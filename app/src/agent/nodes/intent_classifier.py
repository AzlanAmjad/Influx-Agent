"""
intent_classifier node – single LLM call with structured JSON output.

Classifies the user query into (is_influx_relevant, is_schema_valid,
task_type, confidence, reason) using a compact prompt with the schema
injected as ``database:name`` lines.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.src.agent.llm import extract_json, get_llm, last_user_message
from app.src.agent.state import IntentClassification, AgentState

log = logging.getLogger(__name__)


def _normalize_classification(
    parsed: dict,
) -> IntentClassification:
    """
    Normalize model output into strict schema with safe defaults.
    """
    raw_task_type = parsed.get("task_type")
    if raw_task_type not in {"query", "anomaly", "unsupported"}:
        raw_task_type = "unsupported"

    raw_conf = parsed.get("confidence")
    if isinstance(raw_conf, (int, float)):
        confidence = float(raw_conf)
    else:
        confidence = 0.6 if raw_task_type in {"query", "anomaly"} else 0.2

    confidence = max(0.0, min(1.0, confidence))

    normalized = {
        "is_influx_relevant": bool(parsed.get("is_influx_relevant", False)),
        "is_schema_valid": bool(parsed.get("is_schema_valid", False)),
        "task_type": raw_task_type,
        "confidence": confidence,
        "reason": str(parsed.get("reason") or "No reason provided."),
    }
    return IntentClassification.model_validate(normalized)

# ── prompt helpers ────────────────────────────────────────────────────────────

def _schema_snippet(filtered_schema: list[dict]) -> str:
    """Render schema as compact db:name lines to minimize prompt tokens."""
    if not filtered_schema:
        return "(schema unavailable)"
    lines = [
        f"- {m['database']}:{m['name']}"
        for m in filtered_schema
    ]
    return "\n".join(lines)


def _system_prompt(snippet: str) -> str:
    return f"""\
Classify user intent for an InfluxDB analytics agent.

Known schema entries:
{snippet}

Return ONLY JSON with exact keys:

{{
    "is_influx_relevant": true|false,
    "is_schema_valid": true|false,
  "task_type": "query" | "anomaly" | "unsupported",
    "confidence": 0.0-1.0,
    "reason": "short reason"
}}

Rules:
- query = retrieval, reporting, understanding.
- anomaly = fault/anomaly detection.
- unsupported = not solvable with this schema.
- Keep reason under 20 words.
"""


# ── node ──────────────────────────────────────────────────────────────────────

def classify_intent_node(state: AgentState) -> dict:
    """
    Call the LLM once with a tight structured-output prompt and parse the
    JSON response into IntentClassification fields.

    On any parse failure the node gracefully degrades to 'unsupported' so
    downstream guardrails always receive a fully-populated state.
    """
    user_content = last_user_message(state["messages"])
    snippet = _schema_snippet(state["schema"])
    system = _system_prompt(snippet)

    log.debug("classify_intent  user=%r", user_content[:120])

    llm = get_llm(state["model"])
    response = llm.invoke(
        [SystemMessage(content=system), HumanMessage(content=user_content)]
    )

    log.debug("classify_intent  raw_response=%r", str(response.content)[:200])

    try:
        parsed = extract_json(response.content)
        clf = _normalize_classification(parsed)
    except Exception as exc:  # noqa: BLE001
        log.warning("classify_intent  parse_error=%s", exc)
        return {
            "is_influx_relevant": False,
            "is_schema_valid": False,
            "task_type": "unsupported",
            "confidence": 0.0,
            "reason": f"Classification response could not be parsed: {exc}",
            "error": f"Intent classifier parse error: {exc}",
        }

    log.debug(
        "classify_intent  task_type=%s  confidence=%s  relevant=%s",
        clf.task_type,
        clf.confidence,
        clf.is_influx_relevant,
    )
    return {
        "is_influx_relevant": clf.is_influx_relevant,
        "is_schema_valid": clf.is_schema_valid,
        "task_type": clf.task_type,
        "confidence": clf.confidence,
        "reason": clf.reason,
        "error": None,
    }
