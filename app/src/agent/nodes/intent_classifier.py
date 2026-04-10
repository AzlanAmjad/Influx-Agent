"""
intent_classifier node – single LLM call with structured JSON output.

Uses ChatOllama (format="json", temperature=0) to classify the user query
into the IntentClassification contract.  The filtered schema snippet is
injected into the system prompt – never the full schema.
"""

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from app.src.agent.state import IntentClassification, IntentState
from app.src.core.config import settings


def _extract_json_payload(content: str | dict) -> dict:
    """Best-effort JSON extraction from Ollama response content."""
    if isinstance(content, dict):
        return content

    if not isinstance(content, str):
        raise ValueError("Classifier response content is not a string/dict")

    text = content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: extract first JSON object if model prepends/appends text.
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found in classifier response")
    return json.loads(match.group(0))


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

def classify_intent_node(state: IntentState) -> dict:
    """
    Call the LLM once with a tight structured-output prompt and parse the
    JSON response into IntentClassification fields.

    On any parse failure the node gracefully degrades to 'unsupported' so
    downstream guardrails always receive a fully-populated state.
    """
    messages: list[dict] = state["messages"]
    model: str = state["model"]

    user_content: str = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"),
        "",
    )

    snippet = _schema_snippet(state["schema"])
    system = _system_prompt(snippet)

    llm = ChatOllama(
        base_url=settings.ollama_host,
        model=model,
        format="json",   # Ollama native JSON-mode – enforces valid JSON output
        temperature=0,   # fully deterministic classification
    )

    response = llm.invoke(
        [SystemMessage(content=system), HumanMessage(content=user_content)]
    )

    try:
        parsed = _extract_json_payload(response.content)
        clf = _normalize_classification(parsed)
    except Exception as exc:  # noqa: BLE001
        return {
            "is_influx_relevant": False,
            "is_schema_valid": False,
            "task_type": "unsupported",
            "confidence": 0.0,
            "reason": f"Classification response could not be parsed: {exc}",
            "error": f"Intent classifier parse error: {exc}",
        }

    return {
        "is_influx_relevant": clf.is_influx_relevant,
        "is_schema_valid": clf.is_schema_valid,
        "task_type": clf.task_type,
        "confidence": clf.confidence,
        "reason": clf.reason,
        "error": None,
    }
