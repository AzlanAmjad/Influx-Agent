"""
resolve_time node – LLM extracts time boundaries from the user query.

Single-purpose node: determines the InfluxQL time range (start / end)
from the user's natural-language question.

This node **never fails**.  On any parse error it falls back to the
default window (``now() - 6h`` → ``now()``), so downstream nodes always
receive a valid time_range.
"""

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.src.agent.llm import extract_json, get_llm, last_user_message
from app.src.agent.state import AgentState

log = logging.getLogger(__name__)

_DEFAULT_START = "now() - 6h"
_DEFAULT_END = "now()"

# Matches valid InfluxQL relative-time expressions like:
#   now()   now() - 6h   now() - 30m   now() - 1d
_TIME_EXPR_RE = re.compile(
    r"now\(\)"           # literal now()
    r"(?:\s*-\s*"        # optional: minus sign
    r"\d+[smhdw]"        # digit(s) + unit (s/m/h/d/w)
    r")?"
)


def _sanitise_time(raw: str, default: str) -> str:
    """
    Extract a valid InfluxQL time expression from *raw*, stripping any
    stray characters the LLM may have appended (e.g. trailing ``}``).

    Returns *default* when nothing valid can be found.
    """
    match = _TIME_EXPR_RE.search(raw)
    if match:
        return match.group(0)
    return default


def _system_prompt() -> str:
    return f"""\
Extract the time range from the user's question for an InfluxDB query.

Return ONLY JSON: {{"start": "<InfluxQL time expr>", "end": "<InfluxQL time expr>"}}

Rules:
- Use InfluxQL time expressions: now(), now() - 1h, now() - 30m, now() - 6h, etc.
- Maximum time window is 6 hours.
- If the user doesn't mention a time range, use start="{_DEFAULT_START}" end="{_DEFAULT_END}".
- "last hour" → now() - 1h / now()
- "last 30 minutes" → now() - 30m / now()
- "past 3 hours" → now() - 3h / now()
"""


# ── node ──────────────────────────────────────────────────────────────────────

def resolve_time_node(state: AgentState) -> dict:
    """
    Ask the LLM to infer a time range from the user's question.

    Always returns a valid ``time_range`` dict – falls back to the
    default 6-hour window on any failure.
    """
    user_content = last_user_message(state["messages"])

    log.debug("resolve_time  user=%r", user_content[:120])

    llm = get_llm(state["model"])
    response = llm.invoke(
        [SystemMessage(content=_system_prompt()), HumanMessage(content=user_content)]
    )

    log.debug("resolve_time  raw_response=%r", str(response.content)[:200])

    try:
        parsed = extract_json(response.content)
        start = _sanitise_time(str(parsed.get("start", "")), _DEFAULT_START)
        end = _sanitise_time(str(parsed.get("end", "")), _DEFAULT_END)
    except Exception as exc:  # noqa: BLE001
        log.warning("resolve_time  parse_error=%s – using defaults", exc)
        start, end = _DEFAULT_START, _DEFAULT_END

    time_range = {"start": start, "end": end}
    log.debug("resolve_time  result=%s", time_range)
    return {"time_range": time_range}
