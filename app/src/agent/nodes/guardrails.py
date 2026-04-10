"""
guardrails node – single deterministic gate applied after intent classification.

Checks (in order):
  1. Query must be InfluxDB-relevant.
  2. task_type must be a routable pipeline (query / anomaly).
  3. Confidence must clear the minimum threshold.

Sets ``error`` on any failure; the graph's conditional edge routes to
``unsupported_response`` when error is non-None.
"""

import logging

from app.src.agent.state import AgentState

log = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD: float = 0.55
_ROUTABLE_TASK_TYPES: frozenset[str] = frozenset({"query", "anomaly"})


def guardrails_node(state: AgentState) -> dict:
    """Single hard-gate node: relevance → task type → confidence."""
    log.debug(
        "guardrails  relevant=%s  task_type=%s  confidence=%s",
        state.get("is_influx_relevant"),
        state.get("task_type"),
        state.get("confidence"),
    )
    if not state.get("is_influx_relevant"):
        return {
            "task_type": "unsupported",
            "error": (
                "Query is not relevant to the InfluxDB schema. "
                f"Reason: {state.get('reason', 'unspecified')}"
            ),
        }

    task_type: str | None = state.get("task_type")
    if task_type not in _ROUTABLE_TASK_TYPES:
        return {
            "task_type": "unsupported",
            "error": (
                f"Task type '{task_type}' is not supported by this agent. "
                f"Reason: {state.get('reason', 'unspecified')}"
            ),
        }

    confidence: float = state.get("confidence") or 0.0
    if confidence < _CONFIDENCE_THRESHOLD:
        return {
            "task_type": "unsupported",
            "error": (
                f"Classification confidence {confidence:.2f} is below the "
                f"required threshold of {_CONFIDENCE_THRESHOLD:.2f}. "
                "Please rephrase your query."
            ),
        }

    return {}
