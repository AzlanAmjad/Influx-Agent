"""
unsupported_response node – terminal node for unsupported intents.

Reached when the guardrails determine the user's request cannot be served,
either because it is unrelated to the InfluxDB schema or the agent has no
pipeline capable of handling the classified task type.

Sets ``state["response"]`` to a clear, user-facing explanation so the route
layer has nothing to assemble – it just reads the finished message.
"""

from app.src.agent.state import IntentState

# Friendly preamble shown before the classifier's reason.
_PREAMBLE = (
    "I'm not able to help with that request. "
    "This agent only supports querying and anomaly detection "
    "over the connected InfluxDB system.\n\n"
)


def unsupported_response_node(state: IntentState) -> dict:
    """
    Build a user-facing explanation from whatever the guardrails recorded.

    Priority:
      1. ``error``  – set by a guardrail after a hard constraint violation.
      2. ``reason`` – set by the classifier as a plain-language explanation.
      3. Fallback   – generic message when neither is available.
    """
    detail: str = (
        state.get("error")
        or state.get("reason")
        or "The request could not be classified or is outside this agent's scope."
    )

    return {"response": f"{_PREAMBLE}{detail}"}
