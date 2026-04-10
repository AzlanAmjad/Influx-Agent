"""
Chat route – OpenAI-compatible ``POST /api/chat`` endpoint.

Delegates to :class:`AgentService` which runs the full LangGraph agent
and returns the terminal state.  The response is shaped into an
OpenAI ChatCompletion object with additional ``intent`` and ``plan``
metadata.
"""

import logging
import time
import uuid

from fastapi import APIRouter, HTTPException

from app.src.agent.state import AgentState
from app.src.core.config import settings
from app.src.schemas.chat import ChatRequest
from app.src.services.agent_service import AgentService

log = logging.getLogger(__name__)

router = APIRouter()

_service = AgentService(default_model=settings.default_model)


def _get_response(state: AgentState) -> str:
    """Read the final user-facing message written by the terminal node."""
    return state.get("response") or "An unexpected error occurred."


@router.post("/chat")
def chat(req: ChatRequest):
    """
    Entry point for the LangGraph agent.

    Runs the full agent graph and returns an OpenAI-compatible response.
    The terminal node (unsupported_response, query_pipeline, anomaly_pipeline)
    is responsible for setting state["response"].
    """
    try:
        log.debug("POST /api/chat  model=%s  messages=%d", req.model, len(req.messages))
        state = _service.run(req.messages, req.model)
        log.debug(
            "Graph finished  task_type=%s  confidence=%s  databases=%s",
            state.get("task_type"),
            state.get("confidence"),
            state.get("databases"),
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or settings.default_model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": _get_response(state),
                },
                "finish_reason": "stop",
            }
        ],
        "intent": {
            "is_influx_relevant": state.get("is_influx_relevant"),
            "is_schema_valid":    state.get("is_schema_valid"),
            "task_type":          state.get("task_type"),
            "confidence":         state.get("confidence"),
            "reason":             state.get("reason"),
            "error":              state.get("error"),
        },
        "plan": {
            "databases":             state.get("databases"),
            "selected_measurements": state.get("selected_measurements"),
            "time_range":            state.get("time_range"),
            "influxql_query":        state.get("influxql_query"),
        },
    }
