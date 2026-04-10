"""
OpenAI-compatible ``/v1/chat/completions`` endpoint.

Implements the subset of the OpenAI ChatCompletion contract that
Open WebUI requires:

  * Non-streaming: returns a full ``chat.completion`` JSON object.
  * Streaming (SSE): returns ``text/event-stream`` with chunked
    ``chat.completion.chunk`` deltas followed by ``data: [DONE]``.
  * ``GET /v1/models``: lists the configured default model so Open
    WebUI can discover it.

This route re-uses the same :data:`AgentService` singleton and
:class:`ChatRequest` schema as the legacy ``/api/chat`` endpoint.
"""

import json
import logging
import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.src.agent.state import AgentState
from app.src.core.config import settings
from app.src.schemas.chat import ChatRequest, Message
from app.src.services.agent_service import AgentService

log = logging.getLogger(__name__)

router = APIRouter()

_service = AgentService(default_model=settings.default_model)


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_response(state: AgentState) -> str:
    """Read the final user-facing message written by the terminal node."""
    return state.get("response") or "An unexpected error occurred."


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


def _now() -> int:
    return int(time.time())


# ── /v1/chat/completions ──────────────────────────────────────────────────────

@router.post("/chat/completions")
def chat_completions(req: ChatRequest):
    """
    OpenAI-compatible chat completion endpoint.

    When ``stream=false`` (default) returns a standard ``chat.completion``
    object.  When ``stream=true`` returns an SSE ``text/event-stream``
    that Open WebUI can render progressively.
    """
    try:
        log.debug(
            "POST /v1/chat/completions  model=%s  messages=%d  stream=%s",
            req.model, len(req.messages), req.stream,
        )
        state = _service.run(req.messages, req.model)
        log.debug(
            "Graph finished  task_type=%s  confidence=%s  databases=%s",
            state.get("task_type"),
            state.get("confidence"),
            state.get("databases"),
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    content = _get_response(state)
    model = req.model or settings.default_model
    completion_id = _completion_id()
    created = _now()

    if req.stream:
        return StreamingResponse(
            _stream_sse(content, model, completion_id, created),
            media_type="text/event-stream",
        )

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


# ── SSE streaming ─────────────────────────────────────────────────────────────

def _stream_sse(
    content: str,
    model: str,
    completion_id: str,
    created: int,
):
    """
    Yield the agent response as Server-Sent Events.

    The full response is already computed so we emit it in a single
    content delta (Open WebUI handles this fine).  A future iteration
    can split the content into smaller chunks for a typewriter effect.
    """
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": content},
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(chunk)}\n\n"

    # Final chunk with finish_reason.
    done_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(done_chunk)}\n\n"
    yield "data: [DONE]\n\n"


# ── /v1/models ────────────────────────────────────────────────────────────────

@router.get("/models")
def list_models():
    """
    Minimal ``GET /v1/models`` so Open WebUI can discover available models.

    Returns the configured default model as the single entry.
    """
    model_id = settings.default_model
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "influx-agent",
            }
        ],
    }
