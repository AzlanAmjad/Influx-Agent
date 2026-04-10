import time
import uuid

from fastapi import APIRouter

from app.src.core.config import settings
from app.src.llm.ollama_client import OllamaClient
from app.src.schemas.chat import ChatRequest
from app.src.services.agent_service import AgentService

router = APIRouter()

_client = OllamaClient(host=settings.ollama_host)
_service = AgentService(_client, default_model=settings.default_model)


def _resp_get(resp, key: str, default=None):
    if isinstance(resp, dict):
        return resp.get(key, default)
    return getattr(resp, key, default)


def _msg_get(message, key: str, default=None):
    if message is None:
        return default
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


@router.post("/chat")
def chat(req: ChatRequest):
    result = _service.run(req.messages, req.model)
    result_message = _resp_get(result, "message")
    created = int(time.time())

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": created,
        "model": _resp_get(result, "model", req.model or settings.default_model),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": _msg_get(result_message, "role", "assistant"),
                    "content": _msg_get(result_message, "content", ""),
                },
                "finish_reason": _resp_get(result, "done_reason", "stop"),
            }
        ],
    }
