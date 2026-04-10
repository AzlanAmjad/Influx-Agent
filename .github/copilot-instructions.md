# Copilot Instructions – Influx-Agent

## Project Overview
Thin HTTP → LangGraph intent-router service that uses remote [Ollama](https://ollama.com) models via `langchain-ollama` and exposes an **OpenAI-compatible** `POST /api/chat` endpoint. Built with FastAPI + Pydantic v2.

## Architecture

```
HTTP client
    │  POST /api/chat  (OpenAI ChatCompletion shape)
    ▼
app/src/api/routes/chat.py   ← formats response into OpenAI shape
    │
    ▼
app/src/services/agent_service.py  ← orchestration; loads schema + invokes graph
    │
    ▼
app/src/agent/graphs/intent_graph.py  ← deterministic routing subgraph
    │
    ├─ filter_schema (deterministic)
    ├─ classify_intent (single LLM structured output)
    └─ guardrails + conditional routing

LLM backend: Ollama server (default: http://192.168.1.157:11434)
```

- **`app/src/core/config.py`** – `pydantic-settings` singleton (`settings`). Reads `OLLAMA_HOST` and `DEFAULT_MODEL` from env / `.env` file.
- **`app/src/schemas/chat.py`** – `Message` and `ChatRequest` Pydantic models; `role` is a `Literal["system","user","assistant"]`.
- Module-level singleton `_service` is instantiated once in `chat.py` at import time using `settings`.

## Critical Import Convention
All internal imports use the **`app.src.*`** namespace (not `app.*`):
```python
from app.src.core.config import settings
from app.src.services.agent_service import AgentService
```

## Key Patterns

### Intent-first execution path
`/api/chat` should invoke `AgentService.classify_intent(...)`, which executes the compiled LangGraph subgraph and returns validated intent state.

### Schema handling
Influx schema lives in-repo at `app/src/data/influx_schema.json`; load once at startup and always pass a filtered subset into prompts.

### Adding new routes
1. Create `app/src/api/routes/<name>.py` with an `APIRouter`.
2. Register in `app/src/main.py`: `app.include_router(<router>, prefix="/api")`.

## Developer Workflows

### Run locally
```bash
# from repo root
python -m app.src.main          # uses uvicorn with reload=True on :8000
```

### Install dependencies
```bash
pip install -e ".[dev]"         # includes httpx, pytest, pytest-asyncio
```

### Test
```bash
pytest                          # uses httpx-based FastAPI TestClient
```

### Configuration override
Create a `.env` file (or export env vars):
```
OLLAMA_HOST=http://localhost:11434
DEFAULT_MODEL=llama3:8b
```

## External Dependencies
| Dependency | Purpose |
|---|---|
| `langchain-ollama` | Talks to the Ollama HTTP API via LangChain |
| `langgraph` + `langchain-core` | Deterministic graph orchestration + message abstractions |
| `fastapi` + `uvicorn` | Web framework & ASGI server |
| `pydantic-settings` | Typed config from env / `.env` |
| `httpx` (dev) | FastAPI `TestClient` transport |

Default Ollama host points to a **remote LAN address** (`192.168.1.157`), not localhost — update `.env` for local development.
