# Copilot Instructions – Influx-Agent

## Project Overview
FastAPI service that routes natural-language questions about InfluxDB time-series data through a deterministic LangGraph agent. Exposes an **OpenAI-compatible** `POST /api/chat` endpoint. LLM calls go to a remote Ollama server via `langchain-ollama`.

## Architecture & Data Flow

```
POST /api/chat (OpenAI shape)
  → app/src/api/routes/chat.py        module-level _service singleton
  → app/src/services/agent_service.py  builds AgentState, invokes compiled graph
  → app/src/agent/graphs/agent_graph.py
      classify_intent (LLM) → guardrails (deterministic)
        → select_database (LLM) → refine_schema (InfluxDB SHOW queries, cached)
        → resolve_time (LLM) → select_measurements (LLM)
        → build_query (deterministic) → execute_query (InfluxDB DataFrameClient)
        → query_pipeline / anomaly_pipeline (stubs) → END
```

- **State contract**: `AgentState` (TypedDict in `app/src/agent/state.py`) is populated incrementally by each node. Every node returns a `dict` of only the keys it changes.
- **Error routing**: Any node can set `state["error"]`; conditional edges route to `unsupported_response → END`.
- **Retry loop**: `execute_query` → `select_measurements` (max 1 retry) when measurements return empty data, using `empty_measurements` as an exclusion list.
- **Terminal nodes** set `state["response"]` — the route handler reads it directly.

## Critical Conventions

### Import paths — always `app.src.*`
```python
from app.src.core.config import settings
from app.src.services.agent_service import AgentService
from app.src.agent.llm import extract_json, get_llm
```

### Node implementation pattern
Every graph node is a plain function `(AgentState) → dict` in `app/src/agent/nodes/`. Follow this structure:
1. Read needed keys from state; return `{"error": "..."}` if inputs are missing.
2. For LLM nodes: use `get_llm(state["model"])` + `SystemMessage`/`HumanMessage`, then `extract_json(response.content)`.
3. Return only the keys the node populates — never spread the full state.
4. Graceful degradation: wrap LLM parsing in try/except; fall back to safe defaults (see `resolve_time` as the model — it **never fails**).

### LLM interaction (`app/src/agent/llm.py`)
- `get_llm(model)` → `ChatOllama` with `format="json"`, `temperature=0`.
- `extract_json(content)` → best-effort JSON extraction (handles raw dict, clean JSON string, or balanced-brace extraction from noisy output).
- `last_user_message(messages)` → extracts the latest user message from state.

### Singletons & startup loading
- `settings` (`app/src/core/config.py`) — `pydantic-settings` from env/`.env`.
- `_service` in `chat.py` — `AgentService` instantiated at import time.
- `load_schema()` — `@lru_cache` loads `app/src/data/influx_schema.json` once.
- `_cache` in `refine_schema.py` — thread-safe `dict` caching live InfluxDB SHOW results per database, never expires within process lifetime. Exposed via `GET /api/schema`.
- `_client` / `_df_client` in `app/src/db/client.py` — module-level InfluxDB v1 clients.

### Schema structure
`influx_schema.json` contains `{"measurements": [...]}` with entries like `{"name": "demand", "database": "meter", "description": "..."}`. Pattern names use `{property}` placeholders (e.g. `"inverter/{property}"`). The `historian` database is the broadest catch-all and is always included alongside the primary DB.

### Adding a new graph node
1. Create `app/src/agent/nodes/<name>.py` with a `<name>_node(state: AgentState) -> dict` function.
2. Register in `build_agent_graph()` in `app/src/agent/graphs/agent_graph.py`: `graph.add_node(...)` + edges.
3. Add any new state keys to `AgentState` in `app/src/agent/state.py` with `| None` and document which node populates them.

### Adding a new API route
1. Create `app/src/api/routes/<name>.py` with an `APIRouter`.
2. Register in `app/src/main.py`: `app.include_router(<router>, prefix="/api")`.

## Developer Workflows

```bash
python -m app.src.main              # uvicorn on :8000, reload=True
pip install -e ".[dev]"             # includes pytest, pytest-asyncio
pytest                              # httpx-based FastAPI TestClient
```

### Configuration (`.env` or env vars)
```
OLLAMA_HOST=http://localhost:11434   # default points to remote LAN 192.168.1.157
DEFAULT_MODEL=qwen2.5:14b
INFLUXDB_HOST=localhost
INFLUXDB_PORT=8086
```

## Key Dependencies
| Package | Role |
|---|---|
| `langchain-ollama` | LLM calls to Ollama HTTP API |
| `langgraph` + `langchain-core` | Graph orchestration + message types |
| `fastapi` + `uvicorn` | HTTP framework |
| `influxdb` (v1 client) | `InfluxDBClient` + `DataFrameClient` for InfluxQL |
| `pandas` | Query result merging (outer-join on time index) |
| `pydantic-settings` | Typed config from env |
