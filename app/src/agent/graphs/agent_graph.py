"""
Agent graph.

Graph topology
──────────────

  START
    │
    ▼
  classify_intent          ← LLM: structured JSON classification
    │
    ▼
  guardrails               ← deterministic: relevance, task type, confidence
    │
    ├─ error → unsupported_response → END
    │
    └─ supported (query | anomaly)
         │
         ▼
       select_database     ← LLM: picks target InfluxDB databases (primary + historian)
         │
         ▼
       refine_schema       ← deterministic: SHOW queries (cached, includes historian)
         │
         ├─ error → unsupported_response → END
         │
         ▼
       resolve_time        ← LLM: extracts time boundaries (never fails)
         │
         ▼
       select_measurements ← LLM: picks relevant measurements
         │
         ├─ error → unsupported_response → END
         │
         ▼
       build_query         ← deterministic: assembles InfluxQL
         │
         ├─ error → unsupported_response → END
         │
         ▼
       execute_query        ← deterministic: runs InfluxQL via DataFrameClient
         │
         ├─ any empty & retry<1 → select_measurements (retry loop, max 1)
         │
         ├─ task_type=query   → query_pipeline   (LLM summary + Markdown table)
         │
         └─ task_type=anomaly → anomaly_pipeline (Markdown table + WIP notice)

All nodes are idempotent except for the single retry loop between
execute_query → select_measurements → build_query → execute_query.
The graph compiles once at startup and is reused across concurrent
requests.
"""

from langgraph.graph import END, StateGraph

from app.src.agent.nodes.build_query import build_query_node
from app.src.agent.nodes.execute_query import execute_query_node
from app.src.agent.nodes.guardrails import guardrails_node
from app.src.agent.nodes.intent_classifier import classify_intent_node
from app.src.agent.nodes.anomaly_pipeline import anomaly_pipeline_node
from app.src.agent.nodes.query_pipeline import query_pipeline_node
from app.src.agent.nodes.refine_schema import refine_schema_node
from app.src.agent.nodes.resolve_time import resolve_time_node
from app.src.agent.nodes.select_database import select_database_node
from app.src.agent.nodes.select_measurements import select_measurements_node
from app.src.agent.nodes.unsupported_response import unsupported_response_node
from app.src.agent.state import AgentState


# ── routing ─────────────────────────────────────────────────────────────────────

def _route_after_guardrails(state: AgentState) -> str:
    if state.get("error"):
        return "unsupported_response"
    if state.get("task_type") in ("query", "anomaly"):
        return "select_database"
    return "unsupported_response"


def _route_after_refine(state: AgentState) -> str:
    """Route to unsupported if the InfluxDB schema fetch failed."""
    if state.get("error"):
        return "unsupported_response"
    return "resolve_time"


def _route_after_select_measurements(state: AgentState) -> str:
    """Route to unsupported if measurement selection failed."""
    if state.get("error"):
        return "unsupported_response"
    return "build_query"


def _route_after_build_query(state: AgentState) -> str:
    """Route to execute_query, or bail on build failure."""
    if state.get("error"):
        return "unsupported_response"
    return "execute_query"


def _route_after_execute_query(state: AgentState) -> str:
    """Route to the correct pipeline, or retry on empty results."""
    # When any queries returned empty and retries remain, execute_query
    # sets empty_measurements and clears query_results → loop back.
    if state.get("empty_measurements") and state.get("query_results") is None:
        return "select_measurements"

    task_type = state.get("task_type")
    if task_type == "query":
        return "query_pipeline"
    if task_type == "anomaly":
        return "anomaly_pipeline"
    return "unsupported_response"


# ── graph factory ─────────────────────────────────────────────────────────────

def build_agent_graph():
    """
    Compile and return the agent graph.

    Call once at application startup; the compiled graph is thread-safe
    and can be reused across concurrent requests.
    """
    graph: StateGraph = StateGraph(AgentState)

    # nodes
    graph.add_node("classify_intent",      classify_intent_node)
    graph.add_node("guardrails",           guardrails_node)
    graph.add_node("select_database",      select_database_node)
    graph.add_node("refine_schema",        refine_schema_node)
    graph.add_node("resolve_time",         resolve_time_node)
    graph.add_node("select_measurements",  select_measurements_node)
    graph.add_node("build_query",          build_query_node)
    graph.add_node("execute_query",        execute_query_node)
    graph.add_node("unsupported_response", unsupported_response_node)
    graph.add_node("query_pipeline",       query_pipeline_node)
    graph.add_node("anomaly_pipeline",     anomaly_pipeline_node)

    # classify → guardrails
    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", "guardrails")

    # guardrails → unsupported | select_database
    graph.add_conditional_edges(
        "guardrails",
        _route_after_guardrails,
        {
            "unsupported_response": "unsupported_response",
            "select_database":      "select_database",
        },
    )

    # select_database → refine_schema (always)
    graph.add_edge("select_database", "refine_schema")

    # refine_schema → unsupported | resolve_time
    graph.add_conditional_edges(
        "refine_schema",
        _route_after_refine,
        {
            "unsupported_response": "unsupported_response",
            "resolve_time":         "resolve_time",
        },
    )

    # resolve_time → select_measurements (always – never fails)
    graph.add_edge("resolve_time", "select_measurements")

    # select_measurements → unsupported | build_query
    graph.add_conditional_edges(
        "select_measurements",
        _route_after_select_measurements,
        {
            "unsupported_response": "unsupported_response",
            "build_query":          "build_query",
        },
    )

    # build_query → unsupported | execute_query
    graph.add_conditional_edges(
        "build_query",
        _route_after_build_query,
        {
            "unsupported_response": "unsupported_response",
            "execute_query":        "execute_query",
        },
    )

    # execute_query → retry(select_measurements) | pipeline | unsupported
    graph.add_conditional_edges(
        "execute_query",
        _route_after_execute_query,
        {
            "select_measurements":  "select_measurements",
            "query_pipeline":       "query_pipeline",
            "anomaly_pipeline":     "anomaly_pipeline",
            "unsupported_response": "unsupported_response",
        },
    )

    # terminal edges
    graph.add_edge("unsupported_response", END)
    graph.add_edge("query_pipeline",       END)
    graph.add_edge("anomaly_pipeline",     END)

    return graph.compile()
