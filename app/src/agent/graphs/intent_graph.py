"""
Agent graph.

Graph topology
──────────────

  START
    │
    ▼
  classify_intent        ← single LLM call (structured JSON)
    │
    ▼
  guardrails             ← relevance, task type, confidence (single node)
    │
    ├─ unsupported → unsupported_response → END
    │
    ├─ task_type=query   → query_pipeline   (stub)
    │
    └─ task_type=anomaly → anomaly_pipeline (stub)

All nodes are idempotent and non-iterative.  The graph compiles once at
startup and is reused across concurrent requests.
"""

from langgraph.graph import END, StateGraph

from app.src.agent.nodes.guardrails import guardrails_node
from app.src.agent.nodes.intent_classifier import classify_intent_node
from app.src.agent.nodes.unsupported_response import unsupported_response_node
from app.src.agent.state import IntentState


# ── routing ───────────────────────────────────────────────────────────────────

def _route_after_guardrails(state: IntentState) -> str:
    if state.get("error"):
        return "unsupported_response"
    task_type = state.get("task_type")
    if task_type == "query":
        return "query_pipeline"
    if task_type == "anomaly":
        return "anomaly_pipeline"
    return "unsupported_response"


# ── pipeline stubs (replaced by real subgraphs in later phases) ───────────────

def _query_pipeline_stub(state: IntentState) -> dict:
    """Placeholder – query-generation pipeline entry point."""
    return {"response": "[query pipeline – not yet implemented]"}


def _anomaly_pipeline_stub(state: IntentState) -> dict:
    """Placeholder – anomaly-detection pipeline entry point."""
    return {"response": "[anomaly pipeline – not yet implemented]"}


# ── graph factory ─────────────────────────────────────────────────────────────

def build_agent_graph():
    """
    Compile and return the agent graph.

    Call once at application startup; the compiled graph is thread-safe
    and can be reused across concurrent requests.
    """
    graph: StateGraph = StateGraph(IntentState)

    graph.add_node("classify_intent",      classify_intent_node)
    graph.add_node("guardrails",           guardrails_node)
    graph.add_node("unsupported_response", unsupported_response_node)
    graph.add_node("query_pipeline",       _query_pipeline_stub)
    graph.add_node("anomaly_pipeline",     _anomaly_pipeline_stub)

    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", "guardrails")

    graph.add_conditional_edges(
        "guardrails",
        _route_after_guardrails,
        {
            "unsupported_response": "unsupported_response",
            "query_pipeline":       "query_pipeline",
            "anomaly_pipeline":     "anomaly_pipeline",
        },
    )

    graph.add_edge("unsupported_response", END)
    graph.add_edge("query_pipeline",       END)
    graph.add_edge("anomaly_pipeline",     END)

    return graph.compile()
