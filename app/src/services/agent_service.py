import logging

from app.src.agent.graphs.agent_graph import build_agent_graph
from app.src.agent.schema_loader import load_schema
from app.src.agent.state import AgentState
from app.src.schemas.chat import Message

log = logging.getLogger(__name__)


class AgentService:
    def __init__(self, default_model: str):
        self.default_model = default_model

        # Load schema and compile the agent graph once at startup.
        self._schema = load_schema()
        self._graph = build_agent_graph()
        log.info(
            "AgentService ready  model=%s  schema_measurements=%d",
            default_model,
            len(self._schema),
        )

    # ── graph execution ────────────────────────────────────────────────────────────

    def run(self, messages: list[Message], model: str | None = None) -> AgentState:
        """
        Execute the full agent graph and return the final state.

        The graph classifies intent, selects a database, refines the
        live schema, plans an InfluxQL query, and routes to the
        appropriate pipeline (query / anomaly / unsupported).
        """
        initial_state: AgentState = {
            "messages":           [m.model_dump() for m in messages],
            "schema":             self._schema,
            "model":              model or self.default_model,
            "is_influx_relevant": None,
            "is_schema_valid":    None,
            "task_type":          None,
            "confidence":         None,
            "reason":             None,
            "error":              None,
            "databases":             None,
            "refined_schema":        None,
            "selected_measurements": None,
            "time_range":            None,
            "influxql_query":        None,
            "query_results":         None,
            "retry_count":           0,
            "response":           None,
        }
        log.debug("Invoking agent graph  model=%s", initial_state["model"])
        return self._graph.invoke(initial_state)
