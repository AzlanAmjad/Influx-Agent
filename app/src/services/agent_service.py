from app.src.agent.graphs.intent_graph import build_agent_graph
from app.src.agent.schema_loader import load_schema
from app.src.agent.state import IntentState
from app.src.schemas.chat import Message


class AgentService:
    def __init__(self, default_model: str):
        self.default_model = default_model

        # Load schema and compile the agent graph once at startup.
        self._schema = load_schema()
        self._graph = build_agent_graph()

    # ── intent classification ────────────────────────────────────────────────

    def classify_intent(self, messages: list[Message], model: str | None = None) -> IntentState:
        """
        Run the intent-classification subgraph and return the final state.

        The returned dict always contains:
          is_influx_relevant, is_schema_valid, task_type,
          confidence, reason, filtered_schema, error
        """
        initial_state: IntentState = {
            "messages":           [m.model_dump() for m in messages],
            "schema":             self._schema,
            "model":              model or self.default_model,
            "is_influx_relevant": None,
            "is_schema_valid":    None,
            "task_type":          None,
            "confidence":         None,
            "reason":             None,
            "error":              None,
            "response":           None,
        }
        return self._graph.invoke(initial_state)
