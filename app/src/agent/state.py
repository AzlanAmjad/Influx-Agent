from typing import Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class IntentClassification(BaseModel):
    """Structured output produced by the LLM intent-classifier node."""

    is_influx_relevant: bool = Field(
        description=(
            "True if the user query concerns data accessible via the known "
            "InfluxDB measurements (battery, inverter, meter, weather, historian)."
        )
    )
    is_schema_valid: bool = Field(
        description=(
            "True if the query maps to at least one measurement present in "
            "the pre-filtered schema snippet provided."
        )
    )
    task_type: Literal["query", "anomaly", "unsupported"] = Field(
        description=(
            "query    – user wants to retrieve, understand, or summarise data; "
            "anomaly  – user wants to detect faults, anomalies, or unexpected patterns; "
            "unsupported – query is unrelated, too vague, or cannot be served from InfluxDB."
        )
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Classifier confidence between 0.0 (none) and 1.0 (certain).",
    )
    reason: str = Field(
        description="One-sentence justification for the classification decision."
    )


class AgentState(TypedDict):
    """
    Shared state contract for the agent graph.

    Populated incrementally as each node executes:
      classify_intent      → is_influx_relevant, is_schema_valid, task_type,
                             confidence, reason
      guardrails           → error  (set on hard-constraint violation)
      select_database      → databases  (primary + historian if applicable)
      refine_schema        → refined_schema  (all selected databases)
      resolve_time         → time_range
      select_measurements  → selected_measurements
      build_query          → influxql_query
      execute_query        → query_results | retry
      terminal nodes       → response  (user-facing reply)
    """

    # ── inputs ────────────────────────────────────────────────────────────────
    messages: list[dict]       # raw chat messages (role / content dicts)
    schema: list[dict]         # full measurements schema loaded at startup
    model: str                 # selected LLM model for this request

    # ── derived by classify_intent node ──────────────────────────────────────
    is_influx_relevant: bool | None
    is_schema_valid: bool | None
    task_type: Literal["query", "anomaly", "unsupported"] | None
    confidence: float | None
    reason: str | None

    # ── routing / error propagation ──────────────────────────────────────────
    error: str | None          # non-None signals early termination

    # ── derived by select_database node ──────────────────────────────────────
    databases: list[str] | None  # InfluxDB databases targeted for this request

    # ── derived by refine_schema node ────────────────────────────────────────
    refined_schema: dict | None  # {db_name: {measurement: {tags, fields}}}

    # ── derived by resolve_time node ─────────────────────────────────────────
    time_range: dict | None    # {"start": <str>, "end": <str>}

    # ── derived by select_measurements node ──────────────────────────────────
    selected_measurements: list[str] | None  # ["db:measurement", …]

    # ── derived by build_query node ──────────────────────────────────────────
    influxql_query: str | None # deterministically assembled InfluxQL

    # ── derived by execute_query node ──────────────────────────────────────
    query_results: dict | None # {"columns": [...], "index": [...], "data": [...]}
    retry_count: int           # number of select_measurements retries (max 1)
    # ── terminal output ──────────────────────────────────────────────────────
    response: str | None       # final user-facing message set by terminal nodes
