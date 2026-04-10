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


class IntentState(TypedDict):
    """
    Shared state contract for the intent-classification subgraph.

    Populated incrementally as each node executes:
      classify_intent  → is_influx_relevant, is_schema_valid, task_type,
                         confidence, reason
      guardrails       → error  (set on hard-constraint violation)
      terminal nodes   → response  (user-facing reply)
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

    # ── terminal output ──────────────────────────────────────────────────────
    response: str | None       # final user-facing message set by terminal nodes
