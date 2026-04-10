"""
build_query node – deterministic InfluxQL assembly.

No LLM call.  Constructs ``SELECT *`` statements from the selected
measurements (``db:name`` pairs), time range, and groups them by
database.  The downstream pipeline nodes can execute these queries
directly against InfluxDB.
"""

import logging
from collections import defaultdict

from app.src.agent.state import AgentState

log = logging.getLogger(__name__)


def _quote(measurement: str) -> str:
    """Always double-quote measurement names for safety."""
    return f'"{measurement}"'


def _parse_db_measurement(entry: str, fallback_db: str) -> tuple[str, str]:
    """
    Split a ``db:measurement`` string.  Falls back to *fallback_db* when
    no colon is present.
    """
    if ":" in entry:
        db, _, name = entry.partition(":")
        return db.strip(), name.strip()
    return fallback_db, entry.strip()


# ── node ──────────────────────────────────────────────────────────────────────

def build_query_node(state: AgentState) -> dict:
    """
    Assemble ``SELECT *`` queries grouped by database.

    Each entry in ``selected_measurements`` is a ``db:name`` string.
    The output ``influxql_query`` contains one or more semicolon-separated
    statements, each annotated with ``-- db:<name>`` so the executor
    knows which database to target.

    Sets ``error`` when required inputs are missing.
    """
    raw_measurements: list[str] = state.get("selected_measurements") or []
    time_range: dict = state.get("time_range") or {}
    databases: list[str] = state.get("databases") or []
    fallback_db: str = databases[0] if databases else ""

    if not raw_measurements:
        return {"error": "No measurements selected – cannot build query."}
    if not fallback_db:
        return {"error": "No databases selected – cannot build query."}

    start = time_range.get("start", "now() - 6h")
    end = time_range.get("end", "now()")

    # Group measurements by database.
    by_db: dict[str, list[str]] = defaultdict(list)
    for entry in raw_measurements:
        if not entry or not entry.strip():
            continue
        db, name = _parse_db_measurement(entry, fallback_db)
        by_db[db].append(name)

    parts: list[str] = []
    total = 0
    for db in sorted(by_db):
        for m in by_db[db]:
            q = f"-- db:{db}\nSELECT * FROM {_quote(m)} WHERE time >= {start} AND time <= {end}"
            parts.append(q)
            total += 1

    influxql = ";\n".join(parts)

    log.debug(
        "build_query  databases=%s  queries=%d  influxql=%r",
        list(by_db.keys()),
        total,
        influxql[:300],
    )
    return {"influxql_query": influxql}
