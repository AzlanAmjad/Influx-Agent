"""
refine_schema node – live InfluxDB SHOW queries, cached per database.

Deterministic node (no LLM).  Hits the InfluxDB HTTP API once per database
and caches the result in-process so repeat requests against the same
database skip the network round-trip entirely.
"""

from __future__ import annotations

import logging
from threading import Lock

from app.src.agent.state import AgentState
from app.src.db.client import show_field_keys, show_measurements, show_tag_keys

log = logging.getLogger(__name__)

# ── in-process cache ──────────────────────────────────────────────────────────
# Keyed by database name.  Values never expire during a single process
# lifetime because the InfluxDB schema is effectively static.

_cache: dict[str, dict] = {}
_cache_lock = Lock()


def _fetch_schema_for_database(database: str) -> dict:
    """
    Query InfluxDB for measurements, tag keys, and field keys.

    Returns::

        {
            "database": "historian",
            "measurements": {
                "battman/soc_schedule": {
                    "tags":   ["forecast_issue_time", ...],
                    "fields": [{"field": "value", "type": "float"}, ...]
                },
                ...
            }
        }
    """
    measurement_names = show_measurements(database)
    log.debug("refine_schema  database=%s  measurements_found=%d", database, len(measurement_names))
    measurements: dict[str, dict] = {}

    for name in measurement_names:
        try:
            tags = show_tag_keys(database, name)
            fields = show_field_keys(database, name)
        except Exception:  # noqa: BLE001
            log.warning("SHOW query failed for %s.%s – skipping", database, name)
            tags, fields = [], []

        measurements[name] = {
            "tags": tags,
            "fields": fields,
        }

    return {
        "database": database,
        "measurements": measurements,
    }


def _get_or_fetch(database: str) -> dict:
    """Return cached schema; fetch on first access."""
    with _cache_lock:
        if database in _cache:
            return _cache[database]

    # Fetch outside the lock to avoid blocking concurrent requests.
    schema = _fetch_schema_for_database(database)

    with _cache_lock:
        # Double-check: another thread may have populated it.
        if database not in _cache:
            _cache[database] = schema
        return _cache[database]


# ── node ──────────────────────────────────────────────────────────────────────

def refine_schema_node(state: AgentState) -> dict:
    """
    Populate ``refined_schema`` with live measurement metadata for every
    database listed in ``databases``.

    The output is a dict keyed by database name::

        {
            "meter":     {"demand": {"tags": [...], "fields": [...]}, …},
            "historian": {"battman/soc_schedule": {…}, …},
        }

    On any InfluxDB connectivity failure for the *primary* (first)
    database the node records an error.  Failures on subsequent
    databases are non-fatal.
    """
    databases: list[str] | None = state.get("databases")
    if not databases:
        return {"error": "No databases selected – cannot refine schema."}

    primary = databases[0]
    refined: dict[str, dict] = {}

    for db in databases:
        try:
            cached = _get_or_fetch(db)
            refined[db] = cached["measurements"]
        except Exception as exc:  # noqa: BLE001
            if db == primary:
                log.exception("Failed to refine schema for primary database '%s'", db)
                return {"error": f"InfluxDB schema query failed: {exc}"}
            # Non-primary fetch is best-effort — don't block the pipeline.
            log.warning("Could not fetch schema for '%s', continuing without it: %s", db, exc)

    log.debug(
        "refine_schema  databases=%s  total_measurements=%d",
        list(refined.keys()),
        sum(len(m) for m in refined.values()),
    )
    return {"refined_schema": refined}
