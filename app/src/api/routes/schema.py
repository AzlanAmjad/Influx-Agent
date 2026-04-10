"""
Schema cache inspection endpoint.

Exposes the in-process refined schema cache populated by the
``refine_schema`` graph node so operators can inspect what the agent
has fetched from InfluxDB without triggering a full graph run.
"""

import logging

from fastapi import APIRouter

from app.src.agent.nodes.refine_schema import _cache, _cache_lock

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/schema")
def get_cached_schema():
    """Return the full refined-schema cache (all databases fetched so far)."""
    with _cache_lock:
        snapshot = dict(_cache)

    log.debug("GET /api/schema  cached_databases=%s", list(snapshot.keys()))
    return {
        "databases": list(snapshot.keys()),
        "schema": {
            db: entry.get("measurements", entry) for db, entry in snapshot.items()
        },
    }


@router.get("/schema/{database}")
def get_cached_schema_for_database(database: str):
    """Return the refined schema for a single database, or 404."""
    with _cache_lock:
        entry = _cache.get(database)

    if entry is None:
        return {"error": f"No cached schema for database '{database}'.", "databases": list(_cache.keys())}

    measurements = entry.get("measurements", entry)
    return {
        "database": database,
        "measurement_count": len(measurements),
        "measurements": measurements,
    }
