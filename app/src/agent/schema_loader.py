"""
Schema loader for InfluxDB measurements.

Loads ``app/src/data/influx_schema.json`` once at startup (LRU-cached).
The full schema is small enough to pass directly into LLM prompts.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent.parent / "data" / "influx_schema.json"


@lru_cache(maxsize=1)
def load_schema() -> list[dict]:
    """Load and cache the full measurements schema from disk."""
    with _SCHEMA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    measurements = data["measurements"]
    log.info("Schema loaded  path=%s  measurements=%d", _SCHEMA_PATH.name, len(measurements))
    return measurements
