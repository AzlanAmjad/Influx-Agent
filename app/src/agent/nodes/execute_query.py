"""
execute_query node – runs assembled InfluxQL against InfluxDB.

Parses the ``-- db:<name>`` annotated query string produced by
``build_query``, executes each statement via the ``DataFrameClient``,
and merges all resulting DataFrames on the shared timestamp index.

The merged result is stored in ``query_results`` as a dict of records
so that the state remains JSON-serialisable.
"""

import logging
import re

import pandas as pd

from app.src.agent.state import AgentState
from app.src.db.client import query_dataframe

log = logging.getLogger(__name__)

# Matches the annotation emitted by build_query:  -- db:<name>
_DB_ANNOTATION_RE = re.compile(r"^--\s*db:\s*(.+)$")


def _parse_queries(raw: str) -> list[tuple[str, str]]:
    """
    Split the annotated InfluxQL string into ``(database, select_stmt)``
    pairs.

    Each block has the shape::

        -- db:historian
        SELECT * FROM "battery/SOC" WHERE time >= now() - 5h AND time <= now()

    Multiple blocks are separated by ``;\\n``.
    """
    pairs: list[tuple[str, str]] = []
    current_db: str | None = None

    for block in raw.split(";"):
        block = block.strip()
        if not block:
            continue

        lines = block.splitlines()
        stmt_lines: list[str] = []

        for line in lines:
            m = _DB_ANNOTATION_RE.match(line.strip())
            if m:
                current_db = m.group(1).strip()
            else:
                stmt_lines.append(line)

        stmt = "\n".join(stmt_lines).strip()
        if stmt and current_db:
            pairs.append((current_db, stmt))

    return pairs


def _prefix_columns(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Prefix every column name with *label* so traces from different
    measurements don't collide after the merge.

    The index (time) is left untouched.
    """
    df = df.copy()
    df.columns = [f"{label}.{col}" for col in df.columns]
    return df


# ── node ──────────────────────────────────────────────────────────────────────

def execute_query_node(state: AgentState) -> dict:
    """
    Execute every InfluxQL statement and merge results on timestamp.

    * Each query's columns are prefixed with ``db:measurement`` so
      traces from different sources are distinguishable.
    * All DataFrames are outer-joined on the time index to preserve
      nulls where measurements don't overlap.
    * The final merged frame is stored as ``query_results`` –
      a dict with ``columns``, ``index`` (ISO timestamps), and
      ``data`` (list of row-lists) for downstream serialisation.

    Sets ``error`` when the incoming ``influxql_query`` is missing or
    every query returns an empty result.
    """
    raw_query: str | None = state.get("influxql_query")
    if not raw_query:
        return {"error": "No InfluxQL query to execute."}

    pairs = _parse_queries(raw_query)
    if not pairs:
        return {"error": "Could not parse any executable statements from influxql_query."}

    retry_count: int = state.get("retry_count", 0)
    log.debug("execute_query  statements=%d  retry_count=%d", len(pairs), retry_count)

    frames: list[pd.DataFrame] = []
    empty_labels: list[str] = []

    for db, stmt in pairs:
        # Extract the measurement name for column prefixing.
        m = re.search(r'FROM\s+"([^"]+)"', stmt, re.IGNORECASE)
        measurement = m.group(1) if m else "unknown"
        label = f"{db}:{measurement}"

        log.debug("  → querying  db=%s  measurement=%s", db, measurement)
        try:
            df = query_dataframe(stmt, database=db)
        except Exception:
            log.exception("  ✗ query failed  db=%s  stmt=%r", db, stmt)
            empty_labels.append(label)
            continue

        if df.empty:
            log.debug("  ∅ empty result  db=%s  measurement=%s", db, measurement)
            empty_labels.append(label)
            continue

        log.debug(
            "  ✓ rows=%d  cols=%s",
            len(df),
            list(df.columns[:6]),
        )
        frames.append(_prefix_columns(df, label))

    # ── retry logic: if ANY measurements came back empty, re-run
    #    select_measurements from scratch (max 1 retry). ─────────────────
    _MAX_RETRIES = 1

    if empty_labels and retry_count < _MAX_RETRIES:
        log.warning(
            "execute_query  empty measurements detected (retry %d/%d)  "
            "empty=%s  – requesting retry",
            retry_count + 1,
            _MAX_RETRIES,
            empty_labels,
        )
        return {
            "retry_count": retry_count + 1,
            # Clear stale downstream state so the retry path rebuilds.
            "selected_measurements": None,
            "influxql_query": None,
            "query_results": None,
        }

    if not frames:
        log.warning("execute_query  all queries empty after %d retries", retry_count)
        return {"query_results": {"columns": [], "index": [], "data": []}}

    # Outer-join on the shared DatetimeIndex (time).
    merged = frames[0]
    for df in frames[1:]:
        merged = merged.join(df, how="outer")

    # Sort chronologically.
    merged.sort_index(inplace=True)

    log.info(
        "execute_query  merged  rows=%d  columns=%d",
        len(merged),
        len(merged.columns),
    )

    # Serialise to a JSON-friendly dict.
    result = {
        "columns": list(merged.columns),
        "index":   [ts.isoformat() for ts in merged.index],
        "data":    merged.where(merged.notna(), None).values.tolist(),
    }
    return {"query_results": result}
