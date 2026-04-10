"""
InfluxDB client wrapper for InfluxQL queries.

Uses the official ``influxdb`` v1 Python client library.
Module-level singletons for both the regular client and the DataFrame
client are created from settings on first import.
"""

import logging

import pandas as pd
from influxdb import DataFrameClient, InfluxDBClient

from app.src.core.config import settings

log = logging.getLogger(__name__)

_client = InfluxDBClient(
    host=settings.influxdb_host,
    port=settings.influxdb_port,
)

_df_client = DataFrameClient(
    host=settings.influxdb_host,
    port=settings.influxdb_port,
)

log.info("InfluxDB client connected  host=%s  port=%s", settings.influxdb_host, settings.influxdb_port)


# ── public helpers ────────────────────────────────────────────────────────────

def show_databases() -> list[str]:
    """Return list of database names on the InfluxDB instance."""
    return [db["name"] for db in _client.get_list_database()]


def show_measurements(database: str) -> list[str]:
    """Return list of measurement names in *database*."""
    result = _client.query("SHOW MEASUREMENTS", database=database)
    return [row["name"] for row in result.get_points()]


def show_tag_keys(database: str, measurement: str) -> list[str]:
    """Return tag key names for a specific measurement."""
    result = _client.query(
        f'SHOW TAG KEYS FROM "{measurement}"', database=database
    )
    return [row["tagKey"] for row in result.get_points()]


def show_field_keys(database: str, measurement: str) -> list[dict]:
    """
    Return field keys with types for a specific measurement.

    Each entry: ``{"field": "<name>", "type": "<influx type>"}``
    """
    result = _client.query(
        f'SHOW FIELD KEYS FROM "{measurement}"', database=database
    )
    return [
        {"field": row["fieldKey"], "type": row.get("fieldType", "unknown")}
        for row in result.get_points()
    ]


def run_influxql(query: str, database: str) -> list[dict]:
    """Execute an arbitrary InfluxQL query and return result points."""
    result = _client.query(query, database=database)
    return list(result.get_points())


def query_dataframe(query: str, database: str) -> pd.DataFrame:
    """
    Execute an InfluxQL query and return a pandas DataFrame.

    The DataFrame uses the InfluxDB ``time`` column as its
    DatetimeIndex.  If the query returns multiple series the
    DataFrameClient returns a ``dict[str, DataFrame]`` – this
    helper concatenates them into a single frame.

    Returns an empty DataFrame when there are no results.
    """
    result = _df_client.query(query, database=database)

    if isinstance(result, dict):
        frames = [df for df in result.values() if not df.empty]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1)

    if isinstance(result, pd.DataFrame):
        return result

    return pd.DataFrame()
