"""
query_pipeline node – LLM-powered summariser with tabular output.

Terminal node for the ``query`` task type.  Renders the queried
time-series data as a Markdown table and asks the LLM to produce a
concise, user-facing summary that relates the results back to the
original question.

The response is structured so Open WebUI renders:
  1. A natural-language summary paragraph.
  2. A full Markdown table of the queried data.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.src.agent.formatting import render_markdown_table
from app.src.agent.llm import get_llm, last_user_message
from app.src.agent.state import AgentState

log = logging.getLogger(__name__)


# ── description lookup ────────────────────────────────────────────────────────

def _measurement_descriptions(
    selected: list[str],
    schema: list[dict],
) -> str:
    """
    Build a compact listing of selected measurements with their
    descriptions sourced from the startup schema JSON.

    Handles both exact matches (``meter:demand``) and pattern-based
    measurements (``historian:inverter/ac_power`` matching the
    ``inverter/{property}`` pattern).
    """
    # Build lookup from schema: exact key + prefix key for patterns.
    desc_map: dict[str, str] = {}
    for entry in schema:
        db = entry.get("database", "")
        name = entry.get("name", "")
        description = entry.get("description", "")
        if not description:
            continue
        desc_map[f"{db}:{name}"] = description
        if "{" in name:
            prefix = name[: name.index("{")]
            desc_map[f"{db}:{prefix}"] = description

    lines: list[str] = []
    for m in selected:
        desc = desc_map.get(m, "")
        if not desc:
            # Prefix matching for pattern-expanded measurements.
            db, _, name = m.partition(":")
            parts = name.split("/")
            for i in range(len(parts), 0, -1):
                prefix_key = f"{db}:{'/'.join(parts[:i])}/"
                if prefix_key in desc_map:
                    desc = desc_map[prefix_key]
                    break
        lines.append(f"  - {m}: {desc}" if desc else f"  - {m}")

    return "\n".join(lines) if lines else "(none)"


# ── prompt helpers ────────────────────────────────────────────────────────────

_SAMPLE_ROWS = 20


def _data_summary_for_prompt(query_results: dict) -> str:
    """Build a compact data summary for the LLM context window.

    Includes row/column counts, time span, and evenly-spaced sample
    rows drawn from across the **entire** time range so the model sees
    a representative downsampled view rather than just the first few
    data points.
    """
    columns: list[str] = query_results.get("columns", [])
    index: list[str] = query_results.get("index", [])
    data: list[list] = query_results.get("data", [])

    if not data:
        return "No data was returned by the query."

    total = len(data)
    parts: list[str] = [
        f"Rows returned: {total}",
        f"Columns: {', '.join(columns)}",
    ]

    if index:
        parts.append(f"Time span: {index[0]} → {index[-1]}")

    # Evenly-spaced sample rows across the full dataset.
    sample_count = min(_SAMPLE_ROWS, total)
    if sample_count >= total:
        sample_indices = list(range(total))
    else:
        # Linspace-style: always include first and last row.
        step = (total - 1) / (sample_count - 1)
        sample_indices = [round(i * step) for i in range(sample_count)]
        # Deduplicate while preserving order (possible with very small datasets).
        sample_indices = list(dict.fromkeys(sample_indices))

    sample_lines: list[str] = []
    for i in sample_indices:
        values = {
            col: data[i][j]
            for j, col in enumerate(columns)
            if data[i][j] is not None
        }
        sample_lines.append(f"  {index[i]}: {values}")

    parts.append(
        f"Sample rows ({len(sample_indices)} evenly spaced across dataset):\n"
        + "\n".join(sample_lines)
    )

    return "\n".join(parts)


def _system_prompt(
    measurement_info: str,
    time_range: dict,
    data_summary: str,
) -> str:
    start = time_range.get("start", "?")
    end = time_range.get("end", "?")
    return f"""\
You are a data analyst summarising InfluxDB time-series query results.

Queried measurements:
{measurement_info}

Time range: {start} → {end}

Query results:
{data_summary}

Provide a clear, concise summary (2–4 sentences) of what the data shows
in relation to the user's question.  Mention key values, trends, or
anything notable.  Do NOT repeat raw data — the user will see a table
below your summary.
"""


# ── node ──────────────────────────────────────────────────────────────────────

def query_pipeline_node(state: AgentState) -> dict:
    """
    Summarise query results via LLM and render a Markdown table.

    The ``response`` contains:
      1. An LLM-generated summary relating the data to the user's question.
      2. A Markdown table of the queried time-series data.
    """
    qr: dict = state.get("query_results") or {}
    selected: list[str] = state.get("selected_measurements") or []
    schema: list[dict] = state.get("schema") or []
    time_range: dict = state.get("time_range") or {}
    user_content = last_user_message(state["messages"])

    # Always render the table (shown even when empty).
    table = render_markdown_table(qr)

    # If there's no data, skip the LLM call entirely.
    if not qr.get("data"):
        return {
            "response": (
                "The query executed successfully but returned no data "
                "for the requested time range.\n\n"
                f"{table}"
            )
        }

    # Build LLM context.
    measurement_info = _measurement_descriptions(selected, schema)
    data_summary = _data_summary_for_prompt(qr)
    system = _system_prompt(measurement_info, time_range, data_summary)

    log.debug(
        "query_pipeline  measurements=%s  rows=%d",
        selected,
        len(qr.get("data", [])),
    )
    try:
        llm = get_llm(state["model"], json_mode=False)
        response = llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=user_content)]
        )
        summary = str(response.content).strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("query_pipeline  summariser failed: %s", exc)
        summary = (
            f"Queried **{len(selected)}** measurement(s) "
            f"({', '.join(selected)}) — "
            f"**{len(qr.get('data', []))}** rows returned."
        )

    log.debug("query_pipeline  summary=%r", summary[:200])
    return {"response": f"{summary}\n\n{table}"}
