"""
Shared output-formatting helpers for terminal pipeline nodes.

Provides Markdown rendering utilities so that Open WebUI (or any
Markdown-capable client) can display query results as proper tables.
"""

_MAX_TABLE_ROWS = 50


def _format_cell(value: object) -> str:
    """Format a single cell value for Markdown display."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _truncate_timestamp(ts: str) -> str:
    """Shorten an ISO timestamp to seconds precision for display.

    Drops sub-second fractional parts while preserving any timezone
    suffix:  ``2026-04-10T14:30:00.000000+00:00`` → ``2026-04-10T14:30:00+00:00``
    """
    dot = ts.find(".")
    if dot == -1:
        return ts
    # Find the start of the timezone suffix after the fractional part.
    rest = ts[dot:]
    for i, ch in enumerate(rest):
        if ch in ("+", "-", "Z") and i > 0:
            return ts[:dot] + rest[i:]
    return ts[:dot]


def render_markdown_table(
    query_results: dict,
    max_rows: int = _MAX_TABLE_ROWS,
) -> str:
    """
    Render ``query_results`` as a Markdown table.

    The table includes a ``time`` index column followed by all data
    columns.  Rows beyond *max_rows* are truncated with a notice.

    Returns a user-friendly message when there is no data.
    """
    columns: list[str] = query_results.get("columns", [])
    index: list[str] = query_results.get("index", [])
    data: list[list] = query_results.get("data", [])

    if not columns or not data:
        return "_No data returned._"

    # Header row.
    header = "| time | " + " | ".join(columns) + " |"
    separator = "|---|" + "---|" * len(columns)

    # Truncate rows if needed.
    truncated = len(data) > max_rows
    display_data = data[:max_rows]
    display_index = index[:max_rows]

    rows: list[str] = []
    for ts, row in zip(display_index, display_data):
        cells = [_format_cell(v) for v in row]
        rows.append(f"| {_truncate_timestamp(ts)} | " + " | ".join(cells) + " |")

    table = "\n".join([header, separator, *rows])
    if truncated:
        table += f"\n\n_Showing {max_rows} of {len(data)} rows._"

    return table
