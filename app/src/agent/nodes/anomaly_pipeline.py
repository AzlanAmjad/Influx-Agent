"""
anomaly_pipeline node – WIP stub with tabular output.

Terminal node for the ``anomaly`` task type.  Renders the queried data
as a Markdown table and informs the user that the anomaly-detection
tool is not yet available.

Once the anomaly-detection tooling is implemented this node will be
replaced by a real subgraph that analyses the data for faults and
unexpected patterns.
"""

import logging

from app.src.agent.formatting import render_markdown_table
from app.src.agent.state import AgentState

log = logging.getLogger(__name__)

_WIP_NOTICE = (
    "⚠️ **Anomaly Detection — Work in Progress**\n\n"
    "The data below has been queried successfully.  However, the "
    "anomaly-detection tool that would analyse it for faults and "
    "unexpected patterns is still under development.\n\n"
    "Once available, this pipeline will automatically run anomaly "
    "detection on the queried traces and highlight any findings.\n\n"
    "---\n\n"
    "### Queried Data\n\n"
)


# ── node ──────────────────────────────────────────────────────────────────────

def anomaly_pipeline_node(state: AgentState) -> dict:
    """
    Render queried data and inform the user that anomaly detection is WIP.

    The response contains:
      1. A notice explaining the feature is under development.
      2. A Markdown table of the queried time-series data so the user
         can at least inspect the raw traces.
    """
    qr: dict = state.get("query_results") or {}
    table = render_markdown_table(qr)

    log.debug(
        "anomaly_pipeline  rows=%d  cols=%d",
        len(qr.get("index", [])),
        len(qr.get("columns", [])),
    )

    return {"response": f"{_WIP_NOTICE}{table}"}
