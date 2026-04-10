"""
select_database node – LLM picks the best InfluxDB database for the query.

Uses a compact prompt listing the available databases (extracted from the
loaded schema) so the model can map the user's intent to a single database.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.src.agent.llm import extract_json, get_llm, last_user_message
from app.src.agent.state import AgentState

log = logging.getLogger(__name__)


def _available_databases(schema: list[dict]) -> dict[str, list[str]]:
    """Group measurement names by database."""
    db_map: dict[str, list[str]] = {}
    for m in schema:
        db_map.setdefault(m["database"], []).append(m["name"])
    return db_map


def _system_prompt(db_map: dict[str, list[str]]) -> str:
    lines: list[str] = []
    for db, measurements in sorted(db_map.items()):
        names = ", ".join(measurements[:8])
        if len(measurements) > 8:
            names += f" … (+{len(measurements) - 8} more)"
        lines.append(f"  {db}: {names}")
    listing = "\n".join(lines)

    return f"""\
Pick the single InfluxDB database most relevant to the user's query.

Available databases and their measurements:
{listing}

Return ONLY JSON: {{"database": "<name>", "reason": "short reason"}}
"""


def _extract_database(content: str | dict, valid: set[str]) -> str:
    """Parse the LLM response and return a validated database name."""
    parsed = extract_json(content)

    db = str(parsed.get("database", "")).strip()
    if db in valid:
        return db

    # Fuzzy fallback: check if the model returned a close match.
    db_lower = db.lower()
    for v in valid:
        if v.lower() == db_lower:
            return v

    raise ValueError(f"Model returned unknown database '{db}'. Valid: {valid}")


# ── node ──────────────────────────────────────────────────────────────────────

def select_database_node(state: AgentState) -> dict:
    """
    Ask the LLM to choose which database is most relevant to the user's
    query.  Falls back to 'historian' (the largest catch-all database) on
    any parse or validation failure.

    Always includes 'historian' alongside the primary database so that
    cross-database queries can pull from both.  If 'historian' *is* the
    primary selection it appears only once.
    """
    db_map = _available_databases(state["schema"])
    valid_dbs = set(db_map.keys())
    user_content = last_user_message(state["messages"])

    llm = get_llm(state["model"])
    system = _system_prompt(db_map)

    response = llm.invoke(
        [SystemMessage(content=system), HumanMessage(content=user_content)]
    )

    try:
        primary = _extract_database(response.content, valid_dbs)
    except Exception:  # noqa: BLE001
        primary = "historian"  # safe default – broadest coverage
        log.warning("select_database  fallback to 'historian'")

    databases = [primary]
    if primary != "historian" and "historian" in valid_dbs:
        databases.append("historian")

    log.debug("select_database  databases=%s", databases)
    return {"databases": databases}
