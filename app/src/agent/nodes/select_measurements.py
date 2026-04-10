"""
select_measurements node – LLM picks relevant measurements.

Single-purpose node: given the refined schema (selected database +
historian) and the user's question, select the fewest measurements from
**any available database** that answer the query.  Returns ``db:name``
pairs so downstream nodes know which database each measurement lives in.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.src.agent.llm import extract_json, get_llm, last_user_message
from app.src.agent.state import AgentState

log = logging.getLogger(__name__)


def _schema_listing(refined_schema: dict, primary_db: str) -> str:
    """
    Render the multi-database refined schema as a compact listing.

    Measurements from the primary database are marked with ★ so the LLM
    gives them priority, but it may also pick from other databases.
    """
    lines: list[str] = []
    for db in sorted(refined_schema, key=lambda d: (d != primary_db, d)):
        measurements = refined_schema[db]
        for name, meta in measurements.items():
            fields_raw = meta.get("fields", [])
            field_str = ", ".join(f["field"] for f in fields_raw[:10]) or "*"
            if len(fields_raw) > 10:
                field_str += f" (+{len(fields_raw) - 10} more)"
            tags = meta.get("tags", [])
            tag_str = ", ".join(tags[:6]) or "(none)"
            marker = "★" if db == primary_db else " "
            lines.append(f"  {marker} {db}:{name}  tags=[{tag_str}]  fields=[{field_str}]")
    return "\n".join(lines)


def _system_prompt(listing: str, primary_db: str) -> str:
    return f"""\
Select the InfluxDB measurements most relevant to the user's query.

Available measurements (★ = primary database):
{listing}

Return ONLY JSON: {{"measurements": ["db:name", "db:name"]}}

Rules:
- Prefer measurements from the '{primary_db}' database (★), but you MAY
  also include measurements from other databases if they are relevant.
- Always include the database prefix (e.g. "meter:demand", "historian:battery/{{property}}").
- Choose the fewest measurements that answer the question.
"""


# ── node ──────────────────────────────────────────────────────────────────────

def _build_lookup(refined_schema: dict) -> dict[str, str]:
    """
    Build a ``measurement_name → database`` lookup from the refined schema.

    If the same measurement name appears in multiple databases the first
    one wins (primary DB is iterated first by the caller).
    """
    lookup: dict[str, str] = {}
    for db, measurements in refined_schema.items():
        for name in measurements:
            if name not in lookup:
                lookup[name] = db
    return lookup


# ── node ──────────────────────────────────────────────────────────────────────

def select_measurements_node(state: AgentState) -> dict:
    """
    Ask the LLM to choose which measurements are relevant.

    Returns ``db:measurement`` strings so build_query knows which database
    each measurement belongs to.  Sets ``error`` when the LLM returns
    nothing parsable or an empty list.
    """
    refined: dict | None = state.get("refined_schema")
    if not refined:
        return {"error": "No refined schema available – cannot select measurements."}

    databases: list[str] = state.get("databases") or []
    primary_db = databases[0] if databases else ""
    user_content = last_user_message(state["messages"])

    retry_count: int = state.get("retry_count", 0)
    if retry_count > 0:
        log.debug("select_measurements  RETRY %d", retry_count)

    listing = _schema_listing(refined, primary_db)
    system = _system_prompt(listing, primary_db)

    log.debug("select_measurements  databases=%s", list(refined.keys()))

    llm = get_llm(state["model"])
    response = llm.invoke(
        [SystemMessage(content=system), HumanMessage(content=user_content)]
    )

    log.debug("select_measurements  raw_response=%r", str(response.content)[:200])

    # Build a name→db lookup so we can resolve bare measurement names to
    # their actual database even when the LLM omits the prefix.
    lookup = _build_lookup(refined)

    try:
        parsed = extract_json(response.content)
        measurements = parsed.get("measurements", [])
        if not isinstance(measurements, list) or not measurements:
            return {"error": "LLM returned no measurements."}

        normalised: list[str] = []
        for m in measurements:
            m = str(m)
            if ":" in m:
                # Already prefixed – trust it.
                normalised.append(m)
            elif m in lookup:
                # Resolve via schema lookup.
                normalised.append(f"{lookup[m]}:{m}")
            else:
                # Not found anywhere – default to primary db.
                normalised.append(f"{primary_db}:{m}")
        measurements = normalised
    except Exception as exc:  # noqa: BLE001
        log.warning("select_measurements  parse_error=%s", exc)
        return {"error": f"Measurement selection failed: {exc}"}

    log.debug("select_measurements  selected=%s", measurements)
    return {"selected_measurements": measurements}
