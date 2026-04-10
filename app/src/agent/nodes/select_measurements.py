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


# ── schema descriptions lookup ────────────────────────────────────────────────

def _build_description_map(schema: list[dict]) -> dict[str, str]:
    """
    Build a ``database/measurement_pattern → description`` lookup from
    the startup schema so we can annotate refined measurements with
    human-readable context.

    Patterns use ``{property}``-style placeholders, so we also store a
    prefix key (everything before the first ``{``) for fuzzy matching.
    """
    desc_map: dict[str, str] = {}
    for entry in schema:
        db = entry.get("database", "")
        name = entry.get("name", "")
        description = entry.get("description", "")
        if not description:
            continue
        # Exact key
        desc_map[f"{db}:{name}"] = description
        # Prefix key for pattern expansion (e.g. "historian:battman/")
        if "{" in name:
            prefix = name[: name.index("{")]
            desc_map[f"{db}:{prefix}"] = description
    return desc_map


def _match_description(
    db: str, measurement: str, desc_map: dict[str, str]
) -> str:
    """Find the best description for a measurement, or return empty."""
    # Try exact match first.
    key = f"{db}:{measurement}"
    if key in desc_map:
        return desc_map[key]
    # Try progressively shorter prefixes (handles pattern expansion).
    parts = measurement.split("/")
    for i in range(len(parts), 0, -1):
        prefix = "/".join(parts[:i]) + "/"
        prefix_key = f"{db}:{prefix}"
        if prefix_key in desc_map:
            return desc_map[prefix_key]
    return ""


def _schema_listing(
    refined_schema: dict,
    primary_db: str,
    schema: list[dict],
) -> str:
    """
    Render the multi-database refined schema as a compact listing.

    Each measurement is annotated with:
    - ★ marker if it belongs to the primary database
    - A short description (from the startup schema) so the LLM
      understands what data each measurement actually contains
    - Field names (first few) for additional context
    """
    desc_map = _build_description_map(schema)

    lines: list[str] = []
    for db in sorted(refined_schema, key=lambda d: (d != primary_db, d)):
        measurements = refined_schema[db]
        for name, meta in measurements.items():
            fields_raw = meta.get("fields", [])
            field_names = [f["field"] for f in fields_raw[:8]]
            field_str = ", ".join(field_names) if field_names else "*"
            if len(fields_raw) > 8:
                field_str += f" (+{len(fields_raw) - 8})"

            desc = _match_description(db, name, desc_map)
            desc_part = f"  -- {desc}" if desc else ""

            marker = "★" if db == primary_db else " "
            lines.append(
                f"  {marker} {db}:{name}  fields=[{field_str}]{desc_part}"
            )
    return "\n".join(lines)


def _system_prompt(
    listing: str,
    primary_db: str,
    excluded: list[str] | None = None,
) -> str:
    exclusion_block = ""
    if excluded:
        names = "\n  ".join(excluded)
        exclusion_block = f"""

DO NOT SELECT these measurements — they were already queried and returned NO DATA:
  {names}
You MUST pick different measurements this time."""

    return f"""\
You are a measurement selector for an InfluxDB time-series database.

TASK: Pick the measurements that contain the data the user is asking about.

Each measurement below is listed as:
  database:measurement_name  fields=[...]  -- description

Read the descriptions carefully — they tell you what data each measurement stores.

=== AVAILABLE MEASUREMENTS ===
{listing}
=== END ==={exclusion_block}

RULES (follow strictly):
1. Match the user's intent to the measurement DESCRIPTION, not just the name.
2. Only pick measurements whose description clearly relates to the user's question.
3. Prefer ★ (primary database) measurements when relevant.
4. Return the FEWEST measurements needed. Usually 1–3 is enough.
5. Always use the "database:measurement" format exactly as shown above.
6. If unsure between two measurements, pick the one whose description is a closer semantic match.
7. If no measurement is a perfect match, pick the CLOSEST available measurement — always return at least one.

Return ONLY this JSON — nothing else:
{{"measurements": ["database:measurement_name"]}}
"""


# ── helpers ───────────────────────────────────────────────────────────────────

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
    schema: list[dict] = state.get("schema") or []
    excluded: list[str] | None = state.get("empty_measurements")

    retry_count: int = state.get("retry_count", 0)
    if retry_count > 0:
        log.debug("select_measurements  RETRY %d  excluding=%s", retry_count, excluded)

    listing = _schema_listing(refined, primary_db, schema)
    system = _system_prompt(listing, primary_db, excluded=excluded)

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
        for raw in measurements:
            entry = str(raw)
            if ":" in entry:
                # Already prefixed – trust it.
                normalised.append(entry)
            elif entry in lookup:
                # Resolve via schema lookup.
                normalised.append(f"{lookup[entry]}:{entry}")
            else:
                # Not found anywhere – default to primary db.
                normalised.append(f"{primary_db}:{entry}")
        measurements = normalised
    except Exception as exc:  # noqa: BLE001
        log.warning("select_measurements  parse_error=%s", exc)
        return {"error": f"Measurement selection failed: {exc}"}

    log.debug("select_measurements  selected=%s", measurements)
    return {"selected_measurements": measurements, "empty_measurements": None}
