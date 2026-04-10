"""
select_measurements node – LLM picks relevant measurements.

Single-purpose node: given the refined schema (selected database +
historian) and the user's question, select the fewest measurements from
**any available database** that answer the query.  Returns ``db:name``
pairs so downstream nodes know which database each measurement lives in.
"""

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.src.agent.llm import extract_json, get_llm, last_user_message
from app.src.agent.state import AgentState

log = logging.getLogger(__name__)


# ── schema pattern matching ───────────────────────────────────────────────────


def _build_pattern_index(
    schema: list[dict], db: str
) -> list[tuple[re.Pattern, str, str]]:
    """
    Build a list of ``(regex, pattern_name, description)`` for one
    database from the startup schema.

    Placeholder conversion rules:
      - The **last** ``{placeholder}`` in a pattern becomes ``.+``
        (matches one or more path segments, including ``/``).  This is
        the "rest of path" wildcard — e.g. ``battery/{property}``
        matches ``battery/soc`` **and** ``battery/1/ac/power``.
      - All **earlier** ``{placeholder}``s become ``[^/]+`` (single
        path segment).  E.g. the ``{id}`` in ``battery/{id}/{property}``
        matches exactly one segment like ``1`` or ``2``.

    This correctly routes ``inverter/1/ac/apparent_power`` (4 segments)
    to ``inverter/{id}/{property}`` instead of leaving it uncategorised.

    Sorted by specificity: more path segments first, then exact matches
    before patterns at the same depth.
    """
    patterns: list[tuple[re.Pattern, str, str, int, bool]] = []
    for entry in schema:
        if entry.get("database") != db:
            continue
        name = entry["name"]
        desc = entry.get("description", "")
        is_exact = "{" not in name
        depth = len(name.split("/"))

        if is_exact:
            regex = re.compile(re.escape(name) + "$")
        else:
            segments = name.split("/")
            # Find which segments are placeholders.
            ph_indices = [
                i for i, s in enumerate(segments)
                if s.startswith("{") and s.endswith("}")
            ]
            last_ph = ph_indices[-1] if ph_indices else -1

            regex_parts: list[str] = []
            for i, seg in enumerate(segments):
                if seg.startswith("{") and seg.endswith("}"):
                    # Last placeholder → rest-of-path; others → single segment.
                    regex_parts.append(".+" if i == last_ph else "[^/]+")
                else:
                    regex_parts.append(re.escape(seg))
            regex = re.compile("/".join(regex_parts) + "$")

        patterns.append((regex, name, desc, depth, is_exact))

    # Most specific first: deepest path wins, then exact beats pattern.
    patterns.sort(key=lambda p: (p[3], p[4]), reverse=True)

    return [(regex, name, desc) for regex, name, desc, _, _ in patterns]


def _match_pattern(
    measurement: str,
    patterns: list[tuple[re.Pattern, str, str]],
) -> tuple[str, str] | None:
    """Return ``(pattern_name, description)`` for the best matching pattern."""
    for regex, pattern_name, desc in patterns:
        if regex.fullmatch(measurement):
            return (pattern_name, desc)
    return None


# Matches a path segment that is a bare integer > 1 (device IDs like /2/, /3/, /15/).
_HIGHER_DEVICE_RE = re.compile(r"(?:^|/)([2-9]|\d{2,})(?:/|$)")


def _filter_device_representatives(
    items: list[tuple[str, dict]],
) -> list[tuple[str, dict]]:
    """
    Keep only device-1 (or non-numeric) measurements for prompt display.

    For per-device patterns like ``battery/{id}/{property}``, the live
    schema contains entries for every device (``battery/1/soc``,
    ``battery/2/soc``, ``battery/3/soc``, …).  Showing all of them
    wastes prompt tokens — the LLM only needs to see the device-1
    examples to understand the naming convention.

    Measurements with no numeric device segment (e.g. ``battman/strategy``)
    pass through unchanged.
    """
    return [(name, meta) for name, meta in items if not _HIGHER_DEVICE_RE.search(name)]


def _schema_listing(
    refined_schema: dict,
    primary_db: str,
    schema: list[dict],
) -> str:
    """
    Render the multi-database refined schema as a **grouped** listing.

    Measurements are categorised by their schema pattern from
    ``influx_schema.json``.  Each group shows:
      - A header with the pattern name, description, and total count.
      - Representative measurements (device-1 only for per-device
        patterns) with field names.

    For patterns containing a numeric device ID (e.g. ``battery/2/soc``,
    ``meter/3/demand``), only device-1 examples are shown in the prompt.
    The group header still shows the total count and the prompt rules
    tell the LLM it can construct names for other device IDs.
    """
    sections: list[str] = []

    for db in sorted(refined_schema, key=lambda d: (d != primary_db, d)):
        measurements = refined_schema[db]
        patterns = _build_pattern_index(schema, db)
        marker = "★" if db == primary_db else " "

        # Group measurements by pattern.
        groups: dict[str, dict] = {}  # pattern_name → {desc, items}
        ungrouped: list[tuple[str, dict]] = []

        for name in sorted(measurements):
            meta = measurements[name]
            match = _match_pattern(name, patterns)
            if match:
                pat_name, desc = match
                if pat_name not in groups:
                    groups[pat_name] = {"description": desc, "items": []}
                groups[pat_name]["items"].append((name, meta))
            else:
                ungrouped.append((name, meta))

        # Render each group.
        for pat_name, group in groups.items():
            desc = group["description"]
            items = group["items"]
            total = len(items)

            # Filter to device-1 representatives for per-device patterns.
            shown = _filter_device_representatives(items)
            hidden = total - len(shown)

            count_label = f"{total} measurement{'s' if total != 1 else ''}"
            sections.append(
                f"── {marker} {db}:{pat_name} ({count_label})  -- {desc}"
            )

            for name, meta in shown:
                fields_raw = meta.get("fields", [])
                field_names = [f["field"] for f in fields_raw[:8]]
                field_str = ", ".join(field_names) if field_names else "*"
                if len(fields_raw) > 8:
                    field_str += f" (+{len(fields_raw) - 8})"
                sections.append(f"       {db}:{name}  fields=[{field_str}]")

            if hidden > 0:
                sections.append(
                    f"       (devices 2+ follow the same pattern — {hidden} omitted)"
                )

        # Render ungrouped measurements (if any).
        if ungrouped:
            shown_ug = _filter_device_representatives(ungrouped)
            hidden_ug = len(ungrouped) - len(shown_ug)
            sections.append(f"── {marker} {db}:(uncategorised) ({len(ungrouped)} measurements)")
            for name, meta in shown_ug:
                fields_raw = meta.get("fields", [])
                field_names = [f["field"] for f in fields_raw[:8]]
                field_str = ", ".join(field_names) if field_names else "*"
                sections.append(f"       {db}:{name}  fields=[{field_str}]")
            if hidden_ug > 0:
                sections.append(
                    f"       (devices 2+ follow the same pattern — {hidden_ug} omitted)"
                )

    return "\n".join(sections)


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

Measurements below are grouped by category.  Each group header shows:
  ── database:pattern_name (count)  -- description

Under each header are example measurements with their fields.

=== AVAILABLE MEASUREMENTS ===
{listing}
=== END ==={exclusion_block}

RULES (follow strictly):
1. Read the group DESCRIPTIONS carefully — they tell you what data each category contains.
2. First identify the correct group, then pick specific measurements from that group.
3. Only pick measurements whose group description clearly relates to the user's question.
4. Prefer ★ (primary database) groups when relevant.
5. Return the FEWEST measurements needed. Usually 1–3 is enough.
6. Always use the "database:measurement_name" format exactly as shown.
7. If a group header shows a pattern like battery/{{id}}/{{property}}, you MAY construct
   a measurement name matching that pattern even if the specific name isn't listed
   in the examples. For example, if the user asks about battery 5 SOC, use
   "historian:battery/5/soc" even if only "battery/1/soc" is shown.
8. If no measurement is a perfect match, pick the CLOSEST available — always return at least one.

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
