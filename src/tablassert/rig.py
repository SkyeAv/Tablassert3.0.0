from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tablassert.biolink import AgentTypes, KnowledgeLevels
from tablassert.enums import SourceStatuses
from tablassert.errors import TablassertError
from tablassert.models import DEFAULT_RIG_UI_EXPLANATION, RIGConfig

#: Edge fields present on EVERY edge of a Tablassert build. The RIG schema says
#: properties carried by every edge need not be reported, so these never appear
#: under ``edge_properties`` in the generated edge type summaries.
CORE_EDGE_FIELDS: frozenset[str] = frozenset(
    {"id", "category", "subject", "object", "predicate", "knowledge_level", "agent_type", "primary_knowledge_source", "sources"}
)

#: Edge columns that describe qualifier context rather than edge properties.
QUALIFIER_FIELDS: frozenset[str] = frozenset({"qualified_predicate"})

#: Maximum distinct qualifier values enumerated inline in a RIG qualifier entry.
QUALIFIER_ENUMERATION_LIMIT: int = 50

#: Factual `source_identifier_types` entry for node categories whose emitted
#: identifiers carry no CURIE prefix; the schema requires the field, and this
#: states what the build actually emitted instead of inventing a namespace.
NO_PREFIX_IDENTIFIERS: str = "Identifiers emitted verbatim; no CURIE prefixes observed in this build."

KNOWLEDGE_LEVEL_VALUES: frozenset[str] = frozenset(k.value for k in KnowledgeLevels)
AGENT_TYPE_VALUES: frozenset[str] = frozenset(a.value for a in AgentTypes)
SOURCE_STATUS_VALUES: frozenset[str] = frozenset(s.value for s in SourceStatuses)


def infores(name: str) -> str:
    """Build an infores CURIE from a graph name in lower kebab case.

    Args:
        name: Graph name (typically snake_case).

    Returns:
        ``"infores:<kebab-name>"``.
    """
    return f"infores:{name.lower().replace('_', '-')}"


def strip_nulls(r: object, bad: set[str] | None = None) -> dict:
    """Remove null keys from an NDJSON-style record.

    Args:
        r: Object expected to be a dict (other types are not stripped).
        bad: Lowercased strings treated as null-equivalent.

    Returns:
        Dict with absent and ``bad``-valued keys removed; recurses into
        nested dicts and lists.

    Notes:
        Drops absent values only, not falsy ones. ``0`` and ``False`` are meaningful
        Biolink values (a ``p_value`` of 0, ``number_of_cases: 0``,
        ``negated: False``), so treating them as null would silently delete the key.
        Kept in step with the Rust port in ``rust/src/json.rs``.
    """
    if bad is None:
        bad = {"na", "nan", "null", "none", ""}
    return {
        k: [strip_nulls(i) if isinstance(i, dict) else i for i in v] if isinstance(v, list) else strip_nulls(v) if isinstance(v, dict) else v
        for k, v in r.items()  # pyright: ignore
        if not (v is None or (isinstance(v, str | list | dict | tuple | set) and len(v) == 0)) and str(v).strip().lower() not in bad
    }


def as_list(v: object) -> list[object]:
    """Coerce scalar and list-like values into a plain list.

    Args:
        v: Any value.

    Returns:
        The original list for lists, ``[]`` for ``None``, otherwise a
        single-element list wrapping the value.
    """
    if isinstance(v, list):
        return v
    if v is None:
        return []
    return [v]


def normalize_biolink_category(v: object) -> str | None:
    """Normalize category strings for RIG target summaries.

    Args:
        v: Raw category value.

    Returns:
        Category with a ``biolink:`` prefix, or ``None`` for empty/non-string
        input.
    """
    if not isinstance(v, str) or not v:
        return None
    if v.startswith("biolink:"):
        return v
    return f"biolink:{v}"


def curie_prefix(v: object) -> str | None:
    """Extract compact identifier prefixes for RIG node type summaries.

    Args:
        v: CURIE string (e.g. ``"CHEBI:1234"``).

    Returns:
        Prefix portion (``"CHEBI"``), or ``None`` when no prefix is present.
    """
    if not isinstance(v, str) or ":" not in v:
        return None
    prefix: str = v.split(":", 1)[0]
    return prefix or None


def clean_values(values: list[object]) -> list[str]:
    """Remove empty values and deduplicate stringified RIG summary values.

    Args:
        values: Raw values, possibly nested in lists.

    Returns:
        Sorted, deduplicated list of non-empty stringified values.
    """
    out: list[str] = []
    for value in values:
        for item in as_list(value):
            if item is None:
                continue
            text: str = str(item).strip()
            if not text or text.lower() in {"na", "nan", "null", "none"}:
                continue
            out.append(text)
    return sorted(set(out))


def unique_dicts(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Deduplicate small RIG summary dictionaries without adding a new dependency.

    Args:
        rows: List of dict rows.

    Returns:
        New list preserving first-occurrence order with duplicates removed.
    """
    seen: set[str] = set()
    out: list[dict[str, object]] = []
    for row in rows:
        key: str = json.dumps(row, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def compose_ui_explanation(prefix: str | None) -> str:
    """Compose the per-edge-type UI explanation for the generated RIG.

    The built-in Tablassert explanation is ALWAYS present. A configured
    ``rig.ui_explanation`` prefix is prepended before it, so a graph can add
    KG-specific context without dropping the standard provenance text.

    Args:
        prefix: Optional configured explanation prefix.

    Returns:
        The composed explanation string.
    """
    text: str = (prefix or "").strip()
    return f"{text} {DEFAULT_RIG_UI_EXPLANATION}" if text else DEFAULT_RIG_UI_EXPLANATION


def _qualifier_field(name: str) -> bool:
    """Whether an edge column describes qualifier context (SPOQ ``Q``)."""
    return name in QUALIFIER_FIELDS or name.endswith("_qualifier")


def _qualifier_entry(prop: str, values: list[str]) -> dict[str, object]:
    """Shape one observed qualifier property into a RIG ``Qualifier`` object.

    CURIE-valued qualifiers are summarized by their identifier prefixes;
    literal-valued ones enumerate the observed values (capped). The
    ``qualified_predicate`` property is the exception: upstream RIGs enumerate
    its predicate CURIEs (e.g. ``[biolink:causes]``), so it always enumerates.
    """
    entry: dict[str, object] = {"property": f"biolink:{prop}"}
    if prop != "qualified_predicate" and values and all(":" in v for v in values):
        entry["value_id_prefixes"] = sorted({v.split(":", 1)[0] for v in values})
        return entry
    entry["value_enumeration"] = values[:QUALIFIER_ENUMERATION_LIMIT]
    if len(values) > QUALIFIER_ENUMERATION_LIMIT:
        entry["value_description"] = f"{len(values)} distinct values observed; the first {QUALIFIER_ENUMERATION_LIMIT} are enumerated."
    return entry


def node_category_map(node_rows: list[dict[str, object]]) -> dict[str, list[str]]:
    """Map emitted node ids to their observed Biolink categories.

    Args:
        node_rows: Final node records (dicts with ``id`` and ``category``).

    Returns:
        Mapping of node id to sorted category CURIEs.
    """
    buckets: dict[str, set[str]] = {}
    for node in node_rows:
        nid: object = node.get("id")
        if not isinstance(nid, str) or not nid:
            continue
        categories: list[str] = clean_values([normalize_biolink_category(v) for v in as_list(node.get("category"))])
        buckets.setdefault(nid, set()).update(categories)
    return {nid: sorted(categories) for nid, categories in buckets.items()}


def rig_node_type_info(nodes: list[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize normalized nodes into RIG node type metadata.

    ``source_identifier_types`` reports the identifier prefixes actually
    emitted by the build -- the reliable namespace signal once source records
    have been entity-resolved. Categories whose identifiers carry no prefix
    get a factual free-text entry instead, because the RIG schema requires
    the field and inventing a namespace would be worse than stating the truth.

    Args:
        nodes: List of node dicts with ``id`` and ``category`` keys.

    Returns:
        Sorted list of ``{"node_category", "source_identifier_types"}`` dicts.
    """
    buckets: dict[str, set[str]] = {}
    for node in nodes:
        prefixes: list[str] = clean_values([curie_prefix(node.get("id"))])
        for category in clean_values([normalize_biolink_category(v) for v in as_list(node.get("category"))]):
            buckets.setdefault(category, set()).update(prefixes)

    return [
        {"node_category": category, "source_identifier_types": sorted(prefixes) or [NO_PREFIX_IDENTIFIERS]}
        for category, prefixes in sorted(buckets.items())
    ]


def rig_edge_type_info(
    edges_path: Path, categories: dict[str, list[str]], ui_explanation: str
) -> tuple[list[dict[str, object]], list[str], int, set[str]]:
    """Summarize the FINAL emitted edges into RIG edge type metadata.

    One entry per observed predicate, aggregating the subject/object categories
    resolved from the emitted node file, KL/AT values, qualifier shapes, edge
    properties, and role-separated knowledge sources. Grouping by predicate matches
    the upstream RIG convention that one edge type may list several subject or
    object categories without implying a full cross-product.

    Args:
        edges_path: Final deduplicated edges NDJSON file.
        categories: Node id -> category map from :func:`node_category_map`.
        ui_explanation: Composed UI explanation applied to every edge type.

    Returns:
        Tuple of ``(edge_type_info, observed edge field names, edge count,
        observed predicate set)``.
    """
    groups: dict[str, dict[str, Any]] = {}
    fields: set[str] = set()
    predicates: set[str] = set()
    count: int = 0
    with edges_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            edge: dict[str, Any] = json.loads(line)
            count += 1
            fields.update(edge.keys())
            predicate: str = str(edge.get("predicate") or "")
            predicates.add(predicate)
            group: dict[str, Any] = groups.setdefault(
                predicate,
                {
                    "subjects": set(),
                    "objects": set(),
                    "kl": set(),
                    "at": set(),
                    "primary": set(),
                    "supporting": set(),
                    "aggregator": set(),
                    "properties": set(),
                    "qualifiers": {},
                },
            )
            group["subjects"].update(categories.get(str(edge.get("subject") or ""), []))
            group["objects"].update(categories.get(str(edge.get("object") or ""), []))
            for key in ("knowledge_level", "agent_type"):
                value: object = edge.get(key)
                if value is not None and str(value).strip():
                    group["kl" if key == "knowledge_level" else "at"].add(str(value))
            for key, value in edge.items():
                if key in CORE_EDGE_FIELDS or key.startswith("original_"):
                    continue
                if _qualifier_field(key):
                    for item in clean_values([value]):
                        group["qualifiers"].setdefault(key, set()).add(item)
                else:
                    group["properties"].add(f"biolink:{key}")
            for source_entry in as_list(edge.get("sources")):
                if not isinstance(source_entry, dict):
                    continue
                resource: str = str(source_entry.get("resource_id") or "")
                role: str = str(source_entry.get("resource_role") or "")
                if resource:
                    if role == "primary_knowledge_source":
                        group["primary"].add(resource)
                    elif role == "supporting_data_source":
                        group["supporting"].add(resource)
                    elif role == "aggregator_knowledge_source":
                        group["aggregator"].add(resource)

    info: list[dict[str, object]] = []
    for predicate in sorted(groups):
        group = groups[predicate]
        entry: dict[str, object] = {
            "subject_categories": sorted(group["subjects"]),
            "predicates": [predicate],
            "object_categories": sorted(group["objects"]),
        }
        if group["qualifiers"]:
            entry["qualifiers"] = [_qualifier_entry(prop, sorted(values)) for prop, values in sorted(group["qualifiers"].items())]
        entry["knowledge_level"] = sorted(group["kl"])
        entry["agent_type"] = sorted(group["at"])
        entry["primary_knowledge_sources"] = sorted(group["primary"])
        if group["supporting"]:
            entry["supporting_data_sources"] = sorted(group["supporting"])
        if group["aggregator"]:
            entry["aggregator_knowledge_sources"] = sorted(group["aggregator"])
        if group["properties"]:
            entry["edge_properties"] = sorted(group["properties"])
        entry["ui_explanation"] = ui_explanation
        info.append(entry)
    return info, sorted(fields), count, predicates


def _dump(model: object) -> dict[str, Any]:
    """Serialize a RIG config sub-model to plain JSON-safe data, omitting nulls."""
    dumped: dict[str, Any] = model.model_dump(mode="json", exclude_none=True)  # pyright: ignore
    return dumped


def build_rig_document(
    name: str,
    version: str,
    rig: RIGConfig,
    nodes_path: Path,
    edges_path: Path,
    node_type_info: list[dict[str, object]],
    edge_type_info: list[dict[str, object]],
    node_count: int,
    edge_count: int,
    node_fields: list[str],
    edge_fields: list[str],
) -> dict[str, object]:
    """Assemble the complete RIG document from config plus observed graph facts.

    Config-authored semantics (source metadata, utility/scope, terms,
    provenance) come from ``rig`` verbatim; mechanical facts (generated
    artifact file entries with their record counts and observed fields, target
    edge/node summaries) are composed here from the final KGX output.

    Args:
        name: Graph name.
        version: Graph version string.
        rig: Validated ``rig:`` graph-config section.
        nodes_path: Final nodes output path.
        edges_path: Final edges output path.
        node_type_info: Node summaries from :func:`rig_node_type_info`.
        edge_type_info: Edge summaries from :func:`rig_edge_type_info`.
        node_count: Number of node records emitted.
        edge_count: Number of edge records emitted.
        node_fields: Sorted union of observed node field names.
        edge_fields: Sorted union of observed edge field names.

    Returns:
        The RIG document ready for audit and serialization.
    """
    nodes_url: str = f"{rig.artifact_base_url}/{nodes_path.name}"
    edges_url: str = f"{rig.artifact_base_url}/{edges_path.name}"

    relevant_files: list[dict[str, object]] = [
        {"file_name": nodes_path.name, "location": nodes_url, "description": "KGX node file generated by this Tablassert build."},
        {"file_name": edges_path.name, "location": edges_url, "description": "KGX edge file generated by this Tablassert build."},
    ]
    for entry in rig.ingest_info.relevant_files or []:
        relevant_files.append(_dump(entry))

    included_content: list[dict[str, object]] = [
        {
            "file_name": nodes_path.name,
            "included_records": f"All {node_count} node records emitted by this build.",
            "fields_used": ", ".join(node_fields),
        },
        {
            "file_name": edges_path.name,
            "included_records": f"All {edge_count} edge records emitted by this build across {len(edge_type_info)} observed edge type group(s).",
            "fields_used": ", ".join(edge_fields),
        },
    ]
    for entry in rig.ingest_info.included_content or []:
        included_content.append(_dump(entry))

    ingest_info: dict[str, object] = {
        # `getattr(..., "value", ...)`: the default factory's enum members bypass
        # `use_enum_values` (defaults are not re-validated), so unwrap them here.
        "ingest_categories": [str(getattr(category, "value", category)) for category in rig.ingest_info.ingest_categories],
        "utility": rig.ingest_info.utility,
        "scope": rig.ingest_info.scope,
        "relevant_files": relevant_files,
        "included_content": included_content,
    }
    if rig.ingest_info.filtered_content:
        ingest_info["filtered_content"] = [_dump(entry) for entry in rig.ingest_info.filtered_content]
    if rig.ingest_info.future_considerations:
        ingest_info["future_considerations"] = [_dump(entry) for entry in rig.ingest_info.future_considerations]
    if rig.ingest_info.additional_notes:
        ingest_info["additional_notes"] = list(rig.ingest_info.additional_notes)

    target_info: dict[str, object] = {"edge_type_info": edge_type_info, "node_type_info": node_type_info}
    if rig.target_info is not None:
        if rig.target_info.future_considerations:
            target_info["future_considerations"] = [_dump(entry) for entry in rig.target_info.future_considerations]
        if rig.target_info.additional_notes:
            target_info["additional_notes"] = list(rig.target_info.additional_notes)

    document: dict[str, object] = {
        "name": rig.name or f"{name} v{version} Resource Ingest Guide",
        "source_info": _dump(rig.source_info),
        "ingest_info": ingest_info,
        "target_info": target_info,
        "provenance_info": _dump(rig.provenance_info),
    }
    if rig.supporting_data_source_info:
        document["supporting_data_source_info"] = [_dump(entry) for entry in rig.supporting_data_source_info]
    return document


def _contains_url(value: str) -> bool:
    return "http://" in value or "https://" in value or "file://" in value


def audit_rig(
    document: dict[str, object],
    rig: RIGConfig,
    nodes_path: Path,
    edges_path: Path,
    section_sources: list[dict[str, object]] | None,
    edge_count: int,
    observed_predicates: set[str],
    observed_node_categories: set[str],
) -> list[str]:
    """Validate the assembled RIG document before it may be written.

    Every check here guards a way a generated RIG could be structurally
    invalid, internally inconsistent, or out of sync with the KGX files it
    describes. Any violation fails the build; the YAML is only written when
    this returns an empty list.

    Args:
        document: Assembled RIG document from :func:`build_rig_document`.
        rig: The validated ``rig:`` config section.
        nodes_path: Final nodes output path (must exist and be referenced).
        edges_path: Final edges output path (must exist and be referenced).
        section_sources: Per-section source descriptors (``local``, ``urls``)
            used to cross-check configured relevant-file entries.
        edge_count: Number of edges emitted (drives edge-type requirements).
        observed_predicates: Predicates observed in the final edges file.
        observed_node_categories: Node categories observed in the final nodes.

    Returns:
        List of human-readable violations (empty when the RIG is PR-worthy).
    """
    violations: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            violations.append(message)

    for key in ("name", "source_info", "ingest_info", "target_info", "provenance_info"):
        require(bool(document.get(key)), f"top-level `{key}` is missing or empty")

    source: dict[str, Any] = document.get("source_info") if isinstance(document.get("source_info"), dict) else {}  # pyright: ignore
    require(str(source.get("infores_id") or "").startswith("infores:"), "source_info.infores_id must be an infores: CURIE")
    terms: object = source.get("terms_of_use_info")
    require(
        isinstance(terms, dict) and any(v is not None and str(v).strip() for v in terms.values()),
        "source_info.terms_of_use_info must carry a non-empty terms/license assessment",
    )
    locations: object = source.get("data_access_locations")
    require(
        isinstance(locations, list) and bool(locations) and all(isinstance(v, str) and _contains_url(v) for v in locations),
        "source_info.data_access_locations must be a non-empty list of URL-bearing entries",
    )
    require(str(source.get("source_status") or "") in SOURCE_STATUS_VALUES, "source_info.source_status must be a valid RIG enum value")

    ingest: dict[str, Any] = document.get("ingest_info") if isinstance(document.get("ingest_info"), dict) else {}  # pyright: ignore
    require(bool(str(ingest.get("utility") or "").strip()), "ingest_info.utility must be non-empty")
    require(bool(str(ingest.get("scope") or "").strip()), "ingest_info.scope must be non-empty")
    relevant: object = ingest.get("relevant_files")
    relevant_list: list[dict[str, Any]] = (
        [e for e in relevant if isinstance(e, dict)] if isinstance(relevant, list) else []  # pyright: ignore
    )
    require(
        isinstance(relevant, list)
        and bool(relevant)
        and len(relevant_list) == len(relevant)
        and all(e.get("file_name") and _contains_url(str(e.get("location") or "")) for e in relevant_list),
        "ingest_info.relevant_files must be a non-empty list of {file_name, location} entries",
    )
    included: object = ingest.get("included_content")
    included_list: list[dict[str, Any]] = [e for e in included if isinstance(e, dict)] if isinstance(included, list) else []  # pyright: ignore
    require(
        isinstance(included, list)
        and bool(included)
        and len(included_list) == len(included)
        and all(e.get("file_name") and e.get("included_records") for e in included_list),
        "ingest_info.included_content entries must each carry file_name and included_records",
    )

    # Generated artifact entries must reference BOTH outputs at the configured URL base...
    by_name: dict[str, dict[str, Any]] = {str(e.get("file_name")): e for e in relevant_list}
    for path in (nodes_path, edges_path):
        artifact: dict[str, Any] | None = by_name.get(path.name)
        require(artifact is not None, f"ingest_info.relevant_files is missing the generated artifact {path.name}")
        if artifact is not None:
            require(
                artifact.get("location") == f"{rig.artifact_base_url}/{path.name}",
                f"artifact {path.name} location must be {rig.artifact_base_url}/{path.name}",
            )
        require(path.is_file(), f"generated KGX artifact {path} does not exist")
    # ...and the local artifact base must be the directory they were written to.
    require(
        Path(rig.artifact_base_path).resolve() == nodes_path.parent.resolve(),
        "rig.artifact_base_path must be the directory the generated artifacts were written to",
    )

    target: dict[str, Any] = document.get("target_info") if isinstance(document.get("target_info"), dict) else {}  # pyright: ignore
    node_types: object = target.get("node_type_info")
    node_list: list[dict[str, Any]] = [e for e in node_types if isinstance(e, dict)] if isinstance(node_types, list) else []  # pyright: ignore
    require(isinstance(node_types, list) and len(node_list) == len(node_types), "target_info.node_type_info must be a list of node type objects")
    # A vacuous build (zero nodes emitted) may carry an empty node_type_info; any build
    # that DID emit nodes must describe every observed category.
    require(
        bool(node_list) or observed_node_categories == set(),
        "target_info.node_type_info must list the emitted node categories when the graph has nodes",
    )
    summary_categories: set[str] = set()
    for entry in node_list:
        require(str(entry.get("node_category") or "").startswith("biolink:"), "node_type_info entries need a biolink: node_category")
        require(bool(entry.get("source_identifier_types")), "node_type_info entries need non-empty source_identifier_types")
        summary_categories.add(str(entry.get("node_category") or ""))
    require(
        summary_categories == observed_node_categories,
        f"node_type_info categories {sorted(summary_categories)} must match observed {sorted(observed_node_categories)}",
    )

    edge_types: object = target.get("edge_type_info")
    require(isinstance(edge_types, list), "target_info.edge_type_info must be a list")
    if edge_count > 0:
        require(bool(edge_types), "target_info.edge_type_info must describe the emitted edges (graph has edges but no edge types)")
    summary_predicates: set[str] = set()
    edge_list: list[dict[str, Any]] = [e for e in edge_types if isinstance(e, dict)] if isinstance(edge_types, list) else []  # pyright: ignore
    for entry in edge_list:
        for key in ("subject_categories", "predicates", "object_categories"):
            values: object = entry.get(key)
            require(
                isinstance(values, list) and bool(values) and all(str(v).startswith("biolink:") for v in values),
                f"edge_type_info.{key} must be a non-empty list of biolink: CURIEs",
            )
        kl: object = entry.get("knowledge_level")
        require(
            isinstance(kl, list) and bool(kl) and all(v in KNOWLEDGE_LEVEL_VALUES for v in kl),
            "edge_type_info.knowledge_level must be a non-empty list of valid enum values",
        )
        at: object = entry.get("agent_type")
        require(
            isinstance(at, list) and bool(at) and all(v in AGENT_TYPE_VALUES for v in at),
            "edge_type_info.agent_type must be a non-empty list of valid enum values",
        )
        pks: object = entry.get("primary_knowledge_sources")
        require(
            isinstance(pks, list) and bool(pks) and all(str(v).startswith("infores:") for v in pks),
            "edge_type_info.primary_knowledge_sources must be a non-empty list of infores: CURIEs",
        )
        require(bool(str(entry.get("ui_explanation") or "").strip()), "edge_type_info.ui_explanation must be non-empty")
        for qualifier in entry.get("qualifiers") or []:
            require(isinstance(qualifier, dict) and bool(qualifier.get("property")), "edge_type_info qualifier entries need a property")
        summary_predicates.update(str(v) for v in entry.get("predicates") or [])
    require(
        summary_predicates == observed_predicates,
        f"edge_type_info predicates {sorted(summary_predicates)} must match observed {sorted(observed_predicates)}",
    )

    provenance: dict[str, Any] = document.get("provenance_info") if isinstance(document.get("provenance_info"), dict) else {}  # pyright: ignore
    contributions: object = provenance.get("contributions")
    require(
        isinstance(contributions, list) and bool(contributions) and all(str(v).strip() for v in contributions),
        "provenance_info.contributions must be a non-empty list of contributor statements",
    )

    # Cross-check author-supplied upstream relevant_files against the table sections:
    # an entry that matches no configured source file or URL is stale documentation.
    configured: list[Any] = list(rig.ingest_info.relevant_files or [])
    sources: list[dict[str, object]] = list(section_sources) if section_sources else []
    if configured and sources:
        local_names: set[str] = {str(s.get("local") or "") for s in sources}
        urls: set[str] = {str(url) for s in sources for url in (s.get("urls") or [])}  # pyright: ignore
        for entry in configured:
            dumped: dict[str, Any] = _dump(entry)
            file_name: str = str(dumped.get("file_name") or "")
            location: str = str(dumped.get("location") or "")
            if file_name not in local_names and location not in urls:
                violations.append(
                    f"rig.ingest_info.relevant_files entry {file_name!r} ({location}) matches no configured table source; "
                    f"known sources: {sorted(local_names)} / {sorted(urls)}"
                )

    return violations


def compile_rig(name: str, version: str, rig: RIGConfig, section_sources: list[dict[str, object]] | None, nodes_path: Path, edges_path: Path) -> None:
    """Build, audit, and write the `.RIG.yaml` alongside the KGX outputs.

    Summaries are streamed from the FINAL deduplicated KGX files so the RIG
    always describes exactly what shipped (post-dedup counts included). The
    document is validated in memory FIRST; the file is only written when the
    audit is clean, so a build can never leave behind a RIG that is
    structurally invalid or out of sync with the KGX it describes.

    Args:
        name: Graph name (output file stem).
        version: Graph version string.
        rig: Validated ``rig:`` graph-config section.
        section_sources: Per-section source descriptors for the relevant-file
            cross-check; ``None`` (direct programmatic builds) skips it.
        nodes_path: Final nodes output path.
        edges_path: Final edges output path.

    Raises:
        TablassertError: With code ``rig-validation-failed`` when the audit
            finds any violation.
    """
    from tablassert.ingests import to_yaml

    node_rows: list[dict[str, object]] = []
    with nodes_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                node_rows.append(json.loads(line))

    categories: dict[str, list[str]] = node_category_map(node_rows)
    node_type_info: list[dict[str, object]] = rig_node_type_info(node_rows)
    node_fields: list[str] = sorted({key for node in node_rows for key, value in node.items() if value is not None})
    ui_explanation: str = compose_ui_explanation(rig.ui_explanation)
    edge_type_info, edge_fields, edge_count, observed_predicates = rig_edge_type_info(edges_path, categories, ui_explanation)
    # `source_files` is an authored config fact (rig.source_files), never scraped
    # from edge `source_record_urls`; when configured it applies to every edge type.
    if rig.source_files:
        for entry in edge_type_info:
            entry["source_files"] = sorted(set(rig.source_files))
    observed_node_categories: set[str] = {str(category) for entry in node_type_info for category in as_list(entry.get("node_category"))}

    document: dict[str, object] = build_rig_document(
        name, version, rig, nodes_path, edges_path, node_type_info, edge_type_info, len(node_rows), edge_count, node_fields, edge_fields
    )
    violations: list[str] = audit_rig(
        document, rig, nodes_path, edges_path, section_sources, edge_count, observed_predicates, observed_node_categories
    )
    if violations:
        detail: str = "\n".join(f"- {violation}" for violation in violations)
        raise TablassertError(f"Generated RIG for {name} v{version} failed validation; nothing was written:\n{detail}", code="rig-validation-failed")

    rig_path: Path = nodes_path.parent / f"{name}_{version}.RIG.yaml"
    to_yaml(rig_path, document)
