from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tablassert._lazy import LazyModule
from tablassert.models import DEFAULT_RIG_UI_EXPLANATION, default_rig_contributions

if TYPE_CHECKING:
    import polars as pl
else:
    pl = LazyModule("polars")


TERMS_OF_USE_WARNING: str = (
    "Terms of use and license information for the upstream sources used to create this KGX were not declared in the "
    "Tablassert graph configuration. Translator source-ingest guidance expects source owners to assess terms of use "
    "before publication or downstream ingest; review the upstream source terms and replace this generated warning with "
    "explicit license or terms information when known."
)


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
        Dict with falsy and ``bad``-valued keys removed; recurses into
        nested dicts and lists.
    """
    if bad is None:
        bad = {"na", "nan", "null", "none", ""}
    return {
        k: [strip_nulls(i) if isinstance(i, dict) else i for i in v] if isinstance(v, list) else strip_nulls(v) if isinstance(v, dict) else v
        for k, v in r.items()  # pyright: ignore
        if v and str(v).strip().lower() not in bad
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


def rig_edge_type_info(lf: pl.LazyFrame, edges_path: Path, ui_explanation: str | None) -> list[dict[str, object]]:
    """Summarize raw edge columns into RIG edge type metadata before node normalization.

    Args:
        lf: Edges LazyFrame to summarize.
        edges_path: Path of the edges file (recorded under ``source_files``).
        ui_explanation: Optional human-readable explanation; falls back to
            ``DEFAULT_RIG_UI_EXPLANATION`` when unset.

    Returns:
        List of per-edge-type dicts, or an empty list when none of the
        expected columns are present.
    """
    names: list[str] = lf.collect_schema().names()
    wanted: list[str] = [
        c
        for c in [
            "subject_category",
            "predicate",
            "object_category",
            "knowledge_level",
            "agent_type",
            "primary_knowledge_source",
            "primary_knowledge_sources",
            "resource_id",
            "upstream_resource_ids",
        ]
        if c in names
    ]
    if not wanted:
        return []

    rows: list[dict[str, Any]] = lf.select(wanted).unique().collect().to_dicts()
    info: list[dict[str, object]] = []
    for row in rows:
        primary_sources: list[str] = clean_values(
            as_list(row.get("primary_knowledge_source"))
            + as_list(row.get("primary_knowledge_sources"))
            + as_list(row.get("resource_id"))
            + as_list(row.get("upstream_resource_ids"))
        )
        edge_type: dict[str, object] = strip_nulls(
            {
                "subject_categories": clean_values([normalize_biolink_category(v) for v in as_list(row.get("subject_category"))]),
                "predicates": clean_values(as_list(row.get("predicate"))),
                "object_categories": clean_values([normalize_biolink_category(v) for v in as_list(row.get("object_category"))]),
                "knowledge_level": row.get("knowledge_level"),
                "agent_type": row.get("agent_type"),
                "primary_knowledge_sources": primary_sources,
                "source_files": [edges_path.name],
                "ui_explanation": ui_explanation or DEFAULT_RIG_UI_EXPLANATION,
            }
        )
        if edge_type:
            info.append(edge_type)
    return info


def rig_node_type_info(nodes: list[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize normalized nodes into RIG node type metadata.

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

    return [strip_nulls({"node_category": category, "source_identifier_types": sorted(prefixes)}) for category, prefixes in sorted(buckets.items())]


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


def compile_rig(
    name: str,
    version: str,
    description: str | None,
    contributions: list[str] | None,
    ui_explanation: str | None,
    tables: list[Path] | None,
    nodes_path: Path,
    edges_path: Path,
    node_type_info: list[dict[str, object]],
    edge_type_info: list[dict[str, object]],
    infores_id: str | None = None,
) -> None:
    """Write Translator Resource Ingest Guide metadata alongside KGX outputs.

    Args:
        name: Graph name.
        version: Graph version string.
        description: Optional human-readable description.
        contributions: Optional contributor list; falls back to defaults.
        ui_explanation: Optional edge-type UI explanation (passed through).
        tables: Optional source table paths (unused; kept for API symmetry).
        nodes_path: Path of the nodes file (recorded in the RIG).
        edges_path: Path of the edges file (recorded in the RIG).
        node_type_info: Precomputed node type summaries.
        edge_type_info: Precomputed edge type summaries.
        infores_id: Optional graph-level infores CURIE; defaults to ``infores(name)``.
    """
    from tablassert.ingests import to_yaml

    rig_path: Path = Path(f"./{name}_{version}.RIG.yaml")
    rig: dict[str, object] = strip_nulls(
        {
            "name": f"{name} v{version}",
            "source_info": {
                "infores_id": infores_id or infores(name),
                "name": name,
                "description": description or f"{name} KGX generated by Tablassert.",
                "data_provision_mechanisms": ["file_download"],
                "data_formats": ["kgx"],
                "data_access_locations": [nodes_path.name, edges_path.name],
                "source_status": "unknown",
                "terms_of_use_info": {"terms_of_use_description": TERMS_OF_USE_WARNING},
            },
            "ingest_info": {
                "ingest_categories": ["translator_knowledge_creator"],
                "utility": "Tablassert converts configured tabular source data into KGX nodes and edges for Translator ingestion.",
                "relevant_files": [nodes_path.name, edges_path.name],
                "included_content": "KGX nodes and edges generated from the graph's configured Tablassert table inputs.",
            },
            "provenance_info": {"contributions": contributions or default_rig_contributions()},
            "target_info": {"edge_type_info": edge_type_info, "node_type_info": node_type_info},
        }
    )
    to_yaml(rig_path, rig)
