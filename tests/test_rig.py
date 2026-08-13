"""Focused tests for the generated Resource Ingest Guide pipeline.

Covers the PR-readiness contract: schema-shaped documents, honest derived
mechanics (artifact files, observed edge/node summaries, role-separated
provenance), and the built-in audit that refuses to write an invalid RIG.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import yaml

from tablassert import lib
from tablassert.errors import TablassertError
from tablassert.models import DEFAULT_RIG_UI_EXPLANATION, RIGConfig
from tablassert.rig import audit_rig, build_rig_document, compose_ui_explanation, rig_edge_type_info, rig_node_type_info


def _source_entry(primary: str, url: str, upstream: list[str] | None = None) -> list[dict[str, Any]]:
    entry: dict[str, Any] = {"id": primary, "resource_id": primary, "resource_role": "primary_knowledge_source", "source_record_urls": [url]}
    if upstream:
        entry["upstream_resource_ids"] = upstream
    return [entry]


def _write_edges(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    edges: Path = tmp_path / "g_1.edges.ndjson"
    edges.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""))
    return edges


def _edge(subject: str, object_: str, primary: str = "infores:test-kg", url: str = "https://example.org/table.tsv", **extra: Any) -> dict[str, Any]:
    return {
        "id": f"edge-{subject}-{object_}",
        "category": ["biolink:Association"],
        "subject": subject,
        "object": object_,
        "predicate": "biolink:related_to",
        "knowledge_level": "statistical_association",
        "agent_type": "data_analysis_pipeline",
        "primary_knowledge_source": primary,
        "sources": _source_entry(primary, url),
        **extra,
    }


def test_compose_ui_explanation_always_keeps_the_default() -> None:
    """The built-in explanation is always present; a configured prefix goes first."""
    assert compose_ui_explanation(None) == DEFAULT_RIG_UI_EXPLANATION
    assert compose_ui_explanation("   ") == DEFAULT_RIG_UI_EXPLANATION
    composed: str = compose_ui_explanation("Custom context.")
    assert composed.startswith("Custom context. ")
    assert composed.endswith(DEFAULT_RIG_UI_EXPLANATION)


def test_rig_node_type_info_reports_emitted_prefixes_and_factual_fallback() -> None:
    """Identifier types are the prefixes actually emitted; prefix-less ids get a factual note."""
    nodes: list[dict[str, object]] = [
        {"id": "HGNC:1", "category": ["biolink:Gene"]},
        {"id": "HGNC:2", "category": ["biolink:Gene"]},
        {"id": "MONDO:1", "category": ["biolink:Disease"]},
        {"id": "free text entity", "category": ["biolink:PhenotypicFeature"]},
    ]
    info: list[dict[str, object]] = rig_node_type_info(nodes)
    by_category: dict[str, list[str]] = {str(entry["node_category"]): entry["source_identifier_types"] for entry in info}  # pyright: ignore
    assert by_category == {
        "biolink:Disease": ["MONDO"],
        "biolink:Gene": ["HGNC"],
        "biolink:PhenotypicFeature": ["Identifiers emitted verbatim; no CURIE prefixes observed in this build."],
    }


def test_rig_edge_type_info_separates_roles_properties_qualifiers_and_files(tmp_path: Path) -> None:
    """Edge summaries carry role-separated sources, observed properties, qualifier shapes, upstream file names."""
    edges: Path = _write_edges(
        tmp_path,
        [
            _edge(
                "HGNC:1",
                "MONDO:1",
                sources=[
                    {
                        "id": "infores:test-kg",
                        "resource_id": "infores:test-kg",
                        "resource_role": "primary_knowledge_source",
                        "upstream_resource_ids": ["infores:pubmed-central"],
                        "source_record_urls": ["https://pmc.ncbi.nlm.nih.gov/bin/table1.xlsx"],
                    },
                    {"id": "infores:pubmed-central", "resource_id": "infores:pubmed-central", "resource_role": "supporting_data_source"},
                    {"id": "infores:aggregator", "resource_id": "infores:aggregator", "resource_role": "aggregator_knowledge_source"},
                ],
                p_value=0.01,
                publications=["PMID:1"],
                object_aspect_qualifier="increased",
            ),
            _edge("HGNC:2", "MONDO:2", object_aspect_qualifier="decreased"),
        ],
    )
    categories: dict[str, list[str]] = {
        "HGNC:1": ["biolink:Gene"],
        "HGNC:2": ["biolink:Gene"],
        "MONDO:1": ["biolink:Disease"],
        "MONDO:2": ["biolink:Disease"],
    }
    info, fields, count, predicates = rig_edge_type_info(edges, categories, "UI TEXT")
    assert count == 2
    assert predicates == {"biolink:related_to"}
    assert "p_value" in fields
    assert "sources" in fields
    assert len(info) == 1
    entry: dict[str, Any] = info[0]  # pyright: ignore
    assert entry["subject_categories"] == ["biolink:Gene"]
    assert entry["predicates"] == ["biolink:related_to"]
    assert entry["object_categories"] == ["biolink:Disease"]
    assert entry["knowledge_level"] == ["statistical_association"]
    assert entry["agent_type"] == ["data_analysis_pipeline"]
    assert entry["primary_knowledge_sources"] == ["infores:test-kg"]
    assert entry["supporting_data_sources"] == ["infores:pubmed-central"]
    assert entry["aggregator_knowledge_sources"] == ["infores:aggregator"]
    assert entry["edge_properties"] == ["biolink:p_value", "biolink:publications"]
    assert entry["ui_explanation"] == "UI TEXT"
    # Source files come from the upstream source_record_urls, never the output NDJSON names.
    assert entry["source_files"] == ["table.tsv", "table1.xlsx"]
    # Literal-valued qualifiers enumerate observed values under their biolink property.
    assert entry["qualifiers"] == [{"property": "biolink:object_aspect_qualifier", "value_enumeration": ["decreased", "increased"]}]


def test_rig_edge_type_info_summarizes_curie_qualifiers_by_prefix(tmp_path: Path) -> None:
    """CURIE-valued qualifiers are summarized by identifier prefix; qualified_predicate enumerates."""
    edges: Path = _write_edges(
        tmp_path, [_edge("HGNC:1", "MONDO:1", disease_context_qualifier="MONDO:0005148", qualified_predicate="biolink:causes")]
    )
    info, _, _, _ = rig_edge_type_info(edges, {"HGNC:1": ["biolink:Gene"], "MONDO:1": ["biolink:Disease"]}, "UI")
    qualifiers: list[dict[str, Any]] = info[0]["qualifiers"]  # pyright: ignore
    assert {"property": "biolink:disease_context_qualifier", "value_id_prefixes": ["MONDO"]} in qualifiers
    assert {"property": "biolink:qualified_predicate", "value_enumeration": ["biolink:causes"]} in qualifiers


def test_build_rig_document_appends_artifact_names_to_configured_bases(tmp_path: Path, rig_factory: Any) -> None:
    """Generated artifact entries carry exact output names at the configured URL base."""
    rig = RIGConfig.model_validate(rig_factory(tmp_path, infores_id="infores:my-kg"))
    nodes_path: Path = tmp_path / "MY_KG_1.0.0.nodes.ndjson"
    edges_path: Path = tmp_path / "MY_KG_1.0.0.edges.ndjson"
    nodes_path.write_text('{"id":"HGNC:1","category":["biolink:Gene"]}\n')
    edges_path.write_text("")

    document = build_rig_document(
        "MY_KG",
        "1.0.0",
        rig,
        nodes_path,
        edges_path,
        node_type_info=[{"node_category": "biolink:Gene", "source_identifier_types": ["HGNC"]}],
        edge_type_info=[],
        node_count=1,
        edge_count=0,
        node_fields=["category", "id"],
        edge_fields=[],
    )

    assert document["name"] == "MY_KG v1.0.0 Resource Ingest Guide"
    relevant: list[dict[str, Any]] = document["ingest_info"]["relevant_files"]  # pyright: ignore
    by_name: dict[str, dict[str, Any]] = {entry["file_name"]: entry for entry in relevant}
    assert by_name["MY_KG_1.0.0.nodes.ndjson"]["location"] == "https://example.org/my-kg/MY_KG_1.0.0.nodes.ndjson"
    assert by_name["MY_KG_1.0.0.edges.ndjson"]["location"] == "https://example.org/my-kg/MY_KG_1.0.0.edges.ndjson"
    included: list[dict[str, Any]] = document["ingest_info"]["included_content"]  # pyright: ignore
    assert included[0]["included_records"] == "All 1 node records emitted by this build."
    assert included[0]["fields_used"] == "category, id"
    assert "0 edge records" in included[1]["included_records"]


def test_audit_rig_passes_a_clean_document(tmp_path: Path, rig_factory: Any) -> None:
    """A well-formed document built from real output files passes the audit."""
    rig = RIGConfig.model_validate(rig_factory(tmp_path, infores_id="infores:clean-kg"))
    nodes_path: Path = tmp_path / "CLEAN_1.nodes.ndjson"
    edges_path: Path = tmp_path / "CLEAN_1.edges.ndjson"
    _write_edges(tmp_path, [_edge("HGNC:1", "MONDO:1")]).rename(edges_path)
    nodes_path.write_text('{"id":"HGNC:1","category":["biolink:Gene"]}\n{"id":"MONDO:1","category":["biolink:Disease"]}\n')

    edge_info, edge_fields, edge_count, predicates = rig_edge_type_info(
        edges_path, {"HGNC:1": ["biolink:Gene"], "MONDO:1": ["biolink:Disease"]}, "UI"
    )
    document = build_rig_document(
        "CLEAN",
        "1",
        rig,
        nodes_path,
        edges_path,
        rig_node_type_info([{"id": "HGNC:1", "category": ["biolink:Gene"]}, {"id": "MONDO:1", "category": ["biolink:Disease"]}]),
        edge_info,
        2,
        edge_count,
        ["category", "id"],
        edge_fields,
    )
    assert audit_rig(document, rig, nodes_path, edges_path, None, edge_count, predicates, {"biolink:Gene", "biolink:Disease"}) == []


def test_audit_rig_flags_structural_and_consistency_violations(tmp_path: Path, rig_factory: Any) -> None:
    """The audit catches missing files, broken artifact references, and summary drift."""
    rig = RIGConfig.model_validate(rig_factory(tmp_path, infores_id="infores:bad-kg"))
    nodes_path: Path = tmp_path / "BAD_1.nodes.ndjson"
    edges_path: Path = tmp_path / "BAD_1.edges.ndjson"
    nodes_path.write_text('{"id":"HGNC:1","category":["biolink:Gene"]}\n')
    edges_path.write_text("")

    document = build_rig_document(
        "BAD",
        "1",
        rig,
        nodes_path,
        edges_path,
        node_type_info=[{"node_category": "biolink:Gene", "source_identifier_types": ["HGNC"]}],
        edge_type_info=[],
        node_count=1,
        edge_count=0,
        node_fields=["id"],
        edge_fields=[],
    )

    # Clean baseline passes.
    assert audit_rig(document, rig, nodes_path, edges_path, None, 0, set(), {"biolink:Gene"}) == []

    # A summary claiming categories/predicates the graph never emitted is rejected.
    violations = audit_rig(document, rig, nodes_path, edges_path, None, 0, {"biolink:related_to"}, {"biolink:Gene"})
    assert any("edge_type_info predicates" in v for v in violations)
    violations = audit_rig(document, rig, nodes_path, edges_path, None, 0, set(), {"biolink:Gene", "biolink:Disease"})
    assert any("node_type_info categories" in v for v in violations)

    # Deleting a generated artifact entry (or the file itself) is rejected.
    tampered: dict[str, Any] = json.loads(json.dumps(document))
    tampered["ingest_info"]["relevant_files"] = [
        entry for entry in tampered["ingest_info"]["relevant_files"] if entry["file_name"] != edges_path.name
    ]
    violations = audit_rig(tampered, rig, nodes_path, edges_path, None, 0, set(), {"biolink:Gene"})
    assert any("missing the generated artifact" in v for v in violations)


def test_compile_graph_rejects_edges_without_primary_provenance(tmp_path: Path, rig_factory: Any) -> None:
    """A graph whose edges lack primary knowledge sources cannot ship a PR-worthy RIG."""
    sub: Path = tmp_path / "sub.parquet"
    pl.DataFrame(
        {
            "subject": ["HGNC:1"],
            "subject_name": ["A"],
            "subject_category": ["gene"],
            "subject_pre_resolution": ["A"],
            "object": ["MONDO:1"],
            "object_name": ["X"],
            "object_category": ["disease"],
            "object_pre_resolution": ["X"],
            "predicate": ["biolink:related_to"],
            "knowledge_level": ["statistical_association"],
            "agent_type": ["data_analysis_pipeline"],
        }
    ).write_parquet(sub)

    with pytest.raises(TablassertError) as exc_info:
        lib.compile_graph([sub], "noprov", "1", rig_factory(tmp_path, infores_id="infores:noprov-kg"))
    assert exc_info.value.code == "rig-validation-failed"
    assert "primary_knowledge_sources" in str(exc_info.value)
    assert not (tmp_path / "noprov_1.RIG.yaml").exists(), "no RIG may be written when the audit fails"


def test_compile_graph_empty_build_emits_empty_summaries(tmp_path: Path, rig_factory: Any) -> None:
    """A vacuous build (all rows filtered away) emits empty -- but present -- type summaries."""
    sub: Path = tmp_path / "sub.parquet"
    pl.DataFrame(
        {
            "subject": pl.Series([], dtype=pl.String),
            "subject_name": pl.Series([], dtype=pl.String),
            "subject_category": pl.Series([], dtype=pl.String),
            "subject_pre_resolution": pl.Series([], dtype=pl.String),
            "object": pl.Series([], dtype=pl.String),
            "object_name": pl.Series([], dtype=pl.String),
            "object_category": pl.Series([], dtype=pl.String),
            "object_pre_resolution": pl.Series([], dtype=pl.String),
            "predicate": pl.Series([], dtype=pl.String),
            "knowledge_level": pl.Series([], dtype=pl.String),
            "agent_type": pl.Series([], dtype=pl.String),
            "primary_knowledge_source": pl.Series([], dtype=pl.String),
        }
    ).write_parquet(sub)

    lib.compile_graph([sub], "nodeonly", "1", rig_factory(tmp_path, infores_id="infores:nodeonly-kg"))
    document: dict[str, Any] = yaml.safe_load((tmp_path / "nodeonly_1.RIG.yaml").read_text())
    assert document["target_info"]["edge_type_info"] == []
    assert document["target_info"]["node_type_info"] == []
    included: list[dict[str, Any]] = document["ingest_info"]["included_content"]
    assert any("0 node records" in entry["included_records"] for entry in included)
    assert any("0 edge records" in entry["included_records"] for entry in included)


def test_compile_graph_multi_source_keeps_configured_relevant_files(tmp_path: Path, rig_factory: Any) -> None:
    """Configured upstream relevant_files survive alongside artifacts when they match table sources."""
    sub: Path = tmp_path / "sub.parquet"
    pl.DataFrame(
        {
            "subject": ["HGNC:1"],
            "subject_name": ["A"],
            "subject_category": ["gene"],
            "subject_pre_resolution": ["A"],
            "object": ["MONDO:1"],
            "object_name": ["X"],
            "object_category": ["disease"],
            "object_pre_resolution": ["X"],
            "predicate": ["biolink:related_to"],
            "knowledge_level": ["knowledge_assertion"],
            "agent_type": ["manual_agent"],
            "primary_knowledge_source": ["infores:multi-kg"],
            "sources": _source_entry("infores:multi-kg", "https://pmc.ncbi.nlm.nih.gov/bin/table1.xlsx", ["infores:pubmed-central"]),
        }
    ).write_parquet(sub)

    rig_dict: dict[str, Any] = rig_factory(tmp_path, infores_id="infores:multi-kg")
    rig_dict["ingest_info"]["relevant_files"] = [
        {"file_name": "table1.xlsx", "location": "https://pmc.ncbi.nlm.nih.gov/bin/table1.xlsx", "description": "Upstream PMC table."}
    ]
    section_sources: list[dict[str, object]] = [{"local": "table1.xlsx", "urls": ["https://pmc.ncbi.nlm.nih.gov/bin/table1.xlsx"]}]
    lib.compile_graph([sub], "multi", "1", rig_dict, section_sources)

    document: dict[str, Any] = yaml.safe_load((tmp_path / "multi_1.RIG.yaml").read_text())
    names: list[str] = [entry["file_name"] for entry in document["ingest_info"]["relevant_files"]]
    assert names == ["multi_1.nodes.ndjson", "multi_1.edges.ndjson", "table1.xlsx"]
    edge_type: dict[str, Any] = document["target_info"]["edge_type_info"][0]
    assert edge_type["source_files"] == ["table1.xlsx"]
    # The default explanation is always present even without a configured prefix.
    assert edge_type["ui_explanation"] == DEFAULT_RIG_UI_EXPLANATION


def test_compile_graph_rejects_stale_configured_relevant_files(tmp_path: Path, rig_factory: Any) -> None:
    """A configured relevant file that matches no table source is stale documentation and fails."""
    sub: Path = tmp_path / "sub.parquet"
    pl.DataFrame(
        {
            "subject": ["HGNC:1"],
            "subject_name": ["A"],
            "subject_category": ["gene"],
            "subject_pre_resolution": ["A"],
            "object": ["MONDO:1"],
            "object_name": ["X"],
            "object_category": ["disease"],
            "object_pre_resolution": ["X"],
            "predicate": ["biolink:related_to"],
            "knowledge_level": ["knowledge_assertion"],
            "agent_type": ["manual_agent"],
            "primary_knowledge_source": ["infores:stale-kg"],
            "sources": _source_entry("infores:stale-kg", "https://example.org/real-table.tsv"),
        }
    ).write_parquet(sub)

    rig_dict: dict[str, Any] = rig_factory(tmp_path, infores_id="infores:stale-kg")
    rig_dict["ingest_info"]["relevant_files"] = [{"file_name": "ghost.tsv", "location": "https://example.org/ghost.tsv"}]
    section_sources: list[dict[str, object]] = [{"local": "real-table.tsv", "urls": ["https://example.org/real-table.tsv"]}]

    with pytest.raises(TablassertError) as exc_info:
        lib.compile_graph([sub], "stale", "1", rig_dict, section_sources)
    assert exc_info.value.code == "rig-validation-failed"
    assert "ghost.tsv" in str(exc_info.value)
    assert not (tmp_path / "stale_1.RIG.yaml").exists()


def test_compile_graph_ui_prefix_is_composed_with_the_default(tmp_path: Path, rig_factory: Any) -> None:
    """rig.ui_explanation is a PREFIX: the default Tablassert explanation is always appended."""
    sub: Path = tmp_path / "sub.parquet"
    pl.DataFrame(
        {
            "subject": ["HGNC:1"],
            "subject_name": ["A"],
            "subject_category": ["gene"],
            "subject_pre_resolution": ["A"],
            "object": ["MONDO:1"],
            "object_name": ["X"],
            "object_category": ["disease"],
            "object_pre_resolution": ["X"],
            "predicate": ["biolink:related_to"],
            "knowledge_level": ["knowledge_assertion"],
            "agent_type": ["manual_agent"],
            "primary_knowledge_source": ["infores:prefix-kg"],
            "sources": _source_entry("infores:prefix-kg", "https://example.org/table.tsv"),
        }
    ).write_parquet(sub)

    lib.compile_graph([sub], "prefix", "1", rig_factory(tmp_path, infores_id="infores:prefix-kg", ui_explanation="Microbiome context."))
    document: dict[str, Any] = yaml.safe_load((tmp_path / "prefix_1.RIG.yaml").read_text())
    explanation: str = document["target_info"]["edge_type_info"][0]["ui_explanation"]
    assert explanation == f"Microbiome context. {DEFAULT_RIG_UI_EXPLANATION}"
