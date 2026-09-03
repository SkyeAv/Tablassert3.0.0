"""Tests for US-005 ``build_and_audit``: ONE deterministic validate->build->QC->coverage mega-tool.

The ``build_and_audit`` core is PURE (base deps + the real Rust redb only) and runs in
the base environment with NO ``importorskip``: it drives the REAL ``validate_pipeline``
+ ``build_pipeline`` (headless ``_NullProgress``) inside an isolated ``workdir`` against
a tiny REAL redb (the offline recipe from ``tests/test_e2e_smoke.py``: ``brca1`` ->
HGNC:1100, ``mapk1`` -> HGNC:6871). The smolagents ``Tool`` wrapper test calls
``pytest.importorskip("smolagents")``. Every test is offline + fast and uses ``tmp_path``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from tablassert import rs
from tablassert.agent import build_and_audit, make_build_and_audit_tool


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def _synonym_row(curie: str, preferred_name: str, names: list[str], category: str) -> dict[str, Any]:
    return {"curie": curie, "preferred_name": preferred_name, "names": names, "types": [category], "taxa": ["NCBITaxon:9606"]}


def _class_row(curie: str, equivalents: list[str]) -> dict[str, Any]:
    return {"id": curie, "equivalent_identifiers": [{"identifier": x} for x in equivalents]}


def _build_real_redb(root: Path) -> Path:
    """Build a tiny REAL fullmap redb: ``brca1`` -> HGNC:1100, ``mapk1`` -> HGNC:6871."""
    root.mkdir(parents=True, exist_ok=True)
    classes: Path = _write_jsonl(root / "classes.ndjson", [_class_row("HGNC:1100", ["NCBIGene:672"])])
    synonyms: Path = _write_jsonl(
        root / "synonyms.ndjson",
        [_synonym_row("HGNC:1100", "BRCA1", ["BRCA1", "brca1"], "Gene"), _synonym_row("HGNC:6871", "MAPK1", ["MAPK1", "mapk1"], "Gene")],
    )
    output: Path = root / "data" / "fullmap.redb"
    rs.build_fullmap_db(output, [classes], [synonyms], threads=2)
    return output


@pytest.fixture
def redb(tmp_path: Path) -> Path:
    """Offline real redb under ``tmp_path`` (absolute path; no chdir needed to build it)."""
    return _build_real_redb(tmp_path / "fullmap")


def _write_table(tmp_path: Path, text: str) -> Path:
    data: Path = tmp_path / "data.tsv"
    data.write_text(text)
    return data


def _section_config(data: Path) -> dict[str, Any]:
    """A bare merged Section config: column A subject, column B object, PMC provenance."""
    return {
        "source": {"kind": "text", "local": str(data), "url": ["https://example.com/data.tsv"], "delimiter": "\t"},
        "statement": {
            "subject": {"method": "column", "encoding": "A"},
            "predicate": "associated_with",
            "object": {"method": "column", "encoding": "B"},
        },
        "provenance": {"repo": "PMC", "publication": "PMC1"},
    }


def _yaml(config: dict[str, Any]) -> str:
    return yaml.safe_dump(config, sort_keys=False)


# --------------------------------------------------------------------------- #
# PURE core tests (base env; no importorskip)
# --------------------------------------------------------------------------- #


def test_build_and_audit_happy_path(tmp_path: Path, redb: Path) -> None:
    """A valid 2-row config builds a KG end-to-end: ok=True, full coverage, artifacts on disk."""
    data: Path = _write_table(tmp_path, "brca1\tmapk1\nbrca1\tmapk1\n")
    result = build_and_audit(_yaml(_section_config(data)), fullmap=redb, workdir=tmp_path)

    assert result["ok"] is True
    assert result["coverage_pct"] == 1.0
    node_count = result["node_count"]
    assert isinstance(node_count, int)
    assert node_count > 0
    edge_count = result["edge_count"]
    assert isinstance(edge_count, int)
    assert edge_count > 0
    assert result["qc_pass_rate"] is None  # qc defaults False

    kgx_path = result["kgx_path"]
    assert isinstance(kgx_path, str)
    assert Path(kgx_path).is_file()
    edges_path = result["edges_path"]
    assert isinstance(edges_path, str)
    assert Path(edges_path).is_file()

    assert result["errors"] == []
    assert result["error_codes"] == []
    assert result["unresolved"] == []


def test_build_and_audit_bad_predicate_surfaces_coded_error(tmp_path: Path, redb: Path) -> None:
    """A bad predicate enum fails validation: ok=False, coded message verbatim, no artifact, no raise."""
    data: Path = _write_table(tmp_path, "brca1\tmapk1\n")
    cfg: dict[str, Any] = _section_config(data)
    cfg["statement"]["predicate"] = "NOT_A_PREDICATE"

    result = build_and_audit(_yaml(cfg), fullmap=redb, workdir=tmp_path)

    assert result["ok"] is False
    errors = result["errors"]
    assert isinstance(errors, list)
    assert errors
    assert isinstance(errors[0], str)
    assert errors[0]
    # The coded error surfaces its code and/or the offending field; the docs URL is appended.
    codes = result["error_codes"]
    assert isinstance(codes, list)
    assert codes or "predicate" in errors[0]
    assert result["kgx_path"] is None
    assert result["node_count"] == 0


def test_build_and_audit_missing_source_fails(tmp_path: Path, redb: Path) -> None:
    """A config missing ``source`` fails validation cleanly: ok=False, no exception."""
    data: Path = _write_table(tmp_path, "brca1\tmapk1\n")
    cfg: dict[str, Any] = _section_config(data)
    del cfg["source"]

    result = build_and_audit(_yaml(cfg), fullmap=redb, workdir=tmp_path)

    assert result["ok"] is False
    errors = result["errors"]
    assert isinstance(errors, list)
    assert errors
    assert result["kgx_path"] is None


def test_build_and_audit_not_yaml(tmp_path: Path, redb: Path) -> None:
    """Invalid YAML and non-mapping YAML both fail cleanly (ok=False), never raising."""
    # Invalid YAML -> YAMLError path.
    bad = build_and_audit("[", fullmap=redb, workdir=tmp_path)
    assert bad["ok"] is False
    bad_errors = bad["errors"]
    assert isinstance(bad_errors, list)
    assert bad_errors

    # Valid YAML but not a mapping -> "config is not a YAML mapping".
    not_mapping = build_and_audit("- 1\n- 2\n", fullmap=redb, workdir=tmp_path)
    assert not_mapping["ok"] is False
    nm_errors = not_mapping["errors"]
    assert isinstance(nm_errors, list)
    assert nm_errors
    assert not_mapping["kgx_path"] is None


def test_build_and_audit_empty_table_no_traceback(tmp_path: Path, redb: Path) -> None:
    """An empty (0 data row) table never tracebacks: whatever the pipeline does, it returns a dict.

    The guarantee is NO uncaught exception; the build may succeed with 0 counts or fail
    with a clean coded error depending on how polars reads an empty source.
    """
    data: Path = _write_table(tmp_path, "")  # exists but has no rows
    result = build_and_audit(_yaml(_section_config(data)), fullmap=redb, workdir=tmp_path)

    assert isinstance(result, dict)
    assert isinstance(result["ok"], bool)
    assert isinstance(result["errors"], list)
    assert isinstance(result["node_count"], int)


def test_build_and_audit_cwd_preserved(tmp_path: Path, redb: Path) -> None:
    """The internal ``contextlib.chdir`` is scoped: the caller's cwd is unchanged after the call."""
    data: Path = _write_table(tmp_path, "brca1\tmapk1\n")
    before: str = os.getcwd()
    build_and_audit(_yaml(_section_config(data)), fullmap=redb, workdir=tmp_path)
    assert os.getcwd() == before


# --------------------------------------------------------------------------- #
# Tool wrapper test (requires the [agent] extra; skips cleanly when absent)
# --------------------------------------------------------------------------- #


def test_build_and_audit_tool(tmp_path: Path, redb: Path) -> None:
    """The lazily-built tool binds the fullmap via closure and returns a JSON audit report."""
    pytest.importorskip("smolagents")
    data: Path = _write_table(tmp_path, "brca1\tmapk1\n")
    tool = make_build_and_audit_tool(lambda: redb)
    assert tool.name == "build_and_audit"

    parsed: dict[str, Any] = json.loads(tool.forward(_yaml(_section_config(data))))
    assert "ok" in parsed
    assert "coverage_pct" in parsed
    assert parsed["ok"] is True


def test_build_and_audit_measures_relative_source_with_correct_cwd(tmp_path: Path, redb: Path) -> None:
    """Regression (review fix 2): coverage is measured INSIDE the build's chdir(workdir).

    A config with a RELATIVE source ``local`` builds fine (the pipeline chdir's into workdir) but used to
    be measured by map_coverage from the ORIGINAL cwd -> frame reproduction failed -> false/unmeasurable
    coverage. build_and_audit now measures inside the same chdir(workdir), so a relative path resolves and
    coverage is a real measurement (1.0 here), never an 'unmeasurable' note.
    """
    workdir: Path = tmp_path / "work"
    workdir.mkdir(parents=True)
    (workdir / "rel.tsv").write_text("brca1\tmapk1\nbrca1\tmapk1\n")
    cfg: dict[str, Any] = {
        "source": {"kind": "text", "local": "rel.tsv", "url": ["https://example.com/rel.tsv"], "delimiter": "\t"},
        "statement": {
            "subject": {"method": "column", "encoding": "A"},
            "predicate": "associated_with",
            "object": {"method": "column", "encoding": "B"},
        },
        "provenance": {"repo": "PMC", "publication": "PMC1"},
    }
    report: dict[str, Any] = build_and_audit(_yaml(cfg), fullmap=redb, workdir=workdir)
    assert report["ok"] is True
    assert report["coverage_pct"] == 1.0  # brca1/mapk1 resolve -> a REAL measurement, not vacuous
    assert not any("unmeasurable" in str(note) for note in report["errors"])


def test_build_and_audit_multi_section_two_files(tmp_path: Path, redb: Path) -> None:
    """W3: a ``{template, sections}`` config with TWO sections (different files) builds ONE graph.

    Each section owns its own ``source`` (a different file); the template carries the shared provenance.
    The build produces nodes/edges from BOTH sections and ``coverage_pct`` is the AGGREGATE (mean) across
    sections, with ``measured`` True iff every section measured.
    """
    t1: Path = tmp_path / "s1.tsv"
    t1.write_text("brca1\tmapk1\nbrca1\tmapk1\n")
    t2: Path = tmp_path / "s2.tsv"
    t2.write_text("mapk1\tbrca1\nmapk1\tbrca1\n")
    cfg: dict[str, Any] = {
        "template": {"provenance": {"repo": "PMC", "publication": "PMC1"}},
        "sections": [
            {
                "source": {"kind": "text", "local": str(t1), "url": ["https://example.com/s1.tsv"], "delimiter": "\t"},
                "statement": {
                    "subject": {"method": "column", "encoding": "A"},
                    "predicate": "associated_with",
                    "object": {"method": "column", "encoding": "B"},
                },
            },
            {
                "source": {"kind": "text", "local": str(t2), "url": ["https://example.com/s2.tsv"], "delimiter": "\t"},
                "statement": {
                    "subject": {"method": "column", "encoding": "A"},
                    "predicate": "associated_with",
                    "object": {"method": "column", "encoding": "B"},
                },
            },
        ],
    }
    result = build_and_audit(_yaml(cfg), fullmap=redb, workdir=tmp_path)
    assert result["ok"] is True
    assert result["measured"] is True
    assert result["coverage_pct"] == 1.0  # both sections fully resolve -> aggregate mean 1.0
    node_count = result["node_count"]
    assert isinstance(node_count, int)
    assert node_count > 0
    edge_count = result["edge_count"]
    assert isinstance(edge_count, int)
    assert edge_count > 0


def _gene_disease_redb(root: Path) -> Path:
    """A fullmap resolving ``brca1`` -> HGNC:1100 (Gene) and ``lung cancer`` -> MONDO:0008903 (Disease).

    The gene~gene fixture above cannot exercise predicate demotion: ``GeneToGeneAssociation``
    leaves ``predicate`` open, so nothing can be forbidden. A gene~disease pair derives
    ``GeneToDiseaseAssociation``, whose enum permits only affects / associated_with / contributes_to.
    """
    root.mkdir(parents=True, exist_ok=True)
    classes: Path = _write_jsonl(root / "classes.ndjson", [_class_row("HGNC:1100", ["NCBIGene:672"])])
    synonyms: Path = _write_jsonl(
        root / "synonyms.ndjson",
        [_synonym_row("HGNC:1100", "BRCA1", ["BRCA1", "brca1"], "Gene"), _synonym_row("MONDO:0008903", "lung cancer", ["lung cancer"], "Disease")],
    )
    output: Path = root / "data" / "fullmap.redb"
    rs.build_fullmap_db(output, [classes], [synonyms], threads=2)
    return output


def test_build_and_audit_reports_biolink_validity(tmp_path: Path, redb: Path) -> None:
    """The audit report carries the Biolink compliance of what it just built."""
    data: Path = _write_table(tmp_path, "brca1\tmapk1\n")
    result = build_and_audit(_yaml(_section_config(data)), fullmap=redb, workdir=tmp_path)

    assert result["ok"] is True
    # Measured, not None: the build produced artifacts, so validity is a real number in [0, 1].
    assert isinstance(result["biolink_valid_pct"], float)
    assert 0.0 <= result["biolink_valid_pct"] <= 1.0
    assert isinstance(result["biolink_valid_pct_strict"], float)
    # The pending exemption can only ever forgive, never accuse.
    assert result["biolink_valid_pct"] >= result["biolink_valid_pct_strict"]
    assert isinstance(result["biolink_problems"], dict)
    assert isinstance(result["demoted_edge_pct"], float)


def test_demoted_edge_pct_catches_a_predicate_its_class_forbids(tmp_path: Path) -> None:
    """The end-to-end proof: the SAME table, two predicates, opposite demotion.

    ``gene_associated_with_condition`` on a gene~disease table is the exact failure the Biolink fix
    measured across 723,595 edges. It does not error -- ``resolve_association_class`` walks up to
    bare ``biolink:Association`` -- so ``demoted_edge_pct`` is the only signal the agent gets.
    """
    fullmap: Path = _gene_disease_redb(tmp_path / "fullmap")
    data: Path = _write_table(tmp_path, "brca1\tlung cancer\n")

    def build(predicate: str, where: str) -> dict[str, Any]:
        config: dict[str, Any] = _section_config(data)
        config["statement"]["predicate"] = predicate
        workdir: Path = tmp_path / where
        workdir.mkdir(parents=True, exist_ok=True)
        return build_and_audit(_yaml(config), fullmap=fullmap, workdir=workdir)

    legal = build("associated_with", "legal")
    forbidden = build("gene_associated_with_condition", "forbidden")

    assert legal["ok"] is True
    assert forbidden["ok"] is True  # a forbidden predicate is NEVER a build error -- that is the point
    assert legal["edge_count"] == forbidden["edge_count"] == 1
    assert legal["demoted_edge_pct"] == 0.0  # keeps GeneToDiseaseAssociation
    assert forbidden["demoted_edge_pct"] == 1.0  # demoted to bare biolink:Association


# --------------------------------------------------------------------------- #
# Actionable audit feedback: predicate_advice + multivalued_suspects + head flag
# --------------------------------------------------------------------------- #


def test_build_and_audit_report_shape_includes_advice_fields(tmp_path: Path, redb: Path) -> None:
    """Every report — success or failure — carries the advice fields with a uniform shape."""
    data: Path = _write_table(tmp_path, "brca1\tmapk1\n")
    ok = build_and_audit(_yaml(_section_config(data)), fullmap=redb, workdir=tmp_path)
    assert ok["ok"] is True
    assert ok["predicate_advice"] == []  # nothing demoted -> no advice
    assert ok["multivalued_suspects"] == []  # nothing unresolved -> no suspects
    assert ok["head"] is False

    bad = build_and_audit("::: not yaml", fullmap=redb, workdir=tmp_path)
    assert bad["ok"] is False
    assert bad["predicate_advice"] == []
    assert bad["multivalued_suspects"] == []
    assert bad["head"] is False


def test_build_and_audit_head_build_is_flagged(tmp_path: Path, redb: Path) -> None:
    """A head build is marked so its sampled edge_count is never compared against a full build's."""
    data: Path = _write_table(tmp_path, "brca1\tmapk1\n" * 10)
    result = build_and_audit(_yaml(_section_config(data)), fullmap=redb, workdir=tmp_path, head=True)
    assert result["ok"] is True
    assert result["head"] is True


def test_predicate_advice_names_the_legal_fix(tmp_path: Path) -> None:
    """End to end: a forbidden predicate yields predicate_advice naming the legal predicates.

    The same demotion as ``test_demoted_edge_pct_catches_a_predicate_its_class_forbids``, asserting
    the ACTIONABLE half: which predicate was demoted, for which category pair, and what is legal.
    """
    fullmap: Path = _gene_disease_redb(tmp_path / "fullmap")
    data: Path = _write_table(tmp_path, "brca1\tlung cancer\n")
    config: dict[str, Any] = _section_config(data)
    config["statement"]["predicate"] = "gene_associated_with_condition"  # forbidden on GeneToDiseaseAssociation

    result = build_and_audit(_yaml(config), fullmap=fullmap, workdir=tmp_path)

    assert result["ok"] is True
    assert result["demoted_edge_pct"] == 1.0
    advice = result["predicate_advice"]
    assert isinstance(advice, list)
    assert len(advice) == 1
    entry = advice[0]
    assert entry["predicate"] == "gene_associated_with_condition"
    assert entry["subject_category"] == "Gene"
    assert entry["object_category"] == "Disease"
    assert entry["association"] == "GeneToDiseaseAssociation"
    assert entry["legal_predicates"] == ["affects", "associated_with", "contributes_to"]
    assert entry["edges"] == 1

    # A legal predicate yields no advice.
    legal = build_and_audit(_yaml(_section_config(data)), fullmap=fullmap, workdir=tmp_path / "legal")
    assert legal["demoted_edge_pct"] == 0.0
    assert legal["predicate_advice"] == []


def test_multivalued_suspects_flags_joined_unresolved_terms(tmp_path: Path, redb: Path) -> None:
    """A subject column of joined cells maps as unusable blobs; the audit names the explode_by fix."""
    data: Path = _write_table(tmp_path, "brca1;mapk1\tmapk1\nmapk1;brca1\tbrca1\n")
    result = build_and_audit(_yaml(_section_config(data)), fullmap=redb, workdir=tmp_path)

    assert result["ok"] is True
    suspects = result["multivalued_suspects"]
    assert isinstance(suspects, list)
    assert len(suspects) == 1
    suspect = suspects[0]
    assert suspect["column"] == "subject"
    assert suspect["separator"] == ";"
    assert suspect["count"] == 2
    assert "explode_by" in suspect["hint"]


def test_multivalued_suspects_skips_exploded_columns(tmp_path: Path, redb: Path) -> None:
    """No suspect when the encoding already explodes: the join resolves per-entity."""
    data: Path = _write_table(tmp_path, "brca1;mapk1\tmapk1\nmapk1;brca1\tbrca1\n")
    cfg: dict[str, Any] = _section_config(data)
    cfg["statement"]["subject"]["explode_by"] = ";"
    result = build_and_audit(_yaml(cfg), fullmap=redb, workdir=tmp_path)
    assert result["ok"] is True
    assert result["coverage_pct"] == 1.0  # the join now resolves per-entity
    assert result["multivalued_suspects"] == []
    assert result["edge_count"] == 4  # 2 rows x 2 genes each


def test_predicate_advice_unit_synthetic_ndjson(tmp_path: Path) -> None:
    """_predicate_advice reads demoted edges + node categories directly, never raises on gaps."""
    from tablassert.agent import _predicate_advice

    nodes: Path = _write_jsonl(
        tmp_path / "nodes.ndjson", [{"id": "HGNC:1100", "category": ["biolink:Gene"]}, {"id": "MONDO:0008903", "category": ["biolink:Disease"]}]
    )
    edges: Path = _write_jsonl(
        tmp_path / "edges.ndjson",
        [
            {
                "subject": "HGNC:1100",
                "predicate": "biolink:gene_associated_with_condition",
                "object": "MONDO:0008903",
                "category": ["biolink:Association"],
            },
            {
                "subject": "HGNC:1100",
                "predicate": "biolink:associated_with",
                "object": "MONDO:0008903",
                "category": ["biolink:GeneToDiseaseAssociation"],
            },
        ],
    )
    advice = _predicate_advice(nodes, edges)
    assert len(advice) == 1
    assert advice[0]["predicate"] == "gene_associated_with_condition"
    legal_predicates = advice[0]["legal_predicates"]
    assert isinstance(legal_predicates, list)
    assert "affects" in legal_predicates

    # Missing artifacts and unknown node ids degrade to no advice, never an error.
    assert _predicate_advice(tmp_path / "absent.nodes", tmp_path / "absent.edges") == []
    orphan_edges: Path = _write_jsonl(
        tmp_path / "orphan.edges.ndjson",
        [{"subject": "X:1", "predicate": "biolink:related_to", "object": "Y:2", "category": ["biolink:Association"]}],
    )
    assert _predicate_advice(nodes, orphan_edges) == []
