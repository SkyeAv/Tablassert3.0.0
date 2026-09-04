"""Tests for US-005: deterministic config compaction + config-size metric.

``compact_config`` shrinks a VALID table config by removing ONLY provably no-op entries —
nulls whose model default is null, empty lists whose model default is empty, and explicit
values equal to a verified Pydantic model default — while preserving semantic non-default
nulls (``taxon: null``), ``nullable: true``, the ``kind`` union discriminator, and every
provenance value. Any invalid input comes back byte-identical. The supervisor integration
(compacted best-config write + ``config_chars`` state metric) is covered in
``test_agent_supervisor.py``; the build equivalence test here drives the REAL pipeline via
``build_and_audit`` against the tiny real ``rs.build_fullmap_db`` redb from the US-007
edge-count fixtures and asserts identical node ids, identical ``(subject, predicate,
object)`` triples, and equal coverage before vs after compaction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from tablassert.agent import _compact_model_dict, _expand_sections, build_and_audit, compact_config, config_size_metric, validate_table_config
from tablassert.models import ManualProvenance, Section
from tests.test_agent_edgecount import FIXTURE_DIR, PAYLOAD, _build_real_redb, _load_config_with_absolute_local


def _flat_config(**subject_extra: Any) -> dict[str, Any]:
    """A minimal valid flat (single-section) Excel config with room for subject extras."""
    return {
        "source": {"kind": "excel", "local": "/tmp/table.xlsx", "url": ["https://e.com/table.xlsx"], "sheet": "Sheet1"},
        "statement": {
            "subject": {"method": "column", "encoding": "A", **subject_extra},
            "predicate": "related_to",
            "object": {"method": "column", "encoding": "B"},
        },
        "provenance": {"repo": "PMC", "publication": "PMC1"},
    }


def _dump(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False)


def _load(yaml_text: str) -> dict[str, Any]:
    loaded: object = yaml.safe_load(yaml_text)
    assert isinstance(loaded, dict)
    return loaded


def _merged_sections(config_yaml: str) -> list[Section]:
    """Validate every merged section of a config exactly the way the W3 gate does."""
    data: dict[str, Any] = _load(config_yaml)
    return [Section.model_validate(section) for section in _expand_sections(data)]


# --------------------------------------------------------------------------- #
# Default / null / empty removal — flat single-section configs
# --------------------------------------------------------------------------- #


def test_compact_removes_explicit_defaults() -> None:
    """Explicit values equal to verified model defaults vanish; non-defaults stay."""
    cfg: dict[str, Any] = _flat_config(taxon=9606)
    cfg["statement"]["predicate"] = "related_to"  # Statement.predicate default
    cfg["statement"]["qualifiers"] = [{"qualifier": "object_direction_qualifier", "method": "column", "encoding": "C", "taxon": 9606}]
    cfg["annotations"] = [{"annotation": "p_value", "method": "value", "encoding": 0.01}]
    original: str = _dump(cfg)
    compacted: dict[str, Any] = _load(compact_config(original))

    assert "taxon" not in compacted["statement"]["subject"], "subject.taxon == default 9606 must be removed"
    assert "predicate" not in compacted["statement"], "predicate == default related_to must be removed"
    assert "method" not in compacted["annotations"][0], "annotation method == default value must be removed"
    assert "taxon" not in compacted["statement"]["qualifiers"][0], "qualifier taxon == default 9606 must be removed"
    # Non-default values survive.
    assert "sheet" not in compacted["source"], "sheet == default Sheet1 must be removed"
    assert compacted["source"]["kind"] == "excel"
    assert compacted["statement"]["subject"]["encoding"] == "A"
    assert compacted["annotations"][0]["encoding"] == 0.01
    assert validate_table_config(_dump(compacted))


def test_compact_removes_sheet_and_delimiter_defaults_but_keeps_kind() -> None:
    """``sheet: Sheet1`` and ``delimiter: ','`` equal defaults; ``kind`` is never dropped."""
    excel: dict[str, Any] = _flat_config()
    excel_compacted: dict[str, Any] = _load(compact_config(_dump(excel)))
    assert "sheet" not in excel_compacted["source"], "sheet == default Sheet1 must be removed"
    assert excel_compacted["source"]["kind"] == "excel", "kind discriminates the source union and must stay"

    text: dict[str, Any] = _flat_config()
    text["source"] = {"kind": "text", "local": "/tmp/table.tsv", "url": ["https://e.com/table.tsv"], "delimiter": ","}
    text_compacted: dict[str, Any] = _load(compact_config(_dump(text)))
    assert "delimiter" not in text_compacted["source"], "delimiter == default ',' must be removed"
    assert text_compacted["source"]["kind"] == "text"


def test_compact_removes_nulls_only_when_null_is_the_default() -> None:
    """``rows: null`` matches its None default; ``taxon: null`` deliberately DISABLES the
    default taxon (9606) and must survive — the models' own tests rely on that meaning."""
    cfg: dict[str, Any] = _flat_config()
    cfg["source"]["rows"] = None
    cfg["statement"]["subject"]["fill"] = None
    cfg["statement"]["object"]["taxon"] = None
    compacted: dict[str, Any] = _load(compact_config(_dump(cfg)))

    assert "rows" not in compacted["source"], "rows: null equals the None default -> removed"
    assert "fill" not in compacted["statement"]["subject"], "fill: null equals the None default -> removed"
    assert compacted["statement"]["object"]["taxon"] is None, "taxon: null is a non-default semantic value -> kept"


def test_compact_preserves_nullable_true_and_qualifier_semantics() -> None:
    """``nullable: true`` changes row-drop behavior and stays; ``nullable: false`` is the default and goes."""
    cfg: dict[str, Any] = _flat_config()
    cfg["statement"]["qualifiers"] = [
        {"qualifier": "object_direction_qualifier", "method": "column", "encoding": "C", "nullable": True},
        {"qualifier": "disease_context_qualifier", "method": "column", "encoding": "D", "nullable": False},
    ]
    qualifiers: list[dict[str, Any]] = _load(compact_config(_dump(cfg)))["statement"]["qualifiers"]
    assert qualifiers[0]["nullable"] is True, "nullable: true is meaningful and must be preserved"
    assert "nullable" not in qualifiers[1], "nullable: false equals the default -> removed"


def test_compact_empty_list_only_when_default_is_empty() -> None:
    """``qualifiers: []`` differs from its None default -> kept; an empty list whose model
    default IS empty (``ManualProvenance.upstream_resource_ids``, default_factory=list) is removed."""
    cfg: dict[str, Any] = _flat_config()
    cfg["statement"]["qualifiers"] = []
    compacted: dict[str, Any] = _load(compact_config(_dump(cfg)))
    assert compacted["statement"]["qualifiers"] == [], "qualifiers default is None, not empty -> [] must stay"

    # White-box: the only empty-list-default field in the Section tree lives under the never-touched
    # provenance subtree, so the rule itself is proven directly against its model.
    compacted_prov: dict[str, Any] = _compact_model_dict(
        {"upstream_resource_ids": [], "knowledge_level": "statistical_association"}, ManualProvenance, None
    )
    assert "upstream_resource_ids" not in compacted_prov, "empty list equal to the model's empty default -> removed"
    assert "knowledge_level" not in compacted_prov, "knowledge_level equals its default -> removed"


def test_compact_never_touches_provenance() -> None:
    """Provenance is the edge's legal attribution: values equal to model defaults stay verbatim."""
    cfg: dict[str, Any] = _flat_config()
    cfg["provenance"] = {
        "repo": "PMC",  # Repositories.PUBMED_CENTRAL default — still must NOT be removed
        "publication": "PMC1",
        "knowledge_level": "statistical_association",  # default — still must NOT be removed
        "agent_type": "data_analysis_pipeline",  # default — still must NOT be removed
    }
    compacted: dict[str, Any] = _load(compact_config(_dump(cfg)))
    assert compacted["provenance"] == cfg["provenance"], "no provenance value may ever be compacted away"


# --------------------------------------------------------------------------- #
# Multi-section {template, sections} handling
# --------------------------------------------------------------------------- #


def _multi_config() -> dict[str, Any]:
    return {
        "template": {
            "provenance": {"repo": "PMC", "publication": "PMC10766526"},
            # A NON-default template value: any section entry equal to the model default at the
            # same path is NOT a no-op (fastmerge would fall back to the template's null).
            "statement": {"subject": {"taxon": None}, "predicate": "associated_with"},
        },
        "sections": [
            {
                "source": {"kind": "excel", "local": "/tmp/p.xlsx", "url": ["https://e.com/p.xlsx"], "sheet": "Sheet1"},
                "statement": {
                    "subject": {"method": "column", "encoding": "A", "taxon": 9606},  # blocked by template taxon: null
                    "object": {"method": "column", "encoding": "B"},
                },
                "annotations": [
                    {"annotation": "effect_size", "method": "column", "encoding": "C"},
                    {"annotation": "effect_type", "method": "value", "encoding": "regression_coefficient"},
                ],
            },
            {
                "source": {"kind": "text", "local": "/tmp/p.tsv", "url": ["https://e.com/p.tsv"], "delimiter": ","},
                "statement": {
                    "subject": {"method": "column", "encoding": "A"},
                    "object": {"method": "column", "encoding": "B", "taxon": 10090},  # non-default -> kept
                },
            },
        ],
    }


def test_compact_multi_section_with_template_protection() -> None:
    """Section entries equal to a default are removed unless the template carries a differing
    value at the same path; merged sections stay semantically identical."""
    original: str = _dump(_multi_config())
    compacted_yaml: str = compact_config(original)
    compacted: dict[str, Any] = _load(compacted_yaml)

    first: dict[str, Any] = compacted["sections"][0]
    assert first["statement"]["subject"]["taxon"] == 9606, "template's taxon: null blocks removing the section's default 9606"
    assert "sheet" not in first["source"], "no template value at source.sheet -> Sheet1 default removed"
    assert "method" not in first["annotations"][1], "annotation method: value is a default inside a list element"
    second: dict[str, Any] = compacted["sections"][1]
    assert "delimiter" not in second["source"]
    assert second["statement"]["object"]["taxon"] == 10090
    assert compacted["template"]["provenance"] == _multi_config()["template"]["provenance"]
    assert compacted["template"]["statement"]["subject"]["taxon"] is None, "template's semantic null survives"

    assert _merged_sections(original) == _merged_sections(compacted_yaml), "merged sections must be identical"


def test_compact_flat_and_template_only_shapes() -> None:
    """Both one-section shapes compact: a bare flat section and a ``{template}``-only config."""
    flat: str = _dump(_flat_config(taxon=9606))
    assert "taxon" not in _load(compact_config(flat))["statement"]["subject"]

    template_only: str = _dump({"template": _flat_config(taxon=9606)})
    compacted: dict[str, Any] = _load(compact_config(template_only))
    assert "taxon" not in compacted["template"]["statement"]["subject"]
    assert validate_table_config(_dump(compacted))


# --------------------------------------------------------------------------- #
# Failure behavior: invalid input unchanged, idempotence
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad_input",
    [
        "{{{ not yaml at all",
        "[1, 2, 3]",  # not a mapping
        _dump({"source": {"kind": "excel"}}),  # missing statement/provenance
        _dump({"template": {"provenance": {"publication": "PMC1"}}, "sections": "not-a-list"}),
        _dump(
            {
                "source": {"kind": "excel", "local": "/x.xlsx", "url": ["https://e.com/x"], "unknown_key": 1},
                "statement": {"subject": {"encoding": "A"}, "object": {"encoding": "B"}},
                "provenance": {"publication": "PMC1"},
            }
        ),
    ],
)
def test_compact_invalid_input_returned_unchanged(bad_input: str) -> None:
    """Any YAML/validation failure returns the EXACT input: never corrupt, never raise."""
    assert compact_config(bad_input) == bad_input


def test_compact_is_idempotent_and_deterministic() -> None:
    """Compacting a compacted config changes nothing further (flat and multi-section)."""
    for config in (_dump(_flat_config(taxon=9606)), _dump(_multi_config())):
        once: str = compact_config(config)
        assert compact_config(once) == once, "compaction must be idempotent"
        assert compact_config(config) == once, "compaction must be deterministic"


def test_compact_semantic_equivalence_via_validated_models() -> None:
    """The compacted config validates to the SAME Section models as the original (flat shape)."""
    original: str = _dump(_flat_config(taxon=9606))
    assert _merged_sections(original) == _merged_sections(compact_config(original))


# --------------------------------------------------------------------------- #
# config_size_metric
# --------------------------------------------------------------------------- #


def test_config_size_metric_counts_sections() -> None:
    flat: str = _dump(_flat_config())
    assert config_size_metric(flat) == {"chars": len(flat), "sections": 1}

    multi: str = _dump(_multi_config())
    assert config_size_metric(multi) == {"chars": len(multi), "sections": 2}

    template_only: str = _dump({"template": _flat_config()})
    assert config_size_metric(template_only)["sections"] == 1, "a template-only config expands to one section"


def test_config_size_metric_tolerates_malformed_shapes() -> None:
    """Documented deterministic fallbacks: never raise, always carry the exact char count."""
    garbage: str = "{{{ not yaml"
    assert config_size_metric(garbage) == {"chars": len(garbage), "sections": 0}
    not_a_mapping: str = "[1, 2, 3]"
    assert config_size_metric(not_a_mapping) == {"chars": len(not_a_mapping), "sections": 0}
    bad_sections: str = _dump({"template": {}, "sections": "oops"})
    assert config_size_metric(bad_sections)["sections"] == 0
    empty_sections: str = _dump({"template": _flat_config(), "sections": []})
    assert config_size_metric(empty_sections)["sections"] == 0


# --------------------------------------------------------------------------- #
# Build equivalence: the REAL pipeline produces the IDENTICAL KG before/after
# --------------------------------------------------------------------------- #


def _ndjson_triples(path: str) -> set[tuple[str, str, str]]:
    return {(edge["subject"], edge["predicate"], edge["object"]) for edge in map(json.loads, Path(path).read_text().splitlines()) if edge}


def _ndjson_node_ids(path: str) -> set[str]:
    return {json.loads(line)["id"] for line in Path(path).read_text().splitlines() if line.strip()}


@pytest.fixture(scope="module")
def tiny_fullmap(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The tiny REAL redb from the US-007 edge-count fixtures (module-scoped: built once)."""
    return _build_real_redb(tmp_path_factory.mktemp("compact-fullmap"))


@pytest.mark.parametrize("shape", ["multi", "flat"])
def test_compaction_is_build_equivalent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_fullmap: Path, shape: str) -> None:
    """Compacted config builds the IDENTICAL knowledge graph: same node ids, same
    ``(subject, predicate, object)`` triples, same coverage — via ``build_and_audit``."""
    monkeypatch.chdir(tmp_path)  # the edge-count harness contract: builds run from a clean cwd
    config: dict[str, Any] = _load_config_with_absolute_local(FIXTURE_DIR / "agent_config.yaml", PAYLOAD)
    if shape == "flat":
        merged: dict[str, Any] = dict(config["sections"][0])
        merged["provenance"] = config["template"]["provenance"]
        config = merged
    original: str = _dump(config)
    compacted: str = compact_config(original)
    assert compacted != original, "the fixture must exercise at least one removable default"
    assert len(compacted) < len(original), "compaction must shrink the config"

    before: dict[str, object] = build_and_audit(original, fullmap=tiny_fullmap, name="COMPACT_EQ", workdir=tmp_path / "before")
    after: dict[str, object] = build_and_audit(compacted, fullmap=tiny_fullmap, name="COMPACT_EQ", workdir=tmp_path / "after")
    assert before["ok"], f"the original config must build: {before['errors']}"
    assert after["ok"], f"the compacted config must build: {after['errors']}"

    assert before["edge_count"] == after["edge_count"], "compaction must not change the edge count"
    assert isinstance(before["edge_count"], int), "the report edge count must be an int"
    assert before["edge_count"] > 0, "the fixture build must emit edges (meaningful equivalence)"
    assert before["node_count"] == after["node_count"], "compaction must not change the node count"
    assert before["coverage_pct"] == after["coverage_pct"], "compaction must not change fullmap coverage"
    assert _ndjson_node_ids(str(before["kgx_path"])) == _ndjson_node_ids(str(after["kgx_path"])), "node id sets must be identical"
    assert _ndjson_triples(str(before["edges_path"])) == _ndjson_triples(str(after["edges_path"])), "edge triples must be identical"
