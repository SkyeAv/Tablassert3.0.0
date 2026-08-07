"""Tests for US-004 ``derive_config`` tool + ``validate_section`` final-answer gate.

The ``validate_section`` / ``section_json_schema`` tests are PURE and run in the
base environment (no ``[agent]`` extra). The smolagents ``Tool`` object tests call
``pytest.importorskip("smolagents")`` so they skip cleanly when the extra is absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from tablassert.agent import make_derive_config_tool, section_json_schema, validate_section
from tablassert.models import Section

FIXTURES: Path = Path(__file__).parent / "fixtures"

# The ALAMV6 table config from docs/configuration/advanced-example.md (template shape),
# exercising the ``template`` branch of validate_section via to_sections + fastmerge.
# Kept on the OLD annotation names (sample_size / relationship_strength) on purpose:
# build-time coercion renames them to supporting_study_size / effect_size, so this
# doubles as the legacy-config backward-compatibility case.
ALAMV6_TEMPLATE: dict[str, Any] = {
    "template": {
        "source": {
            "kind": "excel",
            "local": "./DATALAKE/ALAM.XLSX",
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/instance/11708054/bin/mbio.01679-24-s0006.xlsx",
            "row_slice": [2, "auto"],
            "sheet": "all correlations",
        },
        "statement": {
            "subject": {
                "method": "column",
                "encoding": "A",
                "prioritize": ["OrganismTaxon"],
                "avoid": ["Gene"],
                "remove": ["^NA "],
                "regex": [{"pattern": ".*g__", "replacement": ""}, {"pattern": ";s__", "replacement": " "}, {"pattern": "sp", "replacement": "sp. "}],
            },
            "predicate": "correlated_with",
            "object": {"method": "value", "encoding": "CHEBI:41774"},
        },
        "provenance": {"repo": "PMC", "publication": "PMC11708054"},
        "annotations": [
            {"annotation": "sample_size", "method": "value", "encoding": 9},
            {"annotation": "p_value", "method": "column", "encoding": "C"},
            {"annotation": "relationship_strength", "method": "column", "encoding": "B"},
        ],
    }
}


def _minimal_section() -> dict[str, Any]:
    """A valid bare merged section dict (mirrors tests/fixtures/minimal_section.yaml)."""
    return {
        "source": {"url": "https://example.com/test.tsv", "local": "./test.tsv", "kind": "text", "delimiter": "\t"},
        "statement": {"subject": {"method": "value", "encoding": "BRCA1"}, "object": {"method": "value", "encoding": "TP53"}},
        "provenance": {"repo": "PMC", "publication": "PMC0000000"},
    }


# --------------------------------------------------------------------------- #
# PURE tests (base env; no importorskip)
# --------------------------------------------------------------------------- #


def test_validate_section_accepts_minimal_fixture() -> None:
    """The on-disk minimal section fixture validates as a bare merged section."""
    text: str = (FIXTURES / "minimal_section.yaml").read_text()
    assert validate_section(text) is True


def test_validate_section_accepts_alamv6_template() -> None:
    """A ``{template: {...}}`` table config validates via the to_sections branch."""
    cfg: str = yaml.safe_dump(ALAMV6_TEMPLATE, sort_keys=False)
    assert validate_section(cfg) is True


def test_validate_section_rejects_missing_source() -> None:
    """A section without a ``source`` is rejected (required field)."""
    section: dict[str, Any] = _minimal_section()
    del section["source"]
    assert validate_section(yaml.safe_dump(section)) is False


def test_validate_section_rejects_bad_enum() -> None:
    """An out-of-enum predicate is rejected."""
    section: dict[str, Any] = _minimal_section()
    section["statement"]["predicate"] = "not_a_predicate"
    assert validate_section(yaml.safe_dump(section)) is False


def test_validate_section_rejects_unknown_key() -> None:
    """An extra key is rejected (Section uses extra='forbid')."""
    section: dict[str, Any] = _minimal_section()
    section["bogus"] = 1
    assert validate_section(yaml.safe_dump(section)) is False


@pytest.mark.parametrize("cfg", ["::: not yaml :::", "[", ""])
def test_validate_section_rejects_not_yaml(cfg: str) -> None:
    """Non-dict / unparseable YAML is rejected, never raised."""
    assert validate_section(cfg) is False


@pytest.mark.parametrize("cfg", ["null", "[]", "42", "just a string"])
def test_validate_section_never_raises(cfg: str) -> None:
    """Nasty but parseable inputs return False without raising."""
    assert validate_section(cfg) is False


def test_section_json_schema_matches_model() -> None:
    """section_json_schema mirrors Section.model_json_schema and exposes the core props."""
    schema: dict[str, object] = section_json_schema()
    assert schema == Section.model_json_schema()
    assert "properties" in schema
    properties = schema["properties"]
    assert isinstance(properties, dict)
    assert {"source", "statement", "provenance"} <= set(properties)


# --------------------------------------------------------------------------- #
# Tool object tests (require the [agent] extra; skip cleanly when absent)
# --------------------------------------------------------------------------- #


def test_derive_config_tool_exposes_section_schema() -> None:
    """The lazily-built tool is named derive_config and carries the Section schema."""
    pytest.importorskip("smolagents")
    tool = make_derive_config_tool()
    assert tool.name == "derive_config"
    assert isinstance(tool.output_schema, dict)
    assert tool.output_schema == Section.model_json_schema()


def test_derive_config_tool_forward_passthrough() -> None:
    """forward returns the candidate YAML unchanged (the gate does the validating)."""
    pytest.importorskip("smolagents")
    tool = make_derive_config_tool()
    assert tool.forward("foo: bar") == "foo: bar"


def test_derive_config_tool_description_mentions_schema_gate() -> None:
    """The description tells the agent the output is schema-gated."""
    pytest.importorskip("smolagents")
    tool = make_derive_config_tool()
    assert "schema" in tool.description.lower()


def test_validate_section_never_raises_on_empty_sections() -> None:
    """A ``template: {}`` config with an explicit empty ``sections: []`` returns False, not a raise.

    Regression (review fix 1): ``_merge_first_section`` used to do ``sections[0]`` on the empty
    list -> IndexError, which was NOT in validate_section's except tuple. Because validate_section is
    a smolagents final_answer_checks gate, a raise becomes a hard AgentError that TERMINATES the inner
    run instead of a clean False the agent can recover from. The gate must NEVER raise.
    """
    assert validate_section("template: {}\nsections: []\n") is False
    assert validate_section("template: {}\n") is False  # no sections key -> merges empty template -> invalid
    assert validate_section("template: {}\nsections: []\n") is False  # never raises
