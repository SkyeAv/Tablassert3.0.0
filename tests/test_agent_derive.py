"""Tests for US-004 ``derive_config`` tool + ``validate_section`` final-answer gate.

The ``validate_section`` / ``section_json_schema`` tests are PURE and run in the
base environment (no ``[agent]`` extra). The smolagents ``Tool`` object tests call
``pytest.importorskip("smolagents")`` so they skip cleanly when the extra is absent.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from tablassert.agent import make_derive_config_tool, section_json_schema, table_config_error, validate_section
from tablassert.models import Section

FIXTURES: Path = Path(__file__).parent / "fixtures"

# The ALAMV6 table config from docs/configuration/advanced-example.md (template shape),
# exercising the ``template`` branch of validate_section via to_sections + fastmerge.
# Kept on the OLD annotation names (sample_size / relationship_strength) on purpose:
# build-time coercion renames them to study_size / effect_size, so this doubles as the
# legacy-config backward-compatibility case. `effect_type` has no legacy
# spelling, so it stays canonical -- and it is mandatory here, since the legacy
# `relationship_strength` coerces to `effect_size` and the two must be declared as a pair.
ALAMV6_TEMPLATE: dict[str, Any] = {
    "template": {
        "source": {
            "kind": "excel",
            "local": "./DATALAKE/ALAM.XLSX",
            "url": ["https://pmc.ncbi.nlm.nih.gov/articles/instance/11708054/bin/mbio.01679-24-s0006.xlsx"],
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
            {"annotation": "effect_type", "method": "value", "encoding": "spearmans_rho"},
        ],
    }
}


def _minimal_section() -> dict[str, Any]:
    """A valid bare merged section dict (mirrors tests/fixtures/minimal_section.yaml)."""
    return {
        "source": {"url": ["https://example.com/test.tsv"], "local": "./test.tsv", "kind": "text", "delimiter": "\t"},
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
    """forward returns a VALID candidate YAML unchanged (the gate does the validating)."""
    pytest.importorskip("smolagents")
    tool = make_derive_config_tool()
    valid: str = yaml.safe_dump(ALAMV6_TEMPLATE, sort_keys=False)
    assert tool.forward(valid) == valid


def test_derive_config_tool_forward_returns_the_coded_error_for_an_invalid_config() -> None:
    """An invalid config comes back as its coded error, not silently forwarded.

    The final-answer gate can only answer True/False, so this is the ONLY channel through which the
    model sees the actionable text the coded errors were written to carry.
    """
    pytest.importorskip("smolagents")
    tool = make_derive_config_tool()

    # Structurally wrong: not a Section at all.
    assert tool.forward("foo: bar").startswith("INVALID CONFIG (not forwarded):")

    # A coded Biolink error reaches the agent verbatim, slug and all.
    bad_qualifier: dict[str, Any] = copy.deepcopy(ALAMV6_TEMPLATE)
    bad_qualifier["template"]["statement"]["qualifiers"] = [{"qualifier": "object_direction_qualifier", "method": "value", "encoding": "way up"}]
    message: str = tool.forward(yaml.safe_dump(bad_qualifier, sort_keys=False))
    assert "qualifier-bad-value" in message
    assert "Permitted values include" in message


def test_derive_config_tool_description_mentions_schema_gate() -> None:
    """The description tells the agent the output is schema-gated."""
    pytest.importorskip("smolagents")
    tool = make_derive_config_tool()
    assert "schema" in tool.description.lower()


def test_derive_config_tool_description_carries_us006_guidance() -> None:
    """US-006: the tool description teaches header/row_slice, explode_by, prioritize breadth,
    p_value capture + the effect_size/effect_type pair (drop-with-warning), one section per sheet."""
    pytest.importorskip("smolagents")
    tool = make_derive_config_tool()
    description: str = tool.description
    assert "row_slice: [<first data row>, auto]" in description  # (a) header-row detection
    assert "EXACT sheet name" in description
    assert "explode_by" in description  # (b) delimited multi-entity cells
    assert "EVERY plausible biolink Category" in description  # (c) prioritize breadth
    assert "p_value" in description  # (d) statistics capture
    assert "effect_size (method: column)" in description
    assert "effect_type (method: value)" in description
    assert "dropped with a warning" in description  # US-001 pairing semantics
    assert "ONE section per mappable" in description  # (e) one section per sheet


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


def test_table_config_error_returns_the_actionable_message_the_gate_swallows() -> None:
    """The gates stay boolean (smolagents' contract) but the REASON is no longer thrown away."""
    valid: str = yaml.safe_dump(ALAMV6_TEMPLATE, sort_keys=False)
    assert table_config_error(valid) is None
    assert validate_section(valid) is True

    # `direction_qualifier` is declared in the LinkML schema but attached to no Pydantic class.
    unsatisfiable: dict[str, Any] = copy.deepcopy(ALAMV6_TEMPLATE)
    unsatisfiable["template"]["statement"]["qualifiers"] = [{"qualifier": "direction_qualifier", "method": "value", "encoding": "increased"}]
    message: str | None = table_config_error(yaml.safe_dump(unsatisfiable, sort_keys=False))
    assert message is not None
    assert "qualifier-unsatisfiable" in message
    assert "Use a concrete subtype" in message
    # Still False, still never raises -- only the reason is newly available.
    assert validate_section(yaml.safe_dump(unsatisfiable, sort_keys=False)) is False

    # A permanently disabled field is rejected through both supported config entry points.
    disabled_qualifier: dict[str, Any] = copy.deepcopy(ALAMV6_TEMPLATE)
    disabled_qualifier["template"]["statement"]["qualifiers"] = [
        {"qualifier": "species_context_qualifier", "method": "value", "encoding": "NCBITaxon:9606"}
    ]
    disabled_message: str | None = table_config_error(yaml.safe_dump(disabled_qualifier, sort_keys=False))
    assert disabled_message is not None
    assert "field-disabled" in disabled_message
    assert validate_section(yaml.safe_dump(disabled_qualifier, sort_keys=False)) is False

    disabled_annotation: dict[str, Any] = copy.deepcopy(ALAMV6_TEMPLATE)
    disabled_annotation["template"]["annotations"] = [{"annotation": "species_context_qualifier", "method": "value", "encoding": "NCBITaxon:9606"}]
    annotation_message: str | None = table_config_error(yaml.safe_dump(disabled_annotation, sort_keys=False))
    assert annotation_message is not None
    assert "field-disabled" in annotation_message
    assert validate_section(yaml.safe_dump(disabled_annotation, sort_keys=False)) is False

    # Never raises, whatever it is handed.
    for nasty in ("", "[]", "{", "\x00", "a: [1, 2", "- - -"):
        assert table_config_error(nasty) is None or isinstance(table_config_error(nasty), str)
