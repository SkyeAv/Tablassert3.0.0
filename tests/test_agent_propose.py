"""Tests for US-007 ``propose_config_edit`` — deterministic, constrained NodeEncoding editor.

The core ``propose_config_edit`` tests are PURE and run in the base environment (no
``[agent]`` extra). The smolagents ``Tool`` test calls ``pytest.importorskip("smolagents")``
so it skips cleanly when the extra is absent. ``Categories`` is imported from
``tablassert.biolink`` so the assertions use the EXACT enum ``.value`` strings.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import yaml

from tablassert.agent import make_propose_config_edit_tool, propose_config_edit, validate_section
from tablassert.biolink import Categories

# ``Categories`` is built dynamically; biolink's TYPE_CHECKING stub omits ORGANISM_TAXON, so derive the
# exact ``.value`` strings once (with a waiver for the unstubbed member) and assert against these constants.
ORGANISM_TAXON: str = Categories.ORGANISM_TAXON.value  # pyright: ignore[reportAttributeAccessIssue]
GENE: str = Categories.GENE.value


def _alamv6_section() -> dict[str, Any]:
    """A valid bare merged ALAMV6-style section: subject column A, object literal CHEBI:41774.

    The subject node has NO prioritize/avoid/regex yet so the proposer has room to add them.
    """
    return {
        "source": {
            "kind": "excel",
            "local": "./DATALAKE/ALAM.XLSX",
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/instance/11708054/bin/mbio.01679-24-s0006.xlsx",
            "row_slice": [2, "auto"],
            "sheet": "all correlations",
        },
        "statement": {
            "subject": {"method": "column", "encoding": "A"},
            "predicate": "correlated_with",
            "object": {"method": "value", "encoding": "CHEBI:41774"},
        },
        "provenance": {"repo": "PMC", "publication": "PMC11708054"},
    }


def _taxonomic_report() -> dict[str, Any]:
    """A coverage report whose subject column holds unresolved taxonomic terms."""
    return {
        "overall": 0.0,
        "per_column": {
            "subject": {
                "coverage": 0.0,
                "total": 2,
                "resolved": 0,
                "unresolved": ["lactobacillus;s__rhamnosus", "g__Bacteroides"],
                "method": "column",
            },
            "object": {"coverage": 1.0, "total": 0, "resolved": 0, "unresolved": [], "method": "value"},
        },
        "unresolved": ["lactobacillus;s__rhamnosus", "g__Bacteroides"],
    }


# --------------------------------------------------------------------------- #
# PURE tests (base env; no importorskip)
# --------------------------------------------------------------------------- #


def test_propose_taxonomic_prioritizes_organism_avoids_gene() -> None:
    """Taxonomic unresolved terms -> prioritize OrganismTaxon + avoid Gene (schema-valid, changed)."""
    cfg: dict[str, Any] = _alamv6_section()
    original: str = yaml.safe_dump(cfg, sort_keys=False)
    edited, rationale = propose_config_edit(cfg, _taxonomic_report())

    assert validate_section(edited) is True
    assert edited != original

    subject = yaml.safe_load(edited)["statement"]["subject"]
    assert ORGANISM_TAXON in subject["prioritize"]
    assert GENE in subject["avoid"]
    # rationale references the knobs changed AND at least one unresolved term addressed
    assert ORGANISM_TAXON in rationale
    assert GENE in rationale
    assert "lactobacillus;s__rhamnosus" in rationale or "g__Bacteroides" in rationale


def test_propose_idempotent() -> None:
    """Re-proposing on the already-edited config with the same report does not grow the lists."""
    report: dict[str, Any] = _taxonomic_report()
    edited, _ = propose_config_edit(_alamv6_section(), report)
    edited2, _ = propose_config_edit(edited, report)

    first = yaml.safe_load(edited)["statement"]["subject"]
    second = yaml.safe_load(edited2)["statement"]["subject"]
    assert second["prioritize"] == first["prioritize"]
    assert second["avoid"] == first["avoid"]
    assert second["prioritize"].count(ORGANISM_TAXON) == 1
    assert second["avoid"].count(GENE) == 1


def test_propose_only_touches_nodeencoding() -> None:
    """Edits touch ONLY NodeEncoding fields: source/provenance/predicate (and the value object) are identical."""
    cfg: dict[str, Any] = _alamv6_section()
    edited, _ = propose_config_edit(cfg, _taxonomic_report())

    original = yaml.safe_load(yaml.safe_dump(cfg, sort_keys=False))
    parsed = yaml.safe_load(edited)
    assert parsed["source"] == original["source"]
    assert parsed["provenance"] == original["provenance"]
    assert parsed["statement"]["predicate"] == original["statement"]["predicate"]
    # the object is a method:value literal with no unresolved terms -> untouched
    assert parsed["statement"]["object"] == original["statement"]["object"]


@pytest.mark.parametrize("report", [{}, {"per_column": {}}])
def test_propose_empty_report_no_crash(report: dict[str, Any]) -> None:
    """An empty/per_column-less report yields a schema-valid config + a 'no safe edit' rationale, no exception."""
    edited, rationale = propose_config_edit(_alamv6_section(), report)
    assert validate_section(edited) is True
    assert "no safe edit" in rationale


def test_propose_odd_input_never_raises() -> None:
    """Unparseable config YAML returns a (str, str) tuple and never raises."""
    result = propose_config_edit("::: not yaml", {"per_column": {"subject": {"unresolved": ["x"], "method": "column"}}})
    assert isinstance(result, tuple)
    assert len(result) == 2
    edited, rationale = result
    assert isinstance(edited, str)
    assert isinstance(rationale, str)


def test_propose_exclude_hints() -> None:
    """A top-level exclude_prefixes report hint is extended onto the unresolved column node."""
    report: dict[str, Any] = {
        "overall": 0.0,
        "per_column": {"subject": {"coverage": 0.0, "total": 1, "resolved": 0, "unresolved": ["someunresolvedterm"], "method": "column"}},
        "unresolved": ["someunresolvedterm"],
        "exclude_prefixes": ["OMIM"],
    }
    edited, rationale = propose_config_edit(_alamv6_section(), report)
    assert validate_section(edited) is True
    assert "OMIM" in yaml.safe_load(edited)["statement"]["subject"]["exclude_prefixes"]
    assert "OMIM" in rationale


# --------------------------------------------------------------------------- #
# Tool test (requires the [agent] extra; skips cleanly when absent)
# --------------------------------------------------------------------------- #


def test_propose_tool() -> None:
    """The lazily-built tool returns JSON {config_yaml, rationale} with a schema-valid edit."""
    pytest.importorskip("smolagents")
    tool = make_propose_config_edit_tool()
    assert tool.name == "propose_config_edit"

    original: str = yaml.safe_dump(_alamv6_section(), sort_keys=False)
    payload = json.loads(tool.forward(original, json.dumps(_taxonomic_report())))
    assert "config_yaml" in payload
    assert "rationale" in payload
    assert validate_section(payload["config_yaml"]) is True
