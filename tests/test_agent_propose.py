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

from tablassert.agent import (
    llm_propose_config_edit,
    make_propose_config_edit_tool,
    propose_config_candidates,
    propose_config_edit,
    validate_section,
    validate_table_config,
)
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
            "url": ["https://pmc.ncbi.nlm.nih.gov/articles/instance/11708054/bin/mbio.01679-24-s0006.xlsx"],
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


# --------------------------------------------------------------------------- #
# W2: propose_config_candidates — ranked, distinct, idempotent
# --------------------------------------------------------------------------- #


def _taxonomic_noise_section() -> dict[str, Any]:
    """A bare section whose subject has BOTH taxonomic and noise unresolved terms."""
    return {
        "source": {"kind": "text", "local": "./d.tsv", "url": ["https://e.com/d.tsv"], "delimiter": "\t"},
        "statement": {
            "subject": {"method": "column", "encoding": "A"},
            "predicate": "associated_with",
            "object": {"method": "value", "encoding": "CHEBI:41774"},
        },
        "provenance": {"repo": "PMC", "publication": "PMC1"},
    }


def _taxonomic_noise_report() -> dict[str, Any]:
    return {
        "overall": 0.0,
        "per_column": {"subject": {"coverage": 0.0, "total": 2, "resolved": 0, "unresolved": ["g__Bacteroides", "NA control"], "method": "column"}},
        "unresolved": ["g__Bacteroides", "NA control"],
    }


def test_propose_candidates_ranked_distinct() -> None:
    """When taxonomic AND noise apply, candidates are ranked best-first and DISTINCT.

    Rank 1 is the full edit (taxonomic knobs + noise remove); the narrower single-category variants
    (taxonomic-only, noise-only) follow and differ from the full edit and each other.
    """
    cfg: dict[str, Any] = _taxonomic_noise_section()
    candidates = propose_config_candidates(cfg, _taxonomic_noise_report())

    assert len(candidates) >= 2
    yamls: list[str] = [c[0] for c in candidates]
    assert len(set(yamls)) == len(yamls), "candidates must be distinct"

    # Rank 1 (full edit) has BOTH a taxonomic prioritize and a noise remove...
    full: dict[str, Any] = yaml.safe_load(candidates[0][0])
    assert ORGANISM_TAXON in full["statement"]["subject"]["prioritize"]
    assert full["statement"]["subject"]["remove"]
    # ...while a narrower variant drops one category (taxonomic-only has no remove).
    taxonomic_only = [c for c in candidates if "taxonomic-only" in c[1]]
    assert taxonomic_only
    tax_only_subject: dict[str, Any] = yaml.safe_load(taxonomic_only[0][0])["statement"]["subject"]
    assert ORGANISM_TAXON in tax_only_subject["prioritize"]
    assert "remove" not in tax_only_subject


def test_propose_candidates_idempotent() -> None:
    """Re-proposing on the full edit yields NO new candidates (every knob already present)."""
    cfg: dict[str, Any] = _taxonomic_noise_section()
    candidates = propose_config_candidates(cfg, _taxonomic_noise_report())
    assert candidates
    full_yaml: str = candidates[0][0]
    # The full edit already carries every applicable knob, so re-proposing finds nothing to add.
    assert propose_config_candidates(full_yaml, _taxonomic_noise_report()) == []


def test_propose_candidates_empty_when_no_safe_edit() -> None:
    """A config with no unresolved terms yields no candidates."""
    cfg: dict[str, Any] = _taxonomic_noise_section()
    report: dict[str, Any] = {
        "overall": 1.0,
        "per_column": {"subject": {"coverage": 1.0, "total": 1, "resolved": 1, "unresolved": [], "method": "column"}},
        "unresolved": [],
    }
    assert propose_config_candidates(cfg, report) == []


# --------------------------------------------------------------------------- #
# W1: llm_propose_config_edit — tier-2 reflexion (offline, fake callable model)
# --------------------------------------------------------------------------- #


def _revised_section(new_predicate: str = "correlated_with") -> str:
    """A schema-valid revised config that changes the predicate (a change the deterministic proposer never makes)."""
    cfg: dict[str, Any] = _taxonomic_noise_section()
    cfg["statement"]["predicate"] = new_predicate
    return yaml.safe_dump(cfg, sort_keys=False)


def test_llm_propose_returns_valid_revised_config() -> None:
    """A reflexion model returning a valid revised config (predicate changed) is accepted + gated valid."""
    revised: str = _revised_section("correlated_with")
    result = llm_propose_config_edit(_taxonomic_noise_section_yaml(), _taxonomic_noise_report(), "context", model=lambda prompt: revised)
    assert result is not None
    assert validate_table_config(result)
    assert yaml.safe_load(result)["statement"]["predicate"] == "correlated_with"


def test_llm_propose_extracts_fenced_yaml() -> None:
    """A reflexion model wrapping the config in a ```yaml fence is still extracted + validated."""
    revised: str = _revised_section("interacts_with")
    fenced: str = f"Here is the revised config:\n```yaml\n{revised}\n```\n"
    result = llm_propose_config_edit(_taxonomic_noise_section_yaml(), _taxonomic_noise_report(), "context", model=lambda prompt: fenced)
    assert result is not None
    assert yaml.safe_load(result)["statement"]["predicate"] == "interacts_with"


def test_llm_propose_invalid_returns_none() -> None:
    """A reflexion model returning an invalid config yields None (caller keeps the current best)."""
    result = llm_propose_config_edit(
        _taxonomic_noise_section_yaml(), _taxonomic_noise_report(), "context", model=lambda prompt: "definitely not a config"
    )
    assert result is None


def test_llm_propose_never_raises() -> None:
    """A reflexion model that raises is swallowed -> None (reflexion must never abort the caller)."""

    def boom(prompt: str) -> str:
        raise RuntimeError("synthetic model failure")

    assert llm_propose_config_edit(_taxonomic_noise_section_yaml(), _taxonomic_noise_report(), "context", model=boom) is None


def _taxonomic_noise_section_yaml() -> str:
    return yaml.safe_dump(_taxonomic_noise_section(), sort_keys=False)
