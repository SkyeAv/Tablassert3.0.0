"""Tests for US-007 ``propose_config_edit`` — deterministic, constrained config editor.

Every test here is PURE and runs in the base environment (no ``[agent]`` extra): the proposer,
its ranked ``propose_config_candidates``, and the tier-2 ``llm_propose_config_edit`` are all
offline. US-002 removed the LLM tool wrapper — the supervisor now drives the proposer
deterministically, so no smolagents ``Tool`` test remains. ``Categories`` is imported from
``tablassert.biolink`` so the assertions use the EXACT enum ``.value`` strings.
"""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from tablassert.agent import llm_propose_config_edit, propose_config_candidates, propose_config_edit, validate_section, validate_table_config
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


# --------------------------------------------------------------------------- #
# explode_by rule — joined multi-valued cells in unresolved terms
# --------------------------------------------------------------------------- #


def _joined_section() -> dict[str, Any]:
    """A bare gene~disease section whose subject column joins several genes per cell."""
    return {
        "source": {"kind": "text", "local": "./d.tsv", "url": ["https://e.com/d.tsv"], "delimiter": "\t"},
        "statement": {
            "subject": {"method": "column", "encoding": "A", "prioritize": [GENE]},
            "predicate": "affects",
            "object": {"method": "column", "encoding": "B", "prioritize": ["Disease"]},
        },
        "provenance": {"repo": "PMC", "publication": "PMC1"},
    }


def _joined_report(unresolved: list[str]) -> dict[str, Any]:
    return {
        "overall": 0.0,
        "per_column": {
            "subject": {"coverage": 0.0, "total": len(unresolved), "resolved": 0, "unresolved": unresolved, "method": "column"},
            "object": {"coverage": 1.0, "total": 1, "resolved": 1, "unresolved": [], "method": "column"},
        },
        "unresolved": unresolved,
    }


def test_propose_explode_by_for_joined_terms() -> None:
    """Unresolved terms joining entities with ';' -> explode_by: \";\" on that node (schema-valid)."""
    edited, rationale = propose_config_edit(_joined_section(), _joined_report(["brca1;tp53", "pten;kras"]))

    assert validate_section(edited) is True
    subject = yaml.safe_load(edited)["statement"]["subject"]
    assert subject["explode_by"] == ";"
    assert "explode_by" in rationale
    # A joined cell is NOT taxonomic: the explode rule owns it, no OrganismTaxon misfire.
    assert "prioritize" not in subject or ORGANISM_TAXON not in subject.get("prioritize", [])
    assert "avoid" not in subject


def test_propose_explode_skips_when_already_declared() -> None:
    """A node that already has explode_by is left alone (idempotent)."""
    cfg: dict[str, Any] = _joined_section()
    cfg["statement"]["subject"]["explode_by"] = "|"
    edited, _ = propose_config_edit(cfg, _joined_report(["brca1;tp53", "pten;kras"]))
    assert yaml.safe_load(edited)["statement"]["subject"]["explode_by"] == "|"


def test_propose_explode_ignores_lineage_strings() -> None:
    """Lineage strings carry ';' but are ONE entity: taxonomic knobs fire, explode_by does not."""
    report = _joined_report(["d__bacteria;p__firmicutes;g__escherichia", "d__bacteria;g__bacillus"])
    edited, _ = propose_config_edit(_joined_section(), report)
    subject = yaml.safe_load(edited)["statement"]["subject"]
    assert "explode_by" not in subject
    assert ORGANISM_TAXON in subject["prioritize"]


def test_propose_explode_comma_needs_three_hits() -> None:
    """Commas legitimately appear inside disease names: two comma-terms do not fire, three do."""
    cfg: dict[str, Any] = _joined_section()
    edited, _ = propose_config_edit(cfg, _joined_report(["smith, john", "doe, jane"]))
    assert "explode_by" not in yaml.safe_load(edited)["statement"]["subject"]

    edited, _ = propose_config_edit(cfg, _joined_report(["brca1,tp53", "pten,kras", "egfr,myc"]))
    assert yaml.safe_load(edited)["statement"]["subject"]["explode_by"] == ","


def test_propose_candidates_include_explode_only_variant() -> None:
    """The ranked candidates carry an explode-only narrow variant alongside the full edit.

    The report mixes joined terms with a noise term so the full edit (explode + remove) differs
    from the explode-only variant; otherwise dedup would collapse them.
    """
    candidates = propose_config_candidates(_joined_section(), _joined_report(["brca1;tp53", "pten;kras", "NA control"]))
    assert candidates
    full: dict[str, Any] = yaml.safe_load(candidates[0][0])
    assert full["statement"]["subject"]["explode_by"] == ";"
    assert full["statement"]["subject"]["remove"]  # the noise knob fired too
    explode_only = [c for c in candidates if "explode-only" in c[1]]
    assert explode_only
    variant: dict[str, Any] = yaml.safe_load(explode_only[0][0])["statement"]["subject"]
    assert variant["explode_by"] == ";"
    assert "remove" not in variant


# --------------------------------------------------------------------------- #
# Demoted-predicate fix — driven by the build_and_audit report's predicate_advice
# --------------------------------------------------------------------------- #


def _demotion_audit(predicate: str = "gene_associated_with_condition") -> dict[str, Any]:
    return {
        "demoted_edge_pct": 1.0,
        "predicate_advice": [
            {
                "predicate": predicate,
                "subject_category": "Gene",
                "object_category": "Disease",
                "association": "GeneToDiseaseAssociation",
                "legal_predicates": ["affects", "associated_with", "contributes_to"],
                "edges": 12,
            }
        ],
    }


def _demoted_section() -> dict[str, Any]:
    cfg: dict[str, Any] = _joined_section()
    cfg["statement"]["predicate"] = "gene_associated_with_condition"  # forbidden on GeneToDiseaseAssociation
    return cfg


def test_propose_predicate_fix_with_audit() -> None:
    """A demoted predicate is replaced with a legal one from predicate_advice (schema-valid)."""
    edited, rationale = propose_config_edit(_demoted_section(), _joined_report([]), audit=_demotion_audit())

    assert validate_section(edited) is True
    statement = yaml.safe_load(edited)["statement"]
    assert statement["predicate"] in {"affects", "associated_with", "contributes_to"}
    assert "demoted" in rationale


def test_propose_predicate_fix_requires_audit() -> None:
    """Without the audit report the predicate is NEVER touched (coverage-only behavior preserved)."""
    edited, _ = propose_config_edit(_demoted_section(), _joined_report([]))
    assert yaml.safe_load(edited)["statement"]["predicate"] == "gene_associated_with_condition"


def test_propose_predicate_fix_idempotent() -> None:
    """Once the predicate is legal the advice no longer matches it: a second pass changes nothing."""
    audit = _demotion_audit()
    edited, _ = propose_config_edit(_demoted_section(), _joined_report([]), audit=audit)
    edited2, rationale2 = propose_config_edit(edited, _joined_report([]), audit=audit)
    assert yaml.safe_load(edited2)["statement"]["predicate"] == yaml.safe_load(edited)["statement"]["predicate"]
    assert "demoted" not in rationale2


def test_propose_predicate_fix_ambiguous_skips() -> None:
    """Two advice entries naming the same predicate with DISAGREEING legal sets leave it unchanged."""

    def _entry(subject: str, obj: str, legal: list[str]) -> dict[str, Any]:
        return {
            "predicate": "gene_associated_with_condition",
            "subject_category": subject,
            "object_category": obj,
            "association": "VariantToGeneAssociation",
            "legal_predicates": legal,
            "edges": 3,
        }

    # The section has no prioritize hints, so the two entries cannot be disambiguated.
    cfg: dict[str, Any] = _demoted_section()
    del cfg["statement"]["subject"]["prioritize"]
    del cfg["statement"]["object"]["prioritize"]
    audit: dict[str, Any] = {
        "predicate_advice": [
            _entry("SequenceVariant", "Gene", ["condition_associated_with_gene", "gene_associated_with_condition", "genetically_associated_with"]),
            _entry("Gene", "Disease", ["affects", "associated_with", "contributes_to"]),
        ]
    }
    edited, _ = propose_config_edit(cfg, _joined_report([]), audit=audit)
    assert yaml.safe_load(edited)["statement"]["predicate"] == "gene_associated_with_condition"


def test_propose_predicate_fix_disambiguated_by_prioritize() -> None:
    """Same-predicate advice entries are disambiguated by the section's own prioritize categories."""

    def _entry(subject: str, obj: str, legal: list[str]) -> dict[str, Any]:
        return {
            "predicate": "gene_associated_with_condition",
            "subject_category": subject,
            "object_category": obj,
            "association": "X",
            "legal_predicates": legal,
            "edges": 3,
        }

    audit: dict[str, Any] = {
        "predicate_advice": [
            _entry("SequenceVariant", "Gene", ["condition_associated_with_gene", "genetically_associated_with"]),
            _entry("Gene", "Disease", ["affects", "associated_with", "contributes_to"]),
        ]
    }
    edited, _ = propose_config_edit(_demoted_section(), _joined_report([]), audit=audit)
    # prioritize [Gene] ~ [Disease] selects the second entry's legal set.
    assert yaml.safe_load(edited)["statement"]["predicate"] in {"affects", "associated_with", "contributes_to"}
