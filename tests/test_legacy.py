from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from yaml.constructor import ConstructorError

from tablassert.errors import LegacyDuplicateKeyWarning, LegacySourceUnresolvedError, LegacyUnsupportedSyntaxError, UnpairedEffectAnnotationWarning
from tablassert.ingests import from_yaml, to_sections, to_yaml
from tablassert.legacy import convert_legacy, load_legacy_yaml
from tablassert.models import Section


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "legacy.yaml"
    p.write_text(text)
    return p


def test_duplicate_free_config_matches_safe_load(fixtures_path: Path, recwarn: pytest.WarningsRecorder) -> None:
    """A well-formed legacy config parses identically to ``yaml.safe_load``.

    WHY: the duplicate-merge hook must be a strict no-op on files without duplicates —
    the real TableConfigs-shaped fixture is parsed to an equal object and emits no
    ``LegacyDuplicateKeyWarning``.
    """
    p = fixtures_path / "minimal_section.yaml"
    result = load_legacy_yaml(p)
    assert result == yaml.safe_load(p.read_text())
    assert [w for w in recwarn if issubclass(w.category, LegacyDuplicateKeyWarning)] == []


def test_duplicate_scalar_keys_later_wins_and_warns(tmp_path: Path) -> None:
    """Duplicate scalar keys keep the LAST value but fire exactly one warning naming key and line.

    WHY: ``yaml.safe_load`` silently drops the earlier occurrence; the legacy loader keeps the
    later-wins semantics while surfacing the data loss so a curator can fix the source file.
    """
    p = _write(tmp_path, "name: first\nversion: 1\nname: second\n")
    with pytest.warns(LegacyDuplicateKeyWarning, match=r"Duplicate key `name` .+:3") as record:
        result = load_legacy_yaml(p)
    assert result == {"name": "second", "version": 1}
    assert len(record) == 1


def test_duplicate_nested_dicts_deep_merge(tmp_path: Path) -> None:
    """Duplicate keys whose values are dicts deep-merge instead of overwriting.

    WHY: legacy configs routinely split one logical section across a repeated key; a plain
    ``safe_load`` would keep only the last block and silently lose the first block's fields.
    """
    p = _write(tmp_path, "template:\n  subject:\n    method: value\ntemplate:\n  subject:\n    encoding: BRCA1\n  config: ./a.tsv\n")
    with pytest.warns(LegacyDuplicateKeyWarning, match=r"Duplicate key `template` .+:4"):
        result = load_legacy_yaml(p)
    assert result == {"template": {"subject": {"method": "value", "encoding": "BRCA1"}, "config": "./a.tsv"}}


def test_duplicate_lists_extend(tmp_path: Path) -> None:
    """Duplicate keys whose values are lists concatenate first + later items.

    WHY: a repeated list key in a legacy config almost always means "more entries for the same
    field"; ``safe_load`` would keep only the last list.
    """
    p = _write(tmp_path, "encodings:\n  - BRCA1\nencodings:\n  - TP53\n  - EGFR\n")
    with pytest.warns(LegacyDuplicateKeyWarning, match=r"Duplicate key `encodings` .+:3"):
        result = load_legacy_yaml(p)
    assert result == {"encodings": ["BRCA1", "TP53", "EGFR"]}


def test_duplicate_mixed_types_later_wins(tmp_path: Path) -> None:
    """A duplicate key whose value type changed keeps the later value outright.

    WHY: there is no meaningful merge across types, so the loader matches ``fastmerge``'s
    collision fallback exactly — the two merge paths can never drift.
    """
    p = _write(tmp_path, "source:\n  url:\n    - https://a.example/a.tsv\nsource: ./flat.tsv\n")
    with pytest.warns(LegacyDuplicateKeyWarning, match=r"Duplicate key `source` .+:4"):
        result = load_legacy_yaml(p)
    assert result == {"source": "./flat.tsv"}


def test_duplicate_merge_does_not_mutate_aliased_value(tmp_path: Path) -> None:
    """Merging a duplicate key must not rewrite the mapping shared through a YAML alias.

    WHY: the earlier occurrence is deep-copied before the in-place merge; without that copy,
    merging into ``section`` would also rewrite ``defaults``, since the alias shares one dict.
    """
    p = _write(tmp_path, "defaults: &d\n  a: 1\nsection: *d\nsection:\n  b: 2\n")
    with pytest.warns(LegacyDuplicateKeyWarning, match=r"Duplicate key `section` .+:4"):
        result = load_legacy_yaml(p)
    assert result == {"defaults": {"a": 1}, "section": {"a": 1, "b": 2}}


def test_duplicate_loader_matches_safe_load_on_recursive_alias(tmp_path: Path) -> None:
    """A self-referential alias still yields the same self-referential dict ``safe_load`` builds.

    WHY: eager (``deep=True``) construction was a behavior risk for recursive aliases; this
    pins parity — the cached placeholder is returned for the alias, exactly as in ``safe_load``.
    """
    p = _write(tmp_path, "&a {x: *a}\n")
    result = load_legacy_yaml(p)
    expected = yaml.safe_load("&a {x: *a}\n")
    assert isinstance(result, dict)
    assert isinstance(expected, dict)
    assert result["x"] is result
    assert expected["x"] is expected


def test_duplicate_loader_rejects_unhashable_key_like_safe_load(tmp_path: Path) -> None:
    """An explicit complex key is rejected exactly as ``yaml.safe_load`` rejects it.

    WHY: the merge path must not loosen SafeLoader's error behavior for constructs that
    cannot be dict keys; parity with ``safe_load`` keeps the safety stance honest.
    """
    p = _write(tmp_path, "? [1, 2]\n: value\n")
    with pytest.raises(ConstructorError, match="found unhashable key"):
        load_legacy_yaml(p)


def test_duplicate_loader_rejects_mistagged_non_mapping_like_safe_load(tmp_path: Path) -> None:
    """A ``!!map`` tag forced onto a sequence is rejected exactly as ``yaml.safe_load`` rejects it.

    WHY: explicit tags can lie about node shape; refusing to walk a non-mapping as pairs is
    what the base constructor does, and the duplicate hook must not weaken that.
    """
    p = _write(tmp_path, "!!map [1, 2]\n")
    with pytest.raises(ConstructorError, match="expected a mapping node"):
        load_legacy_yaml(p)


# --- US-003: legacy -> v12 conversion -------------------------------------- #


def _downloads(tmp_path: Path, *paths: str) -> Path:
    """Build a downloads tree of placeholder payload files; return the downloads dir."""
    downloads: Path = tmp_path / "downloads"
    for rel in paths:
        payload: Path = downloads / rel
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload.write_bytes(b"payload")
    return downloads


def _legacy_config(annotations: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """A minimal legacy template-only config; callers swap the annotations list."""
    return {
        "template": {
            "source": {"kind": "text", "local": "./DATALAKE/data.tsv", "url": ["https://example.com/data.tsv"]},
            "statement": {"subject": {"method": "value", "encoding": "BRCA1"}, "object": {"method": "value", "encoding": "TP53"}},
            "provenance": {"repo": "PMC", "publication": "PMC0000000"},
            "annotations": annotations or [],
        }
    }


def _legacy_file(tmp_path: Path, config: dict[str, Any]) -> Path:
    p: Path = tmp_path / "legacy.yaml"
    p.write_text(yaml.safe_dump(config, sort_keys=False))
    return p


def test_convert_template_only_yields_one_section(fixtures_path: Path, tmp_path: Path) -> None:
    """A template-only legacy file converts to exactly one fully-specified section.

    WHY: the shared provenance is hoisted into the template (the v12 idiom), the stale
    ``./DATALAKE`` local is rewritten to the resolved payload, the url becomes the real
    S3 object, the reindex block passes through byte-identical, and the legacy
    ``relationship_strength`` annotation is renamed to ``effect_size`` with its
    method/encoding preserved.
    """
    downloads: Path = _downloads(tmp_path, "PMC7878905/PMC7878905.1/CHENG1.xlsx")
    out: dict[str, Any] = convert_legacy(fixtures_path / "legacy_template_only.yaml", downloads)
    assert set(out) == {"template", "sections"}
    assert out["template"] == {"provenance": {"repo": "PMC", "publication": "PMC7878905"}}
    sections: list[dict[str, Any]] = out["sections"]
    assert len(sections) == 1
    section: dict[str, Any] = sections[0]
    assert "provenance" not in section  # hoisted onto the template
    source: dict[str, Any] = section["source"]
    assert source["local"] == str((downloads / "PMC7878905" / "PMC7878905.1" / "CHENG1.xlsx").resolve())
    assert source["url"] == ["https://pmc-oa-opendata.s3.amazonaws.com/PMC7878905.1/CHENG1.xlsx"]
    assert source["reindex"] == [{"column": "A", "comparison": "ne", "comparator": "N/A"}]  # unchanged passthrough
    by_name: dict[str, dict[str, Any]] = {entry["annotation"]: entry for entry in section["annotations"]}
    assert by_name["effect_size"] == {"annotation": "effect_size", "method": "column", "encoding": "D"}
    assert "relationship_strength" not in by_name


def test_convert_multi_section_expansion(fixtures_path: Path, tmp_path: Path) -> None:
    """Template+sections expand via the SAME fastmerge path production uses.

    WHY: per-section overrides (sheet, object, annotations) merge over the template,
    annotation LISTS extend (template entries first), and the space-separated
    ``relationship strength`` alias in a section renames to ``effect_size``.
    """
    downloads: Path = _downloads(tmp_path, "PMC11530135/PMC11530135.1/CORREIA1.xlsx")
    out: dict[str, Any] = convert_legacy(fixtures_path / "legacy_multi_section.yaml", downloads)
    assert out["template"] == {"provenance": {"repo": "PMC", "publication": "PMC11530135"}}
    sections: list[dict[str, Any]] = out["sections"]
    assert len(sections) == 2
    first: dict[str, Any] = sections[0]
    second: dict[str, Any] = sections[1]
    assert first["source"]["sheet"] == "Data File 4"
    assert second["source"]["sheet"] == "Data File 5"
    assert first["source"]["local"] == second["source"]["local"] == str((downloads / "PMC11530135" / "PMC11530135.1" / "CORREIA1.xlsx").resolve())
    assert [entry["annotation"] for entry in first["annotations"]] == ["sample size", "p value", "effect_size"]
    assert [entry["annotation"] for entry in second["annotations"]] == ["sample size", "p value", "effect size", "effect type"]


def test_convert_divergent_provenances_stay_on_sections(tmp_path: Path) -> None:
    """Sections with DIVERGENT provenance never hoist: template stays empty, each keeps its own.

    WHY: hoisting is only the v12 idiom when EVERY section shares one provenance; when they
    diverge, the converter must return an empty template and leave every provenance on its
    own section — and each section still resolves against its OWN publication's payload dir.
    """
    config: dict[str, Any] = _legacy_config()
    config["template"].pop("provenance")
    config["sections"] = [{"provenance": {"repo": "PMC", "publication": "PMC1111111"}}, {"provenance": {"repo": "PMC", "publication": "PMC2222222"}}]
    p: Path = _legacy_file(tmp_path, config)
    downloads: Path = _downloads(tmp_path, "PMC1111111/PMC1111111.1/data.tsv", "PMC2222222/PMC2222222.1/data.tsv")
    out: dict[str, Any] = convert_legacy(p, downloads)
    assert out["template"] == {}
    sections: list[dict[str, Any]] = out["sections"]
    assert len(sections) == 2
    assert sections[0]["provenance"] == {"repo": "PMC", "publication": "PMC1111111"}
    assert sections[1]["provenance"] == {"repo": "PMC", "publication": "PMC2222222"}
    assert sections[0]["source"]["local"] == str((downloads / "PMC1111111" / "PMC1111111.1" / "data.tsv").resolve())
    assert sections[1]["source"]["local"] == str((downloads / "PMC2222222" / "PMC2222222.1" / "data.tsv").resolve())


def test_convert_output_round_trips_through_section_models(fixtures_path: Path, tmp_path: Path) -> None:
    """The converted config dumps via ``to_yaml`` and every section constructs a ``Section``.

    WHY: the acceptance shape is a real v12 config — dumped, re-read, expanded through
    ``to_sections`` (the production merge), and validated by the Section models with the
    paired ``effect_size``/``effect_type`` annotations kept.
    """
    downloads: Path = _downloads(tmp_path, "PMC7878905/PMC7878905.1/CHENG1.xlsx")
    out: dict[str, Any] = convert_legacy(fixtures_path / "legacy_template_only.yaml", downloads)
    config: Path = tmp_path / "converted.yaml"
    to_yaml(config, out)
    data: Any = from_yaml(config)
    # cast: ``to_sections`` returns one dict per section, but its annotation nests one list too deep.
    expanded: list[dict[str, Any]] = cast("list[dict[str, Any]]", to_sections(data, config))
    assert len(expanded) == 1
    merged: dict[str, Any] = expanded[0]
    merged.pop("config")  # stamped by to_sections; not a Section field
    section: Section = Section.model_validate(merged)
    assert {annotation.annotation for annotation in section.annotations or []} == {"effect_size", "effect_type", "sample_size"}


def test_convert_pairing_gap_passes_through_to_section_validation(tmp_path: Path) -> None:
    """The converter renames the alias but never pair-checks; Section drops the orphan (US-001).

    WHY: pairing policy lives ONLY on ``Section`` — the converted output still carries the
    renamed ``effect_size`` annotation, and constructing the section drops it with the
    ``UnpairedEffectAnnotationWarning`` instead of the converter second-guessing it.
    """
    p: Path = _legacy_file(tmp_path, _legacy_config([{"annotation": "relationship strength", "method": "column", "encoding": "D"}]))
    downloads: Path = _downloads(tmp_path, "PMC0000000/PMC0000000.1/data.tsv")
    out: dict[str, Any] = convert_legacy(p, downloads)
    section: dict[str, Any] = out["sections"][0]
    assert section["annotations"] == [{"annotation": "effect_size", "method": "column", "encoding": "D"}]  # kept, renamed
    merged: dict[str, Any] = {**section, "provenance": out["template"]["provenance"]}
    with pytest.warns(UnpairedEffectAnnotationWarning, match="effect_size"):
        validated: Section = Section.model_validate(merged)
    assert validated.annotations is None  # Section dropped the unpaired half


def _reindex_file(tmp_path: Path, entry: object) -> Path:
    config: dict[str, Any] = _legacy_config()
    config["template"]["source"]["reindex"] = [entry]
    return _legacy_file(tmp_path, config)


@pytest.mark.parametrize(
    "entry",
    [
        {"column": "A", "comparison": "ne", "comparator": 5},  # ne requires a str comparator
        {"column": "TOOLONG", "comparison": "ne", "comparator": "x"},  # column must be A-ZZ
        {"column": "A", "comparison": "gt", "comparator": "N/A"},  # gt requires a numeric comparator
        {"comparison": "ne", "comparator": "x"},  # missing column
    ],
)
def test_convert_invalid_reindex_raises_unsupported(tmp_path: Path, entry: dict[str, Any]) -> None:
    """A reindex entry the v12 ``Reindex`` model rejects fails the WHOLE conversion.

    WHY: ``models.Reindex`` already supports {column, comparison, comparator}; anything it
    rejects must surface as ``legacy-unsupported-syntax`` naming the entry, never leak a
    half-translated filter into the output.
    """
    p: Path = _reindex_file(tmp_path, entry)
    downloads: Path = _downloads(tmp_path, "PMC0000000/PMC0000000.1/data.tsv")
    with pytest.raises(LegacyUnsupportedSyntaxError, match="reindex") as excinfo:
        convert_legacy(p, downloads)
    assert excinfo.value.code == "legacy-unsupported-syntax"
    assert str(excinfo.value).endswith("legacy-unsupported-syntax")


def test_convert_malformed_reindex_shape_raises_unsupported(tmp_path: Path) -> None:
    """Non-list ``reindex`` and non-mapping entries are unsupported constructs too."""
    downloads: Path = _downloads(tmp_path, "PMC0000000/PMC0000000.1/data.tsv")
    not_a_list: dict[str, Any] = _legacy_config()
    not_a_list["template"]["source"]["reindex"] = {"column": "A"}
    with pytest.raises(LegacyUnsupportedSyntaxError, match="reindex"):
        convert_legacy(_legacy_file(tmp_path, not_a_list), downloads)
    with pytest.raises(LegacyUnsupportedSyntaxError, match="reindex"):
        convert_legacy(_reindex_file(tmp_path, "column A"), downloads)


def test_convert_unresolved_source_raises(tmp_path: Path) -> None:
    """No basename match and no fetch -> loud ``legacy-source-unresolved``, never a stale path.

    WHY: the message names the legacy file, the stale ``./DATALAKE`` path, and every
    location tried; the conversion returns NOTHING instead of emitting a config whose
    local points at a file that does not exist.
    """
    p: Path = _legacy_file(tmp_path, _legacy_config())
    downloads: Path = _downloads(tmp_path, "PMC0000000/PMC0000000.1/other.tsv")
    with pytest.raises(LegacySourceUnresolvedError) as excinfo:
        convert_legacy(p, downloads)
    assert excinfo.value.code == "legacy-source-unresolved"
    message: str = str(excinfo.value)
    assert str(p) in message
    assert "./DATALAKE/data.tsv" in message
    assert str(downloads) in message


def test_convert_without_downloads_dir_raises_unresolved(tmp_path: Path) -> None:
    """``downloads=None`` cannot resolve any payload (fetch included — it needs a target)."""
    p: Path = _legacy_file(tmp_path, _legacy_config())
    with pytest.raises(LegacySourceUnresolvedError) as excinfo:
        convert_legacy(p, None)
    assert excinfo.value.code == "legacy-source-unresolved"
    assert "no downloads directory" in str(excinfo.value)
    with pytest.raises(LegacySourceUnresolvedError, match="downloads directory"):
        convert_legacy(p, None, fetch=True)


# Offline PMC S3 seams for the fetch tests (mirrors test_agent_fetch.py): the two
# ``_http_*`` functions are the single I/O seam, routed by URL.
_PMC_VERSION_LISTING: str = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
    "<CommonPrefixes><Prefix>PMC11708054.1/</Prefix></CommonPrefixes>"
    "</ListBucketResult>"
)
_PMC_EMPTY_LISTING: str = '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"></ListBucketResult>'
_PMC_OBJECT_KEYS: list[str] = ["PMC11708054.1/PMC11708054.1.json", "PMC11708054.1/PMC11708054.1.xml", "PMC11708054.1/mbio.01679-24-s0002.xlsx"]


def _patch_pmc(monkeypatch: pytest.MonkeyPatch, *, version_listing: str = _PMC_VERSION_LISTING) -> None:
    """Route the agent's HTTP seams offline (same pattern as test_agent_fetch.py)."""

    def get_text(url: str, *, timeout: int = 120) -> str:
        if "list-type=2" in url and "delimiter=" in url:
            return version_listing
        if "list-type=2" in url:
            contents: str = "".join(f"<Contents><Key>{key}</Key><Size>1</Size></Contents>" for key in _PMC_OBJECT_KEYS)
            return f'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">{contents}</ListBucketResult>'
        if url.endswith(".json"):
            return '{"is_pmc_openaccess": true, "license_code": "CC-BY"}'
        raise AssertionError(f"unexpected text url: {url}")

    def get_bytes(url: str, *, timeout: int = 120) -> bytes:
        return b"FAKEBYTES"

    monkeypatch.setattr("tablassert.agent._http_get_text", get_text)
    monkeypatch.setattr("tablassert.agent._http_get_bytes", get_bytes)


def _pmc_config(local_name: str) -> dict[str, Any]:
    config: dict[str, Any] = _legacy_config()
    config["template"]["source"]["local"] = f"./DATALAKE/{local_name}"
    config["template"]["provenance"]["publication"] = "PMC11708054"
    return config


def test_convert_fetch_downloads_payload_when_no_local_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """fetch=True downloads the article via the agent helpers, then the basename match wins.

    WHY: the payload lands in the canonical ``downloads/<PMC<n>>/<PMC<n>.<v>>/`` layout, so
    the retry resolves the local AND the url becomes the real S3 object for the stem.
    """
    _patch_pmc(monkeypatch)
    p: Path = _legacy_file(tmp_path, _pmc_config("mbio.01679-24-s0002.xlsx"))
    downloads: Path = tmp_path / "downloads"
    downloads.mkdir()
    out: dict[str, Any] = convert_legacy(p, downloads, fetch=True)
    source: dict[str, Any] = out["sections"][0]["source"]
    payload: Path = downloads / "PMC11708054" / "PMC11708054.1" / "mbio.01679-24-s0002.xlsx"
    assert source["local"] == str(payload.resolve())
    assert source["url"] == ["https://pmc-oa-opendata.s3.amazonaws.com/PMC11708054.1/mbio.01679-24-s0002.xlsx"]
    assert payload.read_bytes() == b"FAKEBYTES"


def test_convert_fetch_failure_surfaces_unresolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed fetch leaves the source unresolved: coded error with the cause chained."""
    _patch_pmc(monkeypatch, version_listing=_PMC_EMPTY_LISTING)
    p: Path = _legacy_file(tmp_path, _pmc_config("mbio.01679-24-s0002.xlsx"))
    downloads: Path = tmp_path / "downloads"
    downloads.mkdir()
    with pytest.raises(LegacySourceUnresolvedError, match="PMC open-access fetch") as excinfo:
        convert_legacy(p, downloads, fetch=True)
    assert excinfo.value.code == "legacy-source-unresolved"
    assert isinstance(excinfo.value.__cause__, FileNotFoundError)


def test_convert_keeps_legacy_url_when_payload_stem_unknown(tmp_path: Path) -> None:
    """A payload found outside any ``PMC<n>.<v>`` directory keeps the legacy url verbatim.

    WHY: only a resolved payload under an article-version stem reveals its real S3 object;
    anything else keeps the curator's url rather than fabricating one.
    """
    downloads: Path = _downloads(tmp_path, "data.tsv")  # flat: no PMC<n>.<v> parent
    p: Path = _legacy_file(tmp_path, _legacy_config())
    out: dict[str, Any] = convert_legacy(p, downloads)
    source: dict[str, Any] = out["sections"][0]["source"]
    assert source["local"] == str((downloads / "data.tsv").resolve())
    assert source["url"] == ["https://example.com/data.tsv"]  # legacy url kept verbatim


def test_convert_prefers_publication_dir_on_basename_collision(tmp_path: Path) -> None:
    """One downloads parent can hold many articles; the section's own article wins ties.

    WHY: generic payload names (``data.tsv``) collide across articles, so a match under the
    directory named after ``provenance.publication`` beats an elsewhere match.
    """
    downloads: Path = _downloads(tmp_path, "PMC1111111/PMC1111111.1/data.tsv", "PMC2222222/PMC2222222.1/data.tsv")
    config: dict[str, Any] = _legacy_config()
    config["template"]["provenance"]["publication"] = "PMC2222222"
    p: Path = _legacy_file(tmp_path, config)
    out: dict[str, Any] = convert_legacy(p, downloads)
    source: dict[str, Any] = out["sections"][0]["source"]
    assert source["local"] == str((downloads / "PMC2222222" / "PMC2222222.1" / "data.tsv").resolve())
    assert source["url"] == ["https://pmc-oa-opendata.s3.amazonaws.com/PMC2222222.1/data.tsv"]


@pytest.mark.parametrize(
    "name", ["relationship strength", "relationship_strength", "Relationship-Strength", "RELATIONSHIP STRENGTH", "relationshipstrength"]
)
def test_convert_alias_spellings_rename_to_effect_size(tmp_path: Path, name: str) -> None:
    """Every case/separator spelling of the legacy alias renames to ``effect_size``."""
    p: Path = _legacy_file(tmp_path, _legacy_config([{"annotation": name, "method": "column", "encoding": "D"}]))
    downloads: Path = _downloads(tmp_path, "PMC0000000/PMC0000000.1/data.tsv")
    out: dict[str, Any] = convert_legacy(p, downloads)
    assert out["sections"][0]["annotations"] == [{"annotation": "effect_size", "method": "column", "encoding": "D"}]


def test_convert_non_alias_annotation_names_untouched(tmp_path: Path) -> None:
    """Names that merely CONTAIN the words are not aliases and pass through verbatim."""
    annotations: list[dict[str, Any]] = [
        {"annotation": "relationship_strength_notes", "method": "value", "encoding": "x"},
        {"annotation": "strength relationship", "method": "value", "encoding": "y"},
    ]
    p: Path = _legacy_file(tmp_path, _legacy_config(annotations))
    downloads: Path = _downloads(tmp_path, "PMC0000000/PMC0000000.1/data.tsv")
    out: dict[str, Any] = convert_legacy(p, downloads)
    assert out["sections"][0]["annotations"] == annotations


def test_convert_unsupported_constructs_raise(tmp_path: Path) -> None:
    """Every construct v12 cannot express fails loudly with ``legacy-unsupported-syntax``.

    WHY: conversion is all-or-nothing — no partial silent writes. Covers the root shape,
    unknown top-level/template/section keys, and malformed ``sections`` containers.
    """
    cases: list[tuple[str, str]] = [
        ("[1, 2]\n", "must hold a mapping"),
        ("statement:\n  subject: {method: value}\n", "unknown top-level key"),
        ("template: [1]\n", "`template` must hold a mapping"),
        ("template: {}\nsections: {}\n", "`sections` must hold a list"),
        ("template: {}\nsections: []\n", "empty list"),
        ("template: {}\nsections: [1]\n", "sections[0] must hold a mapping"),
        ("template:\n  frobnicate: 1\n", "frobnicate"),
        ("template: {}\nsections:\n  - frobnicate: 1\n", "frobnicate"),
    ]
    for text, fragment in cases:
        p: Path = tmp_path / "case.yaml"
        p.write_text(text)
        with pytest.raises(LegacyUnsupportedSyntaxError, match=re.escape(fragment)) as excinfo:
            convert_legacy(p, None)
        assert excinfo.value.code == "legacy-unsupported-syntax"


def test_convert_non_mapping_annotation_entries_raise(tmp_path: Path) -> None:
    """A scalar in ``annotations`` (or a non-mapping ``source``) is an unsupported construct."""
    config: dict[str, Any] = _legacy_config()
    config["template"]["annotations"] = ["relationship strength"]
    p: Path = _legacy_file(tmp_path, config)
    downloads: Path = _downloads(tmp_path, "PMC0000000/PMC0000000.1/data.tsv")
    with pytest.raises(LegacyUnsupportedSyntaxError, match="annotations"):
        convert_legacy(p, downloads)
    scalar_source: dict[str, Any] = _legacy_config()
    scalar_source["template"]["source"] = "./DATALAKE/data.tsv"
    with pytest.raises(LegacyUnsupportedSyntaxError, match="`source` must hold a mapping"):
        convert_legacy(_legacy_file(tmp_path, scalar_source), downloads)


def test_convert_module_never_imports_agent_eagerly() -> None:
    """EVERY ``tablassert.agent`` import in legacy.py sits inside a function body.

    WHY: legacy conversion must stay importable in the base environment; the agent module
    is reached only on the fetch/url paths. Walking the WHOLE tree (not just module-level
    statements) makes the guard exhaustive — no agent import may live at module or class
    scope, or in any branch reachable without a call.
    """
    import tablassert.legacy as legacy_mod

    assert legacy_mod.__file__ is not None
    tree: ast.Module = ast.parse(Path(legacy_mod.__file__).read_text())

    def is_agent_import(node: ast.AST) -> bool:
        if isinstance(node, ast.Import):
            return any(alias.name.startswith("tablassert.agent") for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            return (node.module or "").startswith("tablassert.agent")
        return False

    parent_of: dict[ast.AST, ast.AST] = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    agent_imports: list[ast.AST] = [node for node in ast.walk(tree) if is_agent_import(node)]
    assert agent_imports  # the lazy seams exist; the guard below must cover ALL of them
    for node in agent_imports:
        ancestor: ast.AST | None = node
        while ancestor is not None and not isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
            ancestor = parent_of.get(ancestor)
        assert ancestor is not None, f"tablassert.agent import outside a function at line {node.lineno}"
