from __future__ import annotations

import ast
import os
import re
import warnings
from pathlib import Path, PurePosixPath
from typing import Any, cast

import pydantic
import pytest
import yaml
from yaml.constructor import ConstructorError

from tablassert import cli
from tablassert.cli import convert_legacy_command
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


def test_convert_fetch_retries_url_basenames_after_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """After a fetch, the retry matches EVERY candidate — the fetched objects carry the url
    basenames, not the human ``local`` alias, so a local-only retry would never resolve.
    """
    _patch_pmc(monkeypatch)
    config: dict[str, Any] = _legacy_config()
    config["template"]["source"]["local"] = "./DATALAKE/human-alias.xlsx"
    config["template"]["source"]["url"] = ["https://pmc.ncbi.nlm.nih.gov/articles/instance/11708054/bin/mbio.01679-24-s0002.xlsx"]
    config["template"]["provenance"]["publication"] = "PMC11708054"
    p: Path = _legacy_file(tmp_path, config)
    downloads: Path = tmp_path / "downloads"
    downloads.mkdir()
    out: dict[str, Any] = convert_legacy(p, downloads, fetch=True)
    payload: Path = downloads / "PMC11708054" / "PMC11708054.1" / "mbio.01679-24-s0002.xlsx"
    assert out["sections"][0]["source"]["local"] == str(payload.resolve())
    assert payload.read_bytes() == b"FAKEBYTES"


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


def test_convert_url_basename_hit_when_local_basename_misses(tmp_path: Path) -> None:
    """The legacy ``local`` is a human alias; each ``source.url`` basename is a fallback.

    WHY: MOKG configs point ``local`` at ``./DATALAKE/<NAME>.xlsx`` files that exist only on
    the curator's machine, while the ``url`` entries hold the REAL payload filenames — so
    resolution falls back to the url basenames (query/fragment stripped) and the resolved
    payload still gets its url repaired to the real S3 object.
    """
    config: dict[str, Any] = _legacy_config()
    config["template"]["source"]["local"] = "./DATALAKE/HUMAN-ALIAS.xlsx"
    config["template"]["source"]["url"] = ["https://pmc.ncbi.nlm.nih.gov/articles/instance/7878905/bin/real-payload.xlsx?v=2"]
    p: Path = _legacy_file(tmp_path, config)
    downloads: Path = _downloads(tmp_path, "PMC7878905/PMC7878905.1/real-payload.xlsx")
    out: dict[str, Any] = convert_legacy(p, downloads)
    source: dict[str, Any] = out["sections"][0]["source"]
    assert source["local"] == str((downloads / "PMC7878905" / "PMC7878905.1" / "real-payload.xlsx").resolve())
    assert source["url"] == ["https://pmc-oa-opendata.s3.amazonaws.com/PMC7878905.1/real-payload.xlsx"]


def test_convert_local_basename_precedes_url_basenames(tmp_path: Path) -> None:
    """Resolution order is local basename FIRST, then url basenames in declared order.

    WHY: when both exist on disk the curator's local name wins, and when the first url
    basename misses the second one still resolves — the first hit in candidate order wins.
    """
    config: dict[str, Any] = _legacy_config()
    config["template"]["source"]["local"] = "./DATALAKE/human.xlsx"
    config["template"]["source"]["url"] = ["https://a.example/bin/missing.xlsx", "https://b.example/files/second.xlsx"]
    p: Path = _legacy_file(tmp_path, config)
    both: Path = _downloads(tmp_path, "PMC0000000/PMC0000000.1/human.xlsx", "PMC0000000/PMC0000000.1/second.xlsx")
    out: dict[str, Any] = convert_legacy(p, both)
    assert out["sections"][0]["source"]["local"] == str((both / "PMC0000000" / "PMC0000000.1" / "human.xlsx").resolve())
    no_local: Path = _downloads(tmp_path / "alt", "PMC0000000/PMC0000000.1/second.xlsx")
    second: Path = tmp_path / "legacy-second.yaml"
    second.write_text(yaml.safe_dump(config, sort_keys=False))
    out = convert_legacy(second, no_local)
    assert out["sections"][0]["source"]["local"] == str((no_local / "PMC0000000" / "PMC0000000.1" / "second.xlsx").resolve())


def test_convert_unresolved_error_lists_every_tried_basename(tmp_path: Path) -> None:
    """The unresolved error names the local basename AND every url basename that was tried."""
    config: dict[str, Any] = _legacy_config()
    config["template"]["source"]["local"] = "./DATALAKE/human.xlsx"
    config["template"]["source"]["url"] = ["https://a.example/bin/first-payload.xlsx", "https://b.example/files/second-payload.xlsx"]
    p: Path = _legacy_file(tmp_path, config)
    downloads: Path = _downloads(tmp_path, "PMC0000000/PMC0000000.1/other.tsv")
    with pytest.raises(LegacySourceUnresolvedError) as excinfo:
        convert_legacy(p, downloads)
    assert excinfo.value.code == "legacy-source-unresolved"
    message: str = str(excinfo.value)
    for basename in ("human.xlsx", "first-payload.xlsx", "second-payload.xlsx"):
        assert f"basename '{basename}' under {downloads}" in message


def test_convert_malformed_url_shape_raises_unsupported(tmp_path: Path) -> None:
    """A scalar ``source.url`` or a non-string entry is an unsupported construct."""
    downloads: Path = _downloads(tmp_path, "PMC0000000/PMC0000000.1/data.tsv")
    scalar: dict[str, Any] = _legacy_config()
    scalar["template"]["source"]["url"] = "https://example.com/data.tsv"
    with pytest.raises(LegacyUnsupportedSyntaxError, match=r"`source\.url` must hold a list"):
        convert_legacy(_legacy_file(tmp_path, scalar), downloads)
    bad_entry: dict[str, Any] = _legacy_config()
    bad_entry["template"]["source"]["url"] = [42]
    second: Path = tmp_path / "legacy-bad-entry.yaml"
    second.write_text(yaml.safe_dump(bad_entry, sort_keys=False))
    with pytest.raises(LegacyUnsupportedSyntaxError, match=r"every `source\.url` entry must hold a URL string"):
        convert_legacy(second, downloads)


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


def test_convert_dedup_drops_identical_overlay_entries(tmp_path: Path) -> None:
    """An entry declared IDENTICALLY in template and section lands exactly once.

    WHY: the overlay EXTENDS list-valued keys (fastmerge), so the MIN1 case — the same
    ``anatomical_context_qualifier`` on the template and a section — lands twice in the
    merged section; the converter drops the second copy, preserving first-seen order.
    """
    qualifier: dict[str, Any] = {"qualifier": "anatomical_context_qualifier", "method": "value", "encoding": "UBERON:0000955"}
    config: dict[str, Any] = _legacy_config()
    config["template"]["statement"]["qualifiers"] = [dict(qualifier)]
    config["sections"] = [{"statement": {"qualifiers": [dict(qualifier)]}}]
    p: Path = _legacy_file(tmp_path, config)
    downloads: Path = _downloads(tmp_path, "PMC0000000/PMC0000000.1/data.tsv")
    out: dict[str, Any] = convert_legacy(p, downloads)
    assert out["sections"][0]["statement"]["qualifiers"] == [qualifier]  # exactly once


def test_convert_dedup_keeps_distinct_qualifier_entries_and_order(tmp_path: Path) -> None:
    """DISTINCT qualifier entries survive the dedup in first-seen order.

    WHY: template declares the disease qualifier, the section re-declares it identically and
    adds two DISTINCT keys — the overlay yields [disease, disease, anatomical, sex]; the
    dedup collapses the duplicate without disturbing the distinct entries or their order.
    """
    disease: dict[str, Any] = {"qualifier": "disease_context_qualifier", "method": "value", "encoding": "MONDO:0019037"}
    anatomical: dict[str, Any] = {"qualifier": "anatomical_context_qualifier", "method": "value", "encoding": "UBERON:0000955"}
    sex: dict[str, Any] = {"qualifier": "sex_qualifier", "method": "value", "encoding": "PATO:0000384"}
    config: dict[str, Any] = _legacy_config()
    config["template"]["statement"]["qualifiers"] = [dict(disease)]
    config["sections"] = [{"statement": {"qualifiers": [dict(disease), dict(anatomical), dict(sex)]}}]
    p: Path = _legacy_file(tmp_path, config)
    downloads: Path = _downloads(tmp_path, "PMC0000000/PMC0000000.1/data.tsv")
    out: dict[str, Any] = convert_legacy(p, downloads)
    assert out["sections"][0]["statement"]["qualifiers"] == [disease, anatomical, sex]


def test_convert_dedup_leaves_scalar_values_untouched(tmp_path: Path) -> None:
    """Scalar values pass through the dedup untouched — it applies to list entries only.

    WHY: a scalar declared identically on the template and a section (``delimiter`` here) keeps
    fastmerge's later-wins value as a plain scalar; nothing is wrapped or compared.
    """
    config: dict[str, Any] = _legacy_config()
    config["template"]["source"]["delimiter"] = "\t"
    config["sections"] = [{"source": {"delimiter": "\t"}}]
    p: Path = _legacy_file(tmp_path, config)
    downloads: Path = _downloads(tmp_path, "PMC0000000/PMC0000000.1/data.tsv")
    out: dict[str, Any] = convert_legacy(p, downloads)
    delimiter: Any = out["sections"][0]["source"]["delimiter"]
    assert isinstance(delimiter, str)
    assert delimiter == "\t"


def test_convert_same_key_qualifier_section_entry_wins_first_seen_slot(tmp_path: Path) -> None:
    """The MIN1 pattern: a section re-declares the template's qualifier key with a MORE
    SPECIFIC encoding — v12 allows one entry per key, so the section's entry wins the
    first-seen slot while sections without their own entry keep the template's.

    WHY: ``Section`` rejects a qualifier key declared twice (``qualifier-duplicated``); the
    curator's intent of the re-declaration is specialization, i.e. fastmerge's later-wins.
    """
    brain: dict[str, Any] = {"qualifier": "anatomical_context_qualifier", "method": "value", "encoding": "UBERON:0000955"}
    cell: dict[str, Any] = {"qualifier": "anatomical_context_qualifier", "method": "value", "encoding": "CL:0000129"}
    config: dict[str, Any] = _legacy_config()
    config["template"]["statement"]["qualifiers"] = [dict(brain)]
    config["sections"] = [{}, {"statement": {"qualifiers": [dict(cell)]}}]
    p: Path = _legacy_file(tmp_path, config)
    downloads: Path = _downloads(tmp_path, "PMC0000000/PMC0000000.1/data.tsv")
    out: dict[str, Any] = convert_legacy(p, downloads)
    assert out["sections"][0]["statement"]["qualifiers"] == [brain]  # template entry kept
    assert out["sections"][1]["statement"]["qualifiers"] == [cell]  # section entry won
    sections, _ = _validate_converted(tmp_path, out)  # passes Section's one-per-key rule
    assert [[str(q.qualifier) for q in section.statement.qualifiers or []] for section in sections] == [
        ["anatomical_context_qualifier"],
        ["anatomical_context_qualifier"],
    ]


def test_convert_malformed_qualifiers_shape_raises_unsupported(tmp_path: Path) -> None:
    """Non-list ``qualifiers``, non-mapping entries, and empty keys are unsupported constructs."""
    downloads: Path = _downloads(tmp_path, "PMC0000000/PMC0000000.1/data.tsv")
    cases: list[Any] = [{"qualifier": "anatomical_context_qualifier"}, ["anatomical_context_qualifier"], [{"qualifier": ""}]]
    for qualifiers in cases:
        config: dict[str, Any] = _legacy_config()
        config["template"]["statement"]["qualifiers"] = qualifiers
        with pytest.raises(LegacyUnsupportedSyntaxError, match="qualifiers"):
            convert_legacy(_legacy_file(tmp_path, config), downloads)


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


def test_convert_non_list_annotations_raises_unsupported(tmp_path: Path) -> None:
    """A non-list ``annotations`` value is an unsupported construct (never expressible as a ``Section``)."""
    config: dict[str, Any] = _legacy_config()
    config["template"]["annotations"] = "effect size"
    downloads: Path = _downloads(tmp_path, "PMC0000000/PMC0000000.1/data.tsv")
    with pytest.raises(LegacyUnsupportedSyntaxError, match="`annotations` must hold a list") as excinfo:
        convert_legacy(_legacy_file(tmp_path, config), downloads)
    assert excinfo.value.code == "legacy-unsupported-syntax"


def test_convert_non_string_source_local_raises_unsupported(tmp_path: Path) -> None:
    """A non-string ``source.local`` cannot be resolved onto a payload and is unsupported."""
    config: dict[str, Any] = _legacy_config()
    config["template"]["source"]["local"] = 42
    with pytest.raises(LegacyUnsupportedSyntaxError, match=r"`source\.local` must hold a path string") as excinfo:
        convert_legacy(_legacy_file(tmp_path, config), None)
    assert excinfo.value.code == "legacy-unsupported-syntax"


def test_convert_fetch_without_publication_surfaces_unresolved(tmp_path: Path) -> None:
    """fetch=True without ``provenance.publication`` cannot know which article to fetch: unresolved, never a traceback."""
    config: dict[str, Any] = _legacy_config()
    config["template"]["provenance"].pop("publication")
    downloads: Path = tmp_path / "downloads"
    downloads.mkdir()
    with pytest.raises(LegacySourceUnresolvedError, match=r"provenance\.publication") as excinfo:
        convert_legacy(_legacy_file(tmp_path, config), downloads, fetch=True)
    assert excinfo.value.code == "legacy-source-unresolved"


def test_convert_fetch_non_pmc_publication_surfaces_unresolved(tmp_path: Path) -> None:
    """fetch=True with a publication that is not a PMC id stays unresolved (the normalize failure is recorded in ``tried``)."""
    config: dict[str, Any] = _legacy_config()
    config["template"]["provenance"]["publication"] = "not-a-pmc-id"
    downloads: Path = tmp_path / "downloads"
    downloads.mkdir()
    with pytest.raises(LegacySourceUnresolvedError, match="is not a PMC id") as excinfo:
        convert_legacy(_legacy_file(tmp_path, config), downloads, fetch=True)
    assert excinfo.value.code == "legacy-source-unresolved"


@pytest.mark.parametrize("missing", ["source", "statement"])
def test_convert_template_missing_required_block_raises_unsupported(tmp_path: Path, missing: str) -> None:
    """A template missing ``source`` or ``statement`` fails conversion — never a broken ``.v12.yaml``.

    WHY: every expanded section constructs through the Section models, so a required block the
    template never supplied surfaces as ``legacy-unsupported-syntax`` naming the section index
    (and source label when one exists), chained from the pydantic ValidationError, instead of
    writing a config that only fails downstream.
    """
    config: dict[str, Any] = _legacy_config()
    config["template"].pop(missing)
    downloads: Path = _downloads(tmp_path, "PMC0000000/PMC0000000.1/data.tsv")
    with pytest.raises(LegacyUnsupportedSyntaxError, match="section 0") as excinfo:
        convert_legacy(_legacy_file(tmp_path, config), downloads)
    assert excinfo.value.code == "legacy-unsupported-syntax"
    assert "fails `Section` validation" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, pydantic.ValidationError)


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
        assert ancestor is not None, f"tablassert.agent import outside a function at line {getattr(node, 'lineno', 0)}"


# --- US-004: convert-legacy CLI -------------------------------------------- #


def _copy_fixture(fixtures_path: Path, directory: Path, name: str) -> Path:
    """Copy a legacy fixture into ``directory`` so a conversion never writes into the repo."""
    target: Path = directory / name
    target.write_bytes((fixtures_path / name).read_bytes())
    return target


def test_convert_legacy_cli_flag_binding(tmp_path: Path) -> None:
    """``convert-legacy`` binds the positional path plus ``--downloads``/``--fetch``/``--out``.

    WHY: pins the live Cyclopts contract (same pattern as ``validate``'s flag test) — the
    positional and every documented flag bind through the parser without executing.
    """
    legacy: Path = tmp_path / "legacy.yaml"
    downloads: Path = tmp_path / "downloads"
    out: Path = tmp_path / "out"
    fn, bound, _ = cli.APP.parse_args(
        ["convert-legacy", str(legacy), "--downloads", str(downloads), "--fetch", "--out", str(out)], exit_on_error=False
    )
    assert fn is convert_legacy_command
    arguments: dict[str, Any] = dict(bound.arguments)
    assert arguments["legacy_path"] == legacy
    assert arguments["downloads"] == downloads
    assert arguments["fetch"] is True
    assert arguments["out"] == out


def test_convert_legacy_cli_nonexistent_input_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A missing input path fails loud with a usage error (exit 2) before any conversion."""
    with pytest.raises(SystemExit) as excinfo:
        convert_legacy_command(tmp_path / "missing.yaml", None, False, None)
    assert excinfo.value.code == 2
    assert "does not exist" in capsys.readouterr().err


def test_convert_legacy_cli_input_not_file_or_dir_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An existing input that is neither a file nor a directory is a usage error (exit 2)."""
    fifo: Path = tmp_path / "fifo.yaml"
    os.mkfifo(fifo)
    with pytest.raises(SystemExit) as excinfo:
        convert_legacy_command(fifo, None, False, None)
    assert excinfo.value.code == 2
    assert "neither a file nor a directory" in capsys.readouterr().err


def test_convert_legacy_cli_missing_downloads_dir_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A ``--downloads`` path that is not a directory is a usage error (exit 2)."""
    legacy: Path = _legacy_file(tmp_path, _legacy_config())
    with pytest.raises(SystemExit) as excinfo:
        convert_legacy_command(legacy, tmp_path / "nope", False, None)
    assert excinfo.value.code == 2
    assert "--downloads" in capsys.readouterr().err


def test_convert_legacy_cli_out_not_a_directory_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An ``--out`` that exists as a FILE is a usage error (exit 2), never a silent clobber."""
    legacy: Path = _legacy_file(tmp_path, _legacy_config())
    blocker: Path = tmp_path / "blocker"
    blocker.write_text("not a directory")
    with pytest.raises(SystemExit) as excinfo:
        convert_legacy_command(legacy, None, False, blocker)
    assert excinfo.value.code == 2
    assert "--out" in capsys.readouterr().err


def test_convert_legacy_cli_single_file_writes_beside_input(fixtures_path: Path, tmp_path: Path) -> None:
    """Single-file mode writes ``<stem>.v12.yaml`` next to the input by default."""
    legacy: Path = _copy_fixture(fixtures_path, tmp_path, "legacy_template_only.yaml")
    downloads: Path = _downloads(tmp_path, "PMC7878905/PMC7878905.1/CHENG1.xlsx")
    convert_legacy_command(legacy, downloads, False, None)
    converted: Path = tmp_path / "legacy_template_only.v12.yaml"
    assert converted.is_file()
    data: Any = from_yaml(converted)
    assert set(data) == {"template", "sections"}
    section: dict[str, Any] = data["sections"][0]
    assert section["source"]["local"] == str((downloads / "PMC7878905" / "PMC7878905.1" / "CHENG1.xlsx").resolve())


def test_convert_legacy_cli_single_file_out_dir(fixtures_path: Path, tmp_path: Path) -> None:
    """``--out`` redirects the output into the given directory, creating it when missing."""
    legacy: Path = _copy_fixture(fixtures_path, tmp_path, "legacy_template_only.yaml")
    downloads: Path = _downloads(tmp_path, "PMC7878905/PMC7878905.1/CHENG1.xlsx")
    out: Path = tmp_path / "v12"
    convert_legacy_command(legacy, downloads, False, out)
    assert (out / "legacy_template_only.v12.yaml").is_file()
    assert not (tmp_path / "legacy_template_only.v12.yaml").exists()


def test_convert_legacy_cli_unresolved_source_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An unresolvable ``source.local`` exits 1 with its coded message and writes nothing."""
    legacy: Path = _legacy_file(tmp_path, _legacy_config())
    downloads: Path = _downloads(tmp_path, "PMC0000000/PMC0000000.1/other.tsv")
    with pytest.raises(SystemExit) as excinfo:
        convert_legacy_command(legacy, downloads, False, None)
    assert excinfo.value.code == 1
    err: str = capsys.readouterr().err
    assert "legacy-source-unresolved" in err
    assert not (tmp_path / "legacy.v12.yaml").exists()


def test_convert_legacy_cli_fetch_without_downloads_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``--fetch`` without a downloads directory cannot fetch: coded failure, never a hang."""
    legacy: Path = _legacy_file(tmp_path, _legacy_config())
    with pytest.raises(SystemExit) as excinfo:
        convert_legacy_command(legacy, None, True, None)
    assert excinfo.value.code == 1
    assert "legacy-source-unresolved" in capsys.readouterr().err


def test_convert_legacy_cli_single_file_malformed_yaml_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A malformed YAML in single-file mode exits 1 with one FAILED stderr line, never a raw traceback."""
    legacy: Path = tmp_path / "broken.yaml"
    legacy.write_text("template: [\n")
    with pytest.raises(SystemExit) as excinfo:
        convert_legacy_command(legacy, None, False, None)
    assert excinfo.value.code == 1
    err: str = capsys.readouterr().err
    assert f"FAILED {legacy} (error)" in err
    assert not (tmp_path / "broken.v12.yaml").exists()


def test_convert_legacy_cli_batch_status_lines_and_exit(fixtures_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Directory mode prints one status line per file, keeps going, and exits 1 iff ANY failed.

    WHY: two fixtures resolve and write into ``--out`` while the third file's source stays
    unresolved — the batch must not abort on the failure, yet the exit code must be non-zero.
    """
    workdir: Path = tmp_path / "legacy"
    workdir.mkdir()
    _copy_fixture(fixtures_path, workdir, "legacy_template_only.yaml")
    _copy_fixture(fixtures_path, workdir, "legacy_multi_section.yaml")
    _legacy_file(workdir, _legacy_config())  # publication payload absent from downloads -> unresolved
    downloads: Path = _downloads(tmp_path, "PMC7878905/PMC7878905.1/CHENG1.xlsx", "PMC11530135/PMC11530135.1/CORREIA1.xlsx")
    out: Path = tmp_path / "out"
    with pytest.raises(SystemExit) as excinfo:
        convert_legacy_command(workdir, downloads, False, out)
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert captured.out.count("CONVERTED") == 2
    assert "FAILED" in captured.out
    assert "legacy-source-unresolved" in captured.out
    assert "legacy-source-unresolved" in captured.err  # full coded message on stderr
    assert (out / "legacy_template_only.v12.yaml").is_file()
    assert (out / "legacy_multi_section.v12.yaml").is_file()
    assert not (out / "legacy.v12.yaml").exists()


def test_convert_legacy_cli_batch_malformed_yaml_is_failed_not_fatal(fixtures_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A non-coded failure (malformed YAML) gets a FAILED line too and never aborts the batch."""
    workdir: Path = tmp_path / "legacy"
    workdir.mkdir()
    _copy_fixture(fixtures_path, workdir, "legacy_template_only.yaml")
    (workdir / "broken.yaml").write_text("template: [\n")
    downloads: Path = _downloads(tmp_path, "PMC7878905/PMC7878905.1/CHENG1.xlsx")
    with pytest.raises(SystemExit) as excinfo:
        convert_legacy_command(workdir, downloads, False, None)
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "CONVERTED" in captured.out
    assert "FAILED" in captured.out
    assert "(error)" in captured.out
    assert (workdir / "legacy_template_only.v12.yaml").is_file()
    assert not (workdir / "broken.v12.yaml").exists()


def test_convert_legacy_cli_batch_in_place_and_idempotent(fixtures_path: Path, tmp_path: Path) -> None:
    """Without ``--out`` the batch writes beside each input; a rerun skips its own outputs.

    WHY: ``--out`` is optional in directory mode BY DESIGN (documented in docs/cli.md) — the
    ``<stem>.v12.yaml`` suffix keeps outputs beside their inputs, and reruns must skip
    ``*.v12.yaml`` so no ``.v12.v12.yaml`` cascade appears.
    """
    workdir: Path = tmp_path / "legacy"
    workdir.mkdir()
    _copy_fixture(fixtures_path, workdir, "legacy_template_only.yaml")
    downloads: Path = _downloads(tmp_path, "PMC7878905/PMC7878905.1/CHENG1.xlsx")
    convert_legacy_command(workdir, downloads, False, None)
    assert (workdir / "legacy_template_only.v12.yaml").is_file()
    convert_legacy_command(workdir, downloads, False, None)  # rerun: only the source file converts
    assert sorted(p.name for p in workdir.glob("*.yaml")) == ["legacy_template_only.v12.yaml", "legacy_template_only.yaml"]


def test_convert_legacy_cli_empty_directory_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A directory holding no ``*.yaml`` files is a usage error (exit 2) — never a silent pass."""
    empty: Path = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit) as excinfo:
        convert_legacy_command(empty, None, False, None)
    assert excinfo.value.code == 2
    assert "no *.yaml" in capsys.readouterr().err


# --- US-005: synthetic fixtures + MOKG corpus ingestability ------------------- #


def _seed_payloads(tmp_path: Path, raw: dict[str, Any], legacy: Path) -> Path:
    """Materialize one placeholder payload per merged (publication, local basename) pair.

    WHY: every committed legacy fixture must convert against a downloads directory that can
    exist in CI, so the payloads are derived from the fixture's OWN merged sections — the
    production overlay (:func:`to_sections`) supplies them, never a hand-maintained list.
    """
    downloads: Path = tmp_path / "downloads"
    # cast: ``to_sections`` returns one dict per section, but its annotation nests one list too deep.
    expanded: list[dict[str, Any]] = cast("list[dict[str, Any]]", to_sections(raw, legacy))
    for merged in expanded:
        publication: str = merged["provenance"]["publication"]
        name: str = PurePosixPath(merged["source"]["local"]).name
        payload: Path = downloads / publication / f"{publication}.1" / name
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload.write_bytes(b"payload")
    return downloads


def _validate_converted(tmp_path: Path, out: dict[str, Any]) -> tuple[list[Section], list[warnings.WarningMessage]]:
    """Dump a converted config, re-expand it the production way, and construct every Section.

    Returns the validated sections and every warning validation fired, so callers can pin
    the US-001 drop-with-warning behavior without second-guessing the models here.
    """
    config: Path = tmp_path / "converted.yaml"
    to_yaml(config, out)
    data: Any = from_yaml(config)
    # cast: ``to_sections`` returns one dict per section, but its annotation nests one list too deep.
    expanded: list[dict[str, Any]] = cast("list[dict[str, Any]]", to_sections(data, config))
    sections: list[Section] = []
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        for merged in expanded:
            merged.pop("config")  # stamped by to_sections; not a Section field
            sections.append(Section.model_validate(merged))
    return sections, record


#: Committed legacy fixtures expected to convert AND validate fully (a pairing gap may still
#: drop an orphan with UnpairedEffectAnnotationWarning — that is a validated outcome, not a
#: failure). Together with ``legacy_pairing_gap.yaml`` they cover EVERY legacy syntax class:
#: template-only, duplicate-key mapping, reindex blocks, relationship-strength aliases.
_SYNTHETIC_CONVERTIBLE: list[str] = ["legacy_template_only.yaml", "legacy_multi_section.yaml", "legacy_duplicate_keys.yaml"]


@pytest.mark.parametrize("fixture", _SYNTHETIC_CONVERTIBLE)
def test_synthetic_fixture_converts_and_validates(fixtures_path: Path, tmp_path: Path, fixture: str) -> None:
    """Each convertible synthetic fixture parses, converts, and every section validates.

    Fixture-to-syntax-class map: ``legacy_template_only.yaml`` = template-only + reindex
    block + ``relationship_strength`` alias; ``legacy_multi_section.yaml`` = template+sections
    with the space-separated alias; ``legacy_duplicate_keys.yaml`` = duplicate-key mapping.
    No alias spelling may survive conversion under ANY of its case/separator variants.
    """
    path: Path = fixtures_path / fixture
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        raw: Any = load_legacy_yaml(path)
    assert isinstance(raw, dict)
    duplicates: list[warnings.WarningMessage] = [w for w in record if issubclass(w.category, LegacyDuplicateKeyWarning)]
    assert bool(duplicates) == (fixture == "legacy_duplicate_keys.yaml")  # only that fixture merges duplicates
    out: dict[str, Any] = convert_legacy(path, _seed_payloads(tmp_path, raw, path))
    sections, _ = _validate_converted(tmp_path, out)
    assert sections
    for section in sections:
        names: list[str] = [str(annotation.annotation) for annotation in section.annotations or []]
        assert not any("relationship" in name for name in names), f"legacy alias survived conversion: {names}"


def test_synthetic_duplicate_keys_fixture_keeps_both_occurrences(fixtures_path: Path, tmp_path: Path) -> None:
    """The duplicate-key fixture merges BOTH ``template:`` occurrences into one section.

    WHY: ``yaml.safe_load`` would keep only the second block and lose source/provenance
    entirely; the merged conversion must carry the first occurrence's ``source``/provenance
    AND the second's statement/annotations through Section validation.
    """
    path: Path = fixtures_path / "legacy_duplicate_keys.yaml"
    with pytest.warns(LegacyDuplicateKeyWarning, match=r"Duplicate key `template`"):
        raw: Any = load_legacy_yaml(path)
    assert isinstance(raw, dict)
    out: dict[str, Any] = convert_legacy(path, _seed_payloads(tmp_path, raw, path))
    section: dict[str, Any] = out["sections"][0]
    assert section["source"]["sheet"] == "Table 1"  # first occurrence survived the merge
    assert section["statement"]["predicate"] == "correlated_with"  # second occurrence too
    sections, record = _validate_converted(tmp_path, out)
    assert [str(annotation.annotation) for annotation in sections[0].annotations or []] == ["sample size", "effect size", "effect type"]
    assert not [w for w in record if issubclass(w.category, UnpairedEffectAnnotationWarning)]  # paired: nothing dropped


def test_synthetic_pairing_gap_fixture_drops_orphan_with_warning(fixtures_path: Path, tmp_path: Path) -> None:
    """The pairing-gap fixture converts (alias renamed), then Section DROPS the orphan (US-001).

    WHY: the converter must keep the renamed ``effect_size`` annotation — pairing policy
    lives ONLY on ``Section``, which drops the unpaired half with exactly one
    ``UnpairedEffectAnnotationWarning`` and keeps the section.
    """
    path: Path = fixtures_path / "legacy_pairing_gap.yaml"
    raw: Any = load_legacy_yaml(path)
    assert isinstance(raw, dict)
    out: dict[str, Any] = convert_legacy(path, _seed_payloads(tmp_path, raw, path))
    assert out["sections"][0]["annotations"] == [{"annotation": "effect_size", "method": "column", "encoding": "C"}]
    sections, record = _validate_converted(tmp_path, out)
    unpaired: list[warnings.WarningMessage] = [w for w in record if issubclass(w.category, UnpairedEffectAnnotationWarning)]
    assert len(unpaired) == 1
    assert "effect_size" in str(unpaired[0].message)
    assert sections[0].annotations is None  # the orphan was dropped, the section kept


#: The MOKG corpus holds exactly 26 legacy table configs (US-005); a different count means the
#: corpus changed and this acceptance test must be re-evaluated, never silently resized.
_MOKG_FILE_COUNT: int = 26

#: Of the 26 corpus files at least this many must CONVERT (US-008): the human ``local`` names
#: exist on no build machine, but every config's ``url`` points at the real payload object, so
#: url-basename resolution lands the large majority. A lower count means resolution regressed.
_MOKG_MIN_CONVERTED: int = 15


def test_corpus_mokg_convert_or_fail_unresolved(tmp_path: Path) -> None:
    """Env-gated ingestability acceptance over the REAL MOKG corpus (US-005).

    SKIPS with a printed reason when ``TABLASSERT_MOKG_DIR`` is unset. When set, every
    ``*.yaml`` in that directory must parse, overlay (template merged over each section), and
    run the alias/reindex conversion; each file then either converts AND every section
    validates against the downloads directory (``TABLASSERT_MOKG_DOWNLOADS``), or fails
    loudly with EXACTLY ``legacy-source-unresolved`` — no other error class escapes, no file
    is silently skipped. With a downloads directory, at least ``_MOKG_MIN_CONVERTED`` of the
    26 must convert (the ``source.url`` basenames resolve the payloads the human ``local``
    names cannot). ``--fetch`` is deliberately NOT used: the acceptance must stay
    deterministic offline; the docs runbook covers the fetching variant.
    """
    mokg_dir: str | None = os.environ.get("TABLASSERT_MOKG_DIR")
    if not mokg_dir:
        pytest.skip("TABLASSERT_MOKG_DIR is unset; point it at the MOKG legacy corpus (TABLE/MOKG) to run the 26-file ingestability acceptance")
    root: Path = Path(mokg_dir)
    assert root.is_dir(), f"TABLASSERT_MOKG_DIR is not a directory: {root}"
    downloads_env: str | None = os.environ.get("TABLASSERT_MOKG_DOWNLOADS")
    downloads: Path | None = None
    if downloads_env:
        downloads = Path(downloads_env)
        assert downloads.is_dir(), f"TABLASSERT_MOKG_DOWNLOADS is not a directory: {downloads}"

    files: list[Path] = sorted(root.glob("*.yaml"))
    assert len(files) == _MOKG_FILE_COUNT, f"expected {_MOKG_FILE_COUNT} MOKG legacy configs in {root}, found {len(files)}"

    converted: list[str] = []
    unresolved: list[str] = []
    for path in files:
        raw: Any = load_legacy_yaml(path)  # parse: duplicate-merging loader
        assert isinstance(raw, dict), f"{path.name} did not parse to a mapping"
        to_sections(raw, path)  # overlay: template fastmerged over every section
        try:
            out: dict[str, Any] = convert_legacy(path, downloads)  # alias rename + reindex check + source resolution
        except LegacySourceUnresolvedError:  # the class IS the code: it always carries legacy-source-unresolved
            unresolved.append(path.name)
            continue
        # ANY other exception escapes and fails the test: only the two outcomes above are legal.
        _validate_converted(tmp_path, out)
        converted.append(path.name)

    assert len(converted) + len(unresolved) == len(files)  # every file has a verdict: no silent skips
    if downloads is not None:
        assert len(converted) >= _MOKG_MIN_CONVERTED, (
            f"only {len(converted)}/{len(files)} MOKG configs converted; url-basename source resolution regressed (unresolved: {unresolved})"
        )
    print(f"\nMOKG corpus: {len(files)} files | downloads={downloads} | converted={len(converted)} legacy-source-unresolved={len(unresolved)}")
    for name in converted:
        print(f"  CONVERTED {name}")
    for name in unresolved:
        print(f"  FAILED    {name} (legacy-source-unresolved)")
