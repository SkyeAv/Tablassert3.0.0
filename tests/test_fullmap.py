from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import polars as pl
import pytest

import tablassert.cli as cli
import tablassert.lib as lib
from tablassert import rs
from tablassert.biolink import Categories
from tablassert.cli import build_fullmap
from tablassert.fullmap import _TERM_CACHE, ResolveSpec, filter_and_rank, fullmap_db_path, join_matches, lookup_rows, resolve, resolve_batch
from tablassert.lib import to_store


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def synonym_row(curie: str, preferred_name: str, names: list[str], category: str, taxon: str = "NCBITaxon:9606") -> dict[str, Any]:
    return {"curie": curie, "preferred_name": preferred_name, "names": names, "types": [category], "taxa": [taxon]}


def class_row(curie: str, equivalents: list[str]) -> dict[str, Any]:
    return {"id": curie, "equivalent_identifiers": [{"identifier": x} for x in equivalents]}


@pytest.fixture
def fullmap_db(tmp_path: Path) -> Path:
    classes: Path = write_jsonl(tmp_path / "classes.ndjson", [class_row("HGNC:1100", ["NCBIGene:672"])])
    synonyms: Path = write_jsonl(
        tmp_path / "HGNC.ndjson",
        [
            synonym_row("HGNC:1100", "BRCA1", ["BRCA1", "breast cancer 1"], "Gene"),
            synonym_row("HGNC:1", "HUMAN", ["shared"], "Gene", taxon="NCBITaxon:9606"),
            synonym_row("MGI:1", "MOUSE", ["shared"], "Gene", taxon="NCBITaxon:10090"),
            synonym_row("HGNC:2", "GENE HIT", ["ambiguous"], "Gene"),
            synonym_row("MONDO:2", "DISEASE HIT", ["ambiguous"], "Disease", taxon="NCBITaxon:0"),
            synonym_row("HGNC:6871", "MAPK1", ["mapk1"], "Gene"),
            synonym_row("UniProtKB:P28482", "MAPK1 protein", ["mapk1"], "Protein"),
            synonym_row("MONDO:9", "Rare disease", ["contextual"], "Disease", taxon="NCBITaxon:0"),
            synonym_row("HGNC:9", "Gene A", ["contextual"], "Gene"),
            synonym_row("HGNC:10", "Gene B", ["contextual"], "Gene"),
        ],
    )
    output: Path = tmp_path / "data" / "fullmap.redb"
    rs.build_fullmap_db(output, [classes], [synonyms], threads=2)
    return output


def test_empty_resolve_still_writes_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """empty resolve output still writes empty parquet and warns."""
    warnings: list[str] = []

    class DummyLogger:
        def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
            warnings.append(message.format(*args, **kwargs) if kwargs else message)

    monkeypatch.setattr(lib, "logger", DummyLogger())

    source: pl.DataFrame = pl.DataFrame({"subject": ["none", ""], "subject_two": ["none", ""]})
    resolved: pl.LazyFrame = resolve(source.lazy(), "subject", tmp_path / "missing.redb", log=False)

    out: Path = tmp_path / "empty_subgraph.parquet"
    saved: Path = to_store(resolved, out, "config.yaml")
    stored: pl.DataFrame = pl.read_parquet(saved)

    assert saved == out
    assert out.is_file()
    assert stored.height == 0
    assert len(warnings) == 1
    assert "produced 0 rows" in warnings[0]


def test_resolve_uses_fullmap_redb_schema(fullmap_db: Path) -> None:
    """resolve uses embedded fullmap redb schema."""
    source: pl.DataFrame = pl.DataFrame({"subject": ["brca1"], "subject_two": ["brca1"]})

    result: dict[str, Any] = resolve(source.lazy(), "subject", fullmap_db, log=False).collect().to_dicts()[0]

    assert result["subject"] == "HGNC:1100"
    assert result["subject_name"] == "BRCA1"
    assert result["subject_category"] == "biolink:Gene"
    assert result["subject_taxon"] == "NCBITaxon:9606"
    assert result["subject_source"] == "HGNC"
    assert result["subject_source_version"] == rs.fullmap_source_version()


def test_resolve_uses_equivalent_identifier(fullmap_db: Path) -> None:
    """resolve honors equivalent identifiers from class files."""
    source: pl.DataFrame = pl.DataFrame({"subject": ["ncbigene:672"], "subject_two": ["ncbigene672"]})

    result: dict[str, Any] = resolve(source.lazy(), "subject", fullmap_db, log=False).collect().to_dicts()[0]

    assert result["subject"] == "HGNC:1100"


def test_resolve_honors_gene_taxon_filter(fullmap_db: Path) -> None:
    """resolve honors gene taxon filter."""
    source: pl.DataFrame = pl.DataFrame({"subject": ["shared"], "subject_two": ["shared"]})

    result: dict[str, Any] = resolve(source.lazy(), "subject", fullmap_db, taxon="9606", log=False).collect().to_dicts()[0]

    assert result["subject"] == "HGNC:1"
    assert result["subject_name"] == "HUMAN"
    assert result["subject_taxon"] == "NCBITaxon:9606"


def test_resolve_honors_avoid_category(fullmap_db: Path) -> None:
    """resolve honors avoid category."""
    source: pl.DataFrame = pl.DataFrame({"subject": ["ambiguous"], "subject_two": ["ambiguous"]})

    result: dict[str, Any] = resolve(source.lazy(), "subject", fullmap_db, avoid=[Categories.DISEASE], log=False).collect().to_dicts()[0]

    assert result["subject"] == "HGNC:2"
    assert result["subject_category"] == "biolink:Gene"


def test_resolve_honors_prioritize_category(fullmap_db: Path) -> None:
    """resolve honors prioritize category."""
    source: pl.DataFrame = pl.DataFrame({"subject": ["mapk1"], "subject_two": ["mapk1"]})

    result: dict[str, Any] = resolve(source.lazy(), "subject", fullmap_db, prioritize=[Categories.PROTEIN], log=False).collect().to_dicts()[0]

    assert result["subject"] == "UniProtKB:P28482"
    assert result["subject_category"] == "biolink:Protein"


def test_resolve_honors_avoid_category_given_strings(fullmap_db: Path) -> None:
    """avoid accepts plain strings (the build pipeline unwraps Categories via use_enum_values)."""
    source: pl.DataFrame = pl.DataFrame({"subject": ["ambiguous"], "subject_two": ["ambiguous"]})

    result: dict[str, Any] = resolve(source.lazy(), "subject", fullmap_db, avoid=["Disease"], log=False).collect().to_dicts()[0]  # pyright: ignore

    assert result["subject"] == "HGNC:2"
    assert result["subject_category"] == "biolink:Gene"


def test_resolve_honors_prioritize_category_given_strings(fullmap_db: Path) -> None:
    """prioritize accepts plain strings (the build pipeline unwraps Categories via use_enum_values)."""
    source: pl.DataFrame = pl.DataFrame({"subject": ["mapk1"], "subject_two": ["mapk1"]})

    result: dict[str, Any] = resolve(source.lazy(), "subject", fullmap_db, prioritize=["Protein"], log=False).collect().to_dicts()[0]  # pyright: ignore

    assert result["subject"] == "UniProtKB:P28482"
    assert result["subject_category"] == "biolink:Protein"


def test_pr_case_insensitive_preferred() -> None:
    """case-insensitive preferred-name hits outrank synonym-only hits at the same priority."""
    terms: pl.DataFrame = pl.DataFrame({"term": ["brca1"], "nlp_level": [1]})
    raw: pl.DataFrame = pl.DataFrame(
        {
            "term": ["brca1", "brca1"],
            "CURIE": ["HGNC:1100", "HGNC:9999"],
            "PREFERRED_NAME": ["BRCA1", "OTHER"],
            "CATEGORY_NAME": ["Gene", "Gene"],
            "TAXON_ID": [9606, 9606],
            "SOURCE_NAME": ["HGNC", "HGNC"],
            "SOURCE_VERSION": [rs.fullmap_source_version(), rs.fullmap_source_version()],
        }
    )

    matches: pl.DataFrame = filter_and_rank(raw, terms, taxon=None, prioritize=None, avoid=None, column_context=False)

    assert matches["CURIE"].to_list() == ["HGNC:1100"]
    assert matches["PR"].to_list() == [250]


def test_resolve_level_two_fallback(fullmap_db: Path) -> None:
    """resolve falls back to NLP level two matches."""
    source: pl.DataFrame = pl.DataFrame({"subject": ["brca 1"], "subject_two": ["brca1"]})

    result: dict[str, Any] = resolve(source.lazy(), "subject", fullmap_db, log=False).collect().to_dicts()[0]

    assert result["subject"] == "HGNC:1100"
    assert result["subject_nlp_level"] == 2


def test_resolve_column_context_frequency(fullmap_db: Path) -> None:
    """resolve uses column context frequency as final tie breaker."""
    source: pl.DataFrame = pl.DataFrame({"subject": ["contextual"], "subject_two": ["contextual"]})

    result: dict[str, Any] = resolve(source.lazy(), "subject", fullmap_db, log=False).collect().to_dicts()[0]

    assert result["subject_category"] == "biolink:Gene"


def test_resolve_converts_zero_taxon_to_null(fullmap_db: Path) -> None:
    """resolve converts NCBITaxon:0 to null."""
    source: pl.DataFrame = pl.DataFrame({"subject": ["ambiguous"], "subject_two": ["ambiguous"]})

    result: dict[str, Any] = resolve(source.lazy(), "subject", fullmap_db, avoid=[Categories.GENE], log=False).collect().to_dicts()[0]

    assert result["subject"] == "MONDO:2"
    assert result["subject_taxon"] is None


def test_filter_and_rank_honors_avoid_category(fullmap_db: Path) -> None:
    """filter_and_rank applies avoid-category filtering against a pre-fetched raw frame."""
    terms: pl.DataFrame = pl.DataFrame({"term": ["ambiguous"], "nlp_level": [1]})
    raw: pl.DataFrame = pl.DataFrame(rs.lookup_fullmap_terms(fullmap_db, ["ambiguous"]))

    matches: pl.DataFrame = filter_and_rank(raw, terms, taxon=None, prioritize=None, avoid=[Categories.DISEASE], column_context=True)

    assert matches.height == 1
    assert matches["CURIE"].to_list() == ["HGNC:2"]
    assert matches["CATEGORY_NAME"].to_list() == ["Gene"]


def test_filter_and_rank_empty_raw_returns_empty_matches() -> None:
    """filter_and_rank returns empty matches schema when the raw frame has no rows."""
    terms: pl.DataFrame = pl.DataFrame({"term": ["anything"], "nlp_level": [1]})
    matches: pl.DataFrame = filter_and_rank(pl.DataFrame(schema={"term": pl.String}), terms, None, None, None, True)

    assert matches.height == 0
    assert "FREQUENCY" in matches.columns


def test_filter_and_rank_exclude_prefixes_drops_curie_prefix() -> None:
    """exclude_prefixes drops rows whose CURIE prefix (text before the first ':') is listed.

    US-M3: two distinct terms each resolve to one row; excluding the OMIM prefix removes
    only the OMIM term's row and leaves the HGNC row untouched. Distinct terms keep the
    assertion unambiguous (dedup keeps one row per term, so no ranking tie is involved).
    """
    terms: pl.DataFrame = pl.DataFrame({"term": ["brca1", "omimterm"], "nlp_level": [1, 1]})
    raw: pl.DataFrame = pl.DataFrame(
        {
            "term": ["brca1", "omimterm"],
            "CURIE": ["HGNC:1100", "OMIM:123"],
            "PREFERRED_NAME": ["BRCA1", "OMIM THING"],
            "CATEGORY_NAME": ["Gene", "Gene"],
            "TAXON_ID": [9606, 9606],
            "SOURCE_NAME": ["HGNC", "OMIM"],
            "SOURCE_VERSION": [rs.fullmap_source_version(), rs.fullmap_source_version()],
        }
    )

    matches: pl.DataFrame = filter_and_rank(raw, terms, taxon=None, prioritize=None, avoid=None, column_context=False, exclude_prefixes=["OMIM"])

    assert matches["CURIE"].to_list() == ["HGNC:1100"]
    assert matches["term"].to_list() == ["brca1"]


def test_filter_and_rank_exclude_regex_drops_matching_curies() -> None:
    """exclude_regex drops rows whose CURIE matches any supplied pattern.

    A CURIE matching ``^OMIM:\\d+$`` is dropped while a non-matching CURIE survives,
    proving the pattern is applied as a regex against the whole CURIE string.
    """
    terms: pl.DataFrame = pl.DataFrame({"term": ["brca1", "omimterm"], "nlp_level": [1, 1]})
    raw: pl.DataFrame = pl.DataFrame(
        {
            "term": ["brca1", "omimterm"],
            "CURIE": ["HGNC:1100", "OMIM:123"],
            "PREFERRED_NAME": ["BRCA1", "OMIM THING"],
            "CATEGORY_NAME": ["Gene", "Gene"],
            "TAXON_ID": [9606, 9606],
            "SOURCE_NAME": ["HGNC", "OMIM"],
            "SOURCE_VERSION": [rs.fullmap_source_version(), rs.fullmap_source_version()],
        }
    )

    matches: pl.DataFrame = filter_and_rank(raw, terms, taxon=None, prioritize=None, avoid=None, column_context=False, exclude_regex=[r"^OMIM:\d+$"])

    assert matches["CURIE"].to_list() == ["HGNC:1100"]


def test_filter_and_rank_exclude_unicode_regex() -> None:
    """exclude_regex handles unicode patterns (polars str.contains uses the Rust regex engine).

    A CURIE containing ``é`` is dropped by the ``[é€]`` character class while a plain
    ASCII CURIE survives, confirming unicode classes compile and match correctly.
    """
    terms: pl.DataFrame = pl.DataFrame({"term": ["brca1", "unicodeterm"], "nlp_level": [1, 1]})
    raw: pl.DataFrame = pl.DataFrame(
        {
            "term": ["brca1", "unicodeterm"],
            "CURIE": ["HGNC:1100", "HGNC:café"],
            "PREFERRED_NAME": ["BRCA1", "UNICODE THING"],
            "CATEGORY_NAME": ["Gene", "Gene"],
            "TAXON_ID": [9606, 9606],
            "SOURCE_NAME": ["HGNC", "HGNC"],
            "SOURCE_VERSION": [rs.fullmap_source_version(), rs.fullmap_source_version()],
        }
    )

    matches: pl.DataFrame = filter_and_rank(raw, terms, taxon=None, prioritize=None, avoid=None, column_context=False, exclude_regex=["[é€]"])

    assert matches["CURIE"].to_list() == ["HGNC:1100"]


def test_filter_and_rank_default_none_byte_identical() -> None:
    """Byte-identity guard: unset, None, and [] exclude args yield identical output.

    US-M3 critical regression gate: the exclusion filters are purely additive, so with no
    exclusions configured the result must be byte-for-byte identical to the pre-feature
    six-positional call. Covers both column_context paths via a realistic multi-term frame.
    """
    terms: pl.DataFrame = pl.DataFrame({"term": ["brca1", "ambiguous"], "nlp_level": [1, 1]})
    raw: pl.DataFrame = pl.DataFrame(
        {
            "term": ["brca1", "brca1", "ambiguous", "ambiguous"],
            "CURIE": ["HGNC:1100", "HGNC:9999", "HGNC:2", "MONDO:2"],
            "PREFERRED_NAME": ["BRCA1", "OTHER", "GENE HIT", "DISEASE HIT"],
            "CATEGORY_NAME": ["Gene", "Gene", "Gene", "Disease"],
            "TAXON_ID": [9606, 9606, 9606, 0],
            "SOURCE_NAME": ["HGNC", "HGNC", "HGNC", "MONDO"],
            "SOURCE_VERSION": [rs.fullmap_source_version()] * 4,
        }
    )

    baseline: pl.DataFrame = filter_and_rank(raw, terms, None, None, None, True)
    with_none: pl.DataFrame = filter_and_rank(raw, terms, None, None, None, True, exclude_prefixes=None, exclude_regex=None)
    with_empty: pl.DataFrame = filter_and_rank(raw, terms, None, None, None, True, exclude_prefixes=[], exclude_regex=[])

    assert baseline.equals(with_none)
    assert baseline.equals(with_empty)


def test_filter_and_rank_empty_exclude_lists_noop() -> None:
    """Empty exclude lists drop nothing (the guard treats [] exactly like None).

    Even with an OMIM CURIE present that a non-empty list would remove, empty
    exclude_prefixes/exclude_regex keep every row.
    """
    terms: pl.DataFrame = pl.DataFrame({"term": ["brca1", "omimterm"], "nlp_level": [1, 1]})
    raw: pl.DataFrame = pl.DataFrame(
        {
            "term": ["brca1", "omimterm"],
            "CURIE": ["HGNC:1100", "OMIM:123"],
            "PREFERRED_NAME": ["BRCA1", "OMIM THING"],
            "CATEGORY_NAME": ["Gene", "Gene"],
            "TAXON_ID": [9606, 9606],
            "SOURCE_NAME": ["HGNC", "OMIM"],
            "SOURCE_VERSION": [rs.fullmap_source_version(), rs.fullmap_source_version()],
        }
    )

    matches: pl.DataFrame = filter_and_rank(
        raw, terms, taxon=None, prioritize=None, avoid=None, column_context=False, exclude_prefixes=[], exclude_regex=[]
    )

    assert sorted(matches["CURIE"].to_list()) == ["HGNC:1100", "OMIM:123"]


def test_join_matches_coalesces_level_one_hit(fullmap_db: Path) -> None:
    """join_matches coalesces a level one hit back into lf exactly like resolve."""
    lf: pl.LazyFrame = pl.DataFrame({"subject": ["brca1"], "subject_two": ["brca1"]}).lazy()
    terms: pl.DataFrame = pl.DataFrame({"term": ["brca1"], "nlp_level": [1]})
    raw: pl.DataFrame = pl.DataFrame(rs.lookup_fullmap_terms(fullmap_db, ["brca1"]))
    matches: pl.DataFrame = filter_and_rank(raw, terms, None, None, None, True)

    result: dict[str, Any] = join_matches(lf, "subject", matches).collect().to_dicts()[0]

    assert result["subject"] == "HGNC:1100"
    assert result["subject_name"] == "BRCA1"
    assert result["subject_category"] == "biolink:Gene"
    assert "subject_two" not in result


def test_resolve_batch_matches_sequential_resolve_per_column(fullmap_db: Path) -> None:
    """resolve_batch produces the same output as calling resolve once per column in sequence."""
    lf: pl.LazyFrame = pl.DataFrame(
        {
            "subject": ["brca1", "not-a-real-term"],
            "subject_two": ["brca1", "not-a-real-term"],
            "object": ["mapk1", "mapk1"],
            "object_two": ["mapk1", "mapk1"],
        }
    ).lazy()

    sequential: pl.DataFrame = resolve(resolve(lf, "subject", fullmap_db, log=False), "object", fullmap_db, log=False).collect()
    batched: pl.DataFrame = resolve_batch(lf, [ResolveSpec("subject"), ResolveSpec("object")], fullmap_db, log=False).collect()

    cols: list[str] = sorted(sequential.columns)
    assert cols == sorted(batched.columns)
    assert sequential.select(cols).sort(cols).to_dicts() == batched.select(cols).sort(cols).to_dicts()


def test_resolve_batch_handles_asymmetric_term_sets(fullmap_db: Path) -> None:
    """resolve_batch drops only the row whose column failed to resolve, same as sequential resolve."""
    lf: pl.LazyFrame = pl.DataFrame(
        {
            "subject": ["brca1", "not-a-real-term"],
            "subject_two": ["brca1", "not-a-real-term"],
            "object": ["mapk1", "mapk1"],
            "object_two": ["mapk1", "mapk1"],
        }
    ).lazy()

    result: list[dict[str, Any]] = resolve_batch(lf, [ResolveSpec("subject"), ResolveSpec("object")], fullmap_db, log=False).collect().to_dicts()

    assert len(result) == 1
    assert result[0]["subject"] == "HGNC:1100"
    assert result[0]["object"] == "HGNC:6871"


def test_resolve_batch_applies_each_specs_filters_independently(fullmap_db: Path) -> None:
    """resolve_batch applies each spec's own avoid/taxon/prioritize filters independently (no cross-column leakage)."""
    lf: pl.LazyFrame = pl.DataFrame(
        {"subject": ["ambiguous"], "subject_two": ["ambiguous"], "object": ["ambiguous"], "object_two": ["ambiguous"]}
    ).lazy()

    result: dict[str, Any] = (
        resolve_batch(lf, [ResolveSpec("subject", avoid=[Categories.DISEASE]), ResolveSpec("object", avoid=[Categories.GENE])], fullmap_db, log=False)
        .collect()
        .to_dicts()[0]
    )

    assert result["subject"] == "HGNC:2"
    assert result["subject_category"] == "biolink:Gene"
    assert result["object"] == "MONDO:2"
    assert result["object_category"] == "biolink:Disease"


def test_resolve_batch_carries_exclude_specs(fullmap_db: Path) -> None:
    """resolve_batch threads each spec's exclude_prefixes into filter_and_rank.

    US-M3: 'ambiguous' matches both HGNC:2 (Gene) and MONDO:2 (Disease); excluding the
    HGNC prefix drops HGNC:2 so subject resolves deterministically to MONDO:2, proving
    the ResolveSpec exclusion fields reach the hot resolution path.
    """
    lf: pl.LazyFrame = pl.DataFrame({"subject": ["ambiguous"], "subject_two": ["ambiguous"]}).lazy()

    result: dict[str, Any] = resolve_batch(lf, [ResolveSpec("subject", exclude_prefixes=["HGNC"])], fullmap_db, log=False).collect().to_dicts()[0]

    assert result["subject"] == "MONDO:2"
    assert result["subject_category"] == "biolink:Disease"


def test_resolve_batch_makes_one_redb_call_regardless_of_spec_count(fullmap_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """resolve_batch makes one redb lookup regardless of how many node columns are resolved."""
    calls: list[list[str]] = []
    original = rs.lookup_fullmap_terms

    def counting_lookup(db: Path, terms: list[str], threads: Any = None) -> list[dict[str, Any]]:
        calls.append(list(terms))
        return original(db, terms, threads=threads)

    monkeypatch.setattr(rs, "lookup_fullmap_terms", counting_lookup)

    lf: pl.LazyFrame = pl.DataFrame(
        {
            "subject": ["brca1"],
            "subject_two": ["brca1"],
            "object": ["mapk1"],
            "object_two": ["mapk1"],
            "species_context_qualifier": ["shared"],
            "species_context_qualifier_two": ["shared"],
        }
    ).lazy()

    resolve_batch(
        lf, [ResolveSpec("subject"), ResolveSpec("object"), ResolveSpec("species_context_qualifier", taxon="9606")], fullmap_db, log=False
    ).collect()

    assert len(calls) == 1


def test_resolve_batch_three_node_columns_on_sharded_db(fullmap_db: Path) -> None:
    """resolve_batch resolves subject/object/qualifier from ONE sharded fetch.

    US-104 acceptance: the fullmap DB is now a sharded layout (primary
    ``fullmap.redb`` plus sibling ``fullmap.s*.redb`` RECORDS shards). The Python
    consumer only ever hands the PRIMARY path to the rs layer, which derives the
    shard paths internally. This test confirms the sibling shard files actually
    exist on disk, that ``fullmap_db_path`` resolves the primary, then resolves
    three node columns (subject, object, species_context_qualifier) in a single
    ``resolve_batch`` call and asserts every column resolves to the correct
    CURIE. The three columns share one pooled ``rs.lookup_fullmap_terms`` fetch
    (see ``test_resolve_batch_makes_one_redb_call_regardless_of_spec_count``);
    here we prove that pooled fetch fans out across the shards and hydrates all
    three columns. The fixture's diverse terms (brca1, mapk1, shared, ambiguous,
    contextual, ...) hash across multiple of the four shards, so the shared fetch
    genuinely exercises more than one shard file.
    """
    # The sharded layout must actually be on disk: primary + sibling shard files.
    assert fullmap_db.is_file()
    shards: list[Path] = sorted(fullmap_db.parent.glob("fullmap.s*.redb"))
    assert len(shards) >= 2  # default build fans RECORDS out across 4 shards

    # fullmap_db_path resolves the PRIMARY file; the rs layer derives the shards.
    assert fullmap_db_path(fullmap_db) == fullmap_db

    lf: pl.LazyFrame = pl.DataFrame(
        {
            "subject": ["brca1"],
            "subject_two": ["brca1"],
            "object": ["mapk1"],
            "object_two": ["mapk1"],
            "species_context_qualifier": ["shared"],
            "species_context_qualifier_two": ["shared"],
        }
    ).lazy()

    result: dict[str, Any] = (
        resolve_batch(
            lf, [ResolveSpec("subject"), ResolveSpec("object"), ResolveSpec("species_context_qualifier", taxon="9606")], fullmap_db, log=False
        )
        .collect()
        .to_dicts()[0]
    )

    # Each of the three node columns resolves to the correct CURIE from the one
    # shared sharded fetch.
    assert result["subject"] == "HGNC:1100"
    assert result["subject_category"] == "biolink:Gene"
    assert result["object"] == "HGNC:6871"
    assert result["object_category"] == "biolink:Gene"
    assert result["species_context_qualifier"] == "HGNC:1"
    assert result["species_context_qualifier_name"] == "HUMAN"
    assert result["species_context_qualifier_taxon"] == "NCBITaxon:9606"


def test_lookup_threads_match(fullmap_db: Path) -> None:
    """rust lookup is deterministic with one or more threads."""
    single: list[dict[str, Any]] = rs.lookup_fullmap_terms(fullmap_db, ["brca1", "mapk1"], threads=1)
    multi: list[dict[str, Any]] = rs.lookup_fullmap_terms(fullmap_db, ["brca1", "mapk1"], threads=2)
    assert single == multi


def test_fullmap_db_path_variants(tmp_path: Path, fullmap_db: Path) -> None:
    """fullmap base path helper supports file, direct base, and data/fullmap.redb."""
    direct: Path = tmp_path / "fullmap.redb"
    direct.write_bytes(fullmap_db.read_bytes())
    assert fullmap_db_path(fullmap_db) == fullmap_db
    assert fullmap_db_path(tmp_path) == direct
    assert fullmap_db_path(tmp_path / "other") == tmp_path / "other" / "data" / "fullmap.redb"


def test_term_cache_invalidates_across_rebuild(tmp_path: Path) -> None:
    """rebuilding the fullmap DB invalidates the Python term cache.

    US-104: ``_db_cache_key`` keys the caches on the PRIMARY file's resolved path
    plus ``st_mtime`` (never the sibling shard files). ``rs.build_fullmap_db``
    removes the old primary + shards and writes fresh ones, so the primary mtime
    changes on every rebuild and the cached keys no longer match — stale entries
    are orphaned and evicted rather than served. This test builds a DB, warms
    ``_TERM_CACHE`` via ``lookup_rows``, rebuilds DIFFERENT content at the same
    path, and asserts the second lookup returns the NEW curie (not the stale
    cached one). The mtime is bumped explicitly so the test stays deterministic
    even on filesystems with coarse mtime granularity; on Linux ``st_mtime`` has
    sub-second precision, so a real rebuild changes the key on its own.
    """
    output: Path = tmp_path / "fullmap.redb"
    classes: Path = write_jsonl(tmp_path / "classes.ndjson", [class_row("HGNC:1100", ["NCBIGene:672"])])

    # v1: "brca1" resolves to HGNC:1100 and warms the term cache.
    synonyms_v1: Path = write_jsonl(tmp_path / "v1.ndjson", [synonym_row("HGNC:1100", "BRCA1", ["brca1"], "Gene")])
    rs.build_fullmap_db(output, [classes], [synonyms_v1], threads=2)
    first: list[dict[str, object]] = lookup_rows(output, ["brca1"])
    assert first[0]["CURIE"] == "HGNC:1100"
    assert any(term == "brca1" for _path, _mtime, term in _TERM_CACHE)

    # v2: rebuild at the SAME path with different content ("brca1" -> HGNC:2222),
    # then guarantee a distinct mtime so the cache key changes deterministically.
    synonyms_v2: Path = write_jsonl(tmp_path / "v2.ndjson", [synonym_row("HGNC:2222", "BRCA1", ["brca1"], "Gene")])
    rs.build_fullmap_db(output, [classes], [synonyms_v2], threads=2)
    bumped: float = output.stat().st_mtime + 10.0
    os.utime(output, (bumped, bumped))

    second: list[dict[str, object]] = lookup_rows(output, ["brca1"])
    assert second[0]["CURIE"] == "HGNC:2222"  # fresh, not the stale HGNC:1100


def test_build_fullmap_cli_function_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI command builds fullmap redb from downloaded BABEL fixtures."""
    classes: Path = write_jsonl(tmp_path / "classes.ndjson", [class_row("HGNC:1100", ["NCBIGene:672"])])
    synonyms: Path = write_jsonl(tmp_path / "HGNC.ndjson", [synonym_row("HGNC:1100", "BRCA1", ["BRCA1"], "Gene")])
    output: Path = tmp_path / "fullmap.redb"

    def fake_babel_urls(version: str, endpoints: tuple[str, ...], pattern: object) -> list[tuple[str, str]]:
        assert version == "test-version"
        if endpoints == cli.BABEL_CLASS_ENDPOINTS:
            return [("classes.ndjson", "https://example.com/classes.ndjson")]
        return [("HGNC.ndjson", "https://example.com/HGNC.ndjson")]

    def fake_download_babel_file(
        filename: str, url: str, destination: Path, retries: int = 5, on_progress: Callable[[int, int], None] | None = None
    ) -> Path:
        if filename == "classes.ndjson":
            return classes
        return synonyms

    monkeypatch.setattr(cli, "babel_urls", fake_babel_urls)
    monkeypatch.setattr(cli, "download_babel_file", fake_download_babel_file)

    build_fullmap(output=output, cache=tmp_path / "cache", version="test-version", threads=1)

    rows: list[dict[str, Any]] = rs.lookup_fullmap_terms(output, ["brca1"], threads=1)
    assert rows[0]["CURIE"] == "HGNC:1100"


def test_babel_url_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """BABEL URL discovery mirrors fullmap constants and exclusions."""

    class HasFullUrl(Protocol):
        full_url: str

    base: str = "https://stars.renci.org/var/babel_outputs/2026jul22"
    listings: dict[str, bytes] = {
        f"{base}/kgx/": b'<a href="Protein_nodes.jsonl.gz"><a href="Publication_nodes.jsonl.gz"><a href="other.txt">',
        f"{base}/synonyms/": (
            b'<a href="DrugChemicalConflated.txt.gz"><a href="umls.txt.gz"><a href="GeneProteinConflated.txt.gz"><a href="Publication.txt.gz">'
        ),
        f"{base}/synonyms-conflated/": b'<a href="gene.txt.gz"><a href="protein.txt.gz"><a href="smallmolecule.txt.gz">',
    }

    class FakeResponse:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
            return None

        def read(self) -> bytes:
            return self._body

    def fake_urlopen(request: HasFullUrl, timeout: int) -> FakeResponse:
        assert timeout == 60
        return FakeResponse(listings[request.full_url])

    monkeypatch.setattr(cli, "urlopen", fake_urlopen)

    class_urls: list[tuple[str, str]] = cli.babel_urls("2026jul22", cli.BABEL_CLASS_ENDPOINTS, cli.BABEL_CLASS_RE)
    assert class_urls == [("protein_nodes.jsonl.gz", f"{base}/kgx/Protein_nodes.jsonl.gz")]

    synonym_urls: list[tuple[str, str]] = cli.babel_urls("2026jul22", cli.BABEL_SYNONYM_ENDPOINTS, cli.BABEL_SYNONYM_RE)
    assert synonym_urls == [
        ("drugchemicalconflated.txt.gz", f"{base}/synonyms/DrugChemicalConflated.txt.gz"),
        ("umls.txt.gz", f"{base}/synonyms/umls.txt.gz"),
        ("gene.txt.gz", f"{base}/synonyms-conflated/gene.txt.gz"),
        ("protein.txt.gz", f"{base}/synonyms-conflated/protein.txt.gz"),
        ("smallmolecule.txt.gz", f"{base}/synonyms-conflated/smallmolecule.txt.gz"),
    ]
    names: list[str] = [name for name, _ in synonym_urls]
    assert not any(name.startswith(("publication", "geneproteinconflated")) for name in names)


def test_polars_hash_dependency_removed() -> None:
    """polars-hash dependency is not needed for fullmap resolution."""
    pyproject: str = Path("pyproject.toml").read_text()
    assert "polars-hash" not in pyproject


def test_resolve_batch_on_phase_fires_per_column_in_order(fullmap_db: Path) -> None:
    """resolve_batch fires resolve:<col> per spec column in order; output is unchanged vs no callback."""
    lf: pl.LazyFrame = pl.DataFrame({"subject": ["brca1"], "subject_two": ["brca1"], "object": ["mapk1"], "object_two": ["mapk1"]}).lazy()

    phases: list[str] = []
    with_cb: pl.DataFrame = resolve_batch(
        lf, [ResolveSpec("subject"), ResolveSpec("object")], fullmap_db, log=False, on_phase=phases.append
    ).collect()
    without_cb: pl.DataFrame = resolve_batch(lf, [ResolveSpec("subject"), ResolveSpec("object")], fullmap_db, log=False).collect()

    assert phases == ["resolve:subject", "resolve:object"]
    assert with_cb.to_dicts() == without_cb.to_dicts()
