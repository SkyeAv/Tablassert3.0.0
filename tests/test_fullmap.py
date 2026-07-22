from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import polars as pl
import pytest

from tablassert import rs
import tablassert.cli as cli
from tablassert.cli import build_fullmap
import tablassert.lib as lib
from tablassert.enums import Categories
from tablassert.fullmap import ResolveSpec, filter_and_rank, fullmap_db_path, join_matches, query_distinct, resolve, resolve_batch
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
    rs.build_fullmap_db(output, [classes], [synonyms], "2026-07", threads=2, write_batch_size=2)
    return output


def test_query_distinct_empty_terms(tmp_path: Path) -> None:
    """query_distinct returns empty matches schema for empty terms."""
    term: pl.Series = pl.Series("term", [], dtype=pl.String)
    nlp_level: pl.Series = pl.Series("nlp_level", [], dtype=pl.Int64)
    lf: pl.LazyFrame = pl.DataFrame([term, nlp_level]).lazy()

    matches: pl.DataFrame = query_distinct(lf, tmp_path / "missing.redb", None, None, None, True)
    cols: list[str] = [
        "term",
        "CURIE",
        "PREFERRED_NAME",
        "CATEGORY_NAME",
        "TAXON_ID",
        "SOURCE_NAME",
        "SOURCE_VERSION",
        "NLP_LEVEL",
        "PR",
        "FREQUENCY",
    ]

    assert matches.height == 0
    assert matches.columns == cols


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
    """filter_and_rank reproduces query_distinct's avoid-category filtering against a pre-fetched raw frame."""
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


def test_build_fullmap_cli_function_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI command builds fullmap redb from downloaded BABEL fixtures."""
    classes: Path = write_jsonl(tmp_path / "classes.ndjson", [class_row("HGNC:1100", ["NCBIGene:672"])])
    synonyms: Path = write_jsonl(tmp_path / "HGNC.ndjson", [synonym_row("HGNC:1100", "BRCA1", ["BRCA1"], "Gene")])
    output: Path = tmp_path / "fullmap.redb"

    def fake_download_babel_inputs(version: str, cache: Path) -> tuple[list[Path], list[Path]]:
        assert version == "test-version"
        assert cache == tmp_path / "cache"
        return [classes], [synonyms]

    monkeypatch.setattr(cli, "download_babel_inputs", fake_download_babel_inputs)

    build_fullmap(output=output, cache=tmp_path / "cache", version="test-version", threads=1, write_batch_size=1)

    rows: list[dict[str, Any]] = rs.lookup_fullmap_terms(output, ["brca1"], threads=1)
    assert rows[0]["CURIE"] == "HGNC:1100"


def test_babel_url_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """BABEL URL discovery mirrors fullmap constants and exclusions."""

    class HasFullUrl(Protocol):
        full_url: str

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
            return None

        def read(self) -> bytes:
            return b'<a href="Protein_nodes.jsonl.gz"><a href="Publication_nodes.jsonl.gz"><a href="other.txt">'

    def fake_urlopen(request: HasFullUrl, timeout: int) -> FakeResponse:
        assert timeout == 60
        assert "https://stars.renci.org/var/babel_outputs/2025sep1/kgx/" in request.full_url
        return FakeResponse()

    monkeypatch.setattr(cli, "urlopen", fake_urlopen)

    urls: list[tuple[str, str]] = cli.babel_urls("2025sep1", cli.BABEL_CLASS_ENDPOINTS, cli.BABEL_CLASS_RE)
    assert urls == [("protein_nodes.jsonl.gz", "https://stars.renci.org/var/babel_outputs/2025sep1/kgx/Protein_nodes.jsonl.gz")]


def test_polars_hash_dependency_removed() -> None:
    """polars-hash dependency is not needed for fullmap resolution."""
    pyproject: str = Path("pyproject.toml").read_text()
    assert "polars-hash" not in pyproject
