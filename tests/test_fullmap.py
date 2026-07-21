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
from tablassert.fullmap import fullmap_db_path, query_distinct, resolve
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


# ? query_distinct Returns Empty Matches Schema For Empty Terms
def test_query_distinct_empty_terms(tmp_path: Path) -> None:
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


# ? Empty Resolve Output Still Writes Empty Parquet And Warns
def test_empty_resolve_still_writes_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


# ? Resolve Uses Embedded Fullmap Redb Schema
def test_resolve_uses_fullmap_redb_schema(fullmap_db: Path) -> None:
    source: pl.DataFrame = pl.DataFrame({"subject": ["brca1"], "subject_two": ["brca1"]})

    result: dict[str, Any] = resolve(source.lazy(), "subject", fullmap_db, log=False).collect().to_dicts()[0]

    assert result["subject"] == "HGNC:1100"
    assert result["subject_name"] == "BRCA1"
    assert result["subject_category"] == "biolink:Gene"
    assert result["subject_taxon"] == "NCBITaxon:9606"
    assert result["subject_source"] == "HGNC"
    assert result["subject_source_version"] == "2026-07"


# ? Resolve Honors Equivalent Identifiers From Class Files
def test_resolve_uses_equivalent_identifier(fullmap_db: Path) -> None:
    source: pl.DataFrame = pl.DataFrame({"subject": ["ncbigene:672"], "subject_two": ["ncbigene672"]})

    result: dict[str, Any] = resolve(source.lazy(), "subject", fullmap_db, log=False).collect().to_dicts()[0]

    assert result["subject"] == "HGNC:1100"


# ? Resolve Honors Gene Taxon Filter
def test_resolve_honors_gene_taxon_filter(fullmap_db: Path) -> None:
    source: pl.DataFrame = pl.DataFrame({"subject": ["shared"], "subject_two": ["shared"]})

    result: dict[str, Any] = resolve(source.lazy(), "subject", fullmap_db, taxon="9606", log=False).collect().to_dicts()[0]

    assert result["subject"] == "HGNC:1"
    assert result["subject_name"] == "HUMAN"
    assert result["subject_taxon"] == "NCBITaxon:9606"


# ? Resolve Honors Avoid Category
def test_resolve_honors_avoid_category(fullmap_db: Path) -> None:
    source: pl.DataFrame = pl.DataFrame({"subject": ["ambiguous"], "subject_two": ["ambiguous"]})

    result: dict[str, Any] = resolve(source.lazy(), "subject", fullmap_db, avoid=[Categories.DISEASE], log=False).collect().to_dicts()[0]

    assert result["subject"] == "HGNC:2"
    assert result["subject_category"] == "biolink:Gene"


# ? Resolve Honors Prioritize Category
def test_resolve_honors_prioritize_category(fullmap_db: Path) -> None:
    source: pl.DataFrame = pl.DataFrame({"subject": ["mapk1"], "subject_two": ["mapk1"]})

    result: dict[str, Any] = resolve(source.lazy(), "subject", fullmap_db, prioritize=[Categories.PROTEIN], log=False).collect().to_dicts()[0]

    assert result["subject"] == "UniProtKB:P28482"
    assert result["subject_category"] == "biolink:Protein"


# ? Resolve Falls Back To NLP Level Two Matches
def test_resolve_level_two_fallback(fullmap_db: Path) -> None:
    source: pl.DataFrame = pl.DataFrame({"subject": ["brca 1"], "subject_two": ["brca1"]})

    result: dict[str, Any] = resolve(source.lazy(), "subject", fullmap_db, log=False).collect().to_dicts()[0]

    assert result["subject"] == "HGNC:1100"
    assert result["subject_nlp_level"] == 2


# ? Resolve Uses Column Context Frequency As Final Tie Breaker
def test_resolve_column_context_frequency(fullmap_db: Path) -> None:
    source: pl.DataFrame = pl.DataFrame({"subject": ["contextual"], "subject_two": ["contextual"]})

    result: dict[str, Any] = resolve(source.lazy(), "subject", fullmap_db, log=False).collect().to_dicts()[0]

    assert result["subject_category"] == "biolink:Gene"


# ? Resolve Converts NCBITaxon:0 To Null
def test_resolve_converts_zero_taxon_to_null(fullmap_db: Path) -> None:
    source: pl.DataFrame = pl.DataFrame({"subject": ["ambiguous"], "subject_two": ["ambiguous"]})

    result: dict[str, Any] = resolve(source.lazy(), "subject", fullmap_db, avoid=[Categories.GENE], log=False).collect().to_dicts()[0]

    assert result["subject"] == "MONDO:2"
    assert result["subject_taxon"] is None


# ? Rust Lookup Is Deterministic With One Or More Threads
def test_lookup_threads_match(fullmap_db: Path) -> None:
    single: list[dict[str, Any]] = rs.lookup_fullmap_terms(fullmap_db, ["brca1", "mapk1"], threads=1)
    multi: list[dict[str, Any]] = rs.lookup_fullmap_terms(fullmap_db, ["brca1", "mapk1"], threads=2)
    assert single == multi


# ? Fullmap Base Path Helper Supports File, Direct Base, And data/fullmap.redb
def test_fullmap_db_path_variants(tmp_path: Path, fullmap_db: Path) -> None:
    direct: Path = tmp_path / "fullmap.redb"
    direct.write_bytes(fullmap_db.read_bytes())
    assert fullmap_db_path(fullmap_db) == fullmap_db
    assert fullmap_db_path(tmp_path) == direct
    assert fullmap_db_path(tmp_path / "other") == tmp_path / "other" / "data" / "fullmap.redb"


# ? CLI Command Builds Fullmap Redb From Downloaded BABEL Fixtures
def test_build_fullmap_cli_function_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


# ? BABEL URL Discovery Mirrors Fullmap Constants And Exclusions
def test_babel_url_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
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


# ? polars-hash Dependency Is Not Needed For Fullmap Resolution
def test_polars_hash_dependency_removed() -> None:
    pyproject: str = Path("pyproject.toml").read_text()
    assert "polars-hash" not in pyproject
