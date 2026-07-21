from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import polars as pl
import pytest

import tablassert.fullmap as fullmap
import tablassert.lib as lib
from tablassert.enums import Categories
from tablassert.fullmap import query_distinct, resolve
from tablassert.lib import to_store


def datassert_conn(rows: list[dict[str, Any]]) -> Any:
    con: Any = duckdb.connect(":memory:")
    con.execute("CREATE TABLE SOURCES (SOURCE_ID INTEGER, SOURCE_NAME VARCHAR, SOURCE_VERSION VARCHAR)")
    con.execute("CREATE TABLE CATEGORIES (CATEGORY_ID INTEGER, CATEGORY_NAME VARCHAR)")
    con.execute("CREATE TABLE CURIES (CURIE_ID INTEGER, CURIE VARCHAR, PREFERRED_NAME VARCHAR, CATEGORY_ID INTEGER, TAXON_ID BIGINT)")
    con.execute("CREATE TABLE SYNONYMS (SYNONYM VARCHAR, CURIE_ID INTEGER, SOURCE_ID INTEGER)")

    sources: set[tuple[int, str, str]] = set()
    categories: set[tuple[int, str]] = set()
    curies: set[tuple[int, str, str, int, int]] = set()
    synonyms: set[tuple[str, int, int]] = set()
    for row in rows:
        sources.add((int(row["source_id"]), str(row["source_name"]), str(row["source_version"])))
        categories.add((int(row["category_id"]), str(row["category_name"])))
        curies.add((int(row["curie_id"]), str(row["curie"]), str(row["preferred_name"]), int(row["category_id"]), int(row["taxon_id"])))
        synonyms.add((str(row["synonym"]), int(row["curie_id"]), int(row["source_id"])))

    con.executemany("INSERT INTO SOURCES VALUES (?, ?, ?)", list(sources))
    con.executemany("INSERT INTO CATEGORIES VALUES (?, ?)", list(categories))
    con.executemany("INSERT INTO CURIES VALUES (?, ?, ?, ?, ?)", list(curies))
    con.executemany("INSERT INTO SYNONYMS VALUES (?, ?, ?)", list(synonyms))
    return con


def datassert_row(
    synonym: str,
    curie_id: int,
    curie: str,
    preferred_name: str,
    category_id: int,
    category_name: str,
    taxon_id: int = 9606,
    source_id: int = 1,
    source_name: str = "HGNC",
    source_version: str = "2026-07",
) -> dict[str, Any]:
    return {
        "synonym": synonym,
        "curie_id": curie_id,
        "curie": curie,
        "preferred_name": preferred_name,
        "category_id": category_id,
        "category_name": category_name,
        "taxon_id": taxon_id,
        "source_id": source_id,
        "source_name": source_name,
        "source_version": source_version,
    }


# ? query_distinct Returns Empty Matches Schema For Empty Terms
def test_query_distinct_empty_terms() -> None:
    term: pl.Series = pl.Series("term", [], dtype=pl.String)
    nlp_level: pl.Series = pl.Series("nlp_level", [], dtype=pl.Int64)
    shard: pl.Series = pl.Series("shard", [], dtype=pl.Int64)
    lf: pl.LazyFrame = pl.DataFrame([term, nlp_level, shard]).lazy()

    matches: pl.DataFrame = query_distinct(lf, [], None, None, None, True)
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
        def warning(self, message: str) -> None:
            warnings.append(message)

    monkeypatch.setattr(lib, "logger", DummyLogger())

    source: pl.DataFrame = pl.DataFrame({"subject": ["none", ""], "subject_two": ["none", ""]})
    resolved: pl.LazyFrame = resolve(source.lazy(), "subject", [], log=False)

    out: Path = tmp_path / "empty_subgraph.parquet"
    saved: Path = to_store(resolved, out, "config.yaml")
    stored: pl.DataFrame = pl.read_parquet(saved)

    assert saved == out
    assert out.is_file()
    assert stored.height == 0
    assert len(warnings) == 1
    assert "EMPTY SUBGRAPH" in warnings[0]


# ? Resolve Uses Datassert Like DuckDB Schema
def test_resolve_uses_datassert_like_duckdb_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fullmap, "SHARDS", 1)
    con: Any = datassert_conn([datassert_row("brca1", 1, "HGNC:1100", "BRCA1", 1, "Gene")])
    source: pl.DataFrame = pl.DataFrame({"subject": ["brca1"], "subject_two": ["brca1"]})

    result: dict[str, Any] = resolve(source.lazy(), "subject", [con], log=False).collect().to_dicts()[0]

    assert result["subject"] == "HGNC:1100"
    assert result["subject_name"] == "BRCA1"
    assert result["subject_category"] == "biolink:Gene"
    assert result["subject_taxon"] == "NCBITaxon:9606"
    assert result["subject_source"] == "HGNC"
    assert result["subject_source_version"] == "2026-07"


# ? Resolve Honors Datassert Gene Taxon Filter
def test_resolve_honors_datassert_gene_taxon_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fullmap, "SHARDS", 1)
    con: Any = datassert_conn(
        [
            datassert_row("shared", 1, "HGNC:1", "HUMAN", 1, "Gene", taxon_id=9606),
            datassert_row("shared", 2, "MGI:1", "MOUSE", 1, "Gene", taxon_id=10090, source_id=2, source_name="MGI"),
        ]
    )
    source: pl.DataFrame = pl.DataFrame({"subject": ["shared"], "subject_two": ["shared"]})

    result: dict[str, Any] = resolve(source.lazy(), "subject", [con], taxon="9606", log=False).collect().to_dicts()[0]

    assert result["subject"] == "HGNC:1"
    assert result["subject_name"] == "HUMAN"
    assert result["subject_taxon"] == "NCBITaxon:9606"


# ? Resolve Honors Datassert Avoid Category
def test_resolve_honors_datassert_avoid_category(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fullmap, "SHARDS", 1)
    con: Any = datassert_conn(
        [
            datassert_row("ambiguous", 1, "HGNC:2", "GENE HIT", 1, "Gene"),
            datassert_row("ambiguous", 2, "MONDO:2", "DISEASE HIT", 2, "Disease", taxon_id=0, source_id=2, source_name="MONDO"),
        ]
    )
    source: pl.DataFrame = pl.DataFrame({"subject": ["ambiguous"], "subject_two": ["ambiguous"]})

    result: dict[str, Any] = resolve(source.lazy(), "subject", [con], avoid=[Categories.DISEASE], log=False).collect().to_dicts()[0]

    assert result["subject"] == "HGNC:2"
    assert result["subject_category"] == "biolink:Gene"


# ? Resolve Honors Datassert Prioritize Category
def test_resolve_honors_datassert_prioritize_category(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fullmap, "SHARDS", 1)
    con: Any = datassert_conn(
        [
            datassert_row("mapk1", 1, "HGNC:6871", "MAPK1", 1, "Gene"),
            datassert_row("mapk1", 2, "UniProtKB:P28482", "MAPK1 protein", 2, "Protein", source_id=2, source_name="UniProtKB"),
        ]
    )
    source: pl.DataFrame = pl.DataFrame({"subject": ["mapk1"], "subject_two": ["mapk1"]})

    result: dict[str, Any] = resolve(source.lazy(), "subject", [con], prioritize=[Categories.PROTEIN], log=False).collect().to_dicts()[0]

    assert result["subject"] == "UniProtKB:P28482"
    assert result["subject_category"] == "biolink:Protein"
