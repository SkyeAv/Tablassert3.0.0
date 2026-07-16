from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import tablassert.lib as lib
from tablassert.fullmap import query_distinct, resolve
from tablassert.lib import to_store


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
