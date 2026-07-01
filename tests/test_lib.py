from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

import tablassert.lib as lib
from tablassert.ingests import from_yaml
from tablassert.lib import Tcode, idxname, label_edge, strip_nulls


# ? idxname Converts Single Letter Columns
def test_idxname_single_letter() -> None:
    assert idxname("A") == "column_1"
    assert idxname("B") == "column_2"
    assert idxname("Z") == "column_26"


# ? idxname Converts Double Letter Columns
def test_idxname_double_letter() -> None:
    assert idxname("AA") == "column_27"
    assert idxname("AB") == "column_28"
    assert idxname("AZ") == "column_52"


# ? idxname Converts Triple Letter Columns
def test_idxname_triple_letter() -> None:
    assert idxname("AAA") == "column_703"


# ? idxname Returns Column Prefixed String
def test_idxname_format() -> None:
    result: str = idxname("C")
    assert result.startswith("column_")


# ? strip_nulls Removes Null Like Values
def test_strip_nulls_removes_empty_string() -> None:
    r: dict[str, Any] = {"a": "hello", "b": ""}
    result: dict = strip_nulls(r)
    assert "a" in result
    assert "b" not in result


# ? strip_nulls Removes Na Nan Null None
def test_strip_nulls_removes_null_variants() -> None:
    r: dict[str, Any] = {"a": "na", "b": "nan", "c": "null", "d": "none"}
    result: dict = strip_nulls(r)
    assert len(result) == 0


# ? strip_nulls Case Insensitive
def test_strip_nulls_case_insensitive() -> None:
    r: dict[str, Any] = {"a": "NA", "b": "NaN", "c": "NULL", "d": "None"}
    result: dict = strip_nulls(r)
    assert len(result) == 0


# ? strip_nulls Preserves Valid Values
def test_strip_nulls_preserves_valid() -> None:
    r: dict[str, Any] = {"name": "BRCA1", "score": 0.05, "active": True}
    result: dict = strip_nulls(r)
    assert result["name"] == "BRCA1"
    assert result["score"] == 0.05
    assert result["active"] is True


# ? strip_nulls Handles Nested Dicts
def test_strip_nulls_nested_dict() -> None:
    r: dict[str, Any] = {"outer": {"inner": "na", "keep": "yes"}}
    result: dict = strip_nulls(r)
    assert "keep" in result["outer"]
    assert "inner" not in result["outer"]


# ? strip_nulls Handles Lists Of Dicts
def test_strip_nulls_list_of_dicts() -> None:
    r: dict[str, Any] = {"items": [{"a": "keep", "b": ""}, {"a": "also", "c": "null"}]}
    result: dict = strip_nulls(r)
    assert result["items"][0] == {"a": "keep"}
    assert result["items"][1] == {"a": "also"}


# ? strip_nulls Handles Empty Dict
def test_strip_nulls_empty_dict() -> None:
    r: dict[str, Any] = {}
    result: dict = strip_nulls(r)
    assert result == {}


# ? strip_nulls Strips Whitespace Before Check
def test_strip_nulls_whitespace() -> None:
    r: dict[str, Any] = {"a": "  ", "b": " na "}
    result: dict = strip_nulls(r)
    assert len(result) == 0


# ? label_edge Assigns UUID Under Domain
def test_label_edge_assigns_uuid() -> None:
    r: dict[str, Any] = {"subject": "A", "object": "B", "predicate": "treats"}
    result: dict = label_edge(r)  # pyright: ignore
    assert "uuid" in result
    assert isinstance(result["uuid"], str)
    assert len(result["uuid"]) == 36  # ? Standard UUID string length


# ? label_edge UUID Is Deterministic
def test_label_edge_deterministic() -> None:
    r1: dict[str, Any] = {"subject": "A", "object": "B", "predicate": "treats"}
    r2: dict[str, Any] = {"subject": "A", "object": "B", "predicate": "treats"}
    result1: dict = label_edge(r1)  # pyright: ignore
    result2: dict = label_edge(r2)  # pyright: ignore
    assert result1["uuid"] == result2["uuid"]


# ? label_edge Different Data Produces Different UUIDs
def test_label_edge_different_data() -> None:
    r1: dict[str, Any] = {"subject": "A", "predicate": "treats"}
    r2: dict[str, Any] = {"subject": "B", "predicate": "treats"}
    result1: dict = label_edge(r1)  # pyright: ignore
    result2: dict = label_edge(r2)  # pyright: ignore
    assert result1["uuid"] != result2["uuid"]


# ? Tcode collect Threads Downloader Context Into from_url
def test_tcode_collect_threads_download_context(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "number": 7, "config": fixtures_path / "minimal_section.yaml", "store": store}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect([], None, None)  # pyright: ignore
    first_op: tuple[Any, tuple[Any]] = collected[0]

    assert first_op[0].__name__ == "from_url"
    assert first_op[1] == ("https://example.com/test.tsv", Path("test.tsv"), "minimal_section.yaml", "sectionhash")


# ? Tcode Allows Unresolved Value Encodings During Validation
def test_tcode_model_allows_unresolved_value_encoding(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    data["statement"]["subject"] = {"method": "value", "encoding": "Incertae Sedis XI"}

    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "number": 55, "config": fixtures_path / "minimal_section.yaml", "store": store}
    )

    assert tcode_model.statement.subject.method == "value"
    assert tcode_model.statement.subject.encoding == "Incertae Sedis XI"


# ? Tcode collect Enables QC Logging By Default
def test_tcode_collect_skips_qc_by_default(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "number": 7, "config": fixtures_path / "minimal_section.yaml", "store": store}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect([], None, None)  # pyright: ignore
    qc_ops: list[tuple[Any, tuple[Any]]] = [op for op in collected if op[0].__name__ == "fullmap_audit"]

    assert qc_ops == []


# ? Tcode collect Enables QC Logging When Graph QC Is Enabled
def test_tcode_collect_enables_qc_logging(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "number": 7, "config": fixtures_path / "minimal_section.yaml", "store": store, "qc": True}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect([], None, None)  # pyright: ignore
    qc_ops: list[tuple[Any, tuple[Any]]] = [op for op in collected if op[0].__name__ == "fullmap_audit"]

    assert len(qc_ops) == 2
    assert qc_ops[0][1] == ("subject", "sectionhash", "minimal_section.yaml", "passed", True)
    assert qc_ops[1][1] == ("object", "sectionhash", "minimal_section.yaml", "passed", True)


# ? publication_curie Uses PMCID Namespace For PubMed Central
def test_publication_curie_pmc() -> None:
    assert lib.publication_curie("PMC", "PMC1234567") == "PMCID:PMC1234567"


# ? publication_curie Uses Repo Namespace For Non PMC Repositories
def test_publication_curie_pubmed() -> None:
    assert lib.publication_curie("PMID", "11708054") == "PMID:11708054"


# ? Tcode Captures Table Literal Value Before Regex For Column Encoded Nodes
def test_tcode_table_literal_value_before_regex_for_columns(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    data["statement"]["subject"] = {
        "method": "column",
        "encoding": "A",
        "regex": [{"pattern": "\\s+", "replacement": " "}],
    }
    data["statement"]["object"] = {"method": "column", "encoding": "B"}

    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "number": 7, "config": fixtures_path / "minimal_section.yaml", "store": store}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect([], None, None)  # pyright: ignore
    targets: list[str] = [op[1][0] for op in collected if op[0].__name__ == "column" and len(op[1]) > 1]
    assert "subject table literal value" in targets
    assert "object table literal value" in targets

    lit_idx: int = next(
        i for i, op in enumerate(collected) if len(op[1]) > 0 and op[1][0] == "subject table literal value"
    )
    regex_idx: int = next(
        i for i, op in enumerate(collected) if op[0].__name__ == "regex" and len(op[1]) > 0 and op[1][0] == "subject"
    )
    assert lit_idx < regex_idx


# ? Tcode Omits Table Literal Value For Value Encoded Nodes
def test_tcode_table_literal_value_absent_for_value_encoding(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "number": 7, "config": fixtures_path / "minimal_section.yaml", "store": store}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect([], None, None)  # pyright: ignore
    targets: list[str] = [op[1][0] for op in collected if op[0].__name__ == "column" and len(op[1]) > 1]
    assert "subject table literal value" not in targets
    assert "object table literal value" not in targets


# ? resolve_many Skips QC When Disabled
def test_resolve_many_skips_qc(monkeypatch: Any, tmp_path: Path) -> None:
    calls: list[tuple[Any, ...]] = []

    class DummyConn:
        def __enter__(self) -> object:
            return object()

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

    class DummyDuckDB:
        def connect(self, path: Path, read_only: bool = True) -> DummyConn:
            calls.append(("connect", path, read_only))
            return DummyConn()

    def fake_resolve(lf: pl.LazyFrame, col: str, conns: list[object], **kwargs: Any) -> pl.LazyFrame:
        calls.append(("resolve", col, len(conns), kwargs))
        return lf

    def fake_qc(
        lf: pl.LazyFrame,
        col: str,
        section_hash: str,
        config_file: str,
        out: str = "passed",
        log: bool = True,
        provider: str | None = None,
    ) -> pl.LazyFrame:
        calls.append(("qc", col, section_hash, config_file, out, log, provider))
        return lf

    monkeypatch.setattr(lib, "duckdb", DummyDuckDB())
    monkeypatch.setattr(lib, "resolve", fake_resolve)
    monkeypatch.setattr(lib, "fullmap_audit", fake_qc)
    monkeypatch.setattr(lib, "SHARDS", 2)

    result: list[dict[str, Any]] = lib.resolve_many("subject", ["BRCA1", "TP53"], tmp_path, qc=False)

    assert len(result) == 2
    assert any(call[0] == "resolve" for call in calls)
    assert not any(call[0] == "qc" for call in calls)


# ? resolve_many Runs QC With Logging When Enabled
def test_resolve_many_runs_qc(monkeypatch: Any, tmp_path: Path) -> None:
    calls: list[tuple[Any, ...]] = []

    class DummyConn:
        def __enter__(self) -> object:
            return object()

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

    class DummyDuckDB:
        def connect(self, path: Path, read_only: bool = True) -> DummyConn:
            calls.append(("connect", path, read_only))
            return DummyConn()

    def fake_resolve(lf: pl.LazyFrame, col: str, conns: list[object], **kwargs: Any) -> pl.LazyFrame:
        calls.append(("resolve", col, len(conns), kwargs))
        return lf

    def fake_qc(
        lf: pl.LazyFrame,
        col: str,
        section_hash: str,
        config_file: str,
        out: str = "passed",
        log: bool = True,
        provider: str | None = None,
    ) -> pl.LazyFrame:
        calls.append(("qc", col, section_hash, config_file, out, log, provider))
        return lf.with_columns(pl.lit("YES").alias(out))

    monkeypatch.setattr(lib, "duckdb", DummyDuckDB())
    monkeypatch.setattr(lib, "resolve", fake_resolve)
    monkeypatch.setattr(lib, "fullmap_audit", fake_qc)
    monkeypatch.setattr(lib, "SHARDS", 2)

    result: list[dict[str, Any]] = lib.resolve_many("subject", ["BRCA1"], tmp_path, qc=True)

    assert result == [{"subject": "brca1", "original subject": "BRCA1", "subject two": "brca1", "passed": "YES"}]
    assert ("qc", "subject", "", "", "passed", True, None) in calls


# ? sig Uses Exact "p value" Column When Present Alongside Other P-Value Columns
def test_sig_prefers_exact_p_value_column() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"p value": [0.01, 0.1], "adjusted p value": [0.5, 0.5]}).lazy()
    result: pl.DataFrame = lib.sig(lf).collect()
    assert list(result["significant"]) == ["YES", "NO"]


# ? sig Falls Back To Non-Exact P-Value Column When No Exact Match
def test_sig_uses_non_exact_p_value_column() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"adjusted p value": [0.01, 0.1]}).lazy()
    result: pl.DataFrame = lib.sig(lf).collect()
    assert list(result["significant"]) == ["YES", "NO"]


# ? sig Picks Closest Match When Multiple Non-Exact Columns Present
def test_sig_picks_closest_non_exact_match() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"log p value": [0.01], "adjusted p value corrected": [0.5]}).lazy()
    result: pl.DataFrame = lib.sig(lf).collect()
    # "log p value" has higher fuzz.ratio to "p value" than "adjusted p value corrected"
    assert list(result["significant"]) == ["YES"]


# ? sig Returns UNSURE When No P-Value Column Exists
def test_sig_returns_unsure_with_no_p_value_column() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"gene": ["BRCA1"]}).lazy()
    result: pl.DataFrame = lib.sig(lf).collect()
    assert list(result["significant"]) == ["UNSURE"]


# ? sig Marks Null P-Values As UNSURE
def test_sig_marks_null_as_unsure() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"p value": [None, 0.01, 0.1]}).lazy()
    result: pl.DataFrame = lib.sig(lf).collect()
    assert list(result["significant"]) == ["UNSURE", "YES", "NO"]
