from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

import tablassert.lib as lib
from tablassert.ingests import from_yaml
from tablassert.lib import (
    Tcode,
    clean_numeric,
    coerce_pvalue_columns,
    edge_category,
    edge_tables,
    format_numeric,
    idxname,
    infores,
    label_edge,
    numeric_columns,
    parse_edge_name,
    pvalue_target,
    strip_nulls,
)


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
    assert "id" in result
    assert isinstance(result["id"], str)
    assert len(result["id"]) == 36  # ? Standard UUID string length


# ? label_edge UUID Is Deterministic
def test_label_edge_deterministic() -> None:
    r1: dict[str, Any] = {"subject": "A", "object": "B", "predicate": "treats"}
    r2: dict[str, Any] = {"subject": "A", "object": "B", "predicate": "treats"}
    result1: dict = label_edge(r1)  # pyright: ignore
    result2: dict = label_edge(r2)  # pyright: ignore
    assert result1["id"] == result2["id"]


# ? label_edge Different Data Produces Different UUIDs
def test_label_edge_different_data() -> None:
    r1: dict[str, Any] = {"subject": "A", "predicate": "treats"}
    r2: dict[str, Any] = {"subject": "B", "predicate": "treats"}
    result1: dict = label_edge(r1)  # pyright: ignore
    result2: dict = label_edge(r2)  # pyright: ignore
    assert result1["id"] != result2["id"]


# ? Tcode collect Starts Text Sources With Csv Reader
def test_tcode_collect_starts_text_sources_with_csv_reader(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect([], None, None)  # pyright: ignore
    first_op: tuple[Any, tuple[Any]] = collected[0]

    assert first_op[0].__name__ == "csv"
    assert first_op[1] == ("\t",)


# ? Tcode Allows Unresolved Value Encodings During Validation
def test_tcode_model_allows_unresolved_value_encoding(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    data["statement"]["subject"] = {"method": "value", "encoding": "Incertae Sedis XI"}

    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store}
    )

    assert tcode_model.statement.subject.method == "value"
    assert tcode_model.statement.subject.encoding == "Incertae Sedis XI"


# ? Tcode collect Enables QC Logging By Default
def test_tcode_collect_skips_qc_by_default(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect([], None, None)  # pyright: ignore
    qc_ops: list[tuple[Any, tuple[Any]]] = [op for op in collected if op[0].__name__ == "fullmap_audit"]

    assert qc_ops == []


# ? Tcode collect Enables QC Logging When Graph QC Is Enabled
def test_tcode_collect_enables_qc_logging(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store, "qc": True}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect([], None, None)  # pyright: ignore
    qc_ops: list[tuple[Any, tuple[Any]]] = [op for op in collected if op[0].__name__ == "fullmap_audit"]

    assert len(qc_ops) == 2
    assert qc_ops[0][1] == ("subject", "sectionhash", "minimal_section.yaml", "passed", True)
    assert qc_ops[1][1] == ("object", "sectionhash", "minimal_section.yaml", "passed", True)


# ? Tcode collect Adds Biolink Update Date Edge Value
def test_tcode_collect_adds_update_date_value(monkeypatch: Any, fixtures_path: Path) -> None:
    class FixedDate:
        @classmethod
        def today(cls) -> date:
            return date(2026, 7, 16)

    monkeypatch.setattr(lib, "date", FixedDate)
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect([], None, None)  # pyright: ignore
    update_ops: list[tuple[Any, tuple[Any]]] = [op for op in collected if op[0].__name__ == "value" and op[1][0] == "update_date"]

    assert update_ops == [(lib.value, ("update_date", "2026-07-16"))]


# ? publication_curie Uses PMCID Namespace For PubMed Central
def test_publication_curie_pmc() -> None:
    assert lib.publication_curie("PMC", "PMC1234567") == "PMCID:PMC1234567"


# ? publication_curie Uses Repo Namespace For Non PMC Repositories
def test_publication_curie_pubmed() -> None:
    assert lib.publication_curie("PMID", "11708054") == "PMID:11708054"


# ? infores Lower Kebab Cases A Screaming Snake Graph Name With infores Prefix
def test_infores_screaming_snake() -> None:
    assert infores("MULTIOMICS_KG") == "infores:multiomics-kg"


# ? infores Handles Single Word And Tutorial Graph Names
def test_infores_single_and_tutorial() -> None:
    assert infores("TUTORIAL_KG") == "infores:tutorial-kg"
    assert infores("CHEMBL") == "infores:chembl"


# ? normalize Wraps Category In A List And Ensures biolink: Prefix
def test_normalize_category_list_with_biolink_prefix() -> None:
    edges: pl.LazyFrame = pl.DataFrame(
        {
            "subject": ["CURIE:1", "CURIE:2"],
            "subject_name": ["Gene A", "Protein B"],
            "subject_category": ["Gene", "biolink:Protein"],
            "subject_taxon": ["NCBITaxon:9606", "NCBITaxon:9606"],
            "subject_source": ["HGNC", "HGNC"],
            "subject_source_version": ["1", "1"],
        }
    ).lazy()
    nodes, _ = lib.normalize(edges, "subject")
    result: list[Any] = sorted(nodes.collect()["category"].to_list())
    assert result == [["biolink:Gene"], ["biolink:Protein"]]


# ? normalize Keeps Null Categories Null For strip_nulls Removal
def test_normalize_category_null_stays_null() -> None:
    edges: pl.LazyFrame = pl.DataFrame(
        {
            "subject": ["CURIE:1"],
            "subject_name": ["Gene A"],
            "subject_category": pl.Series([None], dtype=pl.String),
            "subject_taxon": ["NCBITaxon:9606"],
            "subject_source": ["HGNC"],
            "subject_source_version": ["1"],
        }
    ).lazy()
    nodes, _ = lib.normalize(edges, "subject")
    result: list[Any] = nodes.collect()["category"].to_list()
    assert result == [None]


# ? Tcode collect Emits resource_id Op When Graph Name Is Provided
def test_tcode_collect_emits_resource_id_when_named(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store, "name": "MULTIOMICS_KG"}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect([], None, None)  # pyright: ignore
    rid_ops: list[tuple[Any, tuple[Any]]] = [op for op in collected if op[0].__name__ == "value" and len(op[1]) > 0 and op[1][0] == "resource_id"]

    assert len(rid_ops) == 1
    assert rid_ops[0][1] == ("resource_id", "infores:multiomics-kg")


# ? Tcode collect Omits resource_id Op When Graph Name Is Absent (Validate Path)
def test_tcode_collect_omits_resource_id_when_unnamed(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect([], None, None)  # pyright: ignore
    rid_ops: list[tuple[Any, tuple[Any]]] = [op for op in collected if op[0].__name__ == "value" and len(op[1]) > 0 and op[1][0] == "resource_id"]

    assert rid_ops == []


# ? Tcode Captures Table Literal Value Before Regex For Column Encoded Nodes
def test_tcode_table_literal_value_before_regex_for_columns(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    data["statement"]["subject"] = {"method": "column", "encoding": "A", "regex": [{"pattern": "\\s+", "replacement": " "}]}
    data["statement"]["object"] = {"method": "column", "encoding": "B"}

    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect([], None, None)  # pyright: ignore
    targets: list[str] = [op[1][0] for op in collected if op[0].__name__ == "column" and len(op[1]) > 1]
    assert "subject_table_literal_value" in targets
    assert "object_table_literal_value" in targets

    lit_idx: int = next(i for i, op in enumerate(collected) if len(op[1]) > 0 and op[1][0] == "subject_table_literal_value")
    regex_idx: int = next(i for i, op in enumerate(collected) if op[0].__name__ == "regex" and len(op[1]) > 0 and op[1][0] == "subject")
    assert lit_idx < regex_idx


# ? Tcode Omits Table Literal Value For Value Encoded Nodes
def test_tcode_table_literal_value_absent_for_value_encoding(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect([], None, None)  # pyright: ignore
    targets: list[str] = [op[1][0] for op in collected if op[0].__name__ == "column" and len(op[1]) > 1]
    assert "subject_table_literal_value" not in targets
    assert "object_table_literal_value" not in targets


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
        lf: pl.LazyFrame, col: str, section_hash: str, config_file: str, out: str = "passed", log: bool = True, provider: str | None = None
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
        lf: pl.LazyFrame, col: str, section_hash: str, config_file: str, out: str = "passed", log: bool = True, provider: str | None = None
    ) -> pl.LazyFrame:
        calls.append(("qc", col, section_hash, config_file, out, log, provider))
        return lf.with_columns(pl.lit("YES").alias(out))

    monkeypatch.setattr(lib, "duckdb", DummyDuckDB())
    monkeypatch.setattr(lib, "resolve", fake_resolve)
    monkeypatch.setattr(lib, "fullmap_audit", fake_qc)
    monkeypatch.setattr(lib, "SHARDS", 2)

    result: list[dict[str, Any]] = lib.resolve_many("subject", ["BRCA1"], tmp_path, qc=True)

    assert result == [{"subject": "brca1", "original_subject": "BRCA1", "subject_two": "brca1", "passed": "YES"}]
    assert ("qc", "subject", "", "", "passed", True, None) in calls


# ? sig Uses Exact "p_value" Column When Present Alongside Other P-Value Columns
def test_sig_prefers_exact_p_value_column() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"p_value": [0.01, 0.1], "adjusted_p_value": [0.5, 0.5]}).lazy()
    result: pl.DataFrame = lib.sig(lf).collect()
    assert list(result["significant"]) == ["YES", "NO"]


# ? sig Falls Back To Non-Exact P-Value Column When No Exact Match
def test_sig_uses_non_exact_p_value_column() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"adjusted_p_value": [0.01, 0.1]}).lazy()
    result: pl.DataFrame = lib.sig(lf).collect()
    assert list(result["significant"]) == ["YES", "NO"]


# ? sig Picks Closest Match When Multiple Non-Exact Columns Present
def test_sig_picks_closest_non_exact_match() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"log_p_value": [0.01], "adjusted_p_value_corrected": [0.5]}).lazy()
    result: pl.DataFrame = lib.sig(lf).collect()
    # "log_p_value" has higher fuzz.ratio to "p_value" than "adjusted_p_value_corrected"
    assert list(result["significant"]) == ["YES"]


# ? sig Returns UNSURE When No P-Value Column Exists
def test_sig_returns_unsure_with_no_p_value_column() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"gene": ["BRCA1"]}).lazy()
    result: pl.DataFrame = lib.sig(lf).collect()
    assert list(result["significant"]) == ["UNSURE"]


# ? sig Marks Null P-Values As UNSURE
def test_sig_marks_null_as_unsure() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"p_value": [None, 0.01, 0.1]}).lazy()
    result: pl.DataFrame = lib.sig(lf).collect()
    assert list(result["significant"]) == ["UNSURE", "YES", "NO"]


# ? sig Marks P-Values Between Cutoff And Threshold As INCONCLUSIVE
def test_sig_marks_inconclusive_band() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"p_value": [0.01, 0.07, 0.1]}).lazy()
    result: pl.DataFrame = lib.sig(lf).collect()
    assert list(result["significant"]) == ["YES", "INCONCLUSIVE", "NO"]


# ? numeric_columns Matches Any Column With P Value In The Name
def test_numeric_columns_matches_p_value_substring() -> None:
    names: list[str] = ["p_value", "adjusted_p_value", "log_p_value", "subject"]
    result: list[str] = numeric_columns(names)
    assert result == ["p_value", "adjusted_p_value", "log_p_value"]
    assert "subject" not in result


# ? numeric_columns Matches Exact Relationship Strength And Sample Size Names
def test_numeric_columns_matches_exact_names() -> None:
    names: list[str] = ["relationship_strength", "sample_size", "cohort"]
    result: list[str] = numeric_columns(names)
    assert "relationship_strength" in result
    assert "sample_size" in result
    assert "cohort" not in result


# ? numeric_columns Is Case Insensitive On The P Value Substring
def test_numeric_columns_case_insensitive() -> None:
    names: list[str] = ["P_Value", "P_VALUE"]
    result: list[str] = numeric_columns(names)
    assert result == ["P_Value", "P_VALUE"]


# ? clean_numeric Coerces Numeric And Scientific Notation Strings To Float64
def test_clean_numeric_parses_numeric_and_scientific() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"p_value": ["1e-8", "0.05", "450"], "sample_size": ["1200", "0.42", "-1.2"]}).lazy()
    result: pl.DataFrame = clean_numeric(lf).collect()
    assert result.schema["p_value"] == pl.Float64
    assert result.schema["sample_size"] == pl.Float64
    assert result["p_value"].to_list() == [1e-8, 0.05, 450.0]
    assert result["sample_size"].to_list() == [1200.0, 0.42, -1.2]


# ? clean_numeric Drops Non Numeric Entries To Null
def test_clean_numeric_nulls_non_numeric() -> None:
    lf: pl.LazyFrame = pl.DataFrame(
        {"p_value": ["1e-8", "N/A", "", "<0.001", "abc"], "relationship_strength": ["0.85", "n/a", "NULL", "x", "y"]}
    ).lazy()
    result: pl.DataFrame = clean_numeric(lf).collect()
    assert result["p_value"].to_list() == [1e-8, None, None, None, None]
    assert result["relationship_strength"].to_list() == [0.85, None, None, None, None]


# ? clean_numeric Leaves Non Matching Columns Untouched
def test_clean_numeric_leaves_non_matching_untouched() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"subject": ["BRCA1", "TP53"], "assertion_method": ["ANOVA", "t-test"], "p_value": ["0.05", "1e-8"]}).lazy()
    result: pl.DataFrame = clean_numeric(lf).collect()
    assert result.schema["subject"] == pl.String
    assert result.schema["assertion_method"] == pl.String
    assert result.schema["p_value"] == pl.Float64
    assert result["subject"].to_list() == ["BRCA1", "TP53"]
    assert result["assertion_method"].to_list() == ["ANOVA", "t-test"]


# ? clean_numeric Is A Noop When No Numeric Columns Are Present
def test_clean_numeric_noop_without_numeric_columns() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"subject": ["BRCA1"], "cohort": ["adult"]}).lazy()
    result: pl.DataFrame = clean_numeric(lf).collect()
    assert result.schema["subject"] == pl.String
    assert result.schema["cohort"] == pl.String


# ? clean_numeric Is Idempotent On Already Float64 Columns
def test_clean_numeric_idempotent_on_float64() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"p_value": [1e-8, 0.05]}).lazy()
    once: pl.DataFrame = clean_numeric(lf).collect()
    twice: pl.DataFrame = clean_numeric(once.lazy()).collect()
    assert twice["p_value"].to_list() == [1e-8, 0.05]
    assert twice.schema["p_value"] == pl.Float64


# ? format_numeric Renders P Value Columns In Scientific Notation
def test_format_numeric_p_value_scientific() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"p_value": ["1e-8", "0.05", "0.001"], "adjusted_p_value": ["0.0001", "0.1", "0.2"]}).lazy()
    result: pl.DataFrame = format_numeric(clean_numeric(lf)).collect()
    assert result["p_value"].to_list() == ["1.0000e-08", "5.0000e-02", "1.0000e-03"]
    assert result["adjusted_p_value"].to_list() == ["1.0000e-04", "1.0000e-01", "2.0000e-01"]
    assert result.schema["p_value"] == pl.String


# ? format_numeric Renders Relationship Strength And Sample Size In Decimal General Format
def test_format_numeric_decimal_general() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"relationship_strength": ["0.85", "0.42", "0.1234"], "sample_size": ["450", "1200", "7"]}).lazy()
    result: pl.DataFrame = format_numeric(clean_numeric(lf)).collect()
    assert result["relationship_strength"].to_list() == ["0.85", "0.42", "0.1234"]
    assert result["sample_size"].to_list() == ["450", "1200", "7"]


# ? format_numeric Preserves Nulls As Null
def test_format_numeric_preserves_nulls() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"p_value": ["1e-8", "N/A", "0.05"]}).lazy()
    result: pl.DataFrame = format_numeric(clean_numeric(lf)).collect()
    assert result["p_value"].to_list() == ["1.0000e-08", None, "5.0000e-02"]


# ? format_numeric Cleans Floating Point Noise To Four Significant Figures
def test_format_numeric_cleans_float_noise() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"relationship_strength": ["0.85000000001", "0.41999999999"]}).lazy()
    result: pl.DataFrame = format_numeric(clean_numeric(lf)).collect()
    assert result["relationship_strength"].to_list() == ["0.85", "0.42"]


# ? format_numeric Is A Noop When No Numeric Columns Are Present
def test_format_numeric_noop_without_numeric_columns() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"subject": ["BRCA1"], "cohort": ["adult"]}).lazy()
    result: pl.DataFrame = format_numeric(lf).collect()
    assert result["subject"].to_list() == ["BRCA1"]
    assert result.schema["subject"] == pl.String


# ? Cleaned And Formatted Null Numeric Values Are Stripped From NDJSON Rows
def test_format_numeric_nulls_stripped_from_ndjson_rows() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"subject": ["BRCA1", "TP53"], "p_value": ["1e-8", "N/A"], "relationship_strength": ["0.85", "0.42"]}).lazy()
    formatted: pl.DataFrame = format_numeric(clean_numeric(lf)).collect()
    rows: list[dict[str, Any]] = [strip_nulls(r) for r in formatted.iter_rows(named=True)]
    assert rows[0] == {"subject": "BRCA1", "p_value": "1.0000e-08", "relationship_strength": "0.85"}
    assert "p_value" not in rows[1]
    assert rows[1]["subject"] == "TP53"
    assert rows[1]["relationship_strength"] == "0.42"


# ? compile_graph Emits Edges And Nodes After Float Formatting Config Removal
def test_compile_graph_emits_ndjson(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    sub: Path = tmp_path / "sub.parquet"
    pl.DataFrame(
        {
            "subject": ["A", "B"],
            "subject_name": ["Alpha", "Beta"],
            "subject_category": ["gene", "gene"],
            "subject_taxon": [None, None],
            "subject_source": [None, None],
            "subject_source_version": [None, None],
            "original_subject": ["A", "B"],
            "object": ["X", "Y"],
            "predicate": ["r", "r"],
            "p_value": ["1.0000e-08", "5.0000e-02"],
            "update_date": ["2026-07-16", "2026-07-16"],
        }
    ).write_parquet(sub)
    lib.compile_graph([sub], "smoke", "1.0.0")
    edges: list[str] = (tmp_path / "smoke_1.0.0.edges.ndjson").read_text().strip().splitlines()
    nodes: list[str] = (tmp_path / "smoke_1.0.0.nodes.ndjson").read_text().strip().splitlines()
    assert len(edges) == 2
    assert all('"id"' in line for line in edges)
    flat: str = "\n".join(edges)
    assert '"p_value":"1.0000e-08"' in flat
    assert '"update_date":"2026-07-16"' in flat
    assert len(nodes) >= 1


# ? compile_graph Keeps Qualifier And Publication Columns On Edges, Out Of Nodes
def test_compile_graph_keeps_qualifiers_and_publications_on_edges(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    sub: Path = tmp_path / "sub.parquet"
    pl.DataFrame(
        {
            "subject": ["A"],
            "subject_name": ["Alpha"],
            "subject_category": ["gene"],
            "subject_taxon": [None],
            "subject_source": [None],
            "subject_source_version": [None],
            "original_subject": ["A"],
            "object": ["X"],
            "predicate": ["r"],
            "disease_context_qualifier": ["MONDO:0005148"],
            "original_disease_context_qualifier": ["MONDO:0005148"],
            "publication": ["PMID:123"],
        }
    ).write_parquet(sub)
    lib.compile_graph([sub], "qual", "1.0.0")
    edges: str = (tmp_path / "qual_1.0.0.edges.ndjson").read_text()
    nodes: str = (tmp_path / "qual_1.0.0.nodes.ndjson").read_text()
    # ! Qualifier And Publication Stay On Edges
    assert "MONDO:0005148" in edges
    assert "PMID:123" in edges
    # ! Neither Becomes A Node
    assert "MONDO:0005148" not in nodes
    assert "PMID:123" not in nodes


# ? sig Computes Significance On A Cleaned Float64 P Value Column
def test_sig_works_on_cleaned_float64() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"p_value": ["1e-8", "0.5", "N/A"]}).lazy()
    cleaned: pl.LazyFrame = clean_numeric(lf)
    result: pl.DataFrame = lib.sig(cleaned).collect()
    assert result["significant"].to_list() == ["YES", "NO", "UNSURE"]


# ? edge_category Maps SmallMolecule + Disease To ChemicalEntityToDiseaseAssociation
def test_edge_category_chemical_to_disease() -> None:
    lf: pl.LazyFrame = pl.LazyFrame({"subject category": ["biolink:SmallMolecule"], "object category": ["biolink:Disease"]})
    result: pl.DataFrame = edge_category(lf).collect()
    assert result["category"].to_list()[0] == ["biolink:ChemicalEntityToDiseaseOrPhenotypicFeatureAssociation"]


# ? edge_category Maps Gene + Disease To GeneToDiseaseAssociation
def test_edge_category_gene_to_disease() -> None:
    lf: pl.LazyFrame = pl.LazyFrame({"subject category": ["biolink:Gene"], "object category": ["biolink:Disease"]})
    result: pl.DataFrame = edge_category(lf).collect()
    assert result["category"].to_list()[0] == ["biolink:GeneToDiseaseAssociation"]


# ? edge_category Bridges Protein To Gene (No ProteinTo* Associations In Biolink)
def test_edge_category_protein_to_disease() -> None:
    lf: pl.LazyFrame = pl.LazyFrame({"subject category": ["biolink:Protein"], "object category": ["biolink:Disease"]})
    result: pl.DataFrame = edge_category(lf).collect()
    assert result["category"].to_list()[0] == ["biolink:GeneToDiseaseAssociation"]


# ? edge_category Falls Back To Generic Association For Unmapped Pairs
def test_edge_category_unmapped_falls_back() -> None:
    lf: pl.LazyFrame = pl.LazyFrame({"subject category": ["biolink:Publication"], "object category": ["biolink:Pathway"]})
    result: pl.DataFrame = edge_category(lf).collect()
    assert result["category"].to_list()[0] == ["biolink:Association"]


# ? edge_category Maps Drug + Disease Through ChemicalEntity Hierarchy
def test_edge_category_drug_to_disease() -> None:
    lf: pl.LazyFrame = pl.LazyFrame({"subject category": ["biolink:Drug"], "object category": ["biolink:Disease"]})
    result: pl.DataFrame = edge_category(lf).collect()
    assert result["category"].to_list()[0] == ["biolink:ChemicalEntityToDiseaseOrPhenotypicFeatureAssociation"]


# ? edge_category Maps SequenceVariant + Disease Through Variant Role
def test_edge_category_variant_to_disease() -> None:
    lf: pl.LazyFrame = pl.LazyFrame({"subject category": ["biolink:SequenceVariant"], "object category": ["biolink:Disease"]})
    result: pl.DataFrame = edge_category(lf).collect()
    assert result["category"].to_list()[0] == ["biolink:VariantToDiseaseAssociation"]


# ? parse_edge_name Parses Standard Name
def test_parse_edge_name_standard() -> None:
    assert parse_edge_name("GeneToDiseaseAssociation") == ("Gene", ["Disease"])


# ? parse_edge_name Splits Multi-Object Names On Or
def test_parse_edge_name_multi_object() -> None:
    assert parse_edge_name("ChemicalEntityToDiseaseOrPhenotypicFeatureAssociation") == ("ChemicalEntity", ["Disease", "PhenotypicFeature"])


# ? parse_edge_name Returns None For Non-Standard Names
def test_parse_edge_name_no_to() -> None:
    assert parse_edge_name("ChemicalGeneInteractionAssociation") is None


# ? edge_tables Returns Same Object On Repeat Calls (Cached)
def test_edge_tables_cached() -> None:
    first: tuple[dict[str, str], dict[str, str]] = edge_tables()
    second: tuple[dict[str, str], dict[str, str]] = edge_tables()
    assert first is second


# ? pvalue_target Matches Common P Value Spellings
def test_pvalue_target_matches_common_spellings() -> None:
    names: list[str] = ["p value", "p-value", "p.value", "pvalue", "P VALUE", "p vals", "p-values", "P"]
    for n in names:
        assert pvalue_target(n) == "p_value", n


# ? pvalue_target Matches Bare P And Padj Style Conventions Found In Real GWAS/DESeq2 Data
def test_pvalue_target_matches_bare_p_and_padj_conventions() -> None:
    plain: list[str] = ["p SMR", "smr p", "p eQTL", "eqtl p", "gwas p", "fisher combined p", "log p"]
    for n in plain:
        assert pvalue_target(n) == "p_value", n

    adjusted: list[str] = ["padj", "p.adj", "adj.P.Val"]
    for n in adjusted:
        assert pvalue_target(n) == "adjusted_p_value", n


# ? pvalue_target Detects Adjusted P Value Variants
def test_pvalue_target_detects_adjusted_variants() -> None:
    names: list[str] = ["adjusted p value", "adjusted-p-value", "adj p value"]
    for n in names:
        assert pvalue_target(n) == "adjusted_p_value", n


# ? pvalue_target Detects Broader Adjustment Synonyms
def test_pvalue_target_detects_broader_adjustment_synonyms() -> None:
    names: list[str] = [
        "FDR",
        "Bonferroni",
        "Holm",
        "false discovery rate",
        "q value",
        "q-value",
        "bonferroni pval",
        "corrected p value",
        "corrected q value",
    ]
    for n in names:
        assert pvalue_target(n) == "adjusted_p_value", n


# ? pvalue_target Does Not Treat Bare Corrected As Adjusted Without A P/Q Value Token
def test_pvalue_target_bare_corrected_is_not_treated_as_adjusted() -> None:
    assert pvalue_target("corrected age") is None
    assert pvalue_target("batch corrected expression") is None


# ? pvalue_target Does Not Treat Bare Adjusted As Adjusted P Value Without A P/Q Value Token
# * Regression For A Real False Positive Found Auditing Production KGX Output: "fully adjusted HR"
# * Is An Adjusted Hazard Ratio, Not A P Value
def test_pvalue_target_bare_adjusted_without_pvalue_context_is_not_treated_as_adjusted() -> None:
    assert pvalue_target("fully adjusted HR") is None
    assert pvalue_target("adjusted odds ratio") is None


# ? pvalue_target Excludes Significance Flag Columns
# * Regression For A Real False Positive Found Auditing Production KGX Output: "bonferroni significance"
# * Is A Categorical Flag Like sig()'s Own "significant" Column, Not The Numeric Value
def test_pvalue_target_excludes_significance_flag_columns() -> None:
    names: list[str] = ["bonferroni significance", "nominal significance", "age significance flag", "significant"]
    for n in names:
        assert pvalue_target(n) is None, n


# ? pvalue_target Excludes Bare Q And Q Statistic Columns
# * Bare "Q" Is Deliberately Not Treated As Q Value Like Since Real Data Also Uses It For
# * Cochran's Q Test Statistic, Unrelated To Storey's Q Value
def test_pvalue_target_excludes_bare_q_and_q_statistic_columns() -> None:
    names: list[str] = ["Q degrees of freedom", "heterogeneity statistic Q", "Cochran Q statistic"]
    for n in names:
        assert pvalue_target(n) is None, n


# ? pvalue_target Excludes Unrelated Columns
def test_pvalue_target_excludes_unrelated_columns() -> None:
    names: list[str] = [
        "sample size",
        "relationship strength",
        "subject",
        "cohort",
        "top value",
        "group value",
        "hazard ratio",
        "odds ratio",
        "probe id",
        "beta",
        "standard error",
    ]
    for n in names:
        assert pvalue_target(n) is None, n


# ? pvalue_target Does Not Conflate Unadjusted With Adjusted
def test_pvalue_target_unadjusted_prefix_not_treated_as_adjusted() -> None:
    assert pvalue_target("unadjusted p value") == "p_value"
    assert pvalue_target("unadjusted HR") is None


# ? coerce_pvalue_columns Renames A Single P Value Column
def test_coerce_pvalue_columns_renames_single_p_value_column() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"p value": [0.01, 0.05]}).lazy()
    result: pl.DataFrame = coerce_pvalue_columns(lf).collect()
    assert "p_value" in result.columns
    assert "p value" not in result.columns
    assert result["p_value"].to_list() == [0.01, 0.05]


# ? coerce_pvalue_columns Renames Both P Value And Adjusted P Value Columns Together
def test_coerce_pvalue_columns_renames_both_p_value_and_adjusted() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"p value": [0.01], "adjusted p value": [0.2]}).lazy()
    result: pl.DataFrame = coerce_pvalue_columns(lf).collect()
    assert result["p_value"].to_list() == [0.01]
    assert result["adjusted_p_value"].to_list() == [0.2]


# ? coerce_pvalue_columns Picks The Best Fuzzy Match Among Multiple Candidates
def test_coerce_pvalue_columns_picks_best_fuzzy_match_among_multiple_candidates() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"log p value": [0.9], "p value": [0.01]}).lazy()
    result: pl.DataFrame = coerce_pvalue_columns(lf).collect()
    assert result["p_value"].to_list() == [0.01]
    assert result["log p value"].to_list() == [0.9]


# ? coerce_pvalue_columns Is A Noop Without P Value Like Columns
def test_coerce_pvalue_columns_noop_without_pvalue_columns() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"subject": ["BRCA1"], "cohort": ["adult"]}).lazy()
    result: pl.DataFrame = coerce_pvalue_columns(lf).collect()
    assert result.columns == ["subject", "cohort"]


# ? coerce_pvalue_columns Is A Noop When Already Canonically Named
def test_coerce_pvalue_columns_noop_when_already_canonical() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"p_value": [0.01]}).lazy()
    result: pl.DataFrame = coerce_pvalue_columns(lf).collect()
    assert result.columns == ["p_value"]
    assert result["p_value"].to_list() == [0.01]


# ? Tcode Coerces P Value Columns After Annotations And Before clean_numeric
# * So Downstream numeric_columns/sig/format_numeric See Already Canonical p_value/adjusted_p_value Names
def test_tcode_collect_coerces_pvalue_before_clean_numeric(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect([], None, None)  # pyright: ignore
    coerce_idx: int = next(i for i, op in enumerate(collected) if op[0].__name__ == "coerce_pvalue_columns")
    clean_idx: int = next(i for i, op in enumerate(collected) if op[0].__name__ == "clean_numeric")

    assert coerce_idx < clean_idx
