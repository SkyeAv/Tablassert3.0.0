from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import polars as pl

import tablassert.lib as lib
from tablassert.enums import ALLOWED_EDGE_FIELDS, Categories, Repositories
from tablassert.fullmap import ResolveSpec
from tablassert.ingests import from_yaml
from tablassert.lib import (
    Tcode,
    clean_numeric,
    coerce_pvalue_columns,
    coerce_study_size_columns,
    drop_not_significant,
    edge_category,
    edge_tables,
    fold_unknown_to_supporting_text,
    format_numeric,
    idx,
    idxname,
    infores,
    numeric_columns,
    parse_edge_name,
    publications,
    pvalue_target,
    study_size_target,
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

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect(Path("/tmp/fullmap.redb"))  # pyright: ignore
    qc_ops: list[tuple[Any, tuple[Any]]] = [op for op in collected if op[0].__name__ == "fullmap_audit"]

    assert qc_ops == []


# ? Tcode collect Enables QC Logging When Graph QC Is Enabled
def test_tcode_collect_enables_qc_logging(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store, "qc": True}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect(Path("/tmp/fullmap.redb"))  # pyright: ignore
    qc_ops: list[tuple[Any, tuple[Any]]] = [op for op in collected if op[0].__name__ == "fullmap_audit"]

    assert len(qc_ops) == 2
    assert qc_ops[0][1] == ("subject", "sectionhash", "minimal_section.yaml", "passed", True)
    assert qc_ops[1][1] == ("object", "sectionhash", "minimal_section.yaml", "passed", True)


# ? Tcode collect Orders drop_not_significant Before resolve_batch In Release Mode
# * Rows That Will Be Dropped For Insignificance Must Never Reach The Expensive Fullmap Resolve Step
def test_tcode_collect_orders_significance_before_resolve_when_release(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash_release.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store, "release": True}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect(Path("/tmp/fullmap.redb"))  # pyright: ignore
    drop_idx: int = next(i for i, op in enumerate(collected) if op[0].__name__ == "drop_not_significant")
    resolve_idx: int = next(i for i, op in enumerate(collected) if op[0].__name__ == "resolve_batch")

    assert drop_idx < resolve_idx


# ? Tcode collect Omits drop_not_significant Without Release But Keeps sig Before resolve_batch
def test_tcode_collect_omits_drop_not_significant_without_release(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash_norelease.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect(Path("/tmp/fullmap.redb"))  # pyright: ignore
    names: list[str] = [op[0].__name__ for op in collected]

    assert "drop_not_significant" not in names
    assert names.index("sig") < names.index("resolve_batch")


# ? Tcode collect Emits Exactly One resolve_batch Op Covering Subject/Object/Qualifiers
def test_tcode_collect_emits_single_resolve_batch_for_all_node_columns(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash_batch.parquet")
    data["statement"]["subject"]["taxon"] = 9606
    data["statement"]["object"]["prioritize"] = ["Gene"]
    data["statement"]["qualifiers"] = [
        {"qualifier": "species_context_qualifier", "method": "value", "encoding": "NCBITaxon:9606", "avoid": ["Disease"]},
        {"qualifier": "anatomical_context_qualifier", "method": "value", "encoding": "UBERON:0000061"},
    ]

    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect(Path("/tmp/fullmap.redb"))  # pyright: ignore
    batch_ops: list[tuple[Any, tuple[Any]]] = [op for op in collected if op[0].__name__ == "resolve_batch"]

    assert len(batch_ops) == 1
    specs: list[ResolveSpec] = batch_ops[0][1][0]
    assert [spec.col for spec in specs] == ["subject", "object", "species_context_qualifier", "anatomical_context_qualifier"]
    assert specs[0].taxon == "9606"
    assert specs[1].prioritize == [Categories.GENE]
    assert specs[2].avoid == [Categories.DISEASE]


# ? Tcode collect Runs Every Node Column's QC Audit After The Single resolve_batch Op
def test_tcode_collect_audits_follow_single_resolve_batch_with_qualifiers(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash_batch_qc.parquet")
    data["statement"]["qualifiers"] = [{"qualifier": "species_context_qualifier", "method": "value", "encoding": "NCBITaxon:9606"}]

    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store, "qc": True}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect(Path("/tmp/fullmap.redb"))  # pyright: ignore
    batch_idx: int = next(i for i, op in enumerate(collected) if op[0].__name__ == "resolve_batch")
    audit_ops: list[tuple[int, tuple[Any, tuple[Any]]]] = [(i, op) for i, op in enumerate(collected) if op[0].__name__ == "fullmap_audit"]

    assert [op[1][0] for _, op in audit_ops] == ["subject", "object", "species_context_qualifier"]
    assert all(i > batch_idx for i, _ in audit_ops)


# ? Tcode collect Runs predicate/edge_category After The Single resolve_batch Op
def test_tcode_collect_edge_ops_follow_resolve_batch(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash_edge_order.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect(Path("/tmp/fullmap.redb"))  # pyright: ignore
    batch_idx: int = next(i for i, op in enumerate(collected) if op[0].__name__ == "resolve_batch")
    predicate_idx: int = next(i for i, op in enumerate(collected) if op[0].__name__ == "value" and op[1][0] == "predicate")
    edge_category_idx: int = next(i for i, op in enumerate(collected) if op[0].__name__ == "edge_category")

    assert batch_idx < predicate_idx
    assert batch_idx < edge_category_idx


# ? Tcode collect Passes The Local Source Path Through To The csv Reader
def test_tcode_collect_passes_local_path_to_csv_reader(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect(Path("/tmp/fullmap.redb"))  # pyright: ignore
    csv_ops: list[tuple[Any, tuple[Any]]] = [op for op in collected if op[0].__name__ == "csv"]

    assert csv_ops[0][1] == (tcode_model.source.local, tcode_model.source.delimiter)  # pyright: ignore


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


# ? upstream_resource_ids Uses PubMed Central InfoRes For PMC Repositories
def test_upstream_resource_ids_pmc() -> None:
    assert lib.upstream_resource_ids(Repositories.PUBMED_CENTRAL) == ["infores:pubmed-central"]


# ? upstream_resource_ids Uses PubMed InfoRes For PMID Repositories
def test_upstream_resource_ids_pubmed() -> None:
    assert lib.upstream_resource_ids(Repositories.PUBMED) == ["infores:pubmed"]


# ? Tcode collect Adds Upstream Resource IDs From Provenance Repository
def test_tcode_collect_adds_upstream_resource_ids(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect(Path("/tmp/fullmap.redb"))  # pyright: ignore
    ops: list[tuple[Any, tuple[Any]]] = [op for op in collected if len(op[1]) > 0 and op[1][0] == "upstream_resource_ids"]
    assert ops[0][1] == ("upstream_resource_ids", ["infores:pubmed-central"])


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

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect(Path("/tmp/fullmap.redb"))  # pyright: ignore
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

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect(Path("/tmp/fullmap.redb"))  # pyright: ignore
    rid_ops: list[tuple[Any, tuple[Any]]] = [op for op in collected if op[0].__name__ == "value" and len(op[1]) > 0 and op[1][0] == "resource_id"]

    assert rid_ops == []


# ? Tcode Emits Source Record URLs As A List Column
def test_tcode_collect_emits_source_record_urls_list(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect(Path("/tmp/fullmap.redb"))  # pyright: ignore
    source_ops: list[tuple[Any, tuple[Any]]] = [op for op in collected if op[0].__name__ == "source_record_urls"]
    url_ops: list[tuple[Any, tuple[Any]]] = [op for op in collected if op[0].__name__ == "value" and len(op[1]) > 0 and op[1][0] == "url"]
    lf: pl.LazyFrame = pl.DataFrame({"subject": ["A"]}).lazy()
    result: pl.DataFrame = source_ops[0][0](lf, *source_ops[0][1]).collect()

    assert len(source_ops) == 1
    assert url_ops == []
    assert "source_record_urls" in result.columns
    assert "url" not in result.columns
    assert result["source_record_urls"].to_list() == [["https://example.com/test.tsv"]]


# ? Tcode Captures Original Value Before Regex For Column Encoded Nodes
def test_tcode_original_value_before_regex_for_columns(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    data["statement"]["subject"] = {"method": "column", "encoding": "A", "regex": [{"pattern": "\\s+", "replacement": " "}]}
    data["statement"]["object"] = {"method": "column", "encoding": "B"}

    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect(Path("/tmp/fullmap.redb"))  # pyright: ignore
    targets: list[str] = [op[1][0] for op in collected if op[0].__name__ == "column" and len(op[1]) > 1]
    assert "original_subject" in targets
    assert "original_object" in targets

    lit_idx: int = next(i for i, op in enumerate(collected) if len(op[1]) > 0 and op[1][0] == "original_subject")
    regex_idx: int = next(i for i, op in enumerate(collected) if op[0].__name__ == "regex" and len(op[1]) > 0 and op[1][0] == "subject")
    assert lit_idx < regex_idx


# ? Tcode Emits Original Value For Value Encoded Nodes
def test_tcode_original_value_present_for_value_encoding(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect(Path("/tmp/fullmap.redb"))  # pyright: ignore
    targets: list[str] = [op[1][0] for op in collected if op[0].__name__ == "column" and len(op[1]) > 1]
    assert "original_subject" in targets
    assert "original_object" in targets


# ? resolve_many Skips QC When Disabled
def test_resolve_many_skips_qc(monkeypatch: Any, tmp_path: Path) -> None:
    calls: list[tuple[Any, ...]] = []

    def fake_resolve(lf: pl.LazyFrame, col: str, db: Path, **kwargs: Any) -> pl.LazyFrame:
        calls.append(("resolve", col, db, kwargs))
        return lf

    def fake_qc(
        lf: pl.LazyFrame, col: str, section_hash: str, config_file: str, out: str = "passed", log: bool = True, provider: str | None = None
    ) -> pl.LazyFrame:
        calls.append(("qc", col, section_hash, config_file, out, log, provider))
        return lf

    monkeypatch.setattr(lib, "resolve", fake_resolve)
    monkeypatch.setattr(lib, "fullmap_audit", fake_qc)

    result: list[dict[str, Any]] = lib.resolve_many("subject", ["BRCA1", "TP53"], tmp_path, qc=False)

    assert len(result) == 2
    assert (
        "resolve",
        "subject",
        tmp_path / "data" / "fullmap.redb",
        {"taxon": None, "prioritize": None, "avoid": None, "column_context": True},
    ) in calls
    assert not any(call[0] == "qc" for call in calls)


# ? resolve_many Runs QC With Logging When Enabled
def test_resolve_many_runs_qc(monkeypatch: Any, tmp_path: Path) -> None:
    calls: list[tuple[Any, ...]] = []

    def fake_resolve(lf: pl.LazyFrame, col: str, db: Path, **kwargs: Any) -> pl.LazyFrame:
        calls.append(("resolve", col, db, kwargs))
        return lf

    def fake_qc(
        lf: pl.LazyFrame, col: str, section_hash: str, config_file: str, out: str = "passed", log: bool = True, provider: str | None = None
    ) -> pl.LazyFrame:
        calls.append(("qc", col, section_hash, config_file, out, log, provider))
        return lf.with_columns(pl.lit("YES").alias(out))

    monkeypatch.setattr(lib, "resolve", fake_resolve)
    monkeypatch.setattr(lib, "fullmap_audit", fake_qc)

    result: list[dict[str, Any]] = lib.resolve_many("subject", ["BRCA1"], tmp_path, qc=True)

    assert result == [{"subject": "brca1", "original_subject": "BRCA1", "subject_two": "brca1", "passed": "YES"}]
    assert ("qc", "subject", "", "", "passed", True, None) in calls


# ? resolve_many Accepts A Direct Fullmap Redb File Path
def test_resolve_many_accepts_direct_fullmap_file(monkeypatch: Any, tmp_path: Path) -> None:
    calls: list[Path] = []
    db: Path = tmp_path / "fullmap.redb"
    db.touch()

    def fake_resolve(lf: pl.LazyFrame, col: str, db_path: Path, **kwargs: Any) -> pl.LazyFrame:
        calls.append(db_path)
        return lf

    monkeypatch.setattr(lib, "resolve", fake_resolve)

    result: list[dict[str, Any]] = lib.resolve_many("subject", ["BRCA1"], db, qc=False)

    assert calls == [db]
    assert result == [{"subject": "brca1", "original_subject": "BRCA1", "subject_two": "brca1"}]


# ? sig Uses Exact "p_value" Column When Present Alongside Other P-Value Columns
def test_sig_prefers_exact_p_value_column() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"p_value": [0.01, 0.1], "adjusted_p_value": [0.5, 0.5]}).lazy()
    result: pl.DataFrame = lib.sig(lf).collect()
    assert list(result["statistical_significance_qualifier"]) == ["biolink:strongly_significant", "biolink:suggestive"]


# ? sig Falls Back To Non-Exact P-Value Column When No Exact Match
def test_sig_uses_non_exact_p_value_column() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"adjusted_p_value": [0.01, 0.1]}).lazy()
    result: pl.DataFrame = lib.sig(lf).collect()
    assert list(result["statistical_significance_qualifier"]) == ["biolink:strongly_significant", "biolink:suggestive"]


# ? sig Picks Closest Match When Multiple Non-Exact Columns Present
def test_sig_picks_closest_non_exact_match() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"log_p_value": [0.01], "adjusted_p_value_corrected": [0.5]}).lazy()
    result: pl.DataFrame = lib.sig(lf).collect()
    # "log_p_value" has higher fuzz.ratio to "p_value" than "adjusted_p_value_corrected"
    assert list(result["statistical_significance_qualifier"]) == ["biolink:strongly_significant"]


# ? sig Omits The Qualifier Column When No P-Value Column Exists (Biolink Class Rule)
# * Edges Are Retained; The Qualifier Is Simply Absent (Not Set)
def test_sig_omits_qualifier_with_no_p_value_column() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"gene": ["BRCA1", "TP53"]}).lazy()
    result: pl.DataFrame = lib.sig(lf).collect()
    assert "statistical_significance_qualifier" not in result.columns
    assert result.height == 2


# ? sig Emits Null (Not UNSURE) For Null P-Values; Edges Are Retained
def test_sig_marks_null_as_null_qualifier() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"p_value": [None, 0.01, 0.1]}).lazy()
    result: pl.DataFrame = lib.sig(lf).collect()
    assert list(result["statistical_significance_qualifier"]) == [None, "biolink:strongly_significant", "biolink:suggestive"]


# ? sig Maps The 0.05 < p <= 0.10 Band To biolink:suggestive
def test_sig_marks_suggestive_band() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"p_value": [0.01, 0.07, 0.1]}).lazy()
    result: pl.DataFrame = lib.sig(lf).collect()
    assert list(result["statistical_significance_qualifier"]) == ["biolink:strongly_significant", "biolink:suggestive", "biolink:suggestive"]


# ? sig Maps p <= 0.001 To biolink:very_strongly_significant (Boundary Included)
def test_sig_very_strongly_significant_band() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"p_value": [1e-8, 0.001, 0.002]}).lazy()
    result: pl.DataFrame = lib.sig(lf).collect()
    assert list(result["statistical_significance_qualifier"]) == [
        "biolink:very_strongly_significant",
        "biolink:very_strongly_significant",
        "biolink:strongly_significant",
    ]


# ? sig Maps The 0.01 < p <= 0.05 Band To biolink:significant (Boundary Included)
def test_sig_significant_band_boundary() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"p_value": [0.05, 0.06]}).lazy()
    result: pl.DataFrame = lib.sig(lf).collect()
    assert list(result["statistical_significance_qualifier"]) == ["biolink:significant", "biolink:suggestive"]


# ? sig Maps p > 0.10 To biolink:not_significant
def test_sig_not_significant_band() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"p_value": [0.11, 0.5]}).lazy()
    result: pl.DataFrame = lib.sig(lf).collect()
    assert list(result["statistical_significance_qualifier"]) == ["biolink:not_significant", "biolink:not_significant"]


# ? drop_not_significant Removes biolink:not_significant Rows While Keeping Null Qualifiers
def test_drop_not_significant_removes_band_keeps_nulls() -> None:
    lf: pl.LazyFrame = pl.DataFrame(
        {
            "subject": ["a", "b", "c", "d"],
            "statistical_significance_qualifier": ["biolink:significant", "biolink:not_significant", None, "biolink:suggestive"],
        }
    ).lazy()
    result: pl.DataFrame = drop_not_significant(lf).collect()
    assert list(result["subject"]) == ["a", "c", "d"]
    assert "biolink:not_significant" not in list(result["statistical_significance_qualifier"])


# ? drop_not_significant Is A No-Op When The Qualifier Column Is Absent
def test_drop_not_significant_noop_without_column() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"subject": ["a", "b"]}).lazy()
    result: pl.DataFrame = drop_not_significant(lf).collect()
    assert result.shape == (2, 1)
    assert list(result["subject"]) == ["a", "b"]


# ? drop_not_significant Keeps Every Band Except biolink:not_significant
def test_drop_not_significant_keeps_all_other_bands() -> None:
    bands: list[Optional[str]] = [
        "biolink:very_strongly_significant",
        "biolink:strongly_significant",
        "biolink:significant",
        "biolink:suggestive",
        "biolink:not_significant",
        None,
    ]
    lf: pl.LazyFrame = pl.DataFrame({"q": bands}).lazy()
    result: pl.DataFrame = drop_not_significant(lf, col="q").collect()
    assert list(result["q"]) == [b for b in bands if b != "biolink:not_significant"]


# ? numeric_columns Matches Any Column With P Value In The Name
def test_numeric_columns_matches_p_value_substring() -> None:
    names: list[str] = ["p_value", "adjusted_p_value", "log_p_value", "subject"]
    result: list[str] = numeric_columns(names)
    assert result == ["p_value", "adjusted_p_value", "log_p_value"]
    assert "subject" not in result


# ? numeric_columns Matches Exact Relationship Strength And Study Size Names
def test_numeric_columns_matches_exact_names() -> None:
    names: list[str] = ["relationship_strength", "sample_size", "supporting_study_size", "cohort"]
    result: list[str] = numeric_columns(names)
    assert "relationship_strength" in result
    assert "sample_size" in result
    assert "supporting_study_size" in result
    assert "cohort" not in result


# ? numeric_columns Is Case Insensitive On The P Value Substring
def test_numeric_columns_case_insensitive() -> None:
    names: list[str] = ["P_Value", "P_VALUE"]
    result: list[str] = numeric_columns(names)
    assert result == ["P_Value", "P_VALUE"]


# ? clean_numeric Coerces Numeric And Scientific Notation Strings To Float64
def test_clean_numeric_parses_numeric_and_scientific() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"p_value": ["1e-8", "0.05", "450"], "supporting_study_size": ["1200", "0.42", "-1.2"]}).lazy()
    result: pl.DataFrame = clean_numeric(lf).collect()
    assert result.schema["p_value"] == pl.Float64
    assert result.schema["supporting_study_size"] == pl.Float64
    assert result["p_value"].to_list() == [1e-8, 0.05, 450.0]
    assert result["supporting_study_size"].to_list() == [1200.0, 0.42, -1.2]


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


# ? format_numeric Renders Relationship Strength And Study Size In Decimal General Format
def test_format_numeric_decimal_general() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"relationship_strength": ["0.85", "0.42", "0.1234"], "supporting_study_size": ["450", "1200", "7"]}).lazy()
    result: pl.DataFrame = format_numeric(clean_numeric(lf)).collect()
    assert result["relationship_strength"].to_list() == ["0.85", "0.42", "0.1234"]
    assert result["supporting_study_size"].to_list() == ["450", "1200", "7"]


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
            "subject_pre_resolution": ["A", "B"],
            "object": ["HGNC:1", "HGNC:2"],
            "object_name": ["Xray", "Yankee"],
            "object_category": ["disease", "disease"],
            "object_taxon": [None, None],
            "object_source": [None, None],
            "object_source_version": [None, None],
            "object_pre_resolution": ["X", "Y"],
            "predicate": ["r", "r"],
            "upstream_resource_ids": [["infores:pubmed-central"], ["infores:pubmed-central"]],
            "knowledge_level": ["knowledge_assertion", "knowledge_assertion"],
            "agent_type": ["manual_agent", "manual_agent"],
            "resource_id": ["infores:smoke", "infores:smoke"],
            "p_value": ["1.0000e-08", "5.0000e-02"],
        }
    ).write_parquet(sub)
    lib.compile_graph([sub], "smoke", "1.0.0", "Smoke graph", None, "Custom UI explanation", None)
    edges: list[str] = (tmp_path / "smoke_1.0.0.edges.ndjson").read_text().strip().splitlines()
    nodes: list[str] = (tmp_path / "smoke_1.0.0.nodes.ndjson").read_text().strip().splitlines()
    rig: dict[str, Any] = lib.strip_nulls(from_yaml(tmp_path / "smoke_1.0.0.RIG.yaml"))
    assert len(edges) == 2
    assert all('"id"' in line for line in edges)
    flat: str = "\n".join(edges)
    assert '"p_value":"1.0000e-08"' in flat
    assert '"upstream_resource_ids":["infores:pubmed-central"]' in flat
    # ! Internal Pre-Resolution Snapshot Is Stripped From Final Edges
    assert "_pre_resolution" not in flat
    assert len(nodes) >= 1
    assert rig["name"] == "smoke v1.0.0"
    assert rig["source_info"]["infores_id"] == "infores:smoke"  # pyright: ignore
    assert rig["source_info"]["description"] == "Smoke graph"  # pyright: ignore
    assert rig["source_info"]["data_access_locations"] == ["smoke_1.0.0.nodes.ndjson", "smoke_1.0.0.edges.ndjson"]  # pyright: ignore
    assert rig["ingest_info"]["relevant_files"] == ["smoke_1.0.0.nodes.ndjson", "smoke_1.0.0.edges.ndjson"]  # pyright: ignore
    assert rig["provenance_info"]["contributions"] == ["Tablassert: KGX and RIG generation"]  # pyright: ignore
    edge_type: dict[str, Any] = rig["target_info"]["edge_type_info"][0]  # pyright: ignore
    assert edge_type["subject_categories"] == ["biolink:gene"]
    assert edge_type["predicates"] == ["r"]
    assert edge_type["object_categories"] == ["biolink:disease"]
    assert edge_type["primary_knowledge_sources"] == ["infores:pubmed-central", "infores:smoke"]
    assert edge_type["ui_explanation"] == "Custom UI explanation"
    node_types: list[dict[str, Any]] = rig["target_info"]["node_type_info"]  # pyright: ignore
    assert {x["node_category"] for x in node_types} == {"biolink:gene", "biolink:disease"}
    assert any(x["source_identifier_types"] == ["HGNC"] for x in node_types)


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
            "subject_pre_resolution": ["A"],
            "object": ["X"],
            "predicate": ["r"],
            "disease_context_qualifier": ["MONDO:0005148"],
            "disease_context_qualifier_pre_resolution": ["MONDO:0005148"],
            "publications": [["PMID:123"]],
        }
    ).write_parquet(sub)
    lib.compile_graph([sub], "qual", "1.0.0")
    edges: str = (tmp_path / "qual_1.0.0.edges.ndjson").read_text()
    nodes: str = (tmp_path / "qual_1.0.0.nodes.ndjson").read_text()
    # ! Qualifier And Publications Stay On Edges
    assert "MONDO:0005148" in edges
    assert "PMID:123" in edges
    # ! Internal Pre-Resolution Snapshots Are Stripped From Final Edges
    assert "_pre_resolution" not in edges
    # ! Neither Becomes A Node
    assert "MONDO:0005148" not in nodes
    assert "PMID:123" not in nodes


# ? dedup_stream Deduplicates And Strips Null Like Values From Node Streams
def test_dedup_stream_nodes(tmp_path: Path) -> None:
    p_in: Path = tmp_path / "nodes.ndjson.tmp"
    p_in.write_text('{"id":"A","drop":"NA"}\n{"id":"A","drop":"NA"}\n{"id":"B"}\n')

    lib.dedup_stream(p_in, is_edges=False)

    assert not p_in.exists()  # ? temp input is removed
    lines: list[str] = (tmp_path / "nodes.ndjson").read_text().strip().splitlines()
    assert lines == ['{"id":"A"}', '{"id":"B"}']


# ? dedup_stream Labels Edges With UUID Shaped ids And Deduplicates
def test_dedup_stream_edges(tmp_path: Path) -> None:
    import json

    p_in: Path = tmp_path / "edges.ndjson.tmp"
    p_in.write_text('{"subject":"A","object":"B","predicate":"r"}\n{"subject":"A","object":"B","predicate":"r"}\n')

    lib.dedup_stream(p_in, is_edges=True)

    assert not p_in.exists()
    lines: list[str] = (tmp_path / "edges.ndjson").read_text().strip().splitlines()
    assert len(lines) == 1  # ? duplicate edges collapse to one
    row: dict = json.loads(lines[0])
    assert "id" in row


# ? sig Computes Significance On A Cleaned Float64 P Value Column
def test_sig_works_on_cleaned_float64() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"p_value": ["1e-8", "0.5", "N/A"]}).lazy()
    cleaned: pl.LazyFrame = clean_numeric(lf)
    result: pl.DataFrame = lib.sig(cleaned).collect()
    assert result["statistical_significance_qualifier"].to_list() == ["biolink:very_strongly_significant", "biolink:not_significant", None]


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
# * Is A Categorical Flag Like sig()'s Own "statistical_significance_qualifier" Column, Not The Numeric Value
def test_pvalue_target_excludes_significance_flag_columns() -> None:
    names: list[str] = [
        "bonferroni significance",
        "nominal significance",
        "age significance flag",
        "significant",
        "statistical_significance_qualifier",
        "statistical significance qualifier",
    ]
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


# ? study_size_target Matches Common Study Size Spellings
def test_study_size_target_matches_common_spellings() -> None:
    names: list[str] = ["n", "N", "sample_size", "sample size", "sample-size", "sample.size", "samplesize", "study size", "cohort size"]
    for n in names:
        assert study_size_target(n) == "supporting_study_size", n


# ? study_size_target Matches Count Synonyms With Explicit Sample/Study Context
def test_study_size_target_matches_count_synonyms() -> None:
    names: list[str] = [
        "sample_count",
        "sample count",
        "number of samples",
        "num_samples",
        "n_samples",
        "samples_n",
        "total_n",
        "total n",
        "participant_count",
        "participants",
        "enrollment",
        "enrollment_count",
    ]
    for n in names:
        assert study_size_target(n) == "supporting_study_size", n


# ? study_size_target Excludes False Positives Without Explicit Study Size Meaning
def test_study_size_target_excludes_false_positives() -> None:
    names: list[str] = [
        "sample_id",
        "sample_name",
        "sample_type",
        "gene",
        "mean",
        "normalized_count",
        "nucleotide",
        "cohort",
        "population",
        "subject",
        "object",
        "predicate",
    ]
    for n in names:
        assert study_size_target(n) is None, n


# ? coerce_study_size_columns Renames Bare N To supporting_study_size
def test_coerce_study_size_columns_renames_n_column() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"n": [120, 450]}).lazy()
    result: pl.DataFrame = coerce_study_size_columns(lf).collect()
    assert "supporting_study_size" in result.columns
    assert "n" not in result.columns
    assert result["supporting_study_size"].to_list() == [120, 450]


# ? coerce_study_size_columns Renames sample_size To supporting_study_size
def test_coerce_study_size_columns_renames_sample_size_column() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"sample_size": [1200]}).lazy()
    result: pl.DataFrame = coerce_study_size_columns(lf).collect()
    assert result.columns == ["supporting_study_size"]
    assert result["supporting_study_size"].to_list() == [1200]


# ? coerce_study_size_columns Picks The Best Candidate And Leaves Others Untouched
def test_coerce_study_size_columns_picks_best_candidate() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"n": [9], "sample size": [1200], "participants": [1250]}).lazy()
    result: pl.DataFrame = coerce_study_size_columns(lf).collect()
    assert result["supporting_study_size"].to_list() == [1200]
    assert result["n"].to_list() == [9]
    assert result["participants"].to_list() == [1250]


# ? coerce_study_size_columns Is A Noop When Already Canonically Named
def test_coerce_study_size_columns_noop_when_already_canonical() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"supporting_study_size": [1200], "sample_size": [999]}).lazy()
    result: pl.DataFrame = coerce_study_size_columns(lf).collect()
    assert result.columns == ["supporting_study_size", "sample_size"]
    assert result["supporting_study_size"].to_list() == [1200]
    assert result["sample_size"].to_list() == [999]


# ? Tcode Coerces P Value Columns After Annotations And Before clean_numeric
# * So Downstream numeric_columns/sig/format_numeric See Already Canonical p_value/adjusted_p_value Names
def test_tcode_collect_coerces_pvalue_before_clean_numeric(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect(Path("/tmp/fullmap.redb"))  # pyright: ignore
    coerce_idx: int = next(i for i, op in enumerate(collected) if op[0].__name__ == "coerce_pvalue_columns")
    clean_idx: int = next(i for i, op in enumerate(collected) if op[0].__name__ == "clean_numeric")

    assert coerce_idx < clean_idx


# ? Tcode Coerces Study Size Columns After Annotations And Before clean_numeric
# * So Downstream numeric_columns/format_numeric See Already Canonical supporting_study_size Names
def test_tcode_collect_coerces_study_size_before_clean_numeric(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect(Path("/tmp/fullmap.redb"))  # pyright: ignore
    coerce_idx: int = next(i for i, op in enumerate(collected) if op[0].__name__ == "coerce_study_size_columns")
    clean_idx: int = next(i for i, op in enumerate(collected) if op[0].__name__ == "clean_numeric")

    assert coerce_idx < clean_idx


# ? Study Size Aliases Become Top-Level Supporting Study Size Fields Before Unknown Folding
def test_coerced_study_size_alias_survives_unknown_folding() -> None:
    lf: pl.LazyFrame = pl.DataFrame(
        {"subject": ["A"], "object": ["B"], "predicate": ["related_to"], "sample_size": [12000], "miscellaneous_notes": ["note"]}
    ).lazy()
    out: pl.DataFrame = fold_unknown_to_supporting_text(coerce_study_size_columns(lf)).collect()
    assert out["supporting_study_size"].to_list() == [12000]
    assert "sample_size" not in out.columns
    assert out["supporting_text"].to_list() == [["miscellaneous_notes: note"]]


# ? publications() Wraps A CURIE Literal As A Single Element list[str] Column
def test_publications_wraps_curie_as_list() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"subject": ["A"]}).lazy()
    out: pl.DataFrame = publications(lf, "PMID:42").collect()
    assert out.schema["publications"] == pl.List(pl.String)
    assert out["publications"].to_list() == [["PMID:42"]]


# ? idx Emits A 1-Based Column Named extracted_from_row_number
def test_idx_emits_one_based_extracted_from_row_number() -> None:
    lf: pl.LazyFrame = pl.LazyFrame({"a": ["x", "y", "z"]})
    out: pl.DataFrame = idx(lf).collect()
    assert "extracted_from_row_number" in out.columns
    assert out["extracted_from_row_number"].to_list() == [1, 2, 3]


# ? fold_unknown_to_supporting_text Is A Noop When Every Column Is On The Allow List
def test_fold_unknown_noop_when_all_allowed() -> None:
    lf: pl.LazyFrame = pl.DataFrame(
        {
            "subject": ["A"],
            "object": ["B"],
            "predicate": ["related_to"],
            "p_value": [0.01],
            "severity_qualifier": ["severe"],
            "publications": [["PMID:1"]],
        }
    ).lazy()
    out: pl.DataFrame = fold_unknown_to_supporting_text(lf).collect()
    # ! Nothing Folded, No supporting_text Column Created
    assert "supporting_text" not in out.columns
    assert set(out.columns) == {"subject", "object", "predicate", "p_value", "severity_qualifier", "publications"}


# ? fold_unknown_to_supporting_text Folds A Single Unknown Column As "col: value"
def test_fold_unknown_single_column() -> None:
    lf: pl.LazyFrame = pl.DataFrame(
        {"subject": ["A"], "object": ["B"], "predicate": ["related_to"], "miscellaneous_notes": ["see smith et al"]}
    ).lazy()
    out: pl.DataFrame = fold_unknown_to_supporting_text(lf).collect()
    assert "miscellaneous_notes" not in out.columns
    assert out.schema["supporting_text"] == pl.List(pl.String)
    assert out["supporting_text"].to_list() == [["miscellaneous_notes: see smith et al"]]


# ? fold_unknown_to_supporting_text Folds Multiple Columns In Deterministic Sorted Order
def test_fold_unknown_multiple_columns_sorted() -> None:
    lf: pl.LazyFrame = pl.DataFrame(
        {
            "subject": ["A"],
            "object": ["B"],
            "predicate": ["related_to"],
            # ! Deliberately Listed Out Of Sort Order To Verify Output Is Sorted By Column Name
            "extracted_from_row_number": ["7"],
            "sheet_name": ["Sheet1"],
            "miscellaneous_flag": ["yes"],
        }
    ).lazy()
    out: pl.DataFrame = fold_unknown_to_supporting_text(lf).collect()
    assert out["supporting_text"].to_list() == [["extracted_from_row_number: 7", "miscellaneous_flag: yes", "sheet_name: Sheet1"]]


# ? fold_unknown_to_supporting_text Skips Null And Empty String Values
def test_fold_unknown_skips_null_and_blank() -> None:
    lf: pl.LazyFrame = pl.DataFrame(
        {
            "subject": ["A", "B", "C"],
            "object": ["X", "Y", "Z"],
            "predicate": ["related_to", "related_to", "related_to"],
            "miscellaneous_notes": ["present", None, "   "],
        }
    ).lazy()
    out: pl.DataFrame = fold_unknown_to_supporting_text(lf).collect()
    rows: list[list[Optional[str]]] = out["supporting_text"].to_list()
    assert rows[0] == ["miscellaneous_notes: present"]
    # ! Null And Whitespace Only Both Yield An Empty List
    assert rows[1] == []
    assert rows[2] == []


# ? fold_unknown_to_supporting_text Appends To Existing list[str] supporting_text
def test_fold_unknown_appends_to_existing_list_supporting_text() -> None:
    lf: pl.LazyFrame = pl.DataFrame(
        {
            "subject": ["A"],
            "object": ["B"],
            "predicate": ["related_to"],
            "supporting_text": [["method: fisher exact"]],
            "miscellaneous_notes": ["see smith et al"],
        }
    ).lazy()
    out: pl.DataFrame = fold_unknown_to_supporting_text(lf).collect()
    assert out["supporting_text"].to_list() == [["method: fisher exact", "miscellaneous_notes: see smith et al"]]


# ? fold_unknown_to_supporting_text Coerces Scalar supporting_text To list[str] Then Appends
def test_fold_unknown_coerces_scalar_supporting_text() -> None:
    lf: pl.LazyFrame = pl.DataFrame(
        {"subject": ["A"], "object": ["B"], "predicate": ["related_to"], "supporting_text": ["plain summary"], "miscellaneous_notes": ["extra"]}
    ).lazy()
    out: pl.DataFrame = fold_unknown_to_supporting_text(lf).collect()
    assert out.schema["supporting_text"] == pl.List(pl.String)
    assert out["supporting_text"].to_list() == [["plain summary", "miscellaneous_notes: extra"]]


# ? fold_unknown_to_supporting_text Never Folds Known Qualifier Columns
def test_fold_unknown_preserves_qualifier_columns() -> None:
    lf: pl.LazyFrame = pl.DataFrame(
        {
            "subject": ["A"],
            "object": ["B"],
            "predicate": ["related_to"],
            "disease_context_qualifier": ["MONDO:0005148"],
            "severity_qualifier": ["severe"],
            "anatomical_context_qualifier": ["UBERON:0000061"],
        }
    ).lazy()
    out: pl.DataFrame = fold_unknown_to_supporting_text(lf).collect()
    # ! No supporting_text Column Materialized Because Nothing Was Foldable
    assert "supporting_text" not in out.columns
    assert "disease_context_qualifier" in out.columns
    assert "severity_qualifier" in out.columns
    assert "anatomical_context_qualifier" in out.columns


# ? PR #1770 Supporting Study Metadata Slots Survive As Top Level Edge Fields, Not Folded
def test_fold_unknown_preserves_supporting_study_metadata_slots() -> None:
    lf: pl.LazyFrame = pl.DataFrame(
        {
            "subject": ["A"],
            "object": ["B"],
            "predicate": ["related_to"],
            "has_supporting_studies": [["PMID:1"]],
            "supporting_study_method_types": [["case-control"]],
            "supporting_study_method_description": ["linear regression adjusted for age"],
            "supporting_study_size": [12000],
            "supporting_study_cohort": ["FINNGEN"],
            "supporting_study_date_range": ["2018-2023"],
            "supporting_study_context": ["European ancestry"],
            "miscellaneous_notes": ["see smith et al"],
        }
    ).lazy()
    out: pl.DataFrame = fold_unknown_to_supporting_text(lf).collect()
    # ! Only The Genuinely Unknown Column Is Folded Into supporting_text
    assert out["supporting_text"].to_list() == [["miscellaneous_notes: see smith et al"]]
    # ! Every PR #1770 Supporting Study Slot Survives As A Top Level Edge Field
    for col in (
        "has_supporting_studies",
        "supporting_study_method_types",
        "supporting_study_method_description",
        "supporting_study_size",
        "supporting_study_cohort",
        "supporting_study_date_range",
        "supporting_study_context",
    ):
        assert col in out.columns


# ? ALLOWED_EDGE_FIELDS Covers Intentional Tablassert Output Columns
def test_allowed_edge_fields_covers_tablassert_pipeline_columns() -> None:
    for col in ("publications", "upstream_resource_ids", "source_record_urls", "p_value", "supporting_text"):
        assert col in ALLOWED_EDGE_FIELDS


# ? PR #1770 Supporting Study Metadata Slots Are Recognized Biolist Edge Fields, Not Folded
def test_allowed_edge_fields_covers_supporting_study_metadata_slots() -> None:
    for col in (
        "has_supporting_studies",
        "supporting_study_method_types",
        "supporting_study_method_description",
        "supporting_study_size",
        "supporting_study_cohort",
        "supporting_study_date_range",
        "supporting_study_context",
    ):
        assert col in ALLOWED_EDGE_FIELDS


# ? compile_graph Folds Non Allow List Annotation Columns Into supporting_text On Edges
def test_compile_graph_folds_unknown_annotations_into_supporting_text(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    sub: Path = tmp_path / "sub.parquet"
    pl.DataFrame(
        {
            "subject": ["A"],
            "object": ["X"],
            "predicate": ["related_to"],
            "miscellaneous_notes": ["see smith et al"],
            "extracted_from_row_number": ["7"],
            "p_value": [0.01],
            "publications": [["PMID:1"]],
        }
    ).write_parquet(sub)
    lib.compile_graph([sub], "fold", "1.0.0")
    edges: str = (tmp_path / "fold_1.0.0.edges.ndjson").read_text()
    # ! Folded Columns No Longer Appear As Top Level JSON Keys On The Edge Object
    assert '"miscellaneous_notes":' not in edges
    assert '"extracted_from_row_number":' not in edges
    # ! But Their Values Survive Inside supporting_text
    assert "miscellaneous_notes: see smith et al" in edges
    assert "extracted_from_row_number: 7" in edges
    # ! Real Biolist Fields Survive As Top Level Fields
    assert '"p_value":0.01' in edges or '"p_value": 0.01' in edges
    assert "PMID:1" in edges
