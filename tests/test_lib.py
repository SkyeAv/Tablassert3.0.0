from __future__ import annotations

import json
from itertools import chain
from pathlib import Path
from typing import Any, Self, cast

import polars as pl

import tablassert.cli as cli
import tablassert.lib as lib
from tablassert import rs
from tablassert.biolink import ALLOWED_EDGE_FIELDS, EFFECT_TYPE_VALUES, UNSATISFIABLE_EDGE_FIELDS, Categories, validate_kgx
from tablassert.coerce import _EFFECT_TYPE_ALIASES, _map_effect_type_value
from tablassert.enums import Repositories
from tablassert.fullmap import ResolveSpec
from tablassert.ingests import from_yaml
from tablassert.lib import (
    HEAD_ROWS,
    Tcode,
    clean_numeric,
    coerce_effect_size_columns,
    coerce_effect_type_columns,
    coerce_pvalue_columns,
    coerce_study_size_columns,
    drop_not_significant,
    edge_category,
    edge_tables,
    effect_size_target,
    effect_type_target,
    fold_unknown_to_supporting_text,
    format_numeric,
    head,
    idx,
    idxname,
    infores,
    numeric_columns,
    parse_edge_name,
    publications,
    pvalue_target,
    retrieval_sources,
    strip_nulls,
    study_size_target,
)


def fake_fullmap_row(term: str, curie: str, name: str, category: str, taxon: int = 0) -> dict[str, object]:
    """Build a fake fullmap lookup row using the Rust extension return schema."""
    return {
        "term": term,
        "CURIE": curie,
        "PREFERRED_NAME": name,
        "CATEGORY_NAME": category,
        "TAXON_ID": taxon,
        "SOURCE_NAME": "TEST",
        "SOURCE_VERSION": "1",
    }


def install_fake_fullmap(monkeypatch: Any, rows: dict[str, list[dict[str, object]]]) -> list[list[str]]:
    """Monkeypatch fullmap lookup and return captured term batches."""
    calls: list[list[str]] = []

    def fake_lookup(db: Path, terms: list[str], threads: int | None = None, return_format: str = "rows") -> list[dict[str, object]]:
        del db, threads, return_format
        calls.append(terms)
        return [row for term in terms for row in rows.get(term, [])]

    monkeypatch.setattr(rs, "lookup_fullmap_terms", fake_lookup)
    return calls


def write_text_section(tmp_path: Path, name: str, section: dict[str, object], rows: list[str]) -> tuple[Path, Path]:
    """Write a source TSV and matching table YAML for a test section."""
    from tablassert.ingests import to_yaml

    table_path: Path = tmp_path / f"{name}.yaml"
    source_path: Path = tmp_path / f"{name}.tsv"
    source_path.write_text("\n".join(rows) + "\n")
    source_config: object = section.get("source", {})
    source_overrides: dict[str, object] = source_config if isinstance(source_config, dict) else {}
    section["source"] = {"url": [f"https://example.com/{name}.tsv"], "local": str(source_path), "kind": "text", "delimiter": "\t", **source_overrides}
    to_yaml(table_path, section)
    return table_path, source_path


class DummyProgress:
    """Minimal progress reporter for build_pipeline smoke tests."""

    def __init__(self) -> None:
        self.stages: list[str] = []
        self.sections: list[str] = []
        self.sub_steps: list[str] = []

    def stage(self, name: str) -> None:
        self.stages.append(name)

    def section_loop(self, n: int, label: str) -> tuple[Any, Any, Any]:
        del n, label

        def start(name: str) -> None:
            self.sections.append(name)

        def advance() -> None:
            return None

        def sub_step(name: str) -> None:
            self.sub_steps.append(name)

        return start, advance, sub_step


class SyncPool:
    """Synchronous drop-in replacement for multiprocessing.Pool in tests."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback

    def map(self, fn: Any, items: list[Any]) -> list[Any]:
        return [fn(item) for item in items]

    def starmap(self, fn: Any, items: object) -> list[Any]:
        return [fn(*item) for item in items]  # pyright: ignore

    def imap_unordered(self, fn: Any, items: object) -> list[Any]:
        # Synchronous stand-in: yields fn(item) for each input (consumed by a for-loop, so a list suffices).
        return [fn(item) for item in items]  # pyright: ignore


def test_idxname_single_letter() -> None:
    """idxname converts single letter columns."""
    assert idxname("A") == "column_1"
    assert idxname("B") == "column_2"
    assert idxname("Z") == "column_26"


def test_idxname_double_letter() -> None:
    """idxname converts double letter columns."""
    assert idxname("AA") == "column_27"
    assert idxname("AB") == "column_28"
    assert idxname("AZ") == "column_52"


def test_idxname_triple_letter() -> None:
    """idxname converts triple letter columns."""
    assert idxname("AAA") == "column_703"


def test_idxname_format() -> None:
    """idxname returns column prefixed string."""
    result: str = idxname("C")
    assert result.startswith("column_")


def test_strip_nulls_removes_empty_string() -> None:
    """strip_nulls removes null like values."""
    r: dict[str, Any] = {"a": "hello", "b": ""}
    result: dict = strip_nulls(r)
    assert "a" in result
    assert "b" not in result


def test_strip_nulls_removes_null_variants() -> None:
    """strip_nulls removes na nan null none."""
    r: dict[str, Any] = {"a": "na", "b": "nan", "c": "null", "d": "none"}
    result: dict = strip_nulls(r)
    assert len(result) == 0


def test_strip_nulls_case_insensitive() -> None:
    """strip_nulls case insensitive."""
    r: dict[str, Any] = {"a": "NA", "b": "NaN", "c": "NULL", "d": "None"}
    result: dict = strip_nulls(r)
    assert len(result) == 0


def test_strip_nulls_preserves_valid() -> None:
    """strip_nulls preserves valid values."""
    r: dict[str, Any] = {"name": "BRCA1", "score": 0.05, "active": True}
    result: dict = strip_nulls(r)
    assert result["name"] == "BRCA1"
    assert result["score"] == 0.05
    assert result["active"] is True


def test_strip_nulls_nested_dict() -> None:
    """strip_nulls handles nested dicts."""
    r: dict[str, Any] = {"outer": {"inner": "na", "keep": "yes"}}
    result: dict = strip_nulls(r)
    assert "keep" in result["outer"]
    assert "inner" not in result["outer"]


def test_strip_nulls_list_of_dicts() -> None:
    """strip_nulls handles lists of dicts."""
    r: dict[str, Any] = {"items": [{"a": "keep", "b": ""}, {"a": "also", "c": "null"}]}
    result: dict = strip_nulls(r)
    assert result["items"][0] == {"a": "keep"}
    assert result["items"][1] == {"a": "also"}


def test_strip_nulls_empty_dict() -> None:
    """strip_nulls handles empty dict."""
    r: dict[str, Any] = {}
    result: dict = strip_nulls(r)
    assert result == {}


def test_strip_nulls_whitespace() -> None:
    """strip_nulls strips whitespace before check."""
    r: dict[str, Any] = {"a": "  ", "b": " na "}
    result: dict = strip_nulls(r)
    assert len(result) == 0


def test_tcode_model_allows_unresolved_value_encoding(fixtures_path: Path) -> None:
    """tcode allows unresolved value encodings during validation."""
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    data["statement"]["subject"] = {"method": "value", "encoding": "Incertae Sedis XI"}

    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store}
    )

    assert tcode_model.statement.subject.method == "value"
    assert tcode_model.statement.subject.encoding == "Incertae Sedis XI"


def test_tcode_collect_skips_qc_by_default(fixtures_path: Path) -> None:
    """tcode collect enables QC logging by default."""
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect(Path("/tmp/fullmap.redb"))  # pyright: ignore
    qc_ops: list[tuple[Any, tuple[Any]]] = [op for op in collected if op[0].__name__ == "fullmap_audit"]

    assert qc_ops == []


def test_tcode_collect_enables_qc_logging(fixtures_path: Path) -> None:
    """tcode collect enables QC logging when graph QC is enabled."""
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


# tcode collect orders drop_not_significant before resolve_batch in release mode
# rows that will be dropped for insignificance must never reach the expensive fullmap resolve step
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


def test_tcode_collect_omits_drop_not_significant_without_release(fixtures_path: Path) -> None:
    """tcode collect omits drop_not_significant without release but keeps sig before resolve_batch."""
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash_norelease.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect(Path("/tmp/fullmap.redb"))  # pyright: ignore
    names: list[str] = [op[0].__name__ for op in collected]

    assert "drop_not_significant" not in names
    assert names.index("sig") < names.index("resolve_batch")


def test_tcode_collect_emits_single_resolve_batch_for_all_node_columns(fixtures_path: Path) -> None:
    """tcode collect emits exactly one resolve_batch op covering subject/object/qualifiers."""
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash_batch.parquet")
    data["statement"]["subject"]["taxon"] = 9606
    data["statement"]["object"]["prioritize"] = ["Gene"]
    data["statement"]["qualifiers"] = [
        {"qualifier": "disease_context_qualifier", "method": "value", "encoding": "MONDO:0000001", "avoid": ["Gene"]},
        {"qualifier": "anatomical_context_qualifier", "method": "value", "encoding": "UBERON:0000061"},
    ]

    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect(Path("/tmp/fullmap.redb"))  # pyright: ignore
    batch_ops: list[tuple[Any, tuple[Any]]] = [op for op in collected if op[0].__name__ == "resolve_batch"]

    assert len(batch_ops) == 1
    specs: list[ResolveSpec] = batch_ops[0][1][0]
    assert [spec.col for spec in specs] == ["subject", "object", "disease_context_qualifier", "anatomical_context_qualifier"]
    assert specs[0].taxon == "9606"
    assert specs[1].prioritize == [Categories.GENE]
    assert specs[2].avoid == [Categories.GENE]


def test_tcode_collect_audits_follow_single_resolve_batch_with_qualifiers(fixtures_path: Path) -> None:
    """tcode collect runs every node column's QC audit after the single resolve_batch op."""
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash_batch_qc.parquet")
    data["statement"]["qualifiers"] = [{"qualifier": "anatomical_context_qualifier", "method": "value", "encoding": "UBERON:0000061"}]

    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store, "qc": True}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect(Path("/tmp/fullmap.redb"))  # pyright: ignore
    batch_idx: int = next(i for i, op in enumerate(collected) if op[0].__name__ == "resolve_batch")
    audit_ops: list[tuple[int, tuple[Any, tuple[Any]]]] = [(i, op) for i, op in enumerate(collected) if op[0].__name__ == "fullmap_audit"]

    assert [op[1][0] for _, op in audit_ops] == ["subject", "object", "anatomical_context_qualifier"]
    assert all(i > batch_idx for i, _ in audit_ops)


def test_tcode_collect_edge_ops_follow_resolve_batch(fixtures_path: Path) -> None:
    """tcode collect runs predicate/edge_category after the single resolve_batch op."""
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


def test_tcode_collect_passes_local_path_to_csv_reader(fixtures_path: Path) -> None:
    """tcode collect passes the local source path through to the csv reader."""
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect(Path("/tmp/fullmap.redb"))  # pyright: ignore
    csv_ops: list[tuple[Any, tuple[Any]]] = [op for op in collected if op[0].__name__ == "csv"]

    assert csv_ops[0][1] == (tcode_model.source.local, tcode_model.source.delimiter)  # pyright: ignore


def test_head_caps_at_n_rows() -> None:
    """head keeps min(n, height) rows, never more than the source height."""
    small: pl.DataFrame = head(pl.LazyFrame({"a": [1, 2, 3]}), n=HEAD_ROWS).collect()
    big: pl.DataFrame = head(pl.LazyFrame({"a": list(range(10))}), n=HEAD_ROWS).collect()

    assert small.height == 3
    assert big.height == 5


def test_head_samples_subset_of_source() -> None:
    """head returns a random subset of the source rows, never inventing or exceeding them."""
    source: list[int] = list(range(10))
    sampled: pl.DataFrame = head(pl.LazyFrame({"a": source}), n=HEAD_ROWS).collect()

    assert sampled.height == HEAD_ROWS
    assert set(sampled["a"].to_list()) <= set(source)

    # A frame shorter than n is returned whole (never sampled beyond its height).
    short: pl.DataFrame = head(pl.LazyFrame({"a": [7, 8, 9]}), n=HEAD_ROWS).collect()
    assert short.height == 3
    assert set(short["a"].to_list()) == {7, 8, 9}


def test_tcode_collect_omits_head_by_default(fixtures_path: Path) -> None:
    """tcode collect omits the head op unless head mode is requested."""
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect(Path("/tmp/fullmap.redb"))  # pyright: ignore
    names: list[str] = [op[0].__name__ for op in collected]

    assert "head" not in names


def test_tcode_collect_inserts_head_after_row_filters_before_resolve_when_head(fixtures_path: Path) -> None:
    """head op sits after idx/row-filters and before resolve_batch so resolve only sees HEAD_ROWS rows."""
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash_head.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store, "head": True}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect(Path("/tmp/fullmap.redb"))  # pyright: ignore
    names: list[str] = [op[0].__name__ for op in collected]

    head_idx: int = names.index("head")
    resolve_idx: int = names.index("resolve_batch")
    idx_idx: int = names.index("idx")
    annotations_idx: int = next(i for i, op in enumerate(collected) if op[0].__name__ == "column" and op[1][0].startswith("original_"))

    assert idx_idx < head_idx < resolve_idx
    assert head_idx < annotations_idx
    assert collected[head_idx][1] == (HEAD_ROWS,)


def test_publication_curie_pmc() -> None:
    """publication_curie uses PMCID namespace for PubMed central."""
    assert lib.publication_curie("PMC", "PMC1234567") == "PMCID:PMC1234567"


def test_publication_curie_pubmed() -> None:
    """publication_curie uses repo namespace for non PMC repositories."""
    assert lib.publication_curie("PMID", "11708054") == "PMID:11708054"


def test_infores_screaming_snake() -> None:
    """infores lower kebab cases a screaming snake graph name with infores prefix."""
    assert infores("MULTIOMICS_KG") == "infores:multiomics-kg"


def test_infores_single_and_tutorial() -> None:
    """infores handles single word and tutorial graph names."""
    assert infores("TUTORIAL_KG") == "infores:tutorial-kg"
    assert infores("CHEMBL") == "infores:chembl"


def test_upstream_resource_ids_pmc() -> None:
    """upstream_resource_ids uses PubMed central InfoRes for PMC repositories."""
    assert lib.upstream_resource_ids(Repositories.PUBMED_CENTRAL) == ["infores:pubmed-central"]


def test_upstream_resource_ids_pubmed() -> None:
    """upstream_resource_ids uses PubMed InfoRes for PMID repositories."""
    assert lib.upstream_resource_ids(Repositories.PUBMED) == ["infores:pubmed"]


def test_tcode_collect_nests_upstream_resource_ids_in_sources(fixtures_path: Path) -> None:
    """Upstream resource IDs reach output inside ``sources``, never flat on the edge.

    ``upstream_resource_ids`` has ``domain: retrieval source`` in Biolink, so a
    top-level column would fail validation with ``extra_forbidden``.
    """
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store, "name": "GRAPH_KG"}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect(Path("/tmp/fullmap.redb"))  # pyright: ignore
    assert [op for op in collected if len(op[1]) > 0 and op[1][0] == "upstream_resource_ids"] == []

    source_ops: list[tuple[Any, tuple[Any]]] = [op for op in collected if op[0] is retrieval_sources]
    assert len(source_ops) == 1
    out: pl.DataFrame = source_ops[0][0](pl.LazyFrame({"subject": ["A"]}), *source_ops[0][1]).collect()
    sources: list[dict[str, Any]] = out["sources"].to_list()[0]
    primary: dict[str, Any] = next(s for s in sources if s["resource_role"] == "primary_knowledge_source")
    assert primary["upstream_resource_ids"] == ["infores:pubmed-central"]
    assert {s["resource_id"] for s in sources if s["resource_role"] == "supporting_data_source"} == {"infores:pubmed-central"}


def test_normalize_category_list_with_biolink_prefix() -> None:
    """normalize wraps category in a list and ensures biolink: prefix."""
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


def test_normalize_category_null_stays_null() -> None:
    """normalize keeps null categories null for strip_nulls removal."""
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


def test_derive_species_context_coalesces_taxon() -> None:
    """derive_species_context uses subject taxon first, then object taxon."""
    lf: pl.LazyFrame = pl.DataFrame(
        {"subject_taxon": ["NCBITaxon:9606", None, None], "object_taxon": ["NCBITaxon:10090", "NCBITaxon:9606", None]}
    ).lazy()

    result: list[Any] = lib.derive_species_context(lf).collect()["species_context_qualifier"].to_list()

    assert result == ["NCBITaxon:9606", "NCBITaxon:9606", None]


def test_tcode_collect_emits_primary_knowledge_source_when_named(fixtures_path: Path) -> None:
    """tcode collect emits primary_knowledge_source op when graph name is provided."""
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store, "name": "MULTIOMICS_KG"}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect(Path("/tmp/fullmap.redb"))  # pyright: ignore
    pks_ops: list[tuple[Any, tuple[Any]]] = [
        op for op in collected if op[0].__name__ == "value" and len(op[1]) > 0 and op[1][0] == "primary_knowledge_source"
    ]

    assert len(pks_ops) == 1
    assert pks_ops[0][1] == ("primary_knowledge_source", "infores:multiomics-kg")


def test_tcode_collect_omits_primary_knowledge_source_when_unnamed(fixtures_path: Path) -> None:
    """tcode collect omits primary_knowledge_source op when graph name is absent (validate path)."""
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect(Path("/tmp/fullmap.redb"))  # pyright: ignore
    pks_ops: list[tuple[Any, tuple[Any]]] = [
        op for op in collected if op[0].__name__ == "value" and len(op[1]) > 0 and op[1][0] == "primary_knowledge_source"
    ]

    assert pks_ops == []


def test_tcode_collect_manual_provenance_overrides_auto_sources(fixtures_path: Path) -> None:
    """manual provenance overrides upstream/publication/KL/AT; PKS still derives from graph infores."""
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    data["provenance"] = {
        "override": {
            "upstream_resource_ids": ["infores:upstream-source"],
            "publications": ["PMCID:PMC9999999"],
            "knowledge_level": "knowledge_assertion",
            "agent_type": "manual_agent",
        }
    }
    store: Path = Path("/tmp/sectionhash.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store, "name": "GRAPH_KG", "infores": "infores:graph-source"}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect(Path("/tmp/fullmap.redb"))  # pyright: ignore
    values: dict[str, object] = {str(op[1][0]): op[1][1] for op in collected if op[0].__name__ == "value" and len(op[1]) >= 2}
    pub_ops = [op for op in collected if op[0] is publications]

    assert values["primary_knowledge_source"] == "infores:graph-source"
    # Manual upstream infores reach output nested in `sources`, not flat on the edge.
    assert "upstream_resource_ids" not in values
    source_args: tuple[Any, ...] = next(op[1] for op in collected if op[0] is retrieval_sources)
    assert source_args[1] == ["infores:upstream-source"]
    assert values["knowledge_level"] == "knowledge_assertion"
    assert values["agent_type"] == "manual_agent"
    assert pub_ops[0][1] == (["PMCID:PMC9999999"],)


def test_tcode_collect_uses_graph_infores_when_no_section_override(fixtures_path: Path) -> None:
    """graph infores overrides the derived infores(name) primary knowledge source."""
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store, "name": "GRAPH_KG", "infores": "infores:custom-graph"}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect(Path("/tmp/fullmap.redb"))  # pyright: ignore
    pks_ops = [op for op in collected if op[0].__name__ == "value" and len(op[1]) > 0 and op[1][0] == "primary_knowledge_source"]

    assert pks_ops[0][1] == ("primary_knowledge_source", "infores:custom-graph")


def test_tcode_collect_nests_source_record_urls_in_sources(fixtures_path: Path) -> None:
    """Source record URLs hang off the primary ``RetrievalSource``, not the edge."""
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store, "name": "GRAPH_KG"}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect(Path("/tmp/fullmap.redb"))  # pyright: ignore
    source_ops: list[tuple[Any, tuple[Any]]] = [op for op in collected if op[0] is retrieval_sources]
    url_ops: list[tuple[Any, tuple[Any]]] = [op for op in collected if op[0].__name__ == "value" and len(op[1]) > 0 and op[1][0] == "url"]
    result: pl.DataFrame = source_ops[0][0](pl.LazyFrame({"subject": ["A"]}), *source_ops[0][1]).collect()

    assert len(source_ops) == 1
    assert url_ops == []
    assert "source_record_urls" not in result.columns
    primary: dict[str, Any] = next(s for s in result["sources"].to_list()[0] if s["resource_role"] == "primary_knowledge_source")
    assert primary["source_record_urls"] == ["https://example.com/test.tsv"]


def test_tcode_original_value_before_regex_for_columns(fixtures_path: Path) -> None:
    """tcode captures original value before regex for column encoded nodes."""
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


def test_tcode_original_value_present_for_value_encoding(fixtures_path: Path) -> None:
    """tcode emits original value for value encoded nodes."""
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect(Path("/tmp/fullmap.redb"))  # pyright: ignore
    targets: list[str] = [op[1][0] for op in collected if op[0].__name__ == "column" and len(op[1]) > 1]
    assert "original_subject" in targets
    assert "original_object" in targets


def test_resolve_many_skips_qc(monkeypatch: Any, tmp_path: Path) -> None:
    """resolve_many skips QC when disabled."""
    calls: list[tuple[Any, ...]] = []

    def fake_resolve(lf: pl.LazyFrame, col: str, db: Path, **kwargs: Any) -> pl.LazyFrame:
        calls.append(("resolve", col, db, kwargs))
        return lf

    def fake_qc(lf: pl.LazyFrame, col: str, section_hash: str, config_file: str, out: str = "passed", log: bool = True) -> pl.LazyFrame:
        calls.append(("qc", col, section_hash, config_file, out, log))
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


def test_resolve_many_runs_qc(monkeypatch: Any, tmp_path: Path) -> None:
    """resolve_many runs QC with logging when enabled."""
    calls: list[tuple[Any, ...]] = []

    def fake_resolve(lf: pl.LazyFrame, col: str, db: Path, **kwargs: Any) -> pl.LazyFrame:
        calls.append(("resolve", col, db, kwargs))
        return lf

    def fake_qc(lf: pl.LazyFrame, col: str, section_hash: str, config_file: str, out: str = "passed", log: bool = True) -> pl.LazyFrame:
        calls.append(("qc", col, section_hash, config_file, out, log))
        return lf.with_columns(pl.lit("YES").alias(out))

    monkeypatch.setattr(lib, "resolve", fake_resolve)
    monkeypatch.setattr(lib, "fullmap_audit", fake_qc)

    result: list[dict[str, Any]] = lib.resolve_many("subject", ["BRCA1"], tmp_path, qc=True)

    assert result == [{"subject": "brca1", "original_subject": "BRCA1", "subject_two": "brca1", "passed": "YES"}]
    assert ("qc", "subject", "", "", "passed", True) in calls


def test_resolve_many_accepts_direct_fullmap_file(monkeypatch: Any, tmp_path: Path) -> None:
    """resolve_many accepts a direct fullmap redb file path."""
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


def test_sig_prefers_exact_p_value_column() -> None:
    """sig uses exact "p_value" column when present alongside other P-Value columns."""
    lf: pl.LazyFrame = pl.DataFrame({"p_value": [0.01, 0.1], "adjusted_p_value": [0.5, 0.5]}).lazy()
    result: pl.DataFrame = lib.sig(lf).collect()
    assert list(result["statistical_significance_qualifier"]) == ["biolink:strongly_significant", "biolink:suggestive"]


def test_sig_uses_non_exact_p_value_column() -> None:
    """sig falls back to non-exact P-Value column when no exact match."""
    lf: pl.LazyFrame = pl.DataFrame({"adjusted_p_value": [0.01, 0.1]}).lazy()
    result: pl.DataFrame = lib.sig(lf).collect()
    assert list(result["statistical_significance_qualifier"]) == ["biolink:strongly_significant", "biolink:suggestive"]


def test_sig_picks_closest_non_exact_match() -> None:
    """sig prefers a raw p-value bucket over an adjusted one when both are non-exact."""
    lf: pl.LazyFrame = pl.DataFrame({"log_p_value": [0.01], "adjusted_p_value_corrected": [0.5]}).lazy()
    result: pl.DataFrame = lib.sig(lf).collect()
    # "log_p_value" -> raw p_value bucket; "adjusted_p_value_corrected" -> adjusted bucket.
    # The raw bucket is preferred, so 0.01 -> strongly_significant (not 0.5 -> not_significant).
    assert list(result["statistical_significance_qualifier"]) == ["biolink:strongly_significant"]


# sig omits the qualifier column when no P-Value column exists (biolink class rule)
# edges are retained; the qualifier is simply absent (not set)
def test_sig_omits_qualifier_with_no_p_value_column() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"gene": ["BRCA1", "TP53"]}).lazy()
    result: pl.DataFrame = lib.sig(lf).collect()
    assert "statistical_significance_qualifier" not in result.columns
    assert result.height == 2


def test_sig_marks_null_as_null_qualifier() -> None:
    """sig emits null (not UNSURE) for null P-Values; edges are retained."""
    lf: pl.LazyFrame = pl.DataFrame({"p_value": [None, 0.01, 0.1]}).lazy()
    result: pl.DataFrame = lib.sig(lf).collect()
    assert list(result["statistical_significance_qualifier"]) == [None, "biolink:strongly_significant", "biolink:suggestive"]


def test_sig_marks_suggestive_band() -> None:
    """sig maps the 0.05 < p <= 0.10 band to biolink:suggestive."""
    lf: pl.LazyFrame = pl.DataFrame({"p_value": [0.01, 0.07, 0.1]}).lazy()
    result: pl.DataFrame = lib.sig(lf).collect()
    assert list(result["statistical_significance_qualifier"]) == ["biolink:strongly_significant", "biolink:suggestive", "biolink:suggestive"]


def test_sig_very_strongly_significant_band() -> None:
    """sig maps p <= 0.001 to biolink:very_strongly_significant (boundary included)."""
    lf: pl.LazyFrame = pl.DataFrame({"p_value": [1e-8, 0.001, 0.002]}).lazy()
    result: pl.DataFrame = lib.sig(lf).collect()
    assert list(result["statistical_significance_qualifier"]) == [
        "biolink:very_strongly_significant",
        "biolink:very_strongly_significant",
        "biolink:strongly_significant",
    ]


def test_sig_significant_band_boundary() -> None:
    """sig maps the 0.01 < p <= 0.05 band to biolink:significant (boundary included)."""
    lf: pl.LazyFrame = pl.DataFrame({"p_value": [0.05, 0.06]}).lazy()
    result: pl.DataFrame = lib.sig(lf).collect()
    assert list(result["statistical_significance_qualifier"]) == ["biolink:significant", "biolink:suggestive"]


def test_sig_not_significant_band() -> None:
    """sig maps p > 0.10 to biolink:not_significant."""
    lf: pl.LazyFrame = pl.DataFrame({"p_value": [0.11, 0.5]}).lazy()
    result: pl.DataFrame = lib.sig(lf).collect()
    assert list(result["statistical_significance_qualifier"]) == ["biolink:not_significant", "biolink:not_significant"]


def test_sig_prefers_raw_p_value_over_adjusted_bucket() -> None:
    """sig derives the qualifier from a raw p-value column, not a co-present adjusted one.

    ``"P"`` classifies as the raw ``p_value`` bucket (bare-P token) and ``"FDR"`` as the
    ``adjusted_p_value`` bucket. The qualifier must follow the raw column: 0.01 maps to
    ``strongly_significant``, whereas the adjusted 0.001 would wrongly yield
    ``very_strongly_significant``.
    """
    lf: pl.LazyFrame = pl.DataFrame({"P": [0.01], "FDR": [0.001]}).lazy()
    result: pl.DataFrame = lib.sig(lf).collect()
    assert list(result["statistical_significance_qualifier"]) == ["biolink:strongly_significant"]


def test_sig_canonical_column_wins_over_higher_scoring_alias() -> None:
    """An existing canonical ``p_value`` column wins over a higher-scoring spaced alias.

    Both ``"p_value"`` and ``"p vals"`` land in the raw bucket; the canonical column is
    chosen directly (no fuzzy tiebreak), so banding follows ``p_value``=0.05
    (``significant``) rather than ``p vals``=0.001 (``very_strongly_significant``).
    """
    lf: pl.LazyFrame = pl.DataFrame({"p_value": [0.05], "p vals": [0.001]}).lazy()
    result: pl.DataFrame = lib.sig(lf).collect()
    assert list(result["statistical_significance_qualifier"]) == ["biolink:significant"]


def test_sig_excludes_substring_only_non_pvalue_column() -> None:
    """sig uses pvalue_target, not a naive substring, so a look-alike column is ignored.

    ``"xp_value_x"`` contains the literal ``p_value`` substring (the old selector would
    grab it) but ``pvalue_target`` rejects it: the ``p`` is glued to an alphanumeric on
    both sides, so neither the value token nor the bare-P token matches. With no real
    p-value column the qualifier is omitted (Biolink class rule).
    """
    lf: pl.LazyFrame = pl.DataFrame({"xp_value_x": [0.01], "gene": ["BRCA1"]}).lazy()
    result: pl.DataFrame = lib.sig(lf).collect()
    assert "statistical_significance_qualifier" not in result.columns
    assert result.height == 1


def test_drop_not_significant_removes_band_keeps_nulls() -> None:
    """drop_not_significant removes biolink:not_significant rows while keeping null qualifiers."""
    lf: pl.LazyFrame = pl.DataFrame(
        {
            "subject": ["a", "b", "c", "d"],
            "statistical_significance_qualifier": ["biolink:significant", "biolink:not_significant", None, "biolink:suggestive"],
        }
    ).lazy()
    result: pl.DataFrame = drop_not_significant(lf).collect()
    assert list(result["subject"]) == ["a", "c", "d"]
    assert "biolink:not_significant" not in list(result["statistical_significance_qualifier"])


def test_drop_not_significant_noop_without_column() -> None:
    """drop_not_significant is a no-op when the qualifier column is absent."""
    lf: pl.LazyFrame = pl.DataFrame({"subject": ["a", "b"]}).lazy()
    result: pl.DataFrame = drop_not_significant(lf).collect()
    assert result.shape == (2, 1)
    assert list(result["subject"]) == ["a", "b"]


def test_drop_not_significant_keeps_all_other_bands() -> None:
    """drop_not_significant keeps every band except biolink:not_significant."""
    bands: list[str | None] = [
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


def test_numeric_columns_matches_p_value_substring() -> None:
    """numeric_columns matches any column with P value in the name."""
    names: list[str] = ["p_value", "adjusted_p_value", "log_p_value", "subject"]
    result: list[str] = numeric_columns(names)
    assert result == ["p_value", "adjusted_p_value", "log_p_value"]
    assert "subject" not in result


def test_numeric_columns_matches_exact_names() -> None:
    """numeric_columns matches exact effect size and study size names."""
    names: list[str] = ["effect_size", "supporting_study_size", "cohort", "sample_size", "relationship_strength"]
    result: list[str] = numeric_columns(names)
    assert "effect_size" in result
    assert "supporting_study_size" in result
    assert "cohort" not in result
    # Old names are superseded: coercion renames them before clean_numeric/format_numeric run.
    assert "sample_size" not in result
    assert "relationship_strength" not in result


def test_numeric_columns_case_insensitive() -> None:
    """numeric_columns is case insensitive on the P value substring."""
    names: list[str] = ["P_Value", "P_VALUE"]
    result: list[str] = numeric_columns(names)
    assert result == ["P_Value", "P_VALUE"]


def test_clean_numeric_parses_numeric_and_scientific() -> None:
    """clean_numeric coerces numeric and scientific notation strings to Float64."""
    lf: pl.LazyFrame = pl.DataFrame({"p_value": ["1e-8", "0.05", "450"], "supporting_study_size": ["1200", "0.42", "-1.2"]}).lazy()
    result: pl.DataFrame = clean_numeric(lf).collect()
    assert result.schema["p_value"] == pl.Float64
    assert result.schema["supporting_study_size"] == pl.Float64
    assert result["p_value"].to_list() == [1e-8, 0.05, 450.0]
    assert result["supporting_study_size"].to_list() == [1200.0, 0.42, -1.2]


def test_clean_numeric_nulls_non_numeric() -> None:
    """clean_numeric drops non numeric entries to null."""
    lf: pl.LazyFrame = pl.DataFrame({"p_value": ["1e-8", "N/A", "", "<0.001", "abc"], "effect_size": ["0.85", "n/a", "NULL", "x", "y"]}).lazy()
    result: pl.DataFrame = clean_numeric(lf).collect()
    assert result["p_value"].to_list() == [1e-8, None, None, None, None]
    assert result["effect_size"].to_list() == [0.85, None, None, None, None]


def test_clean_numeric_leaves_non_matching_untouched() -> None:
    """clean_numeric leaves non matching columns untouched."""
    lf: pl.LazyFrame = pl.DataFrame({"subject": ["BRCA1", "TP53"], "assertion_method": ["ANOVA", "t-test"], "p_value": ["0.05", "1e-8"]}).lazy()
    result: pl.DataFrame = clean_numeric(lf).collect()
    assert result.schema["subject"] == pl.String
    assert result.schema["assertion_method"] == pl.String
    assert result.schema["p_value"] == pl.Float64
    assert result["subject"].to_list() == ["BRCA1", "TP53"]
    assert result["assertion_method"].to_list() == ["ANOVA", "t-test"]


def test_clean_numeric_noop_without_numeric_columns() -> None:
    """clean_numeric is a noop when no numeric columns are present."""
    lf: pl.LazyFrame = pl.DataFrame({"subject": ["BRCA1"], "cohort": ["adult"]}).lazy()
    result: pl.DataFrame = clean_numeric(lf).collect()
    assert result.schema["subject"] == pl.String
    assert result.schema["cohort"] == pl.String


def test_clean_numeric_idempotent_on_float64() -> None:
    """clean_numeric is idempotent on already Float64 columns."""
    lf: pl.LazyFrame = pl.DataFrame({"p_value": [1e-8, 0.05]}).lazy()
    once: pl.DataFrame = clean_numeric(lf).collect()
    twice: pl.DataFrame = clean_numeric(once.lazy()).collect()
    assert twice["p_value"].to_list() == [1e-8, 0.05]
    assert twice.schema["p_value"] == pl.Float64


def test_format_numeric_emits_p_values_as_numbers() -> None:
    """P-value columns are emitted as real JSON numbers, not formatted strings.

    Biolink types ``p_value`` / ``adjusted_p_value`` as ``float``; writing
    ``"1.0000e-08"`` yields a file strict consumers reject even though Pydantic's lax
    mode happens to coerce it back.
    """
    lf: pl.LazyFrame = pl.DataFrame({"p_value": ["1e-8", "0.05", "0.001"], "adjusted_p_value": ["0.0001", "0.1", "0.2"]}).lazy()
    result: pl.DataFrame = format_numeric(clean_numeric(lf)).collect()
    assert result["p_value"].to_list() == [1e-08, 0.05, 0.001]
    assert result["adjusted_p_value"].to_list() == [0.0001, 0.1, 0.2]
    assert result.schema["p_value"] == pl.Float64


def test_format_numeric_decimal_general() -> None:
    """format_numeric renders effect size and study size in decimal general format."""
    lf: pl.LazyFrame = pl.DataFrame({"effect_size": ["0.85", "0.42", "0.1234"], "supporting_study_size": ["450", "1200", "7"]}).lazy()
    result: pl.DataFrame = format_numeric(clean_numeric(lf)).collect()
    assert result["effect_size"].to_list() == ["0.85", "0.42", "0.1234"]
    assert result["supporting_study_size"].to_list() == ["450", "1200", "7"]


def test_format_numeric_preserves_nulls() -> None:
    """format_numeric preserves nulls as null."""
    lf: pl.LazyFrame = pl.DataFrame({"p_value": ["1e-8", "N/A", "0.05"]}).lazy()
    result: pl.DataFrame = format_numeric(clean_numeric(lf)).collect()
    assert result["p_value"].to_list() == [1e-08, None, 0.05]


def test_format_numeric_cleans_float_noise() -> None:
    """format_numeric cleans floating point noise to four significant figures."""
    lf: pl.LazyFrame = pl.DataFrame({"effect_size": ["0.85000000001", "0.41999999999"]}).lazy()
    result: pl.DataFrame = format_numeric(clean_numeric(lf)).collect()
    assert result["effect_size"].to_list() == ["0.85", "0.42"]


def test_format_numeric_noop_without_numeric_columns() -> None:
    """format_numeric is a noop when no numeric columns are present."""
    lf: pl.LazyFrame = pl.DataFrame({"subject": ["BRCA1"], "cohort": ["adult"]}).lazy()
    result: pl.DataFrame = format_numeric(lf).collect()
    assert result["subject"].to_list() == ["BRCA1"]
    assert result.schema["subject"] == pl.String


def test_format_numeric_nulls_stripped_from_ndjson_rows() -> None:
    """cleaned and formatted null numeric values are stripped from NDJSON rows."""
    lf: pl.LazyFrame = pl.DataFrame({"subject": ["BRCA1", "TP53"], "p_value": ["1e-8", "N/A"], "effect_size": ["0.85", "0.42"]}).lazy()
    formatted: pl.DataFrame = format_numeric(clean_numeric(lf)).collect()
    rows: list[dict[str, Any]] = [strip_nulls(r) for r in formatted.iter_rows(named=True)]
    assert rows[0] == {"subject": "BRCA1", "p_value": 1e-08, "effect_size": "0.85"}
    assert "p_value" not in rows[1]
    assert rows[1]["subject"] == "TP53"
    assert rows[1]["effect_size"] == "0.42"


def test_compile_graph_emits_ndjson(monkeypatch: Any, tmp_path: Path) -> None:
    """compile_graph emits edges and nodes after float formatting config removal."""
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
            "primary_knowledge_source": [["infores:smoke"], ["infores:smoke"]],
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
    # Retrieval provenance is nested under `sources`, never flat on the edge.
    assert '"upstream_resource_ids":["infores:pubmed-central"]' not in flat
    # internal pre-resolution snapshot is stripped from final edges
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


def test_compile_graph_opens_ndjson_outputs_as_utf8(monkeypatch: Any, tmp_path: Path) -> None:
    """compile_graph passes UTF-8 file handles to Polars NDJSON writers."""
    monkeypatch.chdir(tmp_path)
    original_open: Any = Path.open
    append_encodings: list[tuple[str, str | None]] = []

    def spy_open(
        self: Path, mode: str = "r", buffering: int = -1, encoding: str | None = None, errors: str | None = None, newline: str | None = None
    ) -> Any:
        if mode == "a" and self.name.endswith(".ndjson.tmp"):
            append_encodings.append((self.name, encoding))
            return original_open(self, mode, buffering, encoding or "ascii", errors, newline)
        return original_open(self, mode, buffering, encoding, errors, newline)

    monkeypatch.setattr(Path, "open", spy_open)
    sub: Path = tmp_path / "sub.parquet"
    pl.DataFrame(
        {
            "subject": ["A-é"],
            "subject_name": ["Alpha-é"],
            "subject_category": ["gene"],
            "subject_taxon": [None],
            "subject_source": [None],
            "subject_source_version": [None],
            "subject_pre_resolution": ["A-é"],
            "object": ["HGNC:1"],
            "object_name": ["Xräy"],
            "object_category": ["disease"],
            "object_taxon": [None],
            "object_source": [None],
            "object_source_version": [None],
            "object_pre_resolution": ["Xräy"],
            "predicate": ["r"],
        }
    ).write_parquet(sub)

    lib.compile_graph([sub], "utf8", "1.0.0")

    assert append_encodings == [("utf8_1.0.0.nodes.ndjson.tmp", "utf-8"), ("utf8_1.0.0.edges.ndjson.tmp", "utf-8")]
    assert "Alpha-é" in (tmp_path / "utf8_1.0.0.nodes.ndjson").read_text(encoding="utf-8")
    assert "A-é" in (tmp_path / "utf8_1.0.0.edges.ndjson").read_text(encoding="utf-8")


def test_compile_graph_progress_callbacks_fire_per_subgraph_and_phase(monkeypatch: Any, tmp_path: Path) -> None:
    """compile_graph threads on_phase/on_subgraph: ordered phases, one tick per subgraph, output unchanged."""
    monkeypatch.chdir(tmp_path)

    def write_sub(p: Path, subj: str, obj: str) -> None:
        pl.DataFrame(
            {
                "subject": [subj],
                "subject_name": [subj],
                "subject_category": ["gene"],
                "subject_taxon": [None],
                "subject_source": [None],
                "subject_source_version": [None],
                "subject_pre_resolution": [subj],
                "object": [obj],
                "object_name": [obj],
                "object_category": ["disease"],
                "object_taxon": [None],
                "object_source": [None],
                "object_source_version": [None],
                "object_pre_resolution": [obj],
                "predicate": ["r"],
            }
        ).write_parquet(p)

    sub_a: Path = tmp_path / "a.parquet"
    sub_b: Path = tmp_path / "b.parquet"
    write_sub(sub_a, "A", "X")
    write_sub(sub_b, "B", "Y")

    phases: list[str] = []
    ticks: list[None] = []
    lib.compile_graph([sub_a, sub_b], "cb", "1.0.0", on_phase=phases.append, on_subgraph=lambda: ticks.append(None))

    # scan/normalize fire once per subgraph, then the shared write phases in order.
    assert phases == ["scan", "normalize", "scan", "normalize", "write-nodes", "write-edges", "dedup", "rig"]
    # on_subgraph ticks exactly once per subgraph (this is what drives the bar total).
    assert len(ticks) == 2

    # Callbacks are pure observation: KGX output is byte-identical to a no-callback run.
    lib.compile_graph([sub_a, sub_b], "cb2", "1.0.0")
    for stem in ("edges.ndjson", "nodes.ndjson"):
        assert (tmp_path / f"cb_1.0.0.{stem}").read_bytes() == (tmp_path / f"cb2_1.0.0.{stem}").read_bytes()


def test_compile_graph_keeps_qualifiers_and_publications_on_edges(monkeypatch: Any, tmp_path: Path) -> None:
    """compile_graph keeps qualifier and publication columns on edges, out of nodes."""
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
    # qualifier and publications stay on edges
    assert "MONDO:0005148" in edges
    assert "PMID:123" in edges
    # internal pre-resolution snapshots are stripped from final edges
    assert "_pre_resolution" not in edges
    # neither becomes a node
    assert "MONDO:0005148" not in nodes
    assert "PMID:123" not in nodes


def test_dedup_stream_nodes(tmp_path: Path) -> None:
    """dedup_stream deduplicates and strips null like values from node streams."""
    p_in: Path = tmp_path / "nodes.ndjson.tmp"
    p_in.write_text('{"id":"A","drop":"NA"}\n{"id":"A","drop":"NA"}\n{"id":"B"}\n')

    lib.dedup_stream(p_in, is_edges=False)

    assert not p_in.exists()  # temp input is removed
    lines: list[str] = (tmp_path / "nodes.ndjson").read_text().strip().splitlines()
    assert lines == ['{"id":"A"}', '{"id":"B"}']


def test_dedup_stream_edges(tmp_path: Path) -> None:
    """dedup_stream labels edges with UUID shaped ids and deduplicates."""
    import json

    p_in: Path = tmp_path / "edges.ndjson.tmp"
    p_in.write_text('{"subject":"A","object":"B","predicate":"r"}\n{"subject":"A","object":"B","predicate":"r"}\n')

    lib.dedup_stream(p_in, is_edges=True)

    assert not p_in.exists()
    lines: list[str] = (tmp_path / "edges.ndjson").read_text().strip().splitlines()
    assert len(lines) == 1  # duplicate edges collapse to one
    row: dict = json.loads(lines[0])
    assert "id" in row


def test_sig_works_on_cleaned_float64() -> None:
    """sig computes significance on a cleaned Float64 P value column."""
    lf: pl.LazyFrame = pl.DataFrame({"p_value": ["1e-8", "0.5", "N/A"]}).lazy()
    cleaned: pl.LazyFrame = clean_numeric(lf)
    result: pl.DataFrame = lib.sig(cleaned).collect()
    assert result["statistical_significance_qualifier"].to_list() == ["biolink:very_strongly_significant", "biolink:not_significant", None]


def test_edge_category_chemical_to_disease() -> None:
    """edge_category maps SmallMolecule + disease to ChemicalEntityToDiseaseAssociation."""
    lf: pl.LazyFrame = pl.LazyFrame({"subject category": ["biolink:SmallMolecule"], "object category": ["biolink:Disease"]})
    result: pl.DataFrame = edge_category(lf).collect()
    assert result["category"].to_list()[0] == ["biolink:ChemicalEntityToDiseaseOrPhenotypicFeatureAssociation"]


def test_edge_category_gene_to_disease() -> None:
    """edge_category maps gene + disease to GeneToDiseaseAssociation."""
    lf: pl.LazyFrame = pl.LazyFrame({"subject category": ["biolink:Gene"], "object category": ["biolink:Disease"]})
    result: pl.DataFrame = edge_category(lf).collect()
    assert result["category"].to_list()[0] == ["biolink:GeneToDiseaseAssociation"]


def test_edge_category_protein_to_disease() -> None:
    """edge_category bridges protein to gene (no ProteinTo* associations in biolink)."""
    lf: pl.LazyFrame = pl.LazyFrame({"subject category": ["biolink:Protein"], "object category": ["biolink:Disease"]})
    result: pl.DataFrame = edge_category(lf).collect()
    assert result["category"].to_list()[0] == ["biolink:GeneToDiseaseAssociation"]


def test_edge_category_unmapped_falls_back() -> None:
    """edge_category falls back to generic association for unmapped pairs."""
    lf: pl.LazyFrame = pl.LazyFrame({"subject category": ["biolink:Publication"], "object category": ["biolink:Pathway"]})
    result: pl.DataFrame = edge_category(lf).collect()
    assert result["category"].to_list()[0] == ["biolink:Association"]


def test_edge_category_drug_to_disease() -> None:
    """edge_category maps drug + disease through ChemicalEntity hierarchy."""
    lf: pl.LazyFrame = pl.LazyFrame({"subject category": ["biolink:Drug"], "object category": ["biolink:Disease"]})
    result: pl.DataFrame = edge_category(lf).collect()
    assert result["category"].to_list()[0] == ["biolink:ChemicalEntityToDiseaseOrPhenotypicFeatureAssociation"]


def test_edge_category_variant_to_disease() -> None:
    """edge_category maps SequenceVariant + disease through variant role."""
    lf: pl.LazyFrame = pl.LazyFrame({"subject category": ["biolink:SequenceVariant"], "object category": ["biolink:Disease"]})
    result: pl.DataFrame = edge_category(lf).collect()
    assert result["category"].to_list()[0] == ["biolink:VariantToDiseaseAssociation"]


def test_parse_edge_name_standard() -> None:
    """parse_edge_name parses standard name."""
    assert parse_edge_name("GeneToDiseaseAssociation") == ("Gene", ["Disease"])


def test_parse_edge_name_multi_object() -> None:
    """parse_edge_name splits multi-object names on or."""
    assert parse_edge_name("ChemicalEntityToDiseaseOrPhenotypicFeatureAssociation") == ("ChemicalEntity", ["Disease", "PhenotypicFeature"])


def test_parse_edge_name_no_to() -> None:
    """parse_edge_name returns none for non-standard names."""
    assert parse_edge_name("ChemicalGeneInteractionAssociation") is None


def test_parse_edge_name_embedded_or_not_split() -> None:
    """parse_edge_name does not split on "Or" embedded inside a role name."""
    assert parse_edge_name("OrganismTaxonToOrganismTaxonAssociation") == ("OrganismTaxon", ["OrganismTaxon"])


def test_edge_tables_cached() -> None:
    """edge_tables returns same object on repeat calls (cached)."""
    first: tuple[dict[str, str], dict[str, str]] = edge_tables()
    second: tuple[dict[str, str], dict[str, str]] = edge_tables()
    assert first is second


def test_pvalue_target_matches_common_spellings() -> None:
    """pvalue_target matches common P value spellings."""
    names: list[str] = ["p value", "p-value", "p.value", "pvalue", "P VALUE", "p vals", "p-values", "P"]
    for n in names:
        assert pvalue_target(n) == "p_value", n


def test_pvalue_target_matches_bare_p_and_padj_conventions() -> None:
    """pvalue_target matches bare P and padj style conventions found in real GWAS/DESeq2 data."""
    plain: list[str] = ["p SMR", "smr p", "p eQTL", "eqtl p", "gwas p", "fisher combined p", "log p"]
    for n in plain:
        assert pvalue_target(n) == "p_value", n

    adjusted: list[str] = ["padj", "p.adj", "adj.P.Val"]
    for n in adjusted:
        assert pvalue_target(n) == "adjusted_p_value", n


def test_pvalue_target_detects_adjusted_variants() -> None:
    """pvalue_target detects adjusted P value variants."""
    names: list[str] = ["adjusted p value", "adjusted-p-value", "adj p value"]
    for n in names:
        assert pvalue_target(n) == "adjusted_p_value", n


def test_pvalue_target_detects_broader_adjustment_synonyms() -> None:
    """pvalue_target detects broader adjustment synonyms."""
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


def test_pvalue_target_bare_corrected_is_not_treated_as_adjusted() -> None:
    """pvalue_target does not treat bare corrected as adjusted without a P/Q value token."""
    assert pvalue_target("corrected age") is None
    assert pvalue_target("batch corrected expression") is None


# pvalue_target does not treat bare adjusted as adjusted P value without a P/Q value token
# regression for a real false positive found auditing production KGX output: "fully adjusted HR"
# is an adjusted hazard ratio, not a P value
def test_pvalue_target_bare_adjusted_without_pvalue_context_is_not_treated_as_adjusted() -> None:
    assert pvalue_target("fully adjusted HR") is None
    assert pvalue_target("adjusted odds ratio") is None


# pvalue_target excludes significance flag columns
# regression for a real false positive found auditing production KGX output: "bonferroni significance"
# is a categorical flag like sig()'s own "statistical_significance_qualifier" column, not the numeric value
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


# pvalue_target excludes bare Q and Q statistic columns
# bare "Q" is deliberately not treated as Q value like since real data also uses it for
# cochran's Q test statistic, unrelated to storey's Q value
def test_pvalue_target_excludes_bare_q_and_q_statistic_columns() -> None:
    names: list[str] = ["Q degrees of freedom", "heterogeneity statistic Q", "Cochran Q statistic"]
    for n in names:
        assert pvalue_target(n) is None, n


def test_pvalue_target_excludes_unrelated_columns() -> None:
    """pvalue_target excludes unrelated columns."""
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


def test_pvalue_target_unadjusted_prefix_not_treated_as_adjusted() -> None:
    """pvalue_target does not conflate unadjusted with adjusted."""
    assert pvalue_target("unadjusted p value") == "p_value"
    assert pvalue_target("unadjusted HR") is None


def test_pvalue_target_matches_separator_glued_bare_p() -> None:
    """pvalue_target treats a bare P glued by underscores/dots/hyphens as a P value.

    Real GWAS outputs glue qualifiers onto a standalone P with underscores
    ("raw_p", "snp_p", "p_nominal"); these are the underscore-delimited siblings
    of the already-supported space-delimited forms ("gwas p", "log p").
    """
    plain: list[str] = [
        "raw_p",
        "nominal_p",
        "unadjusted_p",
        "p_nominal",
        "p_two_tailed",
        "two_sided_p_value",
        "wald_p",
        "p_wald",
        "snp_p",
        "assoc_p",
        "meta_p",
        "empirical_p",
        "perm_p",
        "permutation_p",
    ]
    for n in plain:
        assert pvalue_target(n) == "p_value", n


def test_pvalue_target_matches_numbered_pvalue_conventions() -> None:
    """pvalue_target matches numbered P/Q value columns from multi-phenotype outputs."""
    plain: list[str] = ["pvalue1", "p_value_1", "pvalue2"]
    for n in plain:
        assert pvalue_target(n) == "p_value", n

    adjusted: list[str] = ["padj_1", "qvalue1"]
    for n in adjusted:
        assert pvalue_target(n) == "adjusted_p_value", n


def test_pvalue_target_excludes_alnum_glued_p_and_near_misses() -> None:
    """pvalue_target excludes P glued to a letter/digit and value-like near misses.

    Guards the separator-glued bare-P expansion: gene/protein and chemistry names
    ("p53", "p16", "pH", "protein", "phosphate") and words merely ending in P
    ("peak value", "probability") must not be read as P values.
    """
    names: list[str] = ["pH", "protein", "phosphate", "p53", "p16", "peak value", "probability", "q statistic", "adjusted hazard ratio"]
    for n in names:
        assert pvalue_target(n) is None, n


def test_coerce_pvalue_columns_renames_single_p_value_column() -> None:
    """coerce_pvalue_columns renames a single P value column."""
    lf: pl.LazyFrame = pl.DataFrame({"p value": [0.01, 0.05]}).lazy()
    result: pl.DataFrame = coerce_pvalue_columns(lf).collect()
    assert "p_value" in result.columns
    assert "p value" not in result.columns
    assert result["p_value"].to_list() == [0.01, 0.05]


def test_coerce_pvalue_columns_renames_both_p_value_and_adjusted() -> None:
    """coerce_pvalue_columns renames both P value and adjusted P value columns together."""
    lf: pl.LazyFrame = pl.DataFrame({"p value": [0.01], "adjusted p value": [0.2]}).lazy()
    result: pl.DataFrame = coerce_pvalue_columns(lf).collect()
    assert result["p_value"].to_list() == [0.01]
    assert result["adjusted_p_value"].to_list() == [0.2]


def test_coerce_pvalue_columns_picks_best_fuzzy_match_among_multiple_candidates() -> None:
    """coerce_pvalue_columns picks the best fuzzy match among multiple candidates."""
    lf: pl.LazyFrame = pl.DataFrame({"log p value": [0.9], "p value": [0.01]}).lazy()
    result: pl.DataFrame = coerce_pvalue_columns(lf).collect()
    assert result["p_value"].to_list() == [0.01]
    assert result["log p value"].to_list() == [0.9]


def test_coerce_pvalue_columns_noop_without_pvalue_columns() -> None:
    """coerce_pvalue_columns is a noop without P value like columns."""
    lf: pl.LazyFrame = pl.DataFrame({"subject": ["BRCA1"], "cohort": ["adult"]}).lazy()
    result: pl.DataFrame = coerce_pvalue_columns(lf).collect()
    assert result.columns == ["subject", "cohort"]


def test_coerce_pvalue_columns_noop_when_already_canonical() -> None:
    """coerce_pvalue_columns is a noop when already canonically named."""
    lf: pl.LazyFrame = pl.DataFrame({"p_value": [0.01]}).lazy()
    result: pl.DataFrame = coerce_pvalue_columns(lf).collect()
    assert result.columns == ["p_value"]
    assert result["p_value"].to_list() == [0.01]


def test_coerce_pvalue_columns_keeps_existing_canonical_over_alias() -> None:
    """An existing canonical column wins over a higher-scoring spaced alias (no duplicate rename)."""
    lf: pl.LazyFrame = pl.DataFrame({"p_value": [0.01], "p value": [0.02]}).lazy()
    result: pl.DataFrame = coerce_pvalue_columns(lf).collect()
    assert result.columns == ["p_value", "p value"]
    assert result["p_value"].to_list() == [0.01]
    assert result["p value"].to_list() == [0.02]


def test_study_size_target_matches_common_spellings() -> None:
    """study_size_target matches common study size spellings."""
    names: list[str] = ["n", "N", "sample_size", "sample size", "sample-size", "sample.size", "samplesize", "study size", "cohort size"]
    for n in names:
        assert study_size_target(n) == "supporting_study_size", n


def test_study_size_target_matches_count_synonyms() -> None:
    """study_size_target matches count synonyms with explicit sample/study context."""
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


def test_study_size_target_excludes_false_positives() -> None:
    """study_size_target excludes false positives without explicit study size meaning."""
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


def test_study_size_target_matches_cohort_total_and_study_n_variants() -> None:
    """study_size_target matches cohort counts, mirrored total-n, and study-n labels."""
    names: list[str] = [
        "cohort_n",
        "cohort n",
        "cohort_count",
        "cohort count",
        "n_cohort",
        "total_cohort",
        "n_total",
        "n total",
        "study_n",
        "study n",
    ]
    for n in names:
        assert study_size_target(n) == "supporting_study_size", n


def test_study_size_target_matches_enrolled_variants() -> None:
    """study_size_target treats 'enrolled' as an 'enrollment' study-size variant."""
    names: list[str] = ["enrolled", "enrolled_count", "enrolled n"]
    for n in names:
        assert study_size_target(n) == "supporting_study_size", n


def test_study_size_target_excludes_expanded_near_misses() -> None:
    """study_size_target keeps near-misses out after the cohort/enrolled expansion.

    Guards the expansion: adding cohort/enrolled as units must not pull in
    identifier/date columns ("cohort_id", "enrolled_date") or unrelated n-words
    ("nucleotide_variation", "normalization"), and "case"/"subject" units must not
    match "case_control"/"subject_id".
    """
    names: list[str] = [
        "cohort_id",
        "cohort_name",
        "case_control",
        "subject_id",
        "study_id",
        "nucleotide_variation",
        "normalization",
        "sample_size_estimate",
        "enrollment_date",
        "enrolled_date",
        "participant_id",
    ]
    for n in names:
        assert study_size_target(n) is None, n


def test_coerce_study_size_columns_renames_n_column() -> None:
    """coerce_study_size_columns renames bare N to supporting_study_size."""
    lf: pl.LazyFrame = pl.DataFrame({"n": [120, 450]}).lazy()
    result: pl.DataFrame = coerce_study_size_columns(lf).collect()
    assert "supporting_study_size" in result.columns
    assert "n" not in result.columns
    assert result["supporting_study_size"].to_list() == [120, 450]


def test_coerce_study_size_columns_renames_sample_size_column() -> None:
    """coerce_study_size_columns renames sample_size to supporting_study_size."""
    lf: pl.LazyFrame = pl.DataFrame({"sample_size": [1200]}).lazy()
    result: pl.DataFrame = coerce_study_size_columns(lf).collect()
    assert result.columns == ["supporting_study_size"]
    assert result["supporting_study_size"].to_list() == [1200]


def test_coerce_study_size_columns_picks_best_candidate() -> None:
    """coerce_study_size_columns picks the best candidate and leaves others untouched."""
    lf: pl.LazyFrame = pl.DataFrame({"n": [9], "sample size": [1200], "participants": [1250]}).lazy()
    result: pl.DataFrame = coerce_study_size_columns(lf).collect()
    assert result["supporting_study_size"].to_list() == [1200]
    assert result["n"].to_list() == [9]
    assert result["participants"].to_list() == [1250]


def test_coerce_study_size_columns_noop_when_already_canonical() -> None:
    """coerce_study_size_columns is a noop when already canonically named."""
    lf: pl.LazyFrame = pl.DataFrame({"supporting_study_size": [1200], "sample_size": [999]}).lazy()
    result: pl.DataFrame = coerce_study_size_columns(lf).collect()
    assert result.columns == ["supporting_study_size", "sample_size"]
    assert result["supporting_study_size"].to_list() == [1200]
    assert result["sample_size"].to_list() == [999]


def test_coerce_study_size_columns_keeps_existing_canonical_over_alias() -> None:
    """An existing canonical column wins over a higher-scoring spaced alias (no duplicate rename)."""
    lf: pl.LazyFrame = pl.DataFrame({"supporting_study_size": [1200], "supporting study size": [999]}).lazy()
    result: pl.DataFrame = coerce_study_size_columns(lf).collect()
    assert result.columns == ["supporting_study_size", "supporting study size"]
    assert result["supporting_study_size"].to_list() == [1200]
    assert result["supporting study size"].to_list() == [999]


# --- Effect-size / effect-type coercion (Biolink PR #1774) --------------------------------------


def test_effect_size_target_matches_common_spellings() -> None:
    """effect_size_target matches common effect size spellings including the old name."""
    names: list[str] = [
        "effect size",
        "effectsize",
        "effect_size",
        "ES",
        "es",
        "relationship_strength",
        "relationship strength",
        "beta",
        "beta coefficient",
        "log2FC",
        "log2 fold change",
        "odds ratio",
        "OR",
        "hazard ratio",
        "HR",
        "risk ratio",
        "correlation",
        "rho",
        "r",
    ]
    for n in names:
        assert effect_size_target(n) == "effect_size", n


def test_effect_size_target_matches_qualified_statistic_names() -> None:
    """effect_size_target matches qualified numeric-statistic column names."""
    names: list[str] = [
        "spearman rho",
        "Spearman's rho",
        "spearman_rho",
        "pearson_r",
        "kendall_tau",
        "correlation coefficient",
        "adjusted odds ratio",
        "effect size estimate",
        "log2_FC",
    ]
    for n in names:
        assert effect_size_target(n) == "effect_size", n


def test_effect_size_target_excludes_false_positives() -> None:
    """effect_size_target excludes identifiers, unrelated columns, and effect-type labels."""
    names: list[str] = [
        "correlation_id",
        "subject",
        "gene",
        "p_value",
        "sample_id",
        "beta_actin",
        "or_value",
        "hr_status",
        "order",
        "effect type",
        "effect metric",
        "statistic type",
        "metric",
    ]
    for n in names:
        assert effect_size_target(n) is None, n


def test_effect_type_target_matches_common_spellings() -> None:
    """effect_type_target matches effect-type/metric label spellings."""
    names: list[str] = [
        "effect type",
        "effect_type",
        "effect metric",
        "statistic type",
        "statistical type",
        "metric type",
        "metric",
        "effect size type",
    ]
    for n in names:
        assert effect_type_target(n) == "effect_type", n


def test_effect_type_target_excludes_unrelated_columns() -> None:
    """effect_type_target excludes effect-size names and unrelated columns."""
    names: list[str] = ["effect_size", "effect size", "metric_value", "subject", "effect", "correlation"]
    for n in names:
        assert effect_type_target(n) is None, n


def test_coerce_effect_size_columns_renames_relationship_strength() -> None:
    """coerce_effect_size_columns renames the old relationship_strength name forward to effect_size."""
    lf: pl.LazyFrame = pl.DataFrame({"relationship_strength": [0.85, 0.42]}).lazy()
    result: pl.DataFrame = coerce_effect_size_columns(lf).collect()
    assert "effect_size" in result.columns
    assert "relationship_strength" not in result.columns
    assert result["effect_size"].to_list() == [0.85, 0.42]


def test_coerce_effect_size_columns_renames_effect_size_like_column() -> None:
    """coerce_effect_size_columns renames an effect-size-like column."""
    lf: pl.LazyFrame = pl.DataFrame({"spearman rho": [0.85]}).lazy()
    result: pl.DataFrame = coerce_effect_size_columns(lf).collect()
    assert result.columns == ["effect_size"]
    assert result["effect_size"].to_list() == [0.85]


def test_coerce_effect_size_columns_picks_best_candidate() -> None:
    """coerce_effect_size_columns picks the best candidate and leaves others untouched."""
    lf: pl.LazyFrame = pl.DataFrame({"effect size estimate": [0.1], "effect size": [0.85], "beta": [0.3]}).lazy()
    result: pl.DataFrame = coerce_effect_size_columns(lf).collect()
    assert result["effect_size"].to_list() == [0.85]
    assert result["effect size estimate"].to_list() == [0.1]
    assert result["beta"].to_list() == [0.3]


def test_coerce_effect_size_columns_noop_without_candidates() -> None:
    """coerce_effect_size_columns is a noop without effect-size-like columns."""
    lf: pl.LazyFrame = pl.DataFrame({"subject": ["BRCA1"], "cohort": ["adult"]}).lazy()
    result: pl.DataFrame = coerce_effect_size_columns(lf).collect()
    assert result.columns == ["subject", "cohort"]


def test_coerce_effect_size_columns_noop_when_already_canonical() -> None:
    """coerce_effect_size_columns is a noop when already canonically named."""
    lf: pl.LazyFrame = pl.DataFrame({"effect_size": [0.85], "beta": [0.3]}).lazy()
    result: pl.DataFrame = coerce_effect_size_columns(lf).collect()
    assert result.columns == ["effect_size", "beta"]
    assert result["effect_size"].to_list() == [0.85]


def test_coerce_effect_size_columns_keeps_existing_canonical_over_alias() -> None:
    """An existing canonical column wins over a higher-scoring spaced alias (no duplicate rename)."""
    lf: pl.LazyFrame = pl.DataFrame({"effect_size": [0.85], "effect size": [0.99]}).lazy()
    result: pl.DataFrame = coerce_effect_size_columns(lf).collect()
    assert result.columns == ["effect_size", "effect size"]
    assert result["effect_size"].to_list() == [0.85]
    assert result["effect size"].to_list() == [0.99]


def test_coerce_effect_type_columns_renames_and_maps_alias_values() -> None:
    """coerce_effect_type_columns renames the column and maps alias values to canonical enum values."""
    lf: pl.LazyFrame = pl.DataFrame({"effect size": [0.85, 1.2, 0.4], "effect type": ["Spearman", "odds ratio", "beta"]}).lazy()
    result: pl.DataFrame = coerce_effect_type_columns(coerce_effect_size_columns(lf)).collect()
    assert "effect_type" in result.columns
    assert "effect type" not in result.columns
    assert result["effect_type"].to_list() == ["spearmans_rho", "odds_ratio", "regression_coefficient"]


def test_coerce_effect_type_columns_maps_canonical_and_case_variants() -> None:
    """coerce_effect_type_columns passes canonical values through case/separator-insensitively."""
    lf: pl.LazyFrame = pl.DataFrame({"effect_size": [0.5, 0.6, 0.7], "effect_type": ["COHENS_D", "Hedges' g", "eta-squared"]}).lazy()
    result: pl.DataFrame = coerce_effect_type_columns(lf).collect()
    assert result["effect_type"].to_list() == ["cohens_d", "hedges_g", "eta_squared"]


def test_coerce_effect_type_columns_fuzzy_fallback_and_unmatched_null() -> None:
    """coerce_effect_type_columns fuzzy-matches close spellings and nulls values matching nothing."""
    lf: pl.LazyFrame = pl.DataFrame(
        {"effect_size": [0.5, 0.6, 0.7], "effect_type": ["spearmans", "pearsons", "totally unrelated garbage xyz"]}
    ).lazy()
    result: pl.DataFrame = coerce_effect_type_columns(lf).collect()
    assert result["effect_type"].to_list() == ["spearmans_rho", "pearsons_r", None]


def test_map_effect_type_value_direct() -> None:
    """_map_effect_type_value handles null input directly (map_elements skips nulls itself)."""
    assert _map_effect_type_value(None) is None
    assert _map_effect_type_value("   ") is None
    assert _map_effect_type_value("Cohen's d") == "cohens_d"
    assert _map_effect_type_value("spearmans") == "spearmans_rho"
    assert _map_effect_type_value("totally unrelated garbage xyz") is None


def test_effect_type_aliases_only_map_to_permissible_values() -> None:
    """Drift guard: every alias canonical is a permissible value, and canonical values round-trip."""
    assert {canonical for _, canonical in _EFFECT_TYPE_ALIASES} <= set(EFFECT_TYPE_VALUES)
    for value in EFFECT_TYPE_VALUES:
        assert _map_effect_type_value(value) == value


def test_coerce_effect_type_columns_nulls_blank_and_null_values() -> None:
    """coerce_effect_type_columns maps null and blank values to null."""
    lf: pl.LazyFrame = pl.DataFrame({"effect_size": [0.1, 0.2], "effect_type": [None, "  "]}).lazy()
    result: pl.DataFrame = coerce_effect_type_columns(lf).collect()
    assert result["effect_type"].to_list() == [None, None]


def test_coerce_effect_type_columns_nulls_where_effect_size_null() -> None:
    """effect_type is nulled on rows whose effect_size is null or non-numeric (Biolink class rule)."""
    lf: pl.LazyFrame = pl.DataFrame({"effect size": ["0.85", None, "n/a"], "effect type": ["Spearman", "OR", "beta"]}).lazy()
    result: pl.DataFrame = coerce_effect_type_columns(coerce_effect_size_columns(lf)).collect()
    assert result["effect_size"].to_list() == ["0.85", None, "n/a"]
    assert result["effect_type"].to_list() == ["spearmans_rho", None, None]


def test_coerce_effect_type_columns_nulls_entirely_without_effect_size() -> None:
    """effect_type is nulled entirely when no effect_size column is present (Biolink class rule)."""
    lf: pl.LazyFrame = pl.DataFrame({"metric": ["OR", "pearson r"]}).lazy()
    result: pl.DataFrame = coerce_effect_type_columns(lf).collect()
    assert result.columns == ["effect_type"]
    assert result["effect_type"].to_list() == [None, None]


def test_coerce_effect_type_columns_picks_best_candidate() -> None:
    """coerce_effect_type_columns picks the best candidate column and leaves others untouched."""
    lf: pl.LazyFrame = pl.DataFrame({"effect_size": [0.85, 0.2], "effect size type": ["Spearman", "OR"], "effect type": ["pearson r", "beta"]}).lazy()
    result: pl.DataFrame = coerce_effect_type_columns(lf).collect()
    assert result["effect_type"].to_list() == ["pearsons_r", "regression_coefficient"]
    assert result["effect size type"].to_list() == ["Spearman", "OR"]


def test_coerce_effect_type_columns_keeps_existing_canonical_over_alias() -> None:
    """An existing canonical column wins over a higher-scoring spaced alias (no duplicate rename)."""
    lf: pl.LazyFrame = pl.DataFrame(
        {"effect_size": [0.85, 0.2], "effect_type": ["odds_ratio", "cohens_d"], "effect type": ["pearson r", "beta"]}
    ).lazy()
    result: pl.DataFrame = coerce_effect_type_columns(lf).collect()
    assert result["effect_type"].to_list() == ["odds_ratio", "cohens_d"]
    assert result["effect type"].to_list() == ["pearson r", "beta"]


def test_coerce_effect_type_columns_noop_without_candidates() -> None:
    """coerce_effect_type_columns is a noop without effect-type-like columns."""
    lf: pl.LazyFrame = pl.DataFrame({"subject": ["BRCA1"], "effect_size": [0.85]}).lazy()
    result: pl.DataFrame = coerce_effect_type_columns(lf).collect()
    assert result.columns == ["subject", "effect_size"]


def test_coerced_effect_size_alias_survives_unknown_folding() -> None:
    """The old relationship_strength name becomes a top-level effect_size field before unknown folding."""
    lf: pl.LazyFrame = pl.DataFrame(
        {"subject": ["A"], "object": ["B"], "predicate": ["related_to"], "relationship_strength": ["0.85"], "miscellaneous_notes": ["note"]}
    ).lazy()
    out: pl.DataFrame = fold_unknown_to_supporting_text(coerce_effect_size_columns(lf)).collect()
    assert out["effect_size"].to_list() == ["0.85"]
    assert "relationship_strength" not in out.columns
    assert out["supporting_text"].to_list() == [["miscellaneous_notes: note"]]


def test_unpicked_relationship_strength_folds_into_supporting_text() -> None:
    """When a better effect-size candidate wins the fuzzy pick, the superseded old name folds."""
    lf: pl.LazyFrame = pl.DataFrame(
        {"subject": ["A"], "object": ["B"], "predicate": ["related_to"], "effect size": ["0.85"], "relationship_strength": ["0.42"]}
    ).lazy()
    out: pl.DataFrame = fold_unknown_to_supporting_text(coerce_effect_size_columns(lf)).collect()
    assert out["effect_size"].to_list() == ["0.85"]
    assert "relationship_strength" not in out.columns
    assert out["supporting_text"].to_list() == [["relationship_strength: 0.42"]]


def test_coerced_effect_type_survives_unknown_folding() -> None:
    """Coerced effect_type values stay top-level edge fields after unknown folding."""
    lf: pl.LazyFrame = pl.DataFrame(
        {
            "subject": ["A"],
            "object": ["B"],
            "predicate": ["related_to"],
            "effect_size": ["0.85"],
            "effect type": ["Spearman"],
            "miscellaneous_notes": ["note"],
        }
    ).lazy()
    out: pl.DataFrame = fold_unknown_to_supporting_text(coerce_effect_type_columns(lf)).collect()
    assert out["effect_size"].to_list() == ["0.85"]
    assert out["effect_type"].to_list() == ["spearmans_rho"]
    assert out["supporting_text"].to_list() == [["miscellaneous_notes: note"]]


# tcode coerces P value columns after annotations and before clean_numeric
# so downstream numeric_columns/sig/format_numeric see already canonical p_value/adjusted_p_value names
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


# tcode coerces study size columns after annotations and before clean_numeric
# so downstream numeric_columns/format_numeric see already canonical supporting_study_size names
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


# tcode coerces effect size and effect type columns after annotations and before clean_numeric
# so downstream numeric_columns/format_numeric see already canonical effect_size names
def test_tcode_collect_coerces_effect_columns_before_clean_numeric(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    store: Path = Path("/tmp/sectionhash.parquet")
    tcode_model: Tcode = Tcode.model_validate(  # pyright: ignore
        {**data, "config": fixtures_path / "minimal_section.yaml", "store": store}
    )

    collected: list[tuple[Any, tuple[Any]]] = tcode_model.collect(Path("/tmp/fullmap.redb"))  # pyright: ignore
    study_idx: int = next(i for i, op in enumerate(collected) if op[0].__name__ == "coerce_study_size_columns")
    size_idx: int = next(i for i, op in enumerate(collected) if op[0].__name__ == "coerce_effect_size_columns")
    type_idx: int = next(i for i, op in enumerate(collected) if op[0].__name__ == "coerce_effect_type_columns")
    clean_idx: int = next(i for i, op in enumerate(collected) if op[0].__name__ == "clean_numeric")

    assert study_idx < size_idx < type_idx < clean_idx


def test_coerced_study_size_alias_survives_unknown_folding() -> None:
    """study size aliases become top-level supporting study size fields before unknown folding."""
    lf: pl.LazyFrame = pl.DataFrame(
        {"subject": ["A"], "object": ["B"], "predicate": ["related_to"], "sample_size": [12000], "miscellaneous_notes": ["note"]}
    ).lazy()
    out: pl.DataFrame = fold_unknown_to_supporting_text(coerce_study_size_columns(lf)).collect()
    assert "sample_size" not in out.columns
    if "supporting_study_size" in UNSATISFIABLE_EDGE_FIELDS:
        # Unattached in the installed model: folded rather than emitted unvalidatably.
        assert "supporting_study_size" not in out.columns
        assert "supporting_study_size: 12000" in out["supporting_text"].to_list()[0]
    else:
        assert out["supporting_study_size"].to_list() == [12000]
    assert "miscellaneous_notes: note" in out["supporting_text"].to_list()[0]


def test_publications_wraps_curie_as_list() -> None:
    """publications() wraps a CURIE literal as a single element list[str] column."""
    lf: pl.LazyFrame = pl.DataFrame({"subject": ["A"]}).lazy()
    out: pl.DataFrame = publications(lf, "PMID:42").collect()
    assert out.schema["publications"] == pl.List(pl.String)
    assert out["publications"].to_list() == [["PMID:42"]]


def test_idx_emits_one_based_extracted_from_row_number() -> None:
    """idx emits a 1-Based column named extracted_from_row_number."""
    lf: pl.LazyFrame = pl.LazyFrame({"a": ["x", "y", "z"]})
    out: pl.DataFrame = idx(lf).collect()
    assert "extracted_from_row_number" in out.columns
    assert out["extracted_from_row_number"].to_list() == [1, 2, 3]


def test_fold_unknown_noop_when_all_allowed() -> None:
    """fold_unknown_to_supporting_text is a noop when every column is on the allow list."""
    lf: pl.LazyFrame = pl.DataFrame(
        {
            "subject": ["A"],
            "object": ["B"],
            "predicate": ["related_to"],
            "p_value": [0.01],
            "disease_context_qualifier": ["MONDO:0005148"],
            "publications": [["PMID:1"]],
        }
    ).lazy()
    out: pl.DataFrame = fold_unknown_to_supporting_text(lf).collect()
    # nothing folded, no supporting_text column created
    assert "supporting_text" not in out.columns
    assert set(out.columns) == {"subject", "object", "predicate", "p_value", "disease_context_qualifier", "publications"}


def test_fold_unknown_single_column() -> None:
    """fold_unknown_to_supporting_text folds a single unknown column as "col: value"."""
    lf: pl.LazyFrame = pl.DataFrame(
        {"subject": ["A"], "object": ["B"], "predicate": ["related_to"], "miscellaneous_notes": ["see smith et al"]}
    ).lazy()
    out: pl.DataFrame = fold_unknown_to_supporting_text(lf).collect()
    assert "miscellaneous_notes" not in out.columns
    assert out.schema["supporting_text"] == pl.List(pl.String)
    assert out["supporting_text"].to_list() == [["miscellaneous_notes: see smith et al"]]


def test_fold_unknown_multiple_columns_sorted() -> None:
    """fold_unknown_to_supporting_text folds multiple columns in deterministic sorted order."""
    lf: pl.LazyFrame = pl.DataFrame(
        {
            "subject": ["A"],
            "object": ["B"],
            "predicate": ["related_to"],
            # deliberately listed out of sort order to verify output is sorted by column name
            "extracted_from_row_number": ["7"],
            "sheet_name": ["Sheet1"],
            "miscellaneous_flag": ["yes"],
        }
    ).lazy()
    out: pl.DataFrame = fold_unknown_to_supporting_text(lf).collect()
    assert out["supporting_text"].to_list() == [["extracted_from_row_number: 7", "miscellaneous_flag: yes", "sheet_name: Sheet1"]]


def test_fold_unknown_skips_null_and_blank() -> None:
    """fold_unknown_to_supporting_text skips null and empty string values."""
    lf: pl.LazyFrame = pl.DataFrame(
        {
            "subject": ["A", "B", "C"],
            "object": ["X", "Y", "Z"],
            "predicate": ["related_to", "related_to", "related_to"],
            "miscellaneous_notes": ["present", None, "   "],
        }
    ).lazy()
    out: pl.DataFrame = fold_unknown_to_supporting_text(lf).collect()
    rows: list[list[str | None]] = out["supporting_text"].to_list()
    assert rows[0] == ["miscellaneous_notes: present"]
    # null and whitespace only both yield an empty list
    assert rows[1] == []
    assert rows[2] == []


def test_fold_unknown_appends_to_existing_list_supporting_text() -> None:
    """fold_unknown_to_supporting_text appends to existing list[str] supporting_text."""
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


def test_fold_unknown_coerces_scalar_supporting_text() -> None:
    """fold_unknown_to_supporting_text coerces scalar supporting_text to list[str] then appends."""
    lf: pl.LazyFrame = pl.DataFrame(
        {"subject": ["A"], "object": ["B"], "predicate": ["related_to"], "supporting_text": ["plain summary"], "miscellaneous_notes": ["extra"]}
    ).lazy()
    out: pl.DataFrame = fold_unknown_to_supporting_text(lf).collect()
    assert out.schema["supporting_text"] == pl.List(pl.String)
    assert out["supporting_text"].to_list() == [["plain summary", "miscellaneous_notes: extra"]]


def test_fold_unknown_preserves_qualifier_columns() -> None:
    """fold_unknown_to_supporting_text never folds known qualifier columns."""
    lf: pl.LazyFrame = pl.DataFrame(
        {
            "subject": ["A"],
            "object": ["B"],
            "predicate": ["related_to"],
            "disease_context_qualifier": ["MONDO:0005148"],
            "anatomical_context_qualifier": ["UBERON:0000061"],
        }
    ).lazy()
    out: pl.DataFrame = fold_unknown_to_supporting_text(lf).collect()
    # no supporting_text column materialized because nothing was foldable
    assert "supporting_text" not in out.columns
    assert "disease_context_qualifier" in out.columns
    assert "anatomical_context_qualifier" in out.columns


SUPPORTING_STUDY_SLOTS: tuple[str, ...] = (
    "supporting_study_method_types",
    "supporting_study_method_description",
    "supporting_study_size",
    "supporting_study_cohort",
    "supporting_study_date_range",
    "supporting_study_context",
)


def test_fold_unknown_tracks_supporting_study_slots_of_installed_model() -> None:
    """The ``supporting_study_*`` slots are folded iff the installed model can hold them.

    ``biolink/biolink-model#1770`` attaches these six to root ``association``. Until it
    ships they are declared in the LinkML schema but on no Pydantic class, so emitting
    them flat produces an unvalidatable edge. The behaviour must be derived from the
    installed model rather than pinned to either state.
    """
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
    # `has_supporting_studies` is a real Association slot in every supported version.
    assert "has_supporting_studies" in out.columns
    for col in SUPPORTING_STUDY_SLOTS:
        assert (col in out.columns) is (col not in UNSATISFIABLE_EDGE_FIELDS), col
    assert "miscellaneous_notes: see smith et al" in out["supporting_text"].to_list()[0]


def test_allowed_edge_fields_covers_tablassert_pipeline_columns() -> None:
    """ALLOWED_EDGE_FIELDS covers intentional tablassert output columns."""
    for col in ("publications", "sources", "p_value", "supporting_text", "has_supporting_studies"):
        assert col in ALLOWED_EDGE_FIELDS


def test_allowed_edge_fields_tracks_supporting_study_slots_of_installed_model() -> None:
    """``supporting_study_*`` membership follows the installed biolink-model exactly."""
    for col in SUPPORTING_STUDY_SLOTS:
        assert (col in ALLOWED_EDGE_FIELDS) is (col not in UNSATISFIABLE_EDGE_FIELDS), col


def test_compile_graph_folds_unknown_annotations_into_supporting_text(monkeypatch: Any, tmp_path: Path) -> None:
    """compile_graph folds non allow list annotation columns into supporting_text on edges."""
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
    # folded columns no longer appear as top level JSON keys on the edge object
    assert '"miscellaneous_notes":' not in edges
    assert '"extracted_from_row_number":' not in edges
    # but their values survive inside supporting_text
    assert "miscellaneous_notes: see smith et al" in edges
    assert "extracted_from_row_number: 7" in edges
    # real biolist fields survive as top level fields
    assert '"p_value":0.01' in edges or '"p_value": 0.01' in edges
    assert "PMID:1" in edges


def test_compile_subgraph_e2e_value_encoded_nodes(monkeypatch: Any, tmp_path: Path) -> None:
    """compile_subgraph resolves value-encoded subject/object nodes into parquet output.

    The fake fullmap returns the legacy flat-row shape (no ``records`` key) for every call
    regardless of ``return_format``, so ``lookup_rows`` takes its legacy-compat path: after the
    ``return_format="pairs"`` query it re-queries the FULL term set (hardening so partially-warm
    caches do not drop already-cached terms). The subject and object terms are therefore fetched
    in two identical FULLY-BATCHED calls — never split per-column — which is the batching
    guarantee this test asserts.
    """
    rows: dict[str, list[dict[str, object]]] = {
        "brca1": [fake_fullmap_row("brca1", "HGNC:1100", "BRCA1", "Gene", 9606)],
        "tp53": [fake_fullmap_row("tp53", "HGNC:11998", "TP53", "Gene", 9606)],
    }
    calls: list[list[str]] = install_fake_fullmap(monkeypatch, rows)
    table_path, _ = write_text_section(
        tmp_path,
        "value_nodes",
        {
            "statement": {"subject": {"method": "value", "encoding": "BRCA1"}, "object": {"method": "value", "encoding": "TP53"}},
            "provenance": {"repo": "PMC", "publication": "PMC0000000"},
        },
        ["ignored"],
    )
    data: Any = from_yaml(table_path)
    store: Path = tmp_path / "value_nodes.parquet"
    tcode_model: Tcode = Tcode.model_validate({**data, "config": table_path, "store": store, "name": "TEST_KG"})  # pyright: ignore

    result_path: Path = lib.compile_subgraph(tcode_model.collect(tmp_path / "fullmap.redb"))  # pyright: ignore
    result: dict[str, Any] = pl.read_parquet(result_path).row(0, named=True)

    assert calls == [["brca1", "tp53"], ["brca1", "tp53"]]  # legacy-shape re-query; both fully batched
    assert result_path == store
    assert result["subject"] == "HGNC:1100"
    assert result["subject_name"] == "BRCA1"
    assert result["subject_category"] == "biolink:Gene"
    assert result["object"] == "HGNC:11998"
    assert result["object_name"] == "TP53"
    assert result["predicate"] == "biolink:related_to"
    assert result["publications"] == ["PMCID:PMC0000000"]
    assert result["primary_knowledge_source"] == "infores:test-kg"


def test_compile_subgraph_e2e_column_cleanup_and_numeric_annotations(monkeypatch: Any, tmp_path: Path) -> None:
    """compile_subgraph applies column encodings, regex cleanup, aliases, and numeric formatting."""
    rows: dict[str, list[dict[str, object]]] = {
        "brca1": [fake_fullmap_row("brca1", "HGNC:1100", "BRCA1", "Gene", 9606)],
        "tp53": [fake_fullmap_row("tp53", "HGNC:11998", "TP53", "Gene", 9606)],
    }
    install_fake_fullmap(monkeypatch, rows)
    table_path, _ = write_text_section(
        tmp_path,
        "column_cleanup",
        {
            "statement": {
                "subject": {"method": "column", "encoding": "A", "regex": [{"pattern": "\\s+", "replacement": ""}], "remove": ["-", "\\[.*\\]"]},
                "object": {"method": "column", "encoding": "B", "remove": ["\\s+"]},
            },
            "annotations": [
                {"annotation": "P Value", "method": "column", "encoding": "C"},
                {"annotation": "sample size", "method": "column", "encoding": "D"},
                {"annotation": "miscellaneous_notes", "method": "column", "encoding": "E"},
            ],
            "provenance": {"repo": "PMID", "publication": "12345"},
        },
        ["BRCA-1 [alias]\tTP 53\t1e-8\t1200\tkept note"],
    )
    data: Any = from_yaml(table_path)
    store: Path = tmp_path / "column_cleanup.parquet"
    tcode_model: Tcode = Tcode.model_validate({**data, "config": table_path, "store": store})  # pyright: ignore

    result_path: Path = lib.compile_subgraph(tcode_model.collect(tmp_path / "fullmap.redb"))  # pyright: ignore
    result: dict[str, Any] = pl.read_parquet(result_path).row(0, named=True)

    assert result["subject"] == "HGNC:1100"
    assert result["original_subject"] == "BRCA-1 [alias]"
    assert result["object"] == "HGNC:11998"
    assert result["original_object"] == "TP 53"
    assert result["p_value"] == 1e-08
    # Both slots are unattached in biolink-model 4.4.3, so they are preserved on the
    # inlined StudyResult instead of being emitted unvalidatably on the edge.
    described: str = result["has_supporting_studies"][next(iter(result["has_supporting_studies"]))]["has_study_results"][0]["description"]
    assert "supporting_study_size=1200" in described
    assert "statistical_significance_qualifier=biolink:very_strongly_significant" in described
    assert result["miscellaneous_notes"] == "kept note"
    assert result["publications"] == ["PMID:12345"]


def test_compile_subgraph_e2e_release_drops_rows_before_fullmap_lookup(monkeypatch: Any, tmp_path: Path) -> None:
    """release-mode subgraph compilation drops not-significant rows before resolution."""
    rows: dict[str, list[dict[str, object]]] = {
        "keptgene": [fake_fullmap_row("keptgene", "HGNC:1", "KEPTGENE", "Gene", 9606)],
        "keptdisease": [fake_fullmap_row("keptdisease", "MONDO:1", "Kept disease", "Disease", 0)],
        "droppedgene": [fake_fullmap_row("droppedgene", "HGNC:2", "DROPPEDGENE", "Gene", 9606)],
        "droppeddisease": [fake_fullmap_row("droppeddisease", "MONDO:2", "Dropped disease", "Disease", 0)],
    }
    calls: list[list[str]] = install_fake_fullmap(monkeypatch, rows)
    table_path, _ = write_text_section(
        tmp_path,
        "release_drop",
        {
            "statement": {"subject": {"method": "column", "encoding": "A"}, "object": {"method": "column", "encoding": "B"}},
            "annotations": [{"annotation": "p_value", "method": "column", "encoding": "C"}],
            "provenance": {"repo": "PMC", "publication": "PMC0000000"},
        },
        ["DroppedGene\tDroppedDisease\t0.5", "KeptGene\tKeptDisease\t0.01"],
    )
    data: Any = from_yaml(table_path)
    store: Path = tmp_path / "release_drop.parquet"
    tcode_model: Tcode = Tcode.model_validate({**data, "config": table_path, "store": store, "release": True})  # pyright: ignore

    result_path: Path = lib.compile_subgraph(tcode_model.collect(tmp_path / "fullmap.redb"))  # pyright: ignore
    result: pl.DataFrame = pl.read_parquet(result_path)
    looked_up: set[str] = set(calls[0])

    assert result.height == 1
    assert result["subject"].to_list() == ["HGNC:1"]
    assert result["object"].to_list() == ["MONDO:1"]
    described = result["has_supporting_studies"].to_list()[0]
    assert (
        "statistical_significance_qualifier=biolink:strongly_significant" in (described[next(iter(described))]["has_study_results"][0]["description"])
    )
    assert "droppedgene" not in looked_up
    assert "droppeddisease" not in looked_up


def test_compile_subgraph_e2e_head_caps_rows_to_five(monkeypatch: Any, tmp_path: Path) -> None:
    """--head randomly samples 5 of 8 rows before fullmap resolution (never more than available)."""
    rows: dict[str, list[dict[str, object]]] = {
        **{f"gene{i}": [fake_fullmap_row(f"gene{i}", f"HGNC:{i}", f"GENE{i}", "Gene", 9606)] for i in range(1, 9)},
        **{f"disease{i}": [fake_fullmap_row(f"disease{i}", f"MONDO:{i}", f"Disease{i}", "Disease", 0)] for i in range(1, 9)},
    }
    calls: list[list[str]] = install_fake_fullmap(monkeypatch, rows)
    table_path, _ = write_text_section(
        tmp_path,
        "head_cap",
        {
            "statement": {"subject": {"method": "column", "encoding": "A"}, "object": {"method": "column", "encoding": "B"}},
            "provenance": {"repo": "PMC", "publication": "PMC0000000"},
        },
        [f"gene{i}\tdisease{i}" for i in range(1, 9)],
    )
    data: Any = from_yaml(table_path)
    store: Path = tmp_path / "head_cap.parquet"
    tcode_model: Tcode = Tcode.model_validate({**data, "config": table_path, "store": store, "head": True})  # pyright: ignore

    result_path: Path = lib.compile_subgraph(tcode_model.collect(tmp_path / "fullmap.redb"))  # pyright: ignore
    result: pl.DataFrame = pl.read_parquet(result_path)
    looked_up: set[str] = set(chain.from_iterable(calls))
    genes: set[str] = {t for t in looked_up if t.startswith("gene")}
    diseases: set[str] = {t for t in looked_up if t.startswith("disease")}

    assert result.height == 5
    assert len(genes) == 5
    assert len(diseases) == 5
    assert genes <= {f"gene{i}" for i in range(1, 9)}
    assert diseases <= {f"disease{i}" for i in range(1, 9)}


def test_compile_subgraph_and_graph_e2e_qualifier_stays_edge_attribute(monkeypatch: Any, tmp_path: Path) -> None:
    """auto-derived species context survives graph export as an edge attribute without creating nodes."""
    monkeypatch.chdir(tmp_path)
    rows: dict[str, list[dict[str, object]]] = {
        "brca1": [fake_fullmap_row("brca1", "HGNC:1100", "BRCA1", "Gene", 9606)],
        "disease x": [fake_fullmap_row("disease x", "MONDO:0000001", "Disease X", "Disease", 0)],
        "homo sapiens": [fake_fullmap_row("homo sapiens", "NCBITaxon:9606", "Homo sapiens", "OrganismTaxon", 9606)],
    }
    install_fake_fullmap(monkeypatch, rows)
    table_path, _ = write_text_section(
        tmp_path,
        "qualifier",
        {
            "statement": {
                "subject": {"method": "value", "encoding": "BRCA1"},
                "object": {"method": "value", "encoding": "Disease X"},
                "qualifiers": [],
            },
            "provenance": {"repo": "PMC", "publication": "PMC0000000"},
        },
        ["ignored"],
    )
    data: Any = from_yaml(table_path)
    store: Path = tmp_path / "qualifier.parquet"
    tcode_model: Tcode = Tcode.model_validate({**data, "config": table_path, "store": store, "name": "QUAL_KG"})  # pyright: ignore

    subgraph: Path = lib.compile_subgraph(tcode_model.collect(tmp_path / "fullmap.redb"))  # pyright: ignore
    # `biolink:Association` has no species_context_qualifier slot, so the value is
    # nulled on the edge and preserved on the inlined StudyResult instead.
    frame: pl.DataFrame = pl.read_parquet(subgraph)
    assert frame["species_context_qualifier"].to_list() == [None]
    assert (
        "species_context_qualifier=NCBITaxon:9606"
        in frame["has_supporting_studies"].to_list()[0][next(iter(frame["has_supporting_studies"].to_list()[0]))]["has_study_results"][0][
            "description"
        ]
    )

    lib.compile_graph([subgraph], "qual", "1.0.0")
    edges: list[dict[str, Any]] = [json.loads(line) for line in (tmp_path / "qual_1.0.0.edges.ndjson").read_text().splitlines()]
    nodes: list[dict[str, Any]] = [json.loads(line) for line in (tmp_path / "qual_1.0.0.nodes.ndjson").read_text().splitlines()]

    # Nulled on the edge (no such slot on biolink:Association) and kept on the study.
    assert "species_context_qualifier" not in edges[0]
    study: dict[str, Any] = edges[0]["has_supporting_studies"]
    assert "species_context_qualifier=NCBITaxon:9606" in study[next(iter(study))]["has_study_results"][0]["description"]
    assert all("species_context_qualifier_pre_resolution" not in edge for edge in edges)
    assert {node["id"] for node in nodes} == {"HGNC:1100", "MONDO:0000001"}
    assert "NCBITaxon:9606" not in {node["id"] for node in nodes}


def test_node_output_reflects_disease_taxon(monkeypatch: Any, tmp_path: Path) -> None:
    """Taxon-bearing disease nodes surface their taxon in KGX nodes NDJSON."""
    monkeypatch.chdir(tmp_path)
    rows: dict[str, list[dict[str, object]]] = {
        "disease taxon": [fake_fullmap_row("disease taxon", "MONDO:50", "Taxon-bearing disease", "Disease", 9606)],
        "brca1": [fake_fullmap_row("brca1", "HGNC:1100", "BRCA1", "Gene", 9606)],
    }
    install_fake_fullmap(monkeypatch, rows)
    table_path, _ = write_text_section(
        tmp_path,
        "disease_taxon_node",
        {
            "statement": {"subject": {"method": "value", "encoding": "Disease Taxon"}, "object": {"method": "value", "encoding": "BRCA1"}},
            "provenance": {"repo": "PMC", "publication": "PMC0000000"},
        },
        ["ignored"],
    )
    data: Any = from_yaml(table_path)
    store: Path = tmp_path / "disease_taxon_node.parquet"
    tcode_model: Tcode = Tcode.model_validate({**data, "config": table_path, "store": store, "name": "DISEASE_TAXON_KG"})  # pyright: ignore

    subgraph: Path = lib.compile_subgraph(tcode_model.collect(tmp_path / "fullmap.redb"))  # pyright: ignore
    lib.compile_graph([subgraph], "disease_taxon", "1.0.0")
    nodes: list[dict[str, Any]] = [json.loads(line) for line in (tmp_path / "disease_taxon_1.0.0.nodes.ndjson").read_text().splitlines()]

    disease_node: dict[str, Any] = next(node for node in nodes if node["id"] == "MONDO:50")
    assert disease_node["in_taxon"] == ["NCBITaxon:9606"]


def test_build_pipeline_e2e_smoke_with_monkeypatched_fullmap(monkeypatch: Any, tmp_path: Path) -> None:
    """build_pipeline runs all six stages and emits KGX/RIG using a monkeypatched fullmap DB."""
    from tablassert.ingests import to_yaml

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".tablassert" / "store").mkdir(parents=True)
    monkeypatch.setattr(cli, "Pool", SyncPool)
    rows: dict[str, list[dict[str, object]]] = {
        "brca1": [fake_fullmap_row("brca1", "HGNC:1100", "BRCA1", "Gene", 9606)],
        "tp53": [fake_fullmap_row("tp53", "HGNC:11998", "TP53", "Gene", 9606)],
    }
    install_fake_fullmap(monkeypatch, rows)
    table_path, _ = write_text_section(
        tmp_path,
        "pipeline_table",
        {
            "statement": {"subject": {"method": "value", "encoding": "BRCA1"}, "object": {"method": "value", "encoding": "TP53"}},
            "provenance": {"repo": "PMC", "publication": "PMC0000000"},
        },
        ["ignored"],
    )
    table_data: Any = from_yaml(table_path)
    to_yaml(table_path, {"template": table_data})
    graph_path: Path = tmp_path / "graph.yaml"
    to_yaml(
        graph_path,
        {
            "name": "PIPELINE_KG",
            "version": "0.1.0",
            "description": "Pipeline smoke graph.",
            "tables": [str(table_path)],
            "fullmap": str(tmp_path / "fullmap.redb"),
        },
    )
    progress: DummyProgress = DummyProgress()

    cli.build_pipeline(graph_path, cast(Any, progress), release=False, qc=False, log=False)
    edge_rows: list[dict[str, Any]] = [json.loads(line) for line in (tmp_path / "PIPELINE_KG_0.1.0.edges.ndjson").read_text().splitlines()]
    node_rows: list[dict[str, Any]] = [json.loads(line) for line in (tmp_path / "PIPELINE_KG_0.1.0.nodes.ndjson").read_text().splitlines()]
    rig: dict[str, Any] = from_yaml(tmp_path / "PIPELINE_KG_0.1.0.RIG.yaml")  # pyright: ignore

    assert progress.stages == [
        "Loading Tables",
        "Extracting Sections",
        "Building TCode",
        "Collecting Instructions",
        "Building Subgraphs",
        "Compiling Graph",
    ]
    assert len(edge_rows) == 1
    assert edge_rows[0]["subject"] == "HGNC:1100"
    assert edge_rows[0]["object"] == "HGNC:11998"
    primary_source: dict[str, Any] = next(x for x in edge_rows[0]["sources"] if x["resource_role"] == "primary_knowledge_source")
    assert primary_source["upstream_resource_ids"] == ["infores:pubmed-central"]
    assert edge_rows[0]["primary_knowledge_source"] == "infores:pipeline-kg"
    assert {row["id"] for row in node_rows} == {"HGNC:1100", "HGNC:11998"}
    assert rig["name"] == "PIPELINE_KG v0.1.0"
    edge_type: dict[str, Any] = rig["target_info"]["edge_type_info"][0]  # pyright: ignore
    assert edge_type["primary_knowledge_sources"] == ["infores:pipeline-kg", "infores:pubmed-central"]

    # The gate: every emitted record must construct as its own Biolink class. Without
    # this, a build can (and previously did) ship files where no record validated.
    report: dict[str, Any] = validate_kgx(tmp_path / "PIPELINE_KG_0.1.0.nodes.ndjson", tmp_path / "PIPELINE_KG_0.1.0.edges.ndjson")
    assert report["ok"], report
    assert report["edges"]["valid"] == report["edges"]["total"] == 1
    assert report["nodes"]["valid"] == report["nodes"]["total"] == 2


def test_build_pipeline_head_mode_isolates_store_and_caps_rows(monkeypatch: Any, tmp_path: Path) -> None:
    """--head caches subgraphs to .head.parquet (never clobbering a full build) and caps to 5 rows."""
    from tablassert.ingests import to_yaml

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".tablassert" / "store").mkdir(parents=True)
    monkeypatch.setattr(cli, "Pool", SyncPool)
    rows: dict[str, list[dict[str, object]]] = {
        **{f"gene{i}": [fake_fullmap_row(f"gene{i}", f"HGNC:{i}", f"GENE{i}", "Gene", 9606)] for i in range(1, 9)},
        **{f"disease{i}": [fake_fullmap_row(f"disease{i}", f"MONDO:{i}", f"Disease{i}", "Disease", 0)] for i in range(1, 9)},
    }
    install_fake_fullmap(monkeypatch, rows)
    table_path, _ = write_text_section(
        tmp_path,
        "head_store",
        {
            "statement": {"subject": {"method": "column", "encoding": "A"}, "object": {"method": "column", "encoding": "B"}},
            "provenance": {"repo": "PMC", "publication": "PMC0000000"},
        },
        [f"gene{i}\tdisease{i}" for i in range(1, 9)],
    )
    table_data: Any = from_yaml(table_path)
    to_yaml(table_path, {"template": table_data})
    graph_path: Path = tmp_path / "graph.yaml"
    to_yaml(
        graph_path,
        {
            "name": "HEAD_KG",
            "version": "0.1.0",
            "description": "Head preview graph.",
            "tables": [str(table_path)],
            "fullmap": str(tmp_path / "fullmap.redb"),
        },
    )
    progress: DummyProgress = DummyProgress()

    cli.build_pipeline(graph_path, cast(Any, progress), release=False, qc=False, log=False, head=True)

    edge_rows: list[dict[str, Any]] = [json.loads(line) for line in (tmp_path / "HEAD_KG_0.1.0.edges.ndjson").read_text().splitlines()]
    store_files: list[Path] = list((tmp_path / ".tablassert" / "store").glob("*.parquet"))

    assert len(edge_rows) == 5
    assert store_files, "expected a cached subgraph parquet"
    assert all(f.name.endswith(".head.parquet") for f in store_files)


def test_compile_subgraph_threads_fine_phases_into_resolve_and_qc(monkeypatch: Any, tmp_path: Path) -> None:
    """compile_subgraph forwards on_phase into resolve_batch/fullmap_audit so fine sub-phases fire in order."""
    rows: dict[str, list[dict[str, object]]] = {
        "brca1": [fake_fullmap_row("brca1", "HGNC:1100", "BRCA1", "Gene", 9606)],
        "tp53": [fake_fullmap_row("tp53", "HGNC:11998", "TP53", "Gene", 9606)],
    }
    install_fake_fullmap(monkeypatch, rows)

    # Safety net: this data passes QC at the exact stage, so BioBERT must never run (keeps the test offline).
    class DummyBioBERT:
        def encode(self, values: list[str]) -> object:
            raise AssertionError("Stage 3 (BioBERT) must not run for exact-match QC data")

    monkeypatch.setattr("tablassert.qc.get_biobert", lambda: DummyBioBERT())

    table_path, _ = write_text_section(
        tmp_path,
        "fine_phases",
        {
            "statement": {"subject": {"method": "value", "encoding": "BRCA1"}, "object": {"method": "value", "encoding": "TP53"}},
            "provenance": {"repo": "PMC", "publication": "PMC0000000"},
        },
        ["ignored"],
    )
    data: Any = from_yaml(table_path)
    store: Path = tmp_path / "fine_phases.parquet"
    tcode_model: Tcode = Tcode.model_validate({**data, "config": table_path, "store": store, "qc": True})  # pyright: ignore

    phases: list[str] = []
    result_path: Path = lib.compile_subgraph(tcode_model.collect(tmp_path / "fullmap.redb"), on_phase=phases.append)  # pyright: ignore

    assert result_path == store
    # Per-column resolve sub-phases fire in spec order, before any QC sub-phase.
    assert phases.index("resolve:subject") < phases.index("resolve:object")
    assert phases.index("resolve:object") < phases.index("qc:exact")
    # QC sub-phases fire for the audits: exact then fuzzy; bert never (exact-match quick exit).
    assert phases.index("qc:exact") < phases.index("qc:fuzzy")
    assert "qc:bert" not in phases


def test_predicate_options_answers_which_predicates_keep_the_class() -> None:
    """The authoring-time helper: which predicates does a (subject, object) pair actually permit?

    Before this existed there was no way to ask, which is how 723,595 edges shipped with
    ``gene_associated_with_condition`` on ``GeneToDiseaseAssociation`` -- a predicate that class
    forbids, so every one of them silently demoted to bare ``biolink:Association``.
    """
    from tablassert.lib import derived_edge_category, predicate_options

    assert derived_edge_category("biolink:Gene", "biolink:Disease") == "biolink:GeneToDiseaseAssociation"
    options = predicate_options("biolink:Gene", "biolink:Disease")
    assert options is not None
    assert options == {"biolink:affects", "biolink:associated_with", "biolink:contributes_to"}
    assert "biolink:gene_associated_with_condition" not in options

    # The bare name works identically (configs are written without the prefix).
    assert predicate_options("Gene", "Disease") == options
    # Roles roll up through CATEGORY_PARENT: Protein is a Gene-role subject.
    assert predicate_options("biolink:Protein", "biolink:Disease") == options
    # gene_associated_with_condition IS legal -- on the variant~gene pair, not gene~disease.
    assert "biolink:gene_associated_with_condition" in (predicate_options("SequenceVariant", "Gene") or set())
    # A pair with no specific association class leaves `predicate` open: nothing to demote.
    assert predicate_options("biolink:OrganismTaxon", "biolink:ChemicalEntity") is None
