from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from tablassert import models
from tablassert.biolink import Categories
from tablassert.enums import Comparisons, EncodingMethods, Repositories
from tablassert.ingests import from_yaml, to_sections
from tablassert.models import (
    DEFAULT_RIG_UI_EXPLANATION,
    Annotation,
    Encoding,
    Excel,
    Graph,
    NodeEncoding,
    Provenance,
    Regex,
    Reindex,
    Section,
    Statement,
    Text,
)

# Repository root (parent of tests/) and the shipped docs examples, used to prove the
# deprecation scaffold stays silent across the real example corpus.
ROOT: Path = Path(__file__).resolve().parent.parent
EXAMPLES: Path = ROOT / "docs" / "examples"


def test_section_from_minimal_yaml(fixtures_path: Path) -> None:
    """valid minimal text section."""
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    section: Section = Section(**data)  # pyright: ignore
    assert section.source.kind == "text"
    assert section.statement.subject.encoding == "BRCA1"
    assert section.statement.object.encoding == "TP53"
    assert section.provenance.repo == "PMC"


def test_graph_rig_defaults() -> None:
    """graph RIG defaults are declared in the model."""
    graph: Graph = Graph(  # pyright: ignore
        name="TEST", version="1.0.0", description="Test graph", tables=[Path("./table.yaml")], fullmap=Path("./fullmap")
    )
    assert graph.contributions == ["Tablassert: KGX and RIG generation"]
    assert graph.ui_explanation == DEFAULT_RIG_UI_EXPLANATION


def test_graph_rejects_removed_enrichment_databases() -> None:
    """graph rejects removed enrichment databases."""
    data: dict[str, Any] = {
        "name": "TEST",
        "version": "1.0.0",
        "description": "Test graph",
        "tables": [Path("./table.yaml")],
        "fullmap": Path("./fullmap"),
        "pubmed_db": Path("./PubMed.db"),
        "pmc_db": Path("./PMCSuppCaptions.db"),
    }
    with pytest.raises(ValidationError):
        Graph.model_validate(data)


def test_graph_rejects_qc_and_log_keys() -> None:
    """graph rejects QC and log keys (moved to `build-graph` CLI flags)."""
    data: dict[str, Any] = {
        "name": "TEST",
        "version": "1.0.0",
        "description": "Test graph",
        "tables": [Path("./table.yaml")],
        "fullmap": Path("./fullmap"),
        "qc": True,
        "log": True,
    }
    with pytest.raises(ValidationError):
        Graph.model_validate(data)


def test_excel_source_valid() -> None:
    """valid minimal excel section."""
    source: Excel = Excel(local=Path("./test.xlsx"), url="https://example.com/test.xlsx", kind="excel", sheet="Sheet1")  # pyright: ignore
    assert source.kind == "excel"
    assert source.sheet == "Sheet1"


def test_text_source_valid() -> None:
    """valid text source."""
    source: Text = Text(local=Path("./test.tsv"), url="https://example.com/test.tsv", kind="text", delimiter="\t")  # pyright: ignore
    assert source.kind == "text"
    assert source.delimiter == "\t"


def test_section_missing_source_raises(fixtures_path: Path) -> None:
    """invalid section missing required source."""
    data: Any = from_yaml(fixtures_path / "invalid_section_missing_source.yaml")
    with pytest.raises(ValidationError):
        Section(**data)


def test_encoding_value_method() -> None:
    """encoding with value method."""
    enc: Encoding = Encoding(method="value", encoding="BRCA1")  # pyright: ignore
    assert enc.method == "value"
    assert enc.encoding == "BRCA1"


def test_encoding_column_method() -> None:
    """encoding with column method."""
    enc: Encoding = Encoding(method="column", encoding="A")  # pyright: ignore
    assert enc.method == "column"
    assert enc.encoding == "A"


def test_encoding_with_optional_fields() -> None:
    """encoding with optional fields."""
    enc: Encoding = Encoding(  # pyright: ignore
        method="value", encoding="test", prefix="PREFIX:", suffix=":SUFFIX", fill="forward", explode_by=";"
    )
    assert enc.prefix == "PREFIX:"
    assert enc.suffix == ":SUFFIX"
    assert enc.fill == "forward"
    assert enc.explode_by == ";"


def test_encoding_with_regex() -> None:
    """encoding with regex."""
    enc: Encoding = Encoding(  # pyright: ignore
        method="value", encoding="test", regex=[{"pattern": r"\s+", "replacement": " "}, {"pattern": r"\.$", "replacement": ""}]
    )
    assert len(enc.regex) == 2  # pyright: ignore
    assert enc.regex[0].pattern == r"\s+"  # pyright: ignore


def test_encoding_with_remove() -> None:
    """encoding with remove patterns."""
    enc: Encoding = Encoding(method="value", encoding="test", remove=[r"\[\d+\]", r"\s+"])  # pyright: ignore
    assert len(enc.remove) == 2  # pyright: ignore


def test_encoding_with_transformations() -> None:
    """encoding with transformations."""
    enc: Encoding = Encoding(  # pyright: ignore
        method="value", encoding=2.0, transformations=[{"function": "pow", "arguments": ["values", 2]}]
    )
    assert len(enc.transformations) == 1  # pyright: ignore
    assert enc.transformations[0].function == "pow"  # pyright: ignore


def test_node_encoding_with_taxon() -> None:
    """NodeEncoding with taxon."""
    node: NodeEncoding = NodeEncoding(method="value", encoding="BRCA1", taxon=9606)  # pyright: ignore
    assert node.taxon == 9606


def test_node_encoding_defaults_to_human_taxon() -> None:
    """NodeEncoding defaults omitted taxon to Homo sapiens."""
    node: NodeEncoding = NodeEncoding(method="value", encoding="BRCA1")  # pyright: ignore
    assert node.taxon == 9606


def test_node_encoding_explicit_null_disables_taxon() -> None:
    """NodeEncoding preserves explicit null taxon to disable taxon filtering."""
    node: NodeEncoding = NodeEncoding(method="value", encoding="BRCA1", taxon=None)  # pyright: ignore
    assert node.taxon is None


def test_qualifier_rejects_species_context_qualifier() -> None:
    """species_context_qualifier is auto-derived and cannot be manually declared."""
    with pytest.raises(ValidationError) as exc_info:
        models.Qualifier(qualifier="species_context_qualifier", method="value", encoding="Homo sapiens")  # pyright: ignore
    assert "qualifier-auto-derived" in str(exc_info.value)


def test_node_encoding_with_prioritize_avoid() -> None:
    """NodeEncoding with prioritize and avoid."""
    node: NodeEncoding = NodeEncoding(  # pyright: ignore
        method="value", encoding="BRCA1", prioritize=[Categories.GENE, Categories.PROTEIN], avoid=[Categories.DISEASE]
    )
    assert len(node.prioritize) == 2  # pyright: ignore
    assert len(node.avoid) == 1  # pyright: ignore


def test_node_encoding_exclude_fields_default_none() -> None:
    """NodeEncoding exclude_prefixes/exclude_regex default to None.

    US-M3: the CURIE-exclusion fields are purely additive; an untouched config must
    keep both as None so the resolution hot path stays byte-for-byte unchanged.
    """
    node: NodeEncoding = NodeEncoding(method="value", encoding="BRCA1")  # pyright: ignore
    assert node.exclude_prefixes is None
    assert node.exclude_regex is None


def test_node_encoding_exclude_regex_invalid_rejected() -> None:
    """Guard: every `exclude_regex` entry must be a polars-compatible regular expression.

    Catches an invalid CURIE-exclusion pattern at config time instead of deep inside a
    multi-hour build; reuses the `regex-bad-pattern` code from the `Regex.pattern` probe.
    """
    with pytest.raises(ValidationError) as exc_info:
        NodeEncoding(method=EncodingMethods.VALUE, encoding="x", exclude_regex=["("])  # pyright: ignore
    assert "regex-bad-pattern" in str(exc_info.value)


def test_node_encoding_exclude_regex_empty_pattern_rejected() -> None:
    """Guard: an empty or whitespace-only `exclude_regex` entry is rejected.

    An empty pattern compiles, but polars ``str.contains("")`` matches EVERY CURIE, so it
    would silently drop all resolution candidates (data loss). Fail loudly at config time
    with the same `regex-bad-pattern` code instead of emptying a build hours in.
    """
    for bad in ("", "   "):
        with pytest.raises(ValidationError) as exc_info:
            NodeEncoding(method=EncodingMethods.VALUE, encoding="x", exclude_regex=[bad])  # pyright: ignore
        assert "regex-bad-pattern" in str(exc_info.value)


def test_node_encoding_exclude_empty_lists_noop() -> None:
    """Empty exclude lists validate and are preserved verbatim.

    Downstream treats ``[]`` identically to ``None`` (a pure no-op), so the model must
    accept and round-trip empty lists rather than coercing or rejecting them.
    """
    node: NodeEncoding = NodeEncoding(  # pyright: ignore
        method="value", encoding="BRCA1", exclude_prefixes=[], exclude_regex=[]
    )
    assert node.exclude_prefixes == []
    assert node.exclude_regex == []


def test_statement_default_predicate() -> None:
    """statement with default predicate."""
    stmt: Statement = Statement(  # pyright: ignore
        subject={"method": "value", "encoding": "A"}, object={"method": "value", "encoding": "B"}
    )
    assert stmt.predicate == "related_to"


def test_statement_explicit_predicate() -> None:
    """statement with explicit predicate."""
    stmt: Statement = Statement(  # pyright: ignore
        subject={"method": "value", "encoding": "A"}, object={"method": "value", "encoding": "B"}, predicate="treats"
    )
    assert stmt.predicate == "treats"


def test_reindex_valid() -> None:
    """reindex valid construction."""
    ri: Reindex = Reindex(column="A", comparison="ne", comparator="")  # pyright: ignore
    assert ri.column == "A"
    assert ri.comparison == "ne"
    assert ri.comparator == ""


def test_reindex_numeric_comparator() -> None:
    """reindex with numeric comparator."""
    ri: Reindex = Reindex(column="B", comparison="gt", comparator=0)  # pyright: ignore
    assert ri.comparator == 0


def test_provenance_valid() -> None:
    """provenance valid construction."""
    p: Provenance = Provenance(  # pyright: ignore
        repo="PMC",  # pyright: ignore[reportArgumentType]
        publication="PMC0000000",
    )
    assert p.repo == "PMC"
    assert p.publication == "PMC0000000"
    assert p.knowledge_level == "statistical_association"
    assert p.agent_type == "data_analysis_pipeline"


def test_provenance_custom_knowledge_level_and_agent_type() -> None:
    """provenance accepts custom KL/AT values."""
    p: Provenance = Provenance(  # pyright: ignore
        repo="PMID",  # pyright: ignore[reportArgumentType]
        publication="12345678",
        knowledge_level="prediction",  # pyright: ignore[reportArgumentType]
        agent_type="computational_model",  # pyright: ignore[reportArgumentType]
    )
    assert p.knowledge_level == "prediction"
    assert p.agent_type == "computational_model"


def test_annotation_valid() -> None:
    """annotation valid construction."""
    a: Annotation = Annotation(annotation="p_value", method="column", encoding="E")  # pyright: ignore
    assert a.annotation == "p_value"
    assert a.encoding == "E"


def test_section_rejects_extra_fields() -> None:
    """section rejects extra fields."""
    with pytest.raises(ValidationError):
        Section(
            source={"local": "./t.tsv", "url": "https://example.com/t.tsv", "kind": "text"},
            statement={"subject": {"method": "value", "encoding": "A"}, "object": {"method": "value", "encoding": "B"}},
            provenance={"repo": "PMC", "publication": "PMC000"},
            unknown_field="bad",  # pyright: ignore
        )


def test_section_with_row_slice() -> None:
    """section with row slice."""
    source: Text = Text(  # pyright: ignore
        local=Path("./test.tsv"), url="https://example.com/test.tsv", kind="text", row_slice=[2, "auto"]
    )
    assert source.row_slice == [2, "auto"]


def test_section_with_rows() -> None:
    """section with rows."""
    source: Text = Text(local=Path("./test.tsv"), url="https://example.com/test.tsv", kind="text", rows=[1, 2, 5])  # pyright: ignore
    assert source.rows == [1, 2, 5]


def test_section_with_reindex() -> None:
    """section with reindex."""
    source: Text = Text(  # pyright: ignore
        local=Path("./test.tsv"), url="https://example.com/test.tsv", kind="text", reindex=[{"column": "A", "comparison": "ne", "comparator": ""}]
    )
    assert len(source.reindex) == 1  # pyright: ignore
    assert source.reindex[0].column == "A"  # pyright: ignore


def test_section_with_qualifiers() -> None:
    """section with qualifiers."""
    section: Section = Section(  # pyright: ignore
        source={"local": "./t.tsv", "url": "https://example.com/t.tsv", "kind": "text"},
        statement={
            "subject": {"method": "value", "encoding": "A"},
            "object": {"method": "value", "encoding": "B"},
            "qualifiers": [{"qualifier": "disease_context_qualifier", "method": "value", "encoding": "MONDO:0005575"}],
        },
        provenance={"repo": "PMC", "publication": "PMC000"},
    )
    assert len(section.statement.qualifiers) == 1  # pyright: ignore


def test_section_with_annotations() -> None:
    """section with annotations."""
    data: dict[str, Any] = {
        "source": {"local": "./t.tsv", "url": "https://example.com/t.tsv", "kind": "text"},
        "statement": {"subject": {"method": "value", "encoding": "A"}, "object": {"method": "value", "encoding": "B"}},
        "provenance": {"repo": "PMC", "publication": "PMC000"},
        "annotations": [
            {"annotation": "p_value", "method": "column", "encoding": "E"},
            {"annotation": "sample_size", "method": "value", "encoding": 28},
        ],
    }
    section: Section = Section.model_validate(data)  # pyright: ignore
    assert len(section.annotations) == 2  # pyright: ignore


def test_reindex_eq_with_non_string_comparator_rejected() -> None:
    """Guard: `eq`/`ne` reindex filters require a str comparator.

    Catches a misconfigured row filter at config time instead of deep inside a multi-hour build.
    """
    with pytest.raises(ValidationError) as exc_info:
        Reindex(column="A", comparison=Comparisons.EQ, comparator=5)
    assert "comparison-bad-comparator-type" in str(exc_info.value)


def test_reindex_ordering_with_non_numeric_comparator_rejected() -> None:
    """Guard: ordering comparisons (`gt`/`ge`/`lt`/`le`) require a numeric comparator.

    Catches a misconfigured row filter at config time instead of deep inside a multi-hour build.
    """
    with pytest.raises(ValidationError) as exc_info:
        Reindex(column="A", comparison=Comparisons.GT, comparator="x")
    assert "comparison-nonnumeric-comparator" in str(exc_info.value)


def test_source_with_both_rows_and_row_slice_rejected() -> None:
    """Guard: a source cannot declare both explicit `rows` and a `row_slice`.

    Catches an ambiguous row-selection config at config time instead of deep inside a multi-hour build.
    """
    with pytest.raises(ValidationError) as exc_info:
        Text(local=Path("./t.tsv"), url="https://example.com/t.tsv", kind="text", rows=[1], row_slice=[1, 5])  # pyright: ignore
    assert "config-rows-and-row-slice-conflict" in str(exc_info.value)


def test_regex_with_invalid_pattern_rejected() -> None:
    """Guard: a regex `pattern` must be a polars-compatible regular expression.

    Catches an invalid replacement pattern at config time instead of deep inside a multi-hour build.
    """
    with pytest.raises(ValidationError) as exc_info:
        Regex(pattern="(", replacement=" ")
    assert "regex-bad-pattern" in str(exc_info.value)


def test_regex_with_invalid_replacement_rejected() -> None:
    """Guard: a regex `replacement` must itself be a polars-compatible regular expression.

    Catches an invalid replacement value at config time instead of deep inside a multi-hour build.
    """
    with pytest.raises(ValidationError) as exc_info:
        Regex(pattern="ok", replacement="(")
    assert "regex-bad-replacement" in str(exc_info.value)


def test_encoding_column_method_with_non_column_encoding_rejected() -> None:
    """Guard: the `column` encoding method requires an Excel-style column name (A-ZZ).

    Catches a mistyped source column reference at config time instead of deep inside a multi-hour build.
    """
    with pytest.raises(ValidationError) as exc_info:
        Encoding(method=EncodingMethods.COLUMN, encoding="not_a_col_123")  # pyright: ignore
    assert "encoding-bad-excel-column" in str(exc_info.value)


def test_encoding_remove_with_invalid_regex_rejected() -> None:
    """Guard: every `remove` entry must be a polars-compatible regular expression.

    Catches an invalid text-cleaning pattern at config time instead of deep inside a multi-hour build.
    """
    with pytest.raises(ValidationError) as exc_info:
        Encoding(method=EncodingMethods.VALUE, encoding="x", remove=["("])  # pyright: ignore
    assert "encoding-bad-remove-entry" in str(exc_info.value)


def test_provenance_pmc_repo_with_non_pmc_publication_rejected() -> None:
    """Guard: a PubMed Central `repo` requires a publication id starting with `PMC`.

    Catches a mismatched provenance namespace at config time instead of deep inside a multi-hour build.
    """
    with pytest.raises(ValidationError) as exc_info:
        Provenance(repo=Repositories.PUBMED_CENTRAL, publication="12345")  # pyright: ignore
    assert "provenance-bad-pmc-id" in str(exc_info.value)


def test_no_deprecation_warnings_for_current_fixtures(fixtures_path: Path, recwarn: pytest.WarningsRecorder) -> None:
    """US-M4: an EMPTY DEPRECATED_KEYS registry makes the before-hook a silent no-op.

    Every shipped fixture and docs example must load and validate with ZERO UserWarnings,
    proving the deprecation scaffold never regresses today's corpus. Scoped to UserWarning so
    unrelated DeprecationWarnings (multiprocessing/polars) cannot interfere with the assertion.
    """
    # Single-section fixture validates directly.
    Section.model_validate(from_yaml(fixtures_path / "minimal_section.yaml"))

    # Template/sections configs expand via to_sections; the stamped `config` path is a
    # Tcode-only key, so drop it before validating the pure Section schema.
    for path in (fixtures_path / "minimal_section_with_sections.yaml", EXAMPLES / "tutorial-table.yaml"):
        raw: Any = from_yaml(path)
        sections: list[dict[str, Any]] = to_sections(raw, path)  # pyright: ignore
        for section in sections:
            section.pop("config", None)
            Section.model_validate(section)

    # Graph example validates directly.
    Graph.model_validate(from_yaml(EXAMPLES / "tutorial-graph.yaml"))

    assert [w for w in recwarn if issubclass(w.category, UserWarning)] == []


def test_deprecated_key_in_registry_warns_but_still_validates(monkeypatch: pytest.MonkeyPatch) -> None:
    """US-M4: a registered deprecated key soft-warns but never blocks validation.

    Monkeypatching a REAL current field name (`delimiter`) into DEPRECATED_KEYS makes the
    before-hook emit a UserWarning while `extra="forbid"` still accepts the key and the value
    round-trips intact -- isolating "warn, don't fail". monkeypatch restores the empty registry
    so no other test observes the temporary entry.
    """
    monkeypatch.setitem(models.DEPRECATED_KEYS, "delimiter", "delimiter is deprecated; use 'sep' instead")
    with pytest.warns(UserWarning, match="sep"):
        source: Text = Text(local=Path("./t.tsv"), url="https://example.com/t.tsv", kind="text", delimiter="\t")  # pyright: ignore
    assert source.delimiter == "\t"


def test_deprecated_hook_is_noop_for_non_dict_input(recwarn: pytest.WarningsRecorder) -> None:
    """US-M4: the before-hook passes non-mapping raw input through untouched.

    pydantic hands a mode="before" validator whatever raw value model_validate receives, so the
    hook must neither warn nor mutate non-dicts (None/list/instance); normal validation still
    rejects the bad shape, proving the scaffold only ever observes dict keys.
    """
    with pytest.raises(ValidationError):
        Section.model_validate(["not", "a", "mapping"])
    assert [w for w in recwarn if issubclass(w.category, UserWarning)] == []
