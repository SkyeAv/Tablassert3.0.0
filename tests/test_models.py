from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from tablassert.enums import Categories
from tablassert.ingests import from_yaml
from tablassert.models import Annotation, Encoding, Excel, Graph, NodeEncoding, Provenance, Reindex, Section, Statement, Text


# ? Valid Minimal Text Section
def test_section_from_minimal_yaml(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    section: Section = Section(**data)  # pyright: ignore
    assert section.syntax == "TC4"
    assert section.source.kind == "text"
    assert section.statement.subject.encoding == "BRCA1"
    assert section.statement.object.encoding == "TP53"
    assert section.provenance.repo == "PMC"


# ? Graph QC Defaults To False
def test_graph_qc_defaults_false() -> None:
    graph: Graph = Graph(  # pyright: ignore
        name="TEST", version="1.0.0", tables=[Path("./table.yaml")], datassert=Path("./datassert")
    )
    assert graph.qc is False


# ? Graph Accepts Explicit QC True
def test_graph_qc_true() -> None:
    graph: Graph = Graph(  # pyright: ignore
        name="TEST", version="1.0.0", qc=True, tables=[Path("./table.yaml")], datassert=Path("./datassert")
    )
    assert graph.qc is True


# ? Valid Minimal Excel Section
def test_excel_source_valid() -> None:
    source: Excel = Excel(local=Path("./test.xlsx"), url="https://example.com/test.xlsx", kind="excel", sheet="Sheet1")  # pyright: ignore
    assert source.kind == "excel"
    assert source.sheet == "Sheet1"


# ? Valid Text Source
def test_text_source_valid() -> None:
    source: Text = Text(local=Path("./test.tsv"), url="https://example.com/test.tsv", kind="text", delimiter="\t")  # pyright: ignore
    assert source.kind == "text"
    assert source.delimiter == "\t"


# ? Invalid Section Missing Required Source
def test_section_missing_source_raises(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "invalid_section_missing_source.yaml")
    with pytest.raises(ValidationError):
        Section(**data)


# ? Encoding With Value Method
def test_encoding_value_method() -> None:
    enc: Encoding = Encoding(method="value", encoding="BRCA1")  # pyright: ignore
    assert enc.method == "value"
    assert enc.encoding == "BRCA1"


# ? Encoding With Column Method
def test_encoding_column_method() -> None:
    enc: Encoding = Encoding(method="column", encoding="A")  # pyright: ignore
    assert enc.method == "column"
    assert enc.encoding == "A"


# ? Encoding With Optional Fields
def test_encoding_with_optional_fields() -> None:
    enc: Encoding = Encoding(  # pyright: ignore
        method="value", encoding="test", prefix="PREFIX:", suffix=":SUFFIX", fill="forward", explode_by=";"
    )
    assert enc.prefix == "PREFIX:"
    assert enc.suffix == ":SUFFIX"
    assert enc.fill == "forward"
    assert enc.explode_by == ";"


# ? Encoding With Regex
def test_encoding_with_regex() -> None:
    enc: Encoding = Encoding(  # pyright: ignore
        method="value", encoding="test", regex=[{"pattern": r"\s+", "replacement": " "}, {"pattern": r"\.$", "replacement": ""}]
    )
    assert len(enc.regex) == 2  # pyright: ignore
    assert enc.regex[0].pattern == r"\s+"  # pyright: ignore


# ? Encoding With Remove Patterns
def test_encoding_with_remove() -> None:
    enc: Encoding = Encoding(method="value", encoding="test", remove=[r"\[\d+\]", r"\s+"])  # pyright: ignore
    assert len(enc.remove) == 2  # pyright: ignore


# ? Encoding With Transformations
def test_encoding_with_transformations() -> None:
    enc: Encoding = Encoding(  # pyright: ignore
        method="value", encoding=2.0, transformations=[{"function": "pow", "arguments": ["values", 2]}]
    )
    assert len(enc.transformations) == 1  # pyright: ignore
    assert enc.transformations[0].function == "pow"  # pyright: ignore


# ? NodeEncoding With Taxon
def test_node_encoding_with_taxon() -> None:
    node: NodeEncoding = NodeEncoding(method="value", encoding="BRCA1", taxon=9606)  # pyright: ignore
    assert node.taxon == 9606


# ? NodeEncoding With Prioritize And Avoid
def test_node_encoding_with_prioritize_avoid() -> None:
    node: NodeEncoding = NodeEncoding(  # pyright: ignore
        method="value", encoding="BRCA1", prioritize=[Categories.GENE, Categories.PROTEIN], avoid=[Categories.DISEASE]
    )
    assert len(node.prioritize) == 2  # pyright: ignore
    assert len(node.avoid) == 1  # pyright: ignore


# ? Statement With Default Predicate
def test_statement_default_predicate() -> None:
    stmt: Statement = Statement(  # pyright: ignore
        subject={"method": "value", "encoding": "A"}, object={"method": "value", "encoding": "B"}
    )
    assert stmt.predicate == "related_to"


# ? Statement With Explicit Predicate
def test_statement_explicit_predicate() -> None:
    stmt: Statement = Statement(  # pyright: ignore
        subject={"method": "value", "encoding": "A"}, object={"method": "value", "encoding": "B"}, predicate="treats"
    )
    assert stmt.predicate == "treats"


# ? Reindex Valid Construction
def test_reindex_valid() -> None:
    ri: Reindex = Reindex(column="A", comparison="ne", comparator="")  # pyright: ignore
    assert ri.column == "A"
    assert ri.comparison == "ne"
    assert ri.comparator == ""


# ? Reindex With Numeric Comparator
def test_reindex_numeric_comparator() -> None:
    ri: Reindex = Reindex(column="B", comparison="gt", comparator=0)  # pyright: ignore
    assert ri.comparator == 0


# ? Provenance Valid Construction
def test_provenance_valid() -> None:
    p: Provenance = Provenance(  # pyright: ignore
        repo="PMC",  # pyright: ignore[reportArgumentType]
        publication="PMC0000000",
    )
    assert p.repo == "PMC"
    assert p.publication == "PMC0000000"


# ? Annotation Valid Construction
def test_annotation_valid() -> None:
    a: Annotation = Annotation(annotation="p value", method="column", encoding="E")  # pyright: ignore
    assert a.annotation == "p value"
    assert a.encoding == "E"


# ? Section Rejects Extra Fields
def test_section_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Section(
            source={"local": "./t.tsv", "url": "https://example.com/t.tsv", "kind": "text"},
            statement={"subject": {"method": "value", "encoding": "A"}, "object": {"method": "value", "encoding": "B"}},
            provenance={"repo": "PMC", "publication": "PMC000"},
            unknown_field="bad",  # pyright: ignore
        )


# ? Section With Row Slice
def test_section_with_row_slice() -> None:
    source: Text = Text(  # pyright: ignore
        local=Path("./test.tsv"), url="https://example.com/test.tsv", kind="text", row_slice=[2, "auto"]
    )
    assert source.row_slice == [2, "auto"]


# ? Section With Rows
def test_section_with_rows() -> None:
    source: Text = Text(local=Path("./test.tsv"), url="https://example.com/test.tsv", kind="text", rows=[1, 2, 5])  # pyright: ignore
    assert source.rows == [1, 2, 5]


# ? Section With Reindex
def test_section_with_reindex() -> None:
    source: Text = Text(  # pyright: ignore
        local=Path("./test.tsv"), url="https://example.com/test.tsv", kind="text", reindex=[{"column": "A", "comparison": "ne", "comparator": ""}]
    )
    assert len(source.reindex) == 1  # pyright: ignore
    assert source.reindex[0].column == "A"  # pyright: ignore


# ? Section With Qualifiers
def test_section_with_qualifiers() -> None:
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


# ? Section With Annotations
def test_section_with_annotations() -> None:
    section: Section = Section(  # pyright: ignore
        source={"local": "./t.tsv", "url": "https://example.com/t.tsv", "kind": "text"},
        statement={"subject": {"method": "value", "encoding": "A"}, "object": {"method": "value", "encoding": "B"}},
        provenance={"repo": "PMC", "publication": "PMC000"},
        annotations=[
            {"annotation": "p value", "method": "column", "encoding": "E"},
            {"annotation": "sample size", "method": "value", "encoding": 28},
        ],
    )
    assert len(section.annotations) == 2  # pyright: ignore
