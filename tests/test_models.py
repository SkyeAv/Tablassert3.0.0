from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest
from pydantic import ValidationError

import tablassert.models as models
from tablassert.enums import Categories
from tablassert.ingests import from_yaml
from tablassert.models import (
    Annotation,
    Contributor,
    Encoding,
    Excel,
    Graph,
    NodeEncoding,
    Provenance,
    Reindex,
    Section,
    Statement,
    Text,
)


# ? Valid Minimal Text Section
def test_section_from_minimal_yaml(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    section: Section = Section(**data)  # pyright: ignore
    assert section.syntax == "TC3"
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
        method="value",
        encoding="test",
        regex=[{"pattern": r"\s+", "replacement": " "}, {"pattern": r"\.$", "replacement": ""}],
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


# ? Contributor Valid Construction
def test_contributor_valid() -> None:
    c: Contributor = Contributor(kind="curation", name="Test User", date="01 JAN 2025")  # pyright: ignore
    assert c.kind == "curation"
    assert c.name == "Test User"


# ? Contributor With Optional Fields
def test_contributor_with_optionals() -> None:
    c: Contributor = Contributor(  # pyright: ignore
        kind="curation",  # pyright: ignore[reportArgumentType]
        name="Test User",
        date="01 JAN 2025",
        organizations=["Org A"],
        comment="Test comment",  # pyright: ignore
    )
    assert c.organizations == ["Org A"]
    assert c.comment == "Test comment"


# ? Provenance Valid Construction
def test_provenance_valid() -> None:
    p: Provenance = Provenance(  # pyright: ignore
        repo="PMC",  # pyright: ignore[reportArgumentType]
        publication="PMC0000000",
        contributors=[{"kind": "curation", "name": "Test", "date": "01 JAN 2025"}],  # pyright: ignore
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
            provenance={
                "repo": "PMC",
                "publication": "PMC000",
                "contributors": [{"kind": "curation", "name": "T", "date": "2025"}],
            },
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
        local=Path("./test.tsv"),
        url="https://example.com/test.tsv",
        kind="text",
        reindex=[{"column": "A", "comparison": "ne", "comparator": ""}],
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
        provenance={
            "repo": "PMC",
            "publication": "PMC000",
            "contributors": [{"kind": "curation", "name": "T", "date": "2025"}],
        },
    )
    assert len(section.statement.qualifiers) == 1  # pyright: ignore


# ? Section With Annotations
def test_section_with_annotations() -> None:
    section: Section = Section(  # pyright: ignore
        source={"local": "./t.tsv", "url": "https://example.com/t.tsv", "kind": "text"},
        statement={"subject": {"method": "value", "encoding": "A"}, "object": {"method": "value", "encoding": "B"}},
        provenance={
            "repo": "PMC",
            "publication": "PMC000",
            "contributors": [{"kind": "curation", "name": "T", "date": "2025"}],
        },
        annotations=[
            {"annotation": "p value", "method": "column", "encoding": "E"},
            {"annotation": "sample size", "method": "value", "encoding": 28},
        ],
    )
    assert len(section.annotations) == 2  # pyright: ignore


# ? Value Encoding Resolves Against Datassert (Context-Aware Pass)
def test_value_encoding_resolves_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_resolve(_lf: Any, _col: str, _conns: list[object], **_kwargs: Any) -> Any:
        return pl.DataFrame({"resolved": ["YES"]}).lazy()

    monkeypatch.setattr(models, "resolve", fake_resolve)
    section: Section = Section.model_validate(
        {
            "source": {"local": "./t.tsv", "url": "https://example.com/t.tsv", "kind": "text"},
            "statement": {
                "subject": {"method": "value", "encoding": "BRCA1"},
                "object": {"method": "value", "encoding": "TP53"},
            },
            "provenance": {
                "repo": "PMC",
                "publication": "PMC000",
                "contributors": [{"kind": "curation", "name": "T", "date": "2025"}],
            },
        },
        context={"conns": [object()]},
    )
    assert section.statement.subject.encoding == "BRCA1"


# ? Value Encoding Fails To Resolve Raises Code 21
def test_value_encoding_resolves_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_resolve_empty(_lf: Any, _col: str, _conns: list[object], **_kwargs: Any) -> Any:
        return pl.DataFrame({"resolved": []}).lazy()

    monkeypatch.setattr(models, "resolve", fake_resolve_empty)
    with pytest.raises(ValidationError) as exc_info:
        Section.model_validate(
            {
                "source": {"local": "./t.tsv", "url": "https://example.com/t.tsv", "kind": "text"},
                "statement": {
                    "subject": {"method": "value", "encoding": "BRCA1"},
                    "object": {"method": "value", "encoding": "TP53"},
                },
                "provenance": {
                    "repo": "PMC",
                    "publication": "PMC000",
                    "contributors": [{"kind": "curation", "name": "T", "date": "2025"}],
                },
            },
            context={"conns": [object()]},
        )
    assert "21 |" in str(exc_info.value)


# ? Value Encoding Validator Skips Without Context
def test_value_encoding_skips_without_context(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_resolve_empty(_lf: Any, _col: str, _conns: list[object], **_kwargs: Any) -> Any:
        return pl.DataFrame({"resolved": []}).lazy()

    monkeypatch.setattr(models, "resolve", fake_resolve_empty)
    section: Section = Section(  # pyright: ignore
        source={"local": "./t.tsv", "url": "https://example.com/t.tsv", "kind": "text"},
        statement={
            "subject": {"method": "value", "encoding": "BRCA1"},
            "object": {"method": "value", "encoding": "TP53"},
        },
        provenance={
            "repo": "PMC",
            "publication": "PMC000",
            "contributors": [{"kind": "curation", "name": "T", "date": "2025"}],
        },
    )
    assert section.statement.subject.encoding == "BRCA1"


# ? Column Method Encodings Are Not Checked
def test_column_encoding_not_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom_resolve(_lf: Any, _col: str, _conns: list[object], **_kwargs: Any) -> Any:
        raise AssertionError("resolve must not be called for column-method encodings")

    monkeypatch.setattr(models, "resolve", boom_resolve)
    section: Section = Section.model_validate(
        {
            "source": {"local": "./t.tsv", "url": "https://example.com/t.tsv", "kind": "text"},
            "statement": {
                "subject": {"method": "column", "encoding": "A"},
                "object": {"method": "column", "encoding": "B"},
            },
            "provenance": {
                "repo": "PMC",
                "publication": "PMC000",
                "contributors": [{"kind": "curation", "name": "T", "date": "2025"}],
            },
        },
        context={"conns": [object()]},
    )
    assert section.statement.subject.encoding == "A"


# ? Qualifier Value Encoding Is Checked Against Datassert
def test_qualifier_value_encoding_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_resolve(lf: Any, col: str, _conns: list[object], **_kwargs: Any) -> Any:
        term: str = str(lf.collect().get_column(col).to_list()[0])
        if term in ("brca1", "tp53"):
            return pl.DataFrame({"resolved": ["YES"]}).lazy()
        return pl.DataFrame({"resolved": []}).lazy()

    monkeypatch.setattr(models, "resolve", fake_resolve)
    with pytest.raises(ValidationError) as exc_info:
        Section.model_validate(
            {
                "source": {"local": "./t.tsv", "url": "https://example.com/t.tsv", "kind": "text"},
                "statement": {
                    "subject": {"method": "value", "encoding": "BRCA1"},
                    "object": {"method": "value", "encoding": "TP53"},
                    "qualifiers": [
                        {"qualifier": "disease_context_qualifier", "method": "value", "encoding": "ZZZNOTAREALGENE123"}
                    ],
                },
                "provenance": {
                    "repo": "PMC",
                    "publication": "PMC000",
                    "contributors": [{"kind": "curation", "name": "T", "date": "2025"}],
                },
            },
            context={"conns": [object()]},
        )
    assert "21 |" in str(exc_info.value)


# ? Real Value Encoding Resolves Against The Datassert Shards
@pytest.mark.datassert
def test_real_value_encoding_resolves(datassert_dir: Path) -> None:
    from contextlib import ExitStack

    import duckdb

    from tablassert.fullmap import SHARDS

    with ExitStack() as stack:
        conns: list[object] = [
            stack.enter_context(duckdb.connect(datassert_dir / "data" / f"{x}.duckdb", read_only=True))
            for x in range(SHARDS)
        ]
        section: Section = Section.model_validate(
            {
                "source": {"local": "./t.tsv", "url": "https://example.com/t.tsv", "kind": "text"},
                "statement": {
                    "subject": {"method": "value", "encoding": "BRCA1"},
                    "object": {"method": "value", "encoding": "TP53"},
                },
                "provenance": {
                    "repo": "PMC",
                    "publication": "PMC000",
                    "contributors": [{"kind": "curation", "name": "T", "date": "2025"}],
                },
            },
            context={"conns": conns},
        )
        assert section.statement.subject.encoding == "BRCA1"


# ? Real Value Encoding Failure Raises Code 21
@pytest.mark.datassert
def test_real_value_encoding_fails(datassert_dir: Path) -> None:
    from contextlib import ExitStack

    import duckdb

    from tablassert.fullmap import SHARDS

    with ExitStack() as stack:
        conns: list[object] = [
            stack.enter_context(duckdb.connect(datassert_dir / "data" / f"{x}.duckdb", read_only=True))
            for x in range(SHARDS)
        ]
        with pytest.raises(ValidationError) as exc_info:
            Section.model_validate(
                {
                    "source": {"local": "./t.tsv", "url": "https://example.com/t.tsv", "kind": "text"},
                    "statement": {
                        "subject": {"method": "value", "encoding": "ZZZNOTAREALGENE123"},
                        "object": {"method": "value", "encoding": "TP53"},
                    },
                    "provenance": {
                        "repo": "PMC",
                        "publication": "PMC000",
                        "contributors": [{"kind": "curation", "name": "T", "date": "2025"}],
                    },
                },
                context={"conns": conns},
            )
        assert "21 |" in str(exc_info.value)
