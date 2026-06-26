from __future__ import annotations

import re
from operator import eq
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Self, Union

import lazy_loader as Lazy
from diskcache import Cache
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, PositiveInt, field_validator, model_validator

from tablassert.enums import (
    Categories,
    Comparisons,
    Contributions,
    EncodingMethods,
    Files,
    FillMethods,
    Functions,
    Predicates,
    Qualifiers,
    Repositories,
    Statuses,
    Syntaxes,
    Tokens,
)

from tablassert.fullmap import resolve
from tablassert.nlp import level_one, level_two

if TYPE_CHECKING:
    import httpx
    import polars as pl
else:
    httpx = Lazy.load("httpx")
    pl = Lazy.load("polars")


class TablaBase(BaseModel):
    model_config: ConfigDict = ConfigDict(  # pyright: ignore
        str_strip_whitespace=False,
        validate_assignment=True,
        use_enum_values=True,
        extra="forbid",
        populate_by_name=True,
    )


class Reindex(TablaBase):
    column: str = Field(
        ..., pattern=r"^[A-Z]{1,3}$", description="Source column letters used for row filtering.", examples=["A", "AA"]
    )
    comparison: Comparisons = Field(
        Comparisons.NE,
        description="Comparison operator used in reindex filtering.",
        examples=[Comparisons.NE, Comparisons.EQ, Comparisons.GT],
    )
    comparator: Union[str, int, float] = Field(
        ..., description="Right-side value compared against the selected column.", examples=["N/A", 0, 1.5]
    )

    @model_validator(mode="after")
    def comparison_datatypes(self: Self) -> Self:
        x: Comparisons = self.comparison
        y: Union[str, int, float] = self.comparator

        if eq(x, Comparisons.NE) or eq(x, Comparisons.EQ):
            if not isinstance(y, str):
                msg: str = f"14 | eq or ne comparisons must have a str comparator, got {type(y)}"
                raise ValueError(msg)
        else:
            if not (isinstance(y, int) or isinstance(y, float)):
                msg = f"15 | all comparisons other than eq or ne must have a float or an int comparator, got {type(y)}"
                raise ValueError(msg)

        return self


CACHE: Path = Path(".cachassert/")
CACHE.mkdir(parents=True, exist_ok=True)

URL_CACHE: Cache = Cache(CACHE)


class BaseSource(TablaBase):
    local: Path = Field(..., description="Local path to read from or download into.")
    url: HttpUrl = Field(..., description="Remote source URL fetched before parsing.")

    @field_validator("url", mode="after")
    @classmethod
    def is_real_url(cls, url: HttpUrl) -> HttpUrl:
        s: str = str(url)

        @URL_CACHE.memoize()
        def check_url(s: str, timeout: float = 15.0) -> None:
            r: Any = httpx.head(s, timeout=timeout, follow_redirects=True)
            if 400 <= r.status_code < 500 and r.status_code != 403:
                r.raise_for_status()

        try:
            check_url(s)
        except Exception as e:
            msg: str = f"12 | not a real url {s} | {e}"
            raise ValueError(msg)

        return url

    rows: Optional[list[PositiveInt]] = Field(
        None, description="Zero-based row indices kept after any row_slice crop.", examples=[[0, 2, 5]]
    )
    row_slice: Optional[list[Union[PositiveInt, Literal[Tokens.AUTO]]]] = Field(
        None,
        description="Two-value row bounds [start, stop]; each value can be an index or 'auto'.",
        examples=[[1, 50], [Tokens.AUTO, 100], [5, Tokens.AUTO]],
    )

    @model_validator(mode="after")
    def no_rows_and_slice(self: Self) -> Self:
        if self.rows and self.row_slice:
            msg: str = "13 | cannot specify rows and row_slice in the same section"
            raise ValueError(msg)

        return self

    reindex: Optional[list[Reindex]] = Field(
        None,
        description="Sequential row filters applied using source column values.",
        examples=[[{"column": "A", "comparison": Comparisons.NE, "comparator": ""}]],
    )


class Excel(BaseSource):
    kind: Literal[Files.EXCEL] = Field(Files.EXCEL, description="Source kind; must be 'excel'.")
    sheet: Optional[str] = Field("Sheet1", description="Worksheet name to read from the workbook.")


class Text(BaseSource):
    kind: Literal[Files.TEXT] = Field(Files.TEXT, description="Source kind; must be 'text'.")
    delimiter: Optional[str] = Field(
        ",", description="Field delimiter for headerless text/CSV scanning.", examples=[",", "\t", "|"]
    )


class Regex(TablaBase):
    pattern: Union[int, float, str] = Field(
        ..., description="Regex pattern passed to string replacement.", examples=["\\s+", "\\.$"]
    )

    @field_validator("pattern", mode="after")
    @classmethod
    def polars_compatible_pattern(cls, pattern: Union[int, float, str]) -> Union[int, float, str]:
        try:
            pl.Series([""]).str.contains(str(pattern))
        except Exception as e:
            msg: str = f"17 | pattern must be a polars compatible regex, got {pattern} | {e}"
            raise ValueError(msg)

        return pattern

    replacement: Union[int, float, str] = Field(
        ..., description="Replacement value used when the pattern matches.", examples=[" ", "", 0]
    )

    @field_validator("replacement", mode="after")
    @classmethod
    def polars_compatible_replacement(cls, replacement: Union[int, float, str]) -> Union[int, float, str]:
        try:
            pl.Series([""]).str.contains(str(replacement))
        except Exception as e:
            msg: str = f"18 | replacement must be a polars compatible regex, got {replacement} | {e}"
            raise ValueError(msg)

        return replacement


class Math(TablaBase):
    function: Functions = Field(..., description="Math function applied during numeric transformation.")
    arguments: list[Union[Literal[Tokens.VALUES], float, int]] = Field(
        ...,
        description="Function arguments; use 'values' to inject the current value.",
        examples=[[Tokens.VALUES, 2], [-1, Tokens.VALUES]],
    )


class Encoding(TablaBase):
    method: EncodingMethods = Field(
        EncodingMethods.VALUE,
        description="Interpret encoding as a literal value or as source column letters.",
        examples=[EncodingMethods.VALUE, EncodingMethods.COLUMN],
    )
    encoding: Union[str, int, float] = Field(
        ..., description="Literal value or source column letters, depending on method.", examples=["A", "BRCA1", 1.0]
    )

    @model_validator(mode="after")
    def excel_style_columns(self: Self) -> Self:
        if eq(self.method, EncodingMethods.COLUMN):
            x: Union[str, int, float] = self.encoding
            if not re.search(r"^[A-Z]{1,3}$", str(x)):
                msg: str = f"16 | encoding must be an excel style alphanumeric column name like A to ZZ, got {x}"
                raise ValueError(msg)

        return self

    regex: Optional[list[Regex]] = Field(
        None,
        description="Ordered regex replacements applied to encoded text.",
        examples=[[{"pattern": "\\s+", "replacement": " "}, {"pattern": "\\.$", "replacement": ""}]],
    )
    fill: Optional[FillMethods] = Field(
        None,
        description="Null fill strategy applied after value extraction.",
        examples=[FillMethods.FORWARD, FillMethods.ZERO],
    )
    remove: Optional[list[Union[int, float, str]]] = Field(
        None,
        description="Regex patterns removed from text (replace with empty string).",
        examples=[["\\[\\d+\\]", "\\s+"]],
    )

    @field_validator("remove", mode="after")
    @classmethod
    def polars_compatible_replacement(
        cls, remove: Optional[list[Union[int, float, str]]]
    ) -> Optional[list[Union[int, float, str]]]:
        if remove:
            for r in remove:
                try:
                    pl.Series([""]).str.contains(str(r))
                except Exception as e:
                    msg: str = f"19 | remove must be contain polars compatible regular expressions, got {r} | {e}"
                    raise ValueError(msg)

        return remove

    prefix: Optional[str] = Field(None, description="String prepended to the encoded value.")
    suffix: Optional[str] = Field(None, description="String appended to the encoded value.")
    explode_by: Optional[str] = Field(
        None, description="Delimiter used to split a value into multiple rows.", examples=[";", "|"]
    )
    transformations: Optional[list[Math]] = Field(
        None,
        description="Ordered math operations applied to numeric values.",
        examples=[[{"function": Functions.POW, "arguments": [Tokens.VALUES, 2]}]],
    )


class NodeEncoding(Encoding):
    taxon: Optional[PositiveInt] = Field(
        None, description="NCBI taxon id used to constrain gene-oriented mapping.", examples=[9606, 10090]
    )
    prioritize: Optional[list[Categories]] = Field(
        None,
        description="Biolink categories ranked higher during entity resolution.",
        examples=[[Categories.GENE, Categories.PROTEIN]],
    )
    avoid: Optional[list[Categories]] = Field(
        None,
        description="Biolink categories excluded during entity resolution.",
        examples=[[Categories.DISEASE, Categories.PHENOTYPIC_FEATURE]],
    )


class Qualifier(NodeEncoding):
    qualifier: Qualifiers = Field(
        ...,
        description="Qualifier predicate key used as the output qualifier column.",
        examples=[Qualifiers.OBJECT_DIRECTION_QUALIFIER, Qualifiers.SUBJECT_CONTEXT_QUALIFIER],
    )


class Statement(TablaBase):
    subject: NodeEncoding = Field(..., description="Subject node encoding and mapping configuration.")
    object: NodeEncoding = Field(..., description="Object node encoding and mapping configuration.")
    predicate: Predicates = Field(Predicates.RELATED_TO, description="Predicate connecting subject and object nodes.")
    qualifiers: Optional[list[Qualifier]] = Field(
        None, description="Optional qualifier nodes attached to the statement."
    )


class Contributor(TablaBase):
    kind: Contributions = Field(Contributions.CURATION, description="Contributor role for this provenance entry.")
    name: str = Field(..., description="Contributor display name.")
    date: str = Field(..., description="Contribution date string preserved as provided.")
    organizations: Optional[list[str]] = Field(None, description="Optional affiliated organizations.")
    comment: Optional[str] = Field(None, description="Optional free-text contributor note.")


class Provenance(TablaBase):
    repo: Repositories = Field(Repositories.PUBMED_CENTRAL, description="Publication identifier namespace prefix.")
    publication: str = Field(
        ...,
        description="Repository-local publication id appended as repo:publication.",
        examples=["12345678", "PMC1234567"],
    )

    @model_validator(mode="after")
    def is_valid_pmc_id(self: Self) -> Self:
        if eq(self.repo, Repositories.PUBMED_CENTRAL):
            publication: str = self.publication
            if not re.search(r"^PMC\d+", publication):
                msg: str = f"20 | pubmed central publications must start with PMC, got {publication}"
                raise ValueError(msg)

        return self

    contributors: list[Contributor] = Field(..., description="Contributor records embedded in output provenance.")


class Annotation(Encoding):
    annotation: str = Field(
        ..., description="Output column name that receives this encoded annotation.", examples=["p_value", "cohort"]
    )

    @field_validator("annotation", mode="after")
    @classmethod
    def clean_annotation(cls, annotation: str) -> str:
        return annotation.replace("_", " ").strip()


def resolves_value_encodings(statement: Statement, conns: list[object]) -> None:
    # ? Resolve Every Value-Method Literal Against The Shared Datassert Shards
    nodes: list[tuple[str, NodeEncoding]] = [("subject", statement.subject), ("object", statement.object)]
    if statement.qualifiers:
        nodes += [(q.qualifier, q) for q in statement.qualifiers]

    for label, node in nodes:
        if not eq(node.method, EncodingMethods.VALUE):
            continue

        term: str = str(node.encoding)
        lf: pl.LazyFrame = pl.DataFrame({"term": [term]}).lazy()
        lf = level_one(lf, "term")
        lf = level_two(lf, "term")
        resolved: pl.DataFrame = resolve(
            lf,
            "term",
            conns,
            taxon=str(node.taxon) if node.taxon else None,
            prioritize=node.prioritize,
            avoid=node.avoid,
            log=False,
            column_context=False,
        ).collect()
        if resolved.height == 0:
            msg: str = f"21 | value encoding {term!r} in {label!r} did not resolve against datassert"
            raise ValueError(msg)


class Section(TablaBase):
    # ? Pydantic "Section" Model And Coercion
    syntax: Syntaxes = Field(Syntaxes.TC3, description="Section configuration syntax version.")
    status: Statuses = Field(Statuses.ALPHA, description="Section maturity label for configuration tracking.")
    source: Union[Excel, Text] = Field(..., description="Input source definition for reading tabular rows.")
    statement: Statement = Field(..., description="Subject-object statement mapping for this section.")
    provenance: Provenance = Field(..., description="Provenance metadata applied to all produced edges.")
    annotations: Optional[list[Annotation]] = Field(
        None, description="Optional extra encoded columns added to each row."
    )

    @field_validator("statement", mode="after")
    @classmethod
    def value_encodings_resolve(cls, statement: Statement, info: Any) -> Statement:
        # ? Ensure Value-Method Encodings Resolve Against The Shared Datassert Shards
        conns: Optional[list[object]] = info.context.get("conns") if info.context else None
        if conns is None:
            return statement  # * skip without shared connections (contextless path)
        resolves_value_encodings(statement, conns)
        return statement


class Graph(TablaBase):
    # ? Pydantic "Graph" Configuration
    syntax: Syntaxes = Field(Syntaxes.GC2, description="Graph configuration syntax version.")
    name: str = Field(..., description="Graph name written into output metadata.")
    version: str = Field(..., description="Graph version label.")
    log: bool = Field(False, description="Whether to log unmatched entities and audit details during graph builds.")
    qc: bool = Field(False, description="Whether to run the QC audit stage during graph builds.")
    tables: list[Path] = Field(
        ..., description="Paths to table YAML files included in this graph.", examples=[["tables/tutorial-table.yaml"]]
    )
    datassert: Path = Field(
        ..., description="Base datassert directory containing data shard DuckDB files.", examples=[".datassert"]
    )
    pubmed_db: Optional[Path] = Field(
        None, description="Optional PubMed sqlite database for MeSH enrichment.", examples=[".datassert/pubmed.sqlite"]
    )
    pmc_db: Optional[Path] = Field(
        None, description="Optional PMC sqlite database for caption enrichment.", examples=[".datassert/pmc.sqlite"]
    )
