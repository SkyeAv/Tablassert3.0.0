from __future__ import annotations

from operator import eq
from pathlib import Path
from typing import Any, Literal, Optional, Self, Union

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

if TYPE_CHECKING:
    import httpx
else:
    httpx = Lazy.load("httpx")


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


class BaseSource(TablaBase):
    local: Path = Field(..., description="Local path to read from or download into.")
    url: HttpUrl = Field(..., description="Remote source URL fetched before parsing.")

    @field_validator("url", mode="after")
    def is_real_url(url: HttpUrl, timeout: float = 3.0) -> HttpUrl:
        s: str = str(url)

        try:
            r: Any = httpx.head(s, timeout=timeout, follow_redirects=True)
            r.raise_for_status()
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
    replacement: Union[int, float, str] = Field(
        ..., description="Replacement value used when the pattern matches.", examples=[" ", "", 0]
    )


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
    remove: Optional[list[str]] = Field(
        None,
        description="Regex patterns removed from text (replace with empty string).",
        examples=[["\\[\\d+\\]", "\\s+"]],
    )
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
    contributors: list[Contributor] = Field(..., description="Contributor records embedded in output provenance.")


class Annotation(Encoding):
    annotation: str = Field(
        ..., description="Output column name that receives this encoded annotation.", examples=["p_value", "cohort"]
    )


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
