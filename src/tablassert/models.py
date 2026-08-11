from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, NonNegativeInt, PositiveInt, field_validator, model_validator

from tablassert._lazy import LazyModule
from tablassert.biolink import (
    BIOLINK_VERSION,
    ENUM_RANGED_QUALIFIERS,
    UNSATISFIABLE_EDGE_FIELDS,
    AgentTypes,
    Categories,
    KnowledgeLevels,
    Predicates,
    Qualifiers,
)
from tablassert.enums import Comparisons, EncodingMethods, Files, FillMethods, Functions, Repositories, Tokens
from tablassert.errors import TablassertErrorCodes, TablassertValidationError

if TYPE_CHECKING:
    import polars as pl
else:
    pl = LazyModule("polars")


# Deprecated config keys -> guidance. EMPTY today: the before-hook below is a silent no-op until a
# future rename registers a key here to WARN on it. The hook only warns and returns the data
# unchanged — a renamed key that is no longer a valid field is STILL rejected by extra="forbid"
# unless the future hook also pops/translates it; registering a key here supplies the warning half only.
DEPRECATED_KEYS: dict[str, str] = {}


class TablaBase(BaseModel):
    model_config: ConfigDict = ConfigDict(  # pyright: ignore
        str_strip_whitespace=False, validate_assignment=True, use_enum_values=True, extra="forbid", populate_by_name=True
    )

    @model_validator(mode="before")
    @classmethod
    def _warn_deprecated_keys(cls, data: Any) -> Any:
        """Soft-warn when a raw input key is registered in ``DEPRECATED_KEYS``.

        Pure observation: returns ``data`` unchanged and never rejects (``extra="forbid"``
        still governs truly-unknown keys). The registry is EMPTY today, so this is a silent
        no-op; a future field rename registers its old key here to warn instead of hard-fail.
        Note: with ``validate_assignment=True`` pydantic also re-runs this on attribute set,
        so a registered key warns on assignment too — inert while the registry is empty.
        """
        if isinstance(data, dict):
            for key in data:
                if key in DEPRECATED_KEYS:
                    warnings.warn(DEPRECATED_KEYS[key], UserWarning, stacklevel=2)
        return data


class Reindex(TablaBase):
    column: str = Field(..., pattern=r"^[A-Z]{1,3}$", description="Source column letters used for row filtering.", examples=["A", "AA"])
    comparison: Comparisons = Field(
        Comparisons.NE, description="Comparison operator used in reindex filtering.", examples=[Comparisons.NE, Comparisons.EQ, Comparisons.GT]
    )
    comparator: str | int | float = Field(..., description="Right-side value compared against the selected column.", examples=["N/A", 0, 1.5])

    @model_validator(mode="after")
    def comparison_datatypes(self: Self) -> Self:
        x: Comparisons = self.comparison
        y: str | int | float = self.comparator

        if x == Comparisons.NE or x == Comparisons.EQ:
            if not isinstance(y, str):
                raise TablassertValidationError(
                    f"`eq`/`ne` comparisons require a str comparator, got {type(y).__name__}.", code="comparison-bad-comparator-type"
                )
        else:
            if not isinstance(y, (int, float)):
                raise TablassertValidationError(
                    f"Comparisons other than `eq`/`ne` require a float or int comparator, got {type(y).__name__}.",
                    code="comparison-nonnumeric-comparator",
                )

        return self


class BaseSource(TablaBase):
    local: Path = Field(..., description="Local path to read from or download into.")
    url: list[HttpUrl] = Field(
        ...,
        min_length=1,
        description="One or more remote source URL(s) recorded as provenance; emitted in the edge `sources` list under the primary entry's `source_record_urls` list and in the RIG. Format-validated only; not fetched.",
    )

    rows: list[NonNegativeInt] | None = Field(None, description="Zero-based row indices kept after any row_slice crop.", examples=[[0, 2, 5]])
    row_slice: list[NonNegativeInt | Literal[Tokens.AUTO]] | None = Field(
        None,
        description="Two-value row bounds [start, stop]; each value can be an index or 'auto'.",
        examples=[[1, 50], [Tokens.AUTO, 100], [5, Tokens.AUTO]],
    )

    @model_validator(mode="after")
    def no_rows_and_slice(self: Self) -> Self:
        if self.rows and self.row_slice:
            raise TablassertValidationError(
                "Cannot specify both `rows` and `row_slice` in the same section.", code="config-rows-and-row-slice-conflict"
            )

        return self

    reindex: list[Reindex] | None = Field(
        None,
        description="Sequential row filters applied using source column values.",
        examples=[[{"column": "A", "comparison": Comparisons.NE, "comparator": ""}]],
    )


class Excel(BaseSource):
    kind: Literal[Files.EXCEL] = Field(Files.EXCEL, description="Source kind; must be 'excel'.")
    sheet: str | None = Field("Sheet1", description="Worksheet name to read from the workbook.")


class Text(BaseSource):
    kind: Literal[Files.TEXT] = Field(Files.TEXT, description="Source kind; must be 'text'.")
    delimiter: str | None = Field(",", description="Field delimiter for headerless text/CSV scanning.", examples=[",", "\t", "|"])


class Regex(TablaBase):
    pattern: int | float | str = Field(..., description="Regex pattern passed to string replacement.", examples=["\\s+", "\\.$"])

    @field_validator("pattern", mode="after")
    @classmethod
    def polars_compatible_pattern(cls, pattern: int | float | str) -> int | float | str:
        try:
            pl.Series([""]).str.contains(str(pattern))
        except Exception as e:
            raise TablassertValidationError(f"`pattern` must be a polars-compatible regex, got {pattern!r}: {e}", code="regex-bad-pattern") from e

        return pattern

    replacement: int | float | str = Field(..., description="Replacement value used when the pattern matches.", examples=[" ", "", 0])

    @field_validator("replacement", mode="after")
    @classmethod
    def polars_compatible_replacement(cls, replacement: int | float | str) -> int | float | str:
        try:
            pl.Series([""]).str.contains(str(replacement))
        except Exception as e:
            raise TablassertValidationError(
                f"`replacement` must be a polars-compatible regex, got {replacement!r}: {e}", code="regex-bad-replacement"
            ) from e

        return replacement


class Math(TablaBase):
    function: Functions = Field(..., description="Math function applied during numeric transformation.")
    arguments: list[Literal[Tokens.VALUES] | float | int] = Field(
        ..., description="Function arguments; use 'values' to inject the current value.", examples=[[Tokens.VALUES, 2], [-1, Tokens.VALUES]]
    )


class Encoding(TablaBase):
    method: EncodingMethods = Field(
        EncodingMethods.VALUE,
        description="Interpret encoding as a literal value or as source column letters.",
        examples=[EncodingMethods.VALUE, EncodingMethods.COLUMN],
    )
    encoding: str | int | float = Field(..., description="Literal value or source column letters, depending on method.", examples=["A", "BRCA1", 1.0])

    @model_validator(mode="after")
    def excel_style_columns(self: Self) -> Self:
        if self.method == EncodingMethods.COLUMN:
            x: str | int | float = self.encoding
            if not re.search(r"^[A-Z]{1,3}$", str(x)):
                raise TablassertValidationError(f"`encoding` must be an Excel-style column name (A-ZZ), got {x!r}.", code="encoding-bad-excel-column")

        return self

    regex: list[Regex] | None = Field(
        None,
        description="Ordered regex replacements applied to encoded text.",
        examples=[[{"pattern": "\\s+", "replacement": " "}, {"pattern": "\\.$", "replacement": ""}]],
    )
    fill: FillMethods | None = Field(
        None, description="Null fill strategy applied after value extraction.", examples=[FillMethods.FORWARD, FillMethods.ZERO]
    )
    remove: list[int | float | str] | None = Field(
        None, description="Regex patterns removed from text (replace with empty string).", examples=[["\\[\\d+\\]", "\\s+"]]
    )

    @field_validator("remove", mode="after")
    @classmethod
    def polars_compatible_replacement(cls, remove: list[int | float | str] | None) -> list[int | float | str] | None:
        if remove:
            for r in remove:
                try:
                    pl.Series([""]).str.contains(str(r))
                except Exception as e:
                    raise TablassertValidationError(
                        f"`remove` entries must be polars-compatible regular expressions, got {r!r}: {e}", code="encoding-bad-remove-entry"
                    ) from e

        return remove

    prefix: str | None = Field(None, description="String prepended to the encoded value.")
    suffix: str | None = Field(None, description="String appended to the encoded value.")
    explode_by: str | None = Field(None, description="Delimiter used to split a value into multiple rows.", examples=[";", "|"])
    transformations: list[Math] | None = Field(
        None,
        description="Ordered math operations applied to numeric values.",
        examples=[[{"function": Functions.POW, "arguments": [Tokens.VALUES, 2]}]],
    )


class NodeEncoding(Encoding):
    taxon: PositiveInt | None = Field(
        9606,
        description=(
            "NCBI taxon id used to constrain taxon-scoped entity resolution. Defaults to 9606 (Homo sapiens); "
            "set null to disable. Applies to all taxon-bearing categories (genes, diseases, phenotypes, proteins, etc.)."
        ),
        examples=[9606, 10090],
    )
    prioritize: list[Categories] | None = Field(
        None, description="Biolink categories ranked higher during entity resolution.", examples=[[Categories.GENE, Categories.PROTEIN]]
    )
    avoid: list[Categories] | None = Field(
        None, description="Biolink categories excluded during entity resolution.", examples=[[Categories.DISEASE, Categories.PHENOTYPIC_FEATURE]]
    )
    exclude_prefixes: list[str] | None = Field(
        None, description="CURIE namespace prefixes (text before the first ':') dropped during entity resolution.", examples=[["OMIM", "NCBIGene"]]
    )
    exclude_regex: list[str] | None = Field(
        None,
        description="Regex patterns; any resolved CURIE matching one is dropped during entity resolution (case-sensitive).",
        examples=[["^OMIM:\\d+$"]],
    )

    @field_validator("exclude_regex", mode="after")
    @classmethod
    def polars_compatible_exclude_regex(cls, exclude_regex: list[str] | None) -> list[str] | None:
        if exclude_regex:
            for pattern in exclude_regex:
                # An empty pattern compiles but str.contains("") matches EVERY CURIE, silently dropping
                # all candidates; reject empty/whitespace-only entries loudly at config time.
                if not str(pattern).strip():
                    raise TablassertValidationError(
                        f"`exclude_regex` entries must be non-empty patterns (an empty pattern matches every CURIE), got {pattern!r}.",
                        code="regex-bad-pattern",
                    )
                try:
                    pl.Series([""]).str.contains(str(pattern))
                except Exception as e:
                    raise TablassertValidationError(
                        f"`exclude_regex` entries must be polars-compatible regular expressions, got {pattern!r}: {e}", code="regex-bad-pattern"
                    ) from e

        return exclude_regex


class Qualifier(NodeEncoding):
    qualifier: Qualifiers = Field(
        ...,
        description="Qualifier predicate key used as the output qualifier column.",
        examples=[Qualifiers.OBJECT_DIRECTION_QUALIFIER, Qualifiers.SUBJECT_CONTEXT_QUALIFIER],
    )

    @property
    def vocabulary(self: Self) -> frozenset[str] | None:
        """Closed value set for this qualifier, or ``None`` when it is CURIE-ranged."""
        return ENUM_RANGED_QUALIFIERS.get(str(self.qualifier))

    @property
    def resolved(self: Self) -> bool:
        """Whether this qualifier's values go through fullmap entity resolution.

        Only CURIE-ranged qualifiers are resolved. Enum-ranged ones carry a literal
        token from a closed Biolink vocabulary and must be passed through verbatim.
        """
        return self.vocabulary is None

    @model_validator(mode="after")
    def reject_auto_derived_qualifiers(self: Self) -> Self:
        """Reject qualifiers that Tablassert derives from resolved node metadata.

        ``species_context_qualifier`` is populated automatically from resolved
        subject/object taxon, so declaring it manually would make fullmap treat
        it as an independently resolved query column and risk conflicting output.
        """
        if self.qualifier == "species_context_qualifier":
            raise TablassertValidationError(
                "species_context_qualifier is auto-derived from resolved subject/object taxon; remove it from qualifiers.",
                code="qualifier-auto-derived",
            )
        return self

    @model_validator(mode="after")
    def reject_unusable_qualifiers(self: Self) -> Self:
        """Reject qualifier slots that no Biolink Pydantic class can hold.

        ``Qualifiers`` is derived from the LinkML *slot* hierarchy, which is strictly
        broader than the set of slots actually attached to a class. Emitting one of
        these produces an edge that can never validate, so fail at config time with a
        pointer rather than silently at ingest time.
        """
        if str(self.qualifier) in UNSATISFIABLE_EDGE_FIELDS:
            raise TablassertValidationError(
                f"{self.qualifier} is declared in the Biolink schema but attached to no association class "
                f"in biolink-model {BIOLINK_VERSION}, so it cannot be emitted on an edge. "
                "Use a concrete subtype of it, or record the value as an annotation.",
                code="qualifier-unsatisfiable",
            )
        return self

    @model_validator(mode="after")
    def enum_ranged_values_are_literals(self: Self) -> Self:
        """Validate literal values for enum-ranged qualifiers against their vocabulary.

        Qualifiers inherit :class:`NodeEncoding` and are therefore entity-resolved
        through the fullmap by default. That is right for CURIE-ranged qualifiers
        (``anatomical_context_qualifier`` -> ``UBERON:0001557``) and wrong for
        enum-ranged ones: ``object_direction_qualifier`` wants the token ``increased``,
        not the resolved CURIE ``UMLS:C0205217``.
        """
        vocabulary: frozenset[str] | None = self.vocabulary
        if vocabulary is None or self.method != EncodingMethods.VALUE:
            return self
        literal: str = str(self.encoding).strip()
        if literal not in vocabulary:
            preview: str = ", ".join(sorted(vocabulary)[:8])
            raise TablassertValidationError(
                f"{self.qualifier} has a closed Biolink vocabulary; got {literal!r}. Permitted values include: {preview}...",
                code="qualifier-bad-value",
            )
        return self


class Statement(TablaBase):
    subject: NodeEncoding = Field(..., description="Subject node encoding and mapping configuration.")
    object: NodeEncoding = Field(..., description="Object node encoding and mapping configuration.")
    predicate: Predicates = Field(Predicates.RELATED_TO, description="Predicate connecting subject and object nodes.")
    qualifiers: list[Qualifier] | None = Field(None, description="Optional qualifier nodes attached to the statement.")


def validate_infores_curie(value: str, code: TablassertErrorCodes) -> str:
    """Validate an ``infores:`` CURIE used for Biolink knowledge-source fields."""
    if not value.startswith("infores:"):
        raise TablassertValidationError(f"InfoRes values must start with `infores:`, got {value!r}.", code=code)
    return value


class ManualProvenance(TablaBase):
    """Manually-specified provenance for non-PMID/PMC source graphs.

    When present under :class:`Provenance`, these values replace the legacy
    repo/publication-derived provenance while keeping the same KL/AT defaults.
    The edge ``primary_knowledge_source`` always derives from the graph-level
    ``infores`` (or ``infores:<graph-name>``); manual infores CURIEs belong in
    ``upstream_resource_ids``.
    """

    upstream_resource_ids: list[str] = Field(
        default_factory=list,
        description="Manual upstream source infores CURIEs emitted instead of the repo-derived source map; the sanctioned place for manual infores.",
        examples=[["infores:my-upstream"]],
    )
    publications: list[str] | None = Field(
        None,
        description="Publication CURIEs emitted verbatim; currently PMCID CURIEs are required for manual provenance.",
        examples=[["PMCID:PMC1234567"]],
    )
    knowledge_level: KnowledgeLevels = Field(
        KnowledgeLevels.STATISTICAL_ASSOCIATION, description="Biolink KL/AT knowledge level applied to produced edges."
    )
    agent_type: AgentTypes = Field(AgentTypes.DATA_ANALYSIS_PIPELINE, description="Biolink KL/AT agent type responsible for produced edges.")

    @field_validator("upstream_resource_ids", mode="after")
    @classmethod
    def upstream_infores_curies(cls, values: list[str]) -> list[str]:
        for value in values:
            validate_infores_curie(value, "override-bad-upstream-infores")
        return values

    @field_validator("publications", mode="after")
    @classmethod
    def pmcid_publications(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        for value in values:
            if not value.startswith("PMCID:"):
                raise TablassertValidationError(
                    f"Manual provenance publications must start with `PMCID:`, got {value!r}.", code="override-bad-publication"
                )
        return values


class Provenance(TablaBase):
    repo: Repositories = Field(Repositories.PUBMED_CENTRAL, description="Publication identifier namespace prefix.")
    publication: str | None = Field(
        None,
        description="Repository-local publication id appended as repo:publication; required unless manual provenance override is set.",
        examples=["12345678", "PMC1234567"],
    )
    knowledge_level: KnowledgeLevels = Field(
        KnowledgeLevels.STATISTICAL_ASSOCIATION, description="Biolink KL/AT knowledge level applied to produced edges."
    )
    agent_type: AgentTypes = Field(AgentTypes.DATA_ANALYSIS_PIPELINE, description="Biolink KL/AT agent type responsible for produced edges.")
    override: ManualProvenance | None = Field(
        None, description="Manual provenance values for non-PMID/PMC sources; when set, replace repo/publication-derived provenance."
    )

    @model_validator(mode="after")
    def is_valid_pmc_id(self: Self) -> Self:
        if self.override is not None:
            if self.publication is not None:
                raise TablassertValidationError(
                    "Specify either `publication` or `override`, not both, in provenance.", code="provenance-publication-and-override"
                )
            return self
        if self.publication is None:
            raise TablassertValidationError("`publication` is required unless provenance.override is set.", code="provenance-missing-publication")
        if self.repo == Repositories.PUBMED_CENTRAL and not re.search(r"^PMC\d+", self.publication):
            raise TablassertValidationError(
                f"PubMed Central publications must start with `PMC`, got {self.publication!r}.", code="provenance-bad-pmc-id"
            )

        return self


class Annotation(Encoding):
    annotation: str = Field(..., description="Output column name that receives this encoded annotation.", examples=["p_value", "cohort"])
    delimiter: str | None = Field(
        None,
        description=(
            "Split the encoded cell on this separator to emit a real JSON array instead of a scalar. "
            "Required for multivalued Biolink slots such as `has_evidence` or `FDA_regulatory_approvals`, "
            "whose consumers iterate the value."
        ),
        examples=["|", ";", ","],
    )

    @field_validator("delimiter", mode="after")
    @classmethod
    def non_empty_delimiter(cls, delimiter: str | None) -> str | None:
        # An empty separator splits into individual characters, which is never intended
        # and is exactly the failure mode a scalar-vs-list mismatch already causes
        # downstream (`publications.extend("PMID:1")` iterating characters).
        if delimiter is not None and not delimiter:
            raise TablassertValidationError("`delimiter` must be a non-empty separator.", code="annotation-bad-delimiter")
        return delimiter

    @field_validator("annotation", mode="after")
    @classmethod
    def clean_annotation(cls, annotation: str) -> str:
        return annotation.lower().strip()


class Section(TablaBase):
    """Pydantic section model and coercion target for a single table configuration."""

    source: Excel | Text = Field(..., description="Input source definition for reading tabular rows.")
    statement: Statement = Field(..., description="Subject-object statement mapping for this section.")
    provenance: Provenance = Field(..., description="Provenance metadata applied to all produced edges.")
    annotations: list[Annotation] | None = Field(None, description="Optional extra encoded columns added to each row.")


DEFAULT_RIG_CONTRIBUTIONS: list[str] = ["Tablassert: KGX and RIG generation"]
DEFAULT_RIG_UI_EXPLANATION: str = (
    "Source Tablassert data provides assertions derived from configured tabular records. "
    "The source record used to create this Translator edge was transformed into a "
    "Biolink association using the subject, predicate, object, provenance, and "
    "annotation mappings declared in Tablassert's Table Configuration."
)


def default_rig_contributions() -> list[str]:
    return DEFAULT_RIG_CONTRIBUTIONS.copy()


class Graph(TablaBase):
    """Pydantic graph configuration model."""

    name: str = Field(..., description="Graph name written into output metadata.")
    version: str = Field(..., description="Graph version label.")
    description: str = Field(..., description="Source scope description written into generated Resource Ingest Guides.")
    contributions: list[str] = Field(
        default_factory=default_rig_contributions, description="Resource Ingest Guide contribution statements for graph provenance."
    )
    ui_explanation: str = Field(DEFAULT_RIG_UI_EXPLANATION, description="Resource Ingest Guide explanation applied to generated edge type metadata.")
    infores: str | None = Field(
        None,
        description="Graph-level primary knowledge source infores CURIE; defaults to infores:<kebab-name> when omitted.",
        examples=["infores:my-kg"],
    )
    tables: list[Path] = Field(..., description="Paths to table YAML files included in this graph.", examples=[["tables/tutorial-table.yaml"]])
    fullmap: Path = Field(..., description="Base fullmap directory or fullmap redb file for entity resolution.", examples=[".fullmap"])

    @field_validator("infores", mode="after")
    @classmethod
    def infores_curie(cls, infores: str | None) -> str | None:
        if infores is None:
            return None
        return validate_infores_curie(infores, "graph-bad-infores")
