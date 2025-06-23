from typing import Annotated, Optional, Literal, Union, Self
from urllib.parse import urlparse, unquote
from pathlib import Path
from pydantic import (
    field_validator,
    model_validator,
    BaseModel,
    HttpUrl,
    Field,
)
import math
import re
import os


def column_name_fallback(column_name: str) -> str:
    return column_name.upper()


def start_end_rows_fallback(
    start: Optional[int], end: Optional[int], rows: Optional[set[int]]
) -> tuple[Optional[int], Optional[int], Optional[set[int]]]:
    fallback = 1
    if (start or end) and rows:
        raise ValueError(
            "You cannot define start/end_at_line_number AND use_row_numbers"
        )
    if (start or end) and not (start and end):
        if not start:
            return fallback, end, rows
        if not end:
            return start, fallback, rows
    return start, end, rows


def biolink_fallback(x: str) -> str:
    if "biolink:" not in x:
        return "biolink:" + x
    return x


# classes starting with "t" are for basic template validation


class ExcelHyperparameters(BaseModel):
    extension: Literal["xlsx", "xls"] = Field(...)
    which_excel_sheet_to_use: str = Field(default="Sheet1")
    start_at_line_number: Optional[int] = Field(default=None)
    end_at_line_number: Optional[int] = Field(default=None)
    use_row_numbers: Optional[set[int]] = Field(default=None)

    @model_validator(mode="after")
    def start_end_rows_fix(self: Self) -> Self:
        start = self.start_at_line_number
        end = self.end_at_line_number
        rows = self.use_row_numbers
        start, end, rows = start_end_rows_fallback(start, end, rows)
        self.start_at_line_number = start
        self.end_at_line_number = end
        self.use_row_numbers = rows
        return self


class tExcelHyperparameters(BaseModel):
    extension: Optional[Literal["xlsx", "xls"]] = Field(default=None)
    which_excel_sheet_to_use: Optional[str] = Field(default=None)
    start_at_line_number: Optional[int] = Field(default=None)
    end_at_line_number: Optional[int] = Field(default=None)
    use_row_numbers: Optional[set[int]] = Field(default=None)


class CsvHyperparameters(BaseModel):
    extension: Literal["csv", "tsv", "txt"] = Field(...)
    file_delimiter: str = Field(default=",")
    start_at_line_number: Optional[int] = Field(default=None)
    end_at_line_number: Optional[int] = Field(default=None)
    use_row_numbers: Optional[set[int]] = Field(default=None)

    @model_validator(mode="after")
    def start_end_rows_fix(self: Self) -> Self:
        start = self.start_at_line_number
        end = self.end_at_line_number
        rows = self.use_row_numbers
        start, end, rows = start_end_rows_fallback(start, end, rows)
        self.start_at_line_number = start
        self.end_at_line_number = end
        self.use_row_numbers = rows
        return self


class tCsvHyperparameters(BaseModel):
    extension: Optional[Literal["csv", "tsv", "txt"]] = Field(default=None)
    file_delimiter: Optional[str] = Field(default=None)
    start_at_line_number: Optional[int] = Field(default=None)
    end_at_line_number: Optional[int] = Field(default=None)
    use_row_numbers: Optional[set[int]] = Field(default=None)


class PdfHyperparameters(BaseModel):
    extension: Literal["pdf"] = Field(...)
    pages_table_is_on: Optional[set[int]] = Field(default=None)
    camelot_flavor: Literal["stream", "lattice"] = Field(default="lattice")


class tPdfHyperparameters(BaseModel):
    extension: Optional[Literal["pdf"]] = Field(default=None)
    pages_table_is_on: Optional[set[int]] = Field(default=None)
    camelot_flavor: Optional[Literal["stream", "lattice"]] = Field(default=None)


DownloadHyperparameters = Annotated[
    Union[ExcelHyperparameters, CsvHyperparameters, PdfHyperparameters],
    Field(discriminator="extension"),
]
tDownloadHyperparameters = Union[
    tExcelHyperparameters, tCsvHyperparameters, tPdfHyperparameters
]


class Location(BaseModel):
    where_to_download_data_from: HttpUrl = Field(...)
    download_hyperparameters: DownloadHyperparameters = Field(...)


class tLocation(BaseModel):
    where_to_download_data_from: Optional[HttpUrl] = Field(default=None)
    download_hyperparameters: Optional[tDownloadHyperparameters] = Field(default=None)


class Provenance(BaseModel):
    article_curie: str = Field(...)
    config_curator_name: str = Field(...)
    config_curator_organization: str = Field(...)

    @field_validator("article_curie", mode="after")
    @classmethod
    def is_article_curie(cls, article_curie: str) -> str:
        accepted_domains = {"PMC:", "PMID:", "doi:"}
        if all(domain not in article_curie for domain in accepted_domains):
            return "PMC:" + article_curie
        return article_curie


class tProvenance(BaseModel):
    article_curie: Optional[str] = Field(default=None)
    config_curator_name: Optional[str] = Field(default=None)
    config_curator_organization: Optional[str] = Field(default=None)


class MathModuleTransformation(BaseModel):
    attribute: str = Field(...)
    arguments: list[Optional[float]] = Field(...)

    @field_validator("attribute", mode="after")
    @classmethod
    def is_math_module_attribute(cls, attribute: str) -> str:
        if not hasattr(math, attribute):
            raise ValueError("Transformation must include a valid math module atribute")
        return attribute

    @field_validator("arguments", mode="after")
    @classmethod
    def arguments_contains_nonetype(
        cls, arguments: list[Optional[float]]
    ) -> list[Optional[float]]:
        if not any(argument is None for argument in arguments):
            raise ValueError("At least one argument must be nonetype")
        return arguments

    @field_validator("arguments", mode="before")
    @classmethod
    def arguments_fallback(
        cls, arguments: list[Optional[float]]
    ) -> list[Optional[float]]:
        return [
            float(argument) if isinstance(argument, int) else argument
            for argument in arguments
        ]


class tMathModuleTransformation(BaseModel):
    attribute: Optional[str] = Field(default=None)
    arguments: Optional[list[Optional[str]]] = Field(default=None)


class Attribute(BaseModel):
    encoding_method: Literal["value", "column_of_values"] = Field(default="value")
    value_for_encoding: str = Field(...)
    math_module_transformations: Optional[MathModuleTransformation] = Field(
        default=None
    )

    @model_validator(mode="after")
    def column_name_fix(self: Self) -> Self:
        encoding = self.encoding_method
        value = self.value_for_encoding
        if encoding in {"column_of_values"}:
            self.value_for_encoding = column_name_fallback(value)
        return self


class tAttribute(BaseModel):
    encoding_method: Optional[Literal["value", "column_of_values"]] = Field(
        default=None
    )
    value_for_encoding: Optional[str] = Field(default=None)
    math_module_transformations: Optional[tMathModuleTransformation] = Field(
        default=None
    )


class Attributes(BaseModel):
    sample_size: Optional[Attribute] = Field(default=None)
    p_value: Optional[Attribute] = Field(default=None)
    multiple_testing_correction_method: Optional[Attribute] = Field(default=None)
    assertion_strength: Optional[Attribute] = Field(default=None)
    assertion_method: Optional[Attribute] = Field(default=None)
    notes: Optional[str] = Field(default=None)


class tAttributes(BaseModel):
    sample_size: Optional[tAttribute] = Field(default=None)
    p_value: Optional[tAttribute] = Field(default=None)
    multiple_testing_correction_method: Optional[tAttribute] = Field(default=None)
    assertion_strength: Optional[tAttribute] = Field(default=None)
    assertion_method: Optional[tAttribute] = Field(default=None)
    notes: Optional[str] = Field(default=None)


class RegularExpression(BaseModel):
    pattern: str = Field(...)
    replacement: str = Field(...)


class tRegularExpression(BaseModel):
    pattern: Optional[str] = Field(default=None)
    replacement: Optional[str] = Field(default=None)


class MappingHyperparameters(BaseModel):
    in_this_organism: Optional[str] = Field(default=None)
    classes_to_prioritize: Optional[set[str]] = Field(default=None)
    classes_to_avoid: Optional[set[str]] = Field(default=None)
    prefix: Optional[str] = Field(default=None)
    suffix: Optional[str] = Field(default=None)
    how_to_fill_column: Optional[
        Literal["forward", "backward", "min", "max", "mean", "zero", "one"]
    ] = Field(default=None)
    strings_to_remove: Optional[set[str]] = Field(default=None)
    regular_expressions: Optional[set[RegularExpression]] = Field(default=None)
    explode_by_delimiter: Optional[str] = Field(default=None)

    @field_validator("classes_to_prioritize", "classes_to_avoid", mode="after")
    @classmethod
    def biolink_priorities_and_avoid(cls, biolink_set: set[str]) -> set[str]:
        if biolink_set:
            return {biolink_fallback(thing) for thing in biolink_set}
        return biolink_set

    @field_validator("in_this_organism", mode="after")
    @classmethod
    def ncbi_taxon_fallback(cls, ncbi_taxon: str) -> str:
        if ncbi_taxon and "NCBITaxon:" not in ncbi_taxon:
            return "NCBITaxon:" + ncbi_taxon
        return ncbi_taxon


class tMappingHyperparameters(BaseModel):
    in_this_organism: Optional[str] = Field(default=None)
    classes_to_prioritize: Optional[set[str]] = Field(default=None)
    classes_to_avoid: Optional[set[str]] = Field(default=None)
    prefix: Optional[str] = Field(default=None)
    suffix: Optional[str] = Field(default=None)
    how_to_fill_column: Optional[
        Literal["forward", "backward", "min", "max", "mean", "zero", "one"]
    ] = Field(default=None)
    strings_to_remove: Optional[set[str]] = Field(default=None)
    regular_expressions: Optional[set[tRegularExpression]] = Field(default=None)
    explode_by_delimiter: Optional[str] = Field(default=None)


class GraphVertex(BaseModel):
    encoding_method: Literal["value", "column_of_values"] = Field(default="value")
    value_for_encoding: str = Field(...)
    mapping_hyperparameters: MappingHyperparameters = Field(...)

    @model_validator(mode="after")
    def column_name_fix(self: Self) -> Self:
        encoding = self.encoding_method
        value = self.value_for_encoding
        if encoding in {"column_of_values"}:
            self.value_for_encoding = column_name_fallback(value)
        return self


class tGraphVertex(BaseModel):
    encoding_method: Optional[Literal["value", "column_of_values"]] = Field(
        default=None
    )
    value_for_encoding: Optional[str] = Field(default=None)
    mapping_hyperparameters: Optional[tMappingHyperparameters] = Field(default=None)


class Triple(BaseModel):
    triple_subject: GraphVertex = Field(...)
    triple_object: GraphVertex = Field(...)
    triple_predicate: str = Field(default="biolink:associated_with")

    @field_validator("triple_predicate", mode="after")
    @classmethod
    def biolink_predicate(cls, predicate: str) -> str:
        return biolink_fallback(predicate)


class tTriple(BaseModel):
    triple_subject: Optional[tGraphVertex] = Field(default=None)
    triple_object: Optional[tGraphVertex] = Field(default=None)
    triple_predicate: Optional[str] = Field(default=None)


class Reindexing(BaseModel):
    mode: Literal["before", "after"] = Field(default="after")
    column: str = Field(...)
    comparison: Literal["ge", "le", "gt", "lt", "eq", "ne"] = Field(...)
    value_for_comparison: Union[str, float] = Field(...)

    @model_validator(mode="after")
    def mode_value_type_checking(self: Self) -> Self:
        mode = self.mode
        value = self.value_for_comparison
        string_modes = {"eq", "ne"}
        float_modes = {"ge", "le", "gt", "lt"}
        if mode in string_modes and not isinstance(value, str):
            self.value_for_comparison = str(value)
        if mode in float_modes and not isinstance(value, float):
            if isinstance(value, int):
                self.value_for_comparison = float(value)
            else:
                raise ValueError("You must specify a numeric type for " + mode)
        return self


class tReindexing(BaseModel):
    mode: Optional[Literal["before", "after"]] = Field(default=None)
    column: Optional[str] = Field(default=None)
    comparison: Optional[Literal["ge", "le", "gt", "lt", "eq", "ne"]] = Field(
        default=None
    )
    value_for_comparison: Optional[Union[str, float]] = Field(default=None)


class Section(BaseModel):
    filepath: Optional[Path] = Field(default=None)
    location: Location = Field(...)
    provenance: Provenance = Field(...)
    attributes: Attributes = Field(...)
    triple: Triple = Field(...)
    reindexing: Optional[Reindexing] = Field(default=None)

    @model_validator(mode="after")
    def filepath_generator(self: Self) -> Self:
        scaffold: Path = Path("tablassert/data_lake".upper())

        def clean_curie(curie: str) -> str:
            split: str = curie.split(":")[-1]
            return re.sub(r"[^A-Za-z0-9 ]+", "", split)

        curie: str = self.provenance.article_curie
        cleaned_curie = clean_curie(curie).upper()

        def get_filename_from_url(url: HttpUrl) -> str:
            parser = urlparse(str(url))
            path: str = parser.path
            filename = os.path.basename(path)
            return unquote(filename).upper()  # cleans API related stuff

        url: HttpUrl = self.location.where_to_download_data_from
        filename: str = get_filename_from_url(url).upper()

        filepath: Path = scaffold / cleaned_curie / filename
        filepath.parent.mkdir(
            parents=True, exist_ok=True
        )  # makes requrired directories
        self.filepath = filepath
        return self


class tSection(BaseModel):
    filepath: Optional[str] = Field(default=None)
    location: Optional[tLocation] = Field(default=None)
    provenance: Optional[tProvenance] = Field(default=None)
    attributes: Optional[tAttributes] = Field(default=None)
    triple: Optional[tTriple] = Field(default=None)
    reindexing: Optional[tReindexing] = Field(default=None)


class TableConfig(BaseModel):
    sections: list[Section] = Field(...)
    template: Optional[tSection] = Field(default=None)
