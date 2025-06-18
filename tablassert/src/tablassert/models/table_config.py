from pydantic import ValidationError, BaseModel, Field
from typing import Annotated, Optional, Literal, Union

class ExcelHyperparameters(BaseModel):
    extension: Literal["xlsx", "xls"] = Field(...)
    which_excel_sheet_to_use: str = Field(default="Sheet1")
    start_at_line_number: int = Field(...)
    end_at_line_number: int = Field(...)
    use_row_numbers: set[int] = Field(...)

class tExcelHyperparameters(BaseModel):
    extension: Optional[Literal["xlsx", "xls"]] = Field(default=None)
    which_excel_sheet_to_use: Optional[str] = Field(default=None)
    start_at_line_number: Optional[int] = Field(default=None)
    end_at_line_number: Optional[int] = Field(default=None)
    use_row_numbers: Optional[set[int]] = Field(default=None)

class CsvHyperparameters(BaseModel):
    extension: Literal["csv", "tsv", "txt"] = Field(...)
    file_delimiter: str = Field(default=",")
    start_at_line_number: int = Field(...)
    end_at_line_number: int = Field(...)
    use_row_numbers: set[int] = Field(...)

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

DownloadHyperparameters = Annotated[Union[ExcelHyperparameters, CsvHyperparameters, PdfHyperparameters], Field(discriminator="extension")]
tDownloadHyperparameters = Union[tExcelHyperparameters, tCsvHyperparameters, tPdfHyperparameters]

class Location(BaseModel):
    where_to_download_data_from: str = Field(...)
    download_hyperparameters: DownloadHyperparameters = Field(...)

class tLocation(BaseModel):
    where_to_download_data_from: Optional[str] = Field(default=None)
    download_hyperparameters: Optional[tDownloadHyperparameters] = Field(default=None)

class Provenance(BaseModel):
    article_curie: str = Field(...)
    config_curator_name: str = Field(default="Omitted")
    config_curator_organization: str = Field(default="Omitted")

class tProvenance(BaseModel):
    article_curie: Optional[str] = Field(default=None)
    config_curator_name: Optional[str] = Field(default=None)
    config_curator_organization: Optional[str] = Field(default=None)

class MathModuleTransformation(BaseModel):
    attribute: str = Field(...)
    arguments: list[Optional[str]] = Field(...)

class tMathModuleTransformation(BaseModel):
    attribute: Optional[str] = Field(default=None)
    arguments: Optional[list[Optional[str]]] = Field(default=None)

class Attribute(BaseModel):
    encoding_method: Literal["value", "column_of_values"] = Field(default="value")
    value_for_encoding: str = Field(...)
    math_module_transformations: Optional[MathModuleTransformation] = Field(default=None)

class tAttribute(BaseModel):
    encoding_method: Optional[Literal["value", "column_of_values"]] = Field(default=None)
    value_for_encoding: Optional[str] = Field(default=None)
    math_module_transformations: Optional[tMathModuleTransformation] = Field(default=None)

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
    in_this_organism: str = Field(default="NCBITaxon:9606")
    classes_to_prioritize: set[str] = Field(...)
    classes_to_avoid: set[str] = Field(...)
    prefix: str = Field(...)
    suffix: str = Field(...)
    how_to_fill_column: Literal["forward", "backward", "min", "max", "mean", "zero", "one"] = Field(default="forward")
    strings_to_remove: set[str] = Field(...)
    regular_expressions: set[RegularExpression] = Field(...)
    explode_by_delimiter: str = Field(...)

class tMappingHyperparameters(BaseModel):
    in_this_organism: Optional[str] = Field(default=None)
    classes_to_prioritize: Optional[set[str]] = Field(default=None)
    classes_to_avoid: Optional[set[str]] = Field(default=None)
    prefix: Optional[str] = Field(default=None)
    suffix: Optional[str] = Field(default=None)
    how_to_fill_column: Optional[Literal["forward", "backward", "min", "max", "mean", "zero", "one"]] = Field(default=None)
    strings_to_remove: Optional[set[str]] = Field(default=None)
    regular_expressions: Optional[set[tRegularExpression]] = Field(default=None)
    explode_by_delimiter: Optional[str] = Field(default=None)

class GraphVertex(BaseModel):
    encoding_method: Literal["value", "column_of_values", "curie", "column_of_curies"] = Field(defult="value")
    value_for_encoding: str = Field(...)
    mapping_hyperparameters: MappingHyperparameters = Field(...)

class tGraphVertex(BaseModel):
    encoding_method: Optional[Literal["value", "column_of_values", "curie", "column_of_curies"]] = Field(default=None)
    value_for_encoding: Optional[str] = Field(default=None)
    mapping_hyperparameters: Optional[tMappingHyperparameters] = Field(default=None)

class Triple(BaseModel):
    triple_subject: GraphVertex = Field(...)
    triple_object: GraphVertex = Field(...)
    triple_predicate: str = Field(default="biolink:associated_with")

class tTriple(BaseModel):
    triple_subject: Optional[tGraphVertex] = Field(default=None)
    triple_object: Optional[tGraphVertex] = Field(default=None)
    triple_predicate: Optional[str] = Field(default=None)

class Reindexing(BaseModel):
    mode: Literal["before", "after"] = Field(default="after")
    column: str = Field(...)
    comparison: Literal["ge", "le", "gt", "lt", "eq", "ne"] = Field(...)
    value_for_comparison: Union[str, float] = Field(...)

class tReindexing(BaseModel):
    mode: Optional[Literal["before", "after"]] = Field(default=None)
    column: Optional[str] = Field(default=None)
    comparison: Optional[Literal["ge", "le", "gt", "lt", "eq", "ne"]] = Field(default=None)
    value_for_comparison: Optional[Union[str, float]] = Field(default=None)

class Section(BaseModel):
    location: = Field(...)
    provenance: Provenance = Field(...)
    attributes: Attributes = Field(...)
    triple: Triple = Field(...)
    reindexing: Optional[Reindexing] = Field(default=None)

class tSection(BaseModel):
    location: Optional[] = Field(default=None)
    provenance: Optional[tProvenance] = Field(default=None)
    attributes: Optional[tAttributes] = Field(default=None)
    triple: Optional[tTriple] = Field(default=None)
    reindexing: Optional[tReindexing] = Field(default=None)

class TableConfig(BaseModel):
    sections: list[Section] = Field(...)
    template: Optional[tSection] = Field(default=None)
