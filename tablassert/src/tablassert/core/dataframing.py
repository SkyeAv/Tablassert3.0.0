from tablassert.src.tablassert.models.table_config import Section, Location, PdfHyperparameters, CsvHyperparameters, ExcelHyperparameters, Provenance, Attributes, Reindexing, MathModuleTransformation, Triple
from tablassert.src.tablassert.models.graph_config import GraphConfig
from tablassert.src.tablassert.utils.io import PydanticModel
from typing import Optional, Literal, Union, Any
from pydantic import HttpUrl
from pathlib import Path
import polars as pl
import math

def take_rows_dataframe(df: pl.DataFrame, rows: set[int]) -> pl.DataFrame:
    return df.select(pl.all().take(indices=list(rows)))  # type: ignore

def single_bounded_slice_dataframe(df: pl.DataFrame, start: Optional[int], end: Optional[int]) -> pl.DataFrame:
    # slice requires both a start and end... to use python slices you need this
    single_bounded_slice: slice = slice(start, end)
    return pl.DataFrame(df.rows()[single_bounded_slice], schema=df.schema) 

def slice_dataframe(df: pl.DataFrame, start: int, end: int) -> pl.DataFrame:
    slice_length: int = end - start
    return df.slice(offset=start, length=slice_length)  # this is faster than single_bounded_slice

def dataframe_preprocessing(df: pl.DataFrame, start: Optional[int], end: Optional[int], rows: Optional[set[int]]) -> pl.DataFrame:
    if (start and end):
        return slice_dataframe(df, start, end)
    elif (start or end):
        return single_bounded_slice_dataframe(df, start, end)
    elif rows:
        return take_rows_dataframe(df, rows)
    else:
        return df

EXCEL_ENGINE: str = "calamine"

def read_excel(DownloadHyperparameters: ExcelHyperparameters, datapath: Path) -> pl.DataFrame:
    sheetname: str = DownloadHyperparameters.which_excel_sheet_to_use
    start: Optional[int] = DownloadHyperparameters.start_at_line_number
    end: Optional[int] = DownloadHyperparameters.end_at_line_number
    rows: Optional[set[int]] = DownloadHyperparameters.use_row_numbers
    df: pl.DataFrame = pl.read_excel(
        source=datapath,
        sheet_name=sheetname,
        engine=EXCEL_ENGINE,
        has_header=False,
        infer_schema_length=None,
    )  # type: ignore
    return dataframe_preprocessing(df, start, end, rows)

def read_csv(DownloadHyperparameters: CsvHyperparameters, datapath: Path) -> pl.DataFrame:
    delimiter: str = DownloadHyperparameters.file_delimiter
    start: Optional[int] = DownloadHyperparameters.start_at_line_number
    end: Optional[int] = DownloadHyperparameters.end_at_line_number
    rows: Optional[set[int]] = DownloadHyperparameters.use_row_numbers
    df: pl.DataFrame = pl.read_csv(
        source=datapath,
        separator=delimiter,
        has_header=False,
        infer_schema_length=None,
    )
    return dataframe_preprocessing(df, start, end, rows)

# add support later (not needed ASAP)
#def read_pdf(DownloadHyperparameters: PdfHyperparameters, datapath: Path) -> pl.DataFrame:
    #return pl.DataFrame()

def invoke(TableLocation: Location, datapath: Path) -> pl.DataFrame:
    DownloadHyperparameters: PydanticModel = TableLocation.download_hyperparameters
    if isinstance(DownloadHyperparameters, ExcelHyperparameters):
        return read_excel(DownloadHyperparameters, datapath)
    elif isinstance(DownloadHyperparameters, CsvHyperparameters):    
        return read_csv(DownloadHyperparameters, datapath)
    raise RuntimeError("Only xls, xlsx, csv, tsv, txt, and pdf are accepted")
    # add support later (not needed ASAP)
    #elif isinstance(DownloadHyperparameters, PdfHyperparameters):   
        #return pl.DataFrame()

def get_excel_style_column_names(column_name: str) -> str:
    index: int = int(column_name[-1])
    excel_style_letters: str = ""
    while index >= 0:
        excel_style_letters = chr(index % 26 + 65) + excel_style_letters
        index = index // 26 - 1
    return excel_style_letters

def apply_excel_style_column_names(df: pl.DataFrame) -> pl.DataFrame:
    return df.rename(lambda column: get_excel_style_column_names(str(column)))

def value_column(df: pl.DataFrame, column: str, value_for_encoding: str) -> pl.DataFrame:
    return df.with_cols(pl.lit(value_for_encoding).alias(column))

def column_of_values_column(df: pl.DataFrame, column: str, value_for_encoding: str) -> pl.DataFrame:
    if value_for_encoding in df.columns:
        return df.with_cols(pl.col(value_for_encoding).alias(column))
    else:
        raise RuntimeError(column + ": Column " + value_for_encoding + " does not exist")

def make_new_column(df: pl.DataFrame, column: str, encoding_method: Literal["value", "column_of_values"], value_for_encoding: str) -> pl.DataFrame:
    match encoding_method:
        case "value":
            return value_column(df, column, value_for_encoding)
        case "column_of_values":
            return column_of_values_column(df, column, value_for_encoding)
        case _:
            raise RuntimeError("Only value and column_of_values encoding methods are supported")

def greater_than_or_equal_to(df: pl.DataFrame, column: str, value_for_comparison: float) -> pl.DataFrame:
    try:
        return df.filter(pl.col() >= value_for_comparison)
    except Exception as e:
        raise RuntimeError('Cannot filter with mode "ge" in ' + column + " (" + str(e) + ")")

def less_than_or_equal_to(df: pl.DataFrame, column: str, value_for_comparison: float) -> pl.DataFrame:
    try:
        return df.filter(pl.col() <= value_for_comparison)
    except Exception as e:
        raise RuntimeError('Cannot filter with mode "le" in ' + column + " (" + str(e) + ")")

def greater_than(df: pl.DataFrame, column: str, value_for_comparison: float) -> pl.DataFrame:
    try:
        return df.filter(pl.col() > value_for_comparison)
    except Exception as e:
        raise RuntimeError('Cannot filter with mode "gt" in ' + column + " (" + str(e) + ")")

def less_than(df: pl.DataFrame, column: str, value_for_comparison: float) -> pl.DataFrame:
    try:
        return df.filter(pl.col() < value_for_comparison)
    except Exception as e:
        raise RuntimeError('Cannot filter with mode "lt" in ' + column + " (" + str(e) + ")")

def equal_to(df: pl.DataFrame, column: str, value_for_comparison: Union[str, float]) -> pl.DataFrame:
    try:
        return df.filter(pl.col() == value_for_comparison)
    except Exception as e:
        raise RuntimeError('Cannot filter with mode "eq" in ' + column + " (" + str(e) + ")")

def not_equal_to(df: pl.DataFrame, column: str, value_for_comparison: Union[str, float]) -> pl.DataFrame:
    try:
        return df.filter(pl.col() != value_for_comparison)
    except Exception as e:
        raise RuntimeError('Cannot filter with mode "ne" in ' + column + " (" + str(e) + ")")

def reindex_column(df: pl.DataFrame, column: str, comparison: Literal["ge", "le", "gt", "lt", "eq", "ne"], value_for_comparison: Union[str, float]) -> pl.DataFrame:
    match str(comparison):
        case "ge":
            assert isinstance(value_for_comparison, float)
            return greater_than_or_equal_to(df, column, value_for_comparison)
        case "le":
            assert isinstance(value_for_comparison, float)
            return less_than_or_equal_to(df, column, value_for_comparison)
        case "gt":
            assert isinstance(value_for_comparison, float)
            return greater_than(df, column, value_for_comparison)
        case "lt":
            assert isinstance(value_for_comparison, float)
            return less_than(df, column, value_for_comparison)
        case "eq":
            return equal_to(df, column, value_for_comparison)
        case "ne":
            return not_equal_to(df, column, value_for_comparison)
        case _:
            raise RuntimeError("Only reindexing comparisons ge, le, gt, lt, eq, and ne are valid")

def reindexing_operation(df: pl.DataFrame, ReindexingOperation: Reindexing, target_mode: Literal["before", "after"]) -> pl.DataFrame:
    mode: Literal["before", "after"] = ReindexingOperation.mode
    if str(mode) == str(target_mode):
        column: str = ReindexingOperation.mode
        comparison: Literal["ge", "le", "gt", "lt", "eq", "ne"] = ReindexingOperation
        value_for_comparison: Union[str, float] = ReindexingOperation
        return reindex_column(df, column, comparison, value_for_comparison)
    return df

def math_module_operation(df: pl.DataFrame, column: str, Transformation: MathModuleTransformation) -> pl.DataFrame:
    math_module_attribute: str = Transformation.attribute
    operation: Any = getattr(math, math_module_attribute)
    arguments: list[Optional[float]] = Transformation.arguments
    transformation_operation: Any = lambda x: operation(*[arg if arg is not None else x for arg in arguments])
    return df.with_cols(pl.col(column).apply(transformation_operation).alias(column))

def before_mapping(df: pl.DataFrame, Table: Section, TableLocation: Location) -> pl.DataFrame:

    download_link: HttpUrl = TableLocation.where_to_download_data_from
    download_link = str(download_link)
    df = make_new_column(df, "download_link", "value", download_link)

    DownloadHyperparameters: PydanticModel = TableLocation.download_hyperparameters
    if isinstance(DownloadHyperparameters, ExcelHyperparameters):
        DownloadHyperparameters
    if isinstance(DownloadHyperparameters, CsvHyperparameters):    
        DownloadHyperparameters
    # add support later (not needed ASAP)
    #if isinstance(DownloadHyperparameters, PdfHyperparameters):   

    TableProvenance: Provenance = Table.provenance

    TableAttributes: Attributes = Table.attributes
    for Attribute in TableAttributes:
        attribute_name: str = Attribute.__name__
        encoding_method: str = Attribute.encoding_method
        value_for_encoding: str = Attribute.value_for_encoding
        df = make_new_column(df, attribute_name, encoding_method, value_for_encoding)
        math_module_transformations: Optional[set[MathModuleTransformation]] = Attribute.math_module_transformation
        if math_module_transformations:
            for Transformation in math_module_transformations:
                df = math_module_operation(df, attribute_name, Transformation)

    TableReindexing: set[Reindexing] = Table.reindexing
    for ReindexingOperation in TableReindexing:
        df = reindexing_operation(df, ReindexingOperation, "before")

    return df

def mapping(df: pl.DataFrame, Assertion: Triple) -> pl.DataFrame:
    predicate: str = Assertion.triple_predicate
    df = make_new_column(df, "predicate", "value", predicate)

def after_mapping(df: pl.DataFrame, Table: Section) -> pl.DataFrame:
    
    TableReindexing: set[Reindexing] = Table.reindexing
    for ReindexingOperation in TableReindexing:
        df = reindexing_operation(df, ReindexingOperation, "after")
    
    return df

def dataframing(Table: Section, Graph: GraphConfig, datapath: Path) -> pl.DataFrame:
    TableLocation: Location = Table.location
    df: pl.DataFrame = invoke(TableLocation, datapath)
    df = apply_excel_style_column_names(df)
    df = before_mapping(df, Table, TableLocation)
    Assertion: Triple = Table.triple
    df = mapping(df, Assertion)
    return after_mapping(df, Table)