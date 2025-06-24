from tablassert.src.tablassert.models.table_config import Section, Location, PdfHyperparameters, CsvHyperparameters, ExcelHyperparameters, Provenance, Attributes, Reindexing
from tablassert.src.tablassert.models.graph_config import GraphConfig
from tablassert.src.tablassert.models.io import PydanticModel
from typing import Optional, Literal, Union
from pathlib import Path
import polars as pl

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

def reindex_column(df: pl.DataFrame, column: str, comparison: Literal["ge", "le", "gt", "lt", "eq", "ne"], value_for_comparison: Union[str, float]) -> pl.DataFrame:
    match comparison:
        case "ge":
            return
        case "le":
            return
        case "gt":
            return
        case "lt":
            return
        case "eq":
            return
        case "ne":
            return
        case _:
            return

def before_mapping(df: pl.DataFrame, Table: Section) -> pl.DataFrame:
    TableProvenance: Provenance = Table.provenance
    TableAttributes: Attributes = Table.attributes
    TableReindexing: Reindexing = Table.reindexing

def dataframing(Table: Section, Graph: GraphConfig, datapath: Path) -> pl.DataFrame:
    TableLocation: Location = Table.location
    df: pl.DataFrame = invoke(TableLocation, datapath)
    df = apply_excel_style_column_names(df)
    df = before_mapping(df, Table)
    return df