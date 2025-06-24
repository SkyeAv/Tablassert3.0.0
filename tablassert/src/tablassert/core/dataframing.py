from tablassert.src.tablassert.models.table_config import Section, Location, PdfHyperparameters, CsvHyperparameters, ExcelHyperparameters
from tablassert.src.tablassert.models.io import PydanticModel
from typing import Optional
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
    raise RuntimeError("Either start/end or eows must be specified")

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
    )  # type: ignore
    return dataframe_preprocessing(df, start, end, rows)

def read_csv(DownloadHyperparameters: CsvHyperparameters, datapath: Path) -> pl.DataFrame:
    delimiter: str = DownloadHyperparameters.file_delimiter
    start: Optional[int] = DownloadHyperparameters.start_at_line_number
    end: Optional[int] = DownloadHyperparameters.end_at_line_number
    rows: Optional[set[int]] = DownloadHyperparameters.use_row_numbers
    df: pl.DataFrame = pl.read_csv(
        source=datapath,
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

def dataframing(Table: Section, datapath: Path) -> pl.DataFrame:
    TableLocation: Location = Table.location
    df: pl.DataFrame = invoke(TableLocation, datapath)
    return df