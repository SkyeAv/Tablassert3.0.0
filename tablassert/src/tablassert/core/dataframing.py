from tablassert.src.tablassert.models.table_config import Section, Location, PdfHyperparameters, CsvHyperparameters, ExcelHyperparameters
from tablassert.src.tablassert.models.io import PydanticModel
from pathlib import Path
import polars as pl

EXCEL_ENGINE: str = "calamine"

def read_excel(DownloadHyperparameters: ExcelHyperparameters, datapath: Path) -> pl.DataFrame:
    DownloadHyperparameters.
    return pl.read_excel(
        soruce=datapath,
        sheet_name=
        engine=EXCEL_ENGINE,
    )

def read_csv(DownloadHyperparameters: CsvHyperparameters, datapath: Path) -> pl.DataFrame:
    return pl.read_csv(
        soruce=datapath,
    )

# add support later (not needed ASAP)
#def read_pdf(DownloadHyperparameters: PdfHyperparameters, datapath: Path) -> pl.DataFrame:
    #return pl.DataFrame()

def invoke(TableLocation: Location, datapath: Path) -> pl.DataFrame:
    DownloadHyperparameters: PydanticModel = TableLocation.download_hyperparameters
    if isinstance(DownloadHyperparameters, ExcelHyperparameters):
        return read_excel(DownloadHyperparameters, datapath)
    elif isinstance(DownloadHyperparameters, CsvHyperparameters):    
        return read_csv(DownloadHyperparameters, datapath)
    # add support later (not needed ASAP)
    #elif isinstance(DownloadHyperparameters, PdfHyperparameters):   
        #return pl.DataFrame()   

def dataframing(Table: Section, datapath: Path) -> pl.DataFrame:
    TableLocation: Location = Table.location
    df: pl.DataFrame = invoke(TableLocation, datapath)