from tablassert.src.tablassert.models.table_config import TableConfig
import polars as pl

EXCEL_ENGINE: str = "calamine"



"""
def read_excel(config: TableConfig) -> pl.DataFrame:
    Location = TableConfig.location
    filename: str = TableConfig.filename
    return pl.read_excel(
        filename,
        sheet_name= ,
        engine="calamine",
    )
"""
