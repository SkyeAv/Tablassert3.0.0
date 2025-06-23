from tablassert.src.tablassert.models.table_config import TableConfig
import polars as pl

EXCEL_ENGINE = "calamine"

def read_excel(config: TableConfig) -> pl.DataFrame:
    Location = TableConfig.location
    filename: str = Location.
    return pl.read_excel()

