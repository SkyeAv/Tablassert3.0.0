from typing import Any, Optional
from diskcache import Cache
from loguru import logger
import polars as pl

# diskcache setup (sqlite caches)
fullmap3cache: Cache = Cache("TABLASSERT/CACHE/FULLMAP3", max_size=1e9)
babelcache: Cache = Cache("TABLASSERT/CACHE/BABEL", max_size=1e10)
kg2cache: Cache = Cache("TABLASSERT/CACHE/KG2", max_size=1e9)

def connect(sqlitepath: str) -> Database:
    db: Database = Database(sqlitepath)
    db.enable_wal()
    return db

def slicing(df: pl.DataFrame, download_hyperparameters: dict[str, Any]) -> pl.DataFrame:
    start: Optional[int] = download_hyperparameters.get("start_at_line_number")
    end: Optional[int] = download_hyperparameters.get("end_at_line_number")
    rows: Optional[list[int]] = download_hyperparameters.get("use_row_numbers")
    if start or end:
        height: int = df.height
        # the -1 is to convert from excels 1 based indexing to polars 0 based indexing
        if not start:
            slice_length: int = height - (end - 1)
            return df.slice(offset=0, length=slice_length)
        elif not end:
            start = start - 1
            slice_length = height - start
            return df.slice(offset=start, length=slice_length)
        else:
            slice_length = end - start
            return df.slice(offset=(start - 1), length=slice_length)
    elif rows:
        return df.select(pl.all().take(indices=rows))
    else:
        return df

def load_csv(posix_filepath: str, download_hyperparameters: dict[str, Any]) -> pl.DataFrame:
    delimiter: str = download_hyperparameters["file_delimiter"]
    df: pl.DataFrame = pl.read_csv(source=posix_filepath, separator=delimiter, has_header=False, infer_schema_length=None)
    return slicing(df, download_hyperparameters)

EXCEL_ENGINE: str = "xlsx2csv"

def load_excel(posix_filepath: str, download_hyperparameters: dict[str, Any]) -> pl.DataFrame:
    sheetname: str = download_hyperparameters["which_excel_sheet_to_use"]
    df = pl.read_excel(source=posix_filepath, sheet_name=sheetname, engine=EXCEL_ENGINE, has_header=False, infer_schema_length=None)
    return slicing(df, download_hyperparameters)

def initate(posix_filepath: str, download_hyperparameters: dict[str, Any]) -> pl.DataFrame:
    extension: str = download_hyperparameters["extension"]
    match extension.lower():
        case "xls" | "xlsx":
            return load_excel(posix_filepath, download_hyperparameters)
        case "csv" | "tsv" | "txt":
            return load_csv(posix_filepath, download_hyperparameters)
        case _:
            raise RuntimeError(f"CODE:120 | Tablassert doesn't support {extension}... yet")

def dataframing(subsectionmodel: dict[str, Any], graphmodel: dict[str, dict[str, Any]]) -> pl.DataFrame:
    posix_filepath: str = subsectionmodel["posix_filepath"]
    download_hyperparameters: dict[str, Any] = graphmodel["location"]["download_hyperparameters"]
    df = initate(posix_filepath, download_hyperparameters)
