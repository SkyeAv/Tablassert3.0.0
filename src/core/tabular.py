from typing import Any, Optional
from diskcache import Cache
from loguru import logger
import polars as pl

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

def excel_style_column_name(idx: int) -> str:
    letters: str = ""
    while idx >= 0:
        letters = chr(idx % 26 + 65) + letters
        idx = idx // 26 - 1
    return letters

def new_column(df: pl.DataFrame, column: str, encoding_method: str, value_for_encoding: Optional[str]) -> pl.DataFrame:
    if not value_for_encoding:
        value_for_encoding = "not applicable"
    if encoding_method == "value":
        return df.with_columns(pl.lit(value_for_encoding).alias(column))
    elif encoding_method == "column_of_values":
        return df.with_columns(pl.col(value_for_encoding).alias(column))
    else:
        raise RuntimeError(f"CODE:121 | Unrecognized encoding_method {encoding_method}")

def reindex(df: pl.DataFrame, column: str, comparison: str, value_for_comparison: Union[str, float]) -> pl.DataFrame:
    match comparison:
        case "ge":
            assert isinstance(value_for_comparison, float)
            return df.filter(pl.col(column) >= value_for_comparison)
        case "le":
            assert isinstance(value_for_comparison, float)
            return df.filter(pl.col(column) <= value_for_comparison)
        case "gt":
            assert isinstance(value_for_comparison, float)
            return df.filter(pl.col(column) > value_for_comparison)
        case "lt":
            assert isinstance(value_for_comparison, float)
            return df.filter(pl.col(column) < value_for_comparison)
        case "eq":
            return df.filter(pl.col(column) == value_for_comparison)
        case "ne":
            return df.filter(pl.col(column) != value_for_comparison)
        case _:
            raise RuntimeError(f"CODE: 122 | Only reindexing comparisons ge, le, gt, lt, eq, and ne are valid {comparison}")

def dataframing(subsectionmodel: dict[str, Any], graphmodel: dict[str, dict[str, Any]]) -> pl.DataFrame:
    posix_filepath: str = subsectionmodel["posix_filepath"]
    download_hyperparameters: dict[str, Any] = graphmodel["location"]["download_hyperparameters"]
    df = initate(posix_filepath, download_hyperparameters)
    df = df.rename(columns={old_name: excel_style_column_name(idx) for idx, old_name in enumerate(df.columns)})

# diskcache setup (sqlite caches)
fullmap3cache: Cache = Cache("TABLASSERT/CACHE/FULLMAP3", max_size=1e9)
babelcache: Cache = Cache("TABLASSERT/CACHE/BABEL", max_size=1e10)
kg2cache: Cache = Cache("TABLASSERT/CACHE/KG2", max_size=1e9)

def connect(sqlitepath: str) -> Database:
    db: Database = Database(sqlitepath)
    db.enable_wal()
    return db
