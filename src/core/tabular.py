from typing import Any, Optional, Union
from sqlite_utils import Database
from os.path import basename
from diskcache import Cache
from loguru import logger
import polars as pl
import math

def slicing(df: pl.DataFrame, download_hyperparameters: dict[str, Any]) -> pl.DataFrame:
    start: Optional[int] = download_hyperparameters.get("start_at_line_number")
    end: Optional[int] = download_hyperparameters.get("end_at_line_number")
    rows: Optional[list[int]] = download_hyperparameters.get("use_row_numbers")
    if start or end:
        height: int = df.height
        # the -1 is to convert from excels 1 based indexing to polars 0 based indexing
        if not start and end:
            slice_length: int = height - (end - 1)
            return df.slice(offset=0, length=slice_length)
        elif not end and start:
            start = start - 1
            slice_length = height - start
            return df.slice(offset=start, length=slice_length)
        elif start and end:
            slice_length = end - start
            return df.slice(offset=(start - 1), length=slice_length)
        else:
            raise RuntimeError("CODE 123 | At least one start or end must be defined")
    elif rows:
        return df.select(pl.all().take(indices=rows))  # type: ignore
    else:
        return df

def load_csv(posix_filepath: str, download_hyperparameters: dict[str, Any]) -> pl.DataFrame:
    delimiter: str = download_hyperparameters["file_delimiter"]
    df: pl.DataFrame = pl.read_csv(source=posix_filepath, separator=delimiter, has_header=False, infer_schema=False)
    return slicing(df, download_hyperparameters)

EXCEL_ENGINE: str = "xlsx2csv"

def load_excel(posix_filepath: str, download_hyperparameters: dict[str, Any]) -> pl.DataFrame:
    sheetname: str = download_hyperparameters["which_excel_sheet_to_use"]
    df = pl.read_excel(source=posix_filepath, sheet_name=sheetname, engine=EXCEL_ENGINE, has_header=False, infer_schema=False)  # type: ignore
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
            # assertions are for typechecking
            assert isinstance(value_for_comparison, float)
            return df.filter(pl.col(column).cast(pl.Float64) >= value_for_comparison)
        case "le":
            assert isinstance(value_for_comparison, float)
            return df.filter(pl.col(column).cast(pl.Float64) <= value_for_comparison)
        case "gt":
            assert isinstance(value_for_comparison, float)
            return df.filter(pl.col(column).cast(pl.Float64) > value_for_comparison)
        case "lt":
            assert isinstance(value_for_comparison, float)
            return df.filter(pl.col(column).cast(pl.Float64) < value_for_comparison)
        case "eq":
            return df.filter(pl.col(column) == value_for_comparison)
        case "ne":
            return df.filter(pl.col(column) != value_for_comparison)
        case _:
            raise RuntimeError(f"CODE: 122 | Only reindexing comparisons ge, le, gt, lt, eq, and ne are valid {comparison}")

def apply_math_module(df: pl.DataFrame, name: str, transformation: dict[str, Any]) -> pl.DataFrame:
    transformation_operation = lambda x: getattr(math, transformation["attribute"])(*[arg if arg is not None else x for arg in transformation["arguments"]])
    return df.with_columns(pl.col(name).cast(pl.Float64).map_elements(transformation_operation).alias(name))

# diskcache setup (sqlite caches)
fullmap3cache: Cache = Cache("TABLASSERT/CACHE/FULLMAP3", max_size=1e9)
babelcache: Cache = Cache("TABLASSERT/CACHE/BABEL", max_size=1e10)
kg2cache: Cache = Cache("TABLASSERT/CACHE/KG2", max_size=1e9)

def connect(sqlitepath: str) -> Database:
    db: Database = Database(sqlitepath)
    db.enable_wal()  # type: ignore
    return db

def dataframing(subsectionmodel: dict[str, Any], graphmodel: dict[str, dict[str, Any]]) -> pl.DataFrame:
    posix_filepath: str = subsectionmodel["posix_filepath"]
    download_hyperparameters: dict[str, Any] = subsectionmodel["location"]["download_hyperparameters"]
    df = initate(posix_filepath, download_hyperparameters)
    df = df.rename({old_name: excel_style_column_name(idx) for idx, old_name in enumerate(df.columns)})
    df = new_column(df, "download_link", "value", graphmodel["location"]["where_to_download_data_from"])
    df = new_column(df, "file_name", "value", basename(posix_filepath))
    sqlites: dict[str, str] = graphmodel["location"]["sqlite_databases"]
    pmc: Database = connect(sqlites["pmc"])
    # get filecaptions
    df = new_column(df, "file_caption", "value", basename(posix_filepath))
    df = new_column(df, "extension", "value", download_hyperparameters["extension"])
    df = new_column(df, "excel_sheet", "value", download_hyperparameters.get("which_excel_sheet_to_use"))
    provenance: dict[str, str] = subsectionmodel["provenance"]
    df = new_column(df, "article_curie", "value", provenance["article_curie"])
    df = new_column(df, "config_curator_name", "value", provenance["config_curator_name"])
    df = new_column(df, "config_curator_organization", "value", provenance["config_curator_organization"])
    pubmed: Database = connect(sqlites["pubmed"])
    # get pubmed_metadata
    reindexing: Optional[list[dict[str, Any]]] = subsectionmodel["reindexing"]
    if reindexing:
        for operation in reindexing:
            if operation["mode"] == "before":
                df = reindex(df, operation["column"], operation["comparison"], operation["value_for_comparison"])
    attributes: dict[str, Any] = subsectionmodel["attributes"]
    for name, attribute in attributes.items():
        if name != "notes":
            df = new_column(df, name, attribute.get("encoding_method"), attribute.get("value_for_encoding"))
            math_transformations: Optional[list[dict[str, Any]]] = attribute.get("math_module_transformation")
            if math_transformations:
                for transformation in math_transformations:
                    df = apply_math_module(df, name, transformation)
        else:
            df = new_column(df, name, "value", str(attribute))
    triple: dict[str, Any] = subsectionmodel["provenance"]
    for name, spoconfig in triple.items():
        name = name[7:]
        if name == "predicate":
            df = new_column(df, name, "value", str(spoconfig))
        else:
            encoding_method: str = spoconfig["encoding_method"]
            value_for_encoding: str = spoconfig["value_for_encoding"]
            df = new_column(df, f"origonal_{name}", encoding_method, value_for_encoding)
            df = new_column(df, name, encoding_method, value_for_encoding)
            mapping_hyperparameters: dict[str, Any] = spoconfig["mapping_hyperparameters"]
            how_to_fill_column: Optional[str] = mapping_hyperparameters.get("how_to_fill_column")
            if how_to_fill_column:
                df = df.with_columns(pl.col(name).fill_null(strategy=how_to_fill_column))
            explode_by_delimiter: Optional[str] = mapping_hyperparameters.get("explode_by_delimiter")
            if explode_by_delimiter:
                df = df.with_columns(pl.col(name).str.split(explode_by_delimiter)).explode(name)
            regular_expressions: Optional[list[dict[str, Any]]] = mapping_hyperparameters.get("regular_expressions")
            if regular_expressions:
                for regex in regular_expressions:
                    df = df.with_columns(pl.col(name).str.replace_all(regex["pattern"], regex["replacement"]).alias(name))
            substrings_to_remove: Optional[list[str]] = mapping_hyperparameters.get("substrings_to_remove")
            if substrings_to_remove:
                for substring in substrings_to_remove:
                    df = df.with_columns(pl.col(name).str.replace_all(substring, "").alias(name))
            prefix: Optional[str] = mapping_hyperparameters.get("prefix")
            if prefix:
                df = df.with_columns((pl.lit(prefix) + pl.col(name)).alias(name))
            suffix: Optional[str] = mapping_hyperparameters.get("suffix")
            if suffix:     
                df = df.with_columns((pl.col(name) + pl.lit(suffix)).alias(name))
    babel = connect(sqlites["babel"])
    kg2 = connect(sqlites["kg2"])
    # introduce fullmap3
    if reindexing:
        for operation in reindexing:
            if operation["mode"] == "after":
                df = reindex(df, operation["column"], operation["comparison"], operation["value_for_comparison"])
    return df


