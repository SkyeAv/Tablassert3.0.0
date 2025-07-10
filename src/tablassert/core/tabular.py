from typing import Any, Optional, Union, Iterator
from sqlite_utils import Database
from functools import lru_cache
from collections import Counter
from spacy.tokens import Token
from os.path import basename
from diskcache import Cache
from loguru import logger
from pathlib import Path
import polars as pl
import pandas as pd
import sqlite3
import spacy
import math
import time
import re


# I have to do this because of Pool
def initialize_logger() -> None:
    LOG_PATH: Path = Path("TABLASSERT/LOG/mapping.log").resolve()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        LOG_PATH.as_posix(), rotation="250 MB", compression="xz", retention="1 month"
    )
    # added separate log for things that don't map
    DIDNTMAP_LOG_PATH = LOG_PATH.parent / "didntmap.log"
    DIDNTMAP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        DIDNTMAP_LOG_PATH.as_posix(),
        level="WARNING",
        rotation="250 MB",
        compression="xz",
        retention="1 month",
    )


def slicing(df: pl.DataFrame, download_hyperparameters: dict[str, Any]) -> pl.DataFrame:
    start: Optional[int] = download_hyperparameters.get("start_at_line_number")
    end: Optional[int] = download_hyperparameters.get("end_at_line_number")
    rows: Optional[list[int]] = download_hyperparameters.get("use_row_numbers")
    # added to the end of the df not to mess up excel style names, also before the slice for reliable rows
    row_index: pl.Series = pl.Series(
        "extracted_from_row_number", list(range(1, df.height + 1))
    ).cast(
        pl.String
    )  # to correct for the excel style indexing
    df.insert_column(
        len(df.columns), row_index
    )  # because this is the only thing that modifies in place apparently
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
        rows = [row - 1 for row in rows]
        return df.select(pl.all().take(indices=rows))  # type: ignore
    else:
        return df


def load_csv(
    posix_filepath: str, download_hyperparameters: dict[str, Any]
) -> pl.DataFrame:
    delimiter: str = download_hyperparameters["file_delimiter"]
    df: pl.DataFrame = pl.read_csv(
        source=posix_filepath,
        separator=delimiter,
        has_header=False,
        infer_schema=False,
    )
    return slicing(df, download_hyperparameters)


# silly goofy engine here sometimes parses gene symbols to dates... we're stuck with this because our CPU can't support a better engine
DEFAULT_EXCEL_ENGINE: str = "xlsx2csv"
FALLBACK_EXCEL_ENGINE: str = "openpyxl"
XLS_ENGINE: str = "xlrd"


def load_excel(
    posix_filepath: str, download_hyperparameters: dict[str, Any]
) -> pl.DataFrame:
    sheetname: str = download_hyperparameters["which_excel_sheet_to_use"]
    extension: str = download_hyperparameters["extension"]
    if (
        extension == "xls"
    ):  # this is only because xlsx2csv and all of the polars readers don't support the old xls encoding
        pandasdf: pd.DataFrame = pd.read_excel(
            posix_filepath,
            sheet_name=sheetname,
            header=None,
            dtype=str,
            engine=XLS_ENGINE,
        )
        df = pl.from_pandas(pandasdf)  # you need pyarrow in the environment for this
    else:
        try:
            df = pl.read_excel(source=posix_filepath, sheet_name=sheetname, engine=DEFAULT_EXCEL_ENGINE, has_header=False, read_options={"infer_schema": False})  # type: ignore
        except TypeError as e:
            if "NoneType" in str(e):  # for that one weird nonetype bug
                df = pl.read_excel(source=posix_filepath, sheet_name=sheetname, engine=FALLBACK_EXCEL_ENGINE, has_header=False, read_options={"infer_schema": False})  # type: ignore
    return slicing(df, download_hyperparameters)


def initate(
    posix_filepath: str, download_hyperparameters: dict[str, Any]
) -> pl.DataFrame:
    extension: str = download_hyperparameters["extension"]
    match extension.lower():
        case "xls" | "xlsx":
            return load_excel(posix_filepath, download_hyperparameters)
        case "csv" | "tsv" | "txt":
            return load_csv(posix_filepath, download_hyperparameters)
        case _:
            raise RuntimeError(
                f"CODE:120 | Tablassert doesn't support {extension}... yet"
            )


def excel_style_column_name(idx: int) -> str:
    letters: str = ""
    while idx >= 0:
        letters = chr(idx % 26 + 65) + letters
        idx = idx // 26 - 1
    return letters


def new_column(
    df: pl.DataFrame,
    column: str,
    encoding_method: str,
    value_for_encoding: Optional[str],
) -> pl.DataFrame:
    if not value_for_encoding:
        value_for_encoding = "Not applicable"
    if encoding_method == "value":
        return df.with_columns(pl.lit(str(value_for_encoding)).alias(column))
    elif encoding_method == "column_of_values":
        return df.with_columns(pl.col(str(value_for_encoding)).alias(column))
    else:
        raise RuntimeError(f"CODE:121 | Unrecognized encoding_method {encoding_method}")


def reindex(
    df: pl.DataFrame,
    column: str,
    comparison: str,
    value_for_comparison: Union[str, float],
) -> pl.DataFrame:
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
            raise RuntimeError(
                f"CODE: 122 | Only reindexing comparisons ge, le, gt, lt, eq, and ne are valid {comparison}"
            )


def apply_reindexing(
    df: pl.DataFrame, operation: dict[str, str], mode: str
) -> pl.DataFrame:
    if operation["mode"] == mode:
        return reindex(
            df,
            operation["column"],
            operation["comparison"],
            operation["value_for_comparison"],
        )
    return df


def apply_math_module(
    df: pl.DataFrame, name: str, transformation: dict[str, Any]
) -> pl.DataFrame:
    return df.with_columns(
        pl.col(name)
        .cast(pl.Float64)
        .map_elements(
            lambda x: getattr(math, transformation["attribute"])(
                *[arg if arg is not None else x for arg in transformation["arguments"]]
            )
        )
        .alias(name)
    )


def process_attribute(
    df: pl.DataFrame, name: str, attribute: dict[str, Any]
) -> pl.DataFrame:
    if name != "notes":
        df = new_column(
            df,
            name,
            attribute["encoding_method"],
            attribute.get("value_for_encoding"),
        )
        math_transformations: Optional[list[dict[str, Any]]] = attribute.get(
            "math_module_transformation"
        )
        if math_transformations:
            for transformation in math_transformations:
                df = apply_math_module(df, name, transformation)
        return df
    else:
        return new_column(df, name, "value", str(attribute))


# diskcache setup (sqlite caches)
fullmap3cache: Cache = Cache("TABLASSERT/CACHE/FULLMAP3", max_size=1e8)

start: float = 0.0  # default for typechecking


def progress_handler(maxtime: float) -> int:
    if (time.time() - start) >= maxtime:
        return 1
    return 0


def connect(sqlitepath: str, maxtime: float) -> Database:
    db: Database = Database(sqlitepath)
    db.enable_wal()  # type: ignore
    conn = db.conn
    conn.set_progress_handler(lambda: progress_handler(maxtime), 1)
    return db


def levelone(x: Any) -> str:
    return str(x).lower()


DISABLE: list[str] = ["parser", "ner", "textcat"]
MODEL = spacy.load("en_core_web_sm", disable=DISABLE)


def leveltwo(leveloneoutput: str) -> str:
    tokens: list[Token] = MODEL(leveloneoutput)  # type: ignore
    cleaned_tokens: list[str] = [
        token.lemma_  # yield lemma
        for token in tokens  # iterate through tokens
        if not token.is_stop  # is not a stopword
        and not token.is_punct  # is not punctuation
    ]
    sorted_cleaned_unique_tokens: list[str] = sorted(
        list(dict.fromkeys(cleaned_tokens))
    )
    leveltwooutput: str = " ".join(sorted_cleaned_unique_tokens)
    return leveltwooutput


NONWORD_REGEX: Any = re.compile(r"\W+")


def levelthree(leveltwooutput: str) -> str:
    regex: Any = NONWORD_REGEX
    levelthreeoutput: str = re.sub(regex, "", leveltwooutput)
    return levelthreeoutput


ColumnContext: Counter[str] = Counter()


# dynamic query build because f strings only evaluate once at their creation
def babelsql(
    prioritize_placeholders: Optional[str],
    avoid_placeholders: Optional[str],
    taxon: Optional[str],
    most_common: Optional[str],
    level: str,
) -> str:

    babel_levelcondtion: str = {
        "L1": "SYNONYMS.L1 = :input",
        "L2": "SYNONYMS.L2 = :input",
        "L3": "SYNONYMS.L3 = :input",
    }.get(level, "")

    babel_taxoncondition: str = (
        "AND (NAMES.CATEGORY != 'Gene' OR NAMES.TAXON = :taxon)" if taxon else ""
    )
    babel_avoidcondition: str = (
        f"AND NAMES.CATEGORY NOT IN ({avoid_placeholders})"
        if avoid_placeholders
        else ""
    )

    if prioritize_placeholders and most_common:
        babel_orderbyclause: str = f"""
        ORDER BY
            CASE
                WHEN NAMES.CATEGORY IN ({prioritize_placeholders}) AND NAMES.CATEGORY = :most_common THEN 0
                WHEN NAMES.CATEGORY IN ({prioritize_placeholders}) THEN 1
                WHEN NAMES.CATEGORY = :most_common THEN 2
                ELSE 3
            END
        """
    elif prioritize_placeholders:
        babel_orderbyclause = f"""
        ORDER BY
            CASE
                WHEN NAMES.CATEGORY IN ({prioritize_placeholders}) THEN 0
                ELSE 1
            END
        """
    elif most_common:
        babel_orderbyclause = """
        ORDER BY
            CASE
                WHEN NAMES.CATEGORY = :most_common THEN 0
                ELSE 1
            END
        """
    else:
        babel_orderbyclause = ""

    return f"""
    SELECT
        NAMES.CURIE,
        NAMES.CATEGORY,
        NAMES.NAME,
        NAMES.TAXON
        FROM SYNONYMS
        INNER JOIN NAMES ON SYNONYMS.CURIE = NAMES.CURIE
    WHERE
        {babel_levelcondtion}
        {babel_taxoncondition}
        {babel_avoidcondition}
    {babel_orderbyclause}
    """


def kg2sql(
    prioritize_placeholders: Optional[str],
    avoid_placeholders: Optional[str],
    most_common: Optional[str],
    level: str,
) -> str:

    kg2_levelcondition: str = {
        "L1": "nodes.name = :input",
        "L3": "nodes.name_simplified = :input",
    }.get(level, "")

    kg2_avoidcondition: str = (
        f"AND clusters.category NOT IN ({avoid_placeholders})"
        if avoid_placeholders
        else ""
    )

    if prioritize_placeholders and most_common:
        kg2_orderbyclause: str = f"""
        ORDER BY
            CASE
                WHEN clusters.category IN ({prioritize_placeholders}) AND clusters.category = :most_common THEN 0
                WHEN clusters.category IN ({prioritize_placeholders}) THEN 1
                WHEN clusters.category = :most_common THEN 2
                ELSE 3
            END
        """
    elif prioritize_placeholders:
        kg2_orderbyclause = f"""
        ORDER BY
            CASE
                WHEN clusters.category IN ({prioritize_placeholders}) THEN 0
                ELSE 1
            END
        """
    elif most_common:
        kg2_orderbyclause = """
        ORDER BY
            CASE
                WHEN clusters.category = :most_common THEN 0
                ELSE 1
            END
        """
    else:
        kg2_orderbyclause = ""

    return f"""
    SELECT
        clusters.cluster_id,
        clusters.category,
        clusters.name
    FROM nodes
    INNER JOIN clusters ON nodes.cluster_id = clusters.cluster_id
    WHERE
        {kg2_levelcondition}
        {kg2_avoidcondition}
    {kg2_orderbyclause}
    """


def placeholders(
    prioritize: Optional[frozenset[str]],
    avoid: Optional[frozenset[str]],
) -> tuple[Optional[str], Optional[str]]:

    prioritize_placeholders: Optional[str] = (
        ", ".join([f":prioritize{idx}" for idx in range(len(prioritize))])
        if prioritize
        else None
    )
    avoid_placeholders: Optional[str] = (
        ", ".join([f":avoid{idx}" for idx in range(len(avoid))]) if avoid else None
    )

    return (prioritize_placeholders, avoid_placeholders)


# building in a function to improve readability
def sqlparams(
    leveloneoutput: str,
    prioritize: Optional[frozenset[str]],
    avoid: Optional[frozenset[str]],
    taxon: Optional[str],
    most_common: Optional[str],
) -> dict[str, str]:

    sql_params: dict[str, str] = {"input": leveloneoutput}
    if prioritize:
        sql_params.update(
            {f"prioritize{idx}": category for idx, category in enumerate(prioritize)}
        )
    if avoid:
        sql_params.update(
            {f"avoid{idx}": category for idx, category in enumerate(avoid)}
        )
    if taxon:
        sql_params["taxon"] = taxon
    if most_common:
        sql_params["most_common"] = most_common

    return sql_params


def fullmap_struct(
    name: Any,
    curie: Any,
    preferred: Any,
    category: Any,
    taxon: Any,
    level: Any,
    db: Any,
) -> dict[str, Any]:
    return {
        name: curie,
        f"{name}_name": preferred,
        f"{name}_category": f"biolink:{category}",
        f"{name}_mapped_with_taxon": (
            f"NCBITaxon:{taxon}" if taxon else "Not applicable"
        ),
        f"{name}_mapped_with_level": level,
        f"{name}_mapped_with_database": db,
    }


subject_struct: pl.Struct = pl.Struct(
    [
        pl.Field("subject", pl.String),
        pl.Field("subject_name", pl.String),
        pl.Field("subject_category", pl.String),
        pl.Field("subject_mapped_with_taxon", pl.String),
        pl.Field("subject_mapped_with_level", pl.String),
        pl.Field("subject_mapped_with_database", pl.String),
    ]
)


object_struct: pl.Struct = pl.Struct(
    [
        pl.Field("object", pl.String),
        pl.Field("object_name", pl.String),
        pl.Field("object_category", pl.String),
        pl.Field("object_mapped_with_taxon", pl.String),
        pl.Field("object_mapped_with_level", pl.String),
        pl.Field("object_mapped_with_database", pl.String),
    ]
)


def babelresult(name: str, rows: Any, level: str, db: str = "babel") -> dict[str, Any]:
    row: Any = next(rows, {})
    return fullmap_struct(
        name,
        row.get("CURIE"),
        row.get("NAME"),
        row.get("CATEGORY"),
        row.get("TAXON"),
        level,
        db,
    )


def kg2result(name: str, rows: Any, level: str, db: str = "kg2") -> dict[str, Any]:
    row: Any = next(rows, {})
    return fullmap_struct(
        name,
        row.get("cluster_id"),
        row.get("name"),
        row.get("category"),
        None,
        level,
        db,
    )


def collectresults(
    name: str, rows: Any, level: str, db: str, sql_params: dict[str, str]
) -> Optional[dict[str, Any]]:
    if db == "babel":
        result = babelresult(name, rows, level)
    elif db == "kg2":
        result = kg2result(name, rows, level)
    else:
        raise RuntimeError(f"CODE:124 | The database ({db}) is not supported yet")
    if all(v for k, v in result.items() if k != f"{name}_mapped_with_taxon"):
        ColumnContext[str(result[f"{name}_category"])] += 1
        logger.success(f"{str(sql_params)} mapped to {str(result)}")
        return result
    else:
        return None


# generator function because multiprocessing doesn't correctly catch the error in a normal try, except
def safe_query(db: str, sql: str, sql_params: dict[str, str]) -> Iterator[Any]:
    try:
        if db == "babel":
            yield from babel.query(sql, sql_params)  # type: ignore
        elif db == "kg2":
            yield from kg2.query(sql, sql_params)  # type: ignore
        else:
            raise RuntimeError(f"CODE:126 | A method for querying {db} does not exist")
    except sqlite3.OperationalError as e:
        logger.critical(
            f"CODE:125 | {db}, {str(sql_params)} triggered the progress handler {str(e)}"
        )
        return None


@fullmap3cache.memoize()  # type: ignore
def fullmap3(
    name: str,
    unprocessedinput: Any,
    prioritize: Optional[frozenset[str]],
    avoid: Optional[frozenset[str]],
    taxon: Optional[str],
) -> dict[str, Any]:

    prioritize_placeholders, avoid_placeholders = placeholders(prioritize, avoid)

    common_categories: list[Any] = ColumnContext.most_common(1)
    most_common: Optional[str] = (
        str(common_categories[0][0])[8:] if common_categories else None
    )

    level: str = "L1"
    leveloneoutput: str = levelone(unprocessedinput)
    sql_params = sqlparams(leveloneoutput, prioritize, avoid, taxon, most_common)

    global start  # for progress handler
    start = time.time()
    sql: str = babelsql(
        prioritize_placeholders, avoid_placeholders, taxon, most_common, level
    )
    rows: Any = safe_query("babel", sql, sql_params)
    result: Optional[dict[str, Any]] = collectresults(
        name, rows, level, "babel", sql_params
    )
    if result:
        return result

    start = time.time()
    sql = kg2sql(prioritize_placeholders, avoid_placeholders, most_common, level)
    rows = safe_query("kg2", sql, sql_params)
    result = collectresults(name, rows, level, "kg2", sql_params)
    if result:
        return result

    level = "L2"
    leveltwooutput: str = leveltwo(leveloneoutput)
    sql_params["input"] = leveltwooutput

    start = time.time()
    sql = babelsql(
        prioritize_placeholders, avoid_placeholders, taxon, most_common, level
    )
    rows = safe_query("babel", sql, sql_params)
    result = collectresults(name, rows, level, "babel", sql_params)
    if result:
        return result

    level = "L3"
    levelthreeoutput: str = levelthree(leveltwooutput)
    sql_params["input"] = levelthreeoutput

    start = time.time()
    sql = babelsql(
        prioritize_placeholders, avoid_placeholders, taxon, most_common, level
    )
    rows = safe_query("babel", sql, sql_params)
    result = collectresults(name, rows, level, "babel", sql_params)
    if result:
        return result

    levelthreeoutputkg2: str = levelthree(leveloneoutput)
    sql_params["input"] = levelthreeoutputkg2

    start = time.time()
    sql = kg2sql(prioritize_placeholders, avoid_placeholders, most_common, level)
    rows = safe_query("kg2", sql, sql_params)
    result = collectresults(name, rows, level, "kg2", sql_params)
    if result:
        return result

    # for logging
    sql_params["input"] = unprocessedinput
    sql_params["curie"] = article_curie  # type: ignore
    logger.warning(f"{str(sql_params)} failed to map")
    return fullmap_struct(name, None, None, None, None, None, None)


# because the double cache header alters how they both behave
@lru_cache(maxsize=1024)
def cached_fullmap3(
    name: str,
    unprocessedinput: Any,
    prioritize: Optional[frozenset[str]],
    avoid: Optional[frozenset[str]],
    taxon: Optional[str],
) -> dict[str, Any]:
    return fullmap3(name, unprocessedinput, prioritize, avoid, taxon)  # type: ignore


def spocolumn(df: pl.DataFrame, name: str, spoconfig: Any) -> pl.DataFrame:
    name = name[7:]
    if name == "predicate":
        return new_column(df, name, "value", str(spoconfig))
    else:
        encoding_method: str = spoconfig["encoding_method"]
        value_for_encoding: str = spoconfig["value_for_encoding"]
        df = new_column(df, f"original_{name}", encoding_method, value_for_encoding)
        df = new_column(df, name, encoding_method, value_for_encoding)
        mapping_hyperparameters: dict[str, Any] = spoconfig["mapping_hyperparameters"]
        how_to_fill_column: Optional[str] = mapping_hyperparameters.get(
            "how_to_fill_column"
        )
        if how_to_fill_column:
            df = df.with_columns(pl.col(name).fill_null(strategy=how_to_fill_column))  # type: ignore
        explode_by_delimiter: Optional[str] = mapping_hyperparameters.get(
            "explode_by_delimiter"
        )
        if explode_by_delimiter:
            df = df.with_columns(pl.col(name).str.split(explode_by_delimiter)).explode(
                name
            )
        substrings_to_remove: Optional[list[str]] = mapping_hyperparameters.get(
            "substrings_to_remove"
        )
        if substrings_to_remove:
            for substring in substrings_to_remove:
                df = df.with_columns(
                    pl.col(name).str.replace_all(substring, "").alias(name)
                )
        regular_expressions: Optional[list[dict[str, Any]]] = (
            mapping_hyperparameters.get("regular_expressions")
        )
        if regular_expressions:
            for regex in regular_expressions:
                df = df.with_columns(
                    pl.col(name)
                    .str.replace_all(regex["pattern"], regex["replacement"])
                    .alias(name)
                )
        prefix: Optional[str] = mapping_hyperparameters.get("prefix")
        if prefix:
            df = df.with_columns((pl.lit(prefix) + pl.col(name)).alias(name))
        suffix: Optional[str] = mapping_hyperparameters.get("suffix")
        if suffix:
            df = df.with_columns((pl.col(name) + pl.lit(suffix)).alias(name))
        in_this_organism: Optional[str] = mapping_hyperparameters.get(
            "in_this_organism"
        )
        taxon = in_this_organism[10:] if in_this_organism else None
        classes_to_prioritize: Optional[list[str]] = mapping_hyperparameters.get(
            "classes_to_prioritize"
        )
        prioritize = (
            frozenset(priority[8:] for priority in classes_to_prioritize)
            if classes_to_prioritize
            else None
        )
        classes_to_avoid: Optional[list[str]] = mapping_hyperparameters.get(
            "classes_to_avoid"
        )
        avoid = (
            frozenset(void[8:] for void in classes_to_avoid)
            if classes_to_avoid
            else None
        )
        df = df.with_columns(
            pl.col(name)
            .map_elements(
                lambda x: cached_fullmap3(name, x, prioritize, avoid, taxon),
                return_dtype=subject_struct if name == "subject" else object_struct,
                skip_nulls=True,
            )
            .alias(f"{name}_struct")
        )
        ColumnContext.clear()  # remember to reset ColumnContext
        df = df.drop(name).unnest(f"{name}_struct")
        df = df.filter(pl.col(name).is_not_null())
    return df


pubmedmetadatacache: Cache = Cache("TABLASSERT/CACHE/PUBMEDMETADATA", max_size=1e6)


@pubmedmetadatacache.memoize()  # type: ignore  # diskcache because these caches are threadspecific
def pubmed_metadata(article_curie: str) -> dict[str, Any]:

    pubmedsql: str = """
    SELECT
        mesh.mesh_major,
        mesh.mesh,
        info.firstauthor,
        info.journal,
        info.title,
        info.year
    FROM ids
    INNER JOIN mesh ON ids.pmid = mesh.pmid
    INNER JOIN info ON ids.pmid = info.pmid
    WHERE ids.alt = :curie
    """

    global start
    start = time.time()
    rows = list(pubmed.query(pubmedsql, {"curie": article_curie[4:]}))  # type: ignore
    mesh: list[Optional[str]] = [row["mesh"] for row in rows if row]
    mesh_major: list[Optional[str]] = [row["mesh_major"] for row in rows if row]
    mesh_zip: Any = list(
        zip(mesh, mesh_major)
    )  # zip objects aren't reusable for some stupid reason so we have to list them
    domain: list[Optional[str]] = [
        "MESH:" + term for term, importance in mesh_zip if importance == "Y"
    ]
    mesh_terms: list[Optional[str]] = [
        "MESH:" + term for term, importance in mesh_zip if importance == "N"
    ]
    row = rows[0] if rows else {}
    return {
        "domain": ",".join(domain) if domain else "Not applicable",  # type: ignore
        "mesh_terms": ",".join(mesh_terms) if mesh_terms else "Not applicable",  # type: ignore
        "first_author": row.get("firstauthor", "Not applicable"),
        "journal": row.get("journal", "Not applicable"),
        "article_title": row.get("title", "Not applicable"),
        "year_published": str(row.get("year", "Not applicable")),
    }


pubmed_struct: pl.Struct = pl.Struct(
    [
        pl.Field("domain", pl.String),
        pl.Field("mesh_terms", pl.String),
        pl.Field("first_author", pl.String),
        pl.Field("journal", pl.String),
        pl.Field("article_title", pl.String),
        pl.Field("year_published", pl.String),
    ]
)

pmccaptionscache: Cache = Cache("TABLASSERT/CACHE/PMCCAPTIONS", max_size=1e6)


@pmccaptionscache.memoize()  # type: ignore
def pmc_captions(article_curie: str, filename: str) -> Optional[str]:

    pmcsql: str = """
    SELECT caption
    FROM captions
    WHERE pmc = :curie AND file = :filename
    LIMIT 1
    """

    global start
    start = time.time()
    rows = pmc.query(  # type: ignore
        pmcsql, {"curie": article_curie[7:], "filename": basename(filename)}
    )
    row: dict[str, Any] = next(rows, {})
    return row.get("caption")


@lru_cache(maxsize=32)
def is_significant(x: str, p_value_threshold: float) -> str:
    try:
        if float(x) <= p_value_threshold:
            return "YES"
        else:
            return "NO"
    except ValueError:
        if str(x) == "Not applicable":
            return "YES"
        else:
            return "NO"


FINAL_COLUMNS: list[str] = [
    "subject",
    "predicate",
    "object",
    "significant?",
    "domain",
    "mesh_terms",
    "sample_size",
    "p_value",
    "multiple_testing_correction_method",
    "assertion_strength",
    "assertion_method",
    "notes",
    "knowledge_level",
    "agent_type",
    "article_curie",
    "first_author",
    "journal",
    "article_title",
    "year_published",
    "download_link",
    "file_name",
    "extension",
    "excel_sheet",
    "extracted_from_row_number",
    "pmc_file_caption",
    "original_subject",
    "subject_name",
    "subject_category",
    "subject_mapped_with_taxon",
    "subject_mapped_with_database",
    "subject_mapped_with_level",
    "original_object",
    "object_name",
    "object_category",
    "object_mapped_with_taxon",
    "object_mapped_with_database",
    "object_mapped_with_level",
    "config_curator_name",
    "config_curator_organization",
]


def dataframing(
    subsectionmodel: dict[str, Any], graphmodel: dict[str, dict[str, Any]]
) -> pl.DataFrame:
    posix_filepath: str = subsectionmodel["posix_filepath"]
    download_hyperparameters: dict[str, Any] = subsectionmodel["location"][
        "download_hyperparameters"
    ]
    df = initate(posix_filepath, download_hyperparameters)
    df = df.rename(
        {
            old_name: (
                excel_style_column_name(idx)
                if old_name != "extracted_from_row_number"
                else old_name
            )
            for idx, old_name in enumerate(df.columns)
        }
    )
    df = new_column(
        df,
        "download_link",
        "value",
        subsectionmodel["location"]["where_to_download_data_from"],
    )
    df = new_column(df, "file_name", "value", basename(posix_filepath))
    sqlites: dict[str, str] = graphmodel["location"]["sqlite_databases"]
    maxtime: float = graphmodel["hyperparameters"]["sql_progress_handler_timeout"]
    df = new_column(df, "extension", "value", download_hyperparameters["extension"])
    df = new_column(
        df,
        "excel_sheet",
        "value",
        download_hyperparameters.get("which_excel_sheet_to_use"),
    )
    provenance: dict[str, str] = subsectionmodel["provenance"]
    global article_curie
    article_curie = provenance["article_curie"]  # type: ignore
    df = new_column(df, "article_curie", "value", article_curie)  # type: ignore
    df = new_column(
        df, "config_curator_name", "value", provenance["config_curator_name"]
    )
    df = new_column(
        df,
        "config_curator_organization",
        "value",
        provenance["config_curator_organization"],
    )
    global pmc  # databases are global to enable caching because they're unhashable types
    pmc = connect(sqlites["pmc"], maxtime)  # type: ignore
    df = new_column(
        df,
        "pmc_file_caption",
        "value",
        pmc_captions(article_curie, posix_filepath),  # type: ignore
    )
    global pubmed
    pubmed = connect(sqlites["pubmed"], maxtime)  # type: ignore
    df = df.with_columns(
        pl.col("article_curie")
        .map_elements(
            lambda x: pubmed_metadata(x), return_dtype=pubmed_struct, skip_nulls=True
        )
        .alias("pubmed_struct")
    )
    df = df.unnest("pubmed_struct")
    reindexing: Optional[list[dict[str, Any]]] = subsectionmodel["reindexing"]
    if reindexing:
        for operation in reindexing:
            df = apply_reindexing(df, operation, "before")
    attributes: dict[str, Any] = subsectionmodel["attributes"]
    for name, attribute in attributes.items():
        df = process_attribute(df, name, attribute)
    triple: dict[str, Any] = subsectionmodel["triple"]
    global babel
    babel = connect(sqlites["babel"], maxtime)  # type: ignore
    global kg2
    kg2 = connect(sqlites["kg2"], maxtime)  # type: ignore
    for name, spoconfig in triple.items():
        df = spocolumn(df, name, spoconfig)
    if reindexing:
        for operation in reindexing:
            df = apply_reindexing(df, operation, "after")
    df = new_column(df, "knowledge_level", "value", "statistical_association")
    df = new_column(df, "agent_type", "value", "data_analysis_pipeline")
    p_value_threshold: float = graphmodel["hyperparameters"]["maximum_p_value_in_graph"]
    df = df.with_columns(
        pl.col("p_value")
        .map_elements(
            lambda x: is_significant(x, p_value_threshold), return_dtype=pl.String
        )
        .alias("significant?")
    )
    df = df.select(FINAL_COLUMNS)
    return df.drop_nulls()
