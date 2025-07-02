from typing import Any, Optional, Union
from sqlite_utils import Database
from functools import lru_cache
from collections import Counter
from spacy.tokens import Token
from os.path import basename
from diskcache import Cache
from loguru import logger
import polars as pl
import spacy
import math
import time
import re


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


def load_csv(
    posix_filepath: str, download_hyperparameters: dict[str, Any]
) -> pl.DataFrame:
    delimiter: str = download_hyperparameters["file_delimiter"]
    df: pl.DataFrame = pl.read_csv(
        source=posix_filepath, separator=delimiter, has_header=False, infer_schema=False
    )
    return slicing(df, download_hyperparameters)


EXCEL_ENGINE: str = "xlsx2csv"


def load_excel(
    posix_filepath: str, download_hyperparameters: dict[str, Any]
) -> pl.DataFrame:
    sheetname: str = download_hyperparameters["which_excel_sheet_to_use"]
    df = pl.read_excel(source=posix_filepath, sheet_name=sheetname, engine=EXCEL_ENGINE, has_header=False, infer_schema=False)  # type: ignore
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
        value_for_encoding = "not applicable"
    if encoding_method == "value":
        return df.with_columns(pl.lit(value_for_encoding).alias(column))
    elif encoding_method == "column_of_values":
        return df.with_columns(pl.col(value_for_encoding).alias(column))
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


def apply_math_module(
    df: pl.DataFrame, name: str, transformation: dict[str, Any]
) -> pl.DataFrame:
    transformation_operation = lambda x: getattr(math, transformation["attribute"])(
        *[arg if arg is not None else x for arg in transformation["arguments"]]
    )
    return df.with_columns(
        pl.col(name).cast(pl.Float64).map_elements(transformation_operation).alias(name)
    )


def process_attribute(
    df: pl.DataFrame, name: str, attribute: dict[str, Any]
) -> pl.DataFrame:
    if name != "notes":
        df = new_column(
            df,
            name,
            attribute.get("encoding_method"),
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
fullmap3cache: Cache = Cache("TABLASSERT/CACHE/FULLMAP3", max_size=3e10)

start: float = 0.0  # default for typechecking


def progress_handler(maxtime: float = 1.10) -> int:
    if (start - time.time()) >= maxtime:
        return 1
    return 0


def connect(sqlitepath: str) -> Database:
    db: Database = Database(sqlitepath)
    db.enable_wal()  # type: ignore
    conn = db.conn
    conn.set_progress_handler(lambda: progress_handler(), 1)
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
        f"{name}_mapped_with_taxon": f"NCBITaxon:{taxon}",
        f"{name}_mapped_with_level": level,
        f"{name}_mapped_with_database": db,
    }


def babelresults(
    name: str, rows: Any, level: str, db: str = "babel"
) -> Optional[dict[str, Any]]:
    row: Any = next(rows, {})
    struct: dict[str, Any] = fullmap_struct(
        name,
        row.get("CURIE"),
        row.get("NAME"),
        row.get("CATEGORY"),
        row.get("TAXON"),
        level,
        db,
    )
    return (
        struct
        if all(
            value for key, value in struct.items() if key != f"{name}_mapped_with_taxon"
        )
        else None
    )


def kg2results(
    name: str, rows: Any, level: str, db: str = "kg2"
) -> Optional[dict[str, Any]]:
    row: Any = next(rows, {})
    struct = fullmap_struct(
        name,
        row.get("cluster_id"),
        row.get("name"),
        row.get("category"),
        None,
        level,
        db,
    )
    return (
        struct
        if all(
            value for key, value in struct.items() if key != f"{name}_mapped_with_taxon"
        )
        else None
    )


# dynamic query build because f strings only evaluate once at their creation
def babelsql(
    prioritize_placeholders: Optional[str],
    avoid_placeholders: Optional[str],
    taxon: Optional[str],
    most_common: Optional[str],
    level: str,
) -> str:

    babel_levelcondtion: dict[str, str] = {
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

    kg2_levelcondtion: dict[str, str] = {
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

@fullmap3cache.memoize()  # type: ignore
def fullmap3(
    name: str,
    unprocessedinput: Any,
    prioritize: Optional[frozenset[str]],
    avoid: Optional[frozenset[str]],
    taxon: Optional[str],
) -> dict[str, Any]:

    prioritize_placeholders: Optional[str] = (
        ", ".join([f":prioritize{idx}" for idx in range(len(prioritize))])
        if prioritize
        else None
    )
    avoid_placeholders: Optional[str] = (
        ", ".join([f":avoid{idx}" for idx in range(len(avoid))]) if avoid else None
    )
    common_categories: list[Any] = ColumnContext.most_common(1)
    most_common: Optional[str] = (
        str(common_categories[0][0]) if common_categories else None
    )
    leveloneoutput: str = levelone(unprocessedinput)
    sql_params: dict[str, str] = {"input": leveloneoutput}
    level: str = "L1"

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

    global start  # for progress handler
    start = time.time()
    rows: Any = babel.query(babelsql(prioritize_placeholders, avoid_placeholders, taxon, most_common, level), sql_params)  # type: ignore
    result: Optional[dict[str, Any]] = babelresults(name, rows, level)
    if result:
        ColumnContext[str(result[f"{name}_category"])] += 1
        return result

    start = time.time()
    rows = kg2.query(kg2sql(prioritize_placeholders, avoid_placeholders, most_common, level), sql_params)  # type: ignore
    result = kg2results(name, rows, level)
    if result:
        ColumnContext[str(result[f"{name}_category"])] += 1
        return result

    level = "L2"
    leveltwooutput: str = leveltwo(leveloneoutput)
    sql_params["input"] = leveltwooutput

    start = time.time()
    rows = babel.query(babelsql(prioritize_placeholders, avoid_placeholders, taxon, most_common, level), sql_params)  # type: ignore
    result = babelresults(name, rows, level)
    if result:
        ColumnContext[str(result[f"{name}_category"])] += 1
        return result

    start = time.time()
    rows = kg2.query(kg2sql(prioritize_placeholders, avoid_placeholders, most_common, level), sql_params)  # type: ignore
    result = kg2results(name, rows, level)
    if result:
        ColumnContext[str(result[f"{name}_category"])] += 1
        return result

    level = "L3"
    levelthreeoutput: str = levelthree(leveltwooutput)
    sql_params["input"] = leveltwooutput

    start = time.time()
    rows = babel.query(babelsql(prioritize_placeholders, avoid_placeholders, taxon, most_common, level), sql_params)  # type: ignore
    result = babelresults(name, rows, level)
    if result:
        ColumnContext[str(result[f"{name}_category"])] += 1
        return result

    levelthreeoutputkg2: str = levelthree(leveloneoutput)
    sql_params["input"] = levelthreeoutputkg2

    start = time.time()
    rows = kg2.query(kg2sql, sql_params)  # type: ignore
    result = kg2results(name, rows, level)
    if result:
        ColumnContext[str(result[f"{name}_category"])] += 1
        return result

    # add loguru logging here
    return fullmap_struct(name, None, None, None, None, None, None)


def spocolumn(df: pl.DataFrame, name: str, spoconfig: Any) -> pl.DataFrame:
    name = name[7:]
    if name == "predicate":
        return new_column(df, name, "value", str(spoconfig))
    else:
        encoding_method: str = spoconfig["encoding_method"]
        value_for_encoding: str = spoconfig["value_for_encoding"]
        df = new_column(df, f"origonal_{name}", encoding_method, value_for_encoding)
        df = new_column(df, name, encoding_method, value_for_encoding)
        mapping_hyperparameters: dict[str, Any] = spoconfig["mapping_hyperparameters"]
        how_to_fill_column: Optional[str] = str(
            mapping_hyperparameters.get("how_to_fill_column")
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
        substrings_to_remove: Optional[list[str]] = mapping_hyperparameters.get(
            "substrings_to_remove"
        )
        if substrings_to_remove:
            for substring in substrings_to_remove:
                df = df.with_columns(
                    pl.col(name).str.replace_all(substring, "").alias(name)
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
        taxon = in_this_organism[9:] if in_this_organism else None
        classes_to_prioritize: Optional[list[str]] = mapping_hyperparameters.get(
            "classes_to_prioritize"
        )
        prioritize = (
            frozenset(priority[7:] for priority in classes_to_prioritize)
            if classes_to_prioritize
            else None
        )
        classes_to_avoid: Optional[list[str]] = mapping_hyperparameters.get(
            "classes_to_avoid"
        )
        avoid = (
            frozenset(void[7:] for void in classes_to_avoid)
            if classes_to_avoid
            else None
        )
        # introduce fullmap3


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
            old_name: excel_style_column_name(idx)
            for idx, old_name in enumerate(df.columns)
        }
    )
    df = new_column(
        df,
        "download_link",
        "value",
        graphmodel["location"]["where_to_download_data_from"],
    )
    df = new_column(df, "file_name", "value", basename(posix_filepath))
    sqlites: dict[str, str] = graphmodel["location"]["sqlite_databases"]
    global pmc  # databases are global to enable caching because they're unhashable types
    pmc: Database = connect(sqlites["pmc"])  # type: ignore
    # get filecaptions
    df = new_column(df, "file_caption", "value", basename(posix_filepath))
    df = new_column(df, "extension", "value", download_hyperparameters["extension"])
    df = new_column(
        df,
        "excel_sheet",
        "value",
        download_hyperparameters.get("which_excel_sheet_to_use"),
    )
    provenance: dict[str, str] = subsectionmodel["provenance"]
    df = new_column(df, "article_curie", "value", provenance["article_curie"])
    df = new_column(
        df, "config_curator_name", "value", provenance["config_curator_name"]
    )
    df = new_column(
        df,
        "config_curator_organization",
        "value",
        provenance["config_curator_organization"],
    )
    global pubmed
    pubmed: Database = connect(sqlites["pubmed"])  # type: ignore
    # get pubmed_metadata
    reindexing: Optional[list[dict[str, Any]]] = subsectionmodel["reindexing"]
    if reindexing:
        for operation in reindexing:
            df = apply_reindexing(df, operation, "before")
    attributes: dict[str, Any] = subsectionmodel["attributes"]
    for name, attribute in attributes.items():
        df = process_attribute(df, name, attribute)
    triple: dict[str, Any] = subsectionmodel["provenance"]
    global babel
    babel: Database = connect(sqlites["babel"])  # type: ignore
    global kg2
    kg2: Database = connect(sqlites["kg2"])  # type: ignore
    for name, spoconfig in triple.items():
        df = spocolumn(df, name, spoconfig)
    if reindexing:
        for operation in reindexing:
            df = apply_reindexing(df, operation, "after")
    return df
