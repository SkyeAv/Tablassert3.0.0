from __future__ import annotations

from operator import add
from typing import TYPE_CHECKING, Optional

import lazy_loader as Lazy

from tablassert.enums import Categories
from tablassert.log import cat

logger = cat("FULLMAP")

if TYPE_CHECKING:
    import polars as pl
    import polars_hash as plh
else:
    pl = Lazy.load("polars")
    plh = Lazy.load("polars_hash")


SHARDS: int = 10


def empty_matches(column_context: bool) -> pl.DataFrame:
    # ? Creates Empty Fullmap Matches DataFrame With Query Schema
    schema: dict[str, object] = {
        "term": pl.String,
        "CURIE": pl.String,
        "PREFERRED_NAME": pl.String,
        "CATEGORY_NAME": pl.String,
        "TAXON_ID": pl.Int64,
        "SOURCE_NAME": pl.String,
        "SOURCE_VERSION": pl.String,
        "NLP_LEVEL": pl.Int64,
        "PR": pl.Int64,
    }

    if column_context:
        schema["FREQUENCY"] = pl.Int64

    return pl.DataFrame(schema=schema)  # pyright: ignore


def distinct(lf: pl.LazyFrame, l1: str, l2: str, col: str = "term") -> pl.LazyFrame:
    # ? Extract Unique Terms From Two Text Normalization Columns As LazyFrame
    t1: pl.LazyFrame = lf.select(pl.col(l1).alias(col)).unique()
    t1 = t1.with_columns(pl.lit(1).alias("nlp level"))

    t2: pl.LazyFrame = lf.select(pl.col(l2).alias(col)).unique()
    t2 = t2.with_columns(pl.lit(2).alias("nlp level"))

    terms: pl.LazyFrame = pl.concat([t1, t2]).unique(subset=[col], keep="first")

    bad: str = r"^\d+$|^(none|nan|na|null|unknown|not applicable|p value|variable|result|exposure|expression|symbol)$|^$"
    terms = terms.filter(~pl.col(col).str.contains(bad))
    return terms.with_columns((plh.col(col).nchash.xxhash64() % SHARDS).alias("shard"))  # pyright: ignore


def query_builder(prioritize: Optional[list[Categories]], avoid: Optional[list[Categories]], taxon: Optional[str]) -> str:
    # ? Build Query With UNION For Better Index Utilization
    base: str = """
    SELECT
        PA.term,
        CU.CURIE,
        CU.PREFERRED_NAME,
        CA.CATEGORY_NAME,
        CU.TAXON_ID,
        SO.SOURCE_NAME,
        SO.SOURCE_VERSION,
        PA."nlp level" AS NLP_LEVEL,
        CASE
            {priority_case}
            ELSE 50
        END * CASE
            WHEN LOWER(CU.PREFERRED_NAME) = PA.term THEN 1
            ELSE 10
        END AS PR
    FROM SYNONYMS SY
    JOIN SOURCES SO ON SY.SOURCE_ID = SO.SOURCE_ID
    JOIN CURIES CU ON SY.CURIE_ID = CU.CURIE_ID
    JOIN CATEGORIES CA ON CU.CATEGORY_ID = CA.CATEGORY_ID
        {avoid_filter}
    JOIN PARQUET PA ON PA.term = SY.SYNONYM
    {taxon_filter}
"""

    priority_list: str = ", ".join("'" + x + "'" for x in prioritize) if prioritize else ""
    priority_case: str = f"WHEN CA.CATEGORY_NAME IN ({priority_list}) THEN 1" if prioritize else "WHEN TRUE THEN 50"
    avoid_list: str = ", ".join("'" + x + "'" for x in avoid) if avoid else ""
    avoid_filter: str = f"AND CA.CATEGORY_NAME NOT IN ({avoid_list})" if avoid else ""
    taxon_filter: str = f"WHERE CU.TAXON_ID = {taxon} OR CA.CATEGORY_NAME != 'Gene'" if taxon else ""

    return base.format(priority_case=priority_case, avoid_filter=avoid_filter, taxon_filter=taxon_filter)


def query_shard(conn: object, df: pl.DataFrame, query: str) -> pl.DataFrame:
    # ? Query A Single Shard Database For Distinct Terms
    conn.register("PARQUET", df.to_arrow())  # pyright: ignore
    return conn.execute(query).pl()  # pyright: ignore


def deduplicate_result(result: pl.DataFrame, column_context: bool) -> pl.DataFrame:
    sort_by: list[str] = ["term", "PR", "NLP_LEVEL"]
    descending: list[bool] = [False, False, False]

    if column_context:
        frequency: pl.DataFrame = result.group_by("CATEGORY_NAME").agg(pl.len().alias("FREQUENCY"))
        result = result.join(frequency, on="CATEGORY_NAME", how="left")

        sort_by += ["FREQUENCY"]
        descending += [True]

    result = result.sort(sort_by, descending=descending)
    return result.unique(subset=["term"], keep="first")


def query_distinct(
    lf: pl.LazyFrame,
    conns: list[object],
    taxon: Optional[str],
    prioritize: Optional[list[Categories]],
    avoid: Optional[list[Categories]],
    column_context: bool,
) -> pl.DataFrame:
    # ? Query All Shard Databases In Parallel Using Thread Pool
    # * Added Column Prioritization Logic From 4.2.0
    shards: dict[tuple[str], pl.DataFrame] = lf.collect().partition_by("shard", as_dict=True)
    if len(shards) == 0:
        return empty_matches(column_context)

    results: list[pl.DataFrame] = []
    query: str = query_builder(prioritize, avoid, taxon)
    for shard, df in shards.items():
        shard_number: int = int(shard[0])
        conn: object = conns[shard_number]  # type: ignore

        results += [query_shard(conn, df, query)]

    if len(results) == 0:
        return empty_matches(column_context)

    result: pl.DataFrame = pl.concat(results, how="vertical")
    return deduplicate_result(result, column_context)


def log_unmatched(col: str, terms: pl.LazyFrame, matches: pl.DataFrame, section_hash: Optional[str], config_file: Optional[str]) -> None:
    # * Log Unmatched Entities
    level_one: pl.LazyFrame = terms.filter(pl.col("nlp level") == 1)
    antimatches: pl.LazyFrame = level_one.join(matches.lazy().select("term"), left_on="term", right_on="term", how="anti")

    # ! Collection Point: Requires Eager
    unnmatched: pl.DataFrame = antimatches.select("term").unique().collect()
    if unnmatched.height > 0:
        for term in unnmatched.get_column("term").to_list():
            logger.info(f"FAILED | HASH: {section_hash} | CONFIG: {config_file} | COL: {col} | VALUE: {term!r}")


def resolve(
    lf: pl.LazyFrame,
    col: str,
    conns: list[object],
    taxon: Optional[str] = None,
    prioritize: Optional[list[Categories]] = None,
    avoid: Optional[list[Categories]] = None,
    log: bool = True,
    section_hash: Optional[str] = None,
    config_file: Optional[str] = None,
    column_context: bool = True,
    tag: str = " two",
) -> pl.LazyFrame:
    # ? Case Dependant, Provenance Rich Name Entity Recognition
    l1: str = col
    l2: str = add(l1, tag)

    terms: pl.LazyFrame = distinct(lf, l1, l2)
    matches: pl.DataFrame = query_distinct(terms, conns, taxon, prioritize, avoid, column_context)

    if log:
        log_unmatched(col, terms, matches, section_hash, config_file)

    # ! Collection Point: Join After DuckDB Query, Then Re-Lazy
    df: pl.DataFrame = lf.collect()
    result: pl.DataFrame = df.join(matches.filter(pl.col("NLP_LEVEL").eq(1)), left_on=l1, right_on="term", how="left", suffix=" l1")

    l2_matches: pl.DataFrame = matches.filter(pl.col("NLP_LEVEL").eq(2))
    result = result.join(l2_matches, left_on=l2, right_on="term", how="left", suffix=" l2")

    result = result.with_columns(
        [
            pl.when(pl.col("CURIE").is_not_null()).then(pl.col("CURIE")).otherwise(pl.col("CURIE l2")).alias(col),
            pl.when(pl.col("PREFERRED_NAME").is_not_null())
            .then(pl.col("PREFERRED_NAME"))
            .otherwise(pl.col("PREFERRED_NAME l2"))
            .alias(add(col, " name")),
            pl.when(pl.col("CATEGORY_NAME").is_not_null())
            .then(add(pl.lit("biolink:"), pl.col("CATEGORY_NAME")))
            .otherwise(add(pl.lit("biolink:"), pl.col("CATEGORY_NAME l2")))
            .alias(add(col, " category")),
            pl.when(pl.col("TAXON_ID").is_not_null())
            .then(add(pl.lit("NCBITaxon:"), pl.col("TAXON_ID").cast(pl.String)))
            .otherwise(add(pl.lit("NCBITaxon:"), pl.col("TAXON_ID l2").cast(pl.String)))
            .alias(add(col, " taxon")),
            pl.when(pl.col("SOURCE_NAME").is_not_null()).then(pl.col("SOURCE_NAME")).otherwise(pl.col("SOURCE_NAME l2")).alias(add(col, " source")),
            pl.when(pl.col("SOURCE_VERSION").is_not_null())
            .then(pl.col("SOURCE_VERSION"))
            .otherwise(pl.col("SOURCE_VERSION l2"))
            .alias(add(col, " source version")),
            pl.when(pl.col("NLP_LEVEL").is_not_null()).then(pl.col("NLP_LEVEL")).otherwise(pl.col("NLP_LEVEL l2")).alias(add(col, " nlp level")),
        ]
    )

    result = result.select(pl.exclude(r"^(CURIE|PREFERRED_NAME|CATEGORY_NAME|TAXON_ID|SOURCE_NAME|SOURCE_VERSION|NLP_LEVEL|PR|FREQUENCY)( l2)?$"))
    result = result.select(pl.exclude(add(col, " two")))
    result = result.with_columns(pl.col(add(col, " taxon")).replace("NCBITaxon:0", None))
    result = result.filter(pl.col(col).is_not_null())

    return result.lazy()
