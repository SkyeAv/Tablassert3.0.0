from __future__ import annotations

from multiprocessing.pool import ThreadPool
from operator import add
from typing import TYPE_CHECKING, Optional

import lazy_loader as Lazy

from tablassert.enums import Categories
from tablassert.log import logger

if TYPE_CHECKING:
    import polars as pl
    import polars_hash as plh
else:
    pl = Lazy.load("polars")
    plh = Lazy.load("polars_hash")


SHARDS: int = 16


def distinct(lf: pl.LazyFrame, l0: str, l1: str, col: str = "term") -> pl.LazyFrame:
    # ? Extract Unique Terms From Two Text Normalization Columns As LazyFrame
    t0: pl.LazyFrame = lf.select(pl.col(l0).alias(col)).unique()
    t0 = t0.with_columns(pl.lit(0).alias("nlp level"))

    t1: pl.LazyFrame = lf.select(pl.col(l1).alias(col)).unique()
    t1 = t1.with_columns(pl.lit(1).alias("nlp level"))

    terms: pl.LazyFrame = pl.concat([t0, t1]).unique(subset=[col], keep="first")

    bad: str = r"^\d+$|^(none|nan|na|null|unknown)$|^$"
    terms = terms.filter(~pl.col(col).str.contains(bad))
    return terms.with_columns((pl.col(col).chash.xxhash64() % SHARDS).alias("shard"))  # pyright: ignore


def query_builder(
    prioritize: Optional[list[Categories]], avoid: Optional[list[Categories]], taxon: Optional[str]
) -> str:
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

    priority_case: str = (
        f"WHEN CA.CATEGORY_NAME IN ({', '.join(f"'{x}'" for x in prioritize)}) THEN 1"
        if prioritize
        else "WHEN TRUE THEN 50"
    )
    avoid_filter: str = f"AND CA.CATEGORY_NAME NOT IN ({', '.join(f"'{x}'" for x in avoid)})" if avoid else ""
    taxon_filter: str = f"WHERE CU.TAXON_ID = {taxon} OR CA.CATEGORY_NAME != 'Gene'" if taxon else ""

    return base.format(priority_case=priority_case, avoid_filter=avoid_filter, taxon_filter=taxon_filter)


def query_shard(conn: object, df: pl.DataFrame, query: str, column_context: bool) -> pl.DataFrame:
    # ? Query A Single Shard Database For Distinct Terms
    conn.register("PARQUET", df.to_arrow())  # pyright: ignore
    result: pl.DataFrame = conn.execute(query).pl()  # pyright: ignore

    sort_by: list[str] = ["term", "PR", "NLP_LEVEL"]
    descending: list[bool] = [False, False, False]

    if column_context:
        frequency: pl.DataFrame = result.group_by("CATEGORY_NAME").agg(pl.len().alias("FREQUENCY"))
        result = result.join(frequency, on="CATEGORY_NAME", how="left")

        sort_by += ["FREQUENCY"]
        descending += [True]

    result = result.sort(sort_by, descending=descending)
    result = result.unique(subset=["term"], keep="first")
    return result


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
    shards: dict[tuple[str], pl.DataFrame] = (
        lf.sort("shard").collect().partition_by("shard", maintain_order=True, as_dict=True)
    )
    query: str = query_builder(prioritize, avoid, taxon)

    args: list[tuple[object, pl.DataFrame, str, bool]] = [
        (conns[int(shard[0])], df, query, column_context) for shard, df in shards.items()
    ]
    results: list[pl.DataFrame] = ThreadPool(SHARDS).starmap(query_shard, args)

    return pl.concat(results, how="vertical")


def log_unmatched(
    col: str, terms: pl.LazyFrame, matches: pl.DataFrame, section_hash: Optional[str], config_file: Optional[str]
) -> None:
    # * Log Unmatched Entities
    antimatches: pl.LazyFrame = terms.join(matches.lazy().select("term"), left_on="term", right_on="term", how="anti")

    # ! Collection Point: Requires Eager
    unnmatched: pl.DataFrame = antimatches.select("term").unique().collect()
    if unnmatched.height > 0:
        for term in unnmatched.get_column("term").to_list():
            logger.info(
                f"FAILED FULLMAP | STORE: {section_hash} | CONFIG: {config_file} | COL: {col} | VALUE: {term!r}"
            )


def version4(
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
    l0: str = col
    l1: str = add(l0, tag)

    terms: pl.LazyFrame = distinct(lf, l0, l1)
    matches: pl.DataFrame = query_distinct(terms, conns, taxon, prioritize, avoid, column_context)

    if log:
        log_unmatched(col, terms, matches, section_hash, config_file)

    # ! Collection Point: Join After DuckDB Query, Then Re-Lazy
    df: pl.DataFrame = lf.collect()
    result: pl.DataFrame = df.join(
        matches.filter(pl.col("NLP_LEVEL").eq(0)), left_on=l0, right_on="term", how="left", suffix=" l0"
    )

    l1_matches: pl.DataFrame = matches.filter(pl.col("NLP_LEVEL").eq(1))
    result = result.join(l1_matches, left_on=l1, right_on="term", how="left", suffix=" l1")

    result = result.with_columns(
        [
            pl.when(pl.col("CURIE").is_not_null()).then(pl.col("CURIE")).otherwise(pl.col("CURIE l1")).alias(col),
            pl.when(pl.col("PREFERRED_NAME").is_not_null())
            .then(pl.col("PREFERRED_NAME"))
            .otherwise(pl.col("PREFERRED_NAME l1"))
            .alias(add(col, " name")),
            pl.when(pl.col("CATEGORY_NAME").is_not_null())
            .then(add(pl.lit("biolink:"), pl.col("CATEGORY_NAME")))
            .otherwise(add(pl.lit("biolink:"), pl.col("CATEGORY_NAME l1")))
            .alias(add(col, " category")),
            pl.when(pl.col("TAXON_ID").is_not_null())
            .then(add(pl.lit("NCBITaxon:"), pl.col("TAXON_ID").cast(pl.String)))
            .otherwise(add(pl.lit("NCBITaxon:"), pl.col("TAXON_ID l1").cast(pl.String)))
            .alias(add(col, " taxon")),
            pl.when(pl.col("SOURCE_NAME").is_not_null())
            .then(pl.col("SOURCE_NAME"))
            .otherwise(pl.col("SOURCE_NAME l1"))
            .alias(add(col, " source")),
            pl.when(pl.col("SOURCE_VERSION").is_not_null())
            .then(pl.col("SOURCE_VERSION"))
            .otherwise(pl.col("SOURCE_VERSION l1"))
            .alias(add(col, " source version")),
            pl.when(pl.col("NLP_LEVEL").is_not_null())
            .then(pl.col("NLP_LEVEL"))
            .otherwise(pl.col("NLP_LEVEL l1"))
            .alias(add(col, " nlp level")),
        ]
    )

    result = result.select(
        pl.exclude(
            r"^(CURIE|PREFERRED_NAME|CATEGORY_NAME|TAXON_ID|SOURCE_NAME|SOURCE_VERSION|NLP_LEVEL|PR|FREQUENCY)( l1)?$"
        )
    )
    result = result.select(pl.exclude(add(col, " two")))
    result = result.with_columns(pl.col(add(col, " taxon")).replace("NCBITaxon:0", None))
    result = result.filter(pl.col(col).is_not_null())

    return result.lazy()
