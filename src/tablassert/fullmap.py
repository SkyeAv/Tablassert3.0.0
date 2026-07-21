from __future__ import annotations

from operator import add
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import lazy_loader as Lazy

from tablassert import rs
from tablassert.enums import Categories
from tablassert.log import cat

logger = cat("FULLMAP")

if TYPE_CHECKING:
    import polars as pl
else:
    pl = Lazy.load("polars")


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
    t1 = t1.with_columns(pl.lit(1).alias("nlp_level"))

    t2: pl.LazyFrame = lf.select(pl.col(l2).alias(col)).unique()
    t2 = t2.with_columns(pl.lit(2).alias("nlp_level"))

    terms: pl.LazyFrame = pl.concat([t1, t2]).unique(subset=[col], keep="first")

    bad: str = r"^\d+$|^(none|nan|na|null|unknown|not applicable|p_value|variable|result|exposure|expression|symbol)$|^$"
    terms = terms.filter(~pl.col(col).str.contains(bad))
    return terms


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
    db: Path,
    taxon: Optional[str],
    prioritize: Optional[list[Categories]],
    avoid: Optional[list[Categories]],
    column_context: bool,
    threads: Optional[int] = None,
) -> pl.DataFrame:
    # ? Query The Embedded Fullmap Database For Distinct Terms
    # * Added Column Prioritization Logic From 4.2.0
    terms: pl.DataFrame = lf.collect()
    if terms.height == 0:
        return empty_matches(column_context)

    rows: list[dict[str, object]] = rs.lookup_fullmap_terms(db, terms.get_column("term").to_list(), threads=threads)
    if len(rows) == 0:
        return empty_matches(column_context)

    result: pl.DataFrame = pl.DataFrame(rows).join(terms, on="term", how="inner").rename({"nlp_level": "NLP_LEVEL"})
    if avoid:
        avoid_values: list[str] = [x.value for x in avoid]
        result = result.filter(~pl.col("CATEGORY_NAME").is_in(avoid_values))
    if taxon:
        taxon_id: int = int(taxon)
        result = result.filter((pl.col("TAXON_ID") == taxon_id) | (pl.col("CATEGORY_NAME") != Categories.GENE.value))
    if result.height == 0:
        return empty_matches(column_context)

    if prioritize:
        priority_values: list[str] = [x.value for x in prioritize]
        priority: pl.Expr = pl.when(pl.col("CATEGORY_NAME").is_in(priority_values)).then(pl.lit(1)).otherwise(pl.lit(50))
    else:
        priority = pl.lit(50)
    result = result.with_columns(
        (priority * pl.when(pl.col("PREFERRED_NAME").str.to_lowercase() == pl.col("term")).then(pl.lit(1)).otherwise(pl.lit(10))).alias("PR")
    )
    return deduplicate_result(result, column_context)


def fullmap_db_path(fullmap: Path) -> Path:
    # ? Resolves Existing Fullmap Base Paths To The Embedded Redb File
    if fullmap.is_file() or fullmap.suffix == ".redb":
        return fullmap
    direct: Path = fullmap / "fullmap.redb"
    if direct.is_file():
        return direct
    return fullmap / "data" / "fullmap.redb"


def log_unmatched(col: str, terms: pl.LazyFrame, matches: pl.DataFrame, section_hash: Optional[str], config_file: Optional[str]) -> None:
    # * Log Unmatched Entities
    level_one: pl.LazyFrame = terms.filter(pl.col("nlp_level") == 1)
    antimatches: pl.LazyFrame = level_one.join(matches.lazy().select("term"), left_on="term", right_on="term", how="anti")

    # ! Collection Point: Requires Eager
    unnmatched: pl.DataFrame = antimatches.select("term").unique().collect()
    if unnmatched.height > 0:
        for term in unnmatched.get_column("term").to_list():
            logger.info("Unresolved term in {config} ({hash}) col {col}: {term!r}", config=config_file, hash=section_hash, col=col, term=term)


def resolve(
    lf: pl.LazyFrame,
    col: str,
    db: Path,
    taxon: Optional[str] = None,
    prioritize: Optional[list[Categories]] = None,
    avoid: Optional[list[Categories]] = None,
    log: bool = True,
    section_hash: Optional[str] = None,
    config_file: Optional[str] = None,
    column_context: bool = True,
    tag: str = "_two",
    threads: Optional[int] = None,
) -> pl.LazyFrame:
    # ? Case Dependant, Provenance Rich Name Entity Recognition
    l1: str = col
    l2: str = add(l1, tag)

    terms: pl.LazyFrame = distinct(lf, l1, l2)
    matches: pl.DataFrame = query_distinct(terms, db, taxon, prioritize, avoid, column_context, threads=threads)

    if log:
        log_unmatched(col, terms, matches, section_hash, config_file)

    # ! Collection Point: Join After DuckDB Query, Then Re-Lazy
    df: pl.DataFrame = lf.collect()
    result: pl.DataFrame = df.join(matches.filter(pl.col("NLP_LEVEL").eq(1)), left_on=l1, right_on="term", how="left", suffix="_l1")

    l2_matches: pl.DataFrame = matches.filter(pl.col("NLP_LEVEL").eq(2))
    result = result.join(l2_matches, left_on=l2, right_on="term", how="left", suffix="_l2")

    result = result.with_columns(
        [
            pl.when(pl.col("CURIE").is_not_null()).then(pl.col("CURIE")).otherwise(pl.col("CURIE_l2")).alias(col),
            pl.when(pl.col("PREFERRED_NAME").is_not_null())
            .then(pl.col("PREFERRED_NAME"))
            .otherwise(pl.col("PREFERRED_NAME_l2"))
            .alias(add(col, "_name")),
            pl.when(pl.col("CATEGORY_NAME").is_not_null())
            .then(add(pl.lit("biolink:"), pl.col("CATEGORY_NAME")))
            .otherwise(add(pl.lit("biolink:"), pl.col("CATEGORY_NAME_l2")))
            .alias(add(col, "_category")),
            pl.when(pl.col("TAXON_ID").is_not_null())
            .then(add(pl.lit("NCBITaxon:"), pl.col("TAXON_ID").cast(pl.String)))
            .otherwise(add(pl.lit("NCBITaxon:"), pl.col("TAXON_ID_l2").cast(pl.String)))
            .alias(add(col, "_taxon")),
            pl.when(pl.col("SOURCE_NAME").is_not_null()).then(pl.col("SOURCE_NAME")).otherwise(pl.col("SOURCE_NAME_l2")).alias(add(col, "_source")),
            pl.when(pl.col("SOURCE_VERSION").is_not_null())
            .then(pl.col("SOURCE_VERSION"))
            .otherwise(pl.col("SOURCE_VERSION_l2"))
            .alias(add(col, "_source_version")),
            pl.when(pl.col("NLP_LEVEL").is_not_null()).then(pl.col("NLP_LEVEL")).otherwise(pl.col("NLP_LEVEL_l2")).alias(add(col, "_nlp_level")),
        ]
    )

    result = result.select(pl.exclude(r"^(CURIE|PREFERRED_NAME|CATEGORY_NAME|TAXON_ID|SOURCE_NAME|SOURCE_VERSION|NLP_LEVEL|PR|FREQUENCY)(_l2)?$"))
    result = result.select(pl.exclude(add(col, "_two")))
    result = result.with_columns(pl.col(add(col, "_taxon")).replace("NCBITaxon:0", None))
    result = result.filter(pl.col(col).is_not_null())

    return result.lazy()
