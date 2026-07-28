from __future__ import annotations

from collections import OrderedDict
from enum import Enum
from operator import add
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, cast

from tablassert import rs
from tablassert._lazy import LazyModule
from tablassert.biolink import Categories
from tablassert.log import cat

logger = cat("FULLMAP")

_TERM_CACHE: OrderedDict[tuple[Path, float, str], Optional[list[tuple[int, int]]]] = OrderedDict()
_TERM_CACHE_MAX: int = 100_000
_SOURCE_CACHE: dict[tuple[Path, float], tuple[list[str], list[str], list[str], str]] = {}

if TYPE_CHECKING:
    import polars as pl
else:
    pl = LazyModule("polars")


def empty_matches(column_context: bool) -> pl.DataFrame:
    """Build an empty fullmap matches DataFrame using the canonical query schema.

    Args:
        column_context: When ``True``, add the ``FREQUENCY`` column used by
            column-context ranking.

    Returns:
        Zero-row DataFrame with the fullmap matches schema.
    """
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


def _db_cache_key(db: Path) -> tuple[Path, float]:
    """Build a cache key for a fullmap DB that invalidates on rebuild.

    Args:
        db: Path to the fullmap redb file.

    Returns:
        Canonical path and mtime seconds.
    """
    resolved: Path = db.resolve()
    try:
        return resolved, resolved.stat().st_mtime
    except FileNotFoundError:
        return resolved, -1.0


def _remember_term(key: tuple[Path, float, str], value: Optional[list[tuple[int, int]]]) -> None:
    """Store one term lookup in the bounded FIFO cache.

    Args:
        key: Cache key including database path, mtime, and term.
        value: Raw ``(curie_id, source_id)`` records, or ``None`` for misses.
    """
    _TERM_CACHE[key] = value
    while len(_TERM_CACHE) > _TERM_CACHE_MAX:
        _TERM_CACHE.popitem(last=False)


def _dimension_maps(db: Path, cache_key: tuple[Path, float]) -> tuple[list[str], list[str], list[str], str]:
    """Load cached prefix/category/source dimensions for a fullmap DB.

    Args:
        db: Path to the fullmap redb file.
        cache_key: Cache key from ``_db_cache_key``.

    Returns:
        Prefix, category, source, and source-version maps.
    """
    cached: Optional[tuple[list[str], list[str], list[str], str]] = _SOURCE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    source_version: str = rs.fullmap_source_version()
    value: tuple[list[str], list[str], list[str], str] = (
        list(rs.hydrate_prefixes(db)),
        list(rs.hydrate_categories(db)),
        list(rs.hydrate_sources(db)),
        source_version,
    )
    _SOURCE_CACHE.clear()
    _SOURCE_CACHE[cache_key] = value
    return value


def lookup_rows(db: Path, terms: list[str], threads: Optional[int] = None) -> list[dict[str, object]]:
    """Lookup terms using the v2 raw-pair path and hydrate rows once per batch.

    Args:
        db: Path to the fullmap redb file.
        terms: Terms to query.
        threads: Optional thread count forwarded to Rust.

    Returns:
        Hydrated rows matching the legacy ``lookup_fullmap_terms`` shape.
    """
    if not terms:
        return []
    cache_key: tuple[Path, float] = _db_cache_key(db)
    pairs_by_term: dict[str, Optional[list[tuple[int, int]]]] = {}
    misses: list[str] = []
    for term in terms:
        term_key: tuple[Path, float, str] = (cache_key[0], cache_key[1], term)
        if term_key in _TERM_CACHE:
            pairs_by_term[term] = _TERM_CACHE[term_key]
        else:
            misses.append(term)

    if misses:
        try:
            pair_rows: list[dict[str, object]] = rs.lookup_fullmap_terms(db, misses, threads=threads, return_format="pairs")
        except TypeError:
            return rs.lookup_fullmap_terms(db, terms, threads=threads)
        if pair_rows and "records" not in pair_rows[0]:
            return pair_rows
        seen: set[str] = set()
        for row in pair_rows:
            term = str(row["term"])
            records_raw: list[tuple[int, int]] = cast(list[tuple[int, int]], row["records"])
            records: list[tuple[int, int]] = [(int(curie_id), int(source_id)) for curie_id, source_id in records_raw]
            pairs_by_term[term] = records
            _remember_term((cache_key[0], cache_key[1], term), records)
            seen.add(term)
        for term in misses:
            if term not in seen:
                pairs_by_term[term] = None
                _remember_term((cache_key[0], cache_key[1], term), None)

    curie_ids: list[int] = sorted({curie_id for pairs in pairs_by_term.values() if pairs for curie_id, _source_id in pairs})
    if not curie_ids:
        return []
    hydrated: list[dict[str, Any]] = rs.hydrate_curies(db, curie_ids)
    curie_map: dict[int, dict[str, Any]] = dict(zip(curie_ids, hydrated))
    prefixes, categories, sources, source_version = _dimension_maps(db, cache_key)
    rows: list[dict[str, object]] = []
    for term in terms:
        pairs: Optional[list[tuple[int, int]]] = pairs_by_term.get(term)
        if not pairs:
            continue
        for curie_id, source_id in pairs:
            curie: dict[str, Any] = curie_map[curie_id]
            prefix: str = prefixes[int(curie["prefix_id"])]
            category: str = categories[int(curie["category_id"])]
            rows.append(
                {
                    "term": term,
                    "CURIE": add(add(prefix, ":"), str(curie["local_id"])),
                    "PREFERRED_NAME": str(curie["preferred_name"]),
                    "CATEGORY_NAME": category,
                    "TAXON_ID": int(curie["taxon_id"]),
                    "SOURCE_NAME": sources[source_id],
                    "SOURCE_VERSION": source_version,
                }
            )
    return rows


def distinct(lf: pl.LazyFrame, l1: str, l2: str, col: str = "term") -> pl.LazyFrame:
    """Extract unique terms from two text-normalization columns as a LazyFrame.

    Each input column is de-duplicated independently and tagged with its NLP
    level (1 for ``l1``, 2 for ``l2``), then concatenated and de-duplicated
    again keeping the first (lowest-level) occurrence. Purely numeric values,
    common null sentinels, and a fixed list of generic labels are dropped.

    Args:
        lf: Source LazyFrame.
        l1: Level-one (lightly normalized) column name.
        l2: Level-two (heavily normalized) column name.
        col: Output column name for the unified term.

    Returns:
        LazyFrame with one ``col`` column plus ``nlp_level``.
    """
    t1: pl.LazyFrame = lf.select(pl.col(l1).alias(col)).unique()
    t1 = t1.with_columns(pl.lit(1).alias("nlp_level"))

    t2: pl.LazyFrame = lf.select(pl.col(l2).alias(col)).unique()
    t2 = t2.with_columns(pl.lit(2).alias("nlp_level"))

    terms: pl.LazyFrame = pl.concat([t1, t2]).unique(subset=[col], keep="first")

    bad: str = r"^\d+$|^(none|nan|na|null|unknown|not applicable|p_value|variable|result|exposure|expression|symbol)$|^$"
    terms = terms.filter(~pl.col(col).str.contains(bad))
    return terms


def deduplicate_result(result: pl.DataFrame, column_context: bool) -> pl.DataFrame:
    """Sort and de-duplicate fullmap matches so each term keeps its best row.

    When ``column_context`` is set, a per-category ``FREQUENCY`` column is
    attached and used as a high-priority tiebreaker (more common categories
    first).

    Args:
        result: Joined matches with a ``CATEGORY_NAME`` column.
        column_context: Whether to compute/use the frequency tiebreaker.

    Returns:
        DataFrame with one row per ``term``.
    """
    sort_by: list[str] = ["term", "PR", "NLP_LEVEL"]
    descending: list[bool] = [False, False, False]

    if column_context:
        frequency: pl.DataFrame = result.group_by("CATEGORY_NAME").agg(pl.len().alias("FREQUENCY"))
        result = result.join(frequency, on="CATEGORY_NAME", how="left")

        sort_by += ["FREQUENCY"]
        descending += [True]

    result = result.sort(sort_by, descending=descending)
    return result.unique(subset=["term"], keep="first")


def _category_values(categories: list[Any]) -> list[str]:
    """Normalize a ``prioritize``/``avoid`` list to plain category-name strings.

    The build pipeline supplies plain strings (Pydantic ``use_enum_values=True``
    unwraps the ``Categories`` members on ``NodeEncoding``), while direct callers
    may pass ``Categories`` members; accept both.

    Args:
        categories: Category names as ``Categories`` members or plain strings.

    Returns:
        Plain category-name strings.
    """
    return [c.value if isinstance(c, Enum) else c for c in categories]


def filter_and_rank(
    raw: pl.DataFrame,
    terms: pl.DataFrame,
    taxon: Optional[str],
    prioritize: Optional[list[Categories]],
    avoid: Optional[list[Categories]],
    column_context: bool,
) -> pl.DataFrame:
    """Join already-fetched redb rows against one column's own terms, then filter, rank, and dedup.

    Called by ``resolve_batch`` so it can reuse one shared redb fetch per column.

    Args:
        raw: Rows looked up from the fullmap redb.
        terms: Distinct terms for the column being resolved (from ``distinct``).
        taxon: Optional taxon filter applied to gene-category matches.
        prioritize: Categories to boost in ranking.
        avoid: Categories to drop entirely.
        column_context: Whether to compute/use category frequency as a tiebreaker.

    Returns:
        Ranked matches DataFrame with one row per term.
    """
    # Resolve_batch reuses one shared redb fetch per column.
    if raw.height == 0:
        return empty_matches(column_context)

    result: pl.DataFrame = raw.join(terms, on="term", how="inner").rename({"nlp_level": "NLP_LEVEL"})
    if avoid:
        avoid_values: list[str] = _category_values(avoid)
        result = result.filter(~pl.col("CATEGORY_NAME").is_in(avoid_values))
    if taxon:
        taxon_id: int = int(taxon)
        result = result.filter((pl.col("TAXON_ID") == taxon_id) | (pl.col("CATEGORY_NAME") != Categories.GENE.value))
    if result.height == 0:
        return empty_matches(column_context)

    if prioritize:
        priority_values: list[str] = _category_values(prioritize)
        priority: pl.Expr = pl.when(pl.col("CATEGORY_NAME").is_in(priority_values)).then(pl.lit(1)).otherwise(pl.lit(50))
    else:
        priority = pl.lit(50)
    pr_base: pl.Expr = (
        pl.when(pl.col("PREFERRED_NAME") == pl.col("term"))
        .then(pl.lit(1))
        .when((pl.col("PREFERRED_NAME").str.to_lowercase() == pl.col("term")) & (pl.col("NLP_LEVEL") == 1))
        .then(pl.lit(5))
        .otherwise(pl.lit(10))
    )
    result = result.with_columns((priority * pr_base).alias("PR"))
    return deduplicate_result(result, column_context)


def fullmap_db_path(fullmap: Path) -> Path:
    """Resolve a ``fullmap`` model field to the embedded redb file path.

    Accepts either the redb file directly or a base directory. When given a
    directory, checks for ``<dir>/fullmap.redb`` first, then falls back to
    ``<dir>/data/fullmap.redb``.

    Args:
        fullmap: File path or base directory from the ``Graph.fullmap`` field.

    Returns:
        Resolved path to ``fullmap.redb``.
    """
    if fullmap.is_file() or fullmap.suffix == ".redb":
        return fullmap
    direct: Path = fullmap / "fullmap.redb"
    if direct.is_file():
        return direct
    return fullmap / "data" / "fullmap.redb"


def log_unmatched(col: str, terms: pl.LazyFrame, matches: pl.DataFrame, section_hash: Optional[str], config_file: Optional[str]) -> None:
    """Log level-one terms that did not resolve to any CURIE.

    Args:
        col: Column being resolved (for log context).
        terms: Distinct terms with their NLP level.
        matches: Matches actually resolved for this column.
        section_hash: Short section hash (for log context).
        config_file: Originating config file (for log context).
    """
    # Log unmatched entities.
    level_one: pl.LazyFrame = terms.filter(pl.col("nlp_level") == 1)
    antimatches: pl.LazyFrame = level_one.join(matches.lazy().select("term"), left_on="term", right_on="term", how="anti")

    # Collection point: requires eager.
    unnmatched: pl.DataFrame = antimatches.select("term").unique().collect()
    if unnmatched.height > 0:
        for term in unnmatched.get_column("term").to_list():
            logger.info("Unresolved term in {config} ({hash}) col {col}: {term!r}", config=config_file, hash=section_hash, col=col, term=term)


def join_matches(lf: pl.LazyFrame, col: str, matches: pl.DataFrame, tag: str = "_two") -> pl.LazyFrame:
    """Join ranked fullmap matches back into ``lf`` for one column.

    Coalesces level-one and level-two hits per row (level one wins when
    present) and emits derived ``<col>_name``, ``<col>_category``,
    ``<col>_taxon``, ``<col>_source``, ``<col>_source_version`` and
    ``<col>_nlp_level`` columns.

    Args:
        lf: Source LazyFrame (will be collected eagerly for the join).
        col: Column being resolved.
        matches: Ranked matches for this column from ``filter_and_rank``.
        tag: Suffix used to derive the level-two column name.

    Returns:
        New LazyFrame with resolved columns; rows whose ``col`` did not match
        are dropped.

    Notes:
        Split out of ``resolve`` so ``resolve_batch`` can apply per-column
        matches from one shared redb fetch.
    """
    # Split out of resolve so resolve_batch can apply per-column matches from one shared redb fetch.
    l1: str = col
    l2: str = add(l1, tag)

    # Collection point: join after redb query, then re-lazy.
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
    result = result.select(pl.exclude(add(col, tag)))
    result = result.with_columns(pl.col(add(col, "_taxon")).replace("NCBITaxon:0", None))
    result = result.filter(pl.col(col).is_not_null())

    return result.lazy()


class ResolveSpec(NamedTuple):
    """One node column's resolution settings for ``resolve_batch``."""

    col: str
    taxon: Optional[str] = None
    prioritize: Optional[list[Categories]] = None
    avoid: Optional[list[Categories]] = None


def resolve_batch(
    lf: pl.LazyFrame,
    specs: list[ResolveSpec],
    db: Path,
    log: bool = True,
    section_hash: Optional[str] = None,
    config_file: Optional[str] = None,
    column_context: bool = True,
    tag: str = "_two",
    threads: Optional[int] = None,
) -> pl.LazyFrame:
    """Resolve multiple node columns against one shared redb fetch.

    Each column still gets its own taxon/prioritize/avoid filtering and its own
    join back into ``lf``; only the redb round trip itself
    (``rs.lookup_fullmap_terms``) is pooled across columns.

    Args:
        lf: Source LazyFrame.
        specs: One ``ResolveSpec`` per column to resolve.
        db: Path to the fullmap redb file.
        log: When ``True``, log unmatched level-one terms.
        section_hash: Short section hash (for log context).
        config_file: Originating config file (for log context).
        column_context: Whether to compute/use category frequency as a tiebreaker.
        tag: Suffix used to derive level-two column names.
        threads: Optional thread count forwarded to the Rust lookup.

    Returns:
        LazyFrame with resolved columns added.
    """
    # Each column still gets its own taxon/prioritize/avoid filtering and its own join back into lf;
    # only the redb round trip itself (rs.lookup_fullmap_terms) is pooled across columns.
    if not specs:
        return lf

    terms_by_col: dict[str, pl.LazyFrame] = {spec.col: distinct(lf, spec.col, add(spec.col, tag)) for spec in specs}
    collected_terms: dict[str, pl.DataFrame] = {col: terms.collect() for col, terms in terms_by_col.items()}

    union_terms: list[str] = pl.concat([t.select("term") for t in collected_terms.values()]).unique().get_column("term").to_list()

    rows: list[dict[str, object]] = lookup_rows(db, union_terms, threads=threads) if union_terms else []
    raw: pl.DataFrame = pl.DataFrame(rows)

    result: pl.LazyFrame = lf
    for spec in specs:
        terms_df: pl.DataFrame = collected_terms[spec.col]
        matches: pl.DataFrame = filter_and_rank(raw, terms_df, spec.taxon, spec.prioritize, spec.avoid, column_context)
        if log:
            log_unmatched(spec.col, terms_by_col[spec.col], matches, section_hash, config_file)
        result = join_matches(result, spec.col, matches, tag)

    return result


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
    """Case-dependent, provenance-rich named-entity recognition (single-column wrapper).

    Thin convenience wrapper around ``resolve_batch`` for callers resolving a
    single column.

    Args:
        lf: Source LazyFrame.
        col: Column to resolve.
        db: Path to the fullmap redb file.
        taxon: Optional taxon filter applied to gene-category matches.
        prioritize: Categories to boost in ranking.
        avoid: Categories to drop entirely.
        log: When ``True``, log unmatched level-one terms.
        section_hash: Short section hash (for log context).
        config_file: Originating config file (for log context).
        column_context: Whether to compute/use category frequency as a tiebreaker.
        tag: Suffix used to derive the level-two column name.
        threads: Optional thread count forwarded to the Rust lookup.

    Returns:
        LazyFrame with resolved columns added.
    """
    return resolve_batch(
        lf,
        [ResolveSpec(col, taxon, prioritize, avoid)],
        db,
        log=log,
        section_hash=section_hash,
        config_file=config_file,
        column_context=column_context,
        tag=tag,
        threads=threads,
    )
