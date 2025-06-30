from tablassert.src.tablassert.models.graph_config import SqliteDatabases
from sqlite_utils import Database
from typing import Any, Optional
from collections import Counter
from functools import lru_cache
from pydantic import FilePath
from diskcache import Cache
from loguru import logger
from pathlib import Path
import spacy
import time
import re

# all of these are hardcoded because of how hard everything got with circular imports
fullmapcache: Cache = Cache("/tablassert/cache/.fullmap".upper(), max_size=1e10)
babelcache: Cache = Cache("/tablassert/cache/.babel".upper(), max_size=1e10)
kg2cache: Cache = Cache("/tablassert/cache/.kg2".upper(), max_size=1e9)
metadatacache: Cache = Cache("/tablassert/cache/.pubmed_metadata".upper(), max_size=1e6)
captionscache: Cache = Cache("/tablassert/cache/.pubmed_captions".upper(), max_size=1e6)

# logging
log_path: Path = Path("tablassert/log/build.log".upper()).resolve()
log_path.mkdir(parents=True, exist_ok=True)  # ensure log dir exists
logger.add(log_path.as_posix(), rotation="10 MB", retention="10 days", compression="xz")


def new_connection(sqlitepath: str) -> Database:
    return Database(sqlitepath).enable_wal()  # type: ignore


# pubmed lookups aren't frequent enough to justify a combined cache


@metadatacache.memorize()  # type: ignore
def pubmed_metadata(article_curie: str) -> Optional[dict[str, Any]]:
    sql: str = f"""
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
    LIMIT 1
    """
    global start
    start = time.time()
    rows: Any = pubmed.query(sql, {"curie": article_curie})  # type: ignore
    mesh: list[Optional[str]] = [row["mesh.mesh"] for row in rows if row]
    mesh_major: list[Optional[str]] = [row["mesh.mesh_major"] for row in rows if row]
    mesh_zip: Any = zip(mesh, mesh_major)
    domain: list[str] = [term for term, importance in mesh_zip if importance == "Y"]
    mesh_terms: list[str] = [term for term, importance in mesh_zip if importance == "N"]
    row: Any = next(rows, {})
    firstauthor: Optional[str] = row.get("info.firstauthor")
    journal: Optional[str] = row.get("info.journal")
    title: Optional[str] = row.get("info.title")
    year: Optional[str] = row.get("info.year")
    return {
        "domain": domain,
        "mesh_terms": mesh_terms,
        "firstauthor": firstauthor,
        "journal": journal,
        "title": title,
        "year": year,
    }


@captionscache.memorize()  # type: ignore
def file_caption(article_curie: str, filename: str) -> Any:
    sql: str = """
    SELECT caption
    FROM captions
    WHERE pmc = :curie AND file = :filename
    LIMIT 1
    """
    global start
    start = time.time()
    rows: Any = pubmed.query(sql, {"curie": article_curie, "filename": filename})  # type: ignore
    row: Any = next(rows, {})
    return row.get("caption")


# lru cache is 10-100x faster so I cache twice
@lru_cache(maxsize=1024)
def cached_babel_lookup(
    unprocessed_input: str,
    prioritize: Optional[set[str]],
    avoid: Optional[set[str]],
    taxon: Optional[str],
) -> Optional[dict[str, Any]]:
    return babel_lookup(unprocessed_input, prioritize, avoid, taxon)


def dynamic_build(
    level_input: Optional[str],
    prioritize: Optional[set[str]],
    avoid: Optional[set[str]],
    taxon: Optional[str],
) -> tuple[Optional[str], Optional[str], dict[str, str]]:
    prioritize_placeholders: Optional[str] = (
        ", ".join([f":prioritize{index}" for index in range(len(prioritize))])
        if prioritize
        else None
    )
    avoid_placeholders: Optional[str] = (
        ", ".join([f":avoid{index}" for index in range(len(avoid))]) if avoid else None
    )
    most_common: list[Any] = column_context.most_common(1)
    most_common: Optional[str] = str(most_common[0][0]) if most_common else None
    sql_params: dict[str, str] = {"input": level_input}
    if prioritize:
        sql_params.update(
            {
                f":prioritize{index}": category
                for index, category in enumerate(prioritize)
            }
        )
    if avoid:
        sql_params.update(
            {f":avoid{index}": category for index, category in enumerate(avoid)}
        )
    if taxon:
        sql_params[":taxon"] = taxon
    if most_common:
        sql_params[":most_common"] = most_common
    return (prioritize_placeholders, avoid_placeholder, sql_params)


def collect_babelresults(row: Any) -> Optional[dict[str, Any]]:
    curie: str = row.get("NAMES.CURIE")
    category: str = row.get("NAMES.CATEGORY")
    name: str = row.get("NAMES.NAME")
    taxon: str = row.get("NAMES.TAXON")
    if all([curie, category, name, taxon]):
        return {
            "curie": curie,
            "category": category,
            "name": name,
            "taxon": taxon,
        }


level_one: Any = lambda x: str(x).lower()

DISABLE: list[str] = ["parser", "ner", "textcat"]
MODEL = spacy.load("en_core_web_sm", disable=DISABLE)


def level_two(level_two_input: str) -> str:
    tokens: list[str] = MODEL(level_two_input)  # type: ignore
    cleaned_tokens: list[str] = [
        token.lemma_  # yield lemma
        for token in tokens  # iterate through tokens
        if not token.is_stop  # is not a stopword
        and not token.is_punct  # is not punctuation
    ]
    sorted_cleaned_unique_tokens: list[str] = sorted(
        list(dict.fromkeys(cleaned_tokens))
    )
    level_two_output: str = " ".join(sorted_cleaned_unique_tokens)
    return level_two_output


NONWORD_REGEX: Any = re.compile(r"\W+")


def level_three(level_three_input: str) -> str:
    regex: Any = NONWORD_REGEX
    level_three_output: str = re.sub(regex, "", level_three_input)
    return level_three_output


@babelcache.memorize()  # type: ignore
def babel_lookup(
    unprocessed_input: str,
    prioritize: Optional[set[str]],
    avoid: Optional[set[str]],
    taxon: Optional[str],
) -> Optional[dict[str, Any]]:

    level_one_input: str = level_one(unprocessed_input)

    prioritize_placeholders, avoid_placeholder, sql_params = dynamic_build(
        level_one_input, prioritize, avoid, taxon
    )
    level: str = "L1"
    sql: str = f"""
    SELECT
        NAMES.CURIE,
        NAMES.CATEGORY,
        NAMES.NAME,
        NAMES.TAXON
    FROM SYNONYMS
    INNER JOIN NAMES ON SYNONYMS.CURIE = NAMES.CURIE
    WHERE 
        {"SYNONYMS.L1 = :input" if level == "L1" else "SYNONYMS.L2 = :input" if level == "L2" else "SYNONYMS.L3 = :input"}
        {"AND NAMES.TAXON = :taxon" if taxon else ""}
        {f"AND NAMES.CATEGORY NOT IN ({avoid_placeholders})" if avoid_placeholders else ""}
    {f"ORDER BY \n\t CASE \n\t\t WHEN NAMES.CATEGORY IN ({prioritize_placeholders}) AND NAMES.CATEGORY = :most_common THEN 0 \n\t\t WHEN NAMES.CATEGORY IN ({prioritize_placeholders}) THEN 1 \n\t\t WHEN NAMES.CATEGORY = :most_common THEN 2 \n\t\t ELSE 3 \n\t END" if prioritize_placeholders and most_common else f"ORDER BY \n\t CASE \n\t\t WHEN NAMES.CATEGORY IN ({prioritize_placeholders}) THEN 0 \\n\t\t ELSE 1 \n\t END" if prioritize_placeholders else "ORDER BY \n\t CASE \n\t\t WHEN NAMES.CATEGORY = :most_common THEN 0 \n\t\t ELSE 1 \n\t END" if most_common else ""}
    """
    global start
    start = time.time()
    rows: Any = babel.query(sql, sql_params)
    row: Any = next(rows, {})
    result: Optional[dict[str, Any]] = collect_babelresults(row)
    if result:
        return result.update({"level": level})

    level_two_input: str = level_two(level_one_input)
    sql_params["input"] = level_two_input
    level: str = "L2"

    start = time.time()
    rows: Any = babel.query(sql, sql_params)
    row: Any = next(rows, {})
    result: Optional[dict[str, Any]] = collect_babelresults(row)
    if result:
        return result.update({"level": level})

    level_three_input: str = level_three(level_two_input)
    sql_params["input"] = level_three_input
    level: str = "L3"

    start = time.time()
    rows: Any = babel.query(sql, sql_params)
    row: Any = next(rows, {})
    result: Optional[dict[str, Any]] = collect_babelresults(row)
    if result:
        return result.update({"level": level})

    sql_params["input"] = unprocessed_input
    logger.warning()


@lru_cache(maxsize=512)
def cached_kg2_lookup(
    unprocessed_input: str, prioritize: Optional[set[str]], avoid: Optional[set[str]]
) -> Optional[dict[str, Any]]:
    return kg2_lookup(unprocessed_input, prioritize, avoid, taxon)


@kg2cache.memorize()  # type: ignore
def kg2_lookup(
    unprocessed_input: str, prioritize: Optional[set[str]], avoid: Optional[set[str]]
) -> Optional[dict[str, Any]]:

    level_one_input: str = unprocessed_input

    prioritize_placeholders, avoid_placeholder, sql_params = dynamic_build(
        level_one_input, prioritize, avoid, taxon
    )
    level: str = "L1"
    sql: str = f"""
    SELECT
        clusters.cluster_id
        clusters.category
        clusters.name
    FROM id
    INNER JOIN clusters ON nodes.cluster_id = clusters.cluster_id
    WHERE
        {"nodes.name = :input" if level == "L1" else "nodes.name_simplified = :input"}
    {f"ORDER BY \n\t CASE \n\t\t WHEN clusters.category IN ({prioritize_placeholders}) AND clusters.category = :most_common THEN 0 \n\t\t WHEN clusters.category IN ({prioritize_placeholders}) THEN 1 \n\t\t WHEN clusters.category = :most_common THEN 2 \n\t\t ELSE 3 \n\t END" if prioritize_placeholders and most_common else f"ORDER BY \n\t CASE \n\t\t WHEN clusters.category IN ({prioritize_placeholders}) THEN 0 \\n\t\t ELSE 1 \n\t END" if prioritize_placeholders else "ORDER BY \n\t CASE \n\t\t WHEN clusters.category = :most_common THEN 0 \n\t\t ELSE 1 \n\t END" if most_common else ""}
    """

    global start
    start = time.time()
    rows: Any = kg2.query(sql, sql_params)
    row: Any = next(rows, {})

    level_three_input: str = level_three(level_two_input)
    sql_params["input"] = level_three_input
    level: str = "L3"

    level: str = "L3"
    start = time.time()
    rows: Any = babel.query(sql, sql_params)
    row: Any = next(rows, {})
    return


# patch lookups aren't frequent enough to justify a combined cache

"""
Finish adding this after the rest of the pipeline works

@lru_cache(maxsize=16)
def override_lookup(unprocessed_input: str, prioritize: Optional[set[str]], avoid: Optional[set[str]]) -> Optional[]:
    sql: str = "FROM"
    return mapping_patch.query(sql)

@lru_cache(maxsize=32)
def supplement_lookup(unprocessed_input: str, prioritize: Optional[set[str]], avoid: Optional[set[str]]) -> Optional[]:
    sql: str = "FROM"
    return mapping_patch.query(sql)
"""

# counter is global because it's not hashable for the caches
column_context: Counter = Counter()


def reset_column_context() -> None:
    global column_context
    column_context = Counter()


start: float = time.time()


def progress_handler(max_time: float = 1.10) -> int:
    if (start - time.time()) >= max_time:
        return 1
    return 0


def activate_single_sqlite(name: str, path: FilePath) -> None:
    database: Database = new_connection(str(path))
    conn = database.conn
    conn.set_progress_handler(lambda: progress_handler(), 1)
    globals()[name] = database


# databases aren't hashable so I activate them all globally
def activate_sqlites(Sqlites: SqliteDatabases) -> None:
    pubmedpath: FilePath = Sqlites.pubmed
    activate_single_sqlite("pubmed", pubmedpath)
    pmcpath: FilePath = Sqlites.pmc
    activate_single_sqlite("pmc", pmcpath)
    babelpath: FilePath = Sqlites.babel
    activate_single_sqlite("babel", babelpath)
    kg2path: FilePath = Sqlites.kg2
    activate_single_sqlite("kg2", kg2path)
    mapping_patchpath: FilePath = Sqlites.mapping_patch
    activate_single_sqlite("mapping_path", mapping_patchpath)


# @lru_cache(maxsize=2048)
# def cached_fullmap3(unprocessed_input: str, prioritize: Optional[set[str]], avoid: Optional[set[str]], taxon: Optional[str]) -> Optional[]:
# return fullmap3(unprocessed_input, prioritize, avoid, taxon, column_context)

# @fullmapcache.memorize()
# def fullmap3(unprocessed_input: str, prioritize: Optional[set[str]], avoid: Optional[set[str]], taxon: Optional[str]) -> Optional[]:
# return
