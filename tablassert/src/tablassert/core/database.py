from tablassert.src.tablassert.models.graph_config import SqliteDatabases
from sqlite_utils import Database
from typing import Any, Optional
from collections import Counter
from functools import lru_cache
from pydantic import FilePath
from diskcache import Cache
from pathlib import Path

# all of these are hardcoded because of how hard everything got with circular imports
fullmapcache: Cache = Cache("/tablassert/cache/.fullmap".upper(), max_size=1e10)
babelcache: Cache = Cache("/tablassert/cache/.babel".upper(), max_size=1e10)
kg2cache: Cache = Cache("/tablassert/cache/.kg2".upper(), max_size=1e9)
metadatacache: Cache = Cache("/tablassert/cache/.pubmed_metadata".upper(), max_size=1e6)
captionscache: Cache = Cache("/tablassert/cache/.pubmed_captions".upper(), max_size=1e6)

def new_connection(sqlitepath: Path) -> Database:
    return Database(sqlitepath.as_posix()).enable_wal()

# pubmed lookups aren't frequent enough to justify a combined cache

@metadatacache.memorize()
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
    """
    rows: Any = pubmed.query(sql, {"curie": article_curie})
    mesh: Optional[list[str]] = [row["mesh"] for row in rows if row]
    mesh_major: Optional[list[str]] = [row["mesh_major"] for row in rows if row]
    mesh_zip: Any = zip(mesh, mesh_major)
    domain: list[str] = [term for term, importance in mesh_zip if importance == "Y"]
    mesh_terms: list[str] = [term for term, importance in mesh_zip if importance == "N"]
    row: Any = next(rows, {})
    firstauthor: Optional[str] = row.get("firstauthor")
    journal: Optional[str] = row.get("journal")
    title: Optional[str] = row.get("title")
    year: Optional[str] = row.get("year")
    return {
        "domain": domain,
        "mesh_terms": mesh_terms,
        "firstauthor": firstauthor,
        "journal": journal,
        "title": title,
        "year": year,
    }

@captionscache.memorize()
def file_caption(article_curie: str, filename: str) -> Optional[str]:
    sql: str = """
    SELECT caption
    FROM captions
    WHERE pmc = :curie AND file = :filename
    LIMIT 1;
    """
    rows: Any = pubmed.query(sql, {"curie": article_curie, "filename": filename})
    row: Any = next(rows, {})
    return row.get("caption")

# lru cache is 10-100x faster so I cache twice
@lru_cache(maxsize=1024)
def cached_babel_lookup(unprocessed_input: str, prioritize: Optional[set[str]], avoid: Optional[set[str]], taxon: Optional[str]) -> Optional[]:
    return babel_lookup(unprocessed_input, prioritize, avoid, taxon)

def collect_babelresults(rows: Any) -> Optional[dict[str, Any]]

@babelcache.memorize()
def babel_lookup(unprocessed_input: str, prioritize: Optional[set[str]], avoid: Optional[set[str]], taxon: Optional[str]) -> Optional[]:
    prioritize_placeholders: Optional[list[str]] = 
    avoid_placeholders: Optional[list[str]] =
    most_common: list[Any] = column_context.most_common(1)
    if most_common:
        most_common: str = str(most_common[0][0])
    level: str = "L1"
    sql: str = f"""
    SELECT
        NAMES.CURIE,
        NAMES.CATEGORY,
        NAMES.NAME,
        NAMES.TAXON, 
    FROM SYNONYMS
    INNER JOIN NAMES ON SYNONYMS.CURIE = NAMES.CURIE
    WHERE 
        {"SYNONYMS.L1 = :input" if level == "L1" else "SYNONYMS.L2 = :input" if level == "L2" else "SYNONYMS.L3 = :input"}
        {"AND NAMES.TAXON = :taxon" if taxon else ""}
        {f"AND NAMES.CATEGORY NOT IN ({avoid_placeholders})" if avoid_placeholders else ""}
    {f"ORDER BY \n\t CASE \n\t\t WHEN NAMES.CATEGORY IN ({prioritize_placeholders}) AND NAMES.CATEGORY = {most_common} THEN 0 \n\t\t WHEN NAMES.CATEGORY IN ({prioritize_placeholders}) THEN 1 \n\t\t WHEN NAMES.CATEGORY = {most_common} THEN 2 \n\t\t ELSE 3 \n\t END" if prioritize_placeholders and most_common else f"ORDER BY \n\t CASE \n\t\t WHEN NAMES.CATEGORY IN ({prioritize_placeholders}) THEN 0 \\n\t\t ELSE 1 \n\t END" if prioritize_placeholders else f"ORDER BY \n\t CASE \n\t\t WHEN NAMES.CATEGORY = {most_common} THEN 0 \n\t\t ELSE 1 \n\t END" if most_common else "" else ""}
    """
    rows: Any = babel.query(sql)
    return 

@lru_cache(maxsize=512)
def cached_kg2_lookup(unprocessed_input: str, prioritize: Optional[set[str]], avoid: Optional[set[str]]) -> Optional[]:
    return kg2_lookup(unprocessed_input, prioritize, avoid, taxon)

@kg2cache.memorize()
def kg2_lookup(unprocessed_input: str, prioritize: Optional[set[str]], avoid: Optional[set[str]]) -> Optional[]:
    sql: str = """
    FROM
    """
    return kg2.query(sql)

# patch lookups aren't frequent enough to justify a combined cache

@lru_cache(maxsize=16)
def override_lookup(unprocessed_input: str, prioritize: Optional[set[str]], avoid: Optional[set[str]]) -> Optional[]:
    sql: str = """
    FROM
    """
    return mapping_patch.query(sql)

@lru_cache(maxsize=32)
def supplement_lookup(unprocessed_input: str, prioritize: Optional[set[str]], avoid: Optional[set[str]]) -> Optional[]:
    sql: str = """
    FROM
    """
    return mapping_patch.query(sql)

# counter is global because it's not hashable for the caches
column_context: Counter = Counter()

def reset_column_context() -> None:
    global column_context
    column_context = Counter()

# databases aren't hashable so I activate them all globally
def activate_sqlites(Sqlites: SqliteDatabases) -> None:
    pubmedpath: FilePath = Sqlites.pubmed
    global pubmed
    pubmed: Database = new_connection(Path(str(pubmedpath)).resolve())
    pmcpath: FilePath = Sqlites.pmc
    global pmc
    pmc: Database = new_connection(Path(str(pmcpath)).resolve())
    babelpath: FilePath = Sqlites.babel
    global babel
    babel: Database = new_connection(Path(str(babelpath)).resolve())
    kg2path: FilePath = Sqlites.kg2
    global kg2
    kg2: Database = new_connection(Path(str(kg2path)).resolve())
    mapping_patchpath: FilePath = Sqlites.mapping_patch
    global mapping_patch
    mapping_patch: Database = new_connection(Path(str(mapping_patchpath)).resolve())

@lru_cache(maxsize=2048)
def cached_fullmap3(unprocessed_input: str, prioritize: Optional[set[str]], avoid: Optional[set[str]], taxon: Optional[str]) -> Optional[]:
    return fullmap3(unprocessed_input, prioritize, avoid, taxon, column_context)

@fullmapcache.memorize()
def fullmap3(unprocessed_input: str, prioritize: Optional[set[str]], avoid: Optional[set[str]], taxon: Optional[str]) -> Optional[]:
    return