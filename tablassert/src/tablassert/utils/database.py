from tablassert.src.tablassert.core.dataframing import get_diskcache
from sqlite_utils import Database
from typing import Any, Optional
from diskcache import Cache
from pathlib import Path

# all of these are hardcoded because of how hard everything got with circular imports
babelcache: Cache = Cache("/tablassert/cache/.babel".upper(), max_size=1e10)
kg2cache: Cache = Cache("/tablassert/cache/.babel".upper(), max_size=1e9)
metadatacache: Cache = Cache("/tablassert/cache/.pubmed".upper(), max_size=1e6)
captionscache: Cache = Cache("/tablassert/cache/.pubmed".upper(), max_size=1e6)

def new_connection(sqlitepath: Path) -> Database:
    return Database(sqlitepath.as_posix()).enable_wal()

@metadatacache.memorize()
def pubmed_metadata(db: Database, article_curie: str) -> Optional[dict[str, object]]:
    sql: str = """
    """
    return db.query(sql)

@mcaptionscache.memorize()
def file_caption(db: Database, article_curie: str, filename: str) -> Optional[str]:
    sql: str = """
    """
    return db.query(sql)