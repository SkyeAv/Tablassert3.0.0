from tablassert.src.tablassert.core.dataframing import get_diskcache
from sqlite_utils import Database
from diskcache import Cache
from pathlib import Path
from typing import Any

def new_connection(sqlitepath: Path) -> Database:
    return Database(sqlitepath.as_posix()).enable_wal()

def map_
