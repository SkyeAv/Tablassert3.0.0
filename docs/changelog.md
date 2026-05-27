# Changelog

The canonical release history lives in the repository root at [`CHANGELOG.md`](https://github.com/SkyeAv/Tablassert/blob/main/CHANGELOG.md).

## Current Release Notes

## 7.4.9 - 2026-05-26

### Bug Fixes

- Fixed `OSError: Too many open files` during subgraph build at large scales (700+ sections). `with_mesh()` and `with_captions()` in `lib.py` opened a `sqlite_utils.Database` per section but never closed the underlying SQLite connection, leaving FD release to GC. In the tight sequential `compile_subgraph` loop the leaked FDs accumulated past the OS soft limit, causing the next `to_store()` → `df.write_parquet()` (which polars 1.39 routes through `sink_parquet`) to fail opening its target `.storassert/*.parquet`. Both functions now wrap their query bodies in `try:` / `finally: db.conn.close()`.
