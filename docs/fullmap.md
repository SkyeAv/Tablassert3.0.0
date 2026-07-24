# Fullmap

Fullmap is Tablassert's embedded entity-resolution database: a single [redb](https://github.com/cberner/redb) file containing biological synonyms, CURIEs, Biolink categories, taxon IDs, and source provenance, built from NCATS Translator BABEL export files. It powers `resolve()` / `resolve_many()`, mapping free-text strings to standardized identifiers.

Unlike the DuckDB-shard architecture used in earlier versions (built by a separate external `datassert` Go CLI), Fullmap is built entirely in-process by Tablassert's own Rust extension — no external tool or install step is required.

## Build Command

```bash
# Build a fullmap database (downloads BABEL data automatically)
tablassert build-fullmap
```

### Flags

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--output`, `-o` | No | `./fullmap/data/fullmap.redb` | Path to write the built redb file |
| `--cache`, `-c` | No | `./fullmap/downloads/fullmap` | Directory for downloaded BABEL files |
| `--version`, `-v` | No | current BABEL release (see `cli.py`) | BABEL release version to fetch |
| `--threads`, `-t` | No | `None` (~90% of available CPUs) | Worker threads for the parallel build |

### Data Pipeline

The build is a parallel, **memory-bounded** pipeline executed by the Rust extension. Heavy intermediate state is spilled to a temporary directory (`<output>.spill.d`, removed on success) instead of being held in RAM, so a full BABEL build (hundreds of millions of CURIEs) completes within a fixed memory budget. The synonym phase uses **intra-file parallelism** (a producer–consumer pool, below) so the few very large BABEL files are processed by every worker rather than one thread each, and the crate uses the [mimalloc](https://github.com/microsoft/mimalloc) allocator so heavy multi-threaded allocation does not bloat resident memory:

1. **Download** — BABEL class and synonym files are downloaded from RENCI (`https://stars.renci.org/var/babel_outputs`) into `--cache` (resumable, range-request downloads; cached files are reused).
2. **Equivalents index** — Class files are parsed in parallel into sorted on-disk runs, then k-way merged into a single memory-mapped index mapping each primary CURIE to its equivalent identifiers. Only a compact `(hash, offset)` index lives in RAM; the string data is mmap'd.
3. **Synonym pass** — A small pool of producer threads decompresses/reads the synonym files and pushes byte-bounded line-chunks through a bounded channel; the worker threads pull chunks and process the rows in parallel. Because every worker draws from one shared queue, the large files (protein/smallmolecule/gene/drugchemicalconflated) are processed by **all** workers, not one thread each. For each row, the build collects dimension sets (CURIE prefixes, Biolink categories, sources), assigns compact integer CURIE IDs via a hash-keyed dedup map (`xxh3_128(curie) → id`), and accumulates normalized-term → (CURIE, source) postings. Each worker's per-CURIE rows and term postings are drained to bounded on-disk spill runs once its buffer fills, so peak RAM stays flat regardless of input size. Terms matching the lookup path's dead-term filter (purely numeric, or generic labels like `none`/`nan`/`null`) are skipped, since they can never be queried.
4. **Write** — A single redb write transaction emits the dimension tables (`prefixes`, `categories`, `sources`), the `curies` table (streamed from its spill runs), and the `meta` schema tag, followed by the `records` table — a k-way merge of the term spill runs streamed into redb in hash-sorted batches for near-sequential B-tree appends.

### Build Tunables (environment)

Advanced tuning for the build's memory/speed trade-offs. Defaults are safe for a typical large build; override only when targeting an unusual machine.

| Variable | Default | Description |
|----------|---------|-------------|
| `TABLASSERT_FULLMAP_EXCLUDE_PREFIXES` | *(empty)* | Comma-separated CURIE prefixes to drop at build time (e.g. `INCHIKEY,Publication`). Excluding prefixes you never resolve dramatically cuts build time, peak memory, and database size. |
| `TABLASSERT_FULLMAP_CHUNK_BYTES` | `8388608` (8 MiB) | Byte budget per producer→worker line-chunk. Bounded by bytes (not line count) so chunk memory is fixed even for large synonym records. |
| `TABLASSERT_FULLMAP_PRODUCERS` | `clamp(workers/4, 4, #files)` | Number of producer (decompressor) threads. Decompression far outpaces parallel processing, so a handful keeps all workers fed. |
| `TABLASSERT_FULLMAP_LOCAL_SPILL_ENTRIES` | `1000000` | Per-worker term-posting buffer size before spilling a sorted run to disk. Lower → less RAM, more run files. |
| `TABLASSERT_FULLMAP_CURIE_SPILL_ENTRIES` | `250000` | Per-worker CURIE-row buffer size before spilling to disk. Lower → less RAM, more run files. |
| `TABLASSERT_FULLMAP_EQUIV_SPILL_ENTRIES` | `2000000` | Per-thread equivalents buffer size before spilling during the equivalents-index build. |
| `TABLASSERT_FULLMAP_INSERT_BATCH` | `2000000` | Records buffered per hash-sorted batch during the redb write. Larger → faster writes, modestly more RAM. |
| `TABLASSERT_FULLMAP_REDB_CACHE_BYTES` | `2147483648` (2 GiB) | redb write-cache size. |
| `TABLASSERT_FULLMAP_SPILL_DIR` | `<output>.spill.d` | Directory for intermediate spill runs (removed on success). |

### Examples

```bash
# Full build (download, process, and generate the database)
tablassert build-fullmap

# Custom output location and BABEL version
tablassert build-fullmap --output /data/fullmap/fullmap.redb --version 2025sep1

# Tune concurrency for large builds
tablassert build-fullmap --threads 8
```

## Output Artifact

A single redb file (default `./fullmap/data/fullmap.redb`) containing six tables (see `rust/src/fullmap.rs`):

| Table | Description |
|-------|-------------|
| `records` | Normalized term (xxhash `u64`) → serialized list of resolution postings (CURIE id, source id) |
| `prefixes` | Compact `u16` id → CURIE prefix string |
| `categories` | Compact `u16` id → Biolink category string |
| `sources` | Compact `u8` id → source metadata (name/version) |
| `curies` | Compact `u32` id → CURIE record (CURIE, preferred name, category, taxon, source) |
| `meta` | Schema version tag (`tablassert.fullmap.v3`) and the BABEL `source_version` used to build the file |

Lookups (`lookup_fullmap_terms`) check the `meta` schema tag before reading `records`; a mismatched or missing tag raises rather than silently reading incompatible data. Databases built under the older `v1`/`v2` schemas are rejected — there is no automatic schema migration, so a schema bump requires rebuilding via `tablassert build-fullmap`.

## Usage in Graph Config

The `fullmap:` field in a graph configuration points at either the redb file directly or a base directory. Tablassert resolves it via `fullmap_db_path()`:

- If the path is a file or already ends in `.redb`, it's used as-is.
- Else if `<path>/fullmap.redb` exists, that's used.
- Else it falls back to `<path>/data/fullmap.redb` (the `build-fullmap` default layout).

```yaml
# graph-config.yaml
name: my-graph
version: "1.0"
description: Example graph backed by a fullmap entity-resolution database.
fullmap: /path/to/fullmap/   # directory containing data/fullmap.redb, or a direct .redb file
tables:
  - ./TABLE/my-table.yaml
```

## Programmatic Usage

When calling `resolve_many()` directly, pass the fullmap path (file or base directory) as the `fullmap` argument:

```python
from pathlib import Path
from tablassert.lib import resolve_many

results = resolve_many(
    col="gene",
    entities=["TP53", "BRCA1"],
    fullmap=Path("/path/to/fullmap"),
    taxon="9606",
)
```

See [Entity Resolution](api/fullmap.md) for the full `resolve()` API.
