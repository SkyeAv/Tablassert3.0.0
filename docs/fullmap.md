# Fullmap

Build a fullmap once and every `resolve()` / `resolve_many()` call maps free text to the right
biological CURIE — the completeness and freshness of this database directly sets how many of your
entities resolve correctly and how trustworthy the resulting graph is.

Fullmap is Tablassert's embedded entity-resolution database: a small set of [redb](https://github.com/cberner/redb)
files — a primary file holding the dimensions, CURIEs, and schema metadata, plus hash-sharded RECORDS
files (`fullmap.s0.redb` … `fullmap.s15.redb` by default) holding the term→postings index — containing
biological synonyms, CURIEs, Biolink categories, taxon IDs, and source provenance, built from NCATS
Translator BABEL export files.

Fullmap is built entirely in-process by Tablassert's own Rust extension — no external tool or install step required by default (this is an in-process redb shard scheme, not the older external DuckDB shards). If you opt into `build-fullmap --aria2c` / `-a`, only the download stage uses an installed external `aria2c` executable.

## Build Command

```bash
# Build a fullmap database (downloads BABEL data automatically)
tablassert build-fullmap

# Optional: use installed aria2c for resumable segmented BABEL downloads
tablassert build-fullmap --aria2c
```

See the [CLI Reference → build-fullmap](cli.md#build-fullmap) for the complete flag table (output path,
cache directory, BABEL snapshot version, worker threads, and the optional `--aria2c` / `-a` downloader),
their defaults, and more examples. Two facts matter most when planning a build:

- The BABEL **version** flag selects a RENCI BABEL snapshot date (default `2026jul22`) — *not*
  Tablassert's package version. Bumping it fetches a different snapshot and requires rebuilding; the
  value used is recorded in the primary's `meta` table (`source_version`).
- With **threads** left unset, the Rust build caps workers at `min(available_CPUs, MemAvailable_GB / 2)`
  on Linux (reading `MemAvailable:` from `/proc/meminfo`, each worker budgeting ~2 GB of local buffers)
  and falls back to ~90% of CPUs elsewhere, so a large build stays within a fixed memory budget.

### Data Pipeline

The build is a parallel, **memory-bounded** pipeline executed by the Rust extension:

1. **Download** — fetch BABEL class and synonym files from RENCI into the cache (resumable, reused). By default this uses Tablassert's Python downloader; `--aria2c` / `-a` opts into the installed `aria2c` executable, preserving aria2 resume control files across dropped downloads and failing loud if the executable is missing or the download fails.
2. **Equivalents index** — parse class files into sorted on-disk runs, then k-way merge them into a
   memory-mapped index mapping each primary CURIE to its equivalents.
3. **Synonym pass** — a producer/consumer pool streams byte-bounded line-chunks; workers dedup CURIEs,
   accumulate normalized-term → (CURIE, source) postings, and spill per-shard runs to disk so peak RAM
   stays flat regardless of input size.
4. **Write** — one redb transaction writes the primary tables, then `shard_count` independent k-way
   merges write the shard `records` files in parallel (one thread per shard).

??? note "Pipeline details"
    Heavy intermediate state spills to a temporary directory (`<output>.spill.d`, removed on success)
    rather than RAM, so a full BABEL build (hundreds of millions of CURIEs) completes within a fixed
    memory budget; the [mimalloc](https://github.com/microsoft/mimalloc) allocator keeps heavy
    multi-threaded allocation from bloating resident memory.

    - **Download** — files come from `https://stars.renci.org/var/babel_outputs` via resumable,
      range-request downloads; cached files are reused. Passing `--aria2c` / `-a` switches only this
      stage to the installed `aria2c` executable, using aria2's segmented HTTP downloads and retry/resume
      control files while suppressing aria2's own progress UI so Tablassert's progress bar stays clean.
      The progress detail remains file-level (`aria2c downloading`) rather than byte-level in this mode.
    - **Equivalents index** — class files parse in parallel into sorted on-disk runs, k-way merged into a
      single memory-mapped CURIE→equivalents index; only a compact `(hash, offset)` index lives in RAM,
      the string data is mmap'd.
    - **Synonym pass** — uses **intra-file parallelism**: a small pool of producer threads
      decompresses/reads the synonym files and pushes byte-bounded line-chunks through a bounded channel,
      and every worker draws from one shared queue, so the few very large files
      (protein/smallmolecule/gene/drugchemicalconflated) are processed by **all** workers, not one thread
      each. Per row, the build collects dimension sets (CURIE prefixes, Biolink categories, sources),
      assigns compact integer CURIE IDs via a hash-keyed dedup map (`xxh3_128(curie) → id`), and
      accumulates normalized-term → (CURIE, source) postings. Each worker drains its per-CURIE rows and
      term postings to bounded on-disk spill runs once its buffer fills — the term postings partitioned
      per-shard at spill time (each `run_s{shard}_{id}.bin` holds only terms with
      `xxh64(term) & (shards-1)` matching that shard), which keeps peak RAM flat regardless of input size
      and lets the write phase merge each shard independently. Dead terms (purely numeric, or generic
      labels like `none`/`nan`/`null`) are skipped, since they can never be queried.
    - **Write** — a single redb write transaction in the primary file emits the dimension tables
      (`prefixes`, `categories`, `sources`), the `curies` table (streamed from its spill runs), and the
      `meta` schema tag (recording the shard count). The `records` table is then written **in parallel
      across the shard files**: building on the per-shard spill partitioning, the write phase runs
      `shard_count` independent k-way merges — one thread per shard, each merging only its own shard's
      runs and inserting the merged term groups inline into that shard's redb file (one database per
      shard, since redb allows a single writer per file) in hash-sorted batches for near-sequential B-tree
      appends (each batch is appended through redb's end-of-table cursor API, the faster ascending
      bulk-load path; the engine's ascending-key page optimization also lets a key-order-loaded shard
      occupy about half as many pages, so current builds write ~50% smaller shard files at the same
      schema). Because every term's postings already live in its own shard's runs, each merge groups a
      term completely with no cross-shard coordination; as the merge+insert is the bottleneck,
      `shard_count` writers deliver ~N× single-threaded write throughput.

### Build Tunables (environment)

Advanced tuning for the build's memory/speed trade-offs. Defaults are safe for a typical large build;
override only when targeting an unusual machine.

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

## Output Artifact

A primary redb file (default `./fullmap/data/fullmap.redb`) plus its sibling RECORDS shard files
(`fullmap.s0.redb` … `fullmap.s15.redb` — one per shard, named after the output
file stem in the same directory). The shard count is fixed at 16 (the read path
still honors the count recorded in an existing database's `meta` table). Together
they hold six tables (see `rust/src/fullmap.rs`):

| Table | Description |
|-------|-------------|
| `records` | Normalized term (xxhash `u64`) → serialized list of resolution postings (CURIE id, source id). Hash-partitioned across the shard files (`fullmap.s0..s<N-1>.redb`) by `xxh64(term) & (shards-1)`; a term lives in exactly one shard. |
| `prefixes` | Compact `u16` id → CURIE prefix string (primary file) |
| `categories` | Compact `u16` id → Biolink category string (primary file) |
| `sources` | Compact `u8` id → source metadata (name/version) (primary file) |
| `curies` | Compact `u32` id → CURIE record (CURIE, preferred name, category, taxon, source) (primary file) |
| `meta` | Schema version tag (`tablassert.fullmap.v5`), the shard count (`shards`), and the BABEL `source_version` used to build the file (primary file) |

The shard files must remain alongside the primary file — lookups discover them as siblings of the
resolved primary path.

Lookups (`lookup_fullmap_terms`) check the primary's `meta` schema tag before reading `records`, read the
`shards` count to open exactly that many shard files, and fan the query terms out across the shards in
parallel (releasing the GIL, one reader per non-empty shard, re-merged into input order); a mismatched or
missing tag raises rather than silently reading incompatible data. Databases built under the older
`v1`/`v2`/`v3`/`v4` schemas are rejected — there is no automatic schema migration, so a schema bump
(including the v3→v4 move to sharded files and the v4→v5 move to the redb 4 engine) requires rebuilding
via `tablassert build-fullmap`.

Readers open every fullmap file READ-ONLY with a SHARED file lock (redb ≥ 3 `ReadOnlyDatabase`), so any
number of processes can run lookups against the same fullmap concurrently; only a `build-fullmap` rebuild
(an exclusive-lock writer) briefly blocks readers. Each lookup pins one primary-plus-shards file
generation — cached handles are validated against the file's `(dev, ino)` on every use — so a reader
follows a rebuild on the next lookup.

## Usage in Graph Config

The `fullmap:` field in a graph configuration points at either the redb file directly or a base
directory. Tablassert resolves it via `fullmap_db_path()`:

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

Pass the fullmap path (file or base directory) as the `fullmap` argument to `resolve_many()` — see
[Batch Resolution](api/lib.md) for the full reference and example, and
[Entity Resolution](api/fullmap.md) for the lower-level `resolve()` API.
