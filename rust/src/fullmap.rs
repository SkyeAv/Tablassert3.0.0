use flate2::read::GzDecoder;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use rayon::prelude::*;
use redb::{Database, Durability, ReadableTable, TableDefinition};
use rustc_hash::FxHasher;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::hash::{BuildHasher, BuildHasherDefault};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::thread;
use xxhash_rust::xxh3::xxh3_128;
use xxhash_rust::xxh64::xxh64;

const RECORDS: TableDefinition<u64, &[u8]> = TableDefinition::new("records");
const PREFIXES: TableDefinition<u16, &str> = TableDefinition::new("prefixes");
const CATEGORIES: TableDefinition<u16, &str> = TableDefinition::new("categories");
const SOURCES: TableDefinition<u8, &[u8]> = TableDefinition::new("sources");
const CURIES: TableDefinition<u32, &[u8]> = TableDefinition::new("curies");
const META: TableDefinition<&str, &str> = TableDefinition::new("meta");
const SCHEMA_VERSION: &str = "tablassert.fullmap.v4";
const SCHEMA_VERSION_V3: &str = "tablassert.fullmap.v3";
const SCHEMA_VERSION_V2: &str = "tablassert.fullmap.v2";
const SCHEMA_VERSION_V1: &str = "tablassert.fullmap.v1";
const FULLMAP_SOURCE_VERSION: &str = "2026jul22";
/// Compile-time maximum number of on-disk redb shard files the RECORDS table is
/// hash-partitioned across, and the default when `TABLASSERT_FULLMAP_SHARDS` is
/// unset.  Must be a power of two so `term_shard` can route with a bitmask.
/// The runtime shard count (`resolve_shard_count`) is clamped to this cap; raise
/// this const to allow more shard files.  Distinct from the in-memory
/// `SHARD_COUNT` used by the concurrent build maps.
const SHARD_COUNT_SHARDS: usize = 16;
/// A normalized term grouped with its deduplicated `(curie_id, source_id)` pairs.
type TermPairs = (String, Vec<(u32, u8)>);
type PairRecords = Vec<TermPairs>;
/// One k-way-merge heap entry: `(xxh64(term), term, run index, pairs)`,
/// min-ordered by `(hash, term)`.  Ordering by the hash first makes the common
/// heap comparison a single `u64` cmp; the term string is compared only on the
/// rare hash collision.  Spill runs are written sorted by the same `(hash, term)`
/// key (see `spill_run`), so the k-way merge still groups equal terms correctly.
type MergeItem = (Reverse<u64>, Reverse<String>, usize, Vec<(u32, u8)>);
/// A read-path fan-out job: the `(input_index, term)` bucket routed to one shard
/// plus a clone of that shard's handle, so a worker thread owns both outright
/// (no shared receiver or borrow).
type ShardJob = (Vec<(usize, String)>, Arc<Database>);
/// Fast non-cryptographic hash map for the build-hot paths (per-worker term
/// aggregation, dimension/CURIE interning).  `FxHasher` (rustc's own hasher) is
/// a single multiply-XOR pass — ~3-5x faster than std's SipHash for these short
/// string/integer keys.  Collision behavior is irrelevant to build correctness:
/// these maps are internal, and the on-disk RECORDS key is xxh64 (not FxHash).
type FxHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;
/// A producer->worker channel payload: the source id tagging a chunk of raw
/// NDJSON line bytes (one `Vec<u8>` per non-empty line).  Workers parse each line
/// straight from the bytes (`serde_json::from_slice`), so no per-line `String` is
/// ever allocated on the read path.
type LineChunk = (u8, Vec<Vec<u8>>);

/// Database cache keyed by canonical path only.
///
/// redb's `Database::open` updates the file mtime, so keying on `(path, mtime)`
/// made every lookup after the first miss the cache and try to re-open the file,
/// which fails because the first handle still holds redb's exclusive `flock`
/// ("Database already open. Cannot acquire lock.").  Keying on the path alone is
/// safe: within a process the DB is only rebuilt via `build_fullmap_db`, which
/// evicts the cache explicitly, and redb's exclusive lock prevents an external
/// rebuild while we hold a handle.
static DB_CACHE: OnceLock<RwLock<HashMap<PathBuf, Arc<Database>>>> = OnceLock::new();

/// Number of shards for concurrent maps (power of two for mask routing).
const SHARD_COUNT: usize = 64;
const SHARD_MASK: usize = SHARD_COUNT - 1;

// Build tunables (overridable via environment for benchmarking / target tuning).
const DEFAULT_LOCAL_SPILL_ENTRIES: usize = 1_000_000;
const DEFAULT_EQUIV_SPILL_ENTRIES: usize = 2_000_000;
const DEFAULT_REDB_CACHE_BYTES: usize = 2 * 1024 * 1024 * 1024;
const DEFAULT_INSERT_BATCH: usize = 2_000_000;
const DEFAULT_CURIE_SPILL_ENTRIES: usize = 250_000;
/// Byte budget per producer->worker chunk. Bounding by bytes (not line count)
/// keeps each chunk's memory fixed even when synonym lines are large (protein /
/// PUBCHEM records can be ~1-2 KB), so the in-flight line buffer stays bounded
/// regardless of record size while still balancing load across workers.
const DEFAULT_CHUNK_BYTES: usize = 8 * 1024 * 1024;

/// Round `n` DOWN to the nearest power of two (6->4, 3->2, 5->4); powers of two
/// are unchanged and any `n < 1` floors to 1.  The shard mask routing in
/// `term_shard` requires a power-of-two shard count, so the runtime tunable is
/// normalized through this before use.
fn round_down_pow2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let p = n.next_power_of_two();
    if p == n {
        n
    } else {
        p / 2
    }
}

/// Resolve the on-disk RECORDS shard count from `TABLASSERT_FULLMAP_SHARDS`
/// (default `SHARD_COUNT_SHARDS`).  Non-powers-of-two round DOWN to the nearest
/// power of two (6->4, 3->2) and the result is clamped to `<= SHARD_COUNT_SHARDS`,
/// the compile-time cap on shard files (raise the const to allow more).
fn resolve_shard_count() -> usize {
    let raw = env_usize("TABLASSERT_FULLMAP_SHARDS", SHARD_COUNT_SHARDS);
    round_down_pow2(raw).clamp(1, SHARD_COUNT_SHARDS)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
struct FullmapRecord {
    curie: String,
    preferred_name: String,
    category_name: String,
    taxon_id: i64,
    source_name: String,
    source_version: String,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
struct CurieRow {
    prefix_id: u16,
    local_id: String,
    preferred_name: String,
    category_id: u16,
    taxon_id: i32,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
struct SourceRow {
    source_name: String,
}

fn py_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

/// Narrow `value` to its cleaned fixed point — repeated trim + matching /
/// duplicate quote stripping — returning a sub-slice of `value` (zero
/// allocation).  Cleaning only ever narrows to a sub-slice, so the result always
/// borrows from the input.
fn clean_slice(value: &str) -> &str {
    let mut s = value;
    loop {
        let trimmed = s.trim();
        let bytes = trimmed.as_bytes();
        let matching_quotes = bytes.len() >= 2
            && ((bytes[0] == b'\'' && bytes[bytes.len() - 1] == b'\'')
                || (bytes[0] == b'"' && bytes[bytes.len() - 1] == b'"'));
        let duplicate_start = bytes.len() >= 2
            && ((bytes[0] == b'\'' && bytes[1] == b'\'') || (bytes[0] == b'"' && bytes[1] == b'"'));

        let next = if duplicate_start {
            &trimmed[1..]
        } else if matching_quotes {
            &trimmed[1..trimmed.len() - 1]
        } else {
            trimmed
        };

        if next == s {
            return next;
        }
        s = next;
    }
}

/// Clean a value (trim + strip matching/duplicate quotes).  Returns a `Cow` that
/// borrows a sub-slice of the input — zero allocation in every case.
fn clean(value: &str) -> Cow<'_, str> {
    Cow::Borrowed(clean_slice(value))
}

/// Clean AND lowercase in one step (the emit_term hot path): clean to a borrowed
/// sub-slice, then ASCII-fast-path lowercase via `level_one`.  The `Cow` is
/// borrowed in the common case (already-clean, already-lowercase ASCII) and only
/// allocates when the term actually needs lowercasing.
fn clean_and_lower(value: &str) -> Cow<'_, str> {
    level_one(clean_slice(value))
}

/// QC a NORMALIZED (level-one / lowercase) term.  Callers must pass an
/// already-lowercased form (emit_term feeds it the level-one value), so the
/// banned-token checks compare lowercase needles directly and skip a per-term
/// `to_lowercase` heap allocation on the hot path.  Behavior is unchanged:
/// lowercasing is case-only and never affects the empty/tab/newline guards, so
/// QC-ing the lowercase form is exactly equivalent to QC-ing the original value
/// and lowercasing internally.
fn token_qc(value: &str) -> bool {
    // Tripwire (debug builds only; zero release cost): the banned-token checks
    // are case-sensitive against lowercase needles, so a mixed-case caller would
    // silently miss them.  Catch that contract violation early.
    debug_assert_eq!(value, value.to_lowercase());
    !value.is_empty()
        && !value.contains('\t')
        && !value.contains('\n')
        && !value.contains('\r')
        && !value.contains("inchikey")
        && !value.contains("uncharacterized")
        && !value.contains("hypothetical")
}

fn level_one(value: &str) -> Cow<'_, str> {
    if value.is_ascii() {
        // ASCII fast path: lowercasing is a byte operation.  Borrow unchanged if
        // already lowercase; otherwise lowercase into a fresh String.
        if value.bytes().all(|b| !b.is_ascii_uppercase()) {
            Cow::Borrowed(value)
        } else {
            let mut lowered = value.to_string();
            lowered.make_ascii_lowercase();
            Cow::Owned(lowered)
        }
    } else {
        Cow::Owned(value.to_lowercase())
    }
}

fn level_two(value: &str) -> Cow<'_, str> {
    // Fast path: level-two keeps [A-Za-z0-9_]; for an already-lowercase ASCII
    // input that is exactly [a-z0-9_], so when every byte qualifies the filter is
    // a no-op and we borrow unchanged (the common case post-level_one).  Any
    // uppercase or non-ASCII byte falls through to the filtering slow path.
    if value
        .bytes()
        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
    {
        Cow::Borrowed(value)
    } else {
        Cow::Owned(
            value
                .chars()
                .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect(),
        )
    }
}

/// Terms the lookup path can never query: the fullmap `distinct()` bad-regex
/// `^\d+$|^(none|nan|na|null|unknown|not applicable|p_value|variable|result|`
/// `exposure|expression|symbol)$|^$` drops these from the query-term set before
/// lookup, so storing them in the DB is dead weight.  `term` is already a
/// normalized (level-one or level-two) form here.  Skipping them is provably
/// safe: a stored term matching this can never meet a surviving query term.
fn is_dead_term(term: &str) -> bool {
    if term.is_empty() {
        return true;
    }
    if term.bytes().all(|b| b.is_ascii_digit()) {
        return true;
    }
    matches!(
        term,
        "none"
            | "nan"
            | "na"
            | "null"
            | "unknown"
            | "not applicable"
            | "p_value"
            | "variable"
            | "result"
            | "exposure"
            | "expression"
            | "symbol"
    )
}

fn open_reader(path: &Path) -> PyResult<Box<dyn Read>> {
    let file = File::open(path).map_err(py_err)?;
    if path.extension().and_then(|x| x.to_str()) == Some("gz") {
        return Ok(Box::new(GzDecoder::new(file)));
    }
    Ok(Box::new(file))
}

/// A synonym NDJSON row, deserialized with borrowed fields so the hot synonym
/// path never builds a `serde_json::Value` DOM (~7 field allocations per row).
/// `Cow` handles JSON strings that need unescaping (`\uXXXX`) while still
/// borrowing the common plain-ASCII case straight from the line buffer.  Field
/// aliases mirror the legacy key fallbacks (curie|id, types|categories,
/// taxa|taxon, preferred_name|name).
#[derive(Deserialize)]
struct SynonymRow<'a> {
    #[serde(rename = "curie", alias = "id", default, borrow)]
    curie: Option<Cow<'a, str>>,
    #[serde(default, borrow)]
    names: Vec<Cow<'a, str>>,
    #[serde(rename = "types", alias = "categories", default, borrow)]
    types: Vec<Cow<'a, str>>,
    #[serde(rename = "taxa", alias = "taxon", default, borrow)]
    taxa: Vec<Cow<'a, str>>,
    #[serde(rename = "preferred_name", alias = "name", default, borrow)]
    preferred_name: Option<Cow<'a, str>>,
}

/// A class NDJSON row (the equivalent-identifiers source), deserialized with a
/// borrowed `id`.  `equivalent_identifiers` elements are heterogeneously either
/// a bare CURIE string or an object carrying one under identifier|id|curie, so
/// they decode via the untagged `EquivId` (owned — untagged decoding buffers its
/// content, so borrowing buys nothing there and the equiv lists are materialized
/// as owned strings anyway).
#[derive(Deserialize)]
struct ClassRow<'a> {
    #[serde(rename = "id", alias = "curie", default, borrow)]
    id: Option<Cow<'a, str>>,
    #[serde(default)]
    equivalent_identifiers: Vec<EquivId>,
}

/// One `equivalent_identifiers` element: a bare CURIE string or an object with
/// the CURIE under `identifier`/`id`/`curie` (untagged to accept both shapes).
#[derive(Deserialize)]
#[serde(untagged)]
enum EquivId {
    String(String),
    Object(EquivIdObject),
}

#[derive(Deserialize)]
struct EquivIdObject {
    #[serde(rename = "identifier", alias = "id", alias = "curie", default)]
    identifier: Option<String>,
}

/// Stream a class NDJSON file (gz or plain) line-by-line into a reusable byte
/// buffer, decoding each non-empty line into a borrowed `ClassRow` and handing it
/// to `visit`.  Zero per-line `String` allocation (the row borrows from the line
/// buffer); replaces the old `Value`-DOM `for_json_lines`.
fn for_class_lines(
    path: &Path,
    mut visit: impl FnMut(&ClassRow<'_>) -> PyResult<()>,
) -> PyResult<()> {
    let mut reader = BufReader::new(open_reader(path)?);
    let mut line: Vec<u8> = Vec::with_capacity(8192);
    loop {
        line.clear();
        let n = reader.read_until(b'\n', &mut line).map_err(py_err)?;
        if n == 0 {
            break; // clean EOF
        }
        while line.last() == Some(&b'\n') || line.last() == Some(&b'\r') {
            line.pop();
        }
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let row: ClassRow = serde_json::from_slice(&line).map_err(py_err)?;
        visit(&row)?;
    }
    Ok(())
}

fn first_category(row: &SynonymRow<'_>) -> String {
    row.types
        .first()
        .map(Cow::as_ref)
        .unwrap_or("NamedThing")
        .trim_start_matches("biolink:")
        .to_string()
}

fn first_taxon(row: &SynonymRow<'_>) -> i32 {
    row.taxa
        .first()
        .map(Cow::as_ref)
        .unwrap_or_default()
        .trim_start_matches("NCBITaxon:")
        .parse::<i32>()
        .unwrap_or(0)
}

fn split_curie(curie: &str) -> Option<(&str, &str)> {
    curie
        .split_once(':')
        .filter(|(prefix, local)| !prefix.is_empty() && !local.is_empty())
}

/// The CURIE string carried by one `equivalent_identifiers` element, if any.
fn equivalent_id_value(equiv: &EquivId) -> Option<String> {
    match equiv {
        EquivId::String(s) => Some(s.clone()),
        EquivId::Object(o) => o.identifier.clone(),
    }
}

fn class_id_and_equivalents(row: &ClassRow<'_>) -> Option<(String, Vec<String>)> {
    let mut ids: Vec<String> = Vec::new();
    let id = match row.id.as_deref() {
        Some(id) => id.to_owned(),
        None => row
            .equivalent_identifiers
            .first()
            .and_then(equivalent_id_value)?,
    };
    // Primary id is NOT included in the value Vec — process_synonyms already
    // adds the curie itself as a term, saving ~38 GB of redundant storage.
    for equivalent in &row.equivalent_identifiers {
        if let Some(eid) = equivalent_id_value(equivalent) {
            if !ids.contains(&eid) {
                ids.push(eid);
            }
        }
    }
    Some((id, ids))
}

fn source_name(path: &Path) -> String {
    path.file_stem()
        .and_then(|x| x.to_str())
        .unwrap_or("BABEL")
        .trim_end_matches(".ndjson")
        .to_string()
}

fn hydrate_record(
    curie: &CurieRow,
    source_name: String,
    source_version: String,
    prefix: &str,
    category_name: &str,
) -> FullmapRecord {
    FullmapRecord {
        curie: format!("{}:{}", prefix, curie.local_id),
        preferred_name: curie.preferred_name.clone(),
        category_name: category_name.to_string(),
        taxon_id: i64::from(curie.taxon_id),
        source_name,
        source_version,
    }
}

// ---------------------------------------------------------------------------
// Sharded concurrent map (inspired by datassert's sharded curieCounter)
// ---------------------------------------------------------------------------

#[allow(clippy::cast_possible_truncation)] // xxh64->usize: only the low bits feed the shard mask, so truncation is the intended routing
fn shard_index(key: &str) -> usize {
    (xxh64(key.as_bytes(), 0) as usize) & SHARD_MASK
}

/// A sharded concurrent HashMap that routes keys by xxhash to reduce lock
/// contention.  Each shard is independently locked.
struct ShardedMap<V> {
    shards: Vec<RwLock<FxHashMap<String, V>>>,
}

impl<V> ShardedMap<V> {
    fn new() -> Self {
        let mut shards = Vec::with_capacity(SHARD_COUNT);
        for _ in 0..SHARD_COUNT {
            shards.push(RwLock::new(FxHashMap::default()));
        }
        Self { shards }
    }

    /// Get an existing value or insert a new one computed by `f`.
    /// Returns a clone of the value (requires V: Clone).
    fn get_or_insert_with(&self, key: &str, f: impl FnOnce() -> V) -> V
    where
        V: Clone,
    {
        let idx = shard_index(key);
        // Fast path: read lock.
        {
            let shard = self.shards[idx].read().unwrap();
            if let Some(v) = shard.get(key) {
                return v.clone();
            }
        }
        // Slow path: write lock.
        let mut shard = self.shards[idx].write().unwrap();
        if let Some(v) = shard.get(key) {
            return v.clone();
        }
        let v = f();
        shard.insert(key.to_string(), v.clone());
        v
    }
}

/// A sharded concurrent map keyed by the 128-bit hash of a CURIE, assigning
/// dense u32 ids.  Keying by `xxh3_128(curie)` instead of the CURIE string cuts
/// the dedup map's memory several-fold at full scale (hundreds of millions of
/// CURIEs: ~16-byte key vs a heap-allocated string + HashMap overhead), while a
/// 128-bit key makes collisions astronomically unlikely (~1e-21 at 5e8 keys),
/// preserving one stable `curie_id` per unique CURIE.  Mirrors datassert's
/// hash-keyed `curieCounter`, widened from 64 to 128 bits for safety.
struct CurieIdMap {
    shards: Vec<RwLock<FxHashMap<u128, u32>>>,
}

impl CurieIdMap {
    fn new() -> Self {
        let mut shards = Vec::with_capacity(SHARD_COUNT);
        for _ in 0..SHARD_COUNT {
            shards.push(RwLock::new(FxHashMap::default()));
        }
        Self { shards }
    }

    /// Get the id for `hash`, or insert a new one computed by `f`.
    #[allow(clippy::cast_possible_truncation)] // u128 hash->usize: only the low bits select the shard, so truncation is the intended routing
    fn get_or_insert_with(&self, hash: u128, f: impl FnOnce() -> u32) -> u32 {
        let idx = (hash as usize) & SHARD_MASK;
        // Fast path: read lock.
        {
            let shard = self.shards[idx].read().unwrap();
            if let Some(v) = shard.get(&hash) {
                return *v;
            }
        }
        // Slow path: write lock.
        let mut shard = self.shards[idx].write().unwrap();
        if let Some(v) = shard.get(&hash) {
            return *v;
        }
        let v = f();
        shard.insert(hash, v);
        v
    }
}

// ---------------------------------------------------------------------------
// Progress callback (Rust -> Python).  The GIL is released for the heavy work,
// so we re-acquire it briefly here to report phase/file/batch progress.
// ---------------------------------------------------------------------------

struct Progress {
    cb: Py<PyAny>,
}

impl Progress {
    fn call(&self, phase: i32, completed: u64, total: u64, detail: &str) {
        Python::attach(|py| {
            // Progress is best-effort: a failing Python callback must never kill
            // the build, so the call result is deliberately discarded.
            let _ = self.cb.call1(py, (phase, completed, total, detail));
        });
    }
}

// ---------------------------------------------------------------------------
// Spill runs: sorted on-disk chunks of (term -> pairs).  This is the Rust
// analog of datassert's "stage to Parquet, then GROUP BY externally" — it
// bounds the term-aggregation memory and lets the final grouping stream.
//
// Frame format (little-endian):
//   [u32 term_len][term bytes][u32 pair_count][(u32 curie_id, u8 source_id) x pair_count]
// Within a run each term appears once (per-thread dedup) and frames are sorted
// by term bytes.  The same term recurs across runs and is grouped at merge.
// ---------------------------------------------------------------------------

struct RunWriter {
    w: BufWriter<File>,
}

impl RunWriter {
    fn new(path: &Path) -> std::io::Result<Self> {
        Ok(Self {
            w: BufWriter::with_capacity(8 * 1024 * 1024, File::create(path)?),
        })
    }

    #[allow(clippy::cast_possible_truncation)] // frame format stores term/pair counts as u32 lengths
    fn write_term(&mut self, term: &str, pairs: &[(u32, u8)]) -> std::io::Result<()> {
        let tb = term.as_bytes();
        debug_assert!(u32::try_from(tb.len()).is_ok() && u32::try_from(pairs.len()).is_ok());
        self.w.write_all(&(tb.len() as u32).to_le_bytes())?;
        self.w.write_all(tb)?;
        self.w.write_all(&(pairs.len() as u32).to_le_bytes())?;
        for (curie_id, source_id) in pairs {
            self.w.write_all(&curie_id.to_le_bytes())?;
            self.w.write_all(&[*source_id])?;
        }
        Ok(())
    }

    fn finish(mut self) -> std::io::Result<()> {
        self.w.flush()
    }
}

struct RunReader {
    reader: BufReader<File>,
    cur: Option<TermPairs>,
}

impl RunReader {
    fn new(path: &Path) -> std::io::Result<Self> {
        let mut reader = BufReader::with_capacity(1 << 20, File::open(path)?);
        let cur = Self::read_frame(&mut reader)?;
        Ok(Self { reader, cur })
    }

    fn read_frame(r: &mut BufReader<File>) -> std::io::Result<Option<TermPairs>> {
        // Read the first length byte; a clean 0-byte read means EOF (no more frames).
        let mut first = [0u8; 1];
        if r.read(&mut first)? == 0 {
            return Ok(None);
        }
        let mut rest = [0u8; 3];
        r.read_exact(&mut rest)?;
        let term_len = u32::from_le_bytes([first[0], rest[0], rest[1], rest[2]]) as usize;
        let mut term_buf = vec![0u8; term_len];
        r.read_exact(&mut term_buf)?;
        let term = String::from_utf8(term_buf)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let mut cnt_buf = [0u8; 4];
        r.read_exact(&mut cnt_buf)?;
        let cnt = u32::from_le_bytes(cnt_buf) as usize;
        let mut pairs = Vec::with_capacity(cnt);
        for _ in 0..cnt {
            let mut pb = [0u8; 5];
            r.read_exact(&mut pb)?;
            let curie_id = u32::from_le_bytes([pb[0], pb[1], pb[2], pb[3]]);
            pairs.push((curie_id, pb[4]));
        }
        Ok(Some((term, pairs)))
    }

    fn advance(&mut self) -> std::io::Result<()> {
        self.cur = Self::read_frame(&mut self.reader)?;
        Ok(())
    }
}

/// Drain a thread-local term buffer into per-shard sorted run files on disk.
/// Terms are partitioned by `term_shard(term, shard_count)` so each shard's runs
/// hold ONLY that shard's terms; Phase 4 then runs one independent k-way merge
/// per shard with no shared producer (datassert-style write-time partitioning).
/// A file `run_s{shard}_{id}.bin` is written only for a shard that has at least
/// one term.  Returns `(shard, path)` pairs so the caller files each run under
/// its shard's list.  The frame format is unchanged (see `RunWriter`); each
/// per-shard file is sorted by `(xxh64(term), term)`, the hash-prefixed ordering
/// `MergeHeap` merges by (hash-first comparisons are a cheap `u64` cmp).
fn spill_run(
    local: &mut FxHashMap<String, Vec<(u32, u8)>>,
    spill_dir: &Path,
    run_id: usize,
    shard_count: usize,
) -> PyResult<Vec<(usize, PathBuf)>> {
    // Partition the buffer's terms by their destination shard.
    let mut by_shard: Vec<Vec<TermPairs>> = vec![Vec::new(); shard_count];
    for (term, pairs) in local.drain() {
        by_shard[term_shard(&term, shard_count)].push((term, pairs));
    }
    let mut out: Vec<(usize, PathBuf)> = Vec::new();
    for (shard, mut entries) in by_shard.into_iter().enumerate() {
        if entries.is_empty() {
            continue; // only non-empty shards write a file
        }
        // Sort by (xxh64(term), term) so the run stream matches MergeHeap's
        // hash-prefixed heap ordering — required for correct grouping, and makes
        // merge comparisons a cheap u64 cmp (term string compared only on hash
        // collision).  The hash is recomputed per comparison (xxh64 is ~3 ns for
        // short terms) rather than stored, keeping the frame format unchanged.
        entries.sort_unstable_by(|a, b| {
            xxh64(a.0.as_bytes(), 0)
                .cmp(&xxh64(b.0.as_bytes(), 0))
                .then_with(|| a.0.cmp(&b.0))
        });
        let path = spill_dir.join(format!("run_s{shard}_{run_id:08}.bin"));
        let mut w = RunWriter::new(&path).map_err(py_err)?;
        for (term, mut pairs) in entries {
            pairs.sort_unstable();
            pairs.dedup();
            w.write_term(&term, &pairs).map_err(py_err)?;
        }
        w.finish().map_err(py_err)?;
        out.push((shard, path));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Curie-row spill runs: bounded on-disk chunks of (curie_id, encoded CurieRow).
//
// The synonym phase assigns one CurieRow per unique CURIE.  Holding all of them
// in RAM is the dominant full-build memory cost (~50 GB+ at 300-500 M CURIEs),
// so per-thread buffers are drained to these run files once they exceed
// `curie_spill` entries, and Phase 4 streams them straight into the CURIES
// table.  This bounds curie-row memory to O(threads * curie_spill).
//
// Frame format (little-endian): [u32 curie_id][u32 row_len][bincode CurieRow]
// ---------------------------------------------------------------------------

struct CurieRunWriter {
    w: BufWriter<File>,
}

impl CurieRunWriter {
    fn new(path: &Path) -> std::io::Result<Self> {
        Ok(Self {
            w: BufWriter::with_capacity(8 * 1024 * 1024, File::create(path)?),
        })
    }

    #[allow(clippy::cast_possible_truncation)] // frame format stores the encoded row length as a u32
    fn write_row(&mut self, curie_id: u32, encoded: &[u8]) -> std::io::Result<()> {
        debug_assert!(u32::try_from(encoded.len()).is_ok());
        self.w.write_all(&curie_id.to_le_bytes())?;
        self.w.write_all(&(encoded.len() as u32).to_le_bytes())?;
        self.w.write_all(encoded)?;
        Ok(())
    }

    fn finish(mut self) -> std::io::Result<()> {
        self.w.flush()
    }
}

struct CurieRunReader {
    reader: BufReader<File>,
}

impl CurieRunReader {
    fn new(path: &Path) -> std::io::Result<Self> {
        Ok(Self {
            reader: BufReader::with_capacity(1 << 20, File::open(path)?),
        })
    }

    /// Read the next (curie_id, encoded row); None at clean EOF.
    fn next_row(&mut self) -> std::io::Result<Option<(u32, Vec<u8>)>> {
        // Read the first byte to detect EOF; a 1-byte read returns 0 only at a
        // true EOF (multi-byte `read` can legally return a short count mid-file,
        // so we mirror RunReader::read_frame and read_exact the remainder).
        let mut first = [0u8; 1];
        if self.reader.read(&mut first)? == 0 {
            return Ok(None);
        }
        let mut rest = [0u8; 3];
        self.reader.read_exact(&mut rest)?;
        let curie_id = u32::from_le_bytes([first[0], rest[0], rest[1], rest[2]]);
        let mut lb = [0u8; 4];
        self.reader.read_exact(&mut lb)?;
        let len = u32::from_le_bytes(lb) as usize;
        let mut buf = vec![0u8; len];
        self.reader.read_exact(&mut buf)?;
        Ok(Some((curie_id, buf)))
    }
}

/// Drain a thread-local curie-row buffer into a sorted-independent run file on
/// disk.  Each unique curie_id is written exactly once (one row per CURIE hash).
fn spill_curie_run(
    local: &mut Vec<(u32, CurieRow)>,
    spill_dir: &Path,
    run_id: usize,
) -> PyResult<PathBuf> {
    let path = spill_dir.join(format!("curie_run_{run_id:08}.bin"));
    let mut w = CurieRunWriter::new(&path).map_err(py_err)?;
    for (curie_id, row) in local.drain(..) {
        let encoded = bincode::serialize(&row).map_err(py_err)?;
        w.write_row(curie_id, &encoded).map_err(py_err)?;
    }
    w.finish().map_err(py_err)?;
    Ok(path)
}

/// K-way merge of sorted run files, grouping equal terms across runs.
struct MergeHeap {
    readers: Vec<RunReader>,
    heap: BinaryHeap<MergeItem>,
}

impl MergeHeap {
    fn new(paths: &[PathBuf]) -> std::io::Result<Self> {
        let mut readers = Vec::with_capacity(paths.len());
        let mut heap = BinaryHeap::new();
        for (idx, path) in paths.iter().enumerate() {
            let mut rr = RunReader::new(path)?;
            if let Some((term, pairs)) = rr.cur.take() {
                let hash = xxh64(term.as_bytes(), 0);
                heap.push((Reverse(hash), Reverse(term), idx, pairs));
            }
            readers.push(rr);
        }
        Ok(Self { readers, heap })
    }

    /// Return the next term with its merged, sorted, de-duplicated pairs.
    fn next_group(&mut self) -> std::io::Result<Option<TermPairs>> {
        let Some((Reverse(hash), Reverse(term), idx, pairs)) = self.heap.pop() else {
            return Ok(None);
        };
        let mut merged = pairs;
        let mut to_advance = vec![idx];
        // Group every run entry with the same (hash, term).  Hash equality is a
        // necessary precondition for term equality, so comparing the hash first
        // short-circuits the (rare) string comparison on distinct terms.
        while let Some((Reverse(h), Reverse(t), _, _)) = self.heap.peek() {
            if *h != hash || *t != term {
                break;
            }
            let (_, _, i, p) = self.heap.pop().unwrap();
            merged.extend(p);
            to_advance.push(i);
        }
        merged.sort_unstable();
        merged.dedup();
        for i in to_advance {
            self.readers[i].advance()?;
            if let Some((t2, p2)) = self.readers[i].cur.take() {
                let h2 = xxh64(t2.as_bytes(), 0);
                self.heap.push((Reverse(h2), Reverse(t2), i, p2));
            }
        }
        Ok(Some((term, merged)))
    }
}

// ---------------------------------------------------------------------------
// Phase 1: disk-backed equivalents index (sorted file + mmap)
//
// Instead of holding a ~200 GB HashMap<String, Vec<String>> in RAM, we write
// equivalents to a sorted binary file and mmap it.  An in-memory sorted index
// of (hash, offset) pairs (~14 GB for 860 M entries) drives O(log n) binary
// search lookups; the OS page-cache manages the mmap'd string data.
// ---------------------------------------------------------------------------

/// An entry from an equivalents run file: (hash, key, equivs).
type EquivEntry = (u64, String, Vec<String>);
/// Write an equiv entry to a buffered writer (run-file format with hash).
#[allow(clippy::cast_possible_truncation)] // equiv run format stores key/equiv counts and byte lengths as u32
fn write_equiv_entry(
    w: &mut impl Write,
    hash: u64,
    key: &str,
    equivs: &[String],
) -> std::io::Result<()> {
    w.write_all(&hash.to_le_bytes())?;
    let kb = key.as_bytes();
    debug_assert!(u32::try_from(kb.len()).is_ok() && u32::try_from(equivs.len()).is_ok());
    w.write_all(&(kb.len() as u32).to_le_bytes())?;
    w.write_all(kb)?;
    w.write_all(&(equivs.len() as u32).to_le_bytes())?;
    for equiv in equivs {
        let eb = equiv.as_bytes();
        w.write_all(&(eb.len() as u32).to_le_bytes())?;
        w.write_all(eb)?;
    }
    Ok(())
}

/// Read one equiv entry from a buffered reader.  Returns None at clean EOF.
fn read_equiv_entry(r: &mut impl BufRead) -> std::io::Result<Option<EquivEntry>> {
    if r.fill_buf()?.is_empty() {
        return Ok(None);
    }
    let mut hb = [0u8; 8];
    r.read_exact(&mut hb)?;
    let hash = u64::from_le_bytes(hb);
    let mut kl = [0u8; 4];
    r.read_exact(&mut kl)?;
    let key_len = u32::from_le_bytes(kl) as usize;
    let mut key_buf = vec![0u8; key_len];
    r.read_exact(&mut key_buf)?;
    let key = String::from_utf8(key_buf)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let mut ec = [0u8; 4];
    r.read_exact(&mut ec)?;
    let equiv_count = u32::from_le_bytes(ec) as usize;
    let mut equivs = Vec::with_capacity(equiv_count);
    for _ in 0..equiv_count {
        let mut el = [0u8; 4];
        r.read_exact(&mut el)?;
        let elen = u32::from_le_bytes(el) as usize;
        let mut ebuf = vec![0u8; elen];
        r.read_exact(&mut ebuf)?;
        equivs.push(
            String::from_utf8(ebuf)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?,
        );
    }
    Ok(Some((hash, key, equivs)))
}

/// Drain a thread-local equivalents buffer into a sorted run file on disk.
/// Called both mid-file (when the buffer exceeds the spill threshold) and at
/// end-of-file, bounding per-thread anonymous memory during Phase 1a so a
/// single huge class file (e.g. gene_nodes, ~200 M rows) cannot blow up RAM.
fn spill_equiv_local(
    local: &mut HashMap<String, Vec<String>>,
    equiv_dir: &Path,
    run_counter: &AtomicUsize,
    run_paths: &RwLock<Vec<PathBuf>>,
) -> PyResult<()> {
    if local.is_empty() {
        return Ok(());
    }
    let mut entries: Vec<EquivEntry> = local
        .drain()
        .map(|(key, mut ev)| {
            ev.sort();
            ev.dedup();
            (xxh64(key.as_bytes(), 0), key, ev)
        })
        .collect();
    // Sort by (hash, key) so each run's stream matches the k-way merge heap's
    // ordering — required for correct grouping if a hash collision lands two
    // distinct CURIEs in the same run.
    entries.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let run_id = run_counter.fetch_add(1, Ordering::Relaxed);
    let rp = equiv_dir.join(format!("run_{run_id:08}.bin"));
    let mut w = BufWriter::with_capacity(8 * 1024 * 1024, File::create(&rp).map_err(py_err)?);
    for (h, k, e) in &entries {
        write_equiv_entry(&mut w, *h, k, e).map_err(py_err)?;
    }
    w.flush().map_err(py_err)?;
    run_paths.write().unwrap().push(rp);
    Ok(())
}

/// Disk-backed equivalents lookup: sorted hash index + mmap'd string data.
struct EquivIndex {
    /// Sorted xxh64 hashes (binary-search target).
    hashes: Vec<u64>,
    /// Byte offsets into `data` for each entry.
    offsets: Vec<u64>,
    /// Mmap'd data file: entries packed as
    /// `[u32 key_len][key_bytes][u32 equiv_count][u32 elen][equiv_bytes]…`
    data: memmap2::Mmap,
    /// Kept alive for the mmap.
    _file: File,
    /// Path to the data file (for cleanup on drop).
    data_path: PathBuf,
}

impl Drop for EquivIndex {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.data_path);
    }
}

/// Zero-copy iterator over equivalent strings from the mmap'd data.
struct EquivIter<'a> {
    data: &'a [u8],
    pos: usize,
    remaining: u32,
}

impl<'a> Iterator for EquivIter<'a> {
    type Item = &'a str;
    fn next(&mut self) -> Option<&'a str> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let d = &self.data[self.pos..];
        let len = u32::from_le_bytes(d.get(..4)?.try_into().ok()?) as usize;
        self.pos += 4 + len;
        std::str::from_utf8(d.get(4..4 + len)?).ok()
    }
}

impl EquivIndex {
    /// Build the disk-backed equivalents index from class files.
    fn build(
        classes: &[PathBuf],
        spill_dir: &Path,
        progress: Option<&Arc<Progress>>,
    ) -> PyResult<Self> {
        let equiv_dir = spill_dir.join("equiv");
        std::fs::create_dir_all(&equiv_dir).map_err(py_err)?;

        // Phase 1a: parallel read class files → sorted spill runs.
        let run_counter = AtomicUsize::new(0);
        let run_paths: RwLock<Vec<PathBuf>> = RwLock::new(Vec::new());
        let total = classes.len() as u64;
        let done = AtomicUsize::new(0);

        if classes.is_empty() {
            // No class files — write an empty data file.
            let data_path = spill_dir.join("equiv_data.bin");
            std::fs::write(&data_path, b"").map_err(py_err)?;
            let file = File::open(&data_path).map_err(py_err)?;
            let data = unsafe { memmap2::Mmap::map(&file).map_err(py_err)? };
            return Ok(Self {
                hashes: Vec::new(),
                offsets: Vec::new(),
                data,
                _file: file,
                data_path,
            });
        }

        let equiv_spill = env_usize(
            "TABLASSERT_FULLMAP_EQUIV_SPILL_ENTRIES",
            DEFAULT_EQUIV_SPILL_ENTRIES,
        );

        let results: Vec<PyResult<()>> = classes
            .par_iter()
            .map(|path| {
                let mut local: HashMap<String, Vec<String>> = HashMap::new();
                for_class_lines(path, |row| {
                    if let Some((id, equivs)) = class_id_and_equivalents(row) {
                        local.entry(id).or_default().extend(equivs);
                        // Bound per-thread memory: spill a sorted run once the
                        // buffer exceeds the threshold instead of holding an
                        // entire class file (gene_nodes ≈ 200 M rows) in RAM.
                        if local.len() >= equiv_spill {
                            spill_equiv_local(&mut local, &equiv_dir, &run_counter, &run_paths)?;
                        }
                    }
                    Ok(())
                })?;
                spill_equiv_local(&mut local, &equiv_dir, &run_counter, &run_paths)?;

                if let Some(p) = progress {
                    let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                    p.call(
                        0,
                        n as u64,
                        total,
                        &path
                            .file_name()
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_default(),
                    );
                }
                Ok(())
            })
            .collect();
        for r in results {
            r?;
        }
        let run_paths = run_paths.into_inner().unwrap();

        // Phase 1b: read runs → parallel sort → linear group → write data file + build index.
        let data_path = spill_dir.join("equiv_data.bin");
        let mut dw =
            BufWriter::with_capacity(8 * 1024 * 1024, File::create(&data_path).map_err(py_err)?);
        let mut hashes: Vec<u64> = Vec::new();
        let mut offsets: Vec<u64> = Vec::new();
        let mut offset: u64 = 0;

        // The full-build target has enough RAM for the equivalent entries, and a
        // parallel sort avoids the former single-threaded heap merge bottleneck.
        let mut entries: Vec<EquivEntry> = Vec::new();
        for path in &run_paths {
            let file = File::open(path).map_err(py_err)?;
            let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);
            while let Some(entry) = read_equiv_entry(&mut reader).map_err(py_err)? {
                entries.push(entry);
            }
        }
        entries.par_sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        // Throttled Phase-0 progress: report every 1M merged groups (not per
        // group) to bound the GIL re-acquire count. total is 0 (indeterminate).
        let mut entries_merged: u64 = 0;
        let mut iter = entries.into_iter().peekable();
        while let Some((hash, key, mut equivs)) = iter.next() {
            while let Some((h, k, _)) = iter.peek() {
                if *h != hash || *k != key {
                    break;
                }
                let (_, _, e) = iter.next().expect("peeked equivalent entry disappeared");
                equivs.extend(e);
            }
            equivs.sort();
            equivs.dedup();

            // Write data entry (no hash — it's in the index).
            let kb = key.as_bytes();
            dw.write_all(&(kb.len() as u32).to_le_bytes())
                .map_err(py_err)?;
            dw.write_all(kb).map_err(py_err)?;
            dw.write_all(&(equivs.len() as u32).to_le_bytes())
                .map_err(py_err)?;
            let mut entry_len = 4 + kb.len() + 4;
            for equiv in &equivs {
                let eb = equiv.as_bytes();
                dw.write_all(&(eb.len() as u32).to_le_bytes())
                    .map_err(py_err)?;
                dw.write_all(eb).map_err(py_err)?;
                entry_len += 4 + eb.len();
            }
            hashes.push(hash);
            offsets.push(offset);
            offset += entry_len as u64;

            entries_merged += 1;
            if entries_merged.is_multiple_of(1_000_000) {
                if let Some(p) = progress {
                    p.call(0, entries_merged, 0, "merging equivalents");
                }
            }
        }
        dw.flush().map_err(py_err)?;
        drop(dw);

        // Clean up run files (keep the merged data file).
        let _ = std::fs::remove_dir_all(&equiv_dir);

        // Mmap the data file.
        let file = File::open(&data_path).map_err(py_err)?;
        let data = unsafe { memmap2::Mmap::map(&file).map_err(py_err)? };

        Ok(Self {
            hashes,
            offsets,
            data,
            _file: file,
            data_path,
        })
    }

    /// Look up equivalents for `curie`.  Returns a zero-copy iterator over
    /// equivalent CURIE strings from the mmap'd data, or None if not found.
    fn lookup(&self, curie: &str) -> Option<EquivIter<'_>> {
        let hash = xxh64(curie.as_bytes(), 0);
        let start = self.hashes.partition_point(|&h| h < hash);
        let mut idx = start;
        while idx < self.hashes.len() && self.hashes[idx] == hash {
            let off = self.offsets[idx] as usize;
            let d = &self.data[off..];
            let kl = u32::from_le_bytes(d.get(..4)?.try_into().ok()?) as usize;
            let key = std::str::from_utf8(d.get(4..4 + kl)?).ok()?;
            if key == curie {
                let count = u32::from_le_bytes(d.get(4 + kl..8 + kl)?.try_into().ok()?);
                return Some(EquivIter {
                    data: &self.data,
                    pos: off + 8 + kl,
                    remaining: count,
                });
            }
            idx += 1;
        }
        None
    }
}

/// Process a single term through clean_and_lower → token_qc → level_two and
/// insert the resulting normalized forms into `local_terms`.  token_qc runs on
/// the level-one (lowercase) form — exactly equivalent to QC-ing the cleaned
/// value, since lowercasing is case-only — but avoids a redundant per-term
/// `to_lowercase`.  The whole chain is `Cow`: clean/level_one/level_two borrow in
/// the common case (clean, lowercase, `[a-z0-9_]` ASCII), so the ONLY allocation
/// is the `into_owned()` when a form is actually inserted as a map key.
fn emit_term<S: BuildHasher>(
    term: &str,
    pair: (u32, u8),
    local_terms: &mut HashMap<String, Vec<(u32, u8)>, S>,
) {
    let l1 = clean_and_lower(term);
    if !token_qc(&l1) || is_dead_term(&l1) {
        return;
    }
    // Derive the (optional) level-two key while `l1` is still borrowable; the
    // block yields an owned String (allocating only when level-two differs and
    // survives QC) so `l1` can be moved into its own entry afterwards.
    let l2_key: Option<String> = {
        let l2 = level_two(&l1);
        if l2 != l1.as_ref() && token_qc(&l2) && !is_dead_term(&l2) {
            Some(l2.into_owned())
        } else {
            None
        }
    };
    local_terms.entry(l1.into_owned()).or_default().push(pair);
    if let Some(l2) = l2_key {
        local_terms.entry(l2).or_default().push(pair);
    }
}

// ---------------------------------------------------------------------------
// Phases 2+3: single-pass synonym processing.
//
// Intra-file parallelism via a producer-consumer pipeline (NOT parallel
// shards — the output is still a single redb file).  A small pool of PRODUCER
// threads decompresses/reads the synonym files and pushes bounded line-chunks
// through a channel; `worker_count` WORKER threads pull chunks and process the
// rows in parallel.  Because every worker draws from one shared queue, the
// large files (protein/smallmolecule/gene/drugchemicalconflated) are processed
// by ALL workers rather than one thread each, while small files still overlap.
//
// Each worker keeps persistent term / curie-row buffers that spill sorted runs
// to disk at `local_spill` / `curie_spill`, exactly as before, so the run count
// stays bounded by total_terms / local_spill (independent of chunking).  All
// shared dimension/CURIE state is concurrent (sharded maps + atomics + RwLock).
// ---------------------------------------------------------------------------

/// Concurrent state shared across all synonym-phase workers (all interior-
/// mutable / atomic, so it is shared by reference).
struct SynonymShared<'a> {
    prefix_map: &'a ShardedMap<u16>,
    prefix_counter: &'a AtomicU32,
    category_map: &'a ShardedMap<u16>,
    category_counter: &'a AtomicU32,
    curie_map: &'a CurieIdMap,
    curie_counter: &'a AtomicU32,
    equivalents: &'a EquivIndex,
    exclude_prefixes: &'a HashSet<String>,
    spill_dir: &'a Path,
    local_spill: usize,
    curie_spill: usize,
    /// On-disk RECORDS shard count; term spill runs are partitioned across this
    /// many per-shard lists at write time (see `spill_run`).
    shard_count: usize,
    run_counter: &'a AtomicUsize,
    curie_run_counter: &'a AtomicUsize,
    /// Per-shard term spill-run files: `run_paths[shard]` holds only the runs
    /// whose terms hash to `shard`, so Phase 4 can merge each shard independently.
    run_paths: &'a RwLock<Vec<Vec<PathBuf>>>,
    curie_run_paths: &'a RwLock<Vec<PathBuf>>,
}

/// A worker's private, non-shared accumulation buffers.
#[derive(Default)]
struct WorkerBuf {
    terms: FxHashMap<String, Vec<(u32, u8)>>,
    curie_rows: Vec<(u32, CurieRow)>,
}

/// Process one parsed synonym row into a worker's buffers, spilling to disk
/// when a buffer exceeds its budget.
fn process_row(
    sh: &SynonymShared<'_>,
    source_id: u8,
    row: &SynonymRow<'_>,
    buf: &mut WorkerBuf,
) -> PyResult<()> {
    let Some(curie) = row.curie.as_deref() else {
        return Ok(());
    };
    let Some((prefix, local_id)) = split_curie(curie) else {
        return Ok(());
    };
    // Skip CURIEs whose prefix the caller excludes (opt-in via
    // TABLASSERT_FULLMAP_EXCLUDE_PREFIXES) — filtered out downstream anyway.
    if sh.exclude_prefixes.contains(prefix) {
        return Ok(());
    }

    let prefix_id = sh.prefix_map.get_or_insert_with(prefix, || {
        u16::try_from(sh.prefix_counter.fetch_add(1, Ordering::Relaxed))
            .expect("too many fullmap prefixes")
    });
    let category_name = first_category(row);
    let category_id = sh.category_map.get_or_insert_with(&category_name, || {
        u16::try_from(sh.category_counter.fetch_add(1, Ordering::Relaxed))
            .expect("too many fullmap categories")
    });
    let preferred_name = row.preferred_name.as_deref().unwrap_or(curie);
    let taxon_id = first_taxon(row);

    let mut is_new = false;
    let curie_hash = xxh3_128(curie.as_bytes());
    let curie_id = sh.curie_map.get_or_insert_with(curie_hash, || {
        is_new = true;
        sh.curie_counter.fetch_add(1, Ordering::Relaxed)
    });
    if is_new {
        buf.curie_rows.push((
            curie_id,
            CurieRow {
                prefix_id,
                local_id: local_id.to_string(),
                preferred_name: clean(preferred_name).into_owned(),
                category_id,
                taxon_id,
            },
        ));
    }

    let pair = (curie_id, source_id);
    // Borrow each name as a &str straight from the decoded row; emit_term only
    // needs a &str (the Cow derefs).
    for name in &row.names {
        emit_term(name, pair, &mut buf.terms);
    }
    emit_term(curie, pair, &mut buf.terms);
    if let Some(iter) = sh.equivalents.lookup(curie) {
        for equiv in iter {
            emit_term(equiv, pair, &mut buf.terms);
        }
    }

    if buf.terms.len() >= sh.local_spill {
        let run_id = sh.run_counter.fetch_add(1, Ordering::Relaxed);
        let runs = spill_run(&mut buf.terms, sh.spill_dir, run_id, sh.shard_count)?;
        let mut paths = sh.run_paths.write().unwrap();
        for (shard, p) in runs {
            paths[shard].push(p);
        }
    }
    if buf.curie_rows.len() >= sh.curie_spill {
        let run_id = sh.curie_run_counter.fetch_add(1, Ordering::Relaxed);
        let p = spill_curie_run(&mut buf.curie_rows, sh.spill_dir, run_id)?;
        sh.curie_run_paths.write().unwrap().push(p);
    }
    Ok(())
}

/// Drain a worker's remaining buffers to final spill runs.
fn flush_buf(sh: &SynonymShared<'_>, buf: &mut WorkerBuf) -> PyResult<()> {
    if !buf.terms.is_empty() {
        let run_id = sh.run_counter.fetch_add(1, Ordering::Relaxed);
        let runs = spill_run(&mut buf.terms, sh.spill_dir, run_id, sh.shard_count)?;
        let mut paths = sh.run_paths.write().unwrap();
        for (shard, p) in runs {
            paths[shard].push(p);
        }
    }
    if !buf.curie_rows.is_empty() {
        let run_id = sh.curie_run_counter.fetch_add(1, Ordering::Relaxed);
        let p = spill_curie_run(&mut buf.curie_rows, sh.spill_dir, run_id)?;
        sh.curie_run_paths.write().unwrap().push(p);
    }
    Ok(())
}

/// Producer: read one synonym file (gz or plain), group non-empty lines into
/// byte-bounded chunks (~`chunk_bytes` bytes each), and send each chunk (tagged
/// with its source id) into the channel as raw line bytes.  Decompression happens
/// here; JSON parsing/processing happens in the workers.  The bounded channel
/// provides backpressure so a fast decompressor cannot buffer a whole giant file
/// in RAM.
#[allow(clippy::too_many_arguments)]
fn produce_file(
    path: &Path,
    tx: &SyncSender<LineChunk>,
    source_ids: &FxHashMap<String, u8>,
    chunk_bytes: usize,
    progress: Option<&Arc<Progress>>,
    files_done: &AtomicUsize,
    rows_done: &AtomicU64,
    total_files: usize,
) -> PyResult<()> {
    let src_name = source_name(path);
    let source_id = *source_ids
        .get(&src_name)
        .ok_or_else(|| PyRuntimeError::new_err(format!("uninterned source {src_name}")))?;

    // Read into a reusable byte buffer via `read_until` instead of
    // `BufReader::lines()`: the latter allocates a fresh `String` per line AND
    // runs a UTF-8 validation pass, ~1.4B times over a full build.  Here the
    // buffer is allocated once and reused across lines; each non-empty line is
    // pushed into the chunk as an owned `Vec<u8>` that workers parse straight
    // from the bytes (`serde_json::from_slice`), so no per-line `String` exists.
    let mut reader = BufReader::new(open_reader(path)?);
    let mut line: Vec<u8> = Vec::with_capacity(8192);
    let mut chunk: Vec<Vec<u8>> = Vec::new();
    let mut chunk_len: usize = 0;
    // Rows accumulated in the current (not-yet-flushed) chunk; folded into the
    // shared cumulative counter at each flush and at EOF.
    let mut chunk_rows: u64 = 0;
    loop {
        line.clear();
        let n = reader.read_until(b'\n', &mut line).map_err(py_err)?;
        if n == 0 {
            break; // clean EOF
        }
        while line.last() == Some(&b'\n') || line.last() == Some(&b'\r') {
            line.pop();
        }
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        chunk_rows += 1;
        chunk_len += line.len();
        chunk.push(line.clone());
        // Flush once the chunk reaches the byte budget (bounds per-chunk memory
        // regardless of how long individual lines are).
        if chunk_len >= chunk_bytes {
            tx.send((source_id, std::mem::take(&mut chunk)))
                .map_err(py_err)?;
            chunk_len = 0;
            // Per-chunk Phase-1 progress (the big granularity win): report the
            // CUMULATIVE rows processed across the WHOLE synonym phase — a single
            // monotonic unit shared by every producer via `rows_done` — with
            // total=0 (indeterminate; no cheap global row total exists).  A
            // per-file row count here would clash with a file-index count on the
            // same phase's single high-water mark and overflow the bar.
            if let Some(p) = progress {
                let global = rows_done.fetch_add(chunk_rows, Ordering::Relaxed) + chunk_rows;
                p.call(1, global, 0, &format!("{src_name} · {global} rows"));
            }
            chunk_rows = 0;
        }
    }
    if !chunk.is_empty() {
        tx.send((source_id, chunk)).map_err(py_err)?;
    }

    // EOF: fold the final partial chunk's rows into the cumulative count and emit
    // a file-completion tick in the SAME unit (cumulative rows, total=0) so the
    // phase-1 high-water mark stays monotonic and never mixes in a small file
    // index.  The file count rides in the detail string only (not completed/total).
    // This tick is essential for small files that never hit a chunk flush.
    if let Some(p) = progress {
        let global = rows_done.fetch_add(chunk_rows, Ordering::Relaxed) + chunk_rows;
        let n = files_done.fetch_add(1, Ordering::Relaxed) + 1;
        p.call(
            1,
            global,
            0,
            &format!("{src_name} · {global} rows ({n}/{total_files} files)"),
        );
    }
    Ok(())
}

/// Worker: pull line-chunks from the shared receiver and process their rows
/// into a persistent private buffer, spilling as needed; flush on channel
/// close.  The receiver lock is held only for the `recv` call, never during
/// processing, so workers run in parallel.
fn worker_loop(rx: &Mutex<Receiver<LineChunk>>, sh: &SynonymShared<'_>) -> PyResult<()> {
    let mut buf = WorkerBuf::default();
    loop {
        // Lock is dropped at the end of this statement (before processing).
        let job = rx.lock().unwrap().recv();
        match job {
            Ok((source_id, lines)) => {
                for line in lines {
                    let row: SynonymRow = serde_json::from_slice(&line).map_err(py_err)?;
                    process_row(sh, source_id, &row, &mut buf)?;
                }
            }
            Err(_) => break, // all producers finished and the channel drained
        }
    }
    flush_buf(sh, &mut buf)
}

// ---------------------------------------------------------------------------
// Phases 2+3: single-pass synonym processing (parallel over synonym files)
//
// Collects dimensions (prefixes, categories, sources), assigns CURIE IDs,
// builds CurieRows, and aggregates term -> (curie_id, source_id) pairs into
// BOUNDED per-thread buffers that spill sorted runs to disk when they exceed
// `local_spill` entries.  The runs are merged in Phase 4.
// ---------------------------------------------------------------------------

/// Result of the parallel synonym-processing pass.
struct SynonymBuildResult {
    /// prefix string -> u16 id
    prefix_ids: FxHashMap<String, u16>,
    /// category string -> u16 id
    category_ids: FxHashMap<String, u16>,
    /// source string -> u8 id
    source_ids: FxHashMap<String, u8>,
    /// curie-row spill-run files holding (curie_id, encoded CurieRow)
    curie_run_paths: Vec<PathBuf>,
    /// per-shard sorted spill-run files holding term -> pairs; `run_paths[shard]`
    /// holds only the runs whose terms hash to `shard` (see `spill_run`), so
    /// Phase 4 merges each shard independently.
    run_paths: Vec<Vec<PathBuf>>,
}

#[allow(clippy::too_many_arguments)]
fn process_synonyms(
    synonyms: &[PathBuf],
    equivalents: &EquivIndex,
    spill_dir: &Path,
    local_spill: usize,
    curie_spill: usize,
    exclude_prefixes: &HashSet<String>,
    worker_count: usize,
    shard_count: usize,
    chunk_bytes: usize,
    producers: usize,
    progress: Option<&Arc<Progress>>,
) -> PyResult<SynonymBuildResult> {
    // Pre-compute source IDs from filenames (small, deterministic).
    let mut source_ids: FxHashMap<String, u8> = FxHashMap::default();
    for path in synonyms {
        let name = source_name(path);
        let next_id = u8::try_from(source_ids.len())
            .map_err(|_| PyRuntimeError::new_err("too many fullmap sources"))?;
        source_ids.entry(name).or_insert(next_id);
    }

    // Concurrent dimension counters.
    let prefix_map: ShardedMap<u16> = ShardedMap::new();
    let prefix_counter = AtomicU32::new(0);
    let category_map: ShardedMap<u16> = ShardedMap::new();
    let category_counter = AtomicU32::new(0);

    // Concurrent CURIE ID assignment + CurieRow storage.  The dedup map is keyed
    // by xxh3_128(curie) (not the string) to bound memory at full scale.
    let curie_map = CurieIdMap::new();
    let curie_counter = AtomicU32::new(0);
    // Curie-row spill runs (bounded on-disk; see spill_curie_run).
    let curie_run_paths: RwLock<Vec<PathBuf>> = RwLock::new(Vec::new());
    let curie_run_counter = AtomicUsize::new(0);

    // Spill-run bookkeeping: one run list per shard so Phase 4 can merge each
    // shard's runs independently (write-time partitioning in `spill_run`).
    let run_paths: RwLock<Vec<Vec<PathBuf>>> = RwLock::new(vec![Vec::new(); shard_count]);
    let run_counter = AtomicUsize::new(0);
    let files_done = AtomicUsize::new(0);
    // Cumulative rows processed across the whole synonym phase, shared by every
    // producer so Phase-1 progress reports one monotonic unit (see produce_file).
    let rows_done = AtomicU64::new(0);
    let total_files = synonyms.len();

    let shared = SynonymShared {
        prefix_map: &prefix_map,
        prefix_counter: &prefix_counter,
        category_map: &category_map,
        category_counter: &category_counter,
        curie_map: &curie_map,
        curie_counter: &curie_counter,
        equivalents,
        exclude_prefixes,
        spill_dir,
        local_spill,
        curie_spill,
        shard_count,
        run_counter: &run_counter,
        curie_run_counter: &curie_run_counter,
        run_paths: &run_paths,
        curie_run_paths: &curie_run_paths,
    };

    // Bounded channel of line-chunks: backpressure keeps a fast decompressor
    // from buffering a whole giant file in RAM.  Workers pull chunks in
    // parallel from one shared queue, so the large files are processed by ALL
    // workers rather than one thread each.
    let workers = worker_count.max(1);
    // A modest bound keeps workers fed while bounding the in-flight line-buffer
    // memory (bound x chunk_bytes); decompression outpaces processing, so a
    // shallow queue never starves the workers.
    let (tx, rx) = sync_channel::<LineChunk>(workers);
    let rx = Arc::new(Mutex::new(rx));

    // A handful of producers cover all files (decompression is far faster than
    // parallel processing, so only a few are needed to keep the workers fed).
    let file_idx = AtomicUsize::new(0);
    let producer_count = producers.clamp(1, total_files.max(1));

    let scope_result: PyResult<()> = std::thread::scope(|s| {
        // Copyable reference handles so each `move` closure copies the reference
        // rather than moving the underlying (non-Copy) shared state.
        let shared_ref = &shared;
        let file_idx_ref = &file_idx;
        let source_ids_ref = &source_ids;
        let files_done_ref = &files_done;
        let rows_done_ref = &rows_done;
        let mut handles: Vec<std::thread::ScopedJoinHandle<'_, PyResult<()>>> = Vec::new();

        // Workers: persistent private buffers, spilling at local_spill/curie_spill.
        for _ in 0..workers {
            let rx = Arc::clone(&rx);
            handles.push(s.spawn(move || worker_loop(&rx, shared_ref)));
        }

        // Producers: pull files from a shared counter, push line-chunks.
        for _ in 0..producer_count {
            let tx = tx.clone();
            handles.push(s.spawn(move || loop {
                let i = file_idx_ref.fetch_add(1, Ordering::Relaxed);
                if i >= total_files {
                    return Ok(());
                }
                produce_file(
                    &synonyms[i],
                    &tx,
                    source_ids_ref,
                    chunk_bytes,
                    progress,
                    files_done_ref,
                    rows_done_ref,
                    total_files,
                )?;
            }));
        }

        // Drop the main sender so the channel closes once all producer threads
        // finish; workers then drain remaining chunks and exit.
        drop(tx);

        let mut first_err: Option<PyErr> = None;
        for h in handles {
            match h.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                }
                Err(_) => {
                    if first_err.is_none() {
                        first_err = Some(PyRuntimeError::new_err("fullmap build worker panicked"));
                    }
                }
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    });
    scope_result?;

    // Build final prefix_ids / category_ids maps from the sharded counters.
    let mut prefix_ids: FxHashMap<String, u16> = FxHashMap::default();
    for shard in &prefix_map.shards {
        prefix_ids.extend(shard.read().unwrap().iter().map(|(k, v)| (k.clone(), *v)));
    }
    let mut category_ids: FxHashMap<String, u16> = FxHashMap::default();
    for shard in &category_map.shards {
        category_ids.extend(shard.read().unwrap().iter().map(|(k, v)| (k.clone(), *v)));
    }

    let curie_run_paths = curie_run_paths.into_inner().unwrap();
    let run_paths = run_paths.into_inner().unwrap();

    Ok(SynonymBuildResult {
        prefix_ids,
        category_ids,
        source_ids,
        curie_run_paths,
        run_paths,
    })
}

// ---------------------------------------------------------------------------
// Phase 4: write final database — k-way merge of runs streamed into redb
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn write_final_database(
    output: &Path,
    prefix_ids: &FxHashMap<String, u16>,
    category_ids: &FxHashMap<String, u16>,
    source_ids: &FxHashMap<String, u8>,
    curie_run_paths: &[PathBuf],
    run_paths: &[Vec<PathBuf>],
    cache_bytes: usize,
    insert_batch: usize,
    shard_count: usize,
    progress: Option<&Arc<Progress>>,
) -> PyResult<()> {
    let database = redb::Builder::new()
        .set_cache_size(cache_bytes)
        .create(output)
        .map_err(py_err)?;

    // Dimension tables + CURIES + META in one small transaction.
    let write = database.begin_write().map_err(py_err)?;
    {
        let mut prefix_table = write.open_table(PREFIXES).map_err(py_err)?;
        for (value, id) in prefix_ids {
            prefix_table.insert(*id, value.as_str()).map_err(py_err)?;
        }
        drop(prefix_table);

        let mut category_table = write.open_table(CATEGORIES).map_err(py_err)?;
        for (value, id) in category_ids {
            category_table.insert(*id, value.as_str()).map_err(py_err)?;
        }
        drop(category_table);

        let mut source_table = write.open_table(SOURCES).map_err(py_err)?;
        for (value, id) in source_ids {
            let encoded = bincode::serialize(&SourceRow {
                source_name: value.clone(),
            })
            .map_err(py_err)?;
            source_table
                .insert(*id, encoded.as_slice())
                .map_err(py_err)?;
        }
        drop(source_table);

        // Stream curie rows from their spill runs straight into the CURIES
        // table (never held in RAM as a whole).  Each unique curie_id appears
        // exactly once across the runs.
        let mut curie_table = write.open_table(CURIES).map_err(py_err)?;
        for path in curie_run_paths {
            let mut reader = CurieRunReader::new(path).map_err(py_err)?;
            while let Some((curie_id, encoded)) = reader.next_row().map_err(py_err)? {
                curie_table
                    .insert(curie_id, encoded.as_slice())
                    .map_err(py_err)?;
            }
        }
        drop(curie_table);

        let mut meta = write.open_table(META).map_err(py_err)?;
        meta.insert("schema", SCHEMA_VERSION).map_err(py_err)?;
        let shard_count_str = shard_count.to_string();
        meta.insert("shards", shard_count_str.as_str())
            .map_err(py_err)?;
    }
    write.commit().map_err(py_err)?;

    // The primary's dims/CURIES/META transaction is now durable and `database`
    // is never touched again (Phase 4 writes only the separate shard DBs below).
    // Drop it here to release its redb cache and file lock during the long
    // parallel RECORDS write — a free memory win while the shards are built.
    drop(database);

    // Phase 4: one INDEPENDENT k-way merge per shard, run in parallel — one
    // thread per shard, merge + insert inline (datassert-style).  Because the
    // term spill runs were partitioned by `term_shard` AT WRITE TIME, every term
    // in shard `i`'s runs hashes to shard `i`, so each shard's merge groups its
    // terms completely with NO cross-shard coordination — there is no shared
    // producer and no per-shard channel (the single global producer that capped
    // the old design is gone).  Each thread merges only its own shard's runs,
    // hash-sorts the merged groups into bounded batches for B-tree locality,
    // inserts into its shard's RECORDS table, commits with Durability::None,
    // then does a final durable commit.  Every shard file is created above (even
    // one with zero runs), so the `shard_count`-file layout stays stable.
    let shard_databases: Vec<Database> = (0..shard_count)
        .map(|i| {
            redb::Builder::new()
                .set_cache_size(cache_bytes / shard_count)
                .create(shard_path(output, i))
                .map_err(py_err)
        })
        .collect::<PyResult<_>>()?;

    // Owned per-thread progress handle (cheap Arc clone); `Progress::call`
    // re-acquires the GIL via Python::attach, which is safe from many threads.
    let progress: Option<Arc<Progress>> = progress.map(Arc::clone);
    // Estimate a Phase-2 progress total ONLY when a progress callback is present
    // (tests / no-bar builds skip this read pass entirely): the number of
    // term-frames across every shard's spill runs.  This is an UPPER BOUND on the
    // records written (a term recurs across runs and is merged), so the bar
    // climbs toward ~100% during the merge and the final emission below snaps
    // completed to the exact written count.
    let total_estimate: u64 = if progress.is_some() {
        run_paths
            .iter()
            .flatten()
            .map(|path| count_run_frames(path).unwrap_or(0))
            .sum()
    } else {
        0
    };
    // Global record counter shared across shard writer threads.  Its VALUE is
    // monotonic (each thread adds its flushed batch); the delivery ORDER of the
    // per-thread callbacks may vary, and the consumer clamps to a high-water mark.
    let global_written = AtomicU64::new(0);
    let scope_result: PyResult<u64> = std::thread::scope(|s| {
        let mut handles: Vec<std::thread::ScopedJoinHandle<'_, PyResult<u64>>> = Vec::new();
        for i in 0..shard_count {
            // Each thread owns its shard DB handle, its shard's run list, and a
            // progress handle outright — no shared receiver or borrow — but shares
            // the global record counter so progress reports a single count.
            let db = &shard_databases[i];
            let shard_runs: Vec<PathBuf> = run_paths[i].clone();
            let progress = progress.clone();
            let global = &global_written;
            handles.push(s.spawn(move || {
                write_shard_records(
                    db,
                    &shard_runs,
                    insert_batch,
                    i,
                    global,
                    total_estimate,
                    progress.as_ref(),
                )
            }));
        }

        // Join all shard threads; propagate the first error (first-err wins) and
        // sum the per-shard record counts on success.
        let mut first_err: Option<PyErr> = None;
        let mut written: u64 = 0;
        for h in handles {
            match h.join() {
                Ok(Ok(n)) => written += n,
                Ok(Err(e)) => {
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                }
                Err(_) => {
                    if first_err.is_none() {
                        first_err = Some(PyRuntimeError::new_err("fullmap shard writer panicked"));
                    }
                }
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(written),
        }
    });
    let written = scope_result?;

    if let Some(p) = progress {
        p.call(2, written, written, &format!("wrote {written} records"));
    }

    Ok(())
}

/// One shard's complete Phase-4 work: k-way merge ONLY this shard's spill runs
/// (every term in them hashes to this shard, so the merge groups each term fully
/// with no cross-shard coordination), insert the merged groups into the shard's
/// RECORDS table in hash-sorted `insert_batch` batches for B-tree locality, then
/// commit (Durability::None) and do a final durable commit to persist the pages.
/// Returns the number of records written to this shard.
///
/// A shard with zero runs still opens and commits an empty RECORDS table, so an
/// empty shard DB is produced and the `shard_count`-file layout stays stable.
fn write_shard_records(
    database: &Database,
    run_paths: &[PathBuf],
    insert_batch: usize,
    shard_index: usize,
    global_written: &AtomicU64,
    total: u64,
    progress: Option<&Arc<Progress>>,
) -> PyResult<u64> {
    let mut write = database.begin_write().map_err(py_err)?;
    write.set_durability(Durability::None);
    let mut table = write.open_table(RECORDS).map_err(py_err)?;
    let mut merge = MergeHeap::new(run_paths).map_err(py_err)?;
    // Pre-size the batch to the flush threshold so it never regrows (each
    // doubling copied up to ~80 MB of accumulated records at the default 2M
    // batch).  insert_batch == 0 (unbounded) yields a zero-capacity Vec.
    let mut batch: Vec<(u64, Vec<u8>)> = Vec::with_capacity(insert_batch);
    // Reusable bincode scratch buffer: serialize_into reuses one right-sized
    // allocation across the ~1-2B merged groups instead of bincode::serialize
    // growing a fresh Vec per record.  The encoded bytes are cloned into the
    // batch (they must outlive the next clear), but the clone is a memcpy from a
    // pre-sized buffer, not a from-scratch allocation + growth.
    let mut enc_buf: Vec<u8> = Vec::with_capacity(256);
    let mut written: u64 = 0;
    loop {
        let Some((term, pairs)) = merge.next_group().map_err(py_err)? else {
            break;
        };
        let hash = xxh64(term.as_bytes(), 0);
        enc_buf.clear();
        bincode::serialize_into(&mut enc_buf, &(term.as_str(), &pairs)).map_err(py_err)?;
        batch.push((hash, enc_buf.clone()));
        if insert_batch > 0 && batch.len() >= insert_batch {
            let flushed = flush_shard_batch(&mut table, &mut batch)?;
            written += flushed;
            report_shard_progress(progress, global_written, total, shard_index, flushed);
        }
    }
    let flushed = flush_shard_batch(&mut table, &mut batch)?;
    written += flushed;
    report_shard_progress(progress, global_written, total, shard_index, flushed);
    drop(table);
    write.commit().map_err(py_err)?;
    // Final durable commit to persist all pages.
    let write = database.begin_write().map_err(py_err)?;
    write.commit().map_err(py_err)?;
    Ok(written)
}

/// Sort the shard's pending batch by hash and insert it into the shard's RECORDS
/// table, returning the number of records flushed.  Hash-sorted inserts give
/// near-sequential B-tree appends; clearing the buffer keeps memory bounded.
fn flush_shard_batch(
    table: &mut redb::Table<u64, &[u8]>,
    batch: &mut Vec<(u64, Vec<u8>)>,
) -> PyResult<u64> {
    if batch.is_empty() {
        return Ok(0);
    }
    batch.sort_unstable_by_key(|(hash, _)| *hash);
    for (hash, enc) in batch.iter() {
        table.insert(*hash, enc.as_slice()).map_err(py_err)?;
    }
    let flushed = batch.len() as u64;
    batch.clear();
    Ok(flushed)
}

/// Count the term-frames in one spill-run file WITHOUT allocating the pairs:
/// parse the `[u32 term_len][term][u32 pair_count][5*pair_count]` frame headers
/// and skip the payloads.  Used ONLY to estimate a Phase-2 progress total (and
/// only when a progress callback is present).  The count is an UPPER BOUND on the
/// records actually written: a term recurs across runs and is merged/deduped, so
/// the bar climbs toward ~100% during the merge and the final emission snaps
/// completed to the exact written count.  Mirrors `RunReader::read_frame`'s
/// 1-byte EOF detection so a clean trailing read returns the exact count.
fn count_run_frames(path: &Path) -> std::io::Result<u64> {
    let mut r = BufReader::with_capacity(1 << 20, File::open(path)?);
    let mut count: u64 = 0;
    loop {
        let mut first = [0u8; 1];
        if r.read(&mut first)? == 0 {
            return Ok(count);
        }
        let mut rest = [0u8; 3];
        r.read_exact(&mut rest)?;
        let term_len = u32::from_le_bytes([first[0], rest[0], rest[1], rest[2]]) as u64;
        {
            let mut skip = (&mut r).take(term_len);
            std::io::copy(&mut skip, &mut std::io::sink())?;
        }
        let mut cb = [0u8; 4];
        r.read_exact(&mut cb)?;
        let pair_count = u32::from_le_bytes(cb) as u64;
        {
            let mut skip = (&mut r).take(pair_count * 5);
            std::io::copy(&mut skip, &mut std::io::sink())?;
        }
        count += 1;
    }
}

/// Report Phase-2 progress for a flushed batch: add `flushed` to the shared
/// global counter and emit `(2, global, total, "shard {i}: writing {global}")`.
/// The atomic counter VALUE is monotonic (it only grows), but the ORDER in which
/// concurrent shard threads DELIVER these calls to Python is not guaranteed to be
/// sorted (scheduling between `fetch_add` and the call can deliver e.g. 150 then
/// 100).  That is harmless: the consumer (progress.py) clamps to a per-phase
/// high-water mark.  A no-op for an empty flush or when there is no callback.
fn report_shard_progress(
    progress: Option<&Arc<Progress>>,
    global_written: &AtomicU64,
    total: u64,
    shard_index: usize,
    flushed: u64,
) {
    if flushed == 0 {
        return;
    }
    if let Some(p) = progress {
        let global = global_written.fetch_add(flushed, Ordering::Relaxed) + flushed;
        p.call(
            2,
            global,
            total,
            &format!("shard {shard_index}: writing {global} records"),
        );
    }
}

// ---------------------------------------------------------------------------
// Build orchestrator
// ---------------------------------------------------------------------------

/// Route a normalized term to its on-disk RECORDS shard index via xxh64 masked
/// to the shard count.  `shard_count` must be a power of two.  The same hash is
/// the RECORDS key, so a term's shard and its key are derived from one xxh64 call
/// site each (write and read agree).  This is the single routing oracle shared by
/// the writer and the reader.
#[allow(clippy::cast_possible_truncation)] // xxh64->usize: only the low bits feed the shard mask, so truncation is the intended routing
fn term_shard(term: &str, shard_count: usize) -> usize {
    (xxh64(term.as_bytes(), 0) as usize) & (shard_count - 1)
}

/// Sibling shard file path for a primary DB path: `.../fullmap.redb` ->
/// `.../fullmap.s{index}.redb` (same directory, primary file stem preserved).
fn shard_path(primary: &Path, index: usize) -> PathBuf {
    let stem = primary
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "fullmap".to_string());
    let ext = primary
        .extension()
        .map(|e| e.to_string_lossy().into_owned())
        .unwrap_or_else(|| "redb".to_string());
    let name = format!("{stem}.s{index}.{ext}");
    match primary.parent() {
        Some(dir) => dir.join(name),
        None => PathBuf::from(name),
    }
}

fn evict_cached_path(path: &Path) -> PyResult<()> {
    let Some(cache) = DB_CACHE.get() else {
        return Ok(());
    };
    let mut map = cache.write().map_err(py_err)?;
    let primary = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    map.remove(&primary);
    // Evict every shard handle too so a rebuild never reads a stale shard file.
    for index in 0..SHARD_COUNT_SHARDS {
        let shard = shard_path(path, index);
        let canonical = std::fs::canonicalize(&shard).unwrap_or(shard);
        map.remove(&canonical);
    }
    Ok(())
}

fn cache_database(path: &Path, database: Arc<Database>) -> PyResult<()> {
    let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let cache = DB_CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    cache.write().map_err(py_err)?.insert(canonical, database);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_fullmap_inner(
    output: PathBuf,
    classes: Vec<PathBuf>,
    synonyms: Vec<PathBuf>,
    worker_count: usize,
    shard_count: usize,
    progress: Option<Arc<Progress>>,
    local_spill: usize,
    curie_spill: usize,
    exclude_prefixes: HashSet<String>,
    chunk_bytes: usize,
    producers: usize,
    cache_bytes: usize,
    insert_batch: usize,
    spill_dir: PathBuf,
) -> PyResult<()> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(worker_count)
        .build()
        .map_err(py_err)?;

    pool.install(|| {
        // Fresh spill directory for this build.
        if spill_dir.exists() {
            std::fs::remove_dir_all(&spill_dir).map_err(py_err)?;
        }
        std::fs::create_dir_all(&spill_dir).map_err(py_err)?;

        // Phase 1: disk-backed equivalents index from class files.
        let equivalents = EquivIndex::build(&classes, &spill_dir, progress.as_ref())?;

        // Phases 2+3: single-pass synonym processing with bounded spill runs.
        let result = process_synonyms(
            &synonyms,
            &equivalents,
            &spill_dir,
            local_spill,
            curie_spill,
            &exclude_prefixes,
            worker_count,
            shard_count,
            chunk_bytes,
            producers,
            progress.as_ref(),
        )?;
        // Equivalents index (mmap + temp file) is dropped here; the data file
        // is deleted via EquivIndex::Drop.
        drop(equivalents);

        let SynonymBuildResult {
            prefix_ids,
            category_ids,
            source_ids,
            curie_run_paths,
            run_paths,
        } = result;

        // Phase 4: k-way merge of runs streamed into the final database.
        write_final_database(
            &output,
            &prefix_ids,
            &category_ids,
            &source_ids,
            &curie_run_paths,
            &run_paths,
            cache_bytes,
            insert_batch,
            shard_count,
            progress.as_ref(),
        )?;

        // Clean up spill runs on success (left in place on error for inspection).
        let _ = std::fs::remove_dir_all(&spill_dir);
        Ok::<(), PyErr>(())
    })
}

#[pyfunction]
#[pyo3(signature = (output, classes, synonyms, threads=None, progress=None))]
pub fn build_fullmap_db(
    py: Python<'_>,
    output: PathBuf,
    classes: Vec<PathBuf>,
    synonyms: Vec<PathBuf>,
    threads: Option<usize>,
    progress: Option<Py<PyAny>>,
) -> PyResult<()> {
    if synonyms.is_empty() {
        return Err(PyValueError::new_err(
            "at least one synonym file is required",
        ));
    }

    // Phase 4 opens every spill-run file for a shard at once (MergeHeap), across
    // all shards in parallel — large corpora can need thousands of fds per
    // shard. Best-effort: raise our soft NOFILE limit to the hard limit so
    // this isn't capped by a low default (e.g. ulimit -n 1024). Never fails
    // the build if this doesn't succeed.
    let _ = rlimit::increase_nofile_limit(u64::MAX);

    let worker_count = threads
        .unwrap_or_else(|| {
            let cpus = std::thread::available_parallelism()
                .map(std::num::NonZero::get)
                .unwrap_or(1);
            // Cap at available_memory_gb / 2 to prevent swap on memory-constrained
            // machines.  Each thread uses ~400 MB of local buffers; the cap is
            // generous (2 GB/thread) to avoid limiting CPU-bound throughput.
            let avail_kb = std::fs::read_to_string("/proc/meminfo")
                .ok()
                .and_then(|s| {
                    s.lines()
                        .find(|l| l.starts_with("MemAvailable:"))
                        .and_then(|l| {
                            l.split_whitespace()
                                .nth(1)
                                .and_then(|v| v.parse::<usize>().ok())
                        })
                })
                .unwrap_or(0);
            if avail_kb > 0 {
                let avail_gb = avail_kb / (1024 * 1024);
                let mem_cap = (avail_gb / 2).max(1);
                cpus.min(mem_cap)
            } else {
                cpus * 9 / 10
            }
        })
        .max(1);

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(py_err)?;
    }
    evict_cached_path(&output)?;
    if output.exists() {
        std::fs::remove_file(&output).map_err(py_err)?;
    }
    // Remove any stale shard files from a previous build so the new 4-file
    // layout is never mixed with leftover shards.
    for index in 0..SHARD_COUNT_SHARDS {
        let shard = shard_path(&output, index);
        if shard.exists() {
            std::fs::remove_file(&shard).map_err(py_err)?;
        }
    }

    let local_spill = env_usize(
        "TABLASSERT_FULLMAP_LOCAL_SPILL_ENTRIES",
        DEFAULT_LOCAL_SPILL_ENTRIES,
    );
    let curie_spill = env_usize(
        "TABLASSERT_FULLMAP_CURIE_SPILL_ENTRIES",
        DEFAULT_CURIE_SPILL_ENTRIES,
    );
    // Opt-in prefix exclusion: comma-separated CURIE prefixes dropped at build
    // time (e.g. "INCHIKEY,Publication").  Empty/unset = index everything.
    let exclude_prefixes: HashSet<String> = std::env::var("TABLASSERT_FULLMAP_EXCLUDE_PREFIXES")
        .ok()
        .map(|v| {
            v.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();
    // Intra-file parallelism tunables: lines per producer->worker chunk, and the
    // number of producer (decompressor) threads.  Decompression is far faster
    // than parallel processing, so a handful of producers keeps all workers fed.
    let chunk_bytes = env_usize("TABLASSERT_FULLMAP_CHUNK_BYTES", DEFAULT_CHUNK_BYTES);
    let default_producers = (worker_count / 4).max(4).min(synonyms.len().max(1));
    let producers = env_usize("TABLASSERT_FULLMAP_PRODUCERS", default_producers);
    let cache_bytes = env_usize(
        "TABLASSERT_FULLMAP_REDB_CACHE_BYTES",
        DEFAULT_REDB_CACHE_BYTES,
    );
    let insert_batch = env_usize("TABLASSERT_FULLMAP_INSERT_BATCH", DEFAULT_INSERT_BATCH);
    // On-disk RECORDS shard count (one redb file + concurrent writer per shard).
    // Defaults to SHARD_COUNT_SHARDS (16); non-powers-of-two round down,
    // clamped to SHARD_COUNT_SHARDS.
    let shard_count = resolve_shard_count();
    let spill_dir = std::env::var("TABLASSERT_FULLMAP_SPILL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let mut p = output.clone().into_os_string();
            p.push(".spill.d");
            PathBuf::from(p)
        });

    let progress = progress.map(|cb| Arc::new(Progress { cb }));

    // Release the GIL for the whole build so rich's Live display thread can
    // repaint and Ctrl-C works; progress callbacks re-acquire it briefly.
    py.detach(|| {
        build_fullmap_inner(
            output.clone(),
            classes,
            synonyms,
            worker_count,
            shard_count,
            progress,
            local_spill,
            curie_spill,
            exclude_prefixes,
            chunk_bytes,
            producers,
            cache_bytes,
            insert_batch,
            spill_dir,
        )
    })?;

    // Cache the freshly-built database for the read path.
    let database = Arc::new(Database::open(&output).map_err(py_err)?);
    cache_database(&output, database)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Read path
// ---------------------------------------------------------------------------

fn validate_schema(database: &Database) -> PyResult<()> {
    let read = database.begin_read().map_err(py_err)?;
    let meta = read.open_table(META).map_err(py_err)?;
    let schema = meta
        .get("schema")
        .map_err(py_err)?
        .map(|x| x.value().to_string());
    match schema.as_deref() {
        Some(SCHEMA_VERSION) => Ok(()),
        Some(SCHEMA_VERSION_V3 | SCHEMA_VERSION_V2 | SCHEMA_VERSION_V1) => {
            Err(PyRuntimeError::new_err(
                "fullmap DB is outdated; rebuild with 'tablassert build-fullmap'",
            ))
        }
        _ => Err(PyRuntimeError::new_err("unsupported fullmap redb schema")),
    }
}

fn open_cached(db: PathBuf) -> PyResult<Arc<Database>> {
    let canonical = std::fs::canonicalize(&db).unwrap_or(db);
    let cache = DB_CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    if let Some(database) = cache.read().map_err(py_err)?.get(&canonical) {
        return Ok(Arc::clone(database));
    }
    let database = Arc::new(Database::open(&canonical).map_err(py_err)?);
    validate_schema(&database)?;
    let cached = Arc::clone(&database);
    cache.write().map_err(py_err)?.insert(canonical, database);
    Ok(cached)
}

/// Open (and cache) one RECORDS shard by index, deriving its path from the
/// primary DB path.  Shards hold only RECORDS (no META), so they are not
/// schema-validated here — the primary's `validate_schema` gates the layout.
fn open_cached_shard(primary: &Path, index: usize) -> PyResult<Arc<Database>> {
    let path = shard_path(primary, index);
    let canonical = std::fs::canonicalize(&path).unwrap_or(path);
    let cache = DB_CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    if let Some(database) = cache.read().map_err(py_err)?.get(&canonical) {
        return Ok(Arc::clone(database));
    }
    let database = Arc::new(Database::open(&canonical).map_err(py_err)?);
    let cached = Arc::clone(&database);
    cache.write().map_err(py_err)?.insert(canonical, database);
    Ok(cached)
}

/// Read the RECORDS shard count advertised in the primary's META, defaulting to
/// `SHARD_COUNT_SHARDS` when absent.  The write path sets this from the runtime
/// `TABLASSERT_FULLMAP_SHARDS` tunable, so the read path opens exactly the shards
/// that exist and routes with the matching mask.
fn shard_count_of(database: &Database) -> PyResult<usize> {
    let read = database.begin_read().map_err(py_err)?;
    let meta = read.open_table(META).map_err(py_err)?;
    let count = meta
        .get("shards")
        .map_err(py_err)?
        .and_then(|v| v.value().parse::<usize>().ok())
        .unwrap_or(SHARD_COUNT_SHARDS);
    // Round down to a power of two before clamping, mirroring the write path's
    // `resolve_shard_count`: the routing mask `xxh64(term) & (count - 1)` is only
    // correct for powers of two, so a hand-edited non-pow2 META.shards (e.g. 3 ->
    // mask &2) would silently misroute/drop lookups onto a subset of shards.
    // `round_down_pow2(SHARD_COUNT_SHARDS) == SHARD_COUNT_SHARDS` (16 is a pow2),
    // so the default fallback above is preserved.
    Ok(round_down_pow2(count).clamp(1, SHARD_COUNT_SHARDS))
}

/// Open (and cache) all RECORDS shard handles for a primary DB path.  The shard
/// count is read from the primary's META so the read path opens exactly the
/// shards the build wrote.
fn open_cached_shards(primary: &Path) -> PyResult<Vec<Arc<Database>>> {
    let database = open_cached(primary.to_path_buf())?;
    let shard_count = shard_count_of(&database)?;
    (0..shard_count)
        .map(|index| open_cached_shard(primary, index))
        .collect()
}

fn lookup_pair_chunk(shards: &[Arc<Database>], terms: &[String]) -> PyResult<PairRecords> {
    // One read transaction + RECORDS table per shard, opened once; each query
    // term is routed to its shard via `term_shard`.
    let reads: Vec<_> = shards
        .iter()
        .map(|db| db.begin_read().map_err(py_err))
        .collect::<PyResult<_>>()?;
    let tables: Vec<_> = reads
        .iter()
        .map(|read| read.open_table(RECORDS).map_err(py_err))
        .collect::<PyResult<_>>()?;
    let mut out = Vec::new();
    for term in terms {
        let hash = xxh64(term.as_bytes(), 0);
        let table = &tables[term_shard(term, tables.len())];
        if let Some(bytes) = table.get(hash).map_err(py_err)? {
            let (stored_term, records): (String, Vec<(u32, u8)>) =
                bincode::deserialize(bytes.value()).map_err(py_err)?;
            // Verify the term matches (guards against xxh64 collisions).
            if &stored_term == term {
                out.push((term.clone(), records));
            }
        }
    }
    Ok(out)
}

/// Query a single RECORDS shard for the `(input_index, term)` pairs routed to
/// it.  Opens one read transaction + RECORDS table on `shard`, looks up each
/// term by its xxh64 key, and keeps the `stored_term == term` collision guard.
/// Hits are tagged with their original input index so the caller can re-merge
/// every shard's results back into input order.  Pure Rust (no `Python`), so it
/// is safe to run on a worker thread inside a `py.detach` region.
fn lookup_shard_bucket(
    shard: &Database,
    bucket: &[(usize, String)],
) -> PyResult<Vec<(usize, TermPairs)>> {
    let read = shard.begin_read().map_err(py_err)?;
    let table = read.open_table(RECORDS).map_err(py_err)?;
    let mut out = Vec::new();
    for (index, term) in bucket {
        let hash = xxh64(term.as_bytes(), 0);
        if let Some(bytes) = table.get(hash).map_err(py_err)? {
            let (stored_term, records): (String, Vec<(u32, u8)>) =
                bincode::deserialize(bytes.value()).map_err(py_err)?;
            // Verify the term matches (guards against xxh64 collisions).
            if &stored_term == term {
                out.push((*index, (term.clone(), records)));
            }
        }
    }
    Ok(out)
}

/// Fan the query terms out across RECORDS shards and read the non-empty shards
/// concurrently.  Terms are partitioned by `term_shard` (the shared routing
/// oracle) into per-shard buckets; one worker thread per NON-EMPTY shard — capped
/// at `workers` — reads only its own shard's RECORDS, and the tagged hits are
/// re-merged into the original input term order.  With
/// shard_count=SHARD_COUNT_SHARDS and enough workers this is up to
/// SHARD_COUNT_SHARDS concurrent shard reads.  Surplus shards beyond the worker
/// cap are read on the calling thread, which still overlaps with the spawned
/// readers.  Pure Rust end-to-end (no `Python`).
fn lookup_pair_terms_db(
    shards: &[Arc<Database>],
    terms: &[String],
    workers: usize,
) -> PyResult<PairRecords> {
    if terms.is_empty() {
        return Ok(Vec::new());
    }
    let shard_count = shards.len();
    // Single-threaded fast path: one worker, one term, or a single shard.
    if workers <= 1 || terms.len() <= 1 || shard_count <= 1 {
        return lookup_pair_chunk(shards, terms);
    }

    // Partition the query terms into per-shard buckets, tagging each with its
    // input position so hits can be re-merged in the original order afterwards.
    let mut buckets: Vec<Vec<(usize, String)>> = vec![Vec::new(); shard_count];
    for (index, term) in terms.iter().enumerate() {
        buckets[term_shard(term, shard_count)].push((index, term.clone()));
    }

    // One job per NON-EMPTY shard; each job owns its bucket and a clone of its
    // shard handle so worker threads share neither a receiver nor a borrow.
    let mut jobs: Vec<ShardJob> = buckets
        .into_iter()
        .enumerate()
        .filter(|(_, bucket)| !bucket.is_empty())
        .map(|(shard, bucket)| (bucket, Arc::clone(&shards[shard])))
        .collect();

    // Cap concurrent shard reads at `workers`; any surplus shards are read on
    // the calling thread (split_off keeps the first `workers` jobs to spawn).
    let split = jobs.len().min(workers);
    let inline_jobs = jobs.split_off(split);

    let mut handles = Vec::with_capacity(jobs.len());
    for (bucket, shard) in jobs {
        handles.push(thread::spawn(move || lookup_shard_bucket(&shard, &bucket)));
    }

    let mut tagged: Vec<(usize, TermPairs)> = Vec::new();
    for (bucket, shard) in inline_jobs {
        tagged.extend(lookup_shard_bucket(&shard, &bucket)?);
    }
    for handle in handles {
        let mut part = handle
            .join()
            .map_err(|_| PyRuntimeError::new_err("fullmap lookup thread panicked"))??;
        tagged.append(&mut part);
    }

    // Re-merge hits into input term order; misses simply produced no row.
    tagged.sort_by_key(|(index, _)| *index);
    Ok(tagged.into_iter().map(|(_, pairs)| pairs).collect())
}

/// Smallest batch that defaults to parallel shard fan-out when the caller passes
/// no `threads`.  Below this, lookups stay single-threaded: a point/small lookup
/// (and any cache-warm path) finishes faster serially than the cost of spawning
/// shard-reader threads.  At/above it, the per-shard fan-out in
/// `lookup_pair_terms_db` wins.  1024 terms ~= a few ms of serial redb point
/// reads, comfortably above the ~tens-of-µs cost of spawning up to 3 extra
/// threads, so the crossover is safely on the parallel side for real batches
/// while never penalizing small lookups.
const LOOKUP_PARALLEL_MIN: usize = 1024;

/// Default worker count for lookups when the caller passes no `threads`.
/// The production build-graph resolve sends ONE batch of all distinct node-column
/// terms (often huge) with `threads=None`; parallelizing that across the RECORDS
/// shards is the win, so large batches default to `available_parallelism`.  Small
/// batches (< `LOOKUP_PARALLEL_MIN`) stay single-threaded to avoid spawn overhead.
/// The fan-out is already capped by the non-empty shard count inside
/// `lookup_pair_terms_db`, so returning `available_parallelism` yields up to
/// SHARD_COUNT_SHARDS actual shard threads regardless of core count.
fn default_lookup_workers(terms_len: usize) -> usize {
    if terms_len < LOOKUP_PARALLEL_MIN {
        return 1;
    }
    std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1)
}

fn lookup_pair_terms(
    db: PathBuf,
    terms: Vec<String>,
    threads: Option<usize>,
) -> PyResult<PairRecords> {
    let workers = threads
        .unwrap_or_else(|| default_lookup_workers(terms.len()))
        .max(1)
        .min(terms.len().max(1));
    // Open (and schema-validate) the primary, then route pair lookups to shards.
    let _primary = open_cached(db.clone())?;
    let shards = open_cached_shards(&db)?;
    lookup_pair_terms_db(&shards, &terms, workers)
}

fn load_string_table(
    database: &Database,
    table_definition: TableDefinition<u16, &str>,
) -> PyResult<HashMap<u16, String>> {
    let read = database.begin_read().map_err(py_err)?;
    let table = read.open_table(table_definition).map_err(py_err)?;
    let mut out = HashMap::new();
    for item in table.iter().map_err(py_err)? {
        let (id, value) = item.map_err(py_err)?;
        out.insert(id.value(), value.value().to_string());
    }
    Ok(out)
}

fn load_sources(database: &Database) -> PyResult<HashMap<u8, String>> {
    let read = database.begin_read().map_err(py_err)?;
    let table = read.open_table(SOURCES).map_err(py_err)?;
    let mut out = HashMap::new();
    for item in table.iter().map_err(py_err)? {
        let (id, value) = item.map_err(py_err)?;
        let row: SourceRow = bincode::deserialize(value.value()).map_err(py_err)?;
        out.insert(id.value(), row.source_name);
    }
    Ok(out)
}

fn hydrate_curie_rows(database: &Database, curie_ids: &[u32]) -> PyResult<Vec<CurieRow>> {
    let read = database.begin_read().map_err(py_err)?;
    let table = read.open_table(CURIES).map_err(py_err)?;
    let mut out = Vec::with_capacity(curie_ids.len());
    for curie_id in curie_ids {
        let bytes = table.get(*curie_id).map_err(py_err)?.ok_or_else(|| {
            PyRuntimeError::new_err(format!("missing fullmap CURIE id {curie_id}"))
        })?;
        out.push(bincode::deserialize(bytes.value()).map_err(py_err)?);
    }
    Ok(out)
}

fn lookup_terms(
    db: PathBuf,
    terms: Vec<String>,
    threads: Option<usize>,
) -> PyResult<Vec<(String, Vec<FullmapRecord>)>> {
    // Open the primary ONCE for dims/CURIES hydration; pair lookups route to the
    // shard files (a second open of any one file would fail on redb's flock).
    let database = open_cached(db.clone())?;
    let prefix_map = load_string_table(&database, PREFIXES)?;
    let category_map = load_string_table(&database, CATEGORIES)?;
    let source_map = load_sources(&database)?;
    let shards = open_cached_shards(&db)?;
    let workers = threads
        .unwrap_or_else(|| default_lookup_workers(terms.len()))
        .max(1)
        .min(terms.len().max(1));
    let pair_rows = lookup_pair_terms_db(&shards, &terms, workers)?;
    let mut curie_ids: Vec<u32> = pair_rows
        .iter()
        .flat_map(|(_term, pairs)| pairs.iter().map(|(curie_id, _source_id)| *curie_id))
        .collect();
    curie_ids.sort_unstable();
    curie_ids.dedup();
    let hydrated = hydrate_curie_rows(&database, &curie_ids)?;
    let curie_map: HashMap<u32, CurieRow> = curie_ids.into_iter().zip(hydrated).collect();
    let source_version = FULLMAP_SOURCE_VERSION.to_string();
    let mut out = Vec::new();
    for (term, pairs) in pair_rows {
        let mut records = Vec::new();
        for (curie_id, source_id) in pairs {
            let curie = curie_map.get(&curie_id).ok_or_else(|| {
                PyRuntimeError::new_err(format!("missing hydrated CURIE id {curie_id}"))
            })?;
            let prefix = prefix_map.get(&curie.prefix_id).ok_or_else(|| {
                PyRuntimeError::new_err(format!("missing prefix id {}", curie.prefix_id))
            })?;
            let category = category_map.get(&curie.category_id).ok_or_else(|| {
                PyRuntimeError::new_err(format!("missing category id {}", curie.category_id))
            })?;
            let source = source_map
                .get(&source_id)
                .ok_or_else(|| PyRuntimeError::new_err(format!("missing source id {source_id}")))?;
            records.push(hydrate_record(
                curie,
                source.clone(),
                source_version.clone(),
                prefix,
                category,
            ));
        }
        out.push((term, records));
    }
    Ok(out)
}

/// Look up fullmap records for `terms`.  `threads=None` (the production
/// build-graph default) auto-selects the worker count via `default_lookup_workers`:
/// batches >= `LOOKUP_PARALLEL_MIN` fan out across the RECORDS shards in
/// parallel (up to 16 by default), smaller batches stay single-threaded.  An
/// explicit `threads=Some(1)` always forces the serial path.  The GIL is released
/// for the whole lookup.
#[pyfunction]
#[pyo3(signature = (db, terms, threads=None, return_format="rows"))]
pub fn lookup_fullmap_terms<'py>(
    py: Python<'py>,
    db: PathBuf,
    terms: Vec<String>,
    threads: Option<usize>,
    return_format: &str,
) -> PyResult<Bound<'py, PyList>> {
    if return_format == "pairs" {
        // Release the GIL for the whole lookup (pure-Rust shard reads); the
        // PyList is built only after re-acquiring it so rich's Live display
        // thread can repaint and Ctrl-C works mid-lookup.
        let pair_rows = py.detach(move || lookup_pair_terms(db, terms, threads))?;
        let list = PyList::empty(py);
        for (term, pairs) in pair_rows {
            let row = PyDict::new(py);
            row.set_item("term", term)?;
            row.set_item("records", pairs)?;
            list.append(row)?;
        }
        return Ok(list);
    }
    if return_format != "rows" {
        return Err(PyValueError::new_err(
            "return_format must be 'rows' or 'pairs'",
        ));
    }
    // Release the GIL for the whole lookup (pure-Rust shard reads + CURIE/dim
    // hydration against the primary); the PyList is built only after
    // re-acquiring the GIL.
    let rows = py.detach(move || lookup_terms(db, terms, threads))?;
    let list = PyList::empty(py);
    for (term, records) in rows {
        for record in records {
            let row = PyDict::new(py);
            row.set_item("term", &term)?;
            row.set_item("CURIE", record.curie)?;
            row.set_item("PREFERRED_NAME", record.preferred_name)?;
            row.set_item("CATEGORY_NAME", record.category_name)?;
            row.set_item("TAXON_ID", record.taxon_id)?;
            row.set_item("SOURCE_NAME", record.source_name)?;
            row.set_item("SOURCE_VERSION", record.source_version)?;
            list.append(row)?;
        }
    }
    Ok(list)
}

#[pyfunction]
pub fn hydrate_curies<'py>(
    py: Python<'py>,
    db: PathBuf,
    curie_ids: Vec<u32>,
) -> PyResult<Bound<'py, PyList>> {
    let database = open_cached(db)?;
    let rows = hydrate_curie_rows(&database, &curie_ids)?;
    let list = PyList::empty(py);
    for row in rows {
        let item = PyDict::new(py);
        item.set_item("prefix_id", row.prefix_id)?;
        item.set_item("local_id", row.local_id)?;
        item.set_item("preferred_name", row.preferred_name)?;
        item.set_item("category_id", row.category_id)?;
        item.set_item("taxon_id", row.taxon_id)?;
        list.append(item)?;
    }
    Ok(list)
}

#[pyfunction]
pub fn hydrate_sources<'py>(py: Python<'py>, db: PathBuf) -> PyResult<Bound<'py, PyList>> {
    let database = open_cached(db)?;
    let sources = load_sources(&database)?;
    let max_id = sources.keys().copied().max().unwrap_or(0);
    let list = PyList::empty(py);
    for id in 0..=max_id {
        list.append(sources.get(&id).cloned().unwrap_or_default())?;
    }
    Ok(list)
}

#[pyfunction]
pub fn hydrate_prefixes<'py>(py: Python<'py>, db: PathBuf) -> PyResult<Bound<'py, PyList>> {
    let database = open_cached(db)?;
    let prefixes = load_string_table(&database, PREFIXES)?;
    let max_id = prefixes.keys().copied().max().unwrap_or(0);
    let list = PyList::empty(py);
    for id in 0..=max_id {
        list.append(prefixes.get(&id).cloned().unwrap_or_default())?;
    }
    Ok(list)
}

#[pyfunction]
pub fn hydrate_categories<'py>(py: Python<'py>, db: PathBuf) -> PyResult<Bound<'py, PyList>> {
    let database = open_cached(db)?;
    let categories = load_string_table(&database, CATEGORIES)?;
    let max_id = categories.keys().copied().max().unwrap_or(0);
    let list = PyList::empty(py);
    for id in 0..=max_id {
        list.append(categories.get(&id).cloned().unwrap_or_default())?;
    }
    Ok(list)
}

#[pyfunction]
pub fn fullmap_source_version() -> &'static str {
    FULLMAP_SOURCE_VERSION
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Test helper: build with explicit tunables (no Python token / env needed).
    fn build_test(
        output: PathBuf,
        classes: Vec<PathBuf>,
        synonyms: Vec<PathBuf>,
        threads: usize,
        local_spill: usize,
    ) -> PyResult<()> {
        let spill_dir = {
            let mut p = output.clone().into_os_string();
            p.push(".spill.d");
            PathBuf::from(p)
        };
        build_fullmap_inner(
            output,
            classes,
            synonyms,
            threads.max(1),
            SHARD_COUNT_SHARDS,
            None,
            local_spill,
            1_000_000,
            HashSet::new(),
            DEFAULT_CHUNK_BYTES,
            2,
            64 * 1024 * 1024,
            1000,
            spill_dir,
        )
    }

    /// `build_fullmap_db` raises the soft NOFILE limit to the hard limit so
    /// Phase 4's per-shard MergeHeap (which opens every spill-run file for a
    /// shard at once) isn't capped by a low default (e.g. ulimit -n 1024).
    #[test]
    fn increase_nofile_limit_reaches_hard_limit() {
        let (_, hard) = rlimit::Resource::NOFILE.get().expect("getrlimit");
        let new_soft = rlimit::increase_nofile_limit(u64::MAX).expect("raise nofile limit");
        assert_eq!(new_soft, hard);
        let (soft_after, hard_after) = rlimit::Resource::NOFILE.get().expect("getrlimit");
        assert_eq!(soft_after, hard);
        assert_eq!(hard_after, hard);
    }

    #[test]
    fn clean_strips_matching_and_duplicate_quotes() {
        assert_eq!(clean("  'BRCA1'  "), "BRCA1");
        assert_eq!(clean("\"\"TP53\""), "TP53");
    }

    #[test]
    fn token_qc_rejects_banned_tokens() {
        assert!(token_qc("brca1"));
        assert!(!token_qc("hypothetical protein"));
        assert!(!token_qc("line\nbreak"));
    }

    #[test]
    fn level_normalization_matches_fullmap_shape() {
        assert_eq!(level_one("BRCA-1"), "brca-1");
        assert_eq!(level_two("brca-1"), "brca1");
    }

    /// `term_shard` is the single routing oracle shared by the writer and the
    /// reader, so it must be deterministic (same term -> same shard every call,
    /// or reads would look in the wrong file), agree with the raw xxh64-mask
    /// definition, and spread a representative sample across all default shards so the
    /// shard files stay roughly even-sized.
    #[test]
    fn term_shard_is_deterministic_and_balanced() {
        // Deterministic, in range, and equal to an independent xxh64-mask
        // re-derivation for known terms.
        for term in ["brca1", "tp53", "water", "alias disease", "gene42"] {
            let expected = (xxh64(term.as_bytes(), 0) as usize) & (SHARD_COUNT_SHARDS - 1);
            assert_eq!(
                term_shard(term, SHARD_COUNT_SHARDS),
                expected,
                "routing mismatch for {term}"
            );
            assert_eq!(
                term_shard(term, SHARD_COUNT_SHARDS),
                term_shard(term, SHARD_COUNT_SHARDS),
                "non-deterministic for {term}"
            );
            assert!(term_shard(term, SHARD_COUNT_SHARDS) < SHARD_COUNT_SHARDS);
        }

        // Balanced: 1000 distinct synthetic terms hit every shard, none
        // dominating (xxh64 is well-mixed; allow a generous +/- 50% band).
        let mut counts = [0usize; SHARD_COUNT_SHARDS];
        for i in 0..1000 {
            counts[term_shard(&format!("term{i}"), SHARD_COUNT_SHARDS)] += 1;
        }
        let expected = 1000 / SHARD_COUNT_SHARDS;
        for (shard, count) in counts.iter().enumerate() {
            assert!(*count > 0, "shard {shard} received no terms: {counts:?}");
            assert!(
                *count > expected / 2 && *count < expected * 2,
                "shard distribution unbalanced: {counts:?}"
            );
        }
    }

    #[test]
    fn builds_and_reads_redb_records() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let classes = dir.path().join("classes.ndjson");
        let synonyms = dir.path().join("HGNC.ndjson");
        let output = dir.path().join("fullmap.redb");

        let mut class_file = File::create(&classes).unwrap();
        writeln!(
            class_file,
            r#"{{"id":"HGNC:1100","equivalent_identifiers":[{{"identifier":"NCBIGene:672"}}]}}"#
        )
        .unwrap();

        let mut synonym_file = File::create(&synonyms).unwrap();
        writeln!(
            synonym_file,
            r#"{{"curie":"HGNC:1100","preferred_name":"BRCA1","names":["BRCA1"],"types":["Gene"],"taxa":["NCBITaxon:9606"]}}"#
        )
        .unwrap();

        build_test(output.clone(), vec![classes], vec![synonyms], 1, 4_000_000).unwrap();
        let rows = lookup_terms(
            output,
            vec!["brca1".to_string(), "ncbigene672".to_string()],
            Some(1),
        )
        .unwrap();

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].1[0].curie, "HGNC:1100");
        assert_eq!(rows[0].1[0].source_version, FULLMAP_SOURCE_VERSION);
    }

    /// The v4 layout keeps dims+CURIES+META in the primary and moves RECORDS
    /// into sibling shard files.  The primary must NOT carry a RECORDS table,
    /// META must advertise both the schema and the shard count, and every shard
    /// file must exist (even empty) holding a RECORDS table — this is the
    /// on-disk contract the read path relies on.
    #[test]
    fn build_fullmap_db_writes_schema_v4_sharded_layout() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let synonyms = dir.path().join("HGNC.ndjson");
        let output = dir.path().join("fullmap.redb");

        let mut synonym_file = File::create(&synonyms).unwrap();
        writeln!(
            synonym_file,
            r#"{{"curie":"HGNC:1100","preferred_name":"BRCA1","names":["BRCA1"],"types":["Gene"],"taxa":["NCBITaxon:9606"]}}"#
        )
        .unwrap();

        build_test(output.clone(), Vec::new(), vec![synonyms], 1, 4_000_000).unwrap();

        // Primary: META (schema=v4, shards=SHARD_COUNT_SHARDS) + dims + CURIES,
        // but NO RECORDS.
        let database = open_cached(output.clone()).unwrap();
        let read = database.begin_read().unwrap();
        let meta = read.open_table(META).unwrap();
        assert_eq!(meta.get("schema").unwrap().unwrap().value(), SCHEMA_VERSION);
        assert_eq!(
            meta.get("shards").unwrap().unwrap().value(),
            SHARD_COUNT_SHARDS.to_string()
        );
        drop(meta);
        let _prefixes = read.open_table(PREFIXES).unwrap();
        let _categories = read.open_table(CATEGORIES).unwrap();
        let _sources = read.open_table(SOURCES).unwrap();
        let _curies = read.open_table(CURIES).unwrap();
        assert!(
            read.open_table(RECORDS).is_err(),
            "primary must not hold a RECORDS table in the v4 layout"
        );
        drop(read);
        drop(database);

        // All default shard files exist and each holds a RECORDS table.
        for index in 0..SHARD_COUNT_SHARDS {
            let shard = shard_path(&output, index);
            assert!(shard.exists(), "missing shard file {shard:?}");
            let db = Database::open(&shard).unwrap();
            let read = db.begin_read().unwrap();
            let _records = read.open_table(RECORDS).unwrap();
        }

        // The single indexed term still resolves (routed through its shard).
        let rows = lookup_terms(output, vec!["brca1".to_string()], Some(1)).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].1[0].curie, "HGNC:1100");
    }

    /// Per-shard RECORDS row counts for a built primary DB, in shard order.
    /// Used to prove the parallel write routes records identically regardless of
    /// writer count (routing is a pure function of each term's xxh64 hash).
    fn shard_record_counts(primary: &Path) -> Vec<usize> {
        let database = open_cached(primary.to_path_buf()).unwrap();
        let shard_count = shard_count_of(&database).unwrap();
        (0..shard_count)
            .map(|index| {
                let db = open_cached_shard(primary, index).unwrap();
                let read = db.begin_read().unwrap();
                let table = read.open_table(RECORDS).unwrap();
                table.iter().unwrap().count()
            })
            .collect()
    }

    /// `TABLASSERT_FULLMAP_SHARDS` may be any positive integer, but mask routing
    /// only works for powers of two; `round_down_pow2` must round 6->4, 3->2,
    /// 5->4, leave powers unchanged, and floor tiny/zero inputs at 1 so the shard
    /// count is always a valid non-zero mask.
    #[test]
    fn round_down_pow2_rounds_to_nearest_lower_power_of_two() {
        assert_eq!(round_down_pow2(0), 1);
        assert_eq!(round_down_pow2(1), 1);
        assert_eq!(round_down_pow2(2), 2);
        assert_eq!(round_down_pow2(3), 2);
        assert_eq!(round_down_pow2(4), 4);
        assert_eq!(round_down_pow2(5), 4);
        assert_eq!(round_down_pow2(6), 4);
        assert_eq!(round_down_pow2(7), 4);
        assert_eq!(round_down_pow2(8), 8);
        assert_eq!(round_down_pow2(9), 8);
    }

    /// The read path's `shard_count_of` must round a non-power-of-two META.shards
    /// DOWN to a power of two, exactly like the write path's `resolve_shard_count`.
    /// WHY: lookups route with the mask `xxh64(term) & (count - 1)`, which is only
    /// correct for powers of two; a hand-edited META.shards of 3 (mask &2) would
    /// route terms only onto shards {0,2}, silently misrouting/dropping lookups
    /// (shard 1 opened but never queried, shard 3 never opened).  Rounding down
    /// keeps the mask valid.  A missing/unparseable value falls back to the default
    /// `SHARD_COUNT_SHARDS` (16) via `unwrap_or`, itself a power of two so the
    /// round-down leaves it unchanged.  `"0"` PARSES (so the fallback does not fire)
    /// and rounds down to 1, exactly mirroring the write path's `resolve_shard_count`
    /// (`round_down_pow2(0) == 1`); 1 is still a valid mask (&0), so it is safe.
    #[test]
    fn shard_count_of_rounds_non_pow2_meta_down() {
        pyo3::Python::initialize();

        let check = |shards: Option<&str>| -> usize {
            let dir = tempfile::tempdir().unwrap();
            let output = dir.path().join("fullmap.redb");
            let database = Database::create(&output).unwrap();
            let write = database.begin_write().unwrap();
            {
                let mut meta = write.open_table(META).unwrap();
                meta.insert("schema", SCHEMA_VERSION).unwrap();
                if let Some(value) = shards {
                    meta.insert("shards", value).unwrap();
                }
            }
            write.commit().unwrap();
            drop(database);

            let database = Database::open(&output).unwrap();
            let count = shard_count_of(&database).unwrap();
            drop(database);
            count
        };

        assert_eq!(check(Some("3")), 2);
        assert_eq!(check(Some("5")), 4);
        assert_eq!(check(Some("6")), 4);
        assert_eq!(check(Some("7")), 4);
        assert_eq!(check(Some("4")), 4);
        assert_eq!(check(Some("2")), 2);
        assert_eq!(check(Some("1")), 1);
        // Missing / unparseable -> default SHARD_COUNT_SHARDS (16) via unwrap_or.
        assert_eq!(check(None), SHARD_COUNT_SHARDS);
        assert_eq!(check(Some("not-a-number")), SHARD_COUNT_SHARDS);
        // "0" parses (fallback does NOT fire) and rounds down to 1, mirroring the
        // write path; 1 is a valid mask, so this is safe, not a misroute.
        assert_eq!(check(Some("0")), 1);
    }

    /// US-201 eliminated the single global merge producer: Phase 4 now runs one
    /// INDEPENDENT k-way merge per shard in parallel (one thread per shard, doing
    /// the merge and insert inline), so the merge parallelism is across SHARDS and
    /// is no longer a function of the synonym-phase `threads` knob.  The build must
    /// therefore stay serial-equivalent: a threads=1 build and a threads>=shard_count
    /// build (which differ in how the synonym phase partitions work and spill runs)
    /// must yield identical term->CURIE results AND identical per-shard record
    /// counts, because a term's shard is a pure function of its xxh64 hash.  This
    /// guards against any routing / spill-partition / batching regression in the
    /// parallel per-shard merges.
    #[test]
    fn parallel_writers_match_single_writer_build() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let synonyms = dir.path().join("many.ndjson");
        let mut synonym_file = File::create(&synonyms).unwrap();
        for i in 0..200 {
            writeln!(
                synonym_file,
                r#"{{"curie":"HGNC:{i}","preferred_name":"GENE{i}","names":["GENE{i}","alias{i}"],"types":["Gene"],"taxa":["NCBITaxon:9606"]}}"#
            )
            .unwrap();
        }
        drop(synonym_file);

        let out_serial = dir.path().join("serial.redb");
        let out_parallel = dir.path().join("parallel.redb");
        // local_spill=100 forces several sorted runs so each shard's k-way merge
        // has real work; threads=1 vs threads=4 vary only the synonym-phase worker
        // count (Phase 4 always runs one independent merge per shard regardless).
        build_test(
            out_serial.clone(),
            Vec::new(),
            vec![synonyms.clone()],
            1,
            100,
        )
        .unwrap();
        build_test(out_parallel.clone(), Vec::new(), vec![synonyms], 4, 100).unwrap();

        // Per-shard record counts are identical (routing is a pure function of the
        // term hash, independent of thread count) and every shard got >=1 record.
        let serial_counts = shard_record_counts(&out_serial);
        let parallel_counts = shard_record_counts(&out_parallel);
        assert_eq!(serial_counts, parallel_counts);
        assert_eq!(parallel_counts.len(), SHARD_COUNT_SHARDS);
        for count in &parallel_counts {
            assert!(*count > 0, "shard received no records: {parallel_counts:?}");
        }

        // term -> sorted CURIE set is identical across thread counts.
        let probes: Vec<String> = (0..200)
            .flat_map(|i| [format!("gene{i}"), format!("alias{i}")])
            .collect();
        let norm = |db: PathBuf| -> Vec<(String, Vec<String>)> {
            let rows = lookup_terms(db, probes.clone(), Some(4)).unwrap();
            let mut out: Vec<(String, Vec<String>)> = rows
                .into_iter()
                .map(|(t, recs)| {
                    let mut curies: Vec<String> = recs.into_iter().map(|r| r.curie).collect();
                    curies.sort();
                    (t, curies)
                })
                .collect();
            out.sort();
            out
        };
        let serial = norm(out_serial);
        let parallel = norm(out_parallel);
        assert_eq!(serial.len(), 400);
        assert_eq!(serial, parallel);
    }

    /// US-201 shards the term spill runs AT WRITE TIME (one `run_s{shard}_{id}.bin`
    /// per non-empty shard) so Phase 4 can run one INDEPENDENT k-way merge per
    /// shard in parallel, eliminating the single global producer that capped
    /// US-105.  This test pins both halves of that contract:
    ///
    /// 1. WRITE-TIME PARTITIONING: `process_synonyms` must return exactly
    ///    `shard_count` per-shard run lists, every file named `run_s{shard}_*.bin`
    ///    and present on disk, and EVERY term read back from a shard's runs must
    ///    hash to that shard (`term_shard(term, shard_count) == shard`).  A term
    ///    filed under the wrong shard would be merged/inserted in the wrong file
    ///    and silently vanish from lookups, so this is the core correctness
    ///    invariant of the partition.  Multiple runs must exist (small local_spill)
    ///    so the per-shard k-way merge has real work.
    /// 2. PARALLEL MERGE EQUIVALENCE: two full builds over the SAME synonyms but
    ///    with different shard counts (4 vs 2) — i.e. a legacy-compatible count
    ///    and a smaller runtime override, producing different per-shard run
    ///    layouts merged by independent threads — must yield identical
    ///    term -> set(CURIE) results.  Because each term's postings all land in one
    ///    shard and the read path routes by the same `term_shard` oracle, the
    ///    reconstructed CURIE set is shard-count-independent; any cross-shard
    ///    duplication or drop in the parallel merges would break this equality.
    #[test]
    fn per_shard_spill_runs_and_parallel_merges_match_reference() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let synonyms = dir.path().join("many.ndjson");
        let mut synonym_file = File::create(&synonyms).unwrap();
        for i in 0..200 {
            writeln!(
                synonym_file,
                r#"{{"curie":"HGNC:{i}","preferred_name":"GENE{i}","names":["GENE{i}","alias{i}"],"types":["Gene"],"taxa":["NCBITaxon:9606"]}}"#
            )
            .unwrap();
        }
        drop(synonym_file);

        // --- 1. write-time partitioning ------------------------------------
        let spill_dir = dir.path().join("spill");
        std::fs::create_dir_all(&spill_dir).unwrap();
        let equivalents = EquivIndex::build(&[], &spill_dir, None).unwrap();
        let shard_count = SHARD_COUNT_SHARDS;
        // local_spill=30 forces many spills so each shard has several runs and the
        // per-shard k-way merge does real grouping work.
        let result = process_synonyms(
            std::slice::from_ref(&synonyms),
            &equivalents,
            &spill_dir,
            30,
            1_000_000,
            &HashSet::new(),
            4,
            shard_count,
            DEFAULT_CHUNK_BYTES,
            2,
            None,
        )
        .unwrap();
        drop(equivalents);

        // Exactly one run list per shard.
        assert_eq!(result.run_paths.len(), shard_count);

        // Every shard produced runs, each file is named run_s{shard}_*.bin and
        // exists on disk, and every term read back routes to that shard.
        let mut total_runs = 0usize;
        for (shard, paths) in result.run_paths.iter().enumerate() {
            assert!(!paths.is_empty(), "shard {shard} produced no run files");
            let expected_prefix = format!("run_s{shard}_");
            for path in paths {
                let name = path.file_name().unwrap().to_string_lossy().into_owned();
                assert!(
                    name.starts_with(&expected_prefix) && name.ends_with(".bin"),
                    "unexpected run file name {name} in shard {shard}"
                );
                assert!(path.exists(), "run file missing on disk: {path:?}");
                let mut reader = RunReader::new(path).unwrap();
                while let Some((term, _pairs)) = reader.cur.take() {
                    assert_eq!(
                        term_shard(&term, shard_count),
                        shard,
                        "term {term} filed under shard {shard} but hashes elsewhere"
                    );
                    reader.advance().unwrap();
                }
                total_runs += 1;
            }
        }
        // Many runs across the shards prove the per-shard merges each k-way merge
        // more than one run (the whole point of the partition).
        assert!(
            total_runs > shard_count,
            "expected multiple per-shard runs, got {total_runs}"
        );

        // --- 2. parallel merge equivalence (shards=4 vs shards=2) ----------
        let out4 = dir.path().join("s4.redb");
        let out2 = dir.path().join("s2.redb");
        build_fullmap_inner(
            out4.clone(),
            Vec::new(),
            vec![synonyms.clone()],
            4,
            4,
            None,
            100,
            1_000_000,
            HashSet::new(),
            DEFAULT_CHUNK_BYTES,
            2,
            64 * 1024 * 1024,
            1000,
            dir.path().join("s4.spill.d"),
        )
        .unwrap();
        build_fullmap_inner(
            out2.clone(),
            Vec::new(),
            vec![synonyms],
            4,
            2,
            None,
            100,
            1_000_000,
            HashSet::new(),
            DEFAULT_CHUNK_BYTES,
            2,
            64 * 1024 * 1024,
            1000,
            dir.path().join("s2.spill.d"),
        )
        .unwrap();

        let probes: Vec<String> = (0..200)
            .flat_map(|i| [format!("gene{i}"), format!("alias{i}")])
            .collect();
        let norm = |db: PathBuf| -> Vec<(String, Vec<String>)> {
            let rows = lookup_terms(db, probes.clone(), Some(4)).unwrap();
            let mut out: Vec<(String, Vec<String>)> = rows
                .into_iter()
                .map(|(t, recs)| {
                    let mut curies: Vec<String> = recs.into_iter().map(|r| r.curie).collect();
                    curies.sort();
                    (t, curies)
                })
                .collect();
            out.sort();
            out
        };
        let r4 = norm(out4);
        let r2 = norm(out2);
        assert_eq!(r4.len(), 400);
        assert_eq!(r4, r2, "shard count must not change term -> CURIE results");
    }

    /// US-103 fans the read path out across RECORDS shards: `lookup_pair_terms_db`
    /// partitions query terms by `term_shard`, reads each NON-EMPTY shard on its
    /// own thread (capped at `workers`), and re-merges hits in input order.  This
    /// matters because a wrong routing or merge would silently drop, duplicate, or
    /// reorder rows.  The fixture's probe terms provably hash to >=2 different
    /// shards (asserted), misses are interleaved, and the parallel result
    /// (threads=4) must equal the single-threaded chunk path in BOTH content and
    /// order, with hits appearing in probe order and misses yielding no row.
    #[test]
    fn parallel_shard_fanout_merges_in_input_order() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let synonyms = dir.path().join("many.ndjson");
        let output = dir.path().join("fullmap.redb");
        let mut synonym_file = File::create(&synonyms).unwrap();
        for i in 0..120 {
            writeln!(
                synonym_file,
                r#"{{"curie":"HGNC:{i}","preferred_name":"GENE{i}","names":["GENE{i}"],"types":["Gene"],"taxa":["NCBITaxon:9606"]}}"#
            )
            .unwrap();
        }
        drop(synonym_file);
        build_test(output.clone(), Vec::new(), vec![synonyms], 4, 4_000_000).unwrap();

        // Probe terms in a fixed order, interleaving real terms with misses.
        let mut probes: Vec<String> = Vec::new();
        for i in 0..80 {
            probes.push(format!("gene{i}"));
            if i % 7 == 0 {
                probes.push(format!("missing{i}")); // no such term -> miss
            }
        }

        // The real probe terms must span at least 2 shards, otherwise this test
        // would not actually exercise the multi-shard fan-out.
        let spanned: HashSet<usize> = probes
            .iter()
            .filter(|t| !t.starts_with("missing"))
            .map(|t| term_shard(t, SHARD_COUNT_SHARDS))
            .collect();
        assert!(
            spanned.len() >= 2,
            "probe terms must span >=2 shards, got {spanned:?}"
        );

        let shards = open_cached_shards(&output).unwrap();
        assert_eq!(shards.len(), SHARD_COUNT_SHARDS);

        // Parallel fan-out (workers=4) vs the single-threaded chunk path (workers=1)
        // must agree on both content and order.
        let parallel = lookup_pair_terms_db(&shards, &probes, 4).unwrap();
        let serial = lookup_pair_terms_db(&shards, &probes, 1).unwrap();
        assert_eq!(parallel, serial, "parallel fan-out diverged from serial");

        // Hits appear in probe order; misses produced no row.
        let expected_order: Vec<String> = probes
            .iter()
            .filter(|t| !t.starts_with("missing"))
            .cloned()
            .collect();
        let got_order: Vec<String> = parallel.iter().map(|(t, _)| t.clone()).collect();
        assert_eq!(got_order, expected_order, "merge broke input order");

        // End-to-end hydration (through the primary) also agrees across thread
        // counts and yields one row group per hit term.
        let rows_par = lookup_terms(output.clone(), probes.clone(), Some(4)).unwrap();
        let rows_ser = lookup_terms(output, probes, Some(1)).unwrap();
        assert_eq!(rows_par.len(), expected_order.len());
        assert_eq!(rows_par, rows_ser);
    }

    /// The production build-graph resolve calls `lookup_fullmap_terms` with
    /// `threads=None`, so the parallel shard fan-out must kick in from the Rust
    /// DEFAULT alone — not just when a test passes `threads>=2`.  This builds a
    /// large fixture and probes it with a batch that crosses `LOOKUP_PARALLEL_MIN`
    /// and spans every default shard, then asserts: (a) the default worker count
    /// is >1 on any multi-core host, and reaches the 16-shard fan-out cap on a
    /// host with at least 16 CPUs (so `lookup_pair_terms_db` can spawn one reader
    /// per non-empty shard), and (b) `threads=None` returns results IDENTICAL
    /// (content + order) to the forced-serial `threads=Some(1)`, with misses
    /// dropped.  On smaller hosts the maximum-fanout assertion is skipped but
    /// equivalence still holds.
    #[test]
    fn threads_none_defaults_to_parallel_for_large_batch() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let synonyms = dir.path().join("large.ndjson");
        let output = dir.path().join("fullmap.redb");
        let mut synonym_file = File::create(&synonyms).unwrap();
        for i in 0..1100 {
            writeln!(
                synonym_file,
                r#"{{"curie":"HGNC:{i}","preferred_name":"GENE{i}","names":["GENE{i}"],"types":["Gene"],"taxa":["NCBITaxon:9606"]}}"#
            )
            .unwrap();
        }
        drop(synonym_file);
        build_test(output.clone(), Vec::new(), vec![synonyms], 4, 4_000_000).unwrap();

        // Large probe batch (>= LOOKUP_PARALLEL_MIN) with misses interleaved.
        let mut probes: Vec<String> = Vec::new();
        for i in 0..1100 {
            probes.push(format!("gene{i}"));
            if i % 50 == 0 {
                probes.push(format!("absent{i}")); // miss -> dropped
            }
        }
        assert!(
            probes.len() >= LOOKUP_PARALLEL_MIN,
            "batch must cross the parallel threshold, got {}",
            probes.len()
        );

        // Real probe terms provably span every default shard, so on a sufficiently
        // large machine the default worker policy can drive the full 16-shard
        // lookup fan-out.
        let spanned: HashSet<usize> = probes
            .iter()
            .filter(|t| !t.starts_with("absent"))
            .map(|t| term_shard(t, SHARD_COUNT_SHARDS))
            .collect();
        assert_eq!(
            spanned.len(),
            SHARD_COUNT_SHARDS,
            "probes must span every default shard, got {spanned:?}"
        );

        // The Rust default must select >1 worker for this large batch on any
        // multi-core host; after the `.max(1).min(len)` clamp in `lookup_pair_terms`
        // the effective worker count is still >1. On hosts with at least the
        // default shard count, it reaches that count so all non-empty shards may
        // be queried concurrently.
        let cpus = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1);
        let default_workers = default_lookup_workers(probes.len());
        if cpus > 1 {
            assert!(default_workers > 1, "large batch must default to parallel");
            let effective = default_workers.max(1).min(probes.len().max(1));
            assert!(
                effective > 1,
                "clamp must not collapse a large batch to serial"
            );
        }
        if cpus >= SHARD_COUNT_SHARDS {
            assert!(
                default_workers >= SHARD_COUNT_SHARDS,
                "large batch should be able to fan out across all default shards"
            );
        }
        // Explicit threads=Some(1) still forces serial regardless of batch size.
        assert_eq!(default_lookup_workers(0), 1, "empty batch stays serial");
        assert_eq!(
            default_lookup_workers(LOOKUP_PARALLEL_MIN - 1),
            1,
            "sub-threshold batch stays serial"
        );

        // threads=None (production default) == forced-serial Some(1): identical
        // content AND order, misses dropped.
        let via_default = lookup_pair_terms(output.clone(), probes.clone(), None).unwrap();
        let via_serial = lookup_pair_terms(output, probes.clone(), Some(1)).unwrap();
        assert_eq!(
            via_default, via_serial,
            "threads=None diverged from threads=Some(1)"
        );
        let expected_order: Vec<String> = probes
            .iter()
            .filter(|t| !t.starts_with("absent"))
            .cloned()
            .collect();
        let got_order: Vec<String> = via_default.iter().map(|(t, _)| t.clone()).collect();
        assert_eq!(got_order, expected_order, "merge broke input order");
    }

    /// US-103's contract is that `lookup_fullmap_terms` releases the GIL for the
    /// whole lookup: every pure-Rust data-fetch call (`lookup_terms` /
    /// `lookup_pair_terms`) must be wrapped in `py.detach`, with the `PyList`
    /// built only afterwards.  A behavioral GIL test is timing-dependent and
    /// flaky from a Rust `#[test]` (and Python test files are out of scope for
    /// this story), so this is a precise code-level guard: it isolates the
    /// function source and asserts each fetch call appears exactly once and only
    /// inside a `py.detach(move || ...)` wrapper.  If someone removes a detach or
    /// adds a bare fetch call, the read path would silently hold the GIL again
    /// (blocking rich's Live repaint thread) and this test fails loudly.
    #[test]
    fn lookup_fullmap_terms_wraps_every_fetch_in_detach() {
        let src = include_str!("fullmap.rs");
        let start = src
            .find("pub fn lookup_fullmap_terms")
            .expect("lookup_fullmap_terms present");
        let tail = &src[start..];
        let end = tail
            .find("\n#[pyfunction]")
            .expect("next pyfunction bounds fn");
        let body = &tail[..end];

        // Rows fetch: exactly one call, and it is the detached one.
        let rows_calls = body.matches("lookup_terms(db, terms, threads)").count();
        let rows_detached = body
            .matches("py.detach(move || lookup_terms(db, terms, threads)")
            .count();
        assert_eq!(rows_calls, 1, "rows fetch must be called exactly once");
        assert_eq!(
            rows_detached, 1,
            "rows fetch must be wrapped in py.detach (GIL released)"
        );

        // Pairs fetch: exactly one call, and it is the detached one.
        let pair_calls = body
            .matches("lookup_pair_terms(db, terms, threads)")
            .count();
        let pair_detached = body
            .matches("py.detach(move || lookup_pair_terms(db, terms, threads)")
            .count();
        assert_eq!(pair_calls, 1, "pairs fetch must be called exactly once");
        assert_eq!(
            pair_detached, 1,
            "pairs fetch must be wrapped in py.detach (GIL released)"
        );
    }

    /// `TABLASSERT_FULLMAP_SHARDS` makes the shard count runtime-configurable.  A
    /// 2-shard build must write META.shards="2", create exactly s0+s1 (each with a
    /// RECORDS table, even the one that receives no terms), create no higher shard
    /// files up to the compile-time cap, and the read path must open exactly 2
    /// shards (from META) and still resolve every term — proving writer and reader
    /// agree on the runtime shard mask.
    #[test]
    fn runtime_shard_count_builds_and_reads_fewer_shards() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let synonyms = dir.path().join("HGNC.ndjson");
        let output = dir.path().join("fullmap.redb");
        let mut synonym_file = File::create(&synonyms).unwrap();
        for i in 0..50 {
            writeln!(
                synonym_file,
                r#"{{"curie":"HGNC:{i}","preferred_name":"GENE{i}","names":["GENE{i}"],"types":["Gene"],"taxa":["NCBITaxon:9606"]}}"#
            )
            .unwrap();
        }
        drop(synonym_file);

        let spill_dir = {
            let mut p = output.clone().into_os_string();
            p.push(".spill.d");
            PathBuf::from(p)
        };
        // worker_count=4, shard_count=2.
        build_fullmap_inner(
            output.clone(),
            Vec::new(),
            vec![synonyms],
            4,
            2,
            None,
            4_000_000,
            1_000_000,
            HashSet::new(),
            DEFAULT_CHUNK_BYTES,
            2,
            64 * 1024 * 1024,
            1000,
            spill_dir,
        )
        .unwrap();

        // META.shards reflects the runtime count.
        let database = open_cached(output.clone()).unwrap();
        let read = database.begin_read().unwrap();
        let meta = read.open_table(META).unwrap();
        assert_eq!(meta.get("shards").unwrap().unwrap().value(), "2");
        drop(meta);
        drop(read);

        // Exactly s0+s1 exist, each holding a RECORDS table; higher shard files
        // up to the compile-time cap must NOT exist. (Direct opens are scoped so
        // their flocks drop before the cached opens.)
        for index in 0..2 {
            let shard = shard_path(&output, index);
            assert!(shard.exists(), "missing shard file {shard:?}");
            let db = Database::open(&shard).unwrap();
            let read = db.begin_read().unwrap();
            let _records = read.open_table(RECORDS).unwrap();
        }
        for index in 2..SHARD_COUNT_SHARDS {
            assert!(
                !shard_path(&output, index).exists(),
                "shard {index} must not exist for a 2-shard build"
            );
        }

        // Read path opens exactly 2 shards and resolves every term.
        let shards = open_cached_shards(&output).unwrap();
        assert_eq!(shards.len(), 2);
        let terms: Vec<String> = (0..50).map(|i| format!("gene{i}")).collect();
        let rows = lookup_terms(output, terms, Some(4)).unwrap();
        assert_eq!(rows.len(), 50);
    }

    /// Backward compatibility: databases built/advertising the former 4-shard
    /// layout must continue to open exactly those four shard files and route terms
    /// with a 4-shard mask, even though new default builds now write 16 shards.
    #[test]
    fn legacy_four_shard_db_reads_only_four_shards() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let synonyms = dir.path().join("HGNC.ndjson");
        let output = dir.path().join("fullmap.redb");
        let mut synonym_file = File::create(&synonyms).unwrap();
        for i in 0..80 {
            writeln!(
                synonym_file,
                r#"{{"curie":"HGNC:{i}","preferred_name":"GENE{i}","names":["GENE{i}"],"types":["Gene"],"taxa":["NCBITaxon:9606"]}}"#
            )
            .unwrap();
        }
        drop(synonym_file);

        let spill_dir = {
            let mut p = output.clone().into_os_string();
            p.push(".spill.d");
            PathBuf::from(p)
        };
        build_fullmap_inner(
            output.clone(),
            Vec::new(),
            vec![synonyms],
            4,
            4,
            None,
            4_000_000,
            1_000_000,
            HashSet::new(),
            DEFAULT_CHUNK_BYTES,
            2,
            64 * 1024 * 1024,
            1000,
            spill_dir,
        )
        .unwrap();

        let database = open_cached(output.clone()).unwrap();
        let read = database.begin_read().unwrap();
        let meta = read.open_table(META).unwrap();
        assert_eq!(meta.get("shards").unwrap().unwrap().value(), "4");
        drop(meta);
        drop(read);

        let shards = open_cached_shards(&output).unwrap();
        assert_eq!(shards.len(), 4);
        for index in 0..4 {
            assert!(
                shard_path(&output, index).exists(),
                "missing legacy shard {index}"
            );
        }
        for index in 4..SHARD_COUNT_SHARDS {
            assert!(
                !shard_path(&output, index).exists(),
                "new default shard {index} must not be required for legacy 4-shard DBs"
            );
        }

        let terms: Vec<String> = (0..80).map(|i| format!("gene{i}")).collect();
        let rows = lookup_terms(output, terms, Some(SHARD_COUNT_SHARDS)).unwrap();
        assert_eq!(rows.len(), 80);
    }

    #[test]
    fn lookup_rejects_v1_schema() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("fullmap.redb");
        let database = Database::create(&output).unwrap();
        let write = database.begin_write().unwrap();
        {
            let mut meta = write.open_table(META).unwrap();
            meta.insert("schema", SCHEMA_VERSION_V1).unwrap();
        }
        write.commit().unwrap();
        drop(database);

        let err = lookup_terms(output, vec!["brca1".to_string()], Some(1)).unwrap_err();
        assert!(err
            .to_string()
            .contains("fullmap DB is outdated; rebuild with 'tablassert build-fullmap'"));
    }

    /// v3 was the last single-file schema; the sharded read path must reject a
    /// v3 primary as outdated (rebuild hint), not as a generic unsupported
    /// schema, so migrating users get actionable guidance.
    #[test]
    fn lookup_rejects_v3_schema_as_outdated() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("fullmap.redb");
        let database = Database::create(&output).unwrap();
        let write = database.begin_write().unwrap();
        {
            let mut meta = write.open_table(META).unwrap();
            meta.insert("schema", SCHEMA_VERSION_V3).unwrap();
        }
        write.commit().unwrap();
        drop(database);

        let err = lookup_terms(output, vec!["brca1".to_string()], Some(1)).unwrap_err();
        assert!(err
            .to_string()
            .contains("fullmap DB is outdated; rebuild with 'tablassert build-fullmap'"));
    }

    /// `evict_cached_path` must drop the primary AND all default shard handles so
    /// a rebuild never serves a stale file.  Verified by caching the primary plus
    /// all shard handles, evicting, and confirming none of the canonical paths
    /// remain in the cache map.
    #[test]
    fn evict_cached_path_removes_primary_and_all_shards() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let synonyms = dir.path().join("HGNC.ndjson");
        let output = dir.path().join("fullmap.redb");
        let mut synonym_file = File::create(&synonyms).unwrap();
        writeln!(
            synonym_file,
            r#"{{"curie":"HGNC:1100","preferred_name":"BRCA1","names":["BRCA1"],"types":["Gene"],"taxa":["NCBITaxon:9606"]}}"#
        )
        .unwrap();
        drop(synonym_file);

        build_test(output.clone(), Vec::new(), vec![synonyms], 1, 4_000_000).unwrap();

        // Populate the cache with the primary + all default shards.
        let _primary = open_cached(output.clone()).unwrap();
        let shards = open_cached_shards(&output).unwrap();
        assert_eq!(shards.len(), SHARD_COUNT_SHARDS);

        let cache = DB_CACHE.get().unwrap();
        {
            let map = cache.read().unwrap();
            assert!(map.contains_key(&std::fs::canonicalize(&output).unwrap()));
            for index in 0..SHARD_COUNT_SHARDS {
                let key = std::fs::canonicalize(shard_path(&output, index)).unwrap();
                assert!(map.contains_key(&key), "shard {index} not cached");
            }
        }

        evict_cached_path(&output).unwrap();

        // After eviction none of the primary/shard paths remain cached.
        {
            let map = cache.read().unwrap();
            assert!(!map.contains_key(&std::fs::canonicalize(&output).unwrap()));
            for index in 0..SHARD_COUNT_SHARDS {
                let key = std::fs::canonicalize(shard_path(&output, index)).unwrap();
                assert!(!map.contains_key(&key), "shard {index} survived eviction");
            }
        }
    }

    #[test]
    fn build_fullmap_db_deduplicates_curie_rows() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let synonyms = dir.path().join("HGNC.ndjson");
        let output = dir.path().join("fullmap.redb");

        let mut synonym_file = File::create(&synonyms).unwrap();
        writeln!(
            synonym_file,
            r#"{{"curie":"HGNC:1100","preferred_name":"BRCA1","names":["BRCA1","breast cancer 1"],"types":["Gene"],"taxa":["NCBITaxon:9606"]}}"#
        )
        .unwrap();

        build_test(output.clone(), Vec::new(), vec![synonyms], 1, 4_000_000).unwrap();

        let database = open_cached(output).unwrap();
        let read = database.begin_read().unwrap();
        let curies = read.open_table(CURIES).unwrap();
        assert_eq!(curies.iter().unwrap().count(), 1);
    }

    #[test]
    fn curie_run_roundtrip_spans_buffer_boundary() {
        // 20k rows -> a run file larger than the 1 MB BufReader capacity, so
        // next_row must correctly handle records spanning a buffer refill.
        let dir = tempfile::tempdir().unwrap();
        let mut rows: Vec<(u32, CurieRow)> = Vec::new();
        for i in 0..20000u32 {
            rows.push((
                i,
                CurieRow {
                    prefix_id: (i % 100) as u16,
                    local_id: format!("ID{i:06}"),
                    preferred_name: format!(
                        "Preferred Name Number {i} with padding to grow the record size"
                    ),
                    category_id: (i % 50) as u16,
                    taxon_id: i as i32,
                },
            ));
        }
        let path = spill_curie_run(&mut rows, dir.path(), 0).unwrap();
        assert!(std::fs::metadata(&path).unwrap().len() > (1 << 20));
        let mut reader = CurieRunReader::new(&path).unwrap();
        let mut count = 0u32;
        while let Some((id, encoded)) = reader.next_row().unwrap() {
            let row: CurieRow = bincode::deserialize(&encoded).unwrap();
            assert_eq!(id, count);
            assert_eq!(row.local_id, format!("ID{count:06}"));
            count += 1;
        }
        assert_eq!(count, 20000);
    }

    #[test]
    fn curie_rows_spill_to_disk_and_stream_back() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let synonyms = dir.path().join("multi.ndjson");
        let output = dir.path().join("fullmap.redb");

        let mut synonym_file = File::create(&synonyms).unwrap();
        // Five distinct CURIEs; with curie_spill=2 this forces multiple curie runs.
        for i in 0..5 {
            writeln!(
                synonym_file,
                r#"{{"curie":"HGNC:{i}","preferred_name":"GENE{i}","names":["GENE{i}"],"types":["Gene"],"taxa":["NCBITaxon:9606"]}}"#
            )
            .unwrap();
        }

        let spill_dir = {
            let mut p = output.clone().into_os_string();
            p.push(".spill.d");
            PathBuf::from(p)
        };
        // curie_spill = 2 forces the curie-row buffer to spill to disk run files,
        // exercising spill_curie_run + the Phase-4 streaming reader.
        build_fullmap_inner(
            output.clone(),
            Vec::new(),
            vec![synonyms],
            1,
            SHARD_COUNT_SHARDS,
            None,
            4_000_000,
            2,
            HashSet::new(),
            DEFAULT_CHUNK_BYTES,
            2,
            64 * 1024 * 1024,
            1000,
            spill_dir,
        )
        .unwrap();

        let database = open_cached(output.clone()).unwrap();
        let read = database.begin_read().unwrap();
        let curies = read.open_table(CURIES).unwrap();
        assert_eq!(curies.iter().unwrap().count(), 5);
        drop(curies);
        drop(read);
        drop(database);

        // Every gene resolves through the streamed CURIES table.
        let rows = lookup_terms(
            output,
            (0..5).map(|i| format!("gene{i}")).collect(),
            Some(1),
        )
        .unwrap();
        let mut got: Vec<String> = rows
            .iter()
            .flat_map(|(_, recs)| recs.iter().map(|r| r.curie.clone()))
            .collect();
        got.sort();
        got.dedup();
        let want: Vec<String> = (0..5).map(|i| format!("HGNC:{i}")).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn dead_term_filter_drops_numeric_synonyms() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let synonyms = dir.path().join("HGNC.ndjson");
        let output = dir.path().join("fullmap.redb");

        let mut synonym_file = File::create(&synonyms).unwrap();
        // "12345" matches the distinct() bad-regex (^\d+$) and must be dropped;
        // "realname" survives.
        writeln!(
            synonym_file,
            r#"{{"curie":"HGNC:1100","preferred_name":"BRCA1","names":["12345","realname"],"types":["Gene"],"taxa":["NCBITaxon:9606"]}}"#
        )
        .unwrap();

        build_test(output.clone(), Vec::new(), vec![synonyms], 1, 4_000_000).unwrap();

        let alive = lookup_terms(output.clone(), vec!["realname".to_string()], Some(1)).unwrap();
        assert_eq!(alive.len(), 1);
        assert_eq!(alive[0].1[0].curie, "HGNC:1100");

        let dead = lookup_terms(output, vec!["12345".to_string()], Some(1)).unwrap();
        assert!(dead.is_empty() || dead[0].1.is_empty());
    }

    #[test]
    fn prefix_exclusion_drops_matching_rows() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let synonyms = dir.path().join("mixed.ndjson");
        let output = dir.path().join("fullmap.redb");

        let mut synonym_file = File::create(&synonyms).unwrap();
        writeln!(
            synonym_file,
            r#"{{"curie":"HGNC:1","preferred_name":"GENEA","names":["GENEA"],"types":["Gene"],"taxa":["NCBITaxon:9606"]}}"#
        )
        .unwrap();
        writeln!(
            synonym_file,
            r#"{{"curie":"CHEBI:2","preferred_name":"water","names":["water"],"types":["ChemicalEntity"],"taxa":[]}}"#
        )
        .unwrap();

        let spill_dir = {
            let mut p = output.clone().into_os_string();
            p.push(".spill.d");
            PathBuf::from(p)
        };
        let mut exclude = HashSet::new();
        exclude.insert("HGNC".to_string());
        build_fullmap_inner(
            output.clone(),
            Vec::new(),
            vec![synonyms],
            1,
            SHARD_COUNT_SHARDS,
            None,
            4_000_000,
            1_000_000,
            exclude,
            DEFAULT_CHUNK_BYTES,
            2,
            64 * 1024 * 1024,
            1000,
            spill_dir,
        )
        .unwrap();

        // Only CHEBI:2 survives; HGNC:1 is excluded entirely.
        let database = open_cached(output.clone()).unwrap();
        let read = database.begin_read().unwrap();
        let curies = read.open_table(CURIES).unwrap();
        assert_eq!(curies.iter().unwrap().count(), 1);
        drop(curies);
        drop(read);
        drop(database);

        let kept = lookup_terms(output.clone(), vec!["water".to_string()], Some(1)).unwrap();
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].1[0].curie, "CHEBI:2");

        let dropped = lookup_terms(output, vec!["genea".to_string()], Some(1)).unwrap();
        assert!(dropped.is_empty() || dropped[0].1.is_empty());
    }

    #[test]
    fn intra_file_parallelism_single_file_many_workers() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let synonyms = dir.path().join("big.ndjson");
        let output = dir.path().join("fullmap.redb");

        // One file, 5000 distinct CURIEs.
        let mut synonym_file = File::create(&synonyms).unwrap();
        for i in 0..5000 {
            writeln!(
                synonym_file,
                r#"{{"curie":"HGNC:{i}","preferred_name":"GENE{i}","names":["GENE{i}","alias{i}"],"types":["Gene"],"taxa":["NCBITaxon:9606"]}}"#
            )
            .unwrap();
        }

        let spill_dir = {
            let mut p = output.clone().into_os_string();
            p.push(".spill.d");
            PathBuf::from(p)
        };
        // 4 workers, chunk_bytes=8192 (forces many chunks), 2 producers: one
        // file is split across workers and chunks (intra-file parallelism).
        build_fullmap_inner(
            output.clone(),
            Vec::new(),
            vec![synonyms],
            4,
            SHARD_COUNT_SHARDS,
            None,
            4_000_000,
            1_000_000,
            HashSet::new(),
            8192,
            2,
            64 * 1024 * 1024,
            1000,
            spill_dir,
        )
        .unwrap();

        let database = open_cached(output.clone()).unwrap();
        let read = database.begin_read().unwrap();
        let curies = read.open_table(CURIES).unwrap();
        assert_eq!(curies.iter().unwrap().count(), 5000);
        drop(curies);
        drop(read);
        drop(database);

        // Sample lookups across the file (start / middle / end) all resolve.
        let terms: Vec<String> = [0, 1, 2499, 2500, 4998, 4999]
            .into_iter()
            .map(|i| format!("gene{i}"))
            .collect();
        let rows = lookup_terms(output, terms, Some(4)).unwrap();
        let mut got: Vec<String> = rows
            .iter()
            .flat_map(|(_, recs)| recs.iter().map(|r| r.curie.clone()))
            .collect();
        got.sort();
        got.dedup();
        assert_eq!(got.len(), 6);
    }

    #[test]
    fn build_fullmap_db_rejects_empty_synonym_list() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let err = build_fullmap_db(
                py,
                PathBuf::from("/tmp/should-not-exist.redb"),
                Vec::new(),
                Vec::new(),
                Some(1),
                None,
            )
            .expect_err("empty synonyms should fail");
            assert!(err
                .to_string()
                .contains("at least one synonym file is required"));
        });
    }

    #[test]
    fn builds_records_from_alias_fields() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let synonyms = dir.path().join("BABEL.ndjson");
        let output = dir.path().join("fullmap.redb");

        let mut synonym_file = File::create(&synonyms).unwrap();
        writeln!(
            synonym_file,
            r#"{{"id":"MONDO:1","name":"Alias disease","names":["Alias disease"],"categories":["biolink:Disease"],"taxon":["NCBITaxon:0"]}}"#
        )
        .unwrap();

        build_test(output.clone(), Vec::new(), vec![synonyms], 1, 4_000_000).unwrap();
        let rows = lookup_terms(output, vec!["alias disease".to_string()], Some(1)).unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].1[0].curie, "MONDO:1");
        assert_eq!(rows[0].1[0].preferred_name, "Alias disease");
        assert_eq!(rows[0].1[0].category_name, "Disease");
    }

    #[test]
    fn build_fullmap_db_does_not_index_banned_name_tokens() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let synonyms = dir.path().join("HGNC.ndjson");
        let output = dir.path().join("fullmap.redb");

        let mut synonym_file = File::create(&synonyms).unwrap();
        writeln!(
            synonym_file,
            r#"{{"curie":"HGNC:1","preferred_name":"GENE1","names":["hypothetical protein","GENE1"],"types":["Gene"],"taxa":["NCBITaxon:9606"]}}"#
        )
        .unwrap();

        build_test(output.clone(), Vec::new(), vec![synonyms], 1, 4_000_000).unwrap();
        let rows = lookup_terms(
            output,
            vec!["hypothetical protein".to_string(), "gene1".to_string()],
            Some(1),
        )
        .unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].0, "gene1");
    }

    #[test]
    fn build_fullmap_db_cleans_quoted_names_before_indexing() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let synonyms = dir.path().join("HGNC.ndjson");
        let output = dir.path().join("fullmap.redb");

        let mut synonym_file = File::create(&synonyms).unwrap();
        writeln!(
            synonym_file,
            r#"{{"curie":"HGNC:2","preferred_name":"'Quoted Gene'","names":["\"Quoted Gene\""],"types":["Gene"],"taxa":["NCBITaxon:9606"]}}"#
        )
        .unwrap();

        build_test(output.clone(), Vec::new(), vec![synonyms], 1, 4_000_000).unwrap();
        let rows = lookup_terms(output, vec!["quoted gene".to_string()], Some(1)).unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].1[0].preferred_name, "Quoted Gene");
    }

    /// Forcing a tiny spill threshold produces many sorted runs that the k-way
    /// merge must regroup; the result must match a single-run build exactly.
    #[test]
    fn spill_merge_matches_single_run_build() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let synonyms = dir.path().join("multi.ndjson");
        let mut synonym_file = File::create(&synonyms).unwrap();
        // Several curies sharing overlapping names so terms map to multiple pairs.
        writeln!(synonym_file, r#"{{"curie":"HGNC:1","preferred_name":"Alpha","names":["Alpha","shared"],"types":["Gene"],"taxa":["NCBITaxon:9606"]}}"#).unwrap();
        writeln!(synonym_file, r#"{{"curie":"HGNC:2","preferred_name":"Beta","names":["Beta","shared"],"types":["Gene"],"taxa":["NCBITaxon:9606"]}}"#).unwrap();
        writeln!(synonym_file, r#"{{"curie":"MONDO:1","preferred_name":"Gamma","names":["Gamma","shared"],"types":["Disease"],"taxa":["NCBITaxon:0"]}}"#).unwrap();
        writeln!(synonym_file, r#"{{"curie":"HGNC:3","preferred_name":"Delta","names":["Delta","alpha"],"types":["Gene"],"taxa":["NCBITaxon:10090"]}}"#).unwrap();
        drop(synonym_file);

        let out_big = dir.path().join("big.redb");
        let out_tiny = dir.path().join("tiny.redb");
        // local_spill=1 forces a spill after essentially every term => many runs.
        build_test(
            out_big.clone(),
            Vec::new(),
            vec![synonyms.clone()],
            2,
            4_000_000,
        )
        .unwrap();
        build_test(out_tiny.clone(), Vec::new(), vec![synonyms], 2, 1).unwrap();

        let probes = vec![
            "alpha".to_string(),
            "shared".to_string(),
            "gamma".to_string(),
            "beta".to_string(),
            "delta".to_string(),
        ];
        let big = lookup_terms(out_big, probes.clone(), Some(1)).unwrap();
        let tiny = lookup_terms(out_tiny, probes, Some(1)).unwrap();

        // Same terms resolved, same hydrated records (order-independent compare).
        let norm = |v: Vec<(String, Vec<FullmapRecord>)>| -> Vec<(String, Vec<String>)> {
            let mut out: Vec<(String, Vec<String>)> = v
                .into_iter()
                .map(|(t, recs)| {
                    let mut curies: Vec<String> = recs.into_iter().map(|r| r.curie).collect();
                    curies.sort();
                    (t, curies)
                })
                .collect();
            out.sort();
            out
        };
        // "shared" must resolve to all three curies that list it.
        let shared = tiny
            .iter()
            .find(|(t, _)| t == "shared")
            .map(|(_, recs)| {
                let mut c: Vec<String> = recs.iter().map(|r| r.curie.clone()).collect();
                c.sort();
                c
            })
            .unwrap_or_default();
        assert_eq!(shared, vec!["HGNC:1", "HGNC:2", "MONDO:1"]);
        assert_eq!(norm(big), norm(tiny));
    }

    /// Regression: repeated lookups in one process must not trip redb's flock
    /// (the old mtime-keyed cache re-opened the DB and failed).
    #[test]
    fn repeated_lookups_reuse_cached_database() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let synonyms = dir.path().join("HGNC.ndjson");
        let output = dir.path().join("fullmap.redb");
        let mut synonym_file = File::create(&synonyms).unwrap();
        writeln!(
            synonym_file,
            r#"{{"curie":"HGNC:1100","preferred_name":"BRCA1","names":["BRCA1"],"types":["Gene"],"taxa":["NCBITaxon:9606"]}}"#
        )
        .unwrap();
        drop(synonym_file);

        build_test(output.clone(), Vec::new(), vec![synonyms], 1, 4_000_000).unwrap();

        // Several consecutive lookups (rows path) must all succeed.
        for _ in 0..3 {
            let rows = lookup_terms(output.clone(), vec!["brca1".to_string()], Some(1)).unwrap();
            assert_eq!(rows.len(), 1);
        }
        // open_cached returns the same handle across calls.
        let a = open_cached(output.clone()).unwrap();
        let b = open_cached(output).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
    }

    /// `MergeHeap::next_group` must k-way merge several runs sorted by
    /// `(xxh64(term), term)` (the Tier-3a hash-prefixed ordering): for each term
    /// the pair lists from every run that carries it are concatenated then
    /// sorted+deduped (the pairs within a single run are deliberately
    /// unsorted/duplicated here to prove the merge, not the writer, establishes
    /// the final ordering).  Groups emerge in HASH order (not lexicographic), so
    /// the merged (term -> pairs) map is verified order-independently.  Returns
    /// None at a clean EOF.
    #[test]
    fn merge_heap_merges_overlapping_runs_sorted_and_deduped() {
        let dir = tempfile::tempdir().unwrap();
        let run0 = dir.path().join("r0.bin");
        let run1 = dir.path().join("r1.bin");
        let run2 = dir.path().join("r2.bin");

        // Write each run sorted by (xxh64(term), term) — the MergeHeap invariant.
        fn write_run(path: &Path, mut terms: Vec<(&str, Vec<(u32, u8)>)>) {
            terms.sort_by(|a, b| {
                xxh64(a.0.as_bytes(), 0)
                    .cmp(&xxh64(b.0.as_bytes(), 0))
                    .then_with(|| a.0.cmp(b.0))
            });
            let mut w = RunWriter::new(path).unwrap();
            for (term, pairs) in terms {
                w.write_term(term, &pairs).unwrap();
            }
            w.finish().unwrap();
        }

        write_run(
            &run0,
            vec![
                ("apple", vec![(3, 1), (1, 0), (2, 2)]),
                ("cherry", vec![(5, 0)]),
            ],
        );
        write_run(
            &run1,
            vec![
                ("apple", vec![(1, 0), (9, 3)]),
                ("banana", vec![(7, 2), (7, 2)]),
            ],
        );
        write_run(
            &run2,
            vec![("banana", vec![(4, 1)]), ("cherry", vec![(5, 0), (0, 0)])],
        );

        let paths = vec![run0, run1, run2];
        let mut merge = MergeHeap::new(&paths).unwrap();

        // Drain every group into a term -> pairs map (emergence order is hash
        // order, so compare order-independently).  Draining until None also proves
        // the clean-EOF behavior.
        let mut got: std::collections::BTreeMap<String, Vec<(u32, u8)>> =
            std::collections::BTreeMap::new();
        while let Some((term, pairs)) = merge.next_group().unwrap() {
            got.insert(term, pairs);
        }

        let mut expected: std::collections::BTreeMap<String, Vec<(u32, u8)>> =
            std::collections::BTreeMap::new();
        // "apple" appears in run0 + run1: merged, sorted, deduped.
        expected.insert("apple".to_string(), vec![(1, 0), (2, 2), (3, 1), (9, 3)]);
        // "banana" appears in run1 (dup pair) + run2.
        expected.insert("banana".to_string(), vec![(4, 1), (7, 2)]);
        // "cherry" appears in run0 + run2 (overlapping (5,0) pair).
        expected.insert("cherry".to_string(), vec![(0, 0), (5, 0)]);
        assert_eq!(got, expected);
    }

    /// `RunWriter`/`RunReader` must roundtrip term frames exactly: an empty pair
    /// list, a 100k-pair term, a unicode term, and enough padding terms to grow
    /// the file past the 1 MB BufReader capacity so frames span a buffer refill.
    /// The reader returns frames in file order and reports a clean EOF (cur=None).
    #[test]
    fn term_run_writer_reader_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("terms.bin");

        let big_pairs: Vec<(u32, u8)> = (0..100_000).map(|i| (i, (i % 256) as u8)).collect();
        let unicode = "héllo–wörld·αβγ";

        let mut w = RunWriter::new(&path).unwrap();
        w.write_term("empty", &[]).unwrap();
        w.write_term("big", &big_pairs).unwrap();
        w.write_term(unicode, &[(1, 2)]).unwrap();
        // Padding terms push the file well past the 1 MB BufReader capacity.
        for i in 0..20_000u32 {
            w.write_term(&format!("pad{i:08}"), &[(i, 0), (i, 1)])
                .unwrap();
        }
        w.finish().unwrap();
        assert!(std::fs::metadata(&path).unwrap().len() > (1 << 20));

        let mut reader = RunReader::new(&path).unwrap();

        let (term, pairs) = reader.cur.take().unwrap();
        assert_eq!(term, "empty");
        assert!(pairs.is_empty());
        reader.advance().unwrap();

        let (term, pairs) = reader.cur.take().unwrap();
        assert_eq!(term, "big");
        assert_eq!(pairs, big_pairs);
        reader.advance().unwrap();

        let (term, pairs) = reader.cur.take().unwrap();
        assert_eq!(term, unicode);
        assert_eq!(pairs, vec![(1, 2)]);
        reader.advance().unwrap();

        for i in 0..20_000u32 {
            let (term, pairs) = reader.cur.take().unwrap();
            assert_eq!(term, format!("pad{i:08}"));
            assert_eq!(pairs, vec![(i, 0), (i, 1)]);
            reader.advance().unwrap();
        }
        assert!(reader.cur.is_none(), "clean EOF expected");
    }

    /// `EquivIndex::build` must k-way merge MULTIPLE equiv spill runs (forced here
    /// via a tiny `TABLASSERT_FULLMAP_EQUIV_SPILL_ENTRIES`) so a curie that recurs
    /// across several class files/runs resolves to the union of its equivalents,
    /// sorted and deduped.  Two distinct curies must both resolve, and a miss must
    /// return None.
    #[test]
    fn equiv_index_multi_run_merge_and_lookup() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        // A tiny spill threshold forces one run per row, so X:1 (present in all
        // three files) is merged across several runs in Phase 1b.  A small
        // threshold only changes HOW MANY runs are written, never the merged
        // result, so concurrent builds in other tests stay correct.
        std::env::set_var("TABLASSERT_FULLMAP_EQUIV_SPILL_ENTRIES", "1");

        let write_classes = |name: &str, rows: &[&str]| -> PathBuf {
            let path = dir.path().join(name);
            let mut file = File::create(&path).unwrap();
            for row in rows {
                writeln!(file, "{row}").unwrap();
            }
            path
        };
        let a = write_classes(
            "a.ndjson",
            &[
                r#"{"id":"X:1","equivalent_identifiers":[{"identifier":"E:b"},{"identifier":"E:a"}]}"#,
                r#"{"id":"Y:2","equivalent_identifiers":[{"identifier":"E:z"}]}"#,
            ],
        );
        let b = write_classes(
            "b.ndjson",
            &[
                r#"{"id":"X:1","equivalent_identifiers":[{"identifier":"E:c"},{"identifier":"E:a"}]}"#,
            ],
        );
        let c = write_classes(
            "c.ndjson",
            &[
                r#"{"id":"X:1","equivalent_identifiers":[{"identifier":"E:b"}]}"#,
                r#"{"id":"Z:3","equivalent_identifiers":[{"identifier":"E:q"}]}"#,
            ],
        );

        let spill_dir = dir.path().join("spill");
        std::fs::create_dir_all(&spill_dir).unwrap();
        let index = EquivIndex::build(&[a, b, c], &spill_dir, None).unwrap();
        std::env::remove_var("TABLASSERT_FULLMAP_EQUIV_SPILL_ENTRIES");

        // X:1 appears in all three files: equivs merged across runs, sorted+deduped
        // (E:a duplicated across a+b, E:b duplicated across a+c).
        let x: Vec<String> = index
            .lookup("X:1")
            .expect("X:1 present")
            .map(String::from)
            .collect();
        assert_eq!(
            x,
            vec!["E:a".to_string(), "E:b".to_string(), "E:c".to_string()]
        );

        // Two distinct curies both resolve to their own equiv sets.
        let y: Vec<String> = index
            .lookup("Y:2")
            .expect("Y:2 present")
            .map(String::from)
            .collect();
        assert_eq!(y, vec!["E:z".to_string()]);
        let z: Vec<String> = index
            .lookup("Z:3")
            .expect("Z:3 present")
            .map(String::from)
            .collect();
        assert_eq!(z, vec!["E:q".to_string()]);

        // A curie that was never indexed is a miss.
        assert!(index.lookup("ABSENT:9").is_none());
    }

    /// `emit_term` inserts the level-one form and, when distinct, the level-two
    /// form; skips level-two when it equals level-one; drops dead terms and
    /// token_qc failures; and accumulates one pair per call under each key.
    #[test]
    fn emit_term_inserts_normalized_forms_and_accumulates() {
        // l1 and l2 both inserted when they differ ("BRCA-1" -> "brca-1" + "brca1").
        let mut terms: HashMap<String, Vec<(u32, u8)>> = HashMap::new();
        emit_term("BRCA-1", (1, 0), &mut terms);
        assert_eq!(
            terms.get("brca-1").map(Vec::as_slice),
            Some([(1, 0)].as_slice())
        );
        assert_eq!(
            terms.get("brca1").map(Vec::as_slice),
            Some([(1, 0)].as_slice())
        );
        assert_eq!(terms.len(), 2);

        // l2 skipped when it equals l1 ("TP53" -> "tp53" is already alnum).
        let mut terms: HashMap<String, Vec<(u32, u8)>> = HashMap::new();
        emit_term("TP53", (2, 1), &mut terms);
        assert_eq!(
            terms.get("tp53").map(Vec::as_slice),
            Some([(2, 1)].as_slice())
        );
        assert_eq!(terms.len(), 1, "l2 == l1 must not insert a second key");

        // Dead terms (banned token + all-digit) are dropped at level one.
        let mut terms: HashMap<String, Vec<(u32, u8)>> = HashMap::new();
        emit_term("none", (3, 0), &mut terms);
        emit_term("12345", (3, 0), &mut terms);
        assert!(terms.is_empty(), "dead terms must not be indexed");

        // A token_qc failure skips the term entirely.
        let mut terms: HashMap<String, Vec<(u32, u8)>> = HashMap::new();
        emit_term("hypothetical protein", (4, 0), &mut terms);
        assert!(terms.is_empty(), "qc-failing term must not be indexed");

        // Pair accumulation: the same term called twice keeps both pairs in order.
        let mut terms: HashMap<String, Vec<(u32, u8)>> = HashMap::new();
        emit_term("gene", (5, 0), &mut terms);
        emit_term("gene", (6, 1), &mut terms);
        assert_eq!(
            terms.get("gene").map(Vec::as_slice),
            Some([(5, 0), (6, 1)].as_slice())
        );
    }

    /// `is_dead_term` is an exact-match filter: every named banned token, the
    /// empty string, and all-digit strings are dead; ordinary terms (and terms
    /// that merely CONTAIN a banned token as a substring) are alive.
    #[test]
    fn is_dead_term_classifies_banned_empty_and_numeric() {
        for banned in [
            "none",
            "nan",
            "na",
            "null",
            "unknown",
            "not applicable",
            "p_value",
            "variable",
            "result",
            "exposure",
            "expression",
            "symbol",
        ] {
            assert!(is_dead_term(banned), "{banned} must be dead");
        }
        assert!(is_dead_term(""), "empty must be dead");
        assert!(is_dead_term("0"), "all-digit must be dead");
        assert!(is_dead_term("1234567890"), "all-digit must be dead");
        assert!(!is_dead_term("brca1"), "normal term must be alive");
        assert!(!is_dead_term("gene42"), "normal term must be alive");
        // Exact match, not a substring test.
        assert!(!is_dead_term("none-like"), "substring is not an exact ban");
    }

    /// `validate_schema` must reject a v2 primary as OUTDATED (rebuild hint,
    /// mirroring the v1/v3 tests) and a missing or garbage schema as a generic
    /// UNSUPPORTED schema, so migrating users get actionable guidance while a
    /// corrupt/foreign DB fails loudly.
    #[test]
    fn lookup_rejects_v2_and_unsupported_schemas() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();

        // v2 => outdated (rebuild hint).
        let v2 = dir.path().join("v2.redb");
        let database = Database::create(&v2).unwrap();
        let write = database.begin_write().unwrap();
        {
            let mut meta = write.open_table(META).unwrap();
            meta.insert("schema", SCHEMA_VERSION_V2).unwrap();
        }
        write.commit().unwrap();
        drop(database);
        let err = lookup_terms(v2, vec!["brca1".to_string()], Some(1)).unwrap_err();
        assert!(err
            .to_string()
            .contains("fullmap DB is outdated; rebuild with 'tablassert build-fullmap'"));

        // Missing schema key => unsupported.
        let missing = dir.path().join("missing.redb");
        let database = Database::create(&missing).unwrap();
        let write = database.begin_write().unwrap();
        {
            // Create the META table but write no "schema" entry.
            let meta = write.open_table(META).unwrap();
            drop(meta);
        }
        write.commit().unwrap();
        drop(database);
        let err = lookup_terms(missing, vec!["brca1".to_string()], Some(1)).unwrap_err();
        assert!(err.to_string().contains("unsupported fullmap redb schema"));

        // Garbage schema value => unsupported.
        let garbage = dir.path().join("garbage.redb");
        let database = Database::create(&garbage).unwrap();
        let write = database.begin_write().unwrap();
        {
            let mut meta = write.open_table(META).unwrap();
            meta.insert("schema", "tablassert.fullmap.v999").unwrap();
        }
        write.commit().unwrap();
        drop(database);
        let err = lookup_terms(garbage, vec!["brca1".to_string()], Some(1)).unwrap_err();
        assert!(err.to_string().contains("unsupported fullmap redb schema"));
    }
}
