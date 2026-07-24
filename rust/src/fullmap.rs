use flate2::read::GzDecoder;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use rayon::prelude::*;
use redb::{Database, Durability, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
use std::thread;
use xxhash_rust::xxh3::xxh3_128;
use xxhash_rust::xxh64::xxh64;

const RECORDS: TableDefinition<u64, &[u8]> = TableDefinition::new("records");
const PREFIXES: TableDefinition<u16, &str> = TableDefinition::new("prefixes");
const CATEGORIES: TableDefinition<u16, &str> = TableDefinition::new("categories");
const SOURCES: TableDefinition<u8, &[u8]> = TableDefinition::new("sources");
const CURIES: TableDefinition<u32, &[u8]> = TableDefinition::new("curies");
const META: TableDefinition<&str, &str> = TableDefinition::new("meta");
const SCHEMA_VERSION: &str = "tablassert.fullmap.v3";
const SCHEMA_VERSION_V2: &str = "tablassert.fullmap.v2";
const SCHEMA_VERSION_V1: &str = "tablassert.fullmap.v1";
const FULLMAP_SOURCE_VERSION: &str = "2026sep1";
/// A normalized term grouped with its deduplicated `(curie_id, source_id)` pairs.
type TermPairs = (String, Vec<(u32, u8)>);
type PairRecords = Vec<TermPairs>;
/// One k-way-merge heap entry: `(term, run index, pairs)`, min-ordered by term.
type MergeItem = (Reverse<String>, usize, Vec<(u32, u8)>);

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
const DEFAULT_INSERT_BATCH: usize = 500_000;
const DEFAULT_CURIE_SPILL_ENTRIES: usize = 250_000;

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

fn clean(value: &str) -> String {
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
            return next.to_string();
        }
        s = next;
    }
}

fn token_qc(value: &str) -> bool {
    let lower = value.to_lowercase();
    !value.is_empty()
        && !value.contains('\t')
        && !value.contains('\n')
        && !value.contains('\r')
        && !lower.contains("inchikey")
        && !lower.contains("uncharacterized")
        && !lower.contains("hypothetical")
}

fn level_one(value: &str) -> String {
    value.to_lowercase()
}

fn level_two(value: &str) -> String {
    value
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect()
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

fn for_json_lines(path: &Path, mut visit: impl FnMut(Value) -> PyResult<()>) -> PyResult<()> {
    let reader = BufReader::new(open_reader(path)?);
    for line in reader.lines() {
        let raw = line.map_err(py_err)?;
        if raw.trim().is_empty() {
            continue;
        }
        visit(serde_json::from_str(&raw).map_err(py_err)?)?;
    }
    Ok(())
}

fn string_field(value: &Value, keys: &[&str]) -> Option<String> {
    for key in keys {
        if let Some(s) = value.get(*key).and_then(Value::as_str) {
            return Some(s.to_string());
        }
    }
    None
}

fn string_array(value: &Value, key: &str) -> Vec<String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(String::from)
                .collect()
        })
        .unwrap_or_default()
}

fn first_category(value: &Value) -> String {
    string_array(value, "types")
        .into_iter()
        .next()
        .or_else(|| string_array(value, "categories").into_iter().next())
        .unwrap_or_else(|| "NamedThing".to_string())
        .trim_start_matches("biolink:")
        .to_string()
}

fn first_taxon(value: &Value) -> i32 {
    let taxon = string_array(value, "taxa")
        .into_iter()
        .next()
        .or_else(|| string_array(value, "taxon").into_iter().next())
        .unwrap_or_default();
    taxon
        .trim_start_matches("NCBITaxon:")
        .parse::<i32>()
        .unwrap_or(0)
}

fn split_curie(curie: &str) -> Option<(&str, &str)> {
    curie
        .split_once(':')
        .filter(|(prefix, local)| !prefix.is_empty() && !local.is_empty())
}

fn equivalent_id(value: &Value) -> Option<String> {
    if let Some(s) = value.as_str() {
        return Some(s.to_string());
    }
    string_field(value, &["identifier", "id", "curie"])
}

fn class_id_and_equivalents(row: &Value) -> Option<(String, Vec<String>)> {
    let equivalents = row.get("equivalent_identifiers").and_then(Value::as_array);
    let mut ids = Vec::new();
    let id = string_field(row, &["id", "curie"]).or_else(|| {
        equivalents
            .and_then(|items| items.first())
            .and_then(equivalent_id)
    })?;
    // Primary id is NOT included in the value Vec — process_synonyms already
    // adds the curie itself as a term, saving ~38 GB of redundant storage.
    if let Some(equivalents) = equivalents {
        for equivalent in equivalents {
            if let Some(eid) = equivalent_id(equivalent) {
                if !ids.contains(&eid) {
                    ids.push(eid);
                }
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

fn shard_index(key: &str) -> usize {
    (xxh64(key.as_bytes(), 0) as usize) & SHARD_MASK
}

/// A sharded concurrent HashMap that routes keys by xxhash to reduce lock
/// contention.  Each shard is independently locked.
struct ShardedMap<V> {
    shards: Vec<RwLock<HashMap<String, V>>>,
}

impl<V> ShardedMap<V> {
    fn new() -> Self {
        let mut shards = Vec::with_capacity(SHARD_COUNT);
        for _ in 0..SHARD_COUNT {
            shards.push(RwLock::new(HashMap::new()));
        }
        ShardedMap { shards }
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
    shards: Vec<RwLock<HashMap<u128, u32>>>,
}

impl CurieIdMap {
    fn new() -> Self {
        let mut shards = Vec::with_capacity(SHARD_COUNT);
        for _ in 0..SHARD_COUNT {
            shards.push(RwLock::new(HashMap::new()));
        }
        CurieIdMap { shards }
    }

    /// Get the id for `hash`, or insert a new one computed by `f`.
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
        Ok(RunWriter {
            w: BufWriter::with_capacity(1 << 20, File::create(path)?),
        })
    }

    fn write_term(&mut self, term: &str, pairs: &[(u32, u8)]) -> std::io::Result<()> {
        let tb = term.as_bytes();
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
        Ok(RunReader { reader, cur })
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

/// Drain a thread-local term buffer into a sorted run file on disk.
fn spill_run(
    local: &mut HashMap<String, Vec<(u32, u8)>>,
    spill_dir: &Path,
    run_id: usize,
) -> PyResult<PathBuf> {
    let mut entries: Vec<TermPairs> = local.drain().collect();
    entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    let path = spill_dir.join(format!("run_{:08}.bin", run_id));
    let mut w = RunWriter::new(&path).map_err(py_err)?;
    for (term, mut pairs) in entries {
        pairs.sort_unstable();
        pairs.dedup();
        w.write_term(&term, &pairs).map_err(py_err)?;
    }
    w.finish().map_err(py_err)?;
    Ok(path)
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
        Ok(CurieRunWriter {
            w: BufWriter::with_capacity(1 << 20, File::create(path)?),
        })
    }

    fn write_row(&mut self, curie_id: u32, encoded: &[u8]) -> std::io::Result<()> {
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
        Ok(CurieRunReader {
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
    let path = spill_dir.join(format!("curie_run_{:08}.bin", run_id));
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
                heap.push((Reverse(term), idx, pairs));
            }
            readers.push(rr);
        }
        Ok(MergeHeap { readers, heap })
    }

    /// Return the next term with its merged, sorted, de-duplicated pairs.
    fn next_group(&mut self) -> std::io::Result<Option<TermPairs>> {
        let Some((Reverse(term), idx, pairs)) = self.heap.pop() else {
            return Ok(None);
        };
        let mut merged = pairs;
        let mut to_advance = vec![idx];
        while let Some((Reverse(t), _, _)) = self.heap.peek() {
            if *t != term {
                break;
            }
            let (_, i, p) = self.heap.pop().unwrap();
            merged.extend(p);
            to_advance.push(i);
        }
        merged.sort_unstable();
        merged.dedup();
        for i in to_advance {
            self.readers[i].advance()?;
            if let Some((t2, p2)) = self.readers[i].cur.take() {
                self.heap.push((Reverse(t2), i, p2));
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
/// One k-way-merge heap entry for equivalents: (hash, key, run_idx, equivs).
type EquivMergeItem = (Reverse<u64>, Reverse<String>, usize, Vec<String>);

/// Write an equiv entry to a buffered writer (run-file format with hash).
fn write_equiv_entry(
    w: &mut impl Write,
    hash: u64,
    key: &str,
    equivs: &[String],
) -> std::io::Result<()> {
    w.write_all(&hash.to_le_bytes())?;
    let kb = key.as_bytes();
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
    let rp = equiv_dir.join(format!("run_{:08}.bin", run_id));
    let mut w = BufWriter::with_capacity(1 << 20, File::create(&rp).map_err(py_err)?);
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
            return Ok(EquivIndex {
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
                for_json_lines(path, |row| {
                    if let Some((id, equivs)) = class_id_and_equivalents(&row) {
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

        // Phase 1b: k-way merge runs → write data file + build index.
        let data_path = spill_dir.join("equiv_data.bin");
        let mut dw = BufWriter::with_capacity(1 << 20, File::create(&data_path).map_err(py_err)?);
        let mut hashes: Vec<u64> = Vec::new();
        let mut offsets: Vec<u64> = Vec::new();
        let mut offset: u64 = 0;

        let mut readers: Vec<BufReader<File>> = run_paths
            .iter()
            .map(|p| File::open(p).map(BufReader::new))
            .collect::<std::io::Result<Vec<_>>>()
            .map_err(py_err)?;

        // Min-heap by (hash, key).
        let mut heap: BinaryHeap<EquivMergeItem> = BinaryHeap::new();
        for (idx, reader) in readers.iter_mut().enumerate() {
            if let Some((h, k, e)) = read_equiv_entry(reader).map_err(py_err)? {
                heap.push((Reverse(h), Reverse(k), idx, e));
            }
        }

        while let Some((Reverse(hash), Reverse(key), idx, mut equivs)) = heap.pop() {
            let mut to_advance = vec![idx];
            while let Some((Reverse(h), Reverse(k), _, _)) = heap.peek() {
                if *h != hash || *k != key {
                    break;
                }
                let (_, _, i, e) = heap.pop().unwrap();
                equivs.extend(e);
                to_advance.push(i);
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

            for i in to_advance {
                if let Some((h, k, e)) = read_equiv_entry(&mut readers[i]).map_err(py_err)? {
                    heap.push((Reverse(h), Reverse(k), i, e));
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

        Ok(EquivIndex {
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

/// Process a single term through clean → token_qc → level_one → level_two
/// and insert the resulting normalized forms into `local_terms`.
fn emit_term(term: &str, pair: (u32, u8), local_terms: &mut HashMap<String, Vec<(u32, u8)>>) {
    let cleaned = clean(term);
    if !token_qc(&cleaned) {
        return;
    }
    let l1 = level_one(&cleaned);
    if token_qc(&l1) && !is_dead_term(&l1) {
        local_terms.entry(l1.clone()).or_default().push(pair);
        let l2 = level_two(&l1);
        if l2 != l1 && token_qc(&l2) && !is_dead_term(&l2) {
            local_terms.entry(l2).or_default().push(pair);
        }
    }
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
    prefix_ids: HashMap<String, u16>,
    /// category string -> u16 id
    category_ids: HashMap<String, u16>,
    /// source string -> u8 id
    source_ids: HashMap<String, u8>,
    /// curie-row spill-run files holding (curie_id, encoded CurieRow)
    curie_run_paths: Vec<PathBuf>,
    /// sorted spill-run files holding term -> pairs
    run_paths: Vec<PathBuf>,
}

fn process_synonyms(
    synonyms: &[PathBuf],
    equivalents: &EquivIndex,
    spill_dir: &Path,
    local_spill: usize,
    curie_spill: usize,
    exclude_prefixes: &HashSet<String>,
    progress: Option<&Arc<Progress>>,
) -> PyResult<SynonymBuildResult> {
    // Pre-compute source IDs from filenames (small, deterministic).
    let mut source_ids: HashMap<String, u8> = HashMap::new();
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

    // Spill-run bookkeeping.
    let run_paths: RwLock<Vec<PathBuf>> = RwLock::new(Vec::new());
    let run_counter = AtomicUsize::new(0);
    let files_done = AtomicUsize::new(0);
    let total_files = synonyms.len();

    let results: Vec<PyResult<()>> = synonyms
        .par_iter()
        .map(|path| {
            let src_name = source_name(path);
            let source_id = *source_ids
                .get(&src_name)
                .ok_or_else(|| PyRuntimeError::new_err(format!("uninterned source {src_name}")))?;

            let mut local_curie_rows: Vec<(u32, CurieRow)> = Vec::new();
            let mut local_terms: HashMap<String, Vec<(u32, u8)>> = HashMap::new();
            let mut row_count: u64 = 0;
            let mut spill_count: u32 = 0;

            for_json_lines(path, |row| {
                let Some(curie) = string_field(&row, &["curie", "id"]) else {
                    return Ok(());
                };
                let Some((prefix, local_id)) = split_curie(&curie) else {
                    return Ok(());
                };
                // Skip CURIEs whose prefix the caller excludes (opt-in via
                // TABLASSERT_FULLMAP_EXCLUDE_PREFIXES) — they are filtered out
                // of downstream assertions anyway, so indexing them is wasted.
                if exclude_prefixes.contains(prefix) {
                    return Ok(());
                }
                row_count += 1;

                let prefix_id = prefix_map.get_or_insert_with(prefix, || {
                    u16::try_from(prefix_counter.fetch_add(1, Ordering::Relaxed))
                        .expect("too many fullmap prefixes")
                });

                let category_name = first_category(&row);
                let category_id = category_map.get_or_insert_with(&category_name, || {
                    u16::try_from(category_counter.fetch_add(1, Ordering::Relaxed))
                        .expect("too many fullmap categories")
                });

                let preferred_name = string_field(&row, &["preferred_name", "name"])
                    .unwrap_or_else(|| curie.clone());
                let taxon_id = first_taxon(&row);

                let mut is_new = false;
                let curie_hash = xxh3_128(curie.as_bytes());
                let curie_id = curie_map.get_or_insert_with(curie_hash, || {
                    is_new = true;
                    curie_counter.fetch_add(1, Ordering::Relaxed)
                });

                if is_new {
                    local_curie_rows.push((
                        curie_id,
                        CurieRow {
                            prefix_id,
                            local_id: local_id.to_string(),
                            preferred_name: clean(&preferred_name),
                            category_id,
                            taxon_id,
                        },
                    ));
                }

                let pair = (curie_id, source_id);

                // Inline term processing — avoids building a transient all_terms Vec.
                for name in string_array(&row, "names") {
                    emit_term(&name, pair, &mut local_terms);
                }
                emit_term(&curie, pair, &mut local_terms);
                if let Some(iter) = equivalents.lookup(&curie) {
                    for equiv in iter {
                        emit_term(equiv, pair, &mut local_terms);
                    }
                }

                // Spill the thread-local term buffer when it exceeds the budget.
                if local_terms.len() >= local_spill {
                    let run_id = run_counter.fetch_add(1, Ordering::Relaxed);
                    let p = spill_run(&mut local_terms, spill_dir, run_id)?;
                    run_paths.write().unwrap().push(p);
                    spill_count += 1;
                }

                // Spill curie rows to a disk run once the buffer exceeds the
                // budget, bounding per-thread memory (large files like
                // protein.txt.gz can hold 200 M+ unique CURIEs in one thread).
                if local_curie_rows.len() >= curie_spill {
                    let run_id = curie_run_counter.fetch_add(1, Ordering::Relaxed);
                    let p = spill_curie_run(&mut local_curie_rows, spill_dir, run_id)?;
                    curie_run_paths.write().unwrap().push(p);
                }

                Ok(())
            })?;

            // Flush the remainder of this file as a final run.
            if !local_terms.is_empty() {
                let run_id = run_counter.fetch_add(1, Ordering::Relaxed);
                let p = spill_run(&mut local_terms, spill_dir, run_id)?;
                run_paths.write().unwrap().push(p);
                spill_count += 1;
            }

            if !local_curie_rows.is_empty() {
                let run_id = curie_run_counter.fetch_add(1, Ordering::Relaxed);
                let p = spill_curie_run(&mut local_curie_rows, spill_dir, run_id)?;
                curie_run_paths.write().unwrap().push(p);
            }

            if let Some(p) = progress {
                let n = files_done.fetch_add(1, Ordering::Relaxed) + 1;
                p.call(
                    1,
                    n as u64,
                    total_files as u64,
                    &format!("{src_name} · {row_count} rows · {spill_count} spills"),
                );
            }

            Ok(())
        })
        .collect();

    for result in results {
        result?;
    }

    // Build final prefix_ids / category_ids maps from the sharded counters.
    let mut prefix_ids: HashMap<String, u16> = HashMap::new();
    for shard in &prefix_map.shards {
        prefix_ids.extend(shard.read().unwrap().iter().map(|(k, v)| (k.clone(), *v)));
    }
    let mut category_ids: HashMap<String, u16> = HashMap::new();
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
    prefix_ids: &HashMap<String, u16>,
    category_ids: &HashMap<String, u16>,
    source_ids: &HashMap<String, u8>,
    curie_run_paths: &[PathBuf],
    run_paths: &[PathBuf],
    cache_bytes: usize,
    insert_batch: usize,
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
    }
    write.commit().map_err(py_err)?;

    // K-way merge the sorted runs and stream into RECORDS.  All records go in
    // ONE Durability::None transaction (redb bounds it by the write cache and
    // spills dirty pages to disk), which keeps the file compact; a final durable
    // commit persists everything.  Inserts are hash-sorted in bounded batches for
    // near-sequential B-tree appends.
    let mut merge = MergeHeap::new(run_paths).map_err(py_err)?;
    let mut write = database.begin_write().map_err(py_err)?;
    write.set_durability(Durability::None);
    let mut table = write.open_table(RECORDS).map_err(py_err)?;

    let mut batch: Vec<(u64, Vec<u8>)> = Vec::with_capacity(insert_batch);
    let mut written: u64 = 0;

    loop {
        let Some((term, pairs)) = merge.next_group().map_err(py_err)? else {
            break;
        };
        let encoded = bincode::serialize(&(term.as_str(), &pairs)).map_err(py_err)?;
        batch.push((xxh64(term.as_bytes(), 0), encoded));

        if batch.len() >= insert_batch {
            batch.sort_unstable_by_key(|(hash, _)| *hash);
            for (hash, enc) in &batch {
                table.insert(*hash, enc.as_slice()).map_err(py_err)?;
            }
            written += batch.len() as u64;
            batch.clear();
            if let Some(p) = progress {
                p.call(2, written, 0, &format!("writing {written} records"));
            }
        }
    }

    if !batch.is_empty() {
        batch.sort_unstable_by_key(|(hash, _)| *hash);
        for (hash, enc) in &batch {
            table.insert(*hash, enc.as_slice()).map_err(py_err)?;
        }
        written += batch.len() as u64;
        batch.clear();
    }
    drop(table);
    write.commit().map_err(py_err)?;

    // Final durable commit to persist all pages.
    let write = database.begin_write().map_err(py_err)?;
    write.commit().map_err(py_err)?;

    if let Some(p) = progress {
        p.call(2, written, written, &format!("wrote {written} records"));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Build orchestrator
// ---------------------------------------------------------------------------

fn evict_cached_path(path: &Path) -> PyResult<()> {
    let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let Some(cache) = DB_CACHE.get() else {
        return Ok(());
    };
    cache.write().map_err(py_err)?.remove(&canonical);
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
    progress: Option<Arc<Progress>>,
    local_spill: usize,
    curie_spill: usize,
    exclude_prefixes: HashSet<String>,
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

    let worker_count = threads
        .unwrap_or_else(|| {
            let cpus = std::thread::available_parallelism()
                .map(|n| n.get())
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
    let cache_bytes = env_usize(
        "TABLASSERT_FULLMAP_REDB_CACHE_BYTES",
        DEFAULT_REDB_CACHE_BYTES,
    );
    let insert_batch = env_usize("TABLASSERT_FULLMAP_INSERT_BATCH", DEFAULT_INSERT_BATCH);
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
            progress,
            local_spill,
            curie_spill,
            exclude_prefixes,
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
        Some(SCHEMA_VERSION_V2) | Some(SCHEMA_VERSION_V1) => Err(PyRuntimeError::new_err(
            "fullmap DB is outdated; rebuild with 'tablassert build-fullmap'",
        )),
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

fn lookup_pair_chunk(database: &Database, terms: &[String]) -> PyResult<PairRecords> {
    let read = database.begin_read().map_err(py_err)?;
    let table = read.open_table(RECORDS).map_err(py_err)?;
    let mut out = Vec::new();
    for term in terms {
        let hash = xxh64(term.as_bytes(), 0);
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

fn lookup_pair_terms_db(
    database: Arc<Database>,
    terms: &[String],
    workers: usize,
) -> PyResult<PairRecords> {
    if workers <= 1 || terms.len() <= 1 {
        return lookup_pair_chunk(&database, terms);
    }

    let chunk_size = terms.len().div_ceil(workers);
    let mut handles = Vec::new();
    for chunk in terms.chunks(chunk_size) {
        let database = Arc::clone(&database);
        let chunk_terms = chunk.to_vec();
        handles.push(thread::spawn(move || {
            lookup_pair_chunk(&database, &chunk_terms)
        }));
    }

    let mut out = Vec::new();
    for handle in handles {
        let mut chunk = handle
            .join()
            .map_err(|_| PyRuntimeError::new_err("fullmap lookup thread panicked"))??;
        out.append(&mut chunk);
    }
    Ok(out)
}

fn lookup_pair_terms(
    db: PathBuf,
    terms: Vec<String>,
    threads: Option<usize>,
) -> PyResult<PairRecords> {
    let workers = threads.unwrap_or(1).max(1).min(terms.len().max(1));
    let database = open_cached(db)?;
    lookup_pair_terms_db(database, &terms, workers)
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
    // Open the database ONCE and reuse the handle for the pair lookup (a second
    // open would fail on redb's exclusive flock).
    let database = open_cached(db)?;
    let prefix_map = load_string_table(&database, PREFIXES)?;
    let category_map = load_string_table(&database, CATEGORIES)?;
    let source_map = load_sources(&database)?;
    let workers = threads.unwrap_or(1).max(1).min(terms.len().max(1));
    let pair_rows = lookup_pair_terms_db(Arc::clone(&database), &terms, workers)?;
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
        let list = PyList::empty(py);
        for (term, pairs) in lookup_pair_terms(db, terms, threads)? {
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
    let list = PyList::empty(py);
    for (term, records) in lookup_terms(db, terms, threads)? {
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
            None,
            local_spill,
            1_000_000,
            HashSet::new(),
            64 * 1024 * 1024,
            1000,
            spill_dir,
        )
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

    #[test]
    fn build_fullmap_db_writes_schema_v2_tables() {
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

        let database = open_cached(output).unwrap();
        let read = database.begin_read().unwrap();
        let meta = read.open_table(META).unwrap();
        assert_eq!(meta.get("schema").unwrap().unwrap().value(), SCHEMA_VERSION);
        drop(meta);
        let _prefixes = read.open_table(PREFIXES).unwrap();
        let _categories = read.open_table(CATEGORIES).unwrap();
        let _sources = read.open_table(SOURCES).unwrap();
        let _curies = read.open_table(CURIES).unwrap();
        let _records = read.open_table(RECORDS).unwrap();
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
            None,
            4_000_000,
            2,
            HashSet::new(),
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
            None,
            4_000_000,
            1_000_000,
            exclude,
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
}
