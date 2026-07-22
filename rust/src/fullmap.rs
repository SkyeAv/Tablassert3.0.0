use flate2::read::GzDecoder;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;
use redb::{Database, Durability, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
use std::thread;
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
type CachedDatabaseKey = (PathBuf, std::time::SystemTime);
type PairRecords = Vec<(String, Vec<(u32, u8)>)>;
static DB_CACHE: OnceLock<RwLock<HashMap<CachedDatabaseKey, Arc<Database>>>> = OnceLock::new();

/// Number of shards for concurrent maps (power of two for mask routing).
const SHARD_COUNT: usize = 64;
const SHARD_MASK: usize = SHARD_COUNT - 1;

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

fn clean(mut value: String) -> String {
    loop {
        let trimmed = value.trim().to_string();
        let bytes = trimmed.as_bytes();
        let matching_quotes = bytes.len() >= 2
            && ((bytes[0] == b'\'' && bytes[bytes.len() - 1] == b'\'')
                || (bytes[0] == b'"' && bytes[bytes.len() - 1] == b'"'));
        let duplicate_start = bytes.len() >= 2
            && ((bytes[0] == b'\'' && bytes[1] == b'\'') || (bytes[0] == b'"' && bytes[1] == b'"'));

        let next = if duplicate_start {
            trimmed[1..].to_string()
        } else if matching_quotes {
            trimmed[1..trimmed.len() - 1].to_string()
        } else {
            trimmed
        };

        if next == value {
            return next;
        }
        value = next;
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
    let mut ids = HashSet::new();
    let id = string_field(row, &["id", "curie"]).or_else(|| {
        equivalents
            .and_then(|items| items.first())
            .and_then(equivalent_id)
    })?;
    ids.insert(id.clone());
    if let Some(equivalents) = equivalents {
        for equivalent in equivalents {
            if let Some(eid) = equivalent_id(equivalent) {
                ids.insert(eid);
            }
        }
    }
    Some((id, ids.into_iter().collect()))
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

/// One shard of the term-aggregation map.
type TermShard = HashMap<String, HashSet<(u32, u8)>>;

/// Concrete sharded term-aggregation map: term → set of (curie_id, source_id).
struct TermMap {
    shards: Vec<RwLock<TermShard>>,
}

impl TermMap {
    fn new() -> Self {
        let mut shards = Vec::with_capacity(SHARD_COUNT);
        for _ in 0..SHARD_COUNT {
            shards.push(RwLock::new(HashMap::new()));
        }
        TermMap { shards }
    }
}

// ---------------------------------------------------------------------------
// Phase 1: build in-memory equivalents lookup (parallel over class files)
// ---------------------------------------------------------------------------

fn build_equivalents_map(classes: &[PathBuf]) -> PyResult<HashMap<String, Vec<String>>> {
    let partial: Vec<PyResult<HashMap<String, Vec<String>>>> = classes
        .par_iter()
        .map(|path| {
            let mut local: HashMap<String, Vec<String>> = HashMap::new();
            for_json_lines(path, |row| {
                if let Some((id, equivalents)) = class_id_and_equivalents(&row) {
                    local.insert(id, equivalents);
                }
                Ok(())
            })?;
            Ok(local)
        })
        .collect();

    let mut merged: HashMap<String, Vec<String>> = HashMap::new();
    for result in partial {
        let local = result?;
        merged.extend(local);
    }
    Ok(merged)
}

// ---------------------------------------------------------------------------
// Phases 2+3: single-pass synonym processing (parallel over synonym files)
//
// Collects dimensions (prefixes, categories, sources), assigns CURIE IDs,
// builds CurieRows, and aggregates term → (curie_id, source_id) pairs —
// all in one pass per file.
// ---------------------------------------------------------------------------

/// Result of the parallel synonym-processing pass.
struct SynonymBuildResult {
    /// prefix string → u16 id
    prefix_ids: HashMap<String, u16>,
    /// category string → u16 id
    category_ids: HashMap<String, u16>,
    /// source string → u8 id
    source_ids: HashMap<String, u8>,
    /// curie_id → CurieRow (indexed by id)
    curie_rows: Vec<CurieRow>,
    /// term → set of (curie_id, source_id)
    terms: TermMap,
}

fn process_synonyms(
    synonyms: &[PathBuf],
    equivalents: &HashMap<String, Vec<String>>,
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

    // Concurrent CURIE ID assignment + CurieRow storage.
    let curie_map: ShardedMap<u32> = ShardedMap::new();
    let curie_counter = AtomicU32::new(0);
    // CurieRows stored as (curie_id, CurieRow) pairs; merged after the pass.
    let curie_rows_collected: RwLock<Vec<(u32, CurieRow)>> = RwLock::new(Vec::new());

    // Term aggregation.
    let terms = TermMap::new();

    // Parallel pass over synonym files.
    let results: Vec<PyResult<()>> = synonyms
        .par_iter()
        .map(|path| {
            let src_name = source_name(path);
            let source_id = *source_ids
                .get(&src_name)
                .ok_or_else(|| PyRuntimeError::new_err(format!("uninterned source {src_name}")))?;

            // Thread-local buffers to avoid lock contention in the hot loop.
            let mut local_curie_rows: Vec<(u32, CurieRow)> = Vec::new();
            let mut local_terms: HashMap<String, HashSet<(u32, u8)>> = HashMap::new();

            for_json_lines(path, |row| {
                let Some(curie) = string_field(&row, &["curie", "id"]) else {
                    return Ok(());
                };
                let Some((prefix, local_id)) = split_curie(&curie) else {
                    return Ok(());
                };

                // Get-or-create prefix ID.
                let prefix_id = prefix_map.get_or_insert_with(prefix, || {
                    u16::try_from(prefix_counter.fetch_add(1, Ordering::Relaxed))
                        .expect("too many fullmap prefixes")
                });

                // Get-or-create category ID.
                let category_name = first_category(&row);
                let category_id = category_map.get_or_insert_with(&category_name, || {
                    u16::try_from(category_counter.fetch_add(1, Ordering::Relaxed))
                        .expect("too many fullmap categories")
                });

                // Get-or-create CURIE ID.
                let preferred_name = string_field(&row, &["preferred_name", "name"])
                    .unwrap_or_else(|| curie.clone());
                let taxon_id = first_taxon(&row);

                let mut is_new = false;
                let curie_id = curie_map.get_or_insert_with(&curie, || {
                    is_new = true;
                    curie_counter.fetch_add(1, Ordering::Relaxed)
                });

                if is_new {
                    local_curie_rows.push((
                        curie_id,
                        CurieRow {
                            prefix_id,
                            local_id: local_id.to_string(),
                            preferred_name: clean(preferred_name),
                            category_id,
                            taxon_id,
                        },
                    ));
                }

                let pair = (curie_id, source_id);

                // Collect all terms: names + curie + equivalents.
                let mut all_terms = string_array(&row, "names");
                all_terms.push(curie.clone());
                if let Some(equivs) = equivalents.get(&curie) {
                    all_terms.extend(equivs.iter().cloned());
                }

                // Generate L1 and L2 normalised term keys into the
                // thread-local map (no locking in the hot loop).
                for term in &all_terms {
                    let cleaned = clean(term.clone());
                    if !token_qc(&cleaned) {
                        continue;
                    }
                    let l1 = level_one(&cleaned);
                    if token_qc(&l1) {
                        local_terms.entry(l1.clone()).or_default().insert(pair);
                        let l2 = level_two(&l1);
                        if l2 != l1 && token_qc(&l2) {
                            local_terms.entry(l2).or_default().insert(pair);
                        }
                    }
                }

                Ok(())
            })?;

            // Flush thread-local CurieRows into the shared collection.
            if !local_curie_rows.is_empty() {
                curie_rows_collected
                    .write()
                    .unwrap()
                    .extend(local_curie_rows);
            }

            // Merge thread-local terms into the shared TermMap.
            for (term, pairs) in local_terms {
                let idx = shard_index(&term);
                let mut shard = terms.shards[idx].write().unwrap();
                shard.entry(term).or_default().extend(pairs);
            }

            Ok(())
        })
        .collect();

    // Propagate any errors from parallel tasks.
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

    // Build curie_rows vec indexed by curie_id.
    let collected = curie_rows_collected.into_inner().unwrap();
    let max_id = collected.iter().map(|(id, _)| *id).max().unwrap_or(0);
    let mut curie_rows: Vec<CurieRow> = Vec::with_capacity(max_id as usize + 1);
    // Sort by curie_id so we can place them in order.
    let mut sorted = collected;
    sorted.sort_unstable_by_key(|(id, _)| *id);
    // Fill the vec (IDs are 0..N with no gaps since the counter is sequential).
    for (id, row) in sorted {
        while curie_rows.len() <= id as usize {
            // Safety: IDs are assigned sequentially from 0, so gaps should not
            // occur.  If they do, pad with a placeholder (should never happen).
            curie_rows.push(CurieRow {
                prefix_id: 0,
                local_id: String::new(),
                preferred_name: String::new(),
                category_id: 0,
                taxon_id: 0,
            });
        }
        curie_rows[id as usize] = row;
    }

    Ok(SynonymBuildResult {
        prefix_ids,
        category_ids,
        source_ids,
        curie_rows,
        terms,
    })
}

// ---------------------------------------------------------------------------
// Phase 4: write final database (single write transaction)
// ---------------------------------------------------------------------------

fn write_final_database(output: &Path, result: &SynonymBuildResult) -> PyResult<()> {
    // Pre-serialise all RECORDS entries outside the write transaction.
    // Key is xxh64(term) for fast fixed-width B-tree inserts; the full term
    // is stored inside the value for collision verification on lookup.
    let mut record_entries: Vec<(u64, Vec<u8>)> = Vec::new();
    for shard in &result.terms.shards {
        let shard_read = shard.read().unwrap();
        for (term, pairs) in shard_read.iter() {
            let mut sorted_pairs: Vec<(u32, u8)> = pairs.iter().copied().collect();
            sorted_pairs.sort_unstable();
            let encoded = bincode::serialize(&(term.as_str(), &sorted_pairs)).map_err(py_err)?;
            record_entries.push((xxh64(term.as_bytes(), 0), encoded));
        }
    }
    record_entries.sort_unstable_by_key(|(hash, _)| *hash);

    let database = redb::Builder::new()
        .set_cache_size(8 * 1024 * 1024 * 1024) // 8 GB cache
        .create(output)
        .map_err(py_err)?;

    // Write dimension tables + CURIES + META in one small transaction.
    let write = database.begin_write().map_err(py_err)?;
    {
        let mut prefix_table = write.open_table(PREFIXES).map_err(py_err)?;
        for (value, id) in &result.prefix_ids {
            prefix_table.insert(*id, value.as_str()).map_err(py_err)?;
        }
        drop(prefix_table);

        let mut category_table = write.open_table(CATEGORIES).map_err(py_err)?;
        for (value, id) in &result.category_ids {
            category_table.insert(*id, value.as_str()).map_err(py_err)?;
        }
        drop(category_table);

        let mut source_table = write.open_table(SOURCES).map_err(py_err)?;
        for (value, id) in &result.source_ids {
            let encoded = bincode::serialize(&SourceRow {
                source_name: value.clone(),
            })
            .map_err(py_err)?;
            source_table
                .insert(*id, encoded.as_slice())
                .map_err(py_err)?;
        }
        drop(source_table);

        let mut curie_table = write.open_table(CURIES).map_err(py_err)?;
        for (id, curie) in result.curie_rows.iter().enumerate() {
            let encoded = bincode::serialize(curie).map_err(py_err)?;
            curie_table
                .insert(id as u32, encoded.as_slice())
                .map_err(py_err)?;
        }
        drop(curie_table);

        let mut meta = write.open_table(META).map_err(py_err)?;
        meta.insert("schema", SCHEMA_VERSION).map_err(py_err)?;
    }
    write.commit().map_err(py_err)?;

    // Write RECORDS in batched transactions (1 M rows each) to keep each
    // B-tree mutation set small enough for redb to flush efficiently while
    // still amortising the fsync cost over many rows.
    // Write RECORDS in a single transaction with Durability::None for
    // maximum insert throughput, then a final empty Immediate commit to
    // flush everything to disk.
    let mut write = database.begin_write().map_err(py_err)?;
    write.set_durability(Durability::None);
    {
        let mut table = write.open_table(RECORDS).map_err(py_err)?;
        for (hash, encoded) in &record_entries {
            table.insert(*hash, encoded.as_slice()).map_err(py_err)?;
        }
    }
    write.commit().map_err(py_err)?;
    // Final durable commit to persist all pages.
    let write = database.begin_write().map_err(py_err)?;
    write.commit().map_err(py_err)?;

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
    cache
        .write()
        .map_err(py_err)?
        .retain(|(cached_path, _mtime), _database| cached_path != &canonical);
    Ok(())
}

fn cache_database(path: &Path, database: Arc<Database>) -> PyResult<()> {
    let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let modified = std::fs::metadata(&canonical)
        .and_then(|metadata| metadata.modified())
        .map_err(py_err)?;
    let cache = DB_CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    let mut write = cache.write().map_err(py_err)?;
    write.retain(|(cached_path, _mtime), _database| cached_path != &canonical);
    write.insert((canonical, modified), database);
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (output, classes, synonyms, source_version, threads=None, write_batch_size=50000))]
pub fn build_fullmap_db(
    output: PathBuf,
    classes: Vec<PathBuf>,
    synonyms: Vec<PathBuf>,
    source_version: String,
    threads: Option<usize>,
    write_batch_size: usize,
) -> PyResult<()> {
    if synonyms.is_empty() {
        return Err(PyValueError::new_err(
            "at least one synonym file is required",
        ));
    }
    drop(source_version);
    let _write_batch_size = write_batch_size; // kept for CLI compat; unused in v2 build

    // Thread count: explicit --threads flag wins; default to all CPUs.
    let worker_count = threads
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        })
        .max(1);

    // Build a dedicated rayon pool so we don't disturb the global pool.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(worker_count)
        .build()
        .map_err(py_err)?;

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(py_err)?;
    }
    evict_cached_path(&output)?;
    if output.exists() {
        std::fs::remove_file(&output).map_err(py_err)?;
    }

    pool.install(|| {
        // Phase 1: build in-memory equivalents lookup from class files.
        let equivalents = build_equivalents_map(&classes)?;

        // Phases 2+3: single-pass synonym processing.
        let result = process_synonyms(&synonyms, &equivalents)?;

        // Phase 4: write final database.
        write_final_database(&output, &result)?;

        Ok::<(), PyErr>(())
    })?;

    // Cache the freshly-built database for the read path.
    let database = Arc::new(Database::open(&output).map_err(py_err)?);
    cache_database(&output, database)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Read path (unchanged)
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
    let modified = std::fs::metadata(&canonical)
        .and_then(|metadata| metadata.modified())
        .map_err(py_err)?;
    let key = (canonical.clone(), modified);
    let cache = DB_CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    if let Some(database) = cache.read().map_err(py_err)?.get(&key) {
        return Ok(Arc::clone(database));
    }

    let database = Arc::new(Database::open(&canonical).map_err(py_err)?);
    validate_schema(&database)?;
    let mut write = cache.write().map_err(py_err)?;
    write.retain(|(path, _mtime), _database| path != &canonical);
    let cached = Arc::clone(&database);
    write.insert(key, database);
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

fn lookup_pair_terms(
    db: PathBuf,
    terms: Vec<String>,
    threads: Option<usize>,
) -> PyResult<PairRecords> {
    let workers = threads.unwrap_or(1).max(1).min(terms.len().max(1));
    let database = open_cached(db)?;
    if workers <= 1 || terms.len() <= 1 {
        return lookup_pair_chunk(&database, &terms);
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
    let database = open_cached(db.clone())?;
    let prefix_map = load_string_table(&database, PREFIXES)?;
    let category_map = load_string_table(&database, CATEGORIES)?;
    let source_map = load_sources(&database)?;
    let pair_rows = lookup_pair_terms(db, terms, threads)?;
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

    #[test]
    fn clean_strips_matching_and_duplicate_quotes() {
        assert_eq!(clean("  'BRCA1'  ".to_string()), "BRCA1");
        assert_eq!(clean("\"\"TP53\"".to_string()), "TP53");
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

        build_fullmap_db(
            output.clone(),
            vec![classes],
            vec![synonyms],
            "test-version".to_string(),
            Some(1),
            1,
        )
        .unwrap();
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

        build_fullmap_db(
            output.clone(),
            Vec::new(),
            vec![synonyms],
            "ignored-version".to_string(),
            Some(1),
            1,
        )
        .unwrap();

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

        build_fullmap_db(
            output.clone(),
            Vec::new(),
            vec![synonyms],
            "ignored-version".to_string(),
            Some(1),
            1,
        )
        .unwrap();

        let database = open_cached(output).unwrap();
        let read = database.begin_read().unwrap();
        let curies = read.open_table(CURIES).unwrap();
        assert_eq!(curies.iter().unwrap().count(), 1);
    }

    #[test]
    fn build_fullmap_db_rejects_empty_synonym_list() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("fullmap.redb");

        let err = build_fullmap_db(
            output,
            Vec::new(),
            Vec::new(),
            "test-version".to_string(),
            Some(1),
            1,
        )
        .expect_err("empty synonyms should fail");

        assert!(err
            .to_string()
            .contains("at least one synonym file is required"));
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

        build_fullmap_db(
            output.clone(),
            Vec::new(),
            vec![synonyms],
            "test-version".to_string(),
            Some(1),
            1,
        )
        .unwrap();
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

        build_fullmap_db(
            output.clone(),
            Vec::new(),
            vec![synonyms],
            "test-version".to_string(),
            Some(1),
            1,
        )
        .unwrap();
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

        build_fullmap_db(
            output.clone(),
            Vec::new(),
            vec![synonyms],
            "test-version".to_string(),
            Some(1),
            1,
        )
        .unwrap();
        let rows = lookup_terms(output, vec!["quoted gene".to_string()], Some(1)).unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].1[0].preferred_name, "Quoted Gene");
    }
}
