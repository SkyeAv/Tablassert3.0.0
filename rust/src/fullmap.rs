use flate2::read::GzDecoder;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use redb::{Database, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};
use std::thread;

const RECORDS: TableDefinition<&str, &[u8]> = TableDefinition::new("records");
const PREFIXES: TableDefinition<u16, &str> = TableDefinition::new("prefixes");
const CATEGORIES: TableDefinition<u16, &str> = TableDefinition::new("categories");
const SOURCES: TableDefinition<u8, &[u8]> = TableDefinition::new("sources");
const CURIES: TableDefinition<u32, &[u8]> = TableDefinition::new("curies");
const META: TableDefinition<&str, &str> = TableDefinition::new("meta");
const EQUIVALENTS: TableDefinition<&str, &[u8]> = TableDefinition::new("equivalents");
const TERM_RECORDS: TableDefinition<&str, &[u8]> = TableDefinition::new("term_records");
const SCHEMA_VERSION: &str = "tablassert.fullmap.v2";
const SCHEMA_VERSION_V1: &str = "tablassert.fullmap.v1";
const FULLMAP_SOURCE_VERSION: &str = "2026sep1";
type CachedDatabaseKey = (PathBuf, std::time::SystemTime);
type DimensionIds = (
    HashMap<String, u16>,
    HashMap<String, u16>,
    HashMap<String, u8>,
);
type PairRecords = Vec<(String, Vec<(u32, u8)>)>;
static DB_CACHE: OnceLock<RwLock<HashMap<CachedDatabaseKey, Arc<Database>>>> = OnceLock::new();

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

fn stage_equivalents(database: &Database, classes: &[PathBuf], batch_size: usize) -> PyResult<()> {
    for path in classes {
        let mut rows: Vec<(String, Vec<u8>)> = Vec::new();
        for_json_lines(path, |row| {
            if let Some((id, equivalents)) = class_id_and_equivalents(&row) {
                let encoded = bincode::serialize(&equivalents).map_err(py_err)?;
                rows.push((id, encoded));
            }
            if rows.len() >= batch_size {
                write_bytes_batch(database, EQUIVALENTS, &rows)?;
                rows.clear();
            }
            Ok(())
        })?;
        if !rows.is_empty() {
            write_bytes_batch(database, EQUIVALENTS, &rows)?;
        }
    }
    Ok(())
}

fn staged_pair_keys(key: String, pair: (u32, u8)) -> PyResult<Vec<(String, Vec<u8>)>> {
    let cleaned = clean(key);
    if !token_qc(&cleaned) {
        return Ok(Vec::new());
    }
    let encoded = bincode::serialize(&pair).map_err(py_err)?;
    let mut out = Vec::new();
    let l1 = level_one(&cleaned);
    let l2 = level_two(&l1);
    if token_qc(&l1) {
        out.push((stage_record_key(&l1, &encoded), encoded.clone()));
    }
    if token_qc(&l2) {
        out.push((stage_record_key(&l2, &encoded), encoded));
    }
    Ok(out)
}

fn stage_record_key(term: &str, encoded: &[u8]) -> String {
    format!("{}\0{:x}", term, md5::compute(encoded))
}

fn staged_term(key: &str) -> &str {
    key.split_once('\0').map(|(term, _)| term).unwrap_or(key)
}

fn write_bytes_batch(
    database: &Database,
    table_definition: TableDefinition<&str, &[u8]>,
    rows: &[(String, Vec<u8>)],
) -> PyResult<()> {
    let write = database.begin_write().map_err(py_err)?;
    {
        let mut table = write.open_table(table_definition).map_err(py_err)?;
        for (key, value) in rows {
            table
                .insert(key.as_str(), value.as_slice())
                .map_err(py_err)?;
        }
    }
    write.commit().map_err(py_err)?;
    Ok(())
}

fn initialize_build_tables(database: &Database) -> PyResult<()> {
    let write = database.begin_write().map_err(py_err)?;
    {
        let _equivalents = write.open_table(EQUIVALENTS).map_err(py_err)?;
        let _term_records = write.open_table(TERM_RECORDS).map_err(py_err)?;
        let _prefixes = write.open_table(PREFIXES).map_err(py_err)?;
        let _categories = write.open_table(CATEGORIES).map_err(py_err)?;
        let _sources = write.open_table(SOURCES).map_err(py_err)?;
    }
    write.commit().map_err(py_err)?;
    Ok(())
}

fn equivalent_terms(database: &Database, curie: &str) -> PyResult<Vec<String>> {
    let read = database.begin_read().map_err(py_err)?;
    let table = read.open_table(EQUIVALENTS).map_err(py_err)?;
    let Some(bytes) = table.get(curie).map_err(py_err)? else {
        return Ok(Vec::new());
    };
    bincode::deserialize(bytes.value()).map_err(py_err)
}

fn source_name(path: &Path) -> String {
    path.file_stem()
        .and_then(|x| x.to_str())
        .unwrap_or("BABEL")
        .trim_end_matches(".ndjson")
        .to_string()
}

fn ids_u16(values: BTreeSet<String>, label: &str) -> PyResult<HashMap<String, u16>> {
    values
        .into_iter()
        .enumerate()
        .map(|(idx, value)| {
            let id = u16::try_from(idx)
                .map_err(|_| PyRuntimeError::new_err(format!("too many {label}s")))?;
            Ok((value, id))
        })
        .collect()
}

fn ids_u8(values: BTreeSet<String>, label: &str) -> PyResult<HashMap<String, u8>> {
    values
        .into_iter()
        .enumerate()
        .map(|(idx, value)| {
            let id = u8::try_from(idx)
                .map_err(|_| PyRuntimeError::new_err(format!("too many {label}s")))?;
            Ok((value, id))
        })
        .collect()
}

fn intern_synonym_dimensions(database: &Database, synonyms: &[PathBuf]) -> PyResult<DimensionIds> {
    let mut prefixes = BTreeSet::new();
    let mut categories = BTreeSet::new();
    let mut sources = BTreeSet::new();

    for path in synonyms {
        sources.insert(source_name(path));
        for_json_lines(path, |row| {
            if let Some(curie) = string_field(&row, &["curie", "id"]) {
                if let Some((prefix, _local)) = split_curie(&curie) {
                    prefixes.insert(prefix.to_string());
                }
            }
            categories.insert(first_category(&row));
            Ok(())
        })?;
    }

    let prefix_ids = ids_u16(prefixes, "fullmap prefix")?;
    let category_ids = ids_u16(categories, "fullmap category")?;
    let source_ids = ids_u8(sources, "fullmap source")?;

    let write = database.begin_write().map_err(py_err)?;
    {
        let mut prefix_table = write.open_table(PREFIXES).map_err(py_err)?;
        for (value, id) in &prefix_ids {
            prefix_table.insert(*id, value.as_str()).map_err(py_err)?;
        }
        let mut category_table = write.open_table(CATEGORIES).map_err(py_err)?;
        for (value, id) in &category_ids {
            category_table.insert(*id, value.as_str()).map_err(py_err)?;
        }
        let mut source_table = write.open_table(SOURCES).map_err(py_err)?;
        for (value, id) in &source_ids {
            let encoded = bincode::serialize(&SourceRow {
                source_name: value.clone(),
            })
            .map_err(py_err)?;
            source_table
                .insert(*id, encoded.as_slice())
                .map_err(py_err)?;
        }
    }
    write.commit().map_err(py_err)?;
    Ok((prefix_ids, category_ids, source_ids))
}

fn stage_synonyms(
    database: &Database,
    synonyms: &[PathBuf],
    batch_size: usize,
) -> PyResult<BTreeMap<u32, CurieRow>> {
    let (prefix_ids, category_ids, source_ids) = intern_synonym_dimensions(database, synonyms)?;
    let mut rows: Vec<(String, Vec<u8>)> = Vec::new();
    let mut curie_ids: HashMap<String, u32> = HashMap::new();
    let mut curies: BTreeMap<u32, CurieRow> = BTreeMap::new();

    for path in synonyms {
        let source_name = source_name(path);
        let source_id = *source_ids
            .get(&source_name)
            .ok_or_else(|| PyRuntimeError::new_err(format!("uninterned source {source_name}")))?;
        for_json_lines(path, |row| {
            let Some(curie) = string_field(&row, &["curie", "id"]) else {
                return Ok(());
            };
            let Some((prefix, local_id)) = split_curie(&curie) else {
                return Ok(());
            };
            let preferred_name =
                string_field(&row, &["preferred_name", "name"]).unwrap_or_else(|| curie.clone());
            let category_name = first_category(&row);
            let prefix_id = *prefix_ids
                .get(prefix)
                .ok_or_else(|| PyRuntimeError::new_err(format!("uninterned prefix {prefix}")))?;
            let category_id = *category_ids.get(&category_name).ok_or_else(|| {
                PyRuntimeError::new_err(format!("uninterned category {category_name}"))
            })?;
            let next_id = u32::try_from(curie_ids.len())
                .map_err(|_| PyRuntimeError::new_err("too many fullmap CURIE rows"))?;
            let curie_id = *curie_ids.entry(curie.clone()).or_insert_with(|| {
                curies.insert(
                    next_id,
                    CurieRow {
                        prefix_id,
                        local_id: local_id.to_string(),
                        preferred_name: clean(preferred_name),
                        category_id,
                        taxon_id: first_taxon(&row),
                    },
                );
                next_id
            });
            let pair = (curie_id, source_id);

            let mut terms = string_array(&row, "names");
            terms.push(curie.clone());
            terms.extend(equivalent_terms(database, &curie)?);
            for term in terms {
                rows.extend(staged_pair_keys(term, pair)?);
                if rows.len() >= batch_size {
                    write_bytes_batch(database, TERM_RECORDS, &rows)?;
                    rows.clear();
                }
            }
            Ok(())
        })?;
    }
    if !rows.is_empty() {
        write_bytes_batch(database, TERM_RECORDS, &rows)?;
    }
    Ok(curies)
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

fn sorted_pairs(records: HashSet<(u32, u8)>) -> Vec<(u32, u8)> {
    let mut out: Vec<(u32, u8)> = records.into_iter().collect();
    out.sort_unstable();
    out
}

fn commit_final_records(database: &Database, rows: &[(String, Vec<(u32, u8)>)]) -> PyResult<()> {
    let write = database.begin_write().map_err(py_err)?;
    {
        let mut table = write.open_table(RECORDS).map_err(py_err)?;
        for (term, records) in rows {
            let encoded = bincode::serialize(records).map_err(py_err)?;
            table
                .insert(term.as_str(), encoded.as_slice())
                .map_err(py_err)?;
        }
    }
    write.commit().map_err(py_err)?;
    Ok(())
}

fn finalize_records(
    stage_database: &Database,
    final_database: &Database,
    curies: &BTreeMap<u32, CurieRow>,
    batch_size: usize,
) -> PyResult<()> {
    copy_build_tables(stage_database, final_database, curies)?;
    let read = stage_database.begin_read().map_err(py_err)?;
    let table = read.open_table(TERM_RECORDS).map_err(py_err)?;
    let mut current_term: Option<String> = None;
    let mut current_records: HashSet<(u32, u8)> = HashSet::new();
    let mut final_rows: PairRecords = Vec::new();

    for item in table.iter().map_err(py_err)? {
        let (key, value) = item.map_err(py_err)?;
        let term = staged_term(key.value()).to_string();
        if current_term
            .as_deref()
            .is_some_and(|existing| existing != term)
        {
            let finished_term = current_term.take().unwrap_or_default();
            final_rows.push((finished_term, sorted_pairs(current_records)));
            current_records = HashSet::new();
            if final_rows.len() >= batch_size {
                commit_final_records(final_database, &final_rows)?;
                final_rows.clear();
            }
        }
        current_term = Some(term);
        let record: (u32, u8) = bincode::deserialize(value.value()).map_err(py_err)?;
        current_records.insert(record);
    }

    if let Some(term) = current_term {
        final_rows.push((term, sorted_pairs(current_records)));
    }
    if !final_rows.is_empty() {
        commit_final_records(final_database, &final_rows)?;
    }
    Ok(())
}

fn copy_build_tables(
    stage_database: &Database,
    final_database: &Database,
    curies: &BTreeMap<u32, CurieRow>,
) -> PyResult<()> {
    let stage_read = stage_database.begin_read().map_err(py_err)?;
    let write = final_database.begin_write().map_err(py_err)?;
    {
        let stage_prefixes = stage_read.open_table(PREFIXES).map_err(py_err)?;
        let mut final_prefixes = write.open_table(PREFIXES).map_err(py_err)?;
        for item in stage_prefixes.iter().map_err(py_err)? {
            let (id, value) = item.map_err(py_err)?;
            final_prefixes
                .insert(id.value(), value.value())
                .map_err(py_err)?;
        }
        drop(final_prefixes);

        let stage_categories = stage_read.open_table(CATEGORIES).map_err(py_err)?;
        let mut final_categories = write.open_table(CATEGORIES).map_err(py_err)?;
        for item in stage_categories.iter().map_err(py_err)? {
            let (id, value) = item.map_err(py_err)?;
            final_categories
                .insert(id.value(), value.value())
                .map_err(py_err)?;
        }
        drop(final_categories);

        let stage_sources = stage_read.open_table(SOURCES).map_err(py_err)?;
        let mut final_sources = write.open_table(SOURCES).map_err(py_err)?;
        for item in stage_sources.iter().map_err(py_err)? {
            let (id, value) = item.map_err(py_err)?;
            final_sources
                .insert(id.value(), value.value())
                .map_err(py_err)?;
        }
        drop(final_sources);

        let mut final_curies = write.open_table(CURIES).map_err(py_err)?;
        for (id, curie) in curies {
            let encoded = bincode::serialize(curie).map_err(py_err)?;
            final_curies
                .insert(*id, encoded.as_slice())
                .map_err(py_err)?;
        }
    }
    write.commit().map_err(py_err)?;
    Ok(())
}

fn temp_path(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(suffix);
    PathBuf::from(value)
}

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
    let _threads = threads.unwrap_or(1).max(1);
    let batch_size = write_batch_size.max(1);
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(py_err)?;
    }
    let stage_output = temp_path(&output, ".stage.tmp");
    if stage_output.exists() {
        std::fs::remove_file(&stage_output).map_err(py_err)?;
    }
    evict_cached_path(&output)?;
    if output.exists() {
        std::fs::remove_file(&output).map_err(py_err)?;
    }
    let stage_database = Database::create(&stage_output).map_err(py_err)?;
    let final_database = Arc::new(Database::create(&output).map_err(py_err)?);

    initialize_build_tables(&stage_database)?;
    stage_equivalents(&stage_database, &classes, batch_size)?;
    let curies = stage_synonyms(&stage_database, &synonyms, batch_size)?;
    finalize_records(&stage_database, &final_database, &curies, batch_size)?;

    let write = final_database.begin_write().map_err(py_err)?;
    {
        let mut meta = write.open_table(META).map_err(py_err)?;
        meta.insert("schema", SCHEMA_VERSION).map_err(py_err)?;
    }
    write.commit().map_err(py_err)?;
    cache_database(&output, Arc::clone(&final_database))?;
    drop(stage_database);
    drop(final_database);
    std::fs::remove_file(&stage_output).map_err(py_err)?;
    Ok(())
}

fn validate_schema(database: &Database) -> PyResult<()> {
    let read = database.begin_read().map_err(py_err)?;
    let meta = read.open_table(META).map_err(py_err)?;
    let schema = meta
        .get("schema")
        .map_err(py_err)?
        .map(|x| x.value().to_string());
    match schema.as_deref() {
        Some(SCHEMA_VERSION) => Ok(()),
        Some(SCHEMA_VERSION_V1) => Err(PyRuntimeError::new_err(
            "fullmap DB is v1; rebuild with 'tablassert build-fullmap'",
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
        if let Some(bytes) = table.get(term.as_str()).map_err(py_err)? {
            let records: Vec<(u32, u8)> = bincode::deserialize(bytes.value()).map_err(py_err)?;
            out.push((term.clone(), records));
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
            .contains("fullmap DB is v1; rebuild with 'tablassert build-fullmap'"));
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
