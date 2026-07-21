use flate2::read::GzDecoder;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use redb::{Database, TableDefinition};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;

const RECORDS: TableDefinition<&str, &[u8]> = TableDefinition::new("records");
const META: TableDefinition<&str, &str> = TableDefinition::new("meta");
const SCHEMA_VERSION: &str = "tablassert.fullmap.v1";

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
struct FullmapRecord {
    curie: String,
    preferred_name: String,
    category_name: String,
    taxon_id: i64,
    source_name: String,
    source_version: String,
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

fn read_json_lines(path: &Path) -> PyResult<Vec<Value>> {
    let reader = BufReader::new(open_reader(path)?);
    let mut rows = Vec::new();
    for line in reader.lines() {
        let raw = line.map_err(py_err)?;
        if raw.trim().is_empty() {
            continue;
        }
        rows.push(serde_json::from_str(&raw).map_err(py_err)?);
    }
    Ok(rows)
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

fn first_taxon(value: &Value) -> i64 {
    let taxon = string_array(value, "taxa")
        .into_iter()
        .next()
        .or_else(|| string_array(value, "taxon").into_iter().next())
        .unwrap_or_default();
    taxon
        .trim_start_matches("NCBITaxon:")
        .parse::<i64>()
        .unwrap_or(0)
}

fn equivalent_id(value: &Value) -> Option<String> {
    if let Some(s) = value.as_str() {
        return Some(s.to_string());
    }
    string_field(value, &["identifier", "id", "curie"])
}

fn load_equivalents(classes: &[PathBuf]) -> PyResult<HashMap<String, Vec<String>>> {
    let mut lookup = HashMap::new();
    for path in classes {
        for row in read_json_lines(path)? {
            let id = string_field(&row, &["id", "curie"]);
            let equivalents = row.get("equivalent_identifiers").and_then(Value::as_array);
            let Some(id) = id else { continue };
            let mut ids = HashSet::from([id.clone()]);
            if let Some(equivalents) = equivalents {
                for equivalent in equivalents {
                    if let Some(eid) = equivalent_id(equivalent) {
                        ids.insert(eid);
                    }
                }
            }
            lookup.insert(id, ids.into_iter().collect());
        }
    }
    Ok(lookup)
}

fn insert_key(
    map: &mut HashMap<String, HashSet<FullmapRecord>>,
    key: String,
    record: &FullmapRecord,
) {
    let cleaned = clean(key);
    if !token_qc(&cleaned) {
        return;
    }
    let l1 = level_one(&cleaned);
    let l2 = level_two(&l1);
    if token_qc(&l1) {
        map.entry(l1).or_default().insert(record.clone());
    }
    if token_qc(&l2) {
        map.entry(l2).or_default().insert(record.clone());
    }
}

fn build_records(
    classes: &[PathBuf],
    synonyms: &[PathBuf],
    source_version: &str,
) -> PyResult<HashMap<String, HashSet<FullmapRecord>>> {
    let equivalents = load_equivalents(classes)?;
    let mut map = HashMap::new();

    for path in synonyms {
        let source_name = path
            .file_stem()
            .and_then(|x| x.to_str())
            .unwrap_or("BABEL")
            .trim_end_matches(".ndjson")
            .to_string();
        for row in read_json_lines(path)? {
            let Some(curie) = string_field(&row, &["curie", "id"]) else {
                continue;
            };
            let preferred_name =
                string_field(&row, &["preferred_name", "name"]).unwrap_or_else(|| curie.clone());
            let record = FullmapRecord {
                curie: curie.clone(),
                preferred_name: clean(preferred_name),
                category_name: first_category(&row),
                taxon_id: first_taxon(&row),
                source_name: source_name.clone(),
                source_version: source_version.to_string(),
            };

            let mut terms = string_array(&row, "names");
            terms.push(curie.clone());
            if let Some(ids) = equivalents.get(&curie) {
                terms.extend(ids.iter().cloned());
            }
            for term in terms {
                insert_key(&mut map, term, &record);
            }
        }
    }

    Ok(map)
}

fn sorted_records(records: HashSet<FullmapRecord>) -> Vec<FullmapRecord> {
    let mut out: Vec<FullmapRecord> = records.into_iter().collect();
    out.sort_by(|a, b| {
        (
            &a.curie,
            &a.preferred_name,
            &a.category_name,
            a.taxon_id,
            &a.source_name,
            &a.source_version,
        )
            .cmp(&(
                &b.curie,
                &b.preferred_name,
                &b.category_name,
                b.taxon_id,
                &b.source_name,
                &b.source_version,
            ))
    });
    out
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
    let _threads = threads.unwrap_or(1).max(1);
    let batch_size = write_batch_size.max(1);
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(py_err)?;
    }
    if output.exists() {
        std::fs::remove_file(&output).map_err(py_err)?;
    }

    let map = build_records(&classes, &synonyms, &source_version)?;
    let database = Database::create(output).map_err(py_err)?;

    let mut rows: Vec<(String, Vec<FullmapRecord>)> = map
        .into_iter()
        .map(|(term, records)| (term, sorted_records(records)))
        .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));

    for chunk in rows.chunks(batch_size) {
        let write = database.begin_write().map_err(py_err)?;
        {
            let mut table = write.open_table(RECORDS).map_err(py_err)?;
            for (term, records) in chunk {
                let encoded = bincode::serialize(records).map_err(py_err)?;
                table
                    .insert(term.as_str(), encoded.as_slice())
                    .map_err(py_err)?;
            }
        }
        write.commit().map_err(py_err)?;
    }

    let write = database.begin_write().map_err(py_err)?;
    {
        let mut meta = write.open_table(META).map_err(py_err)?;
        meta.insert("schema", SCHEMA_VERSION).map_err(py_err)?;
        meta.insert("source_version", source_version.as_str())
            .map_err(py_err)?;
    }
    write.commit().map_err(py_err)?;
    Ok(())
}

fn lookup_chunk(
    database: &Database,
    terms: &[String],
) -> PyResult<Vec<(String, Vec<FullmapRecord>)>> {
    let read = database.begin_read().map_err(py_err)?;
    let meta = read.open_table(META).map_err(py_err)?;
    let schema = meta
        .get("schema")
        .map_err(py_err)?
        .map(|x| x.value().to_string());
    if schema.as_deref() != Some(SCHEMA_VERSION) {
        return Err(PyRuntimeError::new_err("unsupported fullmap redb schema"));
    }
    drop(meta);
    let table = read.open_table(RECORDS).map_err(py_err)?;
    let mut out = Vec::new();
    for term in terms {
        if let Some(bytes) = table.get(term.as_str()).map_err(py_err)? {
            let records: Vec<FullmapRecord> =
                bincode::deserialize(bytes.value()).map_err(py_err)?;
            out.push((term.clone(), records));
        }
    }
    Ok(out)
}

fn lookup_terms(
    db: PathBuf,
    terms: Vec<String>,
    threads: Option<usize>,
) -> PyResult<Vec<(String, Vec<FullmapRecord>)>> {
    let workers = threads.unwrap_or(1).max(1).min(terms.len().max(1));
    let database = Arc::new(Database::open(db).map_err(py_err)?);
    if workers <= 1 || terms.len() <= 1 {
        return lookup_chunk(&database, &terms);
    }

    let chunk_size = terms.len().div_ceil(workers);
    let mut handles = Vec::new();
    for chunk in terms.chunks(chunk_size) {
        let database = Arc::clone(&database);
        let chunk_terms = chunk.to_vec();
        handles.push(thread::spawn(move || lookup_chunk(&database, &chunk_terms)));
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

#[pyfunction]
#[pyo3(signature = (db, terms, threads=None, return_format="rows"))]
pub fn lookup_fullmap_terms<'py>(
    py: Python<'py>,
    db: PathBuf,
    terms: Vec<String>,
    threads: Option<usize>,
    return_format: &str,
) -> PyResult<Bound<'py, PyList>> {
    if return_format != "rows" {
        return Err(PyValueError::new_err(
            "only return_format='rows' is supported",
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
    fn level_normalization_matches_datassert_shape() {
        assert_eq!(level_one("BRCA-1"), "brca-1");
        assert_eq!(level_two("brca-1"), "brca1");
    }

    #[test]
    fn builds_and_reads_redb_records() {
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
        assert_eq!(rows[0].1[0].source_version, "test-version");
    }
}
