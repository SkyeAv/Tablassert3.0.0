use crate::json::{stable_json_bytes, strip_nulls};
use crate::uuid::uuid_for_json_object;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde_json::Value;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use xxhash_rust::xxh64::xxh64;

fn runtime_error(error: impl ToString) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

fn label_edge(mut value: Value, domain: &str) -> PyResult<Value> {
    let id: String = uuid_for_json_object(domain, &value)
        .ok_or_else(|| runtime_error("expected JSON object"))?;
    value
        .as_object_mut()
        .ok_or_else(|| runtime_error("expected JSON object"))?
        .insert("id".to_string(), Value::String(id));
    Ok(value)
}

fn finalize_record(value: Value, is_edges: bool, domain: &str) -> PyResult<Option<Value>> {
    let cleaned: Value = strip_nulls(&value);
    let is_empty_object: bool = matches!(&cleaned, Value::Object(map) if map.is_empty());
    if is_empty_object {
        return Ok(None);
    }
    if is_edges {
        label_edge(cleaned, domain).map(Some)
    } else {
        Ok(Some(cleaned))
    }
}

#[pyfunction]
#[pyo3(signature = (input, output, is_edges, domain=None))]
pub fn dedup_ndjson(
    input: PathBuf,
    output: PathBuf,
    is_edges: bool,
    domain: Option<String>,
) -> PyResult<()> {
    let domain: String = domain.unwrap_or_else(|| "TABLASSERT".to_string());
    let reader: BufReader<File> = BufReader::new(File::open(input).map_err(runtime_error)?);
    let mut writer: BufWriter<File> = BufWriter::new(File::create(output).map_err(runtime_error)?);
    let mut seen: HashSet<u64> = HashSet::new();

    reader
        .lines()
        .map(|line| line.map_err(runtime_error))
        .map(|line| {
            line.and_then(|line| serde_json::from_str::<Value>(&line).map_err(runtime_error))
        })
        .map(|value| value.and_then(|value| finalize_record(value, is_edges, &domain)))
        .try_for_each(|record| -> PyResult<()> {
            if let Some(value) = record? {
                let bytes: Vec<u8> = stable_json_bytes(&value).map_err(runtime_error)?;
                if seen.insert(xxh64(&bytes, 0)) {
                    writer.write_all(&bytes).map_err(runtime_error)?;
                    writer.write_all(b"\n").map_err(runtime_error)?;
                }
            }
            Ok(())
        })
}

#[cfg(test)]
mod tests {
    use super::dedup_ndjson;
    use serde_json::Value;
    use std::fs;
    use tempfile::tempdir;
    use uuid::Uuid;

    #[test]
    fn dedup_ndjson_deduplicates_nodes() {
        let dir = tempdir().expect("tempdir");
        let input = dir.path().join("nodes.ndjson.tmp");
        let output = dir.path().join("nodes.ndjson");
        fs::write(
            &input,
            "{\"id\":\"A\",\"drop\":\"NA\"}\n{\"id\":\"A\",\"drop\":\"NA\"}\n{}",
        )
        .expect("write input");

        dedup_ndjson(input, output.clone(), false, None).expect("dedup nodes");

        let lines: Vec<String> = fs::read_to_string(output)
            .expect("read output")
            .lines()
            .map(str::to_string)
            .collect();
        assert_eq!(lines, vec!["{\"id\":\"A\"}".to_string()]);
    }

    #[test]
    fn dedup_ndjson_labels_edges() {
        let dir = tempdir().expect("tempdir");
        let input = dir.path().join("edges.ndjson.tmp");
        let output = dir.path().join("edges.ndjson");
        fs::write(
            &input,
            "{\"subject\":\"A\",\"object\":\"B\",\"predicate\":\"r\"}\n",
        )
        .expect("write input");

        dedup_ndjson(input, output.clone(), true, Some("TABLASSERT".to_string()))
            .expect("dedup edges");

        let line = fs::read_to_string(output).expect("read output");
        let value: Value = serde_json::from_str(line.trim()).expect("json");
        let id = value.get("id").and_then(Value::as_str).expect("edge id");
        Uuid::parse_str(id).expect("valid UUID");
    }

    #[test]
    fn dedup_ndjson_duplicate_edges_get_stable_single_id() {
        let dir = tempdir().expect("tempdir");
        let input = dir.path().join("edges.ndjson.tmp");
        let output = dir.path().join("edges.ndjson");
        fs::write(
            &input,
            concat!(
                "{\"subject\":\"A\",\"object\":\"B\",\"predicate\":\"r\"}\n",
                "{\"subject\":\"A\",\"object\":\"B\",\"predicate\":\"r\"}\n"
            ),
        )
        .expect("write input");

        dedup_ndjson(input, output.clone(), true, Some("TABLASSERT".to_string()))
            .expect("dedup edges");

        let lines: Vec<String> = fs::read_to_string(output)
            .expect("read output")
            .lines()
            .map(str::to_string)
            .collect();
        assert_eq!(lines.len(), 1);
        let value: Value = serde_json::from_str(&lines[0]).expect("json");
        let id = value.get("id").and_then(Value::as_str).expect("edge id");
        Uuid::parse_str(id).expect("valid UUID");
    }

    #[test]
    fn dedup_ndjson_strips_nested_null_like_values() {
        let dir = tempdir().expect("tempdir");
        let input = dir.path().join("nodes.ndjson.tmp");
        let output = dir.path().join("nodes.ndjson");
        fs::write(
            &input,
            "{\"id\":\"A\",\"meta\":{\"drop\":\"none\",\"keep\":\"yes\"},\"items\":[{\"x\":\"NA\",\"y\":\"z\"}]}\n",
        )
        .expect("write input");

        dedup_ndjson(input, output.clone(), false, None).expect("dedup nodes");

        let line = fs::read_to_string(output).expect("read output");
        let value: Value = serde_json::from_str(line.trim()).expect("json");
        assert_eq!(value["meta"], serde_json::json!({"keep":"yes"}));
        assert_eq!(value["items"], serde_json::json!([{"y":"z"}]));
    }

    #[test]
    fn dedup_ndjson_empty_object_only_writes_empty_output() {
        let dir = tempdir().expect("tempdir");
        let input = dir.path().join("nodes.ndjson.tmp");
        let output = dir.path().join("nodes.ndjson");
        fs::write(&input, "{}\n{\"drop\":\"NA\"}\n").expect("write input");

        dedup_ndjson(input, output.clone(), false, None).expect("dedup nodes");

        assert_eq!(fs::read_to_string(output).expect("read output"), "");
    }
}
