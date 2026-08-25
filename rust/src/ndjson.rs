use crate::json::{canonical_json_bytes, emitted_json_bytes, strip_nulls};
use crate::uuid::uuid_for_json_object;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use uuid::Uuid;
use xxhash_rust::xxh64::xxh64;

fn runtime_error(error: impl ToString) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

/// Collision-safe dedup decision for the NODES stream.  Records are bucketed by xxh64
/// but a record is suppressed ONLY when its canonical bytes exactly match an existing
/// entry in the bucket, so two DISTINCT records whose xxh64 collides both survive.
/// (The former `HashSet<u64>` keyed on the hash alone silently dropped the second record
/// on a collision — data loss at scale.)  Returns true when `bytes` is new and recorded.
///
/// Edges do NOT use this path: their `id` is a derived hash, so they dedup on the id
/// itself (see `EdgeIndex`), which both guarantees id uniqueness and keeps 16 bytes per
/// edge instead of the whole record.
fn record_if_new(seen: &mut FxHashMap<u64, Vec<Vec<u8>>>, bytes: &[u8]) -> bool {
    let bucket = seen.entry(xxh64(bytes, 0)).or_default();
    if bucket.iter().any(|existing| existing.as_slice() == bytes) {
        false
    } else {
        bucket.push(bytes.to_vec());
        true
    }
}

fn label_edge(mut value: Value, domain: &str, fields: Option<&[String]>) -> PyResult<Value> {
    let id: String = uuid_for_json_object(domain, &value, fields)
        .ok_or_else(|| runtime_error("expected JSON object"))?;
    value
        .as_object_mut()
        .ok_or_else(|| runtime_error("expected JSON object"))?
        .insert("id".to_string(), Value::String(id));
    Ok(value)
}

/// A record ready to write, plus (for edges) the content hash of the record as it stood
/// BEFORE its `id` was inserted.
///
/// Hashing pre-insertion matters twice over: the id is a pure function of the other
/// fields, so a post-insertion hash would agree whenever the ids agree -- exactly the
/// divergence this is meant to detect -- and it avoids cloning every record just to strip
/// one key back out.
struct Finalized {
    value: Value,
    content: u64,
}

fn finalize_record(
    value: Value,
    is_edges: bool,
    domain: &str,
    fields: Option<&[String]>,
) -> PyResult<Option<Finalized>> {
    let cleaned: Value = strip_nulls(&value);
    let is_empty_object: bool = matches!(&cleaned, Value::Object(map) if map.is_empty());
    if is_empty_object {
        return Ok(None);
    }
    if !is_edges {
        return Ok(Some(Finalized {
            value: cleaned,
            content: 0,
        }));
    }
    let content: u64 = xxh64(&canonical_json_bytes(&cleaned).map_err(runtime_error)?, 0);
    label_edge(cleaned, domain, fields).map(|value| Some(Finalized { value, content }))
}

/// Edge dedup state: derived id -> hash of that edge's canonical, id-free content.
///
/// Keying on the id (not on the whole record, as the nodes path does) is what makes
/// `uuid_fields` safe: once the hash covers only a subset of the record, two records can
/// share an id while differing in bytes, and byte-keyed dedup would happily emit both --
/// duplicate edge ids in the output. It is also ~20-25x cheaper: 24 bytes per edge
/// instead of a full copy of every record (~800 bytes each at the 10M-edge scale).
#[derive(Default)]
struct EdgeIndex {
    seen: FxHashMap<[u8; 16], u64>,
}

/// What to do with an edge whose id has been seen before.
enum EdgeVerdict {
    /// First sighting of this id -- write it.
    Fresh,
    /// Byte-identical to the record that already claimed this id -- suppress it.
    Duplicate,
    /// A DIFFERENT record claims this id: the declared `uuid_fields` are not a key.
    Divergent,
}

impl EdgeIndex {
    fn classify(&mut self, id: [u8; 16], content: u64) -> EdgeVerdict {
        match self.seen.get(&id) {
            None => {
                self.seen.insert(id, content);
                EdgeVerdict::Fresh
            }
            Some(existing) if *existing == content => EdgeVerdict::Duplicate,
            Some(_) => EdgeVerdict::Divergent,
        }
    }
}

fn edge_id_bytes(value: &Value) -> PyResult<[u8; 16]> {
    let id: &str = value
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| runtime_error("labeled edge is missing its id"))?;
    Uuid::parse_str(id)
        .map(Uuid::into_bytes)
        .map_err(|error| runtime_error(format!("edge id {id} is not a UUID: {error}")))
}

/// Recover the record that first claimed `id` so the error can name what actually
/// differs.  Runs ONLY on the failure path -- the build is about to abort, so re-reading
/// the partial output costs nothing in the happy case.
fn find_written_edge(output: &Path, id: &str) -> Option<Value> {
    let reader = BufReader::new(File::open(output).ok()?);
    reader
        .lines()
        .map_while(Result::ok)
        .filter(|line| line.contains(id))
        .find_map(|line| {
            let value: Value = serde_json::from_str(&line).ok()?;
            (value.get("id").and_then(Value::as_str) == Some(id)).then_some(value)
        })
}

/// Top-level keys whose values differ between two records.
fn differing_keys(left: &Value, right: &Value) -> Vec<String> {
    let (Some(left), Some(right)) = (left.as_object(), right.as_object()) else {
        return Vec::new();
    };
    let mut keys: Vec<String> = left
        .keys()
        .chain(right.keys())
        .filter(|key| key.as_str() != "id")
        .filter(|key| left.get(*key) != right.get(*key))
        .cloned()
        .collect();
    keys.sort_unstable();
    keys.dedup();
    keys
}

/// Build the `uuid-fields-not-a-key` diagnostic.
///
/// The author's declared `uuid_fields` do not uniquely identify an edge in this graph, so
/// two genuinely different records derived the same id. Emitting both would ship duplicate
/// ids; silently dropping one would lose data. Fail, and name the fields that would fix it.
fn not_a_key_error(output: &Path, id: &str, incoming: &Value, fields: Option<&[String]>) -> PyErr {
    let declared: String = match fields {
        Some(fields) => fields.join(", "),
        None => "<all fields>".to_string(),
    };
    let differing: String = find_written_edge(output, id)
        .map(|existing| differing_keys(&existing, incoming))
        .filter(|keys| !keys.is_empty())
        .map_or_else(
            || "  (could not recover the first record to diff)".to_string(),
            |keys| format!("  they differ in: {}", keys.join(", ")),
        );
    let context: String = ["subject", "predicate", "object"]
        .iter()
        .filter_map(|key| {
            incoming
                .get(*key)
                .and_then(Value::as_str)
                .map(|value| format!("{key}={value}"))
        })
        .collect::<Vec<String>>()
        .join(" ");
    runtime_error(format!(
        "uuid-fields-not-a-key: declared uuid_fields are not a key for this graph.\n\
         \x20 id {id} is claimed by 2 different edges.\n\
         {differing}\n\
         \x20 declared uuid_fields: {declared}\n\
         \x20 offending edge: {context}\n\
         Add a discriminating field to `uuid_fields` (a qualifier, `has_supporting_studies` \
         for the source row, or the statistic that actually differs)."
    ))
}

#[pyfunction]
#[pyo3(signature = (input, output, is_edges, domain=None, uuid_fields=None))]
pub fn dedup_ndjson(
    input: PathBuf,
    output: PathBuf,
    is_edges: bool,
    domain: Option<String>,
    uuid_fields: Option<Vec<String>>,
) -> PyResult<()> {
    let domain: String = domain.unwrap_or_else(|| "TABLASSERT".to_string());
    let fields: Option<&[String]> = uuid_fields.as_deref();
    let reader: BufReader<File> = BufReader::new(File::open(input).map_err(runtime_error)?);
    let mut writer: BufWriter<File> = BufWriter::new(File::create(&output).map_err(runtime_error)?);
    let mut nodes: FxHashMap<u64, Vec<Vec<u8>>> = FxHashMap::default();
    let mut edges: EdgeIndex = EdgeIndex::default();

    reader
        .lines()
        .map(|line| line.map_err(runtime_error))
        .filter(|line| line.as_ref().map_or(true, |line| !line.trim().is_empty()))
        .map(|line| {
            line.and_then(|line| serde_json::from_str::<Value>(&line).map_err(runtime_error))
        })
        .map(|value| value.and_then(|value| finalize_record(value, is_edges, &domain, fields)))
        .try_for_each(|record| -> PyResult<()> {
            let Some(Finalized { value, content }) = record? else {
                return Ok(());
            };
            let write: bool = if is_edges {
                // Edges dedup on their derived id, so the output can never carry the same
                // id twice: an exact repeat is suppressed and a genuine divergence aborts
                // the build rather than shipping a duplicate.
                match edges.classify(edge_id_bytes(&value)?, content) {
                    EdgeVerdict::Fresh => true,
                    EdgeVerdict::Duplicate => false,
                    EdgeVerdict::Divergent => {
                        writer.flush().map_err(runtime_error)?;
                        let id: &str = value["id"].as_str().unwrap_or_default();
                        return Err(not_a_key_error(&output, id, &value, fields));
                    }
                }
            } else {
                record_if_new(
                    &mut nodes,
                    &emitted_json_bytes(&value).map_err(runtime_error)?,
                )
            };
            if write {
                writer
                    .write_all(&emitted_json_bytes(&value).map_err(runtime_error)?)
                    .map_err(runtime_error)?;
                writer.write_all(b"\n").map_err(runtime_error)?;
            }
            Ok(())
        })?;
    // Flush explicitly and propagate failure: relying on BufWriter's drop-time
    // flush would swallow a final write error and report success with truncated
    // output.
    writer.flush().map_err(runtime_error)
}

#[cfg(test)]
mod tests {
    use super::{dedup_ndjson, differing_keys, record_if_new};
    use rustc_hash::FxHashMap;
    use serde_json::Value;
    use std::fs;
    use tempfile::tempdir;
    use uuid::Uuid;

    #[test]
    fn record_if_new_suppresses_only_exact_byte_duplicates() {
        // WHY: dedup must be collision-safe. Keying on xxh64 alone (the old
        // HashSet<u64>) would drop a distinct record that hashes into an
        // occupied bucket; suppression must require an exact byte match.
        let mut seen: FxHashMap<u64, Vec<Vec<u8>>> = FxHashMap::default();
        // First sighting of a record is new.
        assert!(record_if_new(&mut seen, b"{\"id\":\"A\"}"));
        // A byte-identical record is suppressed.
        assert!(!record_if_new(&mut seen, b"{\"id\":\"A\"}"));
        // Distinct records survive even when they land in the same hash bucket
        // (a real xxh64 collision is impractical to force, so this exercises the
        // byte-exact equality path that decides suppression directly).
        assert!(record_if_new(&mut seen, b"{\"id\":\"B\"}"));
        assert!(!record_if_new(&mut seen, b"{\"id\":\"B\"}"));
    }

    #[test]
    fn dedup_ndjson_keeps_distinct_records() {
        // WHY: two different records must both survive dedup; only an identical
        // duplicate is collapsed.
        let dir = tempdir().expect("tempdir");
        let input = dir.path().join("nodes.ndjson.tmp");
        let output = dir.path().join("nodes.ndjson");
        fs::write(&input, "{\"id\":\"A\"}\n{\"id\":\"B\"}\n{\"id\":\"A\"}\n").expect("write input");

        dedup_ndjson(input, output.clone(), false, None, None).expect("dedup nodes");

        let lines: Vec<String> = fs::read_to_string(output)
            .expect("read output")
            .lines()
            .map(str::to_string)
            .collect();
        assert_eq!(
            lines,
            vec!["{\"id\":\"A\"}".to_string(), "{\"id\":\"B\"}".to_string()]
        );
    }

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

        dedup_ndjson(input, output.clone(), false, None, None).expect("dedup nodes");

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

        dedup_ndjson(
            input,
            output.clone(),
            true,
            Some("TABLASSERT".to_string()),
            None,
        )
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

        dedup_ndjson(
            input,
            output.clone(),
            true,
            Some("TABLASSERT".to_string()),
            None,
        )
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

        dedup_ndjson(input, output.clone(), false, None, None).expect("dedup nodes");

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

        dedup_ndjson(input, output.clone(), false, None, None).expect("dedup nodes");

        assert_eq!(fs::read_to_string(output).expect("read output"), "");
    }

    #[test]
    fn dedup_ndjson_skips_blank_lines() {
        let dir = tempdir().expect("tempdir");
        let input = dir.path().join("nodes.ndjson.tmp");
        let output = dir.path().join("nodes.ndjson");
        fs::write(&input, "\n  \n{\"id\":\"A\"}\n\n{\"id\":\"A\"}\n   \n").expect("write input");

        dedup_ndjson(input, output.clone(), false, None, None).expect("dedup nodes");

        assert_eq!(
            fs::read_to_string(output).expect("read output"),
            "{\"id\":\"A\"}\n"
        );
    }

    fn write_edges(dir: &std::path::Path, body: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        let input = dir.join("edges.ndjson.tmp");
        let output = dir.join("edges.ndjson");
        fs::write(&input, body).expect("write input");
        (input, output)
    }

    fn edge_ids(output: &std::path::Path) -> Vec<String> {
        fs::read_to_string(output)
            .expect("read output")
            .lines()
            .map(|line| {
                serde_json::from_str::<Value>(line).expect("json")["id"]
                    .as_str()
                    .expect("edge id")
                    .to_string()
            })
            .collect()
    }

    #[test]
    fn declared_uuid_fields_hold_the_id_still_across_attribute_edits() {
        // WHY: the whole point of `uuid_fields`. Two builds whose only difference is a
        // p_value must produce the SAME edge id, so downstream sees one edge updated
        // rather than one retired and one created.
        let dir = tempdir().expect("tempdir");
        let fields = Some(vec![
            "subject".to_string(),
            "predicate".to_string(),
            "object".to_string(),
        ]);

        let (before_in, before_out) = write_edges(
            dir.path(),
            "{\"subject\":\"A\",\"predicate\":\"r\",\"object\":\"B\",\"p_value\":\"0.01\"}\n",
        );
        dedup_ndjson(before_in, before_out.clone(), true, None, fields.clone()).expect("dedup");

        let after_dir = tempdir().expect("tempdir");
        let (after_in, after_out) = write_edges(
            after_dir.path(),
            "{\"subject\":\"A\",\"predicate\":\"r\",\"object\":\"B\",\"p_value\":\"0.99\",\"effect_size\":1.5}\n",
        );
        dedup_ndjson(after_in, after_out.clone(), true, None, fields).expect("dedup");

        assert_eq!(edge_ids(&before_out), edge_ids(&after_out));
    }

    #[test]
    fn undeclared_uuid_fields_let_the_id_drift() {
        // WHY: the contrast case -- with no `uuid_fields`, the same attribute edit moves
        // the id. This is the pre-16.0.0 behavior the default still preserves.
        let dir = tempdir().expect("tempdir");
        let (before_in, before_out) = write_edges(
            dir.path(),
            "{\"subject\":\"A\",\"predicate\":\"r\",\"object\":\"B\",\"p_value\":\"0.01\"}\n",
        );
        dedup_ndjson(before_in, before_out.clone(), true, None, None).expect("dedup");

        let after_dir = tempdir().expect("tempdir");
        let (after_in, after_out) = write_edges(
            after_dir.path(),
            "{\"subject\":\"A\",\"predicate\":\"r\",\"object\":\"B\",\"p_value\":\"0.99\"}\n",
        );
        dedup_ndjson(after_in, after_out.clone(), true, None, None).expect("dedup");

        assert_ne!(edge_ids(&before_out), edge_ids(&after_out));
    }

    #[test]
    fn uuid_fields_that_are_not_a_key_abort_the_build() {
        // WHY: narrowing the hash inputs can make two DIFFERENT edges share an id.
        // Shipping both would emit duplicate ids; dropping one would lose data. Fail,
        // and name the fields that would disambiguate.
        let dir = tempdir().expect("tempdir");
        let (input, output) = write_edges(
            dir.path(),
            concat!(
                "{\"subject\":\"A\",\"predicate\":\"r\",\"object\":\"B\",\"p_value\":\"0.01\"}\n",
                "{\"subject\":\"A\",\"predicate\":\"r\",\"object\":\"B\",\"p_value\":\"0.99\"}\n"
            ),
        );
        let fields = Some(vec![
            "subject".to_string(),
            "predicate".to_string(),
            "object".to_string(),
        ]);
        let error = dedup_ndjson(input, output, true, None, fields).expect_err("not a key");
        let message = error.to_string();
        assert!(message.contains("uuid-fields-not-a-key"), "{message}");
        assert!(message.contains("p_value"), "{message}");
        assert!(message.contains("subject=A"), "{message}");
    }

    #[test]
    fn identical_edges_still_collapse_under_declared_uuid_fields() {
        // WHY: a true repeat is not a key violation -- it is the duplicate the deduper
        // exists to collapse.
        let dir = tempdir().expect("tempdir");
        let (input, output) = write_edges(
            dir.path(),
            concat!(
                "{\"subject\":\"A\",\"predicate\":\"r\",\"object\":\"B\",\"p_value\":\"0.01\"}\n",
                "{\"subject\":\"A\",\"predicate\":\"r\",\"object\":\"B\",\"p_value\":\"0.01\"}\n"
            ),
        );
        let fields = Some(vec!["subject".to_string(), "object".to_string()]);
        dedup_ndjson(input, output.clone(), true, None, fields).expect("dedup edges");
        assert_eq!(edge_ids(&output).len(), 1);
    }

    #[test]
    fn key_order_alone_no_longer_ships_a_duplicate_id() {
        // WHY: `uuid_for_json_object` sorts keys but the emitted bytes preserve insertion
        // order, so byte-keyed dedup used to keep BOTH of these -- same id, two lines.
        // Keying on the id collapses them.
        let dir = tempdir().expect("tempdir");
        let (input, output) = write_edges(
            dir.path(),
            concat!(
                "{\"subject\":\"A\",\"object\":\"B\"}\n",
                "{\"object\":\"B\",\"subject\":\"A\"}\n"
            ),
        );
        dedup_ndjson(input, output.clone(), true, None, None).expect("dedup edges");
        assert_eq!(edge_ids(&output).len(), 1);
    }

    #[test]
    fn the_domain_separates_identical_edges_across_graphs() {
        // WHY: two graphs asserting the same triple must not mint the same id, which is
        // what makes a narrow `uuid_fields` safe across KGs.
        let left_dir = tempdir().expect("tempdir");
        let (left_in, left_out) =
            write_edges(left_dir.path(), "{\"subject\":\"A\",\"object\":\"B\"}\n");
        dedup_ndjson(
            left_in,
            left_out.clone(),
            true,
            Some("infores:left".to_string()),
            None,
        )
        .expect("dedup");

        let right_dir = tempdir().expect("tempdir");
        let (right_in, right_out) =
            write_edges(right_dir.path(), "{\"subject\":\"A\",\"object\":\"B\"}\n");
        dedup_ndjson(
            right_in,
            right_out.clone(),
            true,
            Some("infores:right".to_string()),
            None,
        )
        .expect("dedup");

        assert_ne!(edge_ids(&left_out), edge_ids(&right_out));
    }

    #[test]
    fn differing_keys_reports_only_real_differences_and_never_the_id() {
        let left = serde_json::json!({"id": "x", "subject": "A", "p_value": "0.01"});
        let right =
            serde_json::json!({"id": "y", "subject": "A", "p_value": "0.99", "effect_size": 1.0});
        assert_eq!(
            differing_keys(&left, &right),
            vec!["effect_size", "p_value"]
        );
    }
}
