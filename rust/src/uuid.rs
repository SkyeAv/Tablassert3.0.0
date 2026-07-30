use pyo3::prelude::*;
use serde_json::Value;
use uuid::Uuid;

const NIL_NAMESPACE: Uuid = Uuid::from_bytes([0; 16]);

fn uuid3(namespace: Uuid, name: &str) -> Uuid {
    Uuid::new_v3(&namespace, name.as_bytes())
}

fn uuid_part(value: &Value) -> Option<String> {
    match value {
        Value::Null | Value::Bool(false) => None,
        Value::Bool(true) => Some("true".to_string()),
        Value::Number(number) => Some(number.to_string()),
        Value::String(text) => (!text.is_empty()).then_some(text.clone()),
        Value::Array(_) | Value::Object(_) => Some(value.to_string()),
    }
}

pub fn uuid_from_parts(domain: &str, values: impl IntoIterator<Item = String>) -> String {
    let domainspace: Uuid = uuid3(NIL_NAMESPACE, domain);
    // Injective encoding: length-prefix each part as `<byte-len>:<part>` so no two
    // distinct part lists serialize to the same string. A plain separator join is
    // ambiguous whenever a part itself contains the separator (e.g. ["a","x\tb","y"]
    // and ["a","x","b","y"] both join to "a\tx\tb\ty"), which would let two different
    // inputs derive the same UUID. Length-prefixing keeps the derived ID collision-safe
    // with respect to its inputs.
    let mut joined: String = String::new();
    for part in values {
        joined.push_str(&part.len().to_string());
        joined.push(':');
        joined.push_str(&part);
    }
    uuid3(domainspace, &joined).to_string()
}

pub fn uuid_for_json_object(domain: &str, value: &Value) -> Option<String> {
    value
        .as_object()
        .map(|object| {
            // Canonicalize: sort entries by key so the same logical object
            // hashes identically regardless of insertion order (serde_json's
            // preserve_order otherwise leaks key order into the UUID).  Feed each
            // key and its normalized value as SEPARATE parts — never a combined
            // "key=value" string — so the encoding stays injective: a combined form
            // would let {"a":"b=c"} and {"a=b":"c"} collide on the part "a=b=c".
            // Entries whose value normalizes to nothing (null, false, empty — see
            // `uuid_part`) are dropped, key included.
            let mut entries: Vec<(&String, &Value)> = object.iter().collect();
            entries.sort_by_key(|(left, _)| *left);
            let mut parts: Vec<String> = Vec::with_capacity(entries.len() * 2);
            for (key, value) in entries {
                if let Some(part) = uuid_part(value) {
                    parts.push(key.clone());
                    parts.push(part);
                }
            }
            parts
        })
        .map(|values| uuid_from_parts(domain, values))
}

#[pyfunction]
pub fn namespace_uuid(domain: String, values: Vec<String>) -> String {
    uuid_from_parts(&domain, values)
}

#[cfg(test)]
mod tests {
    use super::{uuid_for_json_object, uuid_from_parts};
    use serde_json::json;
    use uuid::Uuid;

    #[test]
    fn uuid_from_parts_is_deterministic() {
        let first = uuid_from_parts("domain", ["a".to_string(), "b".to_string()]);
        let second = uuid_from_parts("domain", ["a".to_string(), "b".to_string()]);
        assert_eq!(first, second);
        Uuid::parse_str(&first).expect("valid UUID");
    }

    #[test]
    fn uuid_for_json_object_returns_uuid_shape() {
        let value = json!({"subject": "A", "object": "B", "predicate": "biolink:related_to"});
        let id = uuid_for_json_object("TABLASSERT", &value).expect("object UUID");
        Uuid::parse_str(&id).expect("valid UUID");
    }

    #[test]
    fn uuid_for_json_object_is_independent_of_key_order() {
        // WHY: the same logical object must yield the same UUID no matter the
        // insertion order of its keys; previously `object.values()` was hashed
        // in preserve_order insertion order, so a reordered object changed ID.
        let forward = json!({"subject": "A", "object": "B", "predicate": "r"});
        let backward = json!({"predicate": "r", "object": "B", "subject": "A"});
        let forward_id = uuid_for_json_object("TABLASSERT", &forward).expect("object UUID");
        let backward_id = uuid_for_json_object("TABLASSERT", &backward).expect("object UUID");
        assert_eq!(forward_id, backward_id);
    }

    #[test]
    fn uuid_for_json_object_distinguishes_distinct_keys() {
        // WHY: feeding key/value pairs (not just values) means objects that
        // share values under different keys get distinct UUIDs.
        let ab = json!({"a": "x", "b": "y"});
        let cd = json!({"c": "x", "d": "y"});
        let ab_id = uuid_for_json_object("TABLASSERT", &ab).expect("object UUID");
        let cd_id = uuid_for_json_object("TABLASSERT", &cd).expect("object UUID");
        assert_ne!(ab_id, cd_id);
    }

    #[test]
    fn uuid_for_json_object_has_no_key_value_boundary_collision() {
        // WHY: keys and values are fed as separate parts, so the boundary between
        // them cannot shift to create a collision. A combined "key=value" encoding
        // would map both objects below to the single part "a=b=c"; separate parts
        // (["a","b=c"] vs ["a=b","c"]) keep them distinct.
        let split_value = json!({"a": "b=c"});
        let split_key = json!({"a=b": "c"});
        let value_id = uuid_for_json_object("TABLASSERT", &split_value).expect("object UUID");
        let key_id = uuid_for_json_object("TABLASSERT", &split_key).expect("object UUID");
        assert_ne!(value_id, key_id);
    }

    #[test]
    fn uuid_from_parts_is_injective_across_part_boundaries() {
        // WHY: length-prefixing each part makes the part-list encoding injective,
        // so a part that contains the old '\t' separator can no longer masquerade
        // as multiple parts. These two part lists used to join to the same string.
        let nested = uuid_from_parts(
            "domain",
            ["a".to_string(), "x\tb".to_string(), "y".to_string()],
        );
        let flat = uuid_from_parts(
            "domain",
            [
                "a".to_string(),
                "x".to_string(),
                "b".to_string(),
                "y".to_string(),
            ],
        );
        assert_ne!(nested, flat);
    }
}
