use pyo3::prelude::*;
use serde_json::{Map, Value};
use uuid::Uuid;

const NIL_NAMESPACE: Uuid = Uuid::from_bytes([0; 16]);

fn uuid3(namespace: Uuid, name: &str) -> Uuid {
    Uuid::new_v3(&namespace, name.as_bytes())
}

/// Recursively sort object keys so a nested value's serialization is independent of
/// insertion order.  `serde_json` is built with `preserve_order`, so `Value::to_string`
/// on a nested object emits INSERTION order -- the top-level sort in
/// `uuid_for_json_object` never reached inside `sources` or `has_supporting_studies`,
/// and a polars struct-field reordering silently re-minted every edge id.  Array order
/// is preserved: it is semantic in JSON, and reordering it would conflate genuinely
/// different sequences.
fn canonicalize(value: &Value) -> Value {
    match value {
        Value::Object(entries) => {
            let mut keys: Vec<&String> = entries.keys().collect();
            keys.sort_unstable();
            let mut sorted: Map<String, Value> = Map::with_capacity(entries.len());
            for key in keys {
                sorted.insert(key.clone(), canonicalize(&entries[key]));
            }
            Value::Object(sorted)
        }
        Value::Array(items) => Value::Array(items.iter().map(canonicalize).collect()),
        _ => value.clone(),
    }
}

fn uuid_part(value: &Value) -> Option<String> {
    match value {
        // `null` never reaches the hash: `strip_nulls` removes those keys before
        // labeling.  Dropping it here too keeps the two passes in agreement.
        Value::Null => None,
        // `false` MUST hash, even though it is falsy.  `strip_nulls` deliberately keeps
        // it (`negated: false` is a meaningful Biolink value), so dropping the key here
        // made `{subject: A, negated: false}` and `{subject: A}` derive the SAME id
        // while remaining distinct records -- a duplicate-id path.
        Value::Bool(flag) => Some(if *flag { "true" } else { "false" }.to_string()),
        Value::Number(number) => Some(number.to_string()),
        Value::String(text) => (!text.is_empty()).then(|| text.clone()),
        Value::Array(_) | Value::Object(_) => Some(canonicalize(value).to_string()),
    }
}

pub fn uuid_from_parts(domain: &str, values: impl IntoIterator<Item = impl AsRef<str>>) -> String {
    let domainspace: Uuid = uuid3(NIL_NAMESPACE, domain);
    // Injective encoding: length-prefix each part as `<byte-len>:<part>` so no two
    // distinct part lists serialize to the same string. A plain separator join is
    // ambiguous whenever a part itself contains the separator (e.g. ["a","x\tb","y"]
    // and ["a","x","b","y"] both join to "a\tx\tb\ty"), which would let two different
    // inputs derive the same UUID. Length-prefixing keeps the derived ID collision-safe
    // with respect to its inputs.
    let mut joined: String = String::new();
    for part in values {
        let part: &str = part.as_ref();
        joined.push_str(&part.len().to_string());
        joined.push(':');
        joined.push_str(part);
    }
    uuid3(domainspace, &joined).to_string()
}

/// Derive an object's UUID, optionally over a declared subset of its keys.
///
/// `fields` is the graph config's `uuid_fields`.  `None` hashes every key (the default,
/// byte-compatible with pre-16.0.0 output); `Some` hashes only the named top-level keys,
/// so an edge's id stops moving when a non-identity attribute (`p_value`, `effect_size`,
/// `supporting_text`) changes.  A declared field absent from the record contributes
/// nothing at all -- neither key nor value -- so `{subject: A}` and
/// `{subject: A, negated: true}` stay distinct.
pub fn uuid_for_json_object(
    domain: &str,
    value: &Value,
    fields: Option<&[String]>,
) -> Option<String> {
    let object: &Map<String, Value> = value.as_object()?;
    // Canonicalize: visit entries in sorted key order so the same logical object
    // hashes identically regardless of insertion order (serde_json's preserve_order
    // otherwise leaks key order into the UUID).  Feed each key and its normalized
    // value as SEPARATE parts -- never a combined "key=value" string -- so the
    // encoding stays injective: a combined form would let {"a":"b=c"} and
    // {"a=b":"c"} collide on the part "a=b=c".  Entries whose value normalizes to
    // nothing (null, empty -- see `uuid_part`) are dropped, key included.
    let mut keys: Vec<&String> = match fields {
        Some(declared) => declared
            .iter()
            .filter(|key| object.contains_key(*key))
            .collect(),
        None => object.keys().collect(),
    };
    keys.sort_unstable();
    keys.dedup();
    let mut parts: Vec<String> = Vec::with_capacity(keys.len() * 2);
    for key in keys {
        if let Some(part) = uuid_part(&object[key]) {
            parts.push(key.clone());
            parts.push(part);
        }
    }
    Some(uuid_from_parts(domain, parts))
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
        let id = uuid_for_json_object("TABLASSERT", &value, None).expect("object UUID");
        Uuid::parse_str(&id).expect("valid UUID");
    }

    #[test]
    fn uuid_for_json_object_is_independent_of_key_order() {
        // WHY: the same logical object must yield the same UUID no matter the
        // insertion order of its keys; previously `object.values()` was hashed
        // in preserve_order insertion order, so a reordered object changed ID.
        let forward = json!({"subject": "A", "object": "B", "predicate": "r"});
        let backward = json!({"predicate": "r", "object": "B", "subject": "A"});
        let forward_id = uuid_for_json_object("TABLASSERT", &forward, None).expect("object UUID");
        let backward_id = uuid_for_json_object("TABLASSERT", &backward, None).expect("object UUID");
        assert_eq!(forward_id, backward_id);
    }

    #[test]
    fn uuid_for_json_object_distinguishes_distinct_keys() {
        // WHY: feeding key/value pairs (not just values) means objects that
        // share values under different keys get distinct UUIDs.
        let ab = json!({"a": "x", "b": "y"});
        let cd = json!({"c": "x", "d": "y"});
        let ab_id = uuid_for_json_object("TABLASSERT", &ab, None).expect("object UUID");
        let cd_id = uuid_for_json_object("TABLASSERT", &cd, None).expect("object UUID");
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
        let value_id = uuid_for_json_object("TABLASSERT", &split_value, None).expect("object UUID");
        let key_id = uuid_for_json_object("TABLASSERT", &split_key, None).expect("object UUID");
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

    #[test]
    fn nested_object_key_order_does_not_change_the_uuid() {
        // WHY: the top-level sort never reached inside nested values, which were hashed
        // via `Value::to_string` under serde_json's `preserve_order`. A polars struct
        // field reordering inside `sources` therefore re-minted every edge id.
        let forward = json!({"subject": "A", "sources": [{"resource_id": "infores:x", "resource_role": "primary_knowledge_source"}]});
        let backward = json!({"subject": "A", "sources": [{"resource_role": "primary_knowledge_source", "resource_id": "infores:x"}]});
        let forward_id = uuid_for_json_object("TABLASSERT", &forward, None).expect("object UUID");
        let backward_id = uuid_for_json_object("TABLASSERT", &backward, None).expect("object UUID");
        assert_eq!(forward_id, backward_id);
    }

    #[test]
    fn nested_array_order_still_changes_the_uuid() {
        // WHY: canonicalization sorts object KEYS only. Array order is semantic, so two
        // different sequences must stay distinguishable.
        let forward = json!({"publications": ["PMID:1", "PMID:2"]});
        let backward = json!({"publications": ["PMID:2", "PMID:1"]});
        let forward_id = uuid_for_json_object("TABLASSERT", &forward, None).expect("object UUID");
        let backward_id = uuid_for_json_object("TABLASSERT", &backward, None).expect("object UUID");
        assert_ne!(forward_id, backward_id);
    }

    #[test]
    fn false_is_hashed_rather_than_dropped() {
        // WHY: `strip_nulls` keeps `false` (`negated: false` is meaningful), so dropping
        // the key here made these two DISTINCT records derive the SAME id.
        let negated = json!({"subject": "A", "negated": false});
        let bare = json!({"subject": "A"});
        let negated_id = uuid_for_json_object("TABLASSERT", &negated, None).expect("object UUID");
        let bare_id = uuid_for_json_object("TABLASSERT", &bare, None).expect("object UUID");
        assert_ne!(negated_id, bare_id);
        // ...and `false` stays distinguishable from `true`.
        let affirmed = json!({"subject": "A", "negated": true});
        let affirmed_id = uuid_for_json_object("TABLASSERT", &affirmed, None).expect("object UUID");
        assert_ne!(negated_id, affirmed_id);
    }

    #[test]
    fn declared_fields_ignore_undeclared_changes() {
        // WHY: the whole point of `uuid_fields` -- an attribute-only edit must not move
        // the id.
        let fields = vec![
            "subject".to_string(),
            "predicate".to_string(),
            "object".to_string(),
        ];
        let before = json!({"subject": "A", "predicate": "r", "object": "B", "p_value": "0.01"});
        let after = json!({"subject": "A", "predicate": "r", "object": "B", "p_value": "0.99", "effect_size": 1.5});
        let before_id =
            uuid_for_json_object("TABLASSERT", &before, Some(&fields)).expect("object UUID");
        let after_id =
            uuid_for_json_object("TABLASSERT", &after, Some(&fields)).expect("object UUID");
        assert_eq!(before_id, after_id);
    }

    #[test]
    fn declared_fields_still_track_declared_changes() {
        let fields = vec![
            "subject".to_string(),
            "predicate".to_string(),
            "object".to_string(),
        ];
        let before = json!({"subject": "A", "predicate": "r", "object": "B"});
        let after = json!({"subject": "A", "predicate": "r", "object": "C"});
        let before_id =
            uuid_for_json_object("TABLASSERT", &before, Some(&fields)).expect("object UUID");
        let after_id =
            uuid_for_json_object("TABLASSERT", &after, Some(&fields)).expect("object UUID");
        assert_ne!(before_id, after_id);
    }

    #[test]
    fn declared_field_order_does_not_matter() {
        // WHY: `uuid_fields` is a set, not a sequence -- reordering the config list must
        // not re-mint every id.
        let forward = vec!["subject".to_string(), "object".to_string()];
        let backward = vec!["object".to_string(), "subject".to_string()];
        let value = json!({"subject": "A", "object": "B", "p_value": "0.01"});
        let forward_id =
            uuid_for_json_object("TABLASSERT", &value, Some(&forward)).expect("object UUID");
        let backward_id =
            uuid_for_json_object("TABLASSERT", &value, Some(&backward)).expect("object UUID");
        assert_eq!(forward_id, backward_id);
    }

    #[test]
    fn a_missing_declared_field_contributes_nothing() {
        // WHY: an absent declared field must not silently alias onto a present one.
        let fields = vec!["subject".to_string(), "negated".to_string()];
        let bare = json!({"subject": "A"});
        let present = json!({"subject": "A", "negated": true});
        let bare_id =
            uuid_for_json_object("TABLASSERT", &bare, Some(&fields)).expect("object UUID");
        let present_id =
            uuid_for_json_object("TABLASSERT", &present, Some(&fields)).expect("object UUID");
        assert_ne!(bare_id, present_id);
    }

    #[test]
    fn distinct_domains_separate_identical_records() {
        // WHY: two graphs may legitimately assert the same triple. Namespacing by the
        // graph's infores keeps their ids apart even under a narrow `uuid_fields`.
        let fields = vec![
            "subject".to_string(),
            "predicate".to_string(),
            "object".to_string(),
        ];
        let value = json!({"subject": "A", "predicate": "r", "object": "B"});
        let left =
            uuid_for_json_object("infores:left-kg", &value, Some(&fields)).expect("object UUID");
        let right =
            uuid_for_json_object("infores:right-kg", &value, Some(&fields)).expect("object UUID");
        assert_ne!(left, right);
    }

    #[test]
    fn uuid_for_json_object_rejects_non_objects() {
        assert!(uuid_for_json_object("TABLASSERT", &json!(["a"]), None).is_none());
        assert!(uuid_for_json_object("TABLASSERT", &json!("a"), None).is_none());
    }

    #[test]
    fn golden_vectors_pin_the_encoding() {
        // WHY: nothing pinned an actual UUID value before, so the derivation could change
        // silently and no assertion would fail. These vectors are the regression tripwire:
        // if one moves, edge ids in every published graph moved with it, and that needs a
        // MAJOR bump plus a CHANGELOG migration note.
        let record = json!({
            "subject": "NCBITaxon:846",
            "predicate": "biolink:affects",
            "object": "FB:FBgn0002557",
            "p_value": "9.6407e-03",
            "sources": [{"resource_id": "infores:multiomicskg", "resource_role": "primary_knowledge_source"}],
        });
        assert_eq!(
            uuid_for_json_object("TABLASSERT", &record, None).expect("object UUID"),
            "1fd95137-b2de-3963-9552-3c8b35d1f758"
        );
        let fields = vec![
            "subject".to_string(),
            "predicate".to_string(),
            "object".to_string(),
        ];
        assert_eq!(
            uuid_for_json_object("infores:multiomicskg", &record, Some(&fields))
                .expect("object UUID"),
            "7cf7352c-114f-3e7a-9e38-70ba678c958f"
        );
        assert_eq!(
            uuid_from_parts("domain", ["a", "b"]),
            "1a8199fb-c8eb-381a-85a2-33ce009c506e"
        );
    }
}
