use serde_json::{Map, Value};

fn is_bad_token(text: &str) -> bool {
    let lowered: String = text.trim().to_ascii_lowercase();
    matches!(lowered.as_str(), "" | "na" | "nan" | "null" | "none")
}

// ? Drops absent values only. Deliberately NOT Python truthiness: `0` and `false` are
// ? meaningful Biolink values (a p_value of 0, `number_of_cases: 0`, `negated: false`),
// ? and treating them as absent silently deletes the key from the emitted record.
fn is_present(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(_) | Value::Number(_) => true,
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(entries) => !entries.is_empty(),
    }
}

// ? Python keep clause `str(v).strip().lower() not in bad`; only strings ever match
fn passes_bad_check(value: &Value) -> bool {
    match value {
        Value::String(text) => !is_bad_token(text),
        _ => true,
    }
}

fn keep(value: &Value) -> bool {
    is_present(value) && passes_bad_check(value)
}

// ? Python value transform: lists recurse only into dict items and keep scalars verbatim;
// nested objects recurse; scalars pass through unchanged.
fn transform(value: &Value) -> Value {
    match value {
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|item| match item {
                    Value::Object(_) => strip_nulls(item),
                    _ => item.clone(),
                })
                .collect(),
        ),
        Value::Object(_) => strip_nulls(value),
        _ => value.clone(),
    }
}

// ? Faithful port of lib.strip_nulls. Returns an object (possibly empty); the caller drops
// empty records (mirrors Python dedup_stream's `if r:`).
pub fn strip_nulls(value: &Value) -> Value {
    match value {
        Value::Object(entries) => {
            let mut kept: Map<String, Value> = Map::new();
            for (key, val) in entries {
                if keep(val) {
                    kept.insert(key.clone(), transform(val));
                }
            }
            Value::Object(kept)
        }
        _ => value.clone(),
    }
}

/// Serialize in the record's own key order.  This is the EMITTED form -- what actually
/// gets written to the NDJSON -- so it must not reorder anything.  It is deliberately not
/// a canonical form: use `canonical_json_bytes` when comparing two records for equality.
pub fn emitted_json_bytes(value: &Value) -> serde_json::Result<Vec<u8>> {
    serde_json::to_vec(value)
}

/// Serialize with every object's keys sorted, recursively.
///
/// `serde_json` is built with `preserve_order`, so plain `to_vec` leaks insertion order:
/// two logically identical records that arrived with different key order produce
/// different bytes.  The edge deduper compares records for equality, so it needs a form
/// where "same content" means "same bytes"; array order is preserved because it is
/// semantic.
pub fn canonical_json_bytes(value: &Value) -> serde_json::Result<Vec<u8>> {
    serde_json::to_vec(&canonical_value(value))
}

fn canonical_value(value: &Value) -> Value {
    match value {
        Value::Object(entries) => {
            let mut keys: Vec<&String> = entries.keys().collect();
            keys.sort_unstable();
            let mut sorted: Map<String, Value> = Map::with_capacity(entries.len());
            for key in keys {
                sorted.insert(key.clone(), canonical_value(&entries[key]));
            }
            Value::Object(sorted)
        }
        Value::Array(items) => Value::Array(items.iter().map(canonical_value).collect()),
        _ => value.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::strip_nulls;
    use serde_json::json;

    #[test]
    fn strip_nulls_removes_absent_and_null_like_values() {
        let value = json!({
            "keep": "BRCA1",
            "empty": "",
            "blank": "  ",
            "na": "NA",
            "null": null,
            "empty_array": [],
            "empty_object": {},
            "nested": {"drop": "null", "keep": true}
        });

        let result = strip_nulls(&value);
        assert_eq!(result, json!({"keep": "BRCA1", "nested": {"keep": true}}));
    }

    #[test]
    fn strip_nulls_keeps_zero_and_false() {
        // ! `0` and `false` are meaningful Biolink values (a p_value of 0,
        // ! `number_of_cases: 0`, `negated: false`). Treating them as absent - as
        // ! Python truthiness would - silently deletes the key from the record.
        let value = json!({"p_value": 0, "number_of_cases": 0, "negated": false, "rate": 0.0});

        let result = strip_nulls(&value);
        assert_eq!(
            result,
            json!({"p_value": 0, "number_of_cases": 0, "negated": false, "rate": 0.0})
        );
    }

    #[test]
    fn strip_nulls_keeps_emptied_nested_dict_and_list_scalars() {
        // ! Faithful Python semantics: a nested dict that empties stays as {}, and list
        // items are kept verbatim (only dict items recurse) including falsey scalars.
        let value = json!({
            "nest_all_null": {"a": "null", "b": ""},
            "mixed_list": ["keep", 0, false, null, {"x": "null"}]
        });

        let result = strip_nulls(&value);
        assert_eq!(
            result,
            json!({
                "nest_all_null": {},
                "mixed_list": ["keep", 0, false, null, {}]
            })
        );
    }
}
