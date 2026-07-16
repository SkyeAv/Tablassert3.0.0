use serde_json::{Map, Number, Value};

fn is_bad_token(text: &str) -> bool {
    let lowered: String = text.trim().to_ascii_lowercase();
    matches!(lowered.as_str(), "" | "na" | "nan" | "null" | "none")
}

fn is_zero_number(number: &Number) -> bool {
    number.as_i64().is_some_and(|x| x == 0)
        || number.as_u64().is_some_and(|x| x == 0)
        || number.as_f64().is_some_and(|x| x == 0.0)
}

// ? Mirrors Python truthiness (`if v`) for JSON values
fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => !is_zero_number(number),
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
    is_truthy(value) && passes_bad_check(value)
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
            for (key, val) in entries.iter() {
                if keep(val) {
                    kept.insert(key.clone(), transform(val));
                }
            }
            Value::Object(kept)
        }
        _ => value.clone(),
    }
}

pub fn stable_json_bytes(value: &Value) -> serde_json::Result<Vec<u8>> {
    serde_json::to_vec(value)
}

#[cfg(test)]
mod tests {
    use super::strip_nulls;
    use serde_json::json;

    #[test]
    fn strip_nulls_removes_falsey_and_null_like_values() {
        let value = json!({
            "keep": "BRCA1",
            "empty": "",
            "blank": "  ",
            "na": "NA",
            "zero": 0,
            "false": false,
            "empty_array": [],
            "empty_object": {},
            "nested": {"drop": "null", "keep": true}
        });

        let result = strip_nulls(&value);
        assert_eq!(result, json!({"keep": "BRCA1", "nested": {"keep": true}}));
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
