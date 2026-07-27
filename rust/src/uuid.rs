use pyo3::prelude::*;
use serde_json::Value;
use uuid::Uuid;

const NIL_NAMESPACE: Uuid = Uuid::from_bytes([0; 16]);

fn uuid3(namespace: Uuid, name: &str) -> Uuid {
    Uuid::new_v3(&namespace, name.as_bytes())
}

fn uuid_part(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::Bool(false) => None,
        Value::Bool(true) => Some("true".to_string()),
        Value::Number(number) => Some(number.to_string()),
        Value::String(text) => (!text.is_empty()).then_some(text.clone()),
        Value::Array(_) | Value::Object(_) => Some(value.to_string()),
    }
}

pub fn uuid_from_parts(domain: &str, values: impl IntoIterator<Item = String>) -> String {
    let domainspace: Uuid = uuid3(NIL_NAMESPACE, domain);
    let joined: String = values.into_iter().collect::<Vec<String>>().join("\t");
    uuid3(domainspace, &joined).to_string()
}

pub fn uuid_for_json_object(domain: &str, value: &Value) -> Option<String> {
    value
        .as_object()
        .map(|object| {
            object
                .values()
                .filter_map(uuid_part)
                .collect::<Vec<String>>()
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
}
