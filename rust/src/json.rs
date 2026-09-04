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
///
/// US-003: the sorted-key form is written DIRECTLY into the output buffer instead of
/// first deep-cloning into a canonical `Value` tree -- this runs once per record on
/// every pass, so the clone was the fixed per-record cost.  Scalars are delegated to
/// serde_json's own writers, keeping escaping and number formatting byte-identical to
/// the old clone-then-`to_vec` form (pinned by `canonical_json_bytes_matches_reference_on_fuzz`).
pub fn canonical_json_bytes(value: &Value) -> serde_json::Result<Vec<u8>> {
    let mut buf: Vec<u8> = Vec::with_capacity(128);
    write_canonical(value, &mut buf)?;
    Ok(buf)
}

fn write_canonical(value: &Value, buf: &mut Vec<u8>) -> serde_json::Result<()> {
    match value {
        Value::Object(entries) => {
            let mut keys: Vec<&String> = entries.keys().collect();
            keys.sort_unstable();
            buf.push(b'{');
            let mut first: bool = true;
            for key in keys {
                if !first {
                    buf.push(b',');
                }
                first = false;
                serde_json::to_writer(&mut *buf, key)?;
                buf.push(b':');
                write_canonical(entries.get(key).expect("key came from the same map"), buf)?;
            }
            buf.push(b'}');
        }
        Value::Array(items) => {
            buf.push(b'[');
            let mut first: bool = true;
            for item in items {
                if !first {
                    buf.push(b',');
                }
                first = false;
                write_canonical(item, buf)?;
            }
            buf.push(b']');
        }
        // null / bool / number / string: serde_json's own writer emits exactly the bytes
        // `to_vec` would for the same scalar (ryu/itoa numbers, full string escaping).
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {
            serde_json::to_writer(&mut *buf, value)?
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{canonical_json_bytes, strip_nulls};
    use serde_json::{json, Map, Value};

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

    /// Byte-verbatim copy of the pre-US-003 canonical serializer: deep-clone into a
    /// key-sorted `Value` at every depth, then `to_vec` the clone.  Kept ONLY as the
    /// equivalence oracle for `canonical_json_bytes_matches_reference_on_fuzz` --
    /// production code must never call it (the deep clone was the fixed per-record cost
    /// US-003 removed).
    fn canonical_json_bytes_reference(value: &Value) -> serde_json::Result<Vec<u8>> {
        serde_json::to_vec(&canonical_value_reference(value))
    }

    fn canonical_value_reference(value: &Value) -> Value {
        match value {
            Value::Object(entries) => {
                let mut keys: Vec<&String> = entries.keys().collect();
                keys.sort_unstable();
                let mut sorted: Map<String, Value> = Map::with_capacity(entries.len());
                for key in keys {
                    sorted.insert(key.clone(), canonical_value_reference(&entries[key]));
                }
                Value::Object(sorted)
            }
            Value::Array(items) => {
                Value::Array(items.iter().map(canonical_value_reference).collect())
            }
            _ => value.clone(),
        }
    }

    /// Tiny deterministic PRNG (splitmix64): the fuzz stream must be reproducible across
    /// machines without pulling in a `rand` dependency.
    struct Rng(u64);

    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z: u64 = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn below(&mut self, bound: usize) -> usize {
            (self.next_u64() % u64::try_from(bound).expect("bound fits in u64")) as usize
        }

        fn one_in(&mut self, odds: usize) -> bool {
            self.below(odds) == 0
        }
    }

    fn shuffle<T>(rng: &mut Rng, items: &mut [T]) {
        for high in (1..items.len()).rev() {
            let low: usize = rng.below(high + 1);
            items.swap(high, low);
        }
    }

    /// Strings loaded with escape-sensitive content: quotes, backslashes, control chars,
    /// DEL, NEL/line separators, BMP and non-BMP unicode.
    const FUZZ_STRINGS: [&str; 14] = [
        "",
        "plain",
        "two words",
        "quote \" inside",
        "back\\slash",
        "\"both\" and \\ together",
        "tab\tnewline\nreturn\r",
        "\u{0}\u{1}\u{1f}",
        "\u{7f}",
        "\u{85}\u{2028}\u{2029}",
        "unicode: \u{e9} \u{fc} \u{df} \u{6f22}\u{5b57}",
        "non-bmp: \u{1d11e} \u{1f600} \u{1f980}",
        "\u{10ffff}",
        "trailing backslash\\",
    ];

    /// Keys chosen to stress the byte-wise key sort: digits-vs-letters, case, punctuation
    /// above and below letters, spaces, and multi-byte UTF-8.
    const FUZZ_KEYS: [&str; 12] = [
        "a", "b", "A", "B", "_", "~", "10", "2", "z", "\u{e9}", "zz", "a b",
    ];

    /// Shapes the fuzz stream actually covered, so a passing run cannot be vacuous.
    #[derive(Default)]
    struct FuzzStats {
        multi_key_objects: usize,
        arrays: usize,
        nested_containers: usize,
        escaped_strings: usize,
    }

    fn fuzz_number(rng: &mut Rng) -> Value {
        match rng.below(4) {
            0 => Value::Number(rng.next_u64().into()), // u64, incl. above i64::MAX
            1 => Value::Number((rng.next_u64() as i64).into()), // i64, incl. negatives
            2 => {
                let pool: [f64; 10] = [
                    0.0,
                    -0.0,
                    0.1,
                    -0.1,
                    1e30,
                    1.5e-7,
                    0.3333333333333333,
                    5e-324,
                    1.7976931348623157e308,
                    987654321.5,
                ];
                Value::Number(
                    serde_json::Number::from_f64(pool[rng.below(pool.len())])
                        .expect("pool values are finite"),
                )
            }
            _ => {
                // Random finite fraction with exponent in [-8, 8]: exercises ryu's
                // decimal-vs-exponent formatting boundary and negative signs.
                let mantissa: f64 = (rng.next_u64() % 1_000_000) as f64;
                let exponent: i32 = rng.below(17) as i32 - 8;
                let mut magnitude: f64 = mantissa * 10f64.powi(exponent);
                if rng.one_in(2) {
                    magnitude = -magnitude;
                }
                Value::Number(serde_json::Number::from_f64(magnitude).expect("finite"))
            }
        }
    }

    fn fuzz_value(rng: &mut Rng, depth: usize, stats: &mut FuzzStats) -> Value {
        // Beyond depth 4 only scalars: recursion stays bounded while still nesting deep
        // enough to exercise the recursive sorted-key write.
        match rng.below(if depth >= 4 { 4 } else { 8 }) {
            0 => Value::Null,
            1 => Value::Bool(rng.one_in(2)),
            2 => fuzz_number(rng),
            3 => {
                let text: &str = FUZZ_STRINGS[rng.below(FUZZ_STRINGS.len())];
                if text.chars().any(|c| c == '"' || c == '\\' || c < ' ') {
                    stats.escaped_strings += 1;
                }
                Value::String(text.to_string())
            }
            4 => {
                stats.arrays += 1;
                if depth > 0 {
                    stats.nested_containers += 1;
                }
                let mut items: Vec<Value> = Vec::new();
                for _ in 0..rng.below(5) {
                    items.push(fuzz_value(rng, depth + 1, stats));
                }
                Value::Array(items)
            }
            _ => {
                let mut entries: Map<String, Value> = Map::new();
                for _ in 0..rng.below(5) {
                    let key: String = FUZZ_KEYS[rng.below(FUZZ_KEYS.len())].to_string();
                    entries.insert(key, fuzz_value(rng, depth + 1, stats));
                }
                if entries.len() >= 2 {
                    stats.multi_key_objects += 1;
                }
                if depth > 0 {
                    stats.nested_containers += 1;
                }
                Value::Object(entries)
            }
        }
    }

    #[test]
    fn canonical_json_bytes_matches_reference_on_fuzz() {
        // WHY: US-003 rewrote `canonical_json_bytes` from "deep-clone into a canonical
        // Value, then to_vec" to "write the sorted-key JSON directly into the output
        // buffer".  Node dedup (`record_if_new`) keys on these exact bytes and the
        // default-mode edge content hash hashes them, so even one drifted byte (string
        // escaping, number formatting, key sort, separators) silently shifts node ids,
        // dedup decisions, and edge hashes.  This seeded fuzz drives the new writer and
        // a byte-verbatim copy of the old serializer over nested objects at depth,
        // arrays, every number shape (u64/i64/negative/fractional/exponent), bools,
        // nulls, escape-heavy unicode strings, and empty containers, and demands
        // byte-identical output.
        let mut rng = Rng(0x5EED_2024_0000_0003);
        let mut stats = FuzzStats::default();

        for _ in 0..4096 {
            let value: Value = fuzz_value(&mut rng, 0, &mut stats);
            let direct: Vec<u8> = canonical_json_bytes(&value).expect("canonical");
            let reference: Vec<u8> = canonical_json_bytes_reference(&value).expect("reference");
            assert_eq!(direct, reference, "divergence on: {value}");
        }

        // Key-order permutations of the same object: canonical bytes must not depend on
        // insertion order (serde_json's preserve_order leaks it to plain to_vec).
        let mut order: Vec<usize> = (0..FUZZ_KEYS.len()).collect();
        for _ in 0..128 {
            shuffle(&mut rng, &mut order);
            let count: usize = 1 + rng.below(FUZZ_KEYS.len());
            let entries: Vec<(String, Value)> = order[..count]
                .iter()
                .map(|index| {
                    (
                        FUZZ_KEYS[*index].to_string(),
                        fuzz_value(&mut rng, 1, &mut stats),
                    )
                })
                .collect();
            let mut forward: Map<String, Value> = Map::new();
            for (key, entry) in &entries {
                forward.insert(key.clone(), entry.clone());
            }
            let mut reversed: Map<String, Value> = Map::new();
            for (key, entry) in entries.iter().rev() {
                reversed.insert(key.clone(), entry.clone());
            }
            let forward_value = Value::Object(forward);
            let reversed_value = Value::Object(reversed);
            let direct_forward: Vec<u8> = canonical_json_bytes(&forward_value).expect("canonical");
            let direct_reversed: Vec<u8> =
                canonical_json_bytes(&reversed_value).expect("canonical");
            let reference_forward: Vec<u8> =
                canonical_json_bytes_reference(&forward_value).expect("reference");
            assert_eq!(direct_forward, direct_reversed, "key order leaked");
            assert_eq!(
                direct_forward, reference_forward,
                "divergence on permutation"
            );
        }

        // Duplicate keys in raw JSON: serde_json's parser collapses them last-wins
        // before either serializer sees the Value; pin that both implementations agree
        // on the collapsed form (plus the degenerate scalars/empties).
        let raw_cases: [&str; 8] = [
            r#"{"a":1,"a":2}"#,
            r#"{"b":{"x":[1,2],"x":null},"b":{}}"#,
            "{}",
            "[]",
            "null",
            "true",
            "\"\"",
            "-0.0",
        ];
        for raw in raw_cases {
            let value: Value = serde_json::from_str(raw).expect("parse raw case");
            let direct: Vec<u8> = canonical_json_bytes(&value).expect("canonical");
            let reference: Vec<u8> = canonical_json_bytes_reference(&value).expect("reference");
            assert_eq!(direct, reference, "raw case: {raw}");
        }

        // Non-vacuity: the seeded stream must actually exercise the risky shapes or the
        // equivalence check could pass on a trivially degenerate input.
        assert!(
            stats.multi_key_objects >= 100,
            "expected multi-key objects, got {}",
            stats.multi_key_objects
        );
        assert!(stats.arrays >= 100, "expected arrays, got {}", stats.arrays);
        assert!(
            stats.nested_containers >= 50,
            "expected nested containers, got {}",
            stats.nested_containers
        );
        assert!(
            stats.escaped_strings >= 50,
            "expected escape-heavy strings, got {}",
            stats.escaped_strings
        );
    }
}
