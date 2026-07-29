//! STRICT golden integration tests for the fullmap database build.
//!
//! These pin the EXACT on-disk result of the build pipeline so any future
//! optimization that changes the output (terms indexed, CURIE strings, dimension
//! tables, schema, or shard layout) fails loudly.  They are integration tests
//! (separate crate from the unit tests in `src/fullmap.rs`) and therefore drive
//! the build through the PUBLIC `build_fullmap_db` re-exported at the crate root,
//! then inspect the resulting redb files DIRECTLY:
//!
//! * RECORDS live in the 16 sibling shard files (`fullmap.s{0..15}.redb`), which
//!   the build does NOT cache, so they open cleanly with `Database::open`.
//! * dims/CURIES/META live in the primary (`fullmap.redb`), which `build_fullmap_db`
//!   caches and holds under redb's exclusive flock.  To read it directly we COPY
//!   the committed primary file to a fresh inode (no lock) and open the copy.
//!
//! The canonical comparison form is `term -> sorted(CURIE strings)`, serialized
//! one `term|curie,curie` line per term, sorted by term.  Using CURIE STRINGS
//! (not `curie_id`s) makes the golden invariant under thread count and internal
//! id-assignment order, so it pins the RESULT while tolerating scheduling changes.
//!
//! All tests are OFFLINE and use `tempfile` scratch dirs.

use flate2::write::GzEncoder;
use flate2::Compression;
use redb::{Database, ReadableTable, TableDefinition};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

// redb table definitions — MUST match the names/types in `src/fullmap.rs`.
const RECORDS: TableDefinition<u64, &[u8]> = TableDefinition::new("records");
const PREFIXES: TableDefinition<u16, &str> = TableDefinition::new("prefixes");
const CATEGORIES: TableDefinition<u16, &str> = TableDefinition::new("categories");
const SOURCES: TableDefinition<u8, &[u8]> = TableDefinition::new("sources");
const CURIES: TableDefinition<u32, &[u8]> = TableDefinition::new("curies");
const META: TableDefinition<&str, &str> = TableDefinition::new("meta");

const SCHEMA_VERSION: &str = "tablassert.fullmap.v4";
const SHARD_COUNT: usize = 16;

/// bincode layout MUST match `CurieRow` in `src/fullmap.rs` (field order + types).
#[derive(Deserialize)]
struct CurieRow {
    prefix_id: u16,
    local_id: String,
    preferred_name: String,
    category_id: u16,
    // Required for the bincode layout; not asserted directly here.
    #[allow(dead_code)]
    taxon_id: i32,
}

/// bincode layout MUST match `SourceRow` in `src/fullmap.rs`.
#[derive(Deserialize)]
struct SourceRow {
    source_name: String,
}

// ---------------------------------------------------------------------------
// Fixed embedded fixture.  Deliberately covers: plain ASCII names, unicode names
// (café / naïve), escaped JSON (quotes, backslashes, \uXXXX), alias fields
// (id/name/categories/taxon), dead terms (12345/none/nan), equivalent
// identifiers, multiple names per row, an empty names array, a null
// preferred_name, a row with no names array, and a class row with no
// equivalent_identifiers.  The synonym source file is named "SRC.ndjson" so the
// single source interns as "SRC" (source_id 0).
// ---------------------------------------------------------------------------

const SYNONYM_LINES: &[&str] = &[
    r#"{"curie":"HGNC:1","preferred_name":"Alpha Gene","names":["Alpha Gene","alpha"],"types":["Gene"],"taxa":["NCBITaxon:9606"]}"#,
    r#"{"curie":"HGNC:2","preferred_name":"café","names":["café","naïve"],"types":["Gene"],"taxa":["NCBITaxon:9606"]}"#,
    r#"{"curie":"HGNC:3","preferred_name":"Esc","names":["\"Quoted Name\"","back\\slash","\u00e9t\u00e9"],"types":["Gene"],"taxa":["NCBITaxon:9606"]}"#,
    r#"{"id":"MONDO:1","name":"Alias Disease","names":["alias disease"],"categories":["biolink:Disease"],"taxon":["NCBITaxon:0"]}"#,
    r#"{"curie":"HGNC:4","preferred_name":"Dead","names":["12345","none","nan","realname"],"types":["Gene"],"taxa":["NCBITaxon:9606"]}"#,
    r#"{"curie":"HGNC:5","preferred_name":"Empty Names","names":[],"types":["Gene"],"taxa":["NCBITaxon:9606"]}"#,
    r#"{"curie":"HGNC:6","preferred_name":"Shared Hit","names":["shared"],"types":["Gene"],"taxa":["NCBITaxon:9606"]}"#,
    r#"{"curie":"MONDO:2","preferred_name":"Shared Disease","names":["shared"],"types":["Disease"],"taxa":["NCBITaxon:0"]}"#,
    r#"{"curie":"HGNC:7","preferred_name":null,"names":["nullname"],"types":["Gene"],"taxa":["NCBITaxon:9606"]}"#,
    r#"{"curie":"HGNC:8","preferred_name":"No Names","types":["Gene"],"taxa":["NCBITaxon:9606"]}"#,
    r#"{"curie":"HGNC:9","preferred_name":"Equiv Free","names":["equivfree"],"types":["Gene"],"taxa":["NCBITaxon:9606"]}"#,
    r#"{"curie":"HGNC:10","preferred_name":"Multi A","names":["multi","alpha"],"types":["Gene"],"taxa":["NCBITaxon:9606"]}"#,
    r#"{"curie":"MONDO:3","preferred_name":"Multi B","names":["multi"],"types":["Disease"],"taxa":["NCBITaxon:0"]}"#,
];

const CLASS_LINES: &[&str] = &[
    r#"{"id":"HGNC:1","equivalent_identifiers":[{"identifier":"NCBIGene:100"},{"identifier":"NCBIGene:101"}]}"#,
    r#"{"id":"MONDO:1","equivalent_identifiers":[{"identifier":"DOID:999"}]}"#,
    r#"{"id":"HGNC:9"}"#,
];

/// The pinned canonical output: `term|curie,curie` lines, sorted by term, CURIEs
/// sorted within each term.  Regenerate with:
///   cargo test --test build_golden regenerate_golden -- --ignored --nocapture
const GOLDEN: &str = r#"alias disease|MONDO:1
aliasdisease|MONDO:1
alpha|HGNC:1,HGNC:10
alpha gene|HGNC:1
alphagene|HGNC:1
back\slash|HGNC:3
backslash|HGNC:3
caf|HGNC:2
café|HGNC:2
doid999|MONDO:1
doid:999|MONDO:1
equivfree|HGNC:9
hgnc1|HGNC:1
hgnc10|HGNC:10
hgnc2|HGNC:2
hgnc3|HGNC:3
hgnc4|HGNC:4
hgnc5|HGNC:5
hgnc6|HGNC:6
hgnc7|HGNC:7
hgnc8|HGNC:8
hgnc9|HGNC:9
hgnc:1|HGNC:1
hgnc:10|HGNC:10
hgnc:2|HGNC:2
hgnc:3|HGNC:3
hgnc:4|HGNC:4
hgnc:5|HGNC:5
hgnc:6|HGNC:6
hgnc:7|HGNC:7
hgnc:8|HGNC:8
hgnc:9|HGNC:9
mondo1|MONDO:1
mondo2|MONDO:2
mondo3|MONDO:3
mondo:1|MONDO:1
mondo:2|MONDO:2
mondo:3|MONDO:3
multi|HGNC:10,MONDO:3
nave|HGNC:2
naïve|HGNC:2
ncbigene100|HGNC:1
ncbigene101|HGNC:1
ncbigene:100|HGNC:1
ncbigene:101|HGNC:1
nullname|HGNC:7
quoted name|HGNC:3
quotedname|HGNC:3
realname|HGNC:4
shared|HGNC:6,MONDO:2
t|HGNC:3
été|HGNC:3
"#;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn write_jsonl(path: &Path, lines: &[&str]) {
    let mut file = File::create(path).unwrap();
    for line in lines {
        writeln!(file, "{line}").unwrap();
    }
}

/// Sibling shard path for a primary, mirroring `shard_path` in `src/fullmap.rs`.
fn shard_path(primary: &Path, index: usize) -> PathBuf {
    let stem = primary.file_stem().unwrap().to_string_lossy().into_owned();
    let ext = primary.extension().unwrap().to_string_lossy().into_owned();
    primary.with_file_name(format!("{stem}.s{index}.{ext}"))
}

/// Build the fixed fixture at `<dir>/fullmap.redb` with `threads` workers and
/// return the primary path.  Uses the public `build_fullmap_db` (the production
/// entry point), exactly as Python callers do.
fn build_fixture(dir: &Path, threads: usize) -> PathBuf {
    pyo3::Python::initialize();
    let classes = dir.join("classes.ndjson");
    let synonyms = dir.join("SRC.ndjson");
    write_jsonl(&classes, CLASS_LINES);
    write_jsonl(&synonyms, SYNONYM_LINES);
    let output = dir.join("fullmap.redb");
    pyo3::Python::attach(|py| {
        tablassert_rs::build_fullmap_db(
            py,
            output.clone(),
            vec![classes],
            vec![synonyms],
            Some(threads),
            None,
        )
        .unwrap();
    });
    output
}

/// Open a COPY of the (flock-locked) primary so its dims/CURIES/META tables can
/// be read directly.  The build commits everything before caching the original,
/// so the copied bytes are a complete, consistent database on a fresh inode.
fn open_primary_copy(primary: &Path) -> Database {
    let copy = primary.with_file_name("primary_copy.redb");
    std::fs::copy(primary, &copy).unwrap();
    Database::open(&copy).unwrap()
}

/// Build the canonical `term -> sorted(CURIE strings)` map by iterating EVERY
/// record across ALL shard files and hydrating curie_ids through the primary's
/// CURIES + PREFIXES tables.  Independent of curie_id assignment and thread count.
fn term_curie_map(primary: &Path) -> BTreeMap<String, Vec<String>> {
    // prefix_id -> prefix string, and curie_id -> "prefix:local_id".
    let db = open_primary_copy(primary);
    let read = db.begin_read().unwrap();
    let prefixes = read.open_table(PREFIXES).unwrap();
    let mut prefix_by_id: HashMap<u16, String> = HashMap::new();
    for item in prefixes.iter().unwrap() {
        let (id, value) = item.unwrap();
        prefix_by_id.insert(id.value(), value.value().to_string());
    }
    drop(prefixes);
    let curies = read.open_table(CURIES).unwrap();
    let mut curie_by_id: HashMap<u32, String> = HashMap::new();
    for item in curies.iter().unwrap() {
        let (id, bytes) = item.unwrap();
        let row: CurieRow = bincode::deserialize(bytes.value()).unwrap();
        let prefix = prefix_by_id[&row.prefix_id].clone();
        curie_by_id.insert(id.value(), format!("{}:{}", prefix, row.local_id));
    }
    drop(curies);
    let meta = read.open_table(META).unwrap();
    let shard_count = meta
        .get("shards")
        .unwrap()
        .unwrap()
        .value()
        .parse::<usize>()
        .unwrap();
    drop(meta);
    drop(read);
    drop(db);

    // Iterate every shard's RECORDS, mapping pairs to CURIE strings.
    let mut map: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for index in 0..shard_count {
        let shard_db = Database::open(shard_path(primary, index)).unwrap();
        let shard_read = shard_db.begin_read().unwrap();
        let records = shard_read.open_table(RECORDS).unwrap();
        for item in records.iter().unwrap() {
            let (_hash, bytes) = item.unwrap();
            let (term, pairs): (String, Vec<(u32, u8)>) =
                bincode::deserialize(bytes.value()).unwrap();
            let entry = map.entry(term).or_default();
            for (curie_id, _source_id) in pairs {
                entry.push(curie_by_id[&curie_id].clone());
            }
        }
    }
    for curie_list in map.values_mut() {
        curie_list.sort();
        curie_list.dedup();
    }
    map
}

/// Serialize a `term -> sorted(CURIEs)` map to the canonical multi-line string.
fn canonical_dump(map: &BTreeMap<String, Vec<String>>) -> String {
    let mut out = String::new();
    for (term, curie_list) in map {
        out.push_str(term);
        out.push('|');
        out.push_str(&curie_list.join(","));
        out.push('\n');
    }
    out
}

// ---------------------------------------------------------------------------
// (a) GOLDEN FILE TEST — the critical pin.
// ---------------------------------------------------------------------------

#[test]
fn golden_output_is_pinned() {
    let dir = tempfile::tempdir().unwrap();
    let output = build_fixture(dir.path(), 1);
    let map = term_curie_map(&output);
    let actual = canonical_dump(&map);
    assert!(
        !actual.is_empty(),
        "fixture must produce at least one indexed term"
    );
    assert_eq!(
        GOLDEN, actual,
        "fullmap build output diverged from the pinned golden file"
    );
}

/// Ignored helper: print the current canonical dump so GOLDEN can be regenerated
/// after an intentional, reviewed output change.
#[test]
#[ignore]
fn regenerate_golden() {
    let dir = tempfile::tempdir().unwrap();
    let output = build_fixture(dir.path(), 1);
    let map = term_curie_map(&output);
    println!("===GOLDEN-START===");
    print!("{}", canonical_dump(&map));
    println!("===GOLDEN-END===");
}

// ---------------------------------------------------------------------------
// (b) DETERMINISM — same input built twice yields identical canonical output.
// ---------------------------------------------------------------------------

#[test]
fn deterministic_across_rebuilds() {
    let dir_a = tempfile::tempdir().unwrap();
    let dir_b = tempfile::tempdir().unwrap();
    let out_a = build_fixture(dir_a.path(), 1);
    let out_b = build_fixture(dir_b.path(), 1);
    let dump_a = canonical_dump(&term_curie_map(&out_a));
    let dump_b = canonical_dump(&term_curie_map(&out_b));
    assert!(!dump_a.is_empty());
    assert_eq!(
        dump_a, dump_b,
        "two builds of the same input must be identical"
    );
}

// ---------------------------------------------------------------------------
// (c) THREAD INVARIANCE — threads=1 and threads=4 agree on term -> CURIE strings
// (curie_ids are scheduling-dependent and deliberately NOT compared).
// ---------------------------------------------------------------------------

#[test]
fn thread_count_does_not_change_results() {
    let dir_a = tempfile::tempdir().unwrap();
    let dir_b = tempfile::tempdir().unwrap();
    let serial = term_curie_map(&build_fixture(dir_a.path(), 1));
    let parallel = term_curie_map(&build_fixture(dir_b.path(), 4));
    assert!(!serial.is_empty());
    assert_eq!(
        serial, parallel,
        "thread count must not change term -> CURIE results"
    );
}

// ---------------------------------------------------------------------------
// (d) DIMENSION ROUND-TRIP — every dimension table is complete and consistent.
// ---------------------------------------------------------------------------

#[test]
fn dimension_tables_are_complete_and_consistent() {
    let dir = tempfile::tempdir().unwrap();
    let output = build_fixture(dir.path(), 1);
    let db = open_primary_copy(&output);
    let read = db.begin_read().unwrap();

    let prefixes = read.open_table(PREFIXES).unwrap();
    let mut prefix_ids: HashMap<u16, String> = HashMap::new();
    for item in prefixes.iter().unwrap() {
        let (id, value) = item.unwrap();
        prefix_ids.insert(id.value(), value.value().to_string());
    }
    drop(prefixes);

    let categories = read.open_table(CATEGORIES).unwrap();
    let mut category_ids: HashMap<u16, String> = HashMap::new();
    for item in categories.iter().unwrap() {
        let (id, value) = item.unwrap();
        category_ids.insert(id.value(), value.value().to_string());
    }
    drop(categories);

    let sources = read.open_table(SOURCES).unwrap();
    let mut source_names: Vec<String> = Vec::new();
    for item in sources.iter().unwrap() {
        let (_id, bytes) = item.unwrap();
        let row: SourceRow = bincode::deserialize(bytes.value()).unwrap();
        source_names.push(row.source_name);
    }
    drop(sources);

    // The fixture's synonym CURIEs use exactly the HGNC + MONDO prefixes and the
    // Gene + Disease categories; the single synonym file interns as source "SRC".
    let mut prefix_values: Vec<String> = prefix_ids.values().cloned().collect();
    prefix_values.sort();
    assert_eq!(prefix_values, vec!["HGNC".to_string(), "MONDO".to_string()]);
    let mut category_values: Vec<String> = category_ids.values().cloned().collect();
    category_values.sort();
    assert_eq!(
        category_values,
        vec!["Disease".to_string(), "Gene".to_string()]
    );
    assert_eq!(source_names, vec!["SRC".to_string()]);

    // Every CURIE row's prefix_id and category_id resolve to a valid dimension
    // entry, and the reconstructed CURIE strings are the expected 13 entities.
    let curies = read.open_table(CURIES).unwrap();
    let mut curie_strings: Vec<String> = Vec::new();
    let mut count = 0usize;
    for item in curies.iter().unwrap() {
        let (_id, bytes) = item.unwrap();
        let row: CurieRow = bincode::deserialize(bytes.value()).unwrap();
        let prefix = prefix_ids
            .get(&row.prefix_id)
            .unwrap_or_else(|| panic!("CURIE references missing prefix_id {}", row.prefix_id));
        assert!(
            category_ids.contains_key(&row.category_id),
            "CURIE references missing category_id {}",
            row.category_id
        );
        curie_strings.push(format!("{}:{}", prefix, row.local_id));
        count += 1;
    }
    assert_eq!(count, 13, "fixture must produce exactly 13 CURIE rows");
    curie_strings.sort();
    let expected: Vec<String> = [
        "HGNC:1", "HGNC:10", "HGNC:2", "HGNC:3", "HGNC:4", "HGNC:5", "HGNC:6", "HGNC:7", "HGNC:8",
        "HGNC:9", "MONDO:1", "MONDO:2", "MONDO:3",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    assert_eq!(curie_strings, expected);
}

// ---------------------------------------------------------------------------
// (e) SCHEMA PIN — META advertises the v4 schema and 16 shards.
// ---------------------------------------------------------------------------

#[test]
fn schema_and_shard_count_are_pinned() {
    let dir = tempfile::tempdir().unwrap();
    let output = build_fixture(dir.path(), 1);
    let db = open_primary_copy(&output);
    let read = db.begin_read().unwrap();
    let meta = read.open_table(META).unwrap();
    assert_eq!(meta.get("schema").unwrap().unwrap().value(), SCHEMA_VERSION);
    assert_eq!(meta.get("shards").unwrap().unwrap().value(), "16");
    drop(meta);
    drop(read);
    drop(db);

    // Exactly 16 shard files exist on disk (s0..s15), and no s16.
    for index in 0..SHARD_COUNT {
        assert!(
            shard_path(&output, index).exists(),
            "missing shard file {index}"
        );
    }
    assert!(
        !shard_path(&output, SHARD_COUNT).exists(),
        "s16 must not exist"
    );
}

// ---------------------------------------------------------------------------
// (f) GZ INPUT — a gzipped synonym file yields identical results to plain.
// ---------------------------------------------------------------------------

#[test]
fn gz_input_matches_plain_input() {
    // Plain build.
    let dir_plain = tempfile::tempdir().unwrap();
    let plain = build_fixture(dir_plain.path(), 1);
    let plain_map = term_curie_map(&plain);

    // Gz build: same class + synonym content, synonym file gzipped.  The source
    // name derives from the stem ("SRC.ndjson.gz" -> "SRC.ndjson" -> "SRC"), so
    // it interns identically to the plain build.
    pyo3::Python::initialize();
    let dir_gz = tempfile::tempdir().unwrap();
    let classes = dir_gz.path().join("classes.ndjson");
    write_jsonl(&classes, CLASS_LINES);
    let synonym_text = {
        let mut s = String::new();
        for line in SYNONYM_LINES {
            s.push_str(line);
            s.push('\n');
        }
        s
    };
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(synonym_text.as_bytes()).unwrap();
    let gz_bytes = encoder.finish().unwrap();
    let synonyms_gz = dir_gz.path().join("SRC.ndjson.gz");
    std::fs::write(&synonyms_gz, gz_bytes).unwrap();
    let output_gz = dir_gz.path().join("fullmap.redb");
    pyo3::Python::attach(|py| {
        tablassert_rs::build_fullmap_db(
            py,
            output_gz.clone(),
            vec![classes],
            vec![synonyms_gz],
            Some(1),
            None,
        )
        .unwrap();
    });

    let gz_map = term_curie_map(&output_gz);
    assert!(!gz_map.is_empty());
    assert_eq!(
        plain_map, gz_map,
        "gz input must match the uncompressed equivalent"
    );
}

// ---------------------------------------------------------------------------
// (g) EDGE CASES
// ---------------------------------------------------------------------------

/// An empty synonym file (0 rows) builds an empty DB: no records, no CURIE rows.
#[test]
fn empty_synonym_file_builds_empty_db() {
    pyo3::Python::initialize();
    let dir = tempfile::tempdir().unwrap();
    let classes = dir.path().join("classes.ndjson");
    write_jsonl(&classes, CLASS_LINES);
    let synonyms = dir.path().join("SRC.ndjson");
    write_jsonl(&synonyms, &[]); // 0 rows
    let output = dir.path().join("fullmap.redb");
    pyo3::Python::attach(|py| {
        tablassert_rs::build_fullmap_db(
            py,
            output.clone(),
            vec![classes],
            vec![synonyms],
            Some(1),
            None,
        )
        .unwrap();
    });

    assert!(
        term_curie_map(&output).is_empty(),
        "no rows -> no indexed terms"
    );
    let db = open_primary_copy(&output);
    let read = db.begin_read().unwrap();
    let curies = read.open_table(CURIES).unwrap();
    assert_eq!(
        curies.iter().unwrap().count(),
        0,
        "no rows -> no CURIE rows"
    );
}

/// A synonym row with no `names` array indexes only its CURIE (l1 + l2 forms).
#[test]
fn synonym_row_with_no_names_indexes_only_curie() {
    pyo3::Python::initialize();
    let dir = tempfile::tempdir().unwrap();
    let synonyms = dir.path().join("SRC.ndjson");
    write_jsonl(
        &synonyms,
        &[
            r#"{"curie":"HGNC:8","preferred_name":"No Names","types":["Gene"],"taxa":["NCBITaxon:9606"]}"#,
        ],
    );
    let output = dir.path().join("fullmap.redb");
    pyo3::Python::attach(|py| {
        tablassert_rs::build_fullmap_db(py, output.clone(), vec![], vec![synonyms], Some(1), None)
            .unwrap();
    });

    let map = term_curie_map(&output);
    let terms: Vec<String> = map.keys().cloned().collect();
    // Only the CURIE's level-one ("hgnc:8") and level-two ("hgnc8") forms appear
    // (BTreeMap keys are already sorted; "hgnc8" < "hgnc:8" since '8' < ':').
    assert_eq!(terms, vec!["hgnc8".to_string(), "hgnc:8".to_string()]);
    assert_eq!(map["hgnc:8"], vec!["HGNC:8".to_string()]);
    assert_eq!(map["hgnc8"], vec!["HGNC:8".to_string()]);
}

/// A null `preferred_name` falls back to the CURIE itself on the CURIE row.
#[test]
fn null_preferred_name_falls_back_to_curie() {
    pyo3::Python::initialize();
    let dir = tempfile::tempdir().unwrap();
    let synonyms = dir.path().join("SRC.ndjson");
    write_jsonl(
        &synonyms,
        &[
            r#"{"curie":"HGNC:7","preferred_name":null,"names":["nullname"],"types":["Gene"],"taxa":["NCBITaxon:9606"]}"#,
        ],
    );
    let output = dir.path().join("fullmap.redb");
    pyo3::Python::attach(|py| {
        tablassert_rs::build_fullmap_db(py, output.clone(), vec![], vec![synonyms], Some(1), None)
            .unwrap();
    });

    let db = open_primary_copy(&output);
    let read = db.begin_read().unwrap();
    let curies = read.open_table(CURIES).unwrap();
    let rows: Vec<CurieRow> = curies
        .iter()
        .unwrap()
        .map(|item| bincode::deserialize(item.unwrap().1.value()).unwrap())
        .collect();
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].preferred_name, "HGNC:7",
        "null preferred_name -> CURIE"
    );
}

/// A class row with no `equivalent_identifiers` builds cleanly and adds no
/// equivalent-derived terms (only the synonym's own names + CURIE resolve).
#[test]
fn class_row_without_equivalents_builds() {
    pyo3::Python::initialize();
    let dir = tempfile::tempdir().unwrap();
    let classes = dir.path().join("classes.ndjson");
    write_jsonl(&classes, &[r#"{"id":"HGNC:9"}"#]);
    let synonyms = dir.path().join("SRC.ndjson");
    write_jsonl(
        &synonyms,
        &[
            r#"{"curie":"HGNC:9","preferred_name":"Equiv Free","names":["equivfree"],"types":["Gene"],"taxa":["NCBITaxon:9606"]}"#,
        ],
    );
    let output = dir.path().join("fullmap.redb");
    pyo3::Python::attach(|py| {
        tablassert_rs::build_fullmap_db(
            py,
            output.clone(),
            vec![classes],
            vec![synonyms],
            Some(1),
            None,
        )
        .unwrap();
    });

    let map = term_curie_map(&output);
    // "equivfree" and the CURIE forms resolve; no equivalent-derived terms exist.
    assert_eq!(map["equivfree"], vec!["HGNC:9".to_string()]);
    assert_eq!(map["hgnc:9"], vec!["HGNC:9".to_string()]);
    assert!(
        map.keys()
            .all(|t| ["equivfree", "hgnc:9", "hgnc9"].contains(&t.as_str())),
        "unexpected extra terms: {:?}",
        map.keys().collect::<Vec<_>>()
    );
}
