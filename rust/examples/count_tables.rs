//! Count rows in each table of one or more fullmap redb files.
//! Usage: cargo run --release --example count_tables -- <db1> [db2 ...]
use redb::ReadableTableMetadata;
use redb::TableDefinition;

const RECORDS: TableDefinition<u64, &[u8]> = TableDefinition::new("records");
const PREFIXES: TableDefinition<u16, &str> = TableDefinition::new("prefixes");
const CATEGORIES: TableDefinition<u16, &str> = TableDefinition::new("categories");
const SOURCES: TableDefinition<u8, &[u8]> = TableDefinition::new("sources");
const CURIES: TableDefinition<u32, &[u8]> = TableDefinition::new("curies");

fn main() {
    for arg in std::env::args().skip(1) {
        let db = redb::Database::open(&arg).expect("open db");
        let read = db.begin_read().expect("begin read");
        let records = read
            .open_table(RECORDS)
            .ok()
            .and_then(|t| t.len().ok())
            .unwrap_or(0);
        let curies = read
            .open_table(CURIES)
            .ok()
            .and_then(|t| t.len().ok())
            .unwrap_or(0);
        let prefixes = read
            .open_table(PREFIXES)
            .ok()
            .and_then(|t| t.len().ok())
            .unwrap_or(0);
        let categories = read
            .open_table(CATEGORIES)
            .ok()
            .and_then(|t| t.len().ok())
            .unwrap_or(0);
        let sources = read
            .open_table(SOURCES)
            .ok()
            .and_then(|t| t.len().ok())
            .unwrap_or(0);
        let size = std::fs::metadata(&arg).map(|m| m.len()).unwrap_or(0);
        println!(
            "{arg}: records={records} curies={curies} prefixes={prefixes} categories={categories} sources={sources} file={:.2}GB",
            size as f64 / 1e9
        );
    }
}
