use pyo3::prelude::*;

// mimalloc returns memory to the OS far better than glibc malloc under heavy
// multi-threaded allocation (the synonym phase runs many worker threads each
// making millions of small String/Vec allocations).  Without it, per-thread
// malloc arenas retain freed memory and inflate peak RSS several-fold.
use mimalloc::MiMalloc;
use xxhash_rust::xxh32::xxh32 as xxh32_digest;
use xxhash_rust::xxh64::xxh64 as xxh64_digest;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

mod fullmap;
mod json;
mod ndjson;
mod uuid;

/// XXH32 hex digest (seed 0) of a string's UTF-8 bytes.
///
/// Mirrors Python ``xxhash.xxh32(s.encode()).hexdigest()`` so the pure-Python
/// ``xxhash`` dependency can be dropped in favour of the already-bundled
/// ``xxhash-rust`` crate.
#[pyfunction]
fn xxh32(data: &str) -> String {
    format!("{:08x}", xxh32_digest(data.as_bytes(), 0))
}

/// XXH64 hex digest (seed 0) of a string's UTF-8 bytes.
///
/// Mirrors Python ``xxhash.xxh64(s.encode()).hexdigest()`` (16 lowercase hex
/// chars, zero-padded), complementing ``xxh32`` for callers that want the
/// wider 64-bit hash from the already-bundled ``xxhash-rust`` crate.
#[pyfunction]
fn xxh64(data: &str) -> String {
    format!("{:016x}", xxh64_digest(data.as_bytes(), 0))
}

// Public re-exports of the fullmap `#[pyfunction]`s so Rust integration tests
// (`rust/tests/`) and other non-Python embedders can drive the exact production
// build/read path.  The `fullmap` module itself stays private; only these
// intended entry points are surfaced at the crate root.
pub use fullmap::{
    build_fullmap_db, fullmap_source_version, hydrate_categories, hydrate_curies, hydrate_prefixes,
    hydrate_sources, lookup_fullmap_terms,
};

#[pymodule]
fn rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(fullmap::build_fullmap_db, module)?)?;
    module.add_function(wrap_pyfunction!(fullmap::fullmap_source_version, module)?)?;
    module.add_function(wrap_pyfunction!(fullmap::hydrate_categories, module)?)?;
    module.add_function(wrap_pyfunction!(fullmap::hydrate_curies, module)?)?;
    module.add_function(wrap_pyfunction!(fullmap::hydrate_prefixes, module)?)?;
    module.add_function(wrap_pyfunction!(fullmap::hydrate_sources, module)?)?;
    module.add_function(wrap_pyfunction!(fullmap::lookup_fullmap_terms, module)?)?;
    module.add_function(wrap_pyfunction!(ndjson::dedup_ndjson, module)?)?;
    module.add_function(wrap_pyfunction!(uuid::namespace_uuid, module)?)?;
    module.add_function(wrap_pyfunction!(xxh32, module)?)?;
    module.add_function(wrap_pyfunction!(xxh64, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{xxh32, xxh64};

    #[test]
    fn xxh64_matches_known_digests() {
        // WHY: pin the XXH64 (seed 0) output to values cross-checked against
        // Python's `xxhash.xxh64(...).hexdigest()` so the Rust primitive stays
        // drop-in compatible; 16 lowercase hex chars, zero-padded.
        assert_eq!(xxh64("hello"), "26c7827d889f6da3");
        assert_eq!(xxh64(""), "ef46db3751d8e999");
    }

    #[test]
    fn xxh32_matches_known_digest() {
        // WHY: guard the existing XXH32 primitive's format (8 lowercase hex
        // chars) alongside the new XXH64 one.
        assert_eq!(xxh32(""), "02cc5d05");
    }
}
