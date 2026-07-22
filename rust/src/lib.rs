use pyo3::prelude::*;

mod fullmap;
mod json;
mod ndjson;
mod uuid;

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
    Ok(())
}
