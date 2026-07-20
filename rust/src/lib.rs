use pyo3::prelude::*;

mod json;
mod ndjson;
mod uuid;

#[pymodule]
fn rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(ndjson::dedup_ndjson, module)?)?;
    module.add_function(wrap_pyfunction!(uuid::namespace_uuid, module)?)?;
    Ok(())
}
