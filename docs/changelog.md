# Changelog

The canonical release history lives in the repository root at [`CHANGELOG.md`](https://github.com/SkyeAv/Tablassert/blob/main/CHANGELOG.md).

## Current Release Notes

## 7.5.1 - 2026-07-01

### Changed
- Numeric annotation columns are now coerced and emitted as controlled-notation strings in NDJSON output instead of raw values. Two new pipeline steps wired into `Tcode.collect()` (`lib.py`): `clean_numeric()` lazily casts matching columns to `Float64` with `strict=False` (non-numeric entries drop to null), and `format_numeric()` renders them as strings — p-value columns (any name containing `"p value"`, case-insensitive) in scientific notation (`{:.4e}`), and `relationship strength` / `sample size` in decimal general format (`{:.4g}`, ≥4 significant figures). Non-matching columns are left untouched, and nulls are subsequently dropped by `strip_nulls()`. `math_op()` now also casts with `strict=False` so it tolerates residual junk in numeric annotation columns. `format_numeric()` formats via numpy-backed batch conversion rather than `map_elements` for throughput.
- Removed dead `pl.Config(set_fmt_float=...)` and `pl.Config(float_precision=...)` context managers from `compile_graph()` (`lib.py`); they were no-ops for NDJSON serialization (`write_ndjson` emits raw f64 via serde shortest-repr and ignores float display options), and the `fmt`/`precision` parameters of `compile_graph()` were removed alongside them.

### Added
- Fifteen regression tests in `test_lib.py` covering `numeric_columns()` detection (p-value substring, exact-name match, case-insensitivity), `clean_numeric()` (parse/coerce numeric and scientific notation, null out non-numeric junk, leave non-matching columns untouched, noop, idempotent on Float64), `format_numeric()` (scientific notation for p-value, decimal general format for relationship strength/sample size, null preservation, floating-point-noise cleaning, noop), null-stripped NDJSON rows, `compile_graph()` NDJSON emission, and `sig()` operating over a cleaned Float64 p-value column.
