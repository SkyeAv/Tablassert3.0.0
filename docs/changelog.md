# Changelog

The canonical release history lives in the repository root at [`CHANGELOG.md`](https://github.com/SkyeAv/Tablassert/blob/main/CHANGELOG.md).

## Current Release Notes

## 7.4.13 - 2026-06-30

### Changes

- Removed the datassert prevalidation failure for unresolved `statement.subject` / `statement.object` literal encodings under `method: value`. Graph builds no longer abort during `Tcode.model_validate(...)` for cases like `"Incertae Sedis XI"`; unresolved literal values are now allowed through config validation so downstream runtime handling can decide whether they map or get filtered.
- Added a regression test at the `Tcode.model_validate(...)` layer covering an unresolved `method: value` subject encoding, matching the build-time validation path reported in the field.

## 7.4.12 - 2026-06-29

### Changes

- Expanded the placeholder-term filter regex in `distinct()` (`fullmap.py`) to drop additional non-informative terms during entity resolution. The `bad` pattern now also excludes `not applicable`, `p value`, `variable`, `result`, `exposure`, `expression`, and `symbol` alongside the existing `none`, `nan`, `na`, `null`, and `unknown`, preventing these generic column-header-like values from being sent through resolution and producing spurious CURIE mappings.
