# Changelog

The canonical release history lives in the repository root at [`CHANGELOG.md`](https://github.com/SkyeAv/Tablassert/blob/main/CHANGELOG.md).

## Current Release Notes

## 7.4.14 - 2026-06-30

### Changes

- Extended `sig()` in `lib.py` to select a p-value column by fuzzy matching rather than requiring an exact `"p value"` name. All schema columns whose names contain the substring `"p value"` are now considered candidates; `fuzz.ratio` (rapidfuzz) scores each against the literal `"p value"` and the highest-scoring column is used to compute the `"significant"` output. An exact `"p value"` column scores 100 and is always preferred; columns like `"adjusted p value"` or `"log p value"` are used only when no exact match is present. If no column contains the substring the function continues to emit `"UNSURE"` for all rows.
- Added five regression tests in `test_lib.py` covering: exact-match preference, non-exact fallback, closest-match selection among multiple non-exact candidates, no-p-value column (UNSURE), and null value handling.

## 7.4.13 - 2026-06-30

### Changes

- Removed the datassert prevalidation failure for unresolved `statement.subject` / `statement.object` literal encodings under `method: value`. Graph builds no longer abort during `Tcode.model_validate(...)` for cases like `"Incertae Sedis XI"`; unresolved literal values are now allowed through config validation so downstream runtime handling can decide whether they map or get filtered.
- Added a regression test at the `Tcode.model_validate(...)` layer covering an unresolved `method: value` subject encoding, matching the build-time validation path reported in the field.

## 7.4.12 - 2026-06-29

### Changes

- Expanded the placeholder-term filter regex in `distinct()` (`fullmap.py`) to drop additional non-informative terms during entity resolution. The `bad` pattern now also excludes `not applicable`, `p value`, `variable`, `result`, `exposure`, `expression`, and `symbol` alongside the existing `none`, `nan`, `na`, `null`, and `unknown`, preventing these generic column-header-like values from being sent through resolution and producing spurious CURIE mappings.
