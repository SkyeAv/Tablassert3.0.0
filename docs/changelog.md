# Changelog

The canonical release history lives in the repository root at [`CHANGELOG.md`](https://github.com/SkyeAv/Tablassert/blob/main/CHANGELOG.md).

## Current Release Notes

## 7.4.12 - 2026-06-29

### Changes

- Expanded the placeholder-term filter regex in `distinct()` (`fullmap.py`) to drop additional non-informative terms during entity resolution. The `bad` pattern now also excludes `not applicable`, `p value`, `variable`, `result`, `exposure`, `expression`, and `symbol` alongside the existing `none`, `nan`, `na`, `null`, and `unknown`, preventing these generic column-header-like values from being sent through resolution and producing spurious CURIE mappings.
