# Changelog

The canonical release history lives in the repository root at [`CHANGELOG.md`](https://github.com/SkyeAv/Tablassert/blob/main/CHANGELOG.md).

## Current Release Notes

## 7.4.0 - 2026-05-05

### Changes

- Renamed CLI commands: `build-knowledge-graph` → `build`, `verify-table-configuration-syntax` → `validate`. Version now shown via `--version` flag.
- Added `qc` parameter to `resolve_many()` for optional QC auditing. ONNX Runtime provider is auto-detected via `get_qc_provider()`.
- Added `has_qc_runtime()` helper to `qc.py` and `empty_matches()` helper to `fullmap.py`.
- Rewrote `downloader.py` with `DownloadReceipt`, exception classes, and file validation.

### Bug Fixes

- Fixed tutorial table configuration using header names instead of Excel column letters for encodings.

### Documentation

- Updated all documentation for renamed CLI commands.
- Fixed tutorial and example YAML configurations to use Excel column letter references.
- Updated `resolve_many()` API reference with new `qc` parameter and auto-detected QC provider.
- Fixed CITATION.cff version and CONTRIBUTING.md package list.

For older releases and the full project history, open the root `CHANGELOG.md` in the repository.
