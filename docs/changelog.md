# Changelog

The canonical release history lives in the repository root at [`CHANGELOG.md`](https://github.com/SkyeAv/Tablassert/blob/main/CHANGELOG.md).

## Current Release Notes

## 7.4.1 - 2026-05-05

### Bug Fixes

- Fixed crash in the BUILDING TCODE progress display caused by `format_section_oneline()` calling `.value` on `Tcode.status`, which is a plain string under `use_enum_values=True`.

## 7.4.0 - 2026-05-05

### Changes

- Renamed CLI commands: `build-knowledge-graph` → `build`, `verify-table-configuration-syntax` → `validate`. Version now shown via `--version` flag.
- Added `log` parameter to graph configuration for controlling unmatched entity and audit logging during builds.
- Added `qc` parameter to `resolve_many()` for optional QC auditing. ONNX Runtime provider is auto-detected via `get_qc_provider()`.
- Added `has_qc_runtime()` helper to `qc.py` and `empty_matches()` helper to `fullmap.py`.
- Rewrote `downloader.py` with `DownloadReceipt`, exception classes, and file validation.
- Refactored `cli.py` to use the shared `logger` instance instead of `loguru.logger` directly.
- Unified failure log messages to a single `FAILED` prefix; unmatched entity logging now filters to NLP level 1 terms only.
- Updated progress bar and section display labels to uppercase with pipe separators.

### Bug Fixes

- Fixed tutorial table configuration using header names instead of Excel column letters for encodings.

### Documentation

- Updated all documentation for renamed CLI commands.
- Fixed tutorial and example YAML configurations to use Excel column letter references.
- Updated `resolve_many()` API reference with new `qc` parameter and auto-detected QC provider.
- Documented new `log` graph configuration parameter with example.
- Fixed CITATION.cff version and CONTRIBUTING.md package list.

For older releases and the full project history, open the root `CHANGELOG.md` in the repository.
