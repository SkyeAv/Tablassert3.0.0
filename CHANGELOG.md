# Changelog

All notable changes to this project are documented in this file.

## 7.3.6 - 2026-04-29

### Documentation
- Documented that `publication` must start with `PMC` followed by digits when `repo` is `"PMC"`.

## 7.3.5 - 2026-04-29

### Documentation
- Tightened the table-configuration reference so field requirements, defaults, accepted enum values, row indexing, and column-reference examples match the strict `Section` schema and section-merging behavior implemented in `models.py`, `ingests.py`, and the runtime loader.

## 7.3.4 - 2026-04-28

### Bug Fixes
- Fixed `downloader.from_url()` failing on URLs that trigger an immediate download. The Playwright session now opens a browser context with `accept_downloads=True`, wraps `page.goto()` inside `page.expect_download()`, and tolerates the expected `net::ERR_ABORTED` navigation error that fires when the response is a download rather than a page.

### Documentation
- Documented `miscellaneous notes` as a freetext catch-all annotation in the table configuration and advanced-example pages — used for assay caveats, non-standard units, and qualitative observations that don't map cleanly to a structured field. Supports both `method: value` (constant) and `method: column` (per-row).
- Documented Polars regex constraints for the `regex` and `remove` transforms: patterns are passed to Polars `str.replace_all()` (Rust `regex` crate), so capturing groups (`(...)` / `\1`) and lookarounds (`(?=...)`, `(?<=...)`, `(?!...)`, `(?<!...)`) are not supported and will raise at parse time. Chain simple substitutions instead, or capture residual context in a `miscellaneous notes` annotation.

## 7.3.3 - 2026-04-08

### Bug Fixes
- Changed datassert shard count to 10 (`SHARDS` constant in `fullmap.py`) to correspond to the current datassert database layout.

### Documentation
- Updated shard count references across documentation and examples to reflect the current 10-shard datassert layout.
- Corrected provenance examples so `repo` carries the namespace prefix and `publication` carries the repository-local identifier.

## 7.3.2 - 2026-04-03

### Maintenance
- Updated dependencies. No API changes.

## 7.3.1 - 2026-04-03

### Changes
- Changed `resolve_many()` return type from `dict[str, list[str]]` to `list[dict[str, Any]]` — each resolved entity is now a row dictionary, produced via `to_dicts()`.
- `resolve_many()` now preserves the original input text in an `original {col}` key on each result row.

### Documentation
- Updated `resolve_many()` API reference to match the current function signature, return type, and output format.

## 7.3.0 - 2026-04-03

### New Features
- Added `resolve_many()` to `lib` module — a standalone batch entity resolution function that resolves an iterable of text strings to CURIEs without requiring manual LazyFrame setup, NLP preprocessing, or DuckDB connection management.

### Documentation
- Added detailed API reference page for `resolve_many()` covering function signature, parameters, return value, usage examples, and integration notes.

## 7.2.2 - 2026-04-01

### Bug Fixes
- Fixed Docker publish workflow failing due to mixed-case repository owner in image tags. Hardcoded lowercase `ghcr.io/skyeav/tablassert` and switched trigger to run after autotag completion.

### Maintenance
- Updated PyPI short description.

## 7.2.1 - 2026-04-01

### Maintenance
- Improved PyPI trove classifiers. No API changes.

## 7.2.0 - 2026-03-31

### New Features
- Added `tablassert version` command to display current package version.
- Added autotag GitHub Action for automated version tagging on releases.
- Added PyPI publishing GitHub Action.
- Added Docker image publishing to GitHub Container Registry (ghcr.io).

### Changes
- Sharded datassert entity-resolution database into 16 DuckDB shards for parallel querying.
- Renamed dependency from DBssert to DATASSERT throughout.
- Separated CLI logic into dedicated `cli.py` module.
- Extracted NLP normalization into dedicated `nlp.py` module for cleaner separation of concerns.
- Implemented improved parallelization model for graph compilation.
- Annotated Pydantic model fields with `Field(...)` schema metadata.
- Renamed `fullmap.version4()` to `fullmap.resolve()` for clarity.
- Updated `fullmap` ranking to prioritize case-insensitive exact matches between normalized terms and preferred names.
- Updated `fullmap` term de-duplication to keep first occurrences, improving deterministic output ordering.
- Moved MkDocs to dev-only dependencies.

### Testing
- Added basic pytest suite covering core models, enums, ingests, lib, nlp, and utils.

### Maintenance
- Improved `.gitignore` to exclude common artifacts.

## 7.0.2 - 2026-03-23

### Changes
- Updated package metadata for the 7.0.2 release.
- Added optional `log` and `column_context` controls to `fullmap.resolve()` for more configurable entity-resolution behavior.

### Bug Fixes
- Reworked entity-resolution querying to register terms directly in DuckDB instead of writing temporary parquet files, removing tempfile lifecycle issues in `fullmap` query execution.
- Isolated unmatched-entity logging into a dedicated helper and gated it behind an explicit logging flag.

### Documentation
- Updated API reference docs to match the current `resolve()` function signature and behavior.
- Corrected QC documentation to reflect the implemented fuzzy/BERT validation pipeline.
- Fixed documentation path typos for cache/store artifact directories.

## 7.0.1 - 2026-03-17

### Documentation
- Updated installation docs to reflect `pyproject.toml` extras and added `tablassert[rtcompat]` guidance for systems without required default Polars CPU instructions.

## 7.0.0 - 2026-03-17

### New Features
- Added pre-commit hooks for code quality (ruff linting, formatting, and pyright type checking).
- Enhanced development environment with improved VSCode settings and better gitignore including direnv support.

### Changes
- Migrated dependency management from Nix to UV for improved Python toolchain integration and simpler development workflow.
- Updated GitHub Actions workflows to use UV for deployment and documentation building.
- Removed Docker installation method from documentation to align with current supported usage.
- Removed Nix-specific installation methods and dependencies from the project.
- Removed Chromium dependency as it's no longer required for the core functionality.
- Removed random callable from codebase to simplify dependencies.
- Updated directory naming conventions for better consistency throughout the project.

### Breaking Changes
- Nix is no longer supported for development and installation. Use UV-based installation instead.
- Project now requires Python 3.11+ for compatibility with UV toolchain.

### Documentation
- Completely rewrote installation documentation to reflect UV-based development environment.
- Updated CLI and configuration documentation to remove Nix-specific sections.
- Updated project README with new installation instructions.

## 6.2.1 - 2026-03-12

### Features
- Improved QC auditing with clearer stage behavior and richer failure logging context for section/config/column tracing.
- Improved entity resolution and pipeline behavior for difficult mapping cases, including additional safeguards around nulls, strings, and column-context handling.
- Added optional `pubmed_db` and `pmc_db` graph-configuration support so enrichment can be enabled only when those databases are available.

### Bug Fixes
- Fixed multiple `fullmap` correctness issues, including handling for missing taxon values and unmatched-term edge cases.
- Fixed TCode and transform-path edge cases affecting reindex/math/null-strip behavior during section compilation.
- Fixed integration issues across lazy/eager collection boundaries to reduce incorrect intermediate outputs.

### Performance
- Optimized graph compilation by skipping empty node/edge artifacts and reducing unnecessary downstream work.

### Documentation
- Corrected stale or inaccurate docs from 6.2.0 and aligned CLI, configuration, and API references with current runtime behavior.

## 6.2.0 - 2026-02-27

### New Features
- Added `tablassert verify-table-configuration-syntax <table-config.yaml>` for fast TC3 schema validation without running a full graph build.
- Added rich progress bars across pipeline stages to improve runtime visibility during large graph builds.
- Added automated Docker publishing in CI so container images are built and distributed from the docs workflow.
- Added improved progress messaging and stage-level status updates for entity mapping and build orchestration.

### Changes
- Updated the CLI interface for graph builds from `tablassert -i <graph-config.yaml>` to `tablassert build-knowledge-graph <graph-config.yaml>`.
- Swapped hashing internals to xxHash to improve throughput in high-volume processing paths.
- Updated label-rebuild startup logic so label generation begins with clearer rebuild conditions.
- Refactored AGENTS.md hierarchy into root and scoped instruction files (`docs/`, `nix/`, `lib/tablassert/`) to reduce duplication and clarify ownership.
- Revised docs and installation guidance to align with the 6.2.0 command surface and Docker workflows.

### Breaking Changes
- Graph build invocation now requires the explicit `build-knowledge-graph` subcommand; legacy direct invocation with only `-i` is no longer the primary interface.

### Bug Fixes
- Resolved assignment and small runtime issues captured in recent maintenance commits.
- Applied lint and architecture-documentation cleanup updates to reduce drift and improve maintainability.

For full commit history, run `git log --oneline` in the repository.

## 6.1.0 - Date not tagged in repository metadata

### Notes
- Baseline release prior to the 6.2.0 CLI split and verification-command additions.
