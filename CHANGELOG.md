# Changelog

All notable changes to this project are documented in this file.

## 7.x - 2026-03-17

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
- Project now requires Python 3.13+ for compatibility with UV toolchain.

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
