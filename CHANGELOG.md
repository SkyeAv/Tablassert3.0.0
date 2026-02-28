# Changelog

All notable changes to this project are documented in this file.

## 6.2.0 - 2026-02-27

### New Features
- Added `tablassert-cli verify-table-configuration-syntax <table-config.yaml>` for fast TC3 schema validation without running a full graph build.
- Added rich progress bars across pipeline stages to improve runtime visibility during large graph builds.
- Added automated Docker publishing in CI so container images are built and distributed from the docs workflow.
- Added improved progress messaging and stage-level status updates for entity mapping and build orchestration.

### Changes
- Updated the CLI interface for graph builds from `tablassert-cli -i <graph-config.yaml>` to `tablassert-cli build-knowledge-graph <graph-config.yaml>`.
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
