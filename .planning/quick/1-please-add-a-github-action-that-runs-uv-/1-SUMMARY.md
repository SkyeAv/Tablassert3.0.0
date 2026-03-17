---
phase: quick-1-please-add-a-github-action-that-runs-uv-
plan: 1
subsystem: infra
tags: [github-actions, uv, pypi, oidc, release]
requires: []
provides:
  - Release workflow building wheel and sdist with uv
  - Artifact handoff from build to publish job
  - Trusted publishing path to PyPI with OIDC permissions
affects: [release, packaging, publishing]
tech-stack:
  added: []
  patterns: [release-tag-version-guard, artifact-promotion, trusted-publishing]
key-files:
  created: [.planning/quick/1-please-add-a-github-action-that-runs-uv-/1-SUMMARY.md]
  modified: [.github/workflows/release-pypi.yml]
key-decisions:
  - "Use release artifact promotion (upload/download-artifact) so publish never rebuilds outputs"
  - "Use PyPI trusted publishing (id-token + pypi environment) instead of static credentials"
patterns-established:
  - "Release workflow validates tag-to-version parity before building artifacts"
  - "Publish job consumes build artifacts via needs + download-artifact"
requirements-completed: [BLD-01, PUB-01, PUB-02]
duration: 1m
completed: 2026-03-17
---

# Phase [quick-1] Plan [1]: Release Workflow Summary

**GitHub Actions now builds Tablassert wheel/sdist with uv and publishes the exact built artifacts to PyPI using OIDC trusted publishing.**

## Performance

- **Duration:** 1m
- **Started:** 2026-03-17T14:59:57Z
- **Completed:** 2026-03-17T15:01:11Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Added `.github/workflows/release-pypi.yml` with `release.published` and optional `workflow_dispatch` triggers.
- Implemented build job with `astral-sh/setup-uv`, tag/version validation against `pyproject.toml`, `uv build`, and artifact upload.
- Added publish job with `needs: build`, `id-token: write`, `contents: read`, `environment: pypi`, artifact download, and `pypa/gh-action-pypi-publish`.

## Task Commits

1. **Task 1: Create UV release build workflow scaffold** - `77fba59` (feat)
2. **Task 2: Add secure publish job using built artifacts** - `d4fbfec` (feat)

## Files Created/Modified
- `.github/workflows/release-pypi.yml` - Release workflow with uv build, guarded tag/version check, artifact handoff, and trusted PyPI publish.
- `.planning/quick/1-please-add-a-github-action-that-runs-uv-/1-SUMMARY.md` - Execution summary and task traceability.

## Decisions Made
- Use a manual `workflow_dispatch` input for `release_tag` so maintainers can rerun releases with explicit version context.
- Keep build and publish strictly separated with artifact promotion to guarantee published files are exact build outputs.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

Configure GitHub environment `pypi` and PyPI trusted publisher mapping before first production release.

## Next Phase Readiness

- Release workflow is ready for repository-level environment/protection configuration and first dry-run tag release.
- Trusted publisher mapping in PyPI remains the only external dependency called out in project state.

## Self-Check: PASSED

- FOUND: `.planning/quick/1-please-add-a-github-action-that-runs-uv-/1-SUMMARY.md`
- FOUND: `77fba59`
- FOUND: `d4fbfec`
