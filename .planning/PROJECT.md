# Tablassert Release Automation

## What This Is

This project defines and automates the release workflow for the Tablassert Python CLI so releases are built consistently with UV and published to PyPI through GitHub Actions. It is for maintainers who currently validate the codebase manually and want a repeatable CI/CD path that matches the repository's real behavior.

## Core Value

A tagged release can be built and published to PyPI reliably from GitHub without manual packaging steps.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] GitHub Actions workflow builds Tablassert distribution artifacts with UV on release.
- [ ] GitHub Actions workflow publishes validated artifacts to PyPI.
- [ ] Release pipeline uses secure authentication and avoids hardcoded credentials.

### Out of Scope

- Docker image publishing — currently paused and intentionally excluded.
- Additional product features unrelated to release automation — this initialization only covers packaging and publishing flow.

## Context

The codebase already exists and is manually tested; current repository behavior is treated as source of truth for documentation and release decisions. Recent updates migrated project workflows from Nix to UV, and CLI command usage is now `tablassert`.

## Constraints

- **Tooling**: UV-based Python workflow — release build must run via UV tooling to stay consistent with repository standards.
- **Registry**: PyPI publication target — release outputs must be installable from PyPI.
- **Security**: CI secrets or trusted publishing only — publishing must not expose credentials in workflow files.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use GitHub Actions for release automation | Repository already uses GitHub Actions and this keeps release flow in existing CI/CD surface | — Pending |
| Use UV for build steps | UV is now the project's package and environment tool | — Pending |
| Publish to PyPI from CI | Removes manual release drift and supports reproducible distribution | — Pending |

---
*Last updated: 2026-03-17 after initialization*
