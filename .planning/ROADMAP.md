# Roadmap: Tablassert Release Automation

## Overview

This roadmap delivers a secure, reproducible GitHub Actions release path for the Tablassert CLI: first enforce correct release triggers and deterministic UV builds, then validate and preserve artifacts, then publish through protected PyPI trusted publishing, and finally ensure maintainers can recover safely from bad releases.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [ ] **Phase 1: Release Preconditions and Deterministic Build** - Production releases only trigger with verified tag/version alignment and UV-built distribution outputs.
- [ ] **Phase 2: Artifact Validation and Run Reliability** - Built artifacts are validated, preserved for downstream jobs, and protected from duplicate publish races.
- [ ] **Phase 3: Protected PyPI Publication** - Only validated artifacts are published to PyPI through secure, environment-gated credentials.
- [ ] **Phase 4: Release Recovery Playbook** - Maintainers can follow documented rollback/mitigation steps for bad production releases.

## Phase Details

### Phase 1: Release Preconditions and Deterministic Build
**Goal**: Maintainers can trigger production release runs only from intended release events, with version metadata checks and deterministic UV artifact generation in place.
**Depends on**: Nothing (first phase)
**Requirements**: TRIG-01, TRIG-02, BLD-01
**Success Criteria** (what must be TRUE):
  1. Maintainer can trigger a production release workflow only from approved release/tag events.
  2. Workflow blocks publication path when tag/release metadata does not match package version.
  3. Release run produces both wheel and sdist artifacts using UV for the tagged version.
**Plans**: TBD

### Phase 2: Artifact Validation and Run Reliability
**Goal**: Artifact integrity is proven before publish by validating build outputs, promoting immutable artifacts across jobs, and preventing racing runs for the same version.
**Depends on**: Phase 1
**Requirements**: BLD-02, BLD-03, OPS-01
**Success Criteria** (what must be TRUE):
  1. Maintainer can see build artifacts preserved and transferred unchanged between workflow jobs in the same run.
  2. Workflow fails before publish when artifact metadata/checks are invalid.
  3. Starting duplicate release runs for the same version does not result in multiple competing publish attempts.
**Plans**: TBD

### Phase 3: Protected PyPI Publication
**Goal**: Production publish is a tightly scoped, secure step that uploads only previously validated artifacts through GitHub-protected controls.
**Depends on**: Phase 2
**Requirements**: PUB-01, PUB-02, PUB-03
**Success Criteria** (what must be TRUE):
  1. Publish job uploads only artifacts produced by the validated build/verify jobs from the same workflow run.
  2. Maintainer can complete publish without repository-stored static PyPI secrets in workflow files.
  3. Production publish requires the configured GitHub environment protections before artifact upload proceeds.
**Plans**: TBD

### Phase 4: Release Recovery Playbook
**Goal**: Maintainers can quickly mitigate bad releases with clear, repeatable rollback guidance tailored to PyPI release constraints.
**Depends on**: Phase 3
**Requirements**: OPS-02
**Success Criteria** (what must be TRUE):
  1. Maintainer can find and follow documented mitigation steps when a bad release reaches PyPI.
  2. Maintainer can execute the documented recovery path (for example yank + corrected release) without ad hoc decision-making.
**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Release Preconditions and Deterministic Build | 0/TBD | Not started | - |
| 2. Artifact Validation and Run Reliability | 0/TBD | Not started | - |
| 3. Protected PyPI Publication | 0/TBD | Not started | - |
| 4. Release Recovery Playbook | 0/TBD | Not started | - |
