# Requirements: Tablassert Release Automation

**Defined:** 2026-03-17
**Core Value:** A tagged release can be built and published to PyPI reliably from GitHub without manual packaging steps.

## v1 Requirements

Requirements for initial release automation. Each maps to roadmap phases.

### Triggering and Versioning

- [ ] **TRIG-01**: Maintainer can publish only from release/tag events intended for production releases.
- [ ] **TRIG-02**: Release workflow validates that package version and release/tag metadata are consistent before publish.

### Build and Artifacts

- [x] **BLD-01**: Release workflow builds both sdist and wheel artifacts using UV.
- [ ] **BLD-02**: Build job stores immutable artifacts for downstream jobs in the same workflow run.
- [ ] **BLD-03**: Release workflow fails if artifact metadata is invalid or artifact checks fail.

### Publish and Security

- [x] **PUB-01**: Publish job uploads only artifacts produced by the validated build job.
- [x] **PUB-02**: Publish job uses PyPI trusted publishing (OIDC) or equivalent secure credentials with no hardcoded secrets in repo files.
- [ ] **PUB-03**: Publish step is gated by GitHub environment protections for production PyPI publication.

### Operations and Reliability

- [ ] **OPS-01**: Workflow prevents duplicate/racing publish attempts for the same version.
- [ ] **OPS-02**: Maintainers have documented rollback/mitigation guidance for bad production releases.

## v2 Requirements

Deferred to future release improvements.

### Release Hardening

- **HARD-01**: Maintainer can publish continuously to TestPyPI for pre-production validation.
- **HARD-02**: Workflow performs post-publish install smoke checks from target index.
- **HARD-03**: Workflow enforces stronger provenance/attestation policy.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Docker image publishing | Explicitly paused by maintainers and not required for PyPI package release |
| Feature development unrelated to release automation | This scope is limited to CI/CD packaging and publication reliability |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| TRIG-01 | Phase 1 | Pending |
| TRIG-02 | Phase 1 | Pending |
| BLD-01 | Phase 1 | Complete |
| BLD-02 | Phase 2 | Pending |
| BLD-03 | Phase 2 | Pending |
| PUB-01 | Phase 3 | Complete |
| PUB-02 | Phase 3 | Complete |
| PUB-03 | Phase 3 | Pending |
| OPS-01 | Phase 2 | Pending |
| OPS-02 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 10 total
- Mapped to phases: 10
- Unmapped: 0

---
*Requirements defined: 2026-03-17*
*Last updated: 2026-03-17 after roadmap creation*
