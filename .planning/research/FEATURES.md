# Feature Research

**Domain:** Python package release automation (UV + PyPI via GitHub Actions)
**Researched:** 2026-03-17
**Confidence:** HIGH

## Feature Landscape

### Table Stakes (Users Expect These)

Features maintainers assume exist in a modern Python release pipeline. Missing these makes the workflow feel unsafe or unreliable.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Tag/release-triggered publish flow | Python package releases are expected to publish from immutable tags, not ad-hoc manual commands | LOW | Trigger on `push` tags (`v*`) or `release.published`; prevents accidental publishes from branch pushes |
| Reproducible build artifacts (`sdist` + `wheel`) with UV | PyPI consumers expect both source and wheel dists; maintainers expect one canonical build command | LOW | Use `uv build`; output to `dist/`; artifact should be the exact payload later uploaded |
| Pre-publish validation gate | Teams expect tests/checks before registry publication | MEDIUM | At minimum run smoke tests against built wheel and sdist before publish job |
| Trusted publishing (OIDC) to PyPI | API-tokenless publishing is now the recommended security baseline | MEDIUM | Require GitHub job `permissions.id-token: write` + PyPI trusted publisher mapping; avoid long-lived secrets |
| Environment-gated production publish | Maintainers expect explicit approval/rules before public release | MEDIUM | Use GitHub Environment (`pypi`) with protection rules/reviewers; aligns with PyPA guidance for security |
| Artifact handoff between jobs | Build-once, publish-later separation is standard for auditable CI/CD | LOW | Store `dist/` as workflow artifact; publish job only uploads downloaded artifacts |
| Deterministic action/tool pinning | Supply-chain hardening now expects pinned versions | MEDIUM | Pin `astral-sh/setup-uv` and key actions by stable tag/major policy; pin UV version where feasible |
| Roll-forward + yank rollback playbook | PyPI releases are immutable; teams expect operational recovery guidance | MEDIUM | You cannot overwrite filenames/version uploads; rollback is usually: cut fixed version, optionally yank broken release with reason |

### Differentiators (Competitive Advantage)

Capabilities that move this from "works" to "excellent maintainer experience".

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Dual-lane publish (TestPyPI continuous, PyPI on tagged release) | Catches packaging issues early while keeping production releases intentional | MEDIUM | Publish to TestPyPI on main merges; publish to PyPI on signed/tagged release path |
| Automatic smoke-install verification from published index | Validates real installability, not just local build success | MEDIUM | After publish, `uv run --with <pkg> --no-project` smoke import/CLI check from index |
| Provenance attestations in release pipeline | Improves downstream trust and supports supply-chain verification | LOW | `pypa/gh-action-pypi-publish` generates PEP 740 attestations by default (v1.11+) |
| Concurrency + idempotency controls | Prevents duplicate or racing publishes from multiple events | LOW | Add workflow `concurrency` group by workflow+ref; cancel in-progress for same ref |
| Release metadata automation | Reduces human error and improves release discoverability | MEDIUM | Auto-generate GitHub release notes/changelog from conventional commits or labels |
| Fast-path CI performance tuning for UV | Keeps release latency low without compromising correctness | LOW | Enable UV cache in `setup-uv`; prune cache in CI (`uv cache prune --ci`) |

### Anti-Features (Commonly Requested, Often Problematic)

Features that sound helpful but usually hurt reliability, security, or maintainability.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| "One job does build + publish directly" | Simpler YAML and fewer steps | Removes artifact provenance boundary; harder to audit/retry safely | Keep separate `build` and `publish` jobs with artifact transfer |
| Long-lived PyPI API tokens in repo secrets by default | Familiar setup from older guides | Secret leakage blast radius is high; token rotation burden | Use Trusted Publishing (OIDC); keep tokens only as break-glass fallback |
| Auto-delete/re-upload same version on failure | Desire for quick rollback/fix | PyPI forbids filename reuse; deletes are irreversible and risky | Bump version (patch/post), republish, and yank broken release |
| Publish on every push to main to real PyPI | Fast feedback desire | Pollutes production index, increases bad-release probability | Publish every push to TestPyPI; reserve PyPI for tags/releases |
| Matrix publish from multiple jobs to same version | Parallelism for speed | Race conditions and duplicate upload failures | Build/test in matrix, but publish once from a single release job |

## Feature Dependencies

```text
Trusted publishing (OIDC)
    └──requires──> GitHub Environment configuration (pypi/testpypi)
    └──requires──> Job permissions: id-token: write

Safe PyPI publish
    └──requires──> Build artifacts (uv build)
                        └──requires──> Tagged release trigger

Rollback readiness
    └──requires──> Versioning discipline (no filename reuse)
    └──requires──> Yank procedure + release notes

TestPyPI continuous lane ──enhances──> PyPI tagged lane
Concurrency controls ──protects──> Single publish job
```

### Dependency Notes

- **Trusted publishing requires environment + OIDC permissions:** without both, PyPI cannot mint short-lived publish credentials.
- **Safe publish requires built artifacts first:** build once, then publish exactly those files to avoid drift.
- **Rollback readiness requires versioning discipline:** because PyPI disallows filename reuse, recovery is roll-forward plus optional yank.
- **TestPyPI lane enhances PyPI lane:** catches packaging/upload issues before production releases.

## MVP Definition

### Launch With (v1)

Minimum needed for a robust first release workflow.

- [ ] Tagged release trigger (`v*` or `release.published`) with single publish job
- [ ] `uv build` for wheel + sdist, with smoke test before publish
- [ ] Trusted Publishing to PyPI (`id-token: write`, `pypi` environment protection)
- [ ] Artifact-based separation (`build` job uploads, `publish` job downloads/uploads)
- [ ] Documented rollback runbook: bump version + yank broken release when needed

### Add After Validation (v1.x)

- [ ] TestPyPI continuous lane (main branch) once PyPI lane is stable
- [ ] Concurrency/idempotency guards to eliminate duplicate publish races
- [ ] Release-note automation and post-publish install smoke from index

### Future Consideration (v2+)

- [ ] Signed git tags + stronger provenance policy enforcement (org-level)
- [ ] Advanced policy checks (SBOM verification, stricter attestations gating)

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Trusted Publishing + environment protection | HIGH | MEDIUM | P1 |
| Build/test/publish split with artifacts | HIGH | MEDIUM | P1 |
| Rollback runbook (roll-forward + yank) | HIGH | LOW | P1 |
| TestPyPI continuous lane | MEDIUM | MEDIUM | P2 |
| Concurrency/idempotency controls | MEDIUM | LOW | P2 |
| Release metadata automation | MEDIUM | MEDIUM | P3 |

**Priority key:**
- P1: Must have for launch
- P2: Should have, add when possible
- P3: Nice to have, future consideration

## Competitor Feature Analysis

| Feature | PyPA reference workflow | UV trusted-publishing example | Our Approach |
|---------|--------------------------|-------------------------------|--------------|
| Auth model | Strongly recommends Trusted Publishing | Uses Trusted Publishing with no static credentials | Adopt OIDC-first; token fallback only for break-glass |
| Build + publish shape | Build job + artifact transfer + publish job(s) | Single flow with UV build/smoke/publish | Keep build/publish boundary, use UV commands end-to-end |
| Safety checks | Recommends environment protections/manual approval for `pypi` | Includes wheel+sdist smoke tests | Make smoke tests and rollback runbook mandatory in v1 |

## Sources

- UV GitHub Actions integration guide (official, updated 2026-03-06): https://docs.astral.sh/uv/guides/integration/github/ (HIGH)
- Python Packaging User Guide, GitHub Actions publishing workflow (official, updated 2026-03-17): https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/ (HIGH)
- PyPI Trusted Publishers docs (official): https://docs.pypi.org/trusted-publishers/ (HIGH)
- GitHub Actions OIDC permissions docs (official): https://docs.github.com/en/actions/reference/security/oidc (HIGH)
- GitHub Actions workflow syntax (official, concurrency): https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions (HIGH)
- PyPI Yanking docs (official): https://docs.pypi.org/project-management/yanking/ (HIGH)
- PyPI Help: filename reuse and deletion behavior (official): https://pypi.org/help/#file-name-reuse (HIGH)

---
*Feature research for: Tablassert release automation*
*Researched: 2026-03-17*
