# Project Research Summary

**Project:** Tablassert Release Automation
**Domain:** Python CLI release automation (UV + GitHub Actions + PyPI)
**Researched:** 2026-03-17
**Confidence:** HIGH

## Executive Summary

This project is a secure release automation pipeline for a Python CLI package, where tagged releases are built once with UV and published to PyPI through GitHub Actions. The expert pattern is consistent across PyPA, PyPI, GitHub, and UV docs: isolate build/test from publish, use artifact promotion between jobs, and grant publish credentials only at the final gated step.

The recommended implementation is a two-lane architecture over time: ship an MVP with a protected PyPI lane first, then add a TestPyPI continuous lane for earlier packaging feedback. The MVP should use pinned action/tool versions, `uv build --no-sources`, artifact verification (metadata + smoke install), GitHub Environment protection (`pypi`), and OIDC trusted publishing via `id-token: write` in the publish job only.

The highest risks are security boundary collapse (build and publish combined), trusted publisher config drift (repo/workflow/environment mismatch), and release correctness drift (unpinned dependencies/actions or missing artifact checks). Mitigation is straightforward and should be non-negotiable in roadmap scope: strict job privilege segmentation, immutable artifact handoff, explicit environment approvals, lockfile enforcement, and release runbooks for roll-forward plus yank.

## Key Findings

### Recommended Stack

The stack is mature and strongly documented by official sources. Use GitHub Actions on `ubuntu-latest` with pinned actions, UV `0.10.x` for deterministic builds, and PyPI trusted publishing (OIDC) to avoid static secrets. Keep build and publish in separate jobs and publish only artifacts produced earlier in the same run.

**Core technologies:**
- GitHub Actions (pinned actions, least-privilege permissions): CI/CD orchestration and policy boundary.
- UV `0.10.x` (`setup-uv`, `uv build --no-sources`): canonical build path aligned with current repo tooling.
- PyPI Trusted Publishing (OIDC): tokenless publish auth with lower leakage risk and easier governance.

### Expected Features

For v1, stakeholders expect safe, reproducible, auditable releases from immutable tags, not ad hoc publish commands. Research converges on mandatory guardrails before production publish and explicit operational recovery guidance because PyPI uploads are immutable.

**Must have (table stakes):**
- Tag/release-triggered flow with build -> verify -> publish sequencing.
- Trusted publishing (`id-token: write`) with `pypi` environment protection.
- Build/publish separation via artifacts and pre-publish artifact smoke checks.
- Deterministic tool/action pinning and rollback runbook (bump + yank when needed).

**Should have (competitive):**
- Dual-lane release model: TestPyPI continuous lane + gated PyPI lane.
- Concurrency/idempotency controls to prevent duplicate/racing publishes.
- Post-publish install verification from the target index and release note automation.

**Defer (v2+):**
- Signed tag enforcement and stronger org-level provenance policy gates.
- Advanced supply-chain policy checks (for example SBOM/attestation enforcement).

### Architecture Approach

Use a layered GitHub Actions architecture: trigger gate (`release.published` or strict `v*` tags), unprivileged preflight/build/verify jobs, then a protected publish job bound to `environment: pypi` with `id-token: write`. Major pattern decisions are privilege segmentation by job, artifact promotion (build once/publish once), and environment-gated publishing. This structure is simple enough for MVP but scales to matrix wheel builds by keeping a single final publish gate.

**Major components:**
1. Trigger + preflight gates: ensure real release events and tag/version integrity.
2. Build/verify layer: produce and validate wheel/sdist artifacts with UV and smoke checks.
3. Publish layer: minimal privileged job that downloads immutable artifacts and uploads to PyPI.

### Critical Pitfalls

1. **Single privileged build+publish job** - avoid by splitting jobs and granting OIDC only in publish.
2. **Trusted publisher mapping drift** - avoid by freezing workflow/environment names and checklist validation before release.
3. **Unpinned action/tool versions** - avoid by pinning versions/SHAs and scheduling upgrade smoke checks.
4. **Lockfile/dependency drift in CI** - avoid with `uv lock --check` and `uv sync --locked` gates.
5. **Missing artifact integrity checks** - avoid with `twine check` and install/smoke validation of built artifacts.

## Implications for Roadmap

Based on combined research, suggested phase structure:

### Phase 1: Packaging Baseline and Release Preconditions
**Rationale:** Release automation is unsafe until versioning, lock discipline, and package metadata are deterministic.
**Delivers:** UV build baseline, lockfile checks, tag/version guard script, minimal release trigger contract.
**Addresses:** MVP expectations for reproducible artifacts and safe release triggering.
**Avoids:** Lockfile drift and tag/version mismatch pitfalls.

### Phase 2: Build and Verification Pipeline
**Rationale:** Artifact correctness must be proven before introducing publish privileges.
**Delivers:** Unprivileged preflight/build/verify jobs, artifact upload/download, smoke tests for wheel/sdist, concurrency controls.
**Uses:** `setup-uv`, pinned `checkout`, artifact actions, UV cache strategy.
**Implements:** Artifact promotion and strict `needs`-based ordering.
**Avoids:** Single-job privilege collapse, partial matrix/parallel publish behavior, broken artifact uploads.

### Phase 3: Trusted PyPI Publish and Security Gating
**Rationale:** Publishing should be introduced only after artifacts are validated and boundaries are proven.
**Delivers:** Dedicated publish job with `environment: pypi`, `id-token: write`, trusted publisher mapping, approval controls.
**Addresses:** OIDC-first authentication and production release governance requirements.
**Avoids:** Token misuse, trusted publisher mismatch, unsupported publish topology errors.

### Phase 4: Operational Hardening and DX Enhancements
**Rationale:** Once core flow is stable, add maintainability and confidence improvements.
**Delivers:** TestPyPI lane, post-publish index smoke checks, release-note automation, scheduled pin/update checks, formal rollback/yank playbook.
**Addresses:** Differentiators and ongoing release reliability.
**Avoids:** Release-day surprises from dependency/action drift and weak operational response.

### Phase Ordering Rationale

- Dependencies force this order: deterministic packaging and artifact verification must precede privileged publish.
- Architecture naturally groups work into unprivileged CI first, privileged publish second, hardening third.
- This sequencing directly neutralizes top risks early (privilege boundaries, mapping drift, reproducibility drift).

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 3:** Trusted publisher setup details (exact PyPI mapping fields, environment naming, release event semantics) should be validated against live repo config.
- **Phase 4:** TestPyPI dual-lane strategy and matrix expansion need project-specific decisions (cadence, versioning policy, platform targets).

Phases with standard patterns (skip research-phase):
- **Phase 1:** UV lock/build baseline and tag/version preflight are highly standardized.
- **Phase 2:** Build-once/publish-once artifact pipeline and least-privilege job segmentation are well-documented defaults.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Strong alignment across official UV, PyPA, PyPI, and GitHub docs with concrete version/pinning guidance. |
| Features | HIGH | Table-stakes and prioritization map directly to current ecosystem best practices and official guidance. |
| Architecture | HIGH | Standard reference architecture is consistent across sources and compatible with current repo constraints. |
| Pitfalls | HIGH | Risks are concrete, repeatedly documented, and include actionable CI guardrails. |

**Overall confidence:** HIGH

### Gaps to Address

- Trusted publisher operational details must be validated in the live PyPI project and repository settings before first production publish.
- Release trigger policy needs a final decision (`release.published` vs `push` tag) to avoid duplicate/inconsistent trigger paths.
- If native extensions are introduced later, matrix wheel strategy and atomic artifact aggregation need a follow-up architecture increment.

## Sources

### Primary (HIGH confidence)
- `.planning/research/STACK.md` - stack, version pinning, trusted publishing baseline.
- `.planning/research/FEATURES.md` - feature priorities, MVP vs v1.x/v2 scope, dependency map.
- `.planning/research/ARCHITECTURE.md` - layered workflow architecture, component boundaries, gating patterns.
- `.planning/research/PITFALLS.md` - critical failure modes, phase warnings, CI guardrails.
- https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/ - reference release workflow patterns.
- https://docs.pypi.org/trusted-publishers/ - trusted publishing model and requirements.
- https://docs.github.com/en/actions/reference/security/oidc - OIDC permission model.
- https://docs.astral.sh/uv/guides/integration/github/ - UV GitHub Actions integration patterns.

### Secondary (MEDIUM confidence)
- https://github.com/pypa/gh-action-pypi-publish - implementation constraints and security notes.
- https://github.com/astral-sh/setup-uv - setup action behavior, caching, and version pinning usage.

### Tertiary (LOW confidence)
- None identified in current research set.

---
*Research completed: 2026-03-17*
*Ready for roadmap: yes*
