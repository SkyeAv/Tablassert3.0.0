# Architecture Research

**Domain:** GitHub Actions release architecture for Python package publishing (UV -> PyPI)
**Researched:** 2026-03-17
**Confidence:** HIGH

## Standard Architecture

### System Overview

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ Trigger Layer                                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  Git Tag / GitHub Release (on: release.types=[published])                  │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
┌────────────────────────────────▼────────────────────────────────────────────┐
│ CI Build/Test Layer (no publish credentials)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  preflight job  ->  build job (uv build)  ->  verify job (artifact checks)│
│      │                   │                               │                  │
│      │                   └── upload-artifact: dist-* ───┘                  │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │ needs + success gates
┌────────────────────────────────▼────────────────────────────────────────────┐
│ Protected Publish Layer (privileged, environment-gated)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  publish job (environment: pypi, permissions: id-token: write)             │
│      └── download-artifact(dist-*) -> pypa/gh-action-pypi-publish@release/v1│
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │ OIDC trusted publishing
┌────────────────────────────────▼────────────────────────────────────────────┐
│ Registry/Output Layer                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  PyPI release files (+ attestations in trusted publishing flow)            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| Release trigger | Start release pipeline only for real publish events | `on: release: types: [published]` |
| Preflight job | Fast fail checks before expensive work | Validate tag/version alignment and lockfile state |
| Build job | Produce deterministic `sdist` + `wheel` | `astral-sh/setup-uv` + `uv build` |
| Verify job | Validate exactly what will be published | `twine check`, optional install smoke test from built wheel |
| Artifact broker | Move immutable distributions between jobs | `actions/upload-artifact@v4` / `actions/download-artifact@v5` |
| Publish job | Only job allowed to mint PyPI auth | `environment: pypi` + `id-token: write` + `pypa/gh-action-pypi-publish@release/v1` |
| Environment controls | Human/branch protection around privileged step | GitHub Environment `pypi` with required reviewers/rules |

## Recommended Project Structure

```text
.github/
└── workflows/
    ├── release.yml                 # Main release pipeline (tag/release -> publish)
    └── ci.yml                      # PR/push quality checks (no publish privileges)

scripts/
├── verify-release-version.py       # Optional: tag <-> package version guard
└── smoke-install.sh                # Optional: install/import test from built wheel

dist/                               # Ephemeral build output in workflow jobs
```

### Structure Rationale

- **`release.yml`:** Keep privileged publishing logic isolated from normal CI to minimize accidental permission creep.
- **`ci.yml`:** Reuse most checks without `id-token` or environment access.
- **`scripts/`:** Put release assertions in versioned code so gates are testable and reviewable.

## Architectural Patterns

### Pattern 1: Privilege Segmentation by Job

**What:** Build/verify jobs run with read-only token; publish job alone gets OIDC privilege.
**When to use:** Always for package publishing, especially with trusted publishing.
**Trade-offs:** Slightly more YAML complexity, much lower blast radius.

**Example:**
```yaml
permissions:
  contents: read

jobs:
  build:
    permissions:
      contents: read
  publish:
    permissions:
      contents: read
      id-token: write
```

### Pattern 2: Artifact Promotion (Build Once, Publish Once)

**What:** Build distributions one time, store as immutable artifacts, publish those exact files.
**When to use:** Any release flow where reproducibility matters.
**Trade-offs:** Extra upload/download step, but avoids rebuild drift and partial releases.

**Example:**
```yaml
- uses: actions/upload-artifact@v4
  with:
    name: dist-${{ github.event.release.tag_name }}
    path: dist/

# later in publish job
- uses: actions/download-artifact@v5
  with:
    name: dist-${{ github.event.release.tag_name }}
    path: dist/
```

### Pattern 3: Environment-Gated Publish

**What:** Bind publish job to a `pypi` environment with protection rules and optional required reviewers.
**When to use:** Any production package index publish.
**Trade-offs:** Slower release due to approval gate, significantly safer secret/token boundary.

## Data Flow

### Release Flow (Tag/Release -> PyPI)

```text
Release published event
    ↓
preflight job
    ↓ (needs success)
build job (setup-uv -> uv build)
    ↓
upload dist artifact (immutable)
    ↓
verify job (download artifact -> twine check -> smoke install)
    ↓ (all needs success)
publish job (environment: pypi approval/protection)
    ↓
OIDC token mint (id-token: write)
    ↓
pypa/gh-action-pypi-publish uploads dist/* to PyPI
```

### Permission and Trust Flow

```text
Workflow default: permissions { contents: read }
    ↓
Build/verify jobs: no id-token, no environment secrets
    ↓
Publish job: id-token: write + environment protection rules
    ↓
GitHub OIDC -> PyPI trusted publisher exchange -> short-lived upload auth
```

### Build Order and Gating Controls

1. **Trigger gate:** only `release.published` (not draft create/edit) to avoid accidental publish.
2. **Integrity gate:** `needs` chain enforces strict order: `preflight -> build -> verify -> publish`.
3. **Environment gate:** `publish` waits for environment protection checks/reviewers.
4. **Artifact gate:** publish consumes only named artifact from current run, never rebuilds.
5. **Concurrency gate:** use workflow/job `concurrency` keyed by ref/tag to prevent duplicate simultaneous publishes.

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| Single package, low cadence | Single `build` job + single `publish` job is enough |
| Multiple Python versions/platform wheels | Matrix build jobs, one aggregate verify, one publish job after all matrix jobs succeed |
| Many packages/monorepo | Split per-package build artifacts, keep one publish job per package/index target with explicit `needs` |

### Scaling Priorities

1. **First bottleneck:** Parallel wheel builds and synchronization; solve with matrix builds + one final publish gate.
2. **Second bottleneck:** Release correctness drift; solve with strict preflight checks (tag/version, lockfile, changelog policy).

## Anti-Patterns

### Anti-Pattern 1: Build and Publish in the Same Privileged Job

**What people do:** Run `uv build` and `gh-action-pypi-publish` in one job with `id-token: write`.
**Why it's wrong:** Build-time compromise can directly escalate into publish identity abuse.
**Do this instead:** Build/test in unprivileged jobs; publish only from a minimal, gated job using prebuilt artifacts.

### Anti-Pattern 2: Repository-Level Long-Lived PyPI Tokens

**What people do:** Store broad API tokens in repo secrets and use for all branches/jobs.
**Why it's wrong:** Larger secret exposure surface and harder revocation/auditing.
**Do this instead:** Prefer PyPI trusted publishing (OIDC), scoped to workflow path/repo/env.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| GitHub Actions | Workflow orchestration + job permissions + artifacts | Use workflow-level least privilege, then elevate only publish job |
| PyPI | Trusted Publisher via OIDC | Configure publisher with exact repo + workflow path (+ environment recommended) |
| uv toolchain | Build backend frontend in CI | Install via `astral-sh/setup-uv`, then run `uv build` |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `preflight/build/verify` ↔ `publish` | Artifact handoff + `needs` dependency | No direct secret or OIDC sharing across boundary |
| Release metadata ↔ package metadata | Scripted assertions | Prevent tag/version mismatch before build |

## Sources

- GitHub Actions workflow syntax (`permissions`, `jobs.<job_id>.permissions`, `environment`, `release` triggers): https://docs.github.com/en/actions/automating-your-workflow-with-github-actions/workflow-syntax-for-github-actions and https://docs.github.com/en/actions/writing-workflows/choosing-when-your-workflow-runs/events-that-trigger-workflows
- GitHub Actions OIDC permissions (`id-token: write`): https://docs.github.com/en/actions/reference/security/oidc
- GitHub Actions artifacts and immutability (`upload-artifact@v4` / `download-artifact`): https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/storing-and-sharing-data-from-a-workflow
- PyPI Trusted Publishers (GitHub Actions publisher config and usage): https://docs.pypi.org/trusted-publishers/adding-a-publisher/ and https://docs.pypi.org/trusted-publishers/using-a-publisher/
- `gh-action-pypi-publish` guidance on separation of build/publish, OIDC, Linux job constraints: https://github.com/pypa/gh-action-pypi-publish
- uv build behavior (`uv build` outputs to `dist/`): https://github.com/astral-sh/uv/blob/main/docs/concepts/projects/build.md
- `astral-sh/setup-uv` GitHub Action configuration: https://github.com/astral-sh/setup-uv

---
*Architecture research for: Tablassert release automation CI/CD*
*Researched: 2026-03-17*
