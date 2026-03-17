# Stack Research

**Domain:** Python CLI release automation (GitHub Actions + UV + PyPI)
**Researched:** 2026-03-17
**Confidence:** HIGH

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| GitHub Actions | Hosted runners (`ubuntu-latest`) + pinned actions | CI/CD orchestration for build and release jobs | This is the documented PyPA + PyPI trusted publishing path; it is the standard ecosystem default for open-source Python package release automation. |
| uv (Astral) | `0.10.x` pinned in CI | Build Python distributions with `uv build --no-sources` | Official uv docs explicitly support GitHub Actions and recommend `setup-uv`; `--no-sources` avoids uv-only source overrides leaking into release builds. |
| PyPI Trusted Publishing (OIDC) | Current PyPI trusted publisher flow | Tokenless publish authentication | Official PyPI/PyPA/GitHub guidance recommends trusted publishing over long-lived API tokens; it minimizes secret management and reduces credential leakage risk. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `astral-sh/setup-uv` | `v6` (2025 baseline) pinned to `d0d8abe699bfb85fec6de9f7adb5ae17292296ff` | Install uv and optionally cache uv artifacts | Always for UV-first projects; this is the canonical CI integration maintained by Astral. |
| `actions/setup-python` | `v5` (2025 baseline) pinned to `a26af69be951a213d495a4c3e4e4022e16d87065` | Install pinned Python version from `.python-version`/`pyproject.toml` | Use when you want faster interpreter setup on GitHub-hosted runners or strict Python version control independent of uv. |
| `pypa/gh-action-pypi-publish` | `v1.13.0` pinned to `106e0b0b7c337fa67ed433972f777c6357f78598` | Upload `dist/` artifacts to PyPI/TestPyPI via trusted publishing | Preferred publisher action for PyPI. Keep build and publish in separate jobs and only grant `id-token: write` in publish job. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `actions/checkout` | Fetch repository for build job | Use `v5` pin `93cb6efe18208431cddfb8368fd83d5badbf9bfd`; set `persist-credentials: false` in release workflows. |
| `actions/upload-artifact` + `actions/download-artifact` | Move built `dist/` files between jobs | Use `upload-artifact@v4` pin `ea165f8d65b6e75b540449e92b4886f43607fa02` and `download-artifact@v5` pin `634f93cb2916e3fdff6788551b99b062d0335ce0`. |
| GitHub Environments (`pypi`, optional `testpypi`) | Deployment policy boundary | Use environment protection rules; scope OIDC trust to workflow path + environment in PyPI Trusted Publisher settings. |

## Exact Implementation (Prescriptive)

```yaml
name: release

on:
  push:
    tags:
      - "v*"

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@93cb6efe18208431cddfb8368fd83d5badbf9bfd
        with:
          persist-credentials: false
      - uses: actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065
        with:
          python-version-file: ".python-version"
      - uses: astral-sh/setup-uv@d0d8abe699bfb85fec6de9f7adb5ae17292296ff
        with:
          version: "0.10.x"
          enable-cache: true
      - name: Build distributions
        run: uv build --no-sources
      - name: Store distributions
        uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02
        with:
          name: python-package-distributions
          path: dist/

  publish:
    needs: build
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write
    steps:
      - name: Download distributions
        uses: actions/download-artifact@634f93cb2916e3fdff6788551b99b062d0335ce0
        with:
          name: python-package-distributions
          path: dist/
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@106e0b0b7c337fa67ed433972f777c6357f78598
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| `uv build --no-sources` + `gh-action-pypi-publish` | `python -m build` + `gh-action-pypi-publish` | Use only if the project is not UV-managed. For this project, UV is a hard constraint, so this is not preferred. |
| PyPI Trusted Publishing (OIDC) | PyPI API token secret (`__token__`) | Use only when target index does not support trusted publishing or for temporary migration fallback. |
| Separate build and publish jobs | Single job build+publish | Only acceptable for throwaway/internal pipelines; not for secure release pipelines. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `pypa/gh-action-pypi-publish@master` or floating refs | Deprecated/unstable; non-reproducible and higher supply-chain risk | Pin `release/v1` to a commit SHA (or pin a specific release tag SHA) |
| Long-lived `PYPI_API_TOKEN` secrets as default | Higher credential leakage and rotation burden; no need on GitHub Actions | Trusted Publishing (`id-token: write`, no username/password inputs) |
| Publishing from the same privileged job that runs tests/build scripts | Increases blast radius for dependency/script compromise | Build in low-privilege job, publish in isolated job with only OIDC permission |
| Unpinned Python/toolchain in release workflow | Non-deterministic release artifacts over time | Pin Python source (`.python-version`) and uv version in workflow |

## Stack Patterns by Variant

**If package is pure Python:**
- Use single Linux build job (`ubuntu-latest`) and one publish job.
- Because one wheel + sdist can be built once and uploaded atomically.

**If package has compiled extensions:**
- Use matrix build (typically via `cibuildwheel`) to produce platform wheels, collect all artifacts, then publish once.
- Because PyPA warns against partial asynchronous uploads; atomic publish avoids incomplete releases.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `actions/setup-python@v5` | GitHub-hosted runners; uv `0.10.x` | Good 2025 baseline; set `python-version` or `python-version-file` explicitly. |
| `actions/upload-artifact@v4` | `actions/download-artifact@v5` | Works for job handoff; avoid deprecated `v3`. |
| `pypa/gh-action-pypi-publish@v1` | Linux publish job + `id-token: write` | Docker-based action; PyPA docs note Linux runner expectation. |

## Confidence Assessment

| Area | Level | Reason |
|------|-------|--------|
| Core stack choice | HIGH | Backed by PyPA guide + PyPI trusted publishing docs + GitHub OIDC docs + uv integration docs. |
| Exact action choices | HIGH | Action repos and official docs provide major versions and security pinning guidance. |
| UV-specific build flags | HIGH | uv docs explicitly recommend `uv build --no-sources` for publishing correctness. |

## Sources

- `/astral-sh/setup-uv` (Context7) - setup-uv usage, pinning, caching patterns
- `/actions/setup-python` (Context7) - setup-python major version usage and caching behavior
- `/websites/pypi` (Context7) - trusted publishing requirements (`id-token: write`, `gh-action-pypi-publish`)
- https://docs.astral.sh/uv/guides/integration/github/ - official uv GitHub Actions patterns (published 2026-03-06)
- https://docs.astral.sh/uv/guides/package/ - `uv build --no-sources` and publishing guidance (published 2025-11-24)
- https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/ - PyPA reference workflow and trusted publishing recommendation (updated 2026-03-17)
- https://docs.pypi.org/trusted-publishers/using-a-publisher/ - official PyPI trusted publisher workflow
- https://docs.github.com/en/actions/security-for-github-actions/security-hardening-your-deployments/configuring-openid-connect-in-pypi - GitHub OIDC with PyPI guidance
- https://github.com/pypa/gh-action-pypi-publish - official action security notes (`release/v1`, pinning advice)

---
*Stack research for: UV-based Python package build and PyPI publishing in GitHub Actions*
*Researched: 2026-03-17*
