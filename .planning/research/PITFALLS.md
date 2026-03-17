# Domain Pitfalls

**Domain:** Python release automation with uv + GitHub Actions + PyPI
**Researched:** 2026-03-17
**Confidence:** HIGH

## Critical Pitfalls

### Pitfall 1: Build and publish in the same privileged job

**What goes wrong:**
Teams run `uv build`, tests, and `gh-action-pypi-publish` in one job with PyPI publish permissions, so any compromised build/test step can exfiltrate publish capability or ship tampered artifacts.

**Why it happens:**
Single-job workflows are faster to write and look simpler in demos.

**Prevention strategy (with CI guardrails):**
- Split workflow into at least two jobs: `build` (no `id-token: write`) and `publish` (only download artifacts + publish).
- In publish job, set minimal permissions (`id-token: write`, `contents: read`) and nothing broader.
- Pass artifacts via `actions/upload-artifact` and `actions/download-artifact`; publish only downloaded artifacts.
- Add branch/tag condition so publish only runs on release tags.

**Warning signs:**
- Workflow has one job that both executes arbitrary project commands and publishes.
- `permissions` block is global at workflow level or includes broad scopes in non-publish jobs.
- No `needs: build` dependency before publishing.

**Phase mapping:**
Phase 2 (CI pipeline design) and Phase 3 (secure publish hardening).

---

### Pitfall 2: Trusted Publisher mismatch (environment/workflow/repo fields drift)

**What goes wrong:**
PyPI trusted publishing fails at release time because the configured trusted publisher does not exactly match GitHub repo, workflow filename, or environment name.

**Why it happens:**
Teams rename workflow files/environments after initial setup, or copy config from another repo.

**Prevention strategy (with CI guardrails):**
- Standardize environment names (`pypi`, optional `testpypi`) and freeze workflow filename for publishing.
- Add a release dry-run workflow against TestPyPI before production PyPI.
- Add a pre-release checklist item that compares PyPI trusted publisher fields vs repo settings.
- Keep `id-token: write` scoped only to publish job.

**Warning signs:**
- Release job fails with OIDC/trusted publisher authorization errors.
- Workflow recently renamed or moved.
- Environment in workflow differs from PyPI publisher config.

**Phase mapping:**
Phase 3 (trusted publishing setup) with verification in Phase 4 (release readiness).

---

### Pitfall 3: Unpinned action and tool versions break reproducibility

**What goes wrong:**
A previously green release pipeline starts failing (or behavior changes) because `setup-uv`, `checkout`, or publish actions float to new behavior.

**Why it happens:**
Using moving refs (`@master`, unstable refs, or implicit latest uv versions) seems convenient early.

**Prevention strategy (with CI guardrails):**
- Pin GitHub Actions to stable major or exact tags/SHAs (especially publish action and setup action).
- Pin uv version in `astral-sh/setup-uv` (or version file policy) and review upgrades intentionally.
- Add a scheduled CI smoke job for dependency/action bump detection before release day.

**Warning signs:**
- No explicit `version` for uv setup.
- Action refs include `master`, branch names, or unstable refs.
- Release fails without project code changes.

**Phase mapping:**
Phase 2 (pipeline implementation) and Phase 4 (maintenance hardening).

---

### Pitfall 4: Lockfile and dependency state drift between local and CI

**What goes wrong:**
Maintainers validate locally, but CI resolves differently and build/test/publish behavior diverges from expected output.

**Why it happens:**
`uv.lock` is not enforced in CI, or dependency changes are merged without lock validation.

**Prevention strategy (with CI guardrails):**
- Require `uv lock --check` in CI.
- Use `uv sync --locked` for test/build jobs.
- Tie cache invalidation to `uv.lock` and `pyproject.toml` to avoid stale dependency reuse.
- Block merge if lockfile check fails.

**Warning signs:**
- CI resolves new dependencies on release branch unexpectedly.
- PRs modify `pyproject.toml` without lockfile updates.
- “Works locally, fails in CI” due to dependency version differences.

**Phase mapping:**
Phase 1 (packaging baseline) and Phase 2 (CI validation).

---

### Pitfall 5: Missing artifact integrity checks before upload

**What goes wrong:**
The package uploads successfully, but wheel/sdist is broken (missing files, bad entry point, unusable install).

**Why it happens:**
Teams treat successful `uv build` as sufficient and skip installation/smoke tests on built artifacts.

**Prevention strategy (with CI guardrails):**
- Add artifact smoke tests for both wheel and sdist using isolated commands.
- Run metadata validation (`twine check` equivalent behavior or publish action metadata verification default).
- Fail publish unless smoke tests pass for produced artifacts.

**Warning signs:**
- Release passes CI but users report import/CLI failures immediately.
- Dist contents differ from expected package layout.
- No CI step executes installed artifact from `dist/`.

**Phase mapping:**
Phase 2 (artifact validation pipeline).

---

### Pitfall 6: Publishing partial artifacts from matrix jobs

**What goes wrong:**
Some wheels upload while others fail, resulting in incomplete releases where pip resolution falls back unpredictably to sdist for some platforms.

**Why it happens:**
Teams publish directly from each matrix job or allow unsynchronized parallel uploads.

**Prevention strategy (with CI guardrails):**
- Build per-platform artifacts in matrix jobs, upload all as artifacts, then publish atomically in a single publish job.
- Gate publish with `needs` on all required build/test jobs.
- Keep `skip-existing` disabled for production PyPI so duplicates/race conditions fail loudly.

**Warning signs:**
- Multiple publish steps run in parallel for one release tag.
- PyPI release page shows only a subset of expected files.
- Users on specific platforms build from source unexpectedly.

**Phase mapping:**
Phase 2 (job orchestration) and Phase 3 (publish policy).

---

### Pitfall 7: Misusing unsupported publish contexts (reusable/composite/container)

**What goes wrong:**
Trusted publishing fails or behaves inconsistently when `gh-action-pypi-publish` is invoked from unsupported patterns (notably reusable workflow trusted publishing, composite action wrappers, or job-level containers).

**Why it happens:**
Teams over-abstract CI templates without checking action support boundaries.

**Prevention strategy (with CI guardrails):**
- Keep the actual PyPI publish job in a top-level workflow on GitHub-hosted Ubuntu runner.
- If reusing logic, reuse build/test jobs; keep publish as explicit final job.
- Add CI lint/policy check to reject publish job definitions that use `container:` or invoke through composite wrappers.

**Warning signs:**
- Intermittent OIDC/token exchange errors only in specific workflow topology.
- Publish job runs in custom container or deeply wrapped reusable workflow.
- Team cannot reproduce failures in minimal top-level publish workflow.

**Phase mapping:**
Phase 3 (trusted publish implementation).

---

### Pitfall 8: Environment protection is missing for production publish

**What goes wrong:**
Any pushed tag can trigger production release, increasing accidental or malicious release risk.

**Why it happens:**
Teams configure trusted publishing but skip GitHub Environment protection rules.

**Prevention strategy (with CI guardrails):**
- Use a dedicated `pypi` environment with required reviewers/manual approval.
- Restrict tag pattern and who can create release tags.
- Add release provenance/audit checks (attestations default in recent publish action trusted flow).

**Warning signs:**
- No approval gate before production PyPI upload.
- Maintainers can publish from ad-hoc tags without peer review.
- Release audit trail lacks environment approval events.

**Phase mapping:**
Phase 3 (security controls) and Phase 4 (governance hardening).

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|----------------|------------|
| Phase 1 - Packaging baseline | Lockfile drift, missing metadata/build config | Enforce `uv lock --check`; verify `pyproject.toml` build metadata early |
| Phase 2 - CI build and validation | Single job with privileges, missing artifact smoke tests, partial matrix uploads | Split build/publish jobs, test wheel+sdist artifacts, publish atomically from one job |
| Phase 3 - Trusted PyPI publishing | OIDC misconfig, unsupported workflow topology, over-broad permissions | Top-level publish job, exact trusted publisher mapping, job-scoped `id-token: write` |
| Phase 4 - Release hardening | Unpinned actions/tools, weak environment protections | Pin action/tool versions, require protected `pypi` environment approvals |

## CI Guardrail Checklist

- [ ] Build and publish are separate jobs; publish job only downloads artifacts and uploads.
- [ ] Only publish job has `id-token: write`; no global broad permissions.
- [ ] `uv lock --check` and `uv sync --locked` are enforced in CI.
- [ ] Artifact smoke tests run against both wheel and sdist before publish.
- [ ] Publish action and setup actions are pinned to stable version/sha policy.
- [ ] `pypi` environment exists with protection/approval rules.
- [ ] Publish trigger is tag-gated and controlled.

## Sources

- https://docs.astral.sh/uv/guides/integration/github/ (official uv docs, updated Mar 2026) — HIGH
- https://github.com/astral-sh/setup-uv (official action docs via Context7 and upstream README) — HIGH
- https://docs.pypi.org/trusted-publishers/using-a-publisher/ (official PyPI trusted publishing) — HIGH
- https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/ (PyPA guide, updated Mar 2026) — HIGH
- https://github.com/pypa/gh-action-pypi-publish (official action README/non-goals and supported patterns) — HIGH
