---
phase: quick-1-please-add-a-github-action-that-runs-uv-
plan: 1
type: execute
wave: 1
depends_on: []
files_modified:
  - .github/workflows/release-pypi.yml
autonomous: true
requirements:
  - BLD-01
  - PUB-01
  - PUB-02
must_haves:
  truths:
    - "Maintainer can run a release workflow that builds wheel and sdist with UV."
    - "Built artifacts are the exact inputs used by the publish job."
    - "PyPI upload is performed by GitHub Actions without hardcoded credentials."
  artifacts:
    - path: ".github/workflows/release-pypi.yml"
      provides: "Release workflow with build and publish jobs"
      contains: "uv build, upload/download-artifact, pypa/gh-action-pypi-publish"
  key_links:
    - from: "build job"
      to: "publish job"
      via: "actions/upload-artifact -> actions/download-artifact"
      pattern: "dist/ artifacts"
    - from: "release tag"
      to: "pyproject version"
      via: "workflow validation step"
      pattern: "tag equals project.version"
---

<objective>
Create a single GitHub Actions release workflow that builds distributions with UV and publishes those artifacts to PyPI.

Purpose: Remove manual packaging/publishing drift and make tagged releases reproducible and secure.
Output: `.github/workflows/release-pypi.yml` with guarded release trigger, UV build, artifact handoff, and PyPI publish.
</objective>

<execution_context>
@/home/skyeav/.config/opencode/get-shit-done/workflows/execute-plan.md
@/home/skyeav/.config/opencode/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/PROJECT.md
@.planning/ROADMAP.md
@pyproject.toml
@.github/workflows/docs.yml
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create UV release build workflow scaffold</name>
  <files>.github/workflows/release-pypi.yml</files>
  <action>Create a new workflow triggered by `release.published` (and optional `workflow_dispatch` for maintainers). Add a `build` job on ubuntu-latest that checks out code, installs UV via `astral-sh/setup-uv`, validates that release tag (strip leading `v`) matches `project.version` in `pyproject.toml`, then runs `uv build` to produce wheel and sdist. Upload `dist/*` as a named artifact for downstream jobs. Do not embed PyPI credentials or tokens in workflow code.</action>
  <verify>
    <automated>uv run python -c "import pathlib, yaml; yaml.safe_load(pathlib.Path('.github/workflows/release-pypi.yml').read_text()); print('workflow yaml valid')"</automated>
  </verify>
  <done>Workflow file exists with build job, UV build step, tag/version guard, and artifact upload step.</done>
</task>

<task type="auto">
  <name>Task 2: Add secure publish job using built artifacts</name>
  <files>.github/workflows/release-pypi.yml</files>
  <action>Add a `publish` job that `needs: build`, has minimal permissions (`id-token: write`, `contents: read`), downloads the exact build artifact, and uploads via `pypa/gh-action-pypi-publish` (trusted publishing/OIDC path). Bind the job to a `pypi` environment for protection rules. Ensure publish only runs on successful build and never rebuilds artifacts in this job.</action>
  <verify>
    <automated>python -c "import pathlib,re; t=pathlib.Path('.github/workflows/release-pypi.yml').read_text(); assert 'needs: build' in t and 'id-token: write' in t and 'gh-action-pypi-publish' in t; print('publish wiring present')"</automated>
  </verify>
  <done>Publish job uses downloaded build artifacts and trusted publishing permissions, with no static credential usage in workflow file.</done>
</task>

</tasks>

<verification>
Run a local packaging smoke check and workflow lint-level checks before merge.
</verification>

<success_criteria>
1. Tagged release workflow builds both `.whl` and `.tar.gz` with UV.
2. Publish job consumes artifacts from the build job and uploads to PyPI via `gh-action-pypi-publish`.
3. Workflow file contains no hardcoded PyPI username/password/token secrets.
</success_criteria>

<output>
After completion, create `.planning/quick/1-please-add-a-github-action-that-runs-uv-/1-SUMMARY.md`
</output>
