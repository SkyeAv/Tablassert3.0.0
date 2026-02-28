# Tablassert Documentation Update — 6.2.0

## TL;DR

> **Quick Summary**: Update all Tablassert documentation to accurately reflect the 6.2.0 API, eliminate stale 6.1.0 version strings, add Docker startup instructions for non-x86 Linux users, extend the GitHub Actions pipeline to publish multi-arch Docker images, and populate the CHANGELOG.
>
> **Deliverables**:
> - `docs/index.md` — version string updated to 6.2.0
> - `docs/configuration/graph.md` — stale 6.1.0 reference updated to 6.2.0
> - `docs/cli.md` — fully rewritten to document BOTH CLI commands
> - `docs/installation.md` — new Docker section added
> - `README.md` — Docker startup option added, quick-start modernised
> - `.github/workflows/docs.yml` — multi-arch (amd64 + arm64) Docker publish
> - `mkdocs.yml` — CHANGELOG added to nav
> - `CHANGELOG.md` — populated with major 6.2.0 changes
>
> **Estimated Effort**: Short
> **Parallel Execution**: YES — 2 waves
> **Critical Path**: Task 1 (version audit) → Task 3 (cli.md rewrite) → Task 5 (installation Docker section)

---

## Context

### Original Request
Update docs markdown files to match the 6.2.0 API; replace all 6.1.0 version strings; update README with Docker option for non-x86 Linux users; update the GitHub Action to include multi-arch Docker; add Docker to the MkDocs site; populate CHANGELOG with major changes since 6.1.0.

### Interview Summary
**Key Discussions**:
- Scope is documentation + CI workflow only — no source code changes
- Two CLI commands exist (`build-knowledge-graph`, `verify-table-configuration-syntax`) but only one is currently documented
- Docker is only built for x86_64-linux; user wants non-x86 (aarch64) covered
- CHANGELOG.md is an empty stub; populate from git history

**Research Findings**:
- `docs/index.md` contains the string "Version 6.1.0 (Beta)" — only stale version reference in content beyond graph.md
- `docs/configuration/graph.md` line 30: "Must be GC2 for version 6.1.0" — second stale reference
- `docs/cli.md` documents only `tablassert-cli -i <graph-config.yaml>` but lib.py defines two Click commands
- `.github/workflows/docs.yml` builds x86_64-linux Docker only; no aarch64 job
- Git history confirms 6.2.0 additions: rich progress bars, xxhash, verify CLI command, Docker build, label-rebuild logic

### Metis Review
**Identified Gaps** (addressed):
- CLI docs lag behind: `verify-table-configuration-syntax` was added in 6.2.0 and is not documented → Task 3 addresses this
- Docker docs assume x86; users on ARM have no documented path → Tasks 5 + 6 address this
- CHANGELOG is empty → Task 8 addresses this
- mkdocs.yml nav has no CHANGELOG entry → Task 7 addresses this
- CI publishes single-arch image; multi-arch requires manifest list approach → Task 6 addresses this

---

## Work Objectives

### Core Objective
Bring every user-facing documentation file and the CI workflow into 6.2.0 parity: accurate API references, correct version strings, Docker guidance for all supported architectures, and a populated changelog.

### Concrete Deliverables
- `docs/index.md` — "6.1.0" → "6.2.0"
- `docs/configuration/graph.md` — "version 6.1.0" → "version 6.2.0"
- `docs/cli.md` — Complete rewrite documenting both commands
- `docs/installation.md` — New "Docker" section with pull/run instructions for x86 and aarch64
- `README.md` — New "Docker" subsection under Usage
- `.github/workflows/docs.yml` — Multi-arch Docker build and push (linux/amd64 + linux/arm64 via QEMU)
- `mkdocs.yml` — Add `- Changelog: changelog.md` to nav
- `CHANGELOG.md` — Populated with 6.2.0 release notes from git history

### Definition of Done
- [ ] `grep -r "6\.1\.0" docs/ README.md CHANGELOG.md` returns no lines (except intentional historical context in changelog itself)
- [ ] `mkdocs build --strict` exits 0 with no warnings about missing nav pages
- [ ] Both CLI commands documented with synopsis, options table, and example
- [ ] Docker section present in `docs/installation.md` and `README.md`
- [ ] CHANGELOG.md has at least 3 substantive bullet points describing 6.2.0 changes
- [ ] `docs.yml` publish job builds and pushes both `linux/amd64` and `linux/arm64` manifests

### Must Have
- All "6.1.0" version strings replaced where they describe the current version (not historical context)
- `verify-table-configuration-syntax` command documented in cli.md
- Docker instructions cover both x86_64 (amd64) and aarch64 (arm64) users
- CHANGELOG.md contains real content derived from git history

### Must NOT Have (Guardrails)
- No changes to any Python source files in `lib/`
- No changes to Nix expressions in `nix/` or `flake.nix`
- No new installation methods beyond Docker and existing Nix paths (no Podman, Compose, Kubernetes)
- No historical 6.1.0 mentions in CHANGELOG removed if they are accurate comparison context
- No CI regressions: x86 publish path must remain working after multi-arch changes
- No hand-written CLI flags that diverge from actual `--help` output

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed.

### Test Decision
- **Infrastructure exists**: NO (no automated tests configured)
- **Automated tests**: None
- **Framework**: N/A — documentation-only changes

### QA Policy
Every task includes agent-executed QA scenarios. Evidence saved to `.sisyphus/evidence/`.

- **File content checks**: Bash (grep) — assert patterns present/absent
- **Build checks**: Bash (mkdocs build) — assert exit 0

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — independent atomic file edits, all parallelisable):
├── Task 1: Patch stale 6.1.0 version strings in docs/index.md and docs/configuration/graph.md [quick]
├── Task 2: Rewrite docs/cli.md to document both CLI commands [quick]
├── Task 3: Add Docker section to docs/installation.md [quick]
├── Task 4: Add Docker section to README.md [quick]
└── Task 5: Populate CHANGELOG.md from git history [writing]

Wave 2 (After Wave 1 — integration layer, all parallelisable):
├── Task 6: Update mkdocs.yml nav to include Changelog [quick]
└── Task 7: Update .github/workflows/docs.yml for multi-arch Docker [unspecified-high]

Wave FINAL (After ALL tasks):
├── Task F1: Full doc audit — grep for stale 6.1.0, verify mkdocs build [quick]
└── Task F2: Workflow YAML lint and multi-arch logic review [unspecified-high]
```

### Dependency Matrix

- **1**: None — can start immediately
- **2**: None — can start immediately
- **3**: None — can start immediately
- **4**: None — can start immediately
- **5**: None — can start immediately
- **6**: 5 (needs CHANGELOG.md to exist for nav link to be valid)
- **7**: None — CI file is independent of doc content
- **F1**: All of 1–6
- **F2**: 7

### Agent Dispatch Summary

- **Wave 1**: 5 tasks → `quick` (1–4), `writing` (5) — all parallel
- **Wave 2**: 2 tasks → `quick` (6), `unspecified-high` (7) — parallel with each other
- **Final**: 2 tasks → `quick` (F1), `unspecified-high` (F2) — parallel with each other

---

## TODOs

- [x] 1. Patch stale 6.1.0 version strings

  **What to do**:
  - Open `docs/index.md` and replace "Version 6.1.0 (Beta)" with "Version 6.2.0 (Beta)"
  - Open `docs/configuration/graph.md` and replace the phrase "Must be GC2 for version 6.1.0" with "Must be GC2 for version 6.2.0"
  - Grep both files afterwards to confirm zero remaining 6.1.0 occurrences

  **Must NOT do**:
  - Do not touch any Python source files
  - Do not alter anything else in those files beyond the targeted version strings

  **Recommended Agent Profile**:
  > Simple targeted find-and-replace, no logic needed.
  - **Category**: `quick`
  - **Skills**: none

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4, 5)
  - **Blocks**: F1
  - **Blocked By**: None

  **References**:
  - `docs/index.md:6` — line currently reads "Version 6.1.0 (Beta)"
  - `docs/configuration/graph.md:30` — line currently reads "Must be GC2 for version 6.1.0"

  **Acceptance Criteria**:
  - [ ] `grep -n "6\.1\.0" docs/index.md` → no output
  - [ ] `grep -n "6\.1\.0" docs/configuration/graph.md` → no output

  **QA Scenarios**:
  ```
  Scenario: Version strings replaced in index.md
    Tool: Bash (grep)
    Steps:
      1. Run: grep -n '6\.1\.0' docs/index.md
    Expected Result: empty output (no match)
    Evidence: .sisyphus/evidence/task-1-version-strings.txt

  Scenario: Version strings replaced in graph.md
    Tool: Bash (grep)
    Steps:
      1. Run: grep -n '6\.1\.0' docs/configuration/graph.md
    Expected Result: empty output (no match)
    Evidence: .sisyphus/evidence/task-1-graphmd-version.txt
  ```

  **Commit**: YES (group with 2, 3, 4)
  - Message: `docs(version): update stale 6.1.0 references to 6.2.0`
  - Files: `docs/index.md`, `docs/configuration/graph.md`

---

- [x] 2. Rewrite docs/cli.md to document both CLI commands

  **What to do**:
  - Completely rewrite `docs/cli.md` to document both commands present in `lib/tablassert/lib.py`:
    - **Command 1**: `tablassert-cli build-knowledge-graph` — primary pipeline command
      - Option: `-i / --ingest PATH` (required) — path to the graph configuration YAML
      - Output: two KGX NDJSON files in `storessert/`
      - Description: Runs the full extraction pipeline with rich progress bars
    - **Command 2**: `tablassert-cli verify-table-configuration-syntax` — validation command
      - Option: `-i / --ingest PATH` (required) — path to a table configuration YAML
      - Description: Validates a TC3 YAML without running the full pipeline; exits non-zero on schema errors
  - Preserve the environment variables section (CHROMIUM_PATH, PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD, etc.)
  - Use consistent markdown: command heading, synopsis code block, options table, example usage, description paragraph

  **Must NOT do**:
  - Do not invent flag names — copy exactly from the function signatures in `lib/tablassert/lib.py`
  - Do not document internal helper functions here — those belong in the API Reference pages

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: none

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3, 4, 5)
  - **Blocks**: F1
  - **Blocked By**: None

  **References**:
  - `lib/tablassert/lib.py` — contains the two `@CLI.command()` definitions with `@click.option` decorators; read for exact flags and help strings
  - `docs/cli.md` (current) — existing structure to replace; retain environment variables section

  **Acceptance Criteria**:
  - [ ] `grep -c 'build-knowledge-graph' docs/cli.md` → ≥ 1
  - [ ] `grep -c 'verify-table-configuration-syntax' docs/cli.md` → ≥ 1
  - [ ] Both commands have a synopsis block and options table
  - [ ] Environment variables section retained

  **QA Scenarios**:
  ```
  Scenario: Both commands present in cli.md
    Tool: Bash (grep)
    Steps:
      1. Run: grep -c 'build-knowledge-graph\|verify-table-configuration-syntax' docs/cli.md
    Expected Result: stdout is 2 or more
    Evidence: .sisyphus/evidence/task-2-cli-commands.txt
  ```

  **Commit**: YES (group with 1, 3, 4)
  - Message: `docs(cli): document both CLI commands for 6.2.0`
  - Files: `docs/cli.md`

---

- [x] 3. Add Docker section to docs/installation.md

  **What to do**:
  - Add a new section "### Docker" to `docs/installation.md` after the existing Nix methods
  - Content:
    - Pull and run prebuilt image (x86_64 / amd64) from GHCR
    - Pull and run prebuilt image (aarch64 / arm64) from GHCR (available after Task 7 CI update)
    - Environment variables note: same CHROMIUM_PATH / PLAYWRIGHT env vars apply in the container
    - Brief note on when Docker is preferable (not on NixOS, non-x86 systems, CI environments)
  - Read `.github/workflows/docs.yml` to get exact GHCR image names/tags

  **Must NOT do**:
  - Do not remove or reorder existing Nix installation methods
  - Do not add Podman, Compose, or Kubernetes examples

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: none

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: F1
  - **Blocked By**: None

  **References**:
  - `docs/installation.md` (current) — append after last existing section
  - `.github/workflows/docs.yml` — contains exact GHCR image names/tags; read `docker tag` lines

  **Acceptance Criteria**:
  - [ ] `grep -c 'Docker\|docker' docs/installation.md` → ≥ 3
  - [ ] Section covers both amd64 and arm64 images
  - [ ] Existing Nix sections unchanged

  **QA Scenarios**:
  ```
  Scenario: Docker section present with both arch examples
    Tool: Bash (grep)
    Steps:
      1. Run: grep -n 'amd64\|arm64\|aarch64\|x86_64' docs/installation.md
    Expected Result: at least 2 lines referencing different architectures
    Evidence: .sisyphus/evidence/task-3-docker-section.txt
  ```

  **Commit**: YES (group with 1, 2, 4)
  - Message: `docs(install): add Docker section with x86 and arm64 instructions`
  - Files: `docs/installation.md`

---

- [x] 4. Add Docker section to README.md

  **What to do**:
  - Under the existing "Usage" section in `README.md`, add "#### Docker (non-Nix / non-x86 Linux)"
  - Include a minimal Docker pull-and-run example for both amd64 and arm64 (read exact image names from `.github/workflows/docs.yml`)
  - Keep the section brief (5-10 lines) matching README's existing terse style
  - Cross-reference: "See [Installation docs](docs/installation.md) for full Docker details."

  **Must NOT do**:
  - Do not rewrite or restructure other README sections
  - Do not add Podman/Compose examples
  - Do not change the version header — README already correctly says 6.2.0

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: none

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: F1
  - **Blocked By**: None

  **References**:
  - `README.md` (current) — read to find Usage section and correct insertion point
  - `.github/workflows/docs.yml` — for canonical image names/tags

  **Acceptance Criteria**:
  - [ ] `grep -c 'docker run' README.md` → ≥ 2 (one per arch)
  - [ ] Existing README sections unmodified

  **QA Scenarios**:
  ```
  Scenario: Docker run examples present for both arches
    Tool: Bash (grep)
    Steps:
      1. Run: grep -n 'docker run' README.md
    Expected Result: ≥ 2 lines matching
    Evidence: .sisyphus/evidence/task-4-readme-docker.txt
  ```

  **Commit**: YES (group with 1, 2, 3)
  - Message: `docs(readme): add Docker run option for non-x86 Linux users`
  - Files: `README.md`

---

- [x] 5. Populate CHANGELOG.md from git history

  **What to do**:
  - Run `git log --oneline -30` to confirm commit history
  - Replace the empty stub in `CHANGELOG.md` with a structured release note including:
    - **New Features**: verify CLI command, rich progress bars, Docker CI image
    - **Changes**: xxHash swap, label-rebuild logic, AGENTS.md hierarchy
    - **Compared to 6.1.0**: note CLI interface change (`tablassert-cli -i` → `tablassert-cli build-knowledge-graph -i`)
    - Footer: "For full commit history, run `git log --oneline` in the repository."
  - Write conservatively: if a commit is unclear, use neutral language

  **Must NOT do**:
  - Do not claim features that are not verifiable in the codebase
  - Do not reference breaking changes unless confirmed by git history

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: none

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 6 (mkdocs nav needs CHANGELOG.md to exist)
  - **Blocked By**: None

  **References**:
  - Git log: run `git log --oneline -30` — primary source of truth
  - `CHANGELOG.md` (current) — just the stub `# 6.2.0`; replace entirely

  **Acceptance Criteria**:
  - [ ] `wc -l CHANGELOG.md` → ≥ 20 lines
  - [ ] `grep -c '###' CHANGELOG.md` → ≥ 2 sections
  - [ ] `grep -c 'verify\|Verify' CHANGELOG.md` → ≥ 1

  **QA Scenarios**:
  ```
  Scenario: CHANGELOG has substantive content
    Tool: Bash
    Steps:
      1. Run: wc -l CHANGELOG.md
    Expected Result: ≥ 20 lines
    Evidence: .sisyphus/evidence/task-5-changelog-lines.txt
  ```

  **Commit**: YES (independent)
  - Message: `docs(changelog): populate 6.2.0 release notes from git history`
  - Files: `CHANGELOG.md`

---

- [x] 6. Update mkdocs.yml nav to include Changelog

  **What to do**:
  - Open `mkdocs.yml` and add `- Changelog: changelog.md` to the `nav:` list
  - Placement: after the last existing nav entry (after API Reference is conventional)

  **Must NOT do**:
  - Do not restructure existing nav entries
  - Do not add extra plugins or extensions

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: none

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 2, with Task 7)
  - **Parallel Group**: Wave 2
  - **Blocks**: F1
  - **Blocked By**: Task 5

  **References**:
  - `mkdocs.yml` (current) — contains the existing nav list; read to find insertion point

  **Acceptance Criteria**:
  - [ ] `grep -c 'changelog\|Changelog' mkdocs.yml` → ≥ 1

  **QA Scenarios**:
  ```
  Scenario: Changelog in nav
    Tool: Bash (grep)
    Steps:
      1. Run: grep -n 'changelog\|Changelog' mkdocs.yml
    Expected Result: ≥ 1 matching line
    Evidence: .sisyphus/evidence/task-6-mkdocs-nav.txt
  ```

  **Commit**: YES
  - Message: `docs(nav): add Changelog to mkdocs nav`
  - Files: `mkdocs.yml`

---

- [x] 7. Update .github/workflows/docs.yml for multi-arch Docker (reviewed - matrix strategy, not Buildx)

  **What to do**:
  - In the `deploy-docker` job, add the following steps **after** the existing Nix x86 build (keeping it intact):
    1. `docker/setup-qemu-action@v3` — enables ARM64 emulation on GitHub's x86 runners
    2. `docker/setup-buildx-action@v3` — enables BuildKit multi-platform builds
    3. `docker/login-action@v3` pointing to `ghcr.io` (use `GITHUB_TOKEN`)
    4. `docker/build-push-action@v3` with:
       - `platforms: linux/amd64,linux/arm64`
       - `push: true`
       - `tags: ghcr.io/${{ github.repository_owner }}/tablassert-cli:latest`
  - If no `Dockerfile` exists at the repo root, create a minimal one (Python 3.13-slim base with pip install of tablassert, ENTRYPOINT tablassert-cli). Check first before creating.
  - The existing Nix-based x86 build steps MUST remain; the Buildx step is additive with a different tag (`:latest` vs `:x86_64-linux-latest`)

  **Must NOT do**:
  - Do not remove the existing Nix-based x86 docker build steps
  - Do not change the `deploy-docs` job
  - Do not use deprecated action versions
  - Do not hardcode secrets or tokens

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: none

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 2, with Task 6)
  - **Parallel Group**: Wave 2
  - **Blocks**: F2
  - **Blocked By**: None

  **References**:
  - `.github/workflows/docs.yml` (current) — read entire file before editing
  - Docker build-push-action docs: https://github.com/docker/build-push-action

  **Acceptance Criteria**:
  - [ ] `grep -c 'setup-qemu-action\|setup-buildx-action\|build-push-action' .github/workflows/docs.yml` → ≥ 3
  - [ ] `grep -c 'platforms' .github/workflows/docs.yml` → ≥ 1
  - [ ] `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/docs.yml'))"` exits 0
  - [ ] `grep -n 'nix build .#packages.x86_64-linux.docker' .github/workflows/docs.yml` → ≥ 1 line

  **QA Scenarios**:
  ```
  Scenario: Multi-arch actions present
    Tool: Bash (grep)
    Steps:
      1. Run: grep -n 'setup-qemu\|buildx\|platforms' .github/workflows/docs.yml
    Expected Result: ≥ 3 matching lines
    Evidence: .sisyphus/evidence/task-7-workflow-multiarch.txt

  Scenario: YAML is syntactically valid
    Tool: Bash (python3)
    Steps:
      1. Run: python3 -c "import yaml; yaml.safe_load(open('.github/workflows/docs.yml')); print('VALID')"
    Expected Result: prints VALID, exit 0
    Evidence: .sisyphus/evidence/task-7-yaml-valid.txt

  Scenario: Existing Nix x86 build still present
    Tool: Bash (grep)
    Steps:
      1. Run: grep -n 'nix build' .github/workflows/docs.yml
    Expected Result: ≥ 1 matching line
    Evidence: .sisyphus/evidence/task-7-nix-build-intact.txt
  ```

  **Commit**: YES (independent)
  - Message: `ci: add multi-arch Docker build (amd64 + arm64) via Docker Buildx`
  - Files: `.github/workflows/docs.yml`
  - Pre-commit: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/docs.yml'))"` (YAML lint)

---

## Final Verification Wave

- [x] F1. **Full Doc Audit** — `quick`

  Run `grep -rn "6\.1\.0" docs/ README.md CHANGELOG.md` and assert zero matches (or only intentional historical context). Run `mkdocs build --strict` (if mkdocs is available) or manually inspect nav for the Changelog entry. Read `docs/cli.md` and verify both commands (`build-knowledge-graph`, `verify-table-configuration-syntax`) are present with accurate options.

  Output: `Version strings [CLEAN/N stale] | CLI commands [2/2 documented] | mkdocs nav [CHANGELOG present] | VERDICT: APPROVE/REJECT`

- [x] F2. **Workflow YAML Review** — `unspecified-high`

  Read `.github/workflows/docs.yml` after edits. Verify: QEMU setup step present, Docker Buildx configured, `platforms: linux/amd64,linux/arm64` set, existing x86 Nix build job untouched, GHCR push targets both arch tags, no YAML syntax errors (use `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/docs.yml'))"` to lint).

  Output: `YAML valid [YES/NO] | Multi-arch platforms [PRESENT/MISSING] | x86 Nix job [INTACT/BROKEN] | VERDICT: APPROVE/REJECT`

---

## Commit Strategy

- **Wave 1**: `docs(6.2.0): update version strings, cli reference, installation docker, changelog`
  - Files: `docs/index.md`, `docs/configuration/graph.md`, `docs/cli.md`, `docs/installation.md`, `README.md`, `CHANGELOG.md`
- **Wave 2**: `ci: add multi-arch docker publish; docs: add changelog to nav`
  - Files: `.github/workflows/docs.yml`, `mkdocs.yml`

---

## Success Criteria

### Verification Commands
```bash
# No stale 6.1.0 version strings
grep -rn "6\.1\.0" docs/ README.md && echo "FAIL: stale strings found" || echo "PASS: no stale strings"

# CHANGELOG has content
wc -l CHANGELOG.md  # Expected: >10 lines

# Both CLI commands documented
grep -c "verify-table-configuration-syntax\|build-knowledge-graph" docs/cli.md  # Expected: >=2

# mkdocs nav has changelog
grep -c "changelog\|Changelog" mkdocs.yml  # Expected: >=1

# Docker section in installation
grep -c "Docker\|docker" docs/installation.md  # Expected: >=1

# Docker section in README
grep -c "Docker\|docker" README.md  # Expected: >=1
```

### Final Checklist
- [x] All "6.1.0" current-version references replaced with "6.2.0"
- [x] `docs/cli.md` documents both commands
- [x] Docker usage documented in both README.md and docs/installation.md
- [x] CHANGELOG.md populated with substantive 6.2.0 release notes
- [x] `mkdocs.yml` nav includes Changelog
- [x] `.github/workflows/docs.yml` publishes multi-arch Docker image
