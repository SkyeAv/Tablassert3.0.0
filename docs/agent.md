# Autonomous Agent (`[agent]` extra)

**Why this exists:** hand-authoring a Tablassert config for every PMC supplementary table does not scale.
The optional `[agent]` extra makes it autonomous — point it at **PubMed Central (PMC)** article IDs and it
**derives the config for you**, then builds, audits, and iteratively improves the graph until the entity
resolution *maps* (coverage threshold). The outcome is an **NCATS Translator-compliant KGX knowledge
graph** per article, with the whole loop scored on **quality / cost / wrong tool calls**.

Under the hood it is built on [smolagents](https://github.com/huggingface/smolagents) `CodeAgent` (a
ReAct loop) and [DSPy](https://dspy.ai) GEPA for prompt optimization.

!!! warning "Optional extra"
    The base `tablassert` package does **not** require any of this. `smolagents` and `dspy` are imported
    **lazily** in `tablassert.agent`, so the base install and its test suite are unaffected. Install the
    extra with `pip install tablassert[agent]`.

## Installation

```bash
pip install tablassert[agent]
```

The extra pins:

| Package | Version | Role |
| --- | --- | --- |
| `smolagents` | `==1.26.0` | `CodeAgent` ReAct loop, `OpenAIModel`/`LiteLLMModel`, tools |
| `dspy` | `==3.2.1` | `dspy.GEPA` black-box prompt optimization |
| `litellm` | (any) | optional fallback / rate-limiting model backend |
| `pdfminer.six` | (any) | extract a `.pdf` main text into data-fenced context (`pmc_article_context`) |

## PMC-AWS data source

Tables are fetched from the **new** PMC open-access S3 bucket — the sanctioned bulk path.

| | |
| --- | --- |
| **Bucket** | `s3://pmc-oa-opendata` (us-east-1) |
| **Access** | world-readable, **free**, no credentials (`--no-sign-request`); *not* requester-pays |
| **Download** | public HTTPS `https://pmc-oa-opendata.s3.amazonaws.com/<prefix>/<file>` |
| **Layout** | one prefix per article-version, e.g. `PMC11708054.1/`, containing `PMC<n>.<v>.xml` (JATS), `.pdf`, `.txt`, `.json` (metadata) and the media/supplementary files |

`fetch_pmc_article(pmc_id, outdir)` downloads the **useful** payload for the **latest** article version,
failing fast (cheap checks before any large download and before any model call):

1. Enumerates version prefixes via S3 `list-objects-v2` (`?list-type=2&prefix=PMC<n>.&delimiter=/`) and
   selects the **latest** version (numeric, so `PMC<n>.10` beats `PMC<n>.2`); older versions are ignored.
2. Checks the latest version's `.json` metadata for open access (`is_pmc_openaccess` / a `CC*`
   `license_code`) — **before** any large download (not open access ⇒ `PermissionError` immediately).
3. Enumerates the version's objects (`?list-type=2&prefix=PMC<n>.<v>/`) and confirms a data table is
   present (a file with extension `.xlsx .xls .csv .tsv`) — **before** any large download (none ⇒
   `FileNotFoundError`).
4. Downloads only the **useful** files to `outdir/<prefix>/<file>` and returns their paths: the main text
   (`.xml`/`.nxml`/`.txt`/`.pdf`), the `.json` metadata, and every data table. Binary media (images,
   `.docx`) are skipped. `fetch_pmc_tables` remains as a thin wrapper returning only the table files.

The main text is wired into the agent via the `pmc_article_context` tool, which parses the JATS `.xml`
into a compact, data-fenced summary (title, abstract, section outline, and a supplementary-table manifest
with labels/captions). A `.txt` is a fenced excerpt; a `.pdf` is extracted to a fenced excerpt via
`pdfminer.six` (so PDF-only articles still give the agent main-text context). For Excel tables,
`read_table` lists **all worksheets** and reads a chosen one via `sheet=` (set `source.sheet` in the
config), so the agent can check every table and every sheet before authoring a config.

!!! failure "The old paths are dead"
    The legacy `s3://pmc-open-access` bucket, the FTP `oa_file_list.csv`, and the per-article `tar.gz`
    bundles were **deprecated and removed (Aug 2026)**. The PMC website `/bin/` URLs are unreliable
    (404/HTML stubs) and batch-scraping the website is prohibited. Use the S3 bucket only.

!!! note "Coverage + licensing"
    Only the **open-access subset** of PMC (~half) is available here. Articles are **CC-BY**: cite the
    source and DOI (e.g. PMC11708054 → [10.1128/mbio.01679-24](https://doi.org/10.1128/mbio.01679-24)).

### Local payloads (non-open-access articles)

Only the open-access subset of PMC is fetchable from the bucket. To run the **same** derive/build/improve
pipeline on an article you already hold locally (e.g. a non-open-access paper), pass `--local`:

```bash
# one directory used for every PMC id
tablassert agent PMC11708054 --fullmap ./fullmap --local ./payloads/PMC11708054

# per-article directories
tablassert agent PMC1 PMC2 --fullmap ./fullmap --local PMC1=./payloads/p1 PMC2=./payloads/p2
```

A local payload directory holds the table(s) and (optionally) the article main text. When `--local` is
given for an id, the supervisor locates the files there and **does not fetch from PMC-AWS**; each section's
`source.local` points at the local file (set `source.url` to the original download link if you want the
config to be re-fetchable). A `--local` directory that does not exist fails loud (exit 2).

## Model configuration

The agent talks to an **OpenAI-compatible** endpoint (e.g. a Qwen endpoint). Configuration comes from
CLI flags **and** environment variables — **secrets are never hardcoded**, and the command **fails
loudly** if any required value is unset.

| Flag | Env var | Purpose |
| --- | --- | --- |
| `--model-id`, `-m` | `TABLASSERT_AGENT_MODEL_ID` | model identifier |
| `--api-base`, `-ab` | `TABLASSERT_AGENT_API_BASE` | OpenAI-compatible base URL |
| `--api-key`, `-ak` | `TABLASSERT_AGENT_API_KEY` | API key (secret) |
| `--backend`, `-b` | — | `openai` (default) or `litellm` |

```bash
export TABLASSERT_AGENT_MODEL_ID="qwen3-max"
export TABLASSERT_AGENT_API_BASE="https://YOUR-ENDPOINT.example.com/v1"   # placeholder
export TABLASSERT_AGENT_API_KEY="sk-***"                                  # placeholder — never commit a real key
```

If a value is missing, `tablassert agent` prints a message naming the exact flag/env var and exits
non-zero **before** any model call.

## Running it for real

With network access and a configured endpoint:

```bash
tablassert agent PMC11708054 PMC12345678 \
  --fullmap ./fullmap \
  --map-threshold 0.25 \
  --max-improve-iters 3 \
  --max-steps 20 \
  --state-dir .tablassert/agent
```

Flags: `--max-steps`/`-ms`, `--map-threshold`/`-mt`, `--max-improve-iters`/`-mi`,
`--state-dir`/`-sd`, `--backend {openai,litellm}`/`-b`, plus `--local`/`-l`, `--reflexion`,
`--judge-model`, `--judge-threshold`, and the `--optimize`/`-o` prompt-optimization flags
(`--instructions-file`, `--instructions-out`, `--max-metric-calls`, `--dataset`).
The [CLI reference — `agent`](cli.md#agent) is the authoritative flag table; the list here is a compact
reminder.

### What the supervisor does

The **outer supervisor is deterministic Python** (not an LLM) — smolagents' #1 practice is deterministic
control flow over agentic decisions. For each PMC id it:

1. **Fetches** the latest-version article payload (`fetch_pmc_article`: main text + metadata + all tables;
   fails fast on not-open-access / no-table) and presents **all** candidate tables to the agent.
2. Runs the **inner `CodeAgent`** to *derive* an initial table config (`pmc_article_context` → `read_table`
   → `derive_config`, every section gated by the Section JSON schema). The agent maps **each** mappable
   table/worksheet as its own section — **one config per paper** (see below).
3. **Builds + audits** in one deterministic mega-tool (`build_and_audit`: validate → build → QC → coverage).
4. **Improves** while coverage `< map_threshold` and budget remains: `propose_config_edit` → rebuild →
   **accept iff strictly better** (monotonic — regressions are rejected).
5. **Records** metrics, **checkpoints**, and moves to the next config.

A config that won't map after `--max-improve-iters` is marked `SKIPPED: <reason>` and the supervisor
advances — one difficult article never aborts the batch. A config that **builds** but whose fullmap
coverage **cannot be measured** (an unreproducible source frame) is marked `BUILT_UNMEASURED` — a
terminal **non-failure** that is neither a certified `MAPPED` nor counted as a `SKIPPED`; the best
config is still written and is reusable by the full pipeline. Coverage measurement itself is
multi-cwd: a relative `source.local` is resolved against the build workdir as well as the current
directory before a config is declared unmeasurable.

### Optional gates: reflexion improver & semantic judge

Two opt-in extensions layer on top of the deterministic improve loop (both reuse the configured
endpoint; neither is required):

- **`--reflexion`** — when the deterministic `propose_config_edit` stalls, a tier-2 LLM reflexion
  improver reflects on the coverage feedback and proposes an edit that may change predicate/source
  (same model config).
- **`--judge-model` / `--judge-threshold`** — a semantic judge scores the built output; when
  `--judge-model` is set, `MAPPED` additionally requires the normalized score to clear
  `--judge-threshold` (`0.5` when unset). Without `--judge-model` the coverage gate alone decides.

### Multi-section configs (one per paper)

The agent authors **one table config per paper** that may contain **multiple sections** — one per
mappable supplementary table/worksheet. The config is shaped as `{template, sections}`:

- **`template`** carries the shared per-paper **provenance** (`repo` + `publication`) and nothing else —
  in particular **no `source`**.
- **`sections`** is a list with one entry per table; **each section owns its own `source`** (its own
  `local` path **and** its own `source.url` download link, plus `sheet`/`row_slice`/`delimiter` as
  needed) and its own `statement`. Different sections can therefore reference **different files with
  different download links**.

The final-answer gate (`validate_table_config`) validates **every** section, so a config is accepted
only when all of its sections are schema-valid. `map_coverage` measures each section and reports an
**aggregate** (`overall` = mean of section coverages, `min` = weakest section, `measured` = true iff
every section measured, plus the per-section breakdown under `sections`). `propose_config_edit` edits
each section independently from its own coverage entry. A single-table paper is still one config with
one section. State and storage stay **per-paper**: one best config (`configs/<pmc_id>.yaml`) holding
all sections, with `section_coverages` recorded for visibility.

### Workspace layout & checkpoint / resume

`tablassert agent` uses a **single stable workspace root** — `state_dir` (default `.tablassert/agent`,
override with `--state-dir`). The CLI never sets a separate artifact root, so the checkpoint, the configs,
the fetched downloads, and the build outputs **all co-locate** under it:

```text
.tablassert/agent/                       # = state_dir (the workspace root)
  state.json                             # supervisor checkpoint (atomic; unchanged location)
  graph.yaml                             # SHARED aggregate graph registry (flock-serialized, atomic)
  graph.yaml.lock                        # sidecar lock file for graph.yaml (exclusive flock)
  configs/<pmc_id>.yaml                  # best / accepted config (ALL configs in ONE folder)
  configs/<pmc_id>.derived.yaml          # initial agent-derived config
  downloads/<pmc_id>/<prefix>/...        # fetched PMC payload (main text + metadata + tables) — stable, persists
  builds/<pmc_id>/                       # KGX agent_0.0.1.{nodes,edges}.ndjson + table.yaml + graph.yaml + .tablassert/store — stable
```

| Path | Contents | Lifecycle |
| --- | --- | --- |
| `state.json` | supervisor checkpoint: `{pmc_id, status, config_path, coverage_history[], qc_pass_rate, attempts, last_edits, best_coverage, best_config_path}` per record | written **atomically** (tmp write + `os.replace`) after each config and each improve iteration; git-ignored |
| `graph.yaml` | SHARED aggregate graph registry: one `tables` entry per successful (`MAPPED` / `BUILT_UNMEASURED`) build | maintained under an exclusive `graph.yaml.lock` flock; atomic writes; see [Parallel agents and the shared graph registry](#parallel-agents-and-the-shared-graph-registry) |
| `graph.yaml.lock` | sidecar lock file serializing registry read-modify-write | created on first registration; never deleted |
| `configs/<pmc_id>.yaml` | the best / accepted config for the article | the reuse entry point (below) |
| `configs/<pmc_id>.derived.yaml` | the agent's initial derived config | kept for provenance |
| `downloads/<pmc_id>/<prefix>/` | fetched PMC payload (main text + metadata + tables) | **stable** — persists across runs |
| `builds/<pmc_id>/` | KGX artifacts: `agent_0.0.1.{nodes,edges}.ndjson`, `table.yaml`, `graph.yaml`, `.tablassert/store` | **stable** — the built graph for the article |

Re-running the same command **resumes** from the checkpoint: records already `MAPPED`/`SKIPPED` are
skipped. The `downloads/` payload persists on disk across runs.

### Reusing agent outputs with the full pipeline

The best config's `source.local` points at the downloaded table under `downloads/<pmc_id>/`, so the full
(non-agent) pipeline can reuse the agent's output **without re-fetching**. The agent already writes a
ready-to-build `graph.yaml` (wrapping `table.yaml` with the resolved fullmap) into `builds/<pmc_id>/`:

```bash
cd .tablassert/agent/builds/PMC11708054
tablassert build-kg -f graph.yaml
```

!!! warning "Not relocatable"
    `source.local` in the best config is an **absolute** path into `downloads/<pmc_id>/`. The workspace is
    therefore **not relocatable** — moving or renaming the `.tablassert/agent` folder breaks that reference
    (re-run the agent, or fix `source.local`, after any move).

### Parallel agents and the shared graph registry

Several `tablassert agent` processes can run CONCURRENTLY against the SAME shared `--state-dir` and
each successful build self-registers into ONE aggregate graph config that a single `build-kg` then
builds as a whole:

```bash
# fan out over DISJOINT pmc sets, all pointed at one shared state dir
tablassert agent PMC1 PMC2 --fullmap ./fullmap --state-dir ./shared &
tablassert agent PMC3 PMC4 --fullmap ./fullmap --state-dir ./shared &
wait

# one build of the whole registered graph
tablassert build-kg -f ./shared/graph.yaml
```

Use **disjoint pmc sets**: each process owns its own ids. The shared registry itself is fully
cross-process safe, but the per-process checkpoint (`state.json`) read-modify cycle is not
cross-process locked, so two processes must not own the same pmc id.

**How the registry works.** Every build that ends `MAPPED` or `BUILT_UNMEASURED` (both are
successful builds) UPSERTS its best config into `<state-dir>/graph.yaml`:

- **Concurrency-safe** — each registration takes an EXCLUSIVE `flock` on the
  `<state-dir>/graph.yaml.lock` sidecar around the read-modify-write, then persists atomically
  (tmp file + `os.replace`, the same pattern as `state.json`). No registration can lose or tear
  another process's entry.
- **Upsert by pmc id** — a re-run REPLACES the prior entry for the same pmc id (matched by config
  basename stem); other entries keep their insertion order. Entries are ABSOLUTE paths, so
  `build-kg` works from any CWD.
- **`fullmap` is first-wins** — the first fullmap recorded stays; a later run passing a different
  fullmap keeps the existing value and logs a warning.
- **Self-healing** — a corrupt registry (bad YAML, not a mapping, or failing `Graph.model_validate`)
  is renamed `graph.yaml.corrupt-<UTC timestamp>` and rebuilt fresh with a warning, so unattended
  parallel runs never wedge on a damaged file.
- **Registered statuses** — only `MAPPED` and `BUILT_UNMEASURED` register; `SKIPPED` never does. A
  re-run that SKIPS an already-MAPPED pmc keeps the existing entry (resume skips terminal records
  entirely, so nothing rewrites them). `tablassert rebuild-agent-graph --state-dir ./shared
  --fullmap ./fullmap` reconstructs the registry from `state.json` and prunes stale entries
  (deleted configs, non-registered statuses).

## The tools

| Tool | Kind | Purpose |
| --- | --- | --- |
| `fetch_pmc_article` | function | PMC-AWS download of the useful latest-version payload (main text + metadata + tables), fail-fast |
| `pmc_article_context` | tool | parse the JATS main text into a **data-fenced** summary (title/abstract/sections/supplementary manifest); `.txt`/`.pdf` render a fenced excerpt (PDF via `pdfminer.six`) |
| `read_table` | tool | render a table as **data-fenced, spotlighted** text; lists **all worksheets** of an Excel file (`sheet=`) |
| `derive_config` | tool | author a table config (`template` + one section per table); each section must satisfy `Section.model_json_schema()` |
| `build_and_audit` | tool | **one** deterministic validate→build→QC→coverage mega-tool |
| `map_coverage` | tool | fullmap term-resolution coverage (per-column + overall) |
| `propose_config_edit` | tool | deterministic, constrained `NodeEncoding` edits + rationale |

`build_and_audit` returns coded errors **verbatim** (each carries a docs URL) so the agent can
self-correct the exact offending field.

## Prompt engineering

The agent's `instructions` make the techniques explicit:

- **ReAct + planning** — `CodeAgent` is a ReAct loop; `planning_interval=3` re-plans every few steps.
- **Structured / constrained output** — `derive_config` injects the Section JSON schema; a
  `final_answer_checks=[validate_table_config]` gate means the agent can only terminate with a config
  whose **every section** is schema-valid (multi-section configs are validated section-by-section).
- **Few-shot exemplars** — the tutorial gene~disease section, the ALAMV6 organism~chemical section, and a
  multi-section config (one config, two tables, each section its own source/url).
- **Reflexion-style self-critique** — `propose_config_edit` / `reflexion_improve` reflect on failing rows,
  error codes, and unresolved terms, then make a targeted, schema-valid edit.
- **Error-recovery prompting** — tools return rich coded errors; the prompt directs the agent to read the
  code + message and fix precisely that field, never repeating an unchanged config.
- **Context trimming** — a `step_callback` tallies tokens/steps and failed/wrong/redundant tool calls, and
  trims large old observations to save tokens.

### Prompt-injection defenses

PMC article text and tables are **untrusted data**. Defenses:

- **Data-fence + spotlighting** — `read_table` wraps content in `<<<PMC_DATA_BEGIN>>>` /
  `<<<PMC_DATA_END>>>` preceded by a guardrail; the instructions state that fenced content is DATA, never
  instructions, and any embedded commands are ignored.
- **Minimal authorized imports** — the executor allowlist is just `["yaml"]`, so a hijacked agent cannot
  `import os`/`subprocess`.

## Evaluation & optimization loop

The harness scores every run on three objectives and optimizes them as a black box.

**Deterministic metrics (gate the loop):**

- **Quality** — fullmap mapping coverage, QC audit pass rate, config validity (hard gate), and KG
  node/edge **F1** vs the reference graph.
- **Cost** — `RunResult.token_usage` + step count (the API is free; tokens are the proxy).
- **Reliability** — failed / wrong / redundant tool-call counts from the `ActionStep` logs.

**LLM-as-judge (semantic dimensions only):** a pointwise **0–3** rubric over schema validity, coverage,
QC pass, predicate/category appropriateness, provenance completeness, efficiency, and tool-call
cleanliness — with **position** (both orderings averaged) and **verbosity** bias mitigation. Deterministic
metrics gate the rest; the judge only scores what a metric cannot. Without a judge model, an offline
deterministic heuristic is used.

**Optimizers:**

- **Reflexion** — the simple first-increment retry (`reflexion_improve`).
- **GEPA** — `dspy.GEPA(metric=gepa_metric, candidate_selection_strategy="pareto", …)` optimizes the
  agent's `instructions` + tool `description`s + exemplars as a **black box** from textual feedback
  (`gepa_metric` returns `dspy.Prediction(score=weighted_quality, feedback="<failing rows + error codes +
  wrong-call list>")`). It is system-agnostic, Pareto-native, and needs few rollouts.

**Reporting:** `pareto_frontier(runs)` returns the **non-dominated set** over (quality ↑, cost ↓,
wrong-calls ↓) and its **knee** (best quality per unit cost).

### Real-run prompt optimization (`--optimize`)

GEPA prompt optimization is a first-class CLI path. `tablassert agent --optimize` (`-o`) runs
`dspy.GEPA` and **persists the optimized instructions** instead of running the supervisor.

Following GEPA best practice, the optimizer splits the models: a **strong reflection LM** (`--model-id`)
proposes the few instruction edits, and an optional **fast task LM** (`--task-model`) runs the many
candidate program evaluations. Pointing `--task-model` at a cheap model (e.g. a flash model) keeps the
run fast while the strong model does the thinking; without `--task-model` the reflection LM is used for
both. `--gepa-threads` parallelizes GEPA's candidate **LM forward passes** only — the coverage-scoring
builds stay serialized on the process-wide `_GEPA_BUILD_LOCK` (`agent.py`, since `os.chdir` is
process-global), so a higher thread count does not speed up the expensive build/coverage step.

```bash
# optimize the agent prompt over a dataset of examples, writing the result to a file
tablassert agent PMC11708054 --fullmap ./fullmap --optimize \
  --dataset examples/gepa-dataset.yaml --task-model qwen-flash \
  --max-metric-calls 30 --gepa-threads 4 \
  --instructions-out .tablassert/agent/optimized_instructions.yaml

# later, run the supervisor with the optimized prompt
tablassert agent PMC11708054 --fullmap ./fullmap \
  --instructions-file .tablassert/agent/optimized_instructions.yaml
```

`--dataset` is a YAML/JSON list of examples. Each example carries `table_summary` and
`coverage_feedback` (the program inputs); it MAY also carry:

- `fullmap` — a fullmap path. When present, the GEPA metric scores each proposed config with **real
  fullmap coverage** (via a `build_and_audit` head-sample), so GEPA optimizes the genuine objective
  rather than a validity-only proxy.
- `workdir` — the directory a proposed config's relative `source.local` resolves against (LLMs mimic the
  exemplar's `./downloads/...` paths), so coverage is measured on the actual table.
- `head` — default `true`: score a fast 5-row preview; set `false` for full-fidelity coverage builds.

`--max-metric-calls` bounds the GEPA metric budget. `save_optimized_instructions` /
`load_optimized_instructions` persist and reload the prompt (a `{instructions, descriptions}` mapping).
Without `--instructions-file` the built-in `INSTRUCTIONS` prompt is used. (A real optimization run needs a
live model; the offline suite exercises this path via an injectable `gepa_cls` stub.)

### Golden fixture

`tests/agent_fixtures/PMC11708054/` is an offline replay pair: the ALAMV6 reference config, a small
**synthetic** source table, and a trimmed reference config (CC-BY attribution to PMC11708054; the
reference KGX is computed in-test against a tiny real redb — nothing large is committed). A second
fixture, `tests/agent_fixtures/GENE_DISEASE/`, is a gene~disease config in multi-section
(`{template, sections}`) shape with PMID provenance — used to keep the offline heuristic judge and the
W3 multi-section validation honest on a distinct config.

## Testing

The agent suite is **fully offline** — no live LLM or network. It uses a `FakeModel` smolagents stub,
mocked/snapshotted PMC data, a tiny real redb (`rs.build_fullmap_db`), and injectable GEPA stubs.

```bash
# Agent tests SKIP without the extra and PASS with it:
uv sync --extra agent
uv run pytest -q tests/test_agent_eval.py tests/test_agent_supervisor.py tests/test_agent_assembly.py
```

Without the extra, the base suite stays green and every agent test skips via `pytest.importorskip`.

!!! tip "Telemetry"
    A real `CodeAgent.run` emits HuggingFace telemetry that blocks on a network call. The agent tests set
    `HF_HUB_DISABLE_TELEMETRY=1` / `DO_NOT_TRACK=1` to stay hermetically offline; set the same when running
    fully air-gapped.
