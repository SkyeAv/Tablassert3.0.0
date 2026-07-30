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
with labels/captions). For Excel tables, `read_table` lists **all worksheets** and reads a chosen one via
`sheet=` (set `source.sheet` in the config), so the agent can check every table and every sheet before
authoring a config.

!!! failure "The old paths are dead"
    The legacy `s3://pmc-open-access` bucket, the FTP `oa_file_list.csv`, and the per-article `tar.gz`
    bundles were **deprecated and removed (Aug 2026)**. The PMC website `/bin/` URLs are unreliable
    (404/HTML stubs) and batch-scraping the website is prohibited. Use the S3 bucket only.

!!! note "Coverage + licensing"
    Only the **open-access subset** of PMC (~half) is available here. Articles are **CC-BY**: cite the
    source and DOI (e.g. PMC11708054 → [10.1128/mbio.01679-24](https://doi.org/10.1128/mbio.01679-24)).

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
  --fullmap .fullmap \
  --map-threshold 0.8 \
  --qc-threshold 0.9 \
  --max-improve-iters 3 \
  --max-steps 20 \
  --state-dir .tablassert-agent \
  --executor local
```

Flags: `--max-steps`/`-ms`, `--map-threshold`/`-mt`, `--qc-threshold`/`-qt`, `--max-improve-iters`/`-mi`,
`--state-dir`/`-sd`, `--executor {local,docker}`/`-e`, `--backend {openai,litellm}`/`-b`, `--no-fetch`/`-nf`.
The [CLI reference — `agent`](cli.md#agent) is the authoritative flag table; the list here is a compact
reminder.

### What the supervisor does

The **outer supervisor is deterministic Python** (not an LLM) — smolagents' #1 practice is deterministic
control flow over agentic decisions. For each PMC id it:

1. **Fetches** the latest-version article payload (`fetch_pmc_article`: main text + metadata + all tables;
   fails fast on not-open-access / no-table) and presents **all** candidate tables to the agent.
2. Runs the **inner `CodeAgent`** to *derive* an initial Section config (`pmc_article_context` → `read_table`
   → `derive_config`, gated by the Section JSON schema). The agent picks the table + worksheet to map.
3. **Builds + audits** in one deterministic mega-tool (`build_and_audit`: validate → build → QC → coverage).
4. **Improves** while coverage `< map_threshold` and budget remains: `propose_config_edit` → rebuild →
   **accept iff strictly better** (monotonic — regressions are rejected).
5. **Records** metrics, **checkpoints**, and moves to the next config.

A config that won't map after `--max-improve-iters` is marked `SKIPPED: <reason>` and the supervisor
advances — one difficult article never aborts the batch.

### Workspace layout & checkpoint / resume

`tablassert agent` uses a **single stable workspace root** — `state_dir` (default `.tablassert-agent`,
override with `--state-dir`). The CLI never sets a separate artifact root, so the checkpoint, the configs,
the fetched downloads, and the build outputs **all co-locate** under it:

```
.tablassert-agent/                       # = state_dir (the workspace root)
  state.json                             # supervisor checkpoint (atomic; unchanged location)
  configs/<pmc_id>.yaml                  # best / accepted config (ALL configs in ONE folder)
  configs/<pmc_id>.derived.yaml          # initial agent-derived config
  downloads/<pmc_id>/<prefix>/...        # fetched PMC payload (main text + metadata + tables) — stable, persists
  builds/<pmc_id>/                       # KGX agent_0.0.1.{nodes,edges}.ndjson + table.yaml + graph.yaml + .tablassert/store — stable
```

| Path | Contents | Lifecycle |
| --- | --- | --- |
| `state.json` | supervisor checkpoint: `{pmc_id, status, config_path, coverage_history[], qc_pass_rate, attempts, last_edits, best_coverage, best_config_path}` per record | written **atomically** (tmp write + `os.replace`) after each config and each improve iteration; git-ignored |
| `configs/<pmc_id>.yaml` | the best / accepted config for the article | the reuse entry point (below) |
| `configs/<pmc_id>.derived.yaml` | the agent's initial derived config | kept for provenance |
| `downloads/<pmc_id>/<prefix>/` | fetched PMC payload (main text + metadata + tables) | **stable** — persists across runs; `--no-fetch` replays this snapshot instead of re-downloading |
| `builds/<pmc_id>/` | KGX artifacts: `agent_0.0.1.{nodes,edges}.ndjson`, `table.yaml`, `graph.yaml`, `.tablassert/store` | **stable** — the built graph for the article |

Re-running the same command **resumes** from the checkpoint: records already `MAPPED`/`SKIPPED` are
skipped. Because `downloads/` persists, a resumed or `--no-fetch` run reuses the already-fetched payload
with no re-download.

### Reusing agent outputs with the full pipeline

The best config's `source.local` points at the downloaded table under `downloads/<pmc_id>/`, so the full
(non-agent) pipeline can reuse the agent's output **without re-fetching**:

```bash
tablassert build-kg .tablassert-agent/configs/PMC11708054.yaml --table-config --fullmap ./fullmap
```

!!! warning "Not relocatable"
    `source.local` in the best config is an **absolute** path into `downloads/<pmc_id>/`. The workspace is
    therefore **not relocatable** — moving or renaming the `.tablassert-agent` folder breaks that reference
    (re-run the agent, or fix `source.local`, after any move).

## The tools

| Tool | Kind | Purpose |
| --- | --- | --- |
| `fetch_pmc_article` | function | PMC-AWS download of the useful latest-version payload (main text + metadata + tables), fail-fast |
| `pmc_article_context` | tool | parse the JATS main text into a **data-fenced** summary (title/abstract/sections/supplementary manifest) |
| `read_table` | tool | render a table as **data-fenced, spotlighted** text; lists **all worksheets** of an Excel file (`sheet=`) |
| `derive_config` | tool | author a Section config; `output_schema = Section.model_json_schema()` |
| `build_and_audit` | tool | **one** deterministic validate→build→QC→coverage mega-tool |
| `map_coverage` | tool | fullmap term-resolution coverage (per-column + overall) |
| `propose_config_edit` | tool | deterministic, constrained `NodeEncoding` edits + rationale |

`build_and_audit` returns coded errors **verbatim** (each carries a docs URL) so the agent can
self-correct the exact offending field.

## Prompt engineering

The agent's `instructions` make the techniques explicit:

- **ReAct + planning** — `CodeAgent` is a ReAct loop; `planning_interval=3` re-plans every few steps.
- **Structured / constrained output** — `derive_config` injects the Section JSON schema; a
  `final_answer_checks=[validate_section]` gate means the agent can only terminate with a schema-valid config.
- **Few-shot exemplars** — the tutorial gene~disease config and the ALAMV6 organism~chemical config.
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
- **Sandboxed executor option** — `--executor docker` runs model-written code in a sandbox; the default
  `local` executor is **not** a security boundary (use `docker` for untrusted inputs).

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

### Golden fixture

`tests/agent_fixtures/PMC11708054/` is an offline replay pair: the ALAMV6 reference config, a small
**synthetic** source table, and a trimmed reference config (CC-BY attribution to PMC11708054; the
reference KGX is computed in-test against a tiny real redb — nothing large is committed).

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
