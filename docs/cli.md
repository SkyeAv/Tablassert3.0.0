# CLI Reference

Tablassert extracts knowledge assertions from tabular data into KGX NDJSON. The `tablassert` app
exposes **five subcommands** — `agent`, `build-fullmap`, `build-kg`, `rebuild-agent-graph`,
`validate` — plus an app-level `--version` flag. Run `tablassert --help` (or `<command> --help`)
for the live surface.

## Command index

| Command | Use this to… |
| --- | --- |
| [`agent`](#agent) | Autonomously derive, build, audit, and improve KG configs from PMC articles |
| [`build-fullmap`](#build-fullmap) | Build the embedded fullmap redb used for entity resolution |
| [`build-kg`](#build-kg) | Build a KGX NDJSON knowledge graph from a YAML configuration |
| [`rebuild-agent-graph`](#rebuild-agent-graph) | Rebuild the shared agent graph registry from the supervisor checkpoint |
| [`validate`](#validate) | Validate a graph or table configuration without executing it |
| [`validate-kgx`](#validate-kgx) | Validate built KGX NDJSON against the Biolink Model |

## App flags

These are flags on the root `tablassert` command, **not** subcommands.

| Flag | Description |
| --- | --- |
| `--version` | Print the installed package version as `tablassert <version>` (e.g. `tablassert 10.1.0`) and exit |
| `--help`, `-h` | Show help for the app or a subcommand |

!!! warning "Two different `--version`s"
    The app `--version` prints **Tablassert's package version**. The [`build-fullmap --version`](#build-fullmap)
    flag is unrelated: it selects a **RENCI BABEL snapshot date** (default `2026jul22`).

---

## agent

Use this to autonomously turn one or more PMC articles into audited, improved KG configs and graphs
(fetch → derive config → build + audit → improve until coverage maps). Requires the `[agent]` extra
(`pip install "tablassert[agent]"`); `--optimize` additionally needs the `[optimize]` extra
(`pip install "tablassert[optimize]"`, pulls `dspy`). Both are checked after flag validation and
before any model is built or article fetched, so a missing extra is reported with its install
command instead of surfacing mid-run — see [When an extra is missing](installation.md#when-an-extra-is-missing).

```bash
tablassert agent --fullmap PATH [OPTIONS] PMC-IDS...
```

PMC ids are passed positionally (also accepted as `--pmc-ids`). This page lists the flags; see
[Agent](agent.md) for the full pipeline, workspace layout, checkpoint/resume, and tooling.

| Option | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `PMC-IDS` (`--pmc-ids`) | list[str] | Yes | — | One or more PMC article ids (positional) |
| `--fullmap`, `-f` | Path | Yes | — | Fullmap redb file or base directory |
| `--model-id`, `-m` | str | No | `None` | Model id (env `TABLASSERT_AGENT_MODEL_ID`) |
| `--api-base`, `-ab` | str | No | `None` | OpenAI-compatible base URL (env `TABLASSERT_AGENT_API_BASE`) |
| `--api-key`, `-ak` | str | No | `None` | API key secret (env `TABLASSERT_AGENT_API_KEY`) |
| `--max-steps`, `-ms` | int | No | `20` | Max inner-agent steps per article |
| `--map-threshold`, `-mt` | float | No | `0.25` | Coverage an article must reach to be MAPPED |
| `--max-improve-iters`, `-mi` | int | No | `3` | Max deterministic improve iterations per article |
| `--state-dir`, `-sd` | Path | No | `.tablassert/agent` | Checkpoint/resume workspace directory |
| `--backend`, `-b` | {openai, litellm} | No | `openai` | Model backend |
| `--reflexion` | bool | No | `False` | Enable the tier-2 LLM reflexion improver (same model config) when the deterministic proposer stalls |
| `--judge-model` | str | No | `None` | Model id for the semantic judge gate; MAPPED then also requires the score to clear `--judge-threshold` |
| `--judge-threshold` | float | No | `None` | Semantic judge normalized-score threshold for MAPPED (`0.5` when unset) |
| `--biolink-threshold` | float | No | `0.0` | Minimum Biolink pass rate of the built KGX for MAPPED; `0.0` reports the rate without gating |
| `--local`, `-l` | list[str] | No | `None` | Local payload: one DIR for all ids, or `PMCid=DIR` mappings; skips the PMC-AWS fetch (exit 2 on a missing DIR) |
| `--optimize`, `-o` | bool | No | `False` | Run GEPA prompt optimization and persist optimized instructions instead of running the supervisor |
| `--instructions-file` | Path | No | `None` | Load GEPA-optimized instructions from a prior `--optimize` run |
| `--instructions-out` | Path | No | `None` | Where `--optimize` writes optimized instructions (default `<state-dir>/optimized_instructions.yaml`) |
| `--max-metric-calls` | int | No | `8` | GEPA metric-call budget for `--optimize` |
| `--dataset` | Path | No | `None` | YAML/JSON list of `{table_summary, coverage_feedback}` examples for `--optimize` (an example may also carry `fullmap`, `workdir`, and `head` to score each proposed config with real coverage) |
| `--task-model` | str | No | `None` | Fast model id for GEPA's many program evaluations (cheap task LM + strong reflection LM); `--model-id` is the reflection LM. Defaults to the reflection LM |
| `--gepa-threads` | int | No | `None` | Thread count for GEPA's evaluation pool (`--optimize`) — parallelizes candidate LM forward passes only; coverage-scoring builds stay serialized on `_GEPA_BUILD_LOCK` |

```bash
tablassert agent PMC11708054 --fullmap ./fullmap
```

!!! warning "Secrets"
    Model config comes from the flags above **or** the `TABLASSERT_AGENT_*` environment variables
    (explicit flags win). Secrets are **never** hardcoded or defaulted — a missing value fails loud
    (exit 2) **before** any model is built.

---

## build-fullmap

Use this to obtain the embedded `fullmap.redb` entity-resolution database. By default it first tries
to **download a prebuilt database** published for this Tablassert version; `--force` skips that and
builds from RENCI BABEL exports instead (download class + synonym files, then build a single redb).

```bash
tablassert build-fullmap [ARGS]
```

| Option | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `--output`, `-o` | Path | No | `./fullmap/data/fullmap.redb` | Path to write the redb file (prebuilt extraction or build output) |
| `--cache`, `-c` | Path | No | `./fullmap/downloads` | Directory for downloaded BABEL files when building from scratch (`classes/`, `synonyms/`) |
| `--version`, `-v` | str | No | `2026jul22` | BABEL snapshot date to fetch (a RENCI stamp, **not** Tablassert's version) |
| `--threads`, `-t` | int | No | `None` (auto) | Worker threads for a from-scratch build; auto-capped by memory on Linux (`/proc/meminfo`), else ~90% of CPUs |
| `--aria2c`, `-a` | Flag | No | `False` | Opt into the bundled `aria2c` binary from the `[aria2]` extra for resumable segmented downloads (the prebuilt archive **or** BABEL files); fails loud (exit 2, before any download starts) if the extra is missing or unsupported on the current platform, and on a non-zero aria2c exit |
| `--force`, `-f` | Flag | No | `False` | Skip the prebuilt download and always rebuild from BABEL outputs |

```bash
# Default: download the prebuilt fullmap.tar.zst for this version and extract it (fast)
tablassert build-fullmap --output /data/fullmap/fullmap.redb
# Force a from-scratch rebuild from BABEL outputs (e.g. after a BABEL snapshot bump)
tablassert build-fullmap --force --output /data/fullmap/fullmap.redb
# Accelerate either download with bundled aria2c (`pip install "tablassert[aria2]"`; the multi-GB prebuilt is the ideal aria2 use case)
tablassert build-fullmap --aria2c --output /data/fullmap/fullmap.redb
```

By default `build-fullmap` looks for a prebuilt `fullmap.tar.zst` at
`https://stars.renci.org/var/babel_outputs/<babel-version>/fullmap/<tablassert-version>/` (the version
directory is the **installed Tablassert package version**, never hardcoded), verifies it against the
published `sha256sum.txt`, and stream-extracts it beside `--output`. If no prebuilt exists for this
version (or the download/extract fails), it falls back to a from-scratch BABEL build and logs a
warning. A database already present at `--output` is reused as-is; pass `--force` to rebuild.

See [Fullmap](fullmap.md) for the data pipeline, output schema, and graph-config usage.

---

## build-kg

Use this to build a KGX NDJSON knowledge graph (nodes, edges, and a Resource Ingest Guide) from a
YAML configuration file.

```bash
tablassert build-kg GRAPH-CONFIGURATION-FILE [ARGS]
```

The positional `GRAPH-CONFIGURATION-FILE` (also `--configuration-file`, `-f`) is a **graph** YAML.

| Option | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `GRAPH-CONFIGURATION-FILE` (`--configuration-file`, `-f`) | Path | Yes | — | Graph YAML |
| `--release`, `-r` | Flag | No | `False` | Emit a slim, significant-only graph (drops `biolink:not_significant` edges before resolution) |
| `--qc`, `-q` | Flag | No | `False` | Audit resolved mappings (exact → fuzzy → BioBERT) so low-confidence edges are flagged; requires the `[qc]` extra, checked before the build starts. Also runs a final study stage that asserts over the emitted NDJSON — no duplicate node ids, no undeclared or isolated nodes, no malformed lines or stray whitespace (verbatim `original_*` fields excepted, since they are faithful source copies) — and fails the build (non-zero exit) on any violation |
| `--log`, `-l` | Flag | No | `False` | Enable verbose per-section logging |
| `--head`, `-hd` | Flag | No | `False` | Fast output-shape preview: ≤5 random rows/section, cached to `.head.parquet`, never clobbers a full build |

```bash
tablassert build-kg graph.yaml --qc --log
```

Output is written to `rig.artifact_base_path` (created when missing) as `{name}_{version}.nodes.ndjson`,
`{name}_{version}.edges.ndjson`, and `{name}_{version}.RIG.yaml`; intermediate parquet lands in
`.tablassert/store/`. The RIG document is audited in memory before it is written — an invalid
or incomplete RIG fails the build with `[rig-validation-failed]` and nothing is emitted. See
[Graph Configuration](configuration/graph.md).

??? info "Build progress & stages"
    The build runs six parallel stages — Loading Tables → Extracting Sections → Building TCode →
    Collecting Instructions → Building Subgraphs → Compiling Graph — plus a seventh, Studying Graph,
    only when `--qc` is passed. They run under a three-row live progress block (stage header;
    section bar with count/elapsed/ETA; in-flight item detail). Each completed
    stage prints a green `✓ Stage N · NAME · elapsed` line above the live block. During Building
    Subgraphs the detail line also shows the per-section phase (`load`, `filter`, `clean`, `encode`,
    `resolve`, `qc`, `edge`, `provenance`, `significance`, `finalize`, `write`).

---

## rebuild-agent-graph

Use this to rebuild the SHARED agent graph registry (`<state-dir>/graph.yaml`) from the supervisor
checkpoint (`<state-dir>/state.json`) — e.g. to prune stale entries after deleting configs, or to
recover a hand-edited/damaged registry. Parallel `tablassert agent` runs maintain the registry
incrementally (see [Agent — Parallel agents and the shared graph registry](agent.md#parallel-agents-and-the-shared-graph-registry));
this command reconstructs it deterministically from `state.json`.

```bash
tablassert rebuild-agent-graph [ARGS]
```

| Option | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `--state-dir`, `-sd` | Path | No | `.tablassert/agent` | Agent state directory holding `state.json` + `configs/` |
| `--fullmap`, `-f` | Path | Yes | — | Fullmap redb file or base directory recorded in the registry |

Every `MAPPED` / `BUILT_UNMEASURED` record whose best config still exists on disk becomes a
`tables` entry (absolute path, sorted by pmc id); every other entry — `SKIPPED` records, deleted
configs, stale leftovers — is pruned. The registry `fullmap` is **first-wins**: an existing value
that differs from `--fullmap` is kept (with a warning). The write is concurrency-safe (exclusive
`graph.yaml.lock` flock + atomic replace), so the command never corrupts the registry; run it
while agents are quiescent for a complete snapshot.

```bash
tablassert rebuild-agent-graph --state-dir .tablassert/agent --fullmap ./fullmap
tablassert build-kg -f .tablassert/agent/graph.yaml
```

---

## validate

Use this to validate a configuration against a schema without running the build — ideal for CI and
pre-commit hooks. The required `--schema` flag selects which schema to validate against (the kind is
no longer sniffed from the YAML).

```bash
tablassert validate CONFIGURATION-FILE --schema graph
tablassert validate -f CONFIGURATION-FILE --schema table
```

| Option | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `CONFIGURATION-FILE` (`--configuration-file`, `-f`) | Path | Yes | — | Configuration file to validate |
| `--schema`, `-s` | `graph` \| `table` | Yes | — | Schema to validate against: `graph` validates the `Graph` model **and** every referenced table; `table` validates section syntax only |

Exits non-zero on any schema error. See [Table Configuration](configuration/table.md) and
[Graph Configuration](configuration/graph.md).

```bash
tablassert validate table-config.yaml --schema table
tablassert validate graph.yaml --schema graph
```

---

## validate-kgx

Use this to check that a completed build is actually Biolink-compliant. Where
[`validate`](#validate) checks your *configuration*, `validate-kgx` checks the *output*: every node
and edge is constructed as the Biolink Pydantic class named by its own `category` — the same classes
[`NCATSTranslator/translator-ingests`](https://github.com/NCATSTranslator/translator-ingests) builds
when it ingests your files.

```bash
tablassert validate-kgx --nodes MY_KG_1.0.0.nodes.ndjson --edges MY_KG_1.0.0.edges.ndjson
```

| Option | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `--nodes`, `-n` | Path | Yes | — | Built `*.nodes.ndjson` file to validate |
| `--edges`, `-e` | Path | Yes | — | Built `*.edges.ndjson` file to validate |
| `--limit` | int | No | `20` | Maximum example failures to retain per file |

Failures are grouped by field and error type, so a systematic modelling problem shows up as one line
rather than a million:

```text
biolink-model 4.4.3
nodes: 424141/424141 valid (0 failures)
edges: 2000085/2000085 valid (0 failures)
KGX output is Biolink-compliant.
```

Exits non-zero when any record fails, so it can gate a release in CI. A missing or misspelled path is
reported as `file not found` and also exits non-zero — a file that was never read must never count as
a pass.

Edges carrying `effect_size` / `effect_type` are reported invalid until a `biolink-model` release
ships [#1774](https://github.com/biolink/biolink-model/pull/1774), because 4.4.3 declares neither on
`Association`. Those are counted separately as *pending* rather than treated as defects:

```text
edges: 1200000/2000085 valid (800085 failures; 800085 pending biolink-model support)
```

The strict count is what `ok` and the exit code use; the pending count is what
[`tablassert agent`](#agent) optimizes against, so a deliberate gap never reads as a modelling error.

---

## Typical workflow

1. Author a table config, then a graph config that references it.
2. `tablassert validate graph.yaml --schema graph` — fail fast on schema errors.
3. `tablassert build-kg graph.yaml` — produce KGX NDJSON + RIG (add `--qc` to audit mappings).
4. `tablassert validate-kgx -n MY_KG_1.0.0.nodes.ndjson -e MY_KG_1.0.0.edges.ndjson` — confirm the
   output validates against the Biolink Model before shipping it downstream.

## Next Steps

- **[Tutorial](tutorial.md)** — complete example walkthrough
- **[Configuration Guide](configuration/graph.md)** — YAML configuration reference
- **[Fullmap](fullmap.md)** — entity-resolution database build and schema
- **[Agent](agent.md)** — autonomous PMC → KG pipeline depth
