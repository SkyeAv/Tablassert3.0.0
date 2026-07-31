# CLI Reference

Tablassert extracts knowledge assertions from tabular data into KGX NDJSON. The `tablassert` app
exposes **four subcommands** — `agent`, `build-fullmap`, `build-kg`, `validate` — plus an app-level
`--version` flag. Run `tablassert --help` (or `<command> --help`) for the live surface.

## Command index

| Command | Use this to… |
| --- | --- |
| [`agent`](#agent) | Autonomously derive, build, audit, and improve KG configs from PMC articles |
| [`build-fullmap`](#build-fullmap) | Build the embedded fullmap redb used for entity resolution |
| [`build-kg`](#build-kg) | Build a KGX NDJSON knowledge graph from a YAML configuration |
| [`validate`](#validate) | Validate a graph or table configuration without executing it |

## App flags

These are flags on the root `tablassert` command, **not** subcommands.

| Flag | Description |
| --- | --- |
| `--version` | Print the installed package version as `tablassert <version>` (e.g. `tablassert 8.0.0`) and exit |
| `--help`, `-h` | Show help for the app or a subcommand |

!!! warning "Two different `--version`s"
    The app `--version` prints **Tablassert's package version**. The [`build-fullmap --version`](#build-fullmap)
    flag is unrelated: it selects a **RENCI BABEL snapshot date** (default `2026jul22`).

---

## agent

Use this to autonomously turn one or more PMC articles into audited, improved KG configs and graphs
(fetch → derive config → build + audit → improve until coverage maps). Requires the `[agent]` extra
(`pip install tablassert[agent]`).

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
| `--local`, `-l` | list[str] | No | `None` | Local payload: one DIR for all ids, or `PMCid=DIR` mappings; skips the PMC-AWS fetch (exit 2 on a missing DIR) |
| `--optimize`, `-o` | bool | No | `False` | Run GEPA prompt optimization and persist optimized instructions instead of running the supervisor |
| `--instructions-file` | Path | No | `None` | Load GEPA-optimized instructions from a prior `--optimize` run |
| `--instructions-out` | Path | No | `None` | Where `--optimize` writes optimized instructions (default `<state-dir>/optimized_instructions.yaml`) |
| `--max-metric-calls` | int | No | `8` | GEPA metric-call budget for `--optimize` |
| `--dataset` | Path | No | `None` | YAML/JSON list of `{table_summary, coverage_feedback}` examples for `--optimize` |

```bash
tablassert agent PMC11708054 --fullmap ./fullmap
```

!!! warning "Secrets"
    Model config comes from the flags above **or** the `TABLASSERT_AGENT_*` environment variables
    (explicit flags win). Secrets are **never** hardcoded or defaulted — a missing value fails loud
    (exit 2) **before** any model is built.

---

## build-fullmap

Use this to build the embedded `fullmap.redb` entity-resolution database from RENCI BABEL exports
(download class + synonym files, then build a single redb).

```bash
tablassert build-fullmap [ARGS]
```

| Option | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `--output`, `-o` | Path | No | `./fullmap/data/fullmap.redb` | Path to write the built redb file |
| `--cache`, `-c` | Path | No | `./fullmap/downloads` | Directory for downloaded BABEL files (`classes/`, `synonyms/`) |
| `--version`, `-v` | str | No | `2026jul22` | BABEL snapshot date to fetch (a RENCI stamp, **not** Tablassert's version) |
| `--threads`, `-t` | int | No | `None` (auto) | Worker threads; auto-capped by memory on Linux (`/proc/meminfo`), else ~90% of CPUs |

```bash
tablassert build-fullmap --output /data/fullmap/fullmap.redb
```

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
| `--qc`, `-q` | Flag | No | `False` | Audit resolved mappings (exact → fuzzy → BioBERT) so low-confidence edges are flagged; requires the `[qc]` extra |
| `--log`, `-l` | Flag | No | `False` | Enable verbose per-section logging |
| `--head`, `-hd` | Flag | No | `False` | Fast output-shape preview: ≤5 random rows/section, cached to `.head.parquet`, never clobbers a full build |

```bash
tablassert build-kg graph.yaml --qc --log
```

Output is written to the current directory as `{name}_{version}.nodes.ndjson`,
`{name}_{version}.edges.ndjson`, and `{name}_{version}.RIG.yaml`; intermediate parquet lands in
`.tablassert/store/`. See [Graph Configuration](configuration/graph.md).

??? info "Build progress & stages"
    The build runs six parallel stages — Loading Tables → Extracting Sections → Building TCode →
    Collecting Instructions → Building Subgraphs → Compiling Graph — under a three-row live progress
    block (stage header; section bar with count/elapsed/ETA; in-flight item detail). Each completed
    stage prints a green `✓ Stage N · NAME · elapsed` line above the live block. During Building
    Subgraphs the detail line also shows the per-section phase (`load`, `filter`, `clean`, `encode`,
    `resolve`, `qc`, `edge`, `provenance`, `significance`, `finalize`, `write`).

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

## Typical workflow

1. Author a table config, then a graph config that references it.
2. `tablassert validate graph.yaml --schema graph` — fail fast on schema errors.
3. `tablassert build-kg graph.yaml` — produce KGX NDJSON + RIG (add `--qc` to audit mappings).

## Next Steps

- **[Tutorial](tutorial.md)** — complete example walkthrough
- **[Configuration Guide](configuration/graph.md)** — YAML configuration reference
- **[Fullmap](fullmap.md)** — entity-resolution database build and schema
- **[Agent](agent.md)** — autonomous PMC → KG pipeline depth
