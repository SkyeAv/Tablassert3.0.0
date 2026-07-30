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
| `--map-threshold`, `-mt` | float | No | `0.8` | Coverage an article must reach to be MAPPED |
| `--qc-threshold`, `-qt` | float | No | `0.9` | Target QC pass rate |
| `--max-improve-iters`, `-mi` | int | No | `3` | Max deterministic improve iterations per article |
| `--state-dir`, `-sd` | Path | No | `.tablassert-agent` | Checkpoint/resume workspace directory |
| `--executor`, `-e` | {local, docker} | No | `local` | Code-execution backend; `docker` is the hardened sandbox |
| `--backend`, `-b` | {openai, litellm} | No | `openai` | Model backend |
| `--no-fetch`, `-nf` | Flag | No | `False` | Skip PMC download; reuse the already-fetched snapshot |

```bash
tablassert agent PMC11708054 --fullmap ./fullmap --executor docker
```

!!! warning "Secrets & sandboxing"
    Model config comes from the flags above **or** the `TABLASSERT_AGENT_*` environment variables
    (explicit flags win). Secrets are **never** hardcoded or defaulted — a missing value fails loud
    (exit 2) **before** any model is built. `--executor docker` is the hardened sandbox; the default
    `local` executor runs in-process and is **not** a security boundary (use `docker` for untrusted input).

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
tablassert build-kg CONFIGURATION-FILE [ARGS]
```

By default the positional `CONFIGURATION-FILE` (also `--configuration-file`) is a **graph** YAML.

| Option | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `CONFIGURATION-FILE` (`--configuration-file`) | Path | Yes | — | Graph YAML (or a table YAML with `--table-config`) |
| `--release`, `-r` | Flag | No | `False` | Emit a slim, significant-only graph (drops `biolink:not_significant` edges before resolution) |
| `--qc`, `-q` | Flag | No | `False` | Audit resolved mappings (exact → fuzzy → BioBERT) so low-confidence edges are flagged; requires the `[qc]` extra |
| `--log`, `-l` | Flag | No | `False` | Enable verbose per-section logging |
| `--head`, `-hd` | Flag | No | `False` | Fast output-shape preview: ≤5 random rows/section, cached to `.head.parquet`, never clobbers a full build |
| `--table-config`, `-tc` | Flag | No | `False` | Build/test one table (Section) config without writing a graph config (wrapped in a throwaway `TEMP_KG` graph) |
| `--fullmap`, `-f` | Path | No | `./fullmap` | Fullmap path for the throwaway `TEMP_KG` graph when `--table-config` is passed |

```bash
tablassert build-kg graph.yaml --qc --log
tablassert build-kg table-config.yaml --table-config --fullmap ./fullmap
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

Use this to validate a graph or table configuration without running the build — ideal for CI and
pre-commit hooks. Both forms work: `tablassert validate <file>` and `tablassert validate -f <file>`.

```bash
tablassert validate CONFIGURATION-FILE
tablassert validate -f CONFIGURATION-FILE
```

| Option | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `CONFIGURATION-FILE` (`--configuration-file`, `-f`) | Path | Yes | — | Graph **or** table configuration to validate |

The config kind is detected from the YAML: a mapping with a top-level `tables` key is a **graph**
config (validates the `Graph` model **and** every referenced table); anything else is a **table**
config (validates section syntax). Exits non-zero on any schema error. See
[Table Configuration](configuration/table.md) and [Graph Configuration](configuration/graph.md).

```bash
tablassert validate table-config.yaml
tablassert validate graph.yaml
```

---

## Typical workflow

1. Author a table config, then a graph config that references it.
2. `tablassert validate graph.yaml` — fail fast on schema errors.
3. `tablassert build-kg graph.yaml` — produce KGX NDJSON + RIG (add `--qc` to audit mappings).

## Next Steps

- **[Tutorial](tutorial.md)** — complete example walkthrough
- **[Configuration Guide](configuration/graph.md)** — YAML configuration reference
- **[Fullmap](fullmap.md)** — entity-resolution database build and schema
- **[Agent](agent.md)** — autonomous PMC → KG pipeline depth
