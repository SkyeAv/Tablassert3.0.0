# Stack Research

**Domain:** Declarative tabular-to-knowledge-graph ETL for NCATS Translator-compatible bioinformatics pipelines
**Researched:** 2026-03-12
**Confidence:** HIGH

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended | Confidence |
|------------|---------|---------|-----------------|------------|
| Python | 3.13.x | Runtime for CLI and ETL execution | Translator/LinkML/KGX ecosystem is Python-first, and your project constraint already targets Python 3.13; this maximizes compatibility with Biolink tooling while keeping modern typing/runtime features. | HIGH |
| LinkML | 1.10.0 | Declarative schema/model layer for YAML-driven data contracts | This is the canonical schema framework used by Biolink Model and related ecosystem tooling; it directly supports validation and generation across JSON/RDF/TSV workflows. | HIGH |
| Biolink Model | 4.3.7 | Semantic contract for Translator-aligned node/edge categories and predicates | Translator compatibility depends on Biolink semantics; pinning Biolink avoids silent predicate/category drift across releases. | HIGH |
| KGX | 2.6.0 | Graph exchange/output layer (KGX NDJSON + validation/conversion utilities) | KGX is built for Biolink-aligned KG interchange and is explicitly used in Translator-adjacent workflows; it gives direct path to required NDJSON artifacts. | HIGH |
| pandas | 3.0.1 | Tabular ingestion/transformation baseline | Still the default interoperability layer across bioinformatics ETL libraries; broadest connector and ecosystem compatibility for heterogeneous tabular sources. | HIGH |
| PyArrow | 23.0.1 | Columnar memory + fast IO for parquet/IPC and DataFrame interchange | Arrow gives the fastest common interchange boundary between pandas, DuckDB, and downstream analytics without custom serializers. | HIGH |
| DuckDB | 1.5.0 | SQL-based local analytical engine for heavy joins/dedup/filter over tabular inputs | Best single-node choice for reproducible ETL transforms over large files; avoids Spark-level ops complexity while remaining fast on local/CI runners. | HIGH |

### Supporting Libraries

| Library | Version | Purpose | When to Use | Confidence |
|---------|---------|---------|-------------|------------|
| bmt (Biolink Model Toolkit) | 1.4.6 | Programmatic access to Biolink classes/slots/ancestors and mapping helpers | Use for runtime predicate/category checks and CURIE-to-Biolink lookups during normalization and QC. | HIGH |
| sssom | 0.4.18 | Standardized ontology/entity mapping format + tooling | Use when your resolver emits or consumes mapping tables that need provenance/confidence/justification metadata. | HIGH |
| pandera | 0.29.0 | DataFrame-level contracts and validation checks | Use after ingestion and before KG compilation to fail fast on schema drift and value-domain violations. | MEDIUM |
| pydantic | 2.12.5 | Strongly-typed config/runtime object validation | Use for pipeline config models and strict validation of YAML-expanded runtime settings. | HIGH |
| typer | 0.24.1 | CLI interface and command ergonomics | Use for a maintainable CLI-first UX with typed commands/options and composable subcommands. | HIGH |
| rdflib | 7.6.0 | RDF parsing/serialization interoperability | Use only when a source/consumer boundary requires RDF/OWL conversion beyond KGX defaults. | MEDIUM |
| frictionless | 5.18.1 | Tabular package/resource validation and metadata checks | Use when ingesting external CSV/TSV packages that benefit from explicit tabular metadata QA before transform. | MEDIUM |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| pytest | Test execution for ETL stages | Write stage-level tests (ingest, transform, resolve, emit) with golden KGX fixtures.
| ruff | Linting + formatting | Enforce consistent style and catch import/typing issues early in CI.
| mypy | Static typing checks | Particularly useful for config objects and transformation boundaries.

## Installation

```bash
# Core stack
pip install \
  "linkml==1.10.*" \
  "biolink-model==4.3.*" \
  "kgx==2.6.*" \
  "pandas==3.0.*" \
  "pyarrow==23.0.*" \
  "duckdb==1.5.*"

# Supporting libraries
pip install \
  "bmt==1.4.*" \
  "sssom==0.4.*" \
  "pandera==0.29.*" \
  "pydantic==2.12.*" \
  "typer==0.24.*" \
  "rdflib==7.6.*" \
  "frictionless==5.18.*"

# Dev dependencies
pip install -U pytest ruff mypy
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| DuckDB | Spark | Only when dataset size/SLAs truly require distributed execution across a cluster.
| pandas + PyArrow | Polars | Reasonable if your team is already Polars-native and willing to manage Translator-tooling interop edges.
| pandera | Great Expectations | Choose GX when you need centralized data-quality governance UI/workflows across many teams.
| Typer | Click directly | Choose raw Click only if you need low-level control and can trade off typed CLI ergonomics.

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Building around ad-hoc custom graph JSON (non-KGX contract) | Breaks Translator ecosystem interoperability and increases integration debt. | Emit/validate KGX NDJSON with Biolink-aware checks.
| Treating Neo4j as the canonical build artifact | Adds unnecessary operational coupling for an ETL product whose required output is exchange format files. | Keep file-first build artifacts (KGX NDJSON/TSV), load into graph DB only as downstream optional step.
| Airflow-first orchestration for a single-node CLI ETL | Significant operational overhead and slower iteration for greenfield CLI-focused pipelines. | Keep in-process orchestration; add external scheduler only when multi-tenant scheduling is a real requirement.
| Pydantic v1-era config patterns | V1 is legacy and increasingly misaligned with current Python/runtime ecosystem. | Standardize on Pydantic v2 models.

## Stack Patterns by Variant

**If you are single-node, CLI-first (recommended default):**
- Use `pandas + pyarrow + duckdb` for transform/query layers
- Keep orchestration in Python process + Typer commands
- Emit KGX NDJSON as the canonical artifact

**If you must integrate heterogeneous ontology mappings at scale:**
- Add `sssom`-native mapping tables as first-class ETL assets
- Use `bmt` checks in QC gates to enforce Biolink semantic validity
- Persist mapping provenance/confidence alongside graph edges

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `pandas==3.0.*` | `python>=3.11` | Safe on project baseline Python 3.13.
| `pyarrow==23.0.*` | `python>=3.10` | Works with pandas 3 + DuckDB interchange patterns.
| `duckdb==1.5.*` | `python>=3.10` | Works on Python 3.13; good local analytics default.
| `linkml==1.10.*` | `python>=3.10` | Aligns with modern Python and Biolink ecosystem tooling.
| `kgx==2.6.*` | `python>=3.9` | Supports Python 3.13; designed for Biolink/KGX workflows.
| `bmt==1.4.*` | `python>=3.9` | Pairs with pinned Biolink model versions for consistent lookups.
| `biolink-model==4.3.*` | `python>=3.9` | Pin this to control semantic schema drift between releases.

## Sources

- Context7 `/websites/linkml_io_linkml` — LinkML validation/generation capabilities and tabular/RDF support (HIGH)
- Context7 `/biolink/biolink-model` — Biolink model usage patterns and Translator alignment context (HIGH)
- Context7 `/unionai-oss/pandera` — DataFrame validation backends and role in ETL quality gates (MEDIUM)
- Context7 `/frictionlessdata/frictionless-py` — Tabular validation and metadata-driven checks (MEDIUM)
- PyPI JSON `https://pypi.org/pypi/kgx/json` — latest version `2.6.0`, Python requirement metadata (HIGH)
- PyPI JSON `https://pypi.org/pypi/linkml/json` — latest version `1.10.0`, Python requirement metadata (HIGH)
- PyPI project `https://pypi.org/project/biolink-model/` — latest release `4.3.7` (Feb 25, 2026) (HIGH)
- PyPI project `https://pypi.org/project/bmt/` — latest release `1.4.6` (Oct 6, 2025), note on Biolink pinning (HIGH)
- PyPI project `https://pypi.org/project/sssom/` — latest release `0.4.18` (Dec 19, 2025) (HIGH)
- PyPI JSON `https://pypi.org/pypi/pandas/json` — latest version `3.0.1`, Python requirement metadata (HIGH)
- PyPI JSON `https://pypi.org/pypi/pyarrow/json` — latest version `23.0.1`, Python requirement metadata (HIGH)
- PyPI JSON `https://pypi.org/pypi/duckdb/json` — latest version `1.5.0`, Python requirement metadata (HIGH)
- PyPI JSON `https://pypi.org/pypi/pydantic/json` — latest version `2.12.5`, Python requirement metadata (HIGH)
- PyPI JSON `https://pypi.org/pypi/typer/json` — latest version `0.24.1`, Python requirement metadata (HIGH)
- PyPI JSON `https://pypi.org/pypi/pandera/json` — latest version `0.29.0` (MEDIUM)
- PyPI JSON `https://pypi.org/pypi/frictionless/json` — latest version `5.18.1` (MEDIUM)

---
*Stack research for: declarative tabular-to-knowledge-graph ETL (bioinformatics/Translator context)*
*Researched: 2026-03-12*
