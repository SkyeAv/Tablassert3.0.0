# Architecture Research

**Domain:** Declarative tabular-to-knowledge-graph ETL for biomedical/Translator ecosystems
**Researched:** 2026-03-12
**Confidence:** HIGH

## Standard Architecture

### System Overview

```text
┌────────────────────────────────────────────────────────────────────────────┐
│ Control Plane                                                              │
├────────────────────────────────────────────────────────────────────────────┤
│  CLI/API Trigger -> Run Orchestrator -> Config Compiler -> Run Manifest   │
└────────────────────────────────────────────────────────────────────────────┘
                                  |
                                  v
┌────────────────────────────────────────────────────────────────────────────┐
│ Ingestion + Semantic Normalization                                         │
├────────────────────────────────────────────────────────────────────────────┤
│  Source Adapters -> Row Parser -> Mapping/Transform Engine -> ID Resolver │
│                                            |                               │
│                                            v                               │
│                                  Biolink/KGX Validation Gate              │
└────────────────────────────────────────────────────────────────────────────┘
                                  |
                                  v
┌────────────────────────────────────────────────────────────────────────────┐
│ Graph Assembly + Delivery                                                  │
├────────────────────────────────────────────────────────────────────────────┤
│  Node/Edge Builder -> Dedup/Prune/QC -> Exporters (KGX NDJSON/TSV/JSONL) │
│                                     -> Provenance + Metrics Reports        │
└────────────────────────────────────────────────────────────────────────────┘
                                  |
                                  v
┌────────────────────────────────────────────────────────────────────────────┐
│ Storage Boundaries                                                         │
├────────────────────────────────────────────────────────────────────────────┤
│  Raw snapshots | Staging tables/files | Curated graph artifacts | Reports │
└────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| Run Orchestrator | Build execution plan, enforce stage ordering, emit run IDs and checkpoints | Python CLI entrypoint + explicit stage DAG in code |
| Config Compiler | Parse YAML declarative mapping, validate schema, compile to executable transform plan | Pydantic/LinkML-backed config model with strict validation |
| Source Adapters | Read tabular/JSON/XML remote or local sources, infer/validate columns, normalize input types | Pandas/Polars readers plus adapter classes per source kind |
| Mapping/Transform Engine | Convert source records into node/edge candidates with deterministic rules | Rule executor over compiled mapping spec + pure transform functions |
| Identifier Resolution Service Client | Resolve source IDs/names to canonical CURIEs (dbSERT/Node Norm/SSSOM mappings) | Multi-stage resolver client with exact/fuzzy/embedding fallback |
| Biolink/KGX Validation Gate | Reject or quarantine non-compliant records (required fields, predicates, categories, provenance) | KGX validator + Biolink model toolkit checks + custom quality rules |
| Graph Assembly + QC | Merge partitions, deduplicate entities/edges, prune dangling edges/singletons, compute metrics | DuckDB-based set operations for large files, archive rejected rows |
| Export + Publication | Write KGX NDJSON plus optional TSV/JSONL, emit manifests and run metadata | Sink layer with deterministic sort/order and checksum manifest |

## Recommended Project Structure

```text
src/tablassert/
├── cli/                    # command handlers and argument parsing
├── config/                 # schema for YAML specs and compiler
├── adapters/               # source readers (csv/tsv/json/jsonl/xml/http)
├── transforms/             # declarative rule engine + typed transform ops
├── resolution/             # entity resolution clients and score policies
├── graph/
│   ├── build/              # node/edge assembly and merge logic
│   ├── validate/           # Biolink/KGX + custom checks
│   └── export/             # NDJSON/TSV/JSONL sinks and manifests
├── storage/                # cache, staging, artifact paths, retention policies
├── observability/          # run metrics, QC reports, lineage events
└── orchestration/          # stage sequencing, retry, idempotency controls
```

### Structure Rationale

- **Separation by lifecycle stage:** Ingest, transform, resolve, validate, and export stay decoupled so each stage can be tested independently.
- **Strict boundary around semantic contracts:** `config/` and `graph/validate/` centralize Biolink/KGX constraints so schema drift is caught early.
- **Adapter isolation:** New sources should require new adapter modules, not changes to core graph-building logic.
- **Resolver isolation:** Canonicalization logic evolves quickly; keeping it separate prevents orchestration churn.

## Architectural Patterns

### Pattern 1: Compiled Declarative Pipeline

**What:** Parse YAML spec once into an internal executable plan, then execute against input partitions.
**When to use:** Multiple datasets with similar mapping shape and frequent reruns.
**Trade-offs:** More upfront compiler complexity; much better reproducibility and testability.

**Example:**
```python
plan = compile_config("sources/ctd.yaml")
for row in source_reader(plan.source):
    for op in plan.row_ops:
        row = op.apply(row)
    emit_candidates(plan.emitters, row)
```

### Pattern 2: Stage-Gated Quality Control

**What:** Enforce hard validation gates between major phases (post-ingest, post-resolution, pre-export).
**When to use:** Translator-facing outputs where malformed edges or weak provenance are expensive downstream.
**Trade-offs:** More rejected records and operational noise initially; fewer downstream regressions.

**Example:**
```python
assert_required_columns(table, ["id", "category"])
assert_biolink_predicates(edges)
assert_min_resolution_confidence(edges, threshold=0.80)
```

### Pattern 3: Lakehouse-Style Graph Staging (DuckDB-first)

**What:** Load node/edge candidates into DuckDB staging tables, then perform merge/dedup/prune/export with SQL.
**When to use:** Medium-to-large biomedical files and iterative QC/debug workflows.
**Trade-offs:** Adds a storage layer; dramatically simplifies set-based graph cleanup and reporting.

## Component Boundaries (Who Talks To What)

| Boundary | Communication | Notes |
|----------|---------------|-------|
| CLI -> Orchestrator | Direct function/API call | CLI only starts runs; it must not perform ETL logic directly |
| Orchestrator -> Config Compiler | In-process API | Compiler returns immutable plan object tied to run ID |
| Orchestrator -> Source Adapters | In-process API + file/HTTP I/O | Adapter failures are retriable and isolated per source |
| Transform Engine -> Resolver | In-process client + local DB/HTTP | Resolver returns canonical CURIE + confidence + provenance |
| Transform/Resolver -> Validation Gate | Record stream handoff | Validation decides pass, quarantine, or fail-fast |
| Validation Gate -> Graph Assembly | Staging table/file contract | Only valid, typed candidate records cross boundary |
| Graph Assembly -> Exporters | Table/file handoff | Exporters are pure sinks; no mutation of upstream records |
| Exporters -> Metrics/Lineage | Event emission | Run metrics and manifests are append-only outputs |

## Data Flow

### End-to-End Flow

```text
Run Request
  -> Config Load/Compile
  -> Source Fetch + Parse
  -> Row-Level Transform
  -> Candidate Nodes/Edges
  -> Identifier Resolution + Mapping
  -> Biolink/KGX Validation
  -> Graph Merge + Dedup + Prune
  -> KGX NDJSON Export
  -> QC/Lineage Reports
```

### Key Data Flows

1. **Configuration-to-execution flow:** YAML mapping spec compiles into a deterministic run plan; all downstream stages consume that plan, not raw YAML.
2. **Entity canonicalization flow:** Raw source identifiers flow through resolver stages (exact -> fuzzy -> embedding) and emerge as canonical CURIEs with confidence/provenance.
3. **Validation and quarantine flow:** Non-compliant records are diverted to quarantine datasets with rule violation metadata rather than silently dropped.
4. **Artifact publication flow:** Final node/edge artifacts and reports are versioned by run ID and emitted atomically to avoid partial graph publication.

## Suggested Build Order (Roadmap Dependencies)

1. **Semantic contract layer first** (`config/`, Biolink/KGX constraints)
   - Dependency rationale: all downstream components need stable field contracts and validation rules.
2. **Source adapters + config compiler**
   - Dependency rationale: transform engine requires typed, normalized row inputs and compiled mapping plans.
3. **Transform engine + candidate emitter**
   - Dependency rationale: resolution and graph assembly need standardized node/edge candidates.
4. **Identifier resolution subsystem**
   - Dependency rationale: graph merge and validator need canonical IDs to avoid duplicate entities.
5. **Validation gate + quarantine pipeline**
   - Dependency rationale: prevents invalid artifacts from contaminating graph assembly/export stages.
6. **Graph assembly (dedup/prune/merge) + exporters**
   - Dependency rationale: only after canonical, validated data exists can stable KGX outputs be produced.
7. **Observability and reliability hardening** (metrics, retries, lineage, regression tests)
   - Dependency rationale: hardening is most effective once end-to-end path exists.

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| Prototype to small production (single team, periodic runs) | Single process CLI + local staging + deterministic artifacts is sufficient |
| Growing sources and file volume (many ingests, larger tables) | Shift merge/QC to DuckDB staging, partition by source/date, cache resolver outputs |
| High-volume continuous refresh | Separate control plane from workers, queue per-source jobs, persist run metadata and quarantine stores |

### Scaling Priorities

1. **First bottleneck:** Identifier resolution throughput and cache misses; fix with layered caches and batched lookups.
2. **Second bottleneck:** Dedup/prune over very large edge sets; fix with staged SQL processing and partitioned exports.

## Anti-Patterns

### Anti-Pattern 1: Mixing Extraction and Ontology Decisions in One Function

**What people do:** Parse source rows and assign Biolink categories/predicates inline in adapter code.
**Why it's wrong:** Couples source-specific parsing to semantic policy, making schema changes high-risk.
**Do this instead:** Keep adapters syntax-level only; centralize semantic mapping rules in compiled transform plan.

### Anti-Pattern 2: Late Validation at Export Time Only

**What people do:** Run validation only after full graph assembly.
**Why it's wrong:** Invalid records propagate far, making debugging and rollback expensive.
**Do this instead:** Add stage gates with quarantine after ingestion and after resolution.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| Biolink Model | Version-pinned schema/toolkit lookups | Pin model version per run to keep outputs reproducible |
| KGX tooling | Validator and format sinks | Use as compliance gate for Translator-oriented graph exchange |
| Identifier services (dbSERT, Node Normalizer, SSSOM mappings) | Resolver client abstraction with pluggable backends | Return canonical CURIE + confidence + source provenance |
| Translator TRAPI ecosystem (future serving path) | Optional downstream API adapter from built KG | Keep ETL build boundary separate from query API boundary |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `adapters/` <-> `transforms/` | Typed record objects | Adapter outputs must be schema-checked before transform |
| `transforms/` <-> `resolution/` | Candidate entity DTO + context | Resolution policy should be data-driven, not hardcoded in transforms |
| `resolution/` <-> `graph/build/` | Canonicalized node/edge candidates | Build stage assumes canonical IDs and confidence annotations exist |
| `graph/validate/` <-> `graph/export/` | Validated table/file contract | Export stage is side-effect-only and never repairs invalid data |

## Sources

- KGX documentation (architecture, source/sink/validator modules): https://kgx.readthedocs.io/en/latest/ (HIGH)
- Koza docs (declarative ingest model and DuckDB graph operations pipeline): https://koza.monarchinitiative.org/ and https://koza.monarchinitiative.org/graph-operations/explanation/architecture/ (HIGH)
- Koza Biolink compliance details (required KGX fields, compliance checks): https://koza.monarchinitiative.org/graph-operations/explanation/biolink-compliance/ (HIGH)
- Translator technical docs (federated architecture, KPs, TRAPI/SRI role): https://ncatstranslator.github.io/TranslatorTechnicalDocumentation/architecture/ and https://ncatstranslator.github.io/TranslatorTechnicalDocumentation/architecture/sri/trapi/ (HIGH)
- Biolink model docs (semantic contract for categories/predicates/slots): https://biolink.github.io/biolink-model/ (HIGH)
- LinkML validation docs via Context7 (schema-driven validation patterns): https://github.com/linkml/linkml/blob/main/docs/data/validating-data.rst (HIGH)

---
*Architecture research for: declarative tabular-to-KG ETL in Translator-aligned bioinformatics systems*
*Researched: 2026-03-12*
