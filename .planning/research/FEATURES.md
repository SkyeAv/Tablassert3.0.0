# Feature Research

**Domain:** Declarative tabular-to-knowledge-graph ETL for biomedical/NCATS Translator ecosystems
**Researched:** 2026-03-12
**Confidence:** MEDIUM

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = product feels incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Declarative source configuration (YAML) for tabular/JSON inputs | Koza-style ingest configs and Translator build teams expect config-first onboarding, not custom parser code per source | MEDIUM | Must support schema, field mapping, filters, metadata, and remote/local sources |
| Biolink-aware node/edge modeling and validation | Translator components are built around Biolink categories/predicates and reject low-semantic-quality graphs | HIGH | Include preflight validation for categories/predicates/required slots before export |
| KGX-compliant output (TSV/NDJSON) and format conversion | KGX is the common exchange surface for Translator graph build and downstream serving pipelines | MEDIUM | Export KGX nodes/edges plus JSONL/NDJSON pathways for build/runtime workflows |
| Identifier normalization and synonym handling | Translator interoperability depends on CURIE harmonization and equivalent identifier consolidation | HIGH | Integrate Node Normalizer/dbSERT-style mapping with explicit confidence and fallback behavior |
| Provenance and evidence capture on edges | Translator requires source transparency (`infores`, upstream sources, evidence/publication attributes) | HIGH | Preserve both primary source and aggregator lineage; do not drop upstream IDs during merge |
| CLI-first reproducible runs (validate/build/report) | Existing bioinformatics ETL users rely on scripted, automatable pipelines instead of GUI workflows | MEDIUM | Commands should include validate, transform, compile, and quality report outputs |

### Differentiators (Competitive Advantage)

Features that set the product apart. Not required, but valuable.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Multi-stage entity resolution QC (exact -> fuzzy -> embedding) with policy gates | Converts noisy source data into higher-confidence graph entities while still surfacing uncertain candidates | HIGH | Strong fit to Tablassert core value; expose thresholds and accept/reject audit trail |
| Incremental graph rebuilds with diff-aware outputs | Dramatically lowers rebuild time/cost for frequently updated biomedical sources | HIGH | Use snapshot metadata + change detection to only rebuild affected partitions |
| Built-in graph integrity operations (prune dangling edges, deduplicate, schema harmonization) | Reduces downstream cleanup burden that teams currently script manually in KGX/Koza-adjacent tooling | MEDIUM | Treat as first-class pipeline stages, not ad hoc post-processing scripts |
| Quality scorecards and release gates (Biolink/TRAPI readiness checks) | Gives teams objective pass/fail metrics before publishing a KP or shipping a KG release | MEDIUM | Include semantic, structural, and provenance completeness scores |
| Canonicalized Translator-ready packaging profile | Shortens time from source ETL to deployable Translator KP assets (data + metadata contract) | MEDIUM | Opinionated output profile for Translator compatibility rather than generic export flexibility |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create problems.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Full GUI orchestration/dashboard in v1 | Stakeholders want "non-technical" onboarding | High scope drag, duplicates mature workflow tools, conflicts with current CLI-first user base and milestone goals | Keep CLI + documented YAML patterns + example templates; add thin optional UI later if demand is proven |
| Real-time streaming ingestion for all sources | "Always-fresh" graph sounds attractive | Most biomedical sources are periodic/batch and provenance QA is batch-oriented; streaming adds hard consistency/debugging issues | Support scheduled incremental batch updates with deterministic run manifests |
| Auto-accept low-confidence entity mappings | Appears to improve coverage quickly | Inflates false positives and damages trust in downstream reasoning results | Require confidence thresholds + quarantine bucket + human-review export |
| Over-generalized plugin ecosystem before core hardening | Feels extensible and future-proof | Premature abstraction increases maintenance burden and weakens reliability in core transformation path | Stabilize a narrow, well-tested extension API after MVP pipeline is robust |

## Feature Dependencies

```text
Declarative source configuration
    └──requires──> Schema/field validation
                       └──requires──> Biolink-aware node/edge modeling
                                            └──requires──> KGX/NDJSON export

Identifier normalization
    └──requires──> Mapping resources (Node Normalizer/dbSERT) + confidence policy
                       └──enables──> Provenance/evidence-complete edge output

Quality scorecards/release gates
    └──requires──> Biolink validation + provenance capture + integrity operations

Incremental rebuilds
    └──requires──> Deterministic run manifests + stable IDs + change detection

Auto-accept low-confidence mapping ──conflicts──> Quality scorecards/release gates
```

### Dependency Notes

- **Declarative config requires schema/field validation:** config-first pipelines fail silently without strict contract checks.
- **Biolink modeling precedes export:** valid KGX structure is not enough; semantic correctness must be established before compilation.
- **Normalization enables trustworthy provenance:** equivalent-ID consolidation is needed to avoid duplicate/conflicting lineage on edges.
- **Quality gates depend on integrity + provenance:** release scoring is only meaningful when structural and source metadata checks are present.
- **Low-confidence auto-accept conflicts with quality gates:** permissive mapping undermines pass/fail semantics and downstream trust.

## MVP Definition

### Launch With (v1)

Minimum viable product - what's needed to validate the concept.

- [ ] Declarative YAML ingest + strict schema validation - core onboarding path without custom ETL coding
- [ ] Biolink-aware transform + KGX NDJSON/TSV output - essential Translator-compatible deliverable
- [ ] Identifier normalization with configurable confidence thresholds - baseline interoperability and quality control
- [ ] Provenance/evidence retention in compiled edges - required for Translator trust and auditability
- [ ] CLI workflow (`validate`, `build`, `report`) with deterministic artifacts - reproducible operations for bioinformatics teams

### Add After Validation (v1.x)

Features to add once core is working.

- [ ] Incremental rebuild mode - add once full rebuild correctness and stable IDs are proven
- [ ] Graph integrity operations (prune/dedupe/harmonize) as composable steps - add when users start scaling source count
- [ ] Quality scorecards with policy thresholds - add after baseline metrics from real runs are available

### Future Consideration (v2+)

Features to defer until product-market fit is established.

- [ ] Optional lightweight UI for run visibility - defer until repeated user demand from non-CLI operators
- [ ] Advanced plugin SDK for custom resolvers/transforms - defer until extension points stabilize under production use
- [ ] Near-real-time ingestion for selected feeds - defer until incremental batch pipeline is operationally mature

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Declarative ingest + schema validation | HIGH | MEDIUM | P1 |
| Biolink-aware transform + KGX output | HIGH | HIGH | P1 |
| Identifier normalization + confidence policy | HIGH | HIGH | P1 |
| Provenance/evidence retention | HIGH | HIGH | P1 |
| CLI reproducible workflow | HIGH | MEDIUM | P1 |
| Integrity operations (prune/dedupe/harmonize) | MEDIUM | MEDIUM | P2 |
| Incremental rebuilds | HIGH | HIGH | P2 |
| Quality scorecards/gates | MEDIUM | MEDIUM | P2 |
| Optional UI | LOW | HIGH | P3 |

**Priority key:**
- P1: Must have for launch
- P2: Should have, add when possible
- P3: Nice to have, future consideration

## Competitor Feature Analysis

| Feature | Koza | KGX | Tablassert Approach |
|---------|------|-----|---------------------|
| Declarative ingest config | YAML + semi-declarative transforms | Primarily transform/merge operations once graph data exists | YAML-first ingest with stricter contract validation for source-to-graph compilation |
| Graph exchange and format interoperability | Strong KGX ingest/ops support | Core strength: broad KG format conversion and merge/validate | Keep KGX interoperability, optimize for Translator-targeted NDJSON deliverables |
| Biolink compliance validation | Present via ecosystem integration and graph ops docs | Native validator and Biolink-focused checks | Treat Biolink validation as non-optional release gate |
| Translator semantics/provenance readiness | Indirect (toolkit level) | Translator-oriented but generic utility surface | Opinionated Translator-ready profile (infores lineage + QC outputs + packaging) |
| Resolution quality control | Mapping/translation tables; policy is user-defined | Minimal domain-specific entity QC policy | Multi-stage resolution pipeline with explicit confidence governance |

## Sources

- Koza docs and README (official): https://koza.monarchinitiative.org/ ; https://github.com/monarch-initiative/koza (HIGH)
- KGX docs and README (official): https://biolink.github.io/kgx/ ; https://github.com/biolink/kgx (HIGH)
- Biolink Model docs (official): https://biolink.github.io/biolink-model/ (HIGH)
- Translator Technical Documentation (official): https://ncatstranslator.github.io/TranslatorTechnicalDocumentation/ (HIGH)
- RTX-KG2 Translator page (official project docs): https://ncatstranslator.github.io/TranslatorTechnicalDocumentation/architecture/kp/rtx-kg2/ (MEDIUM)
- Node Normalizer README (official repo): https://github.com/TranslatorSRI/NodeNormalization (MEDIUM)
- Reasoner Validator README (official repo): https://github.com/NCATSTranslator/reasoner-validator (MEDIUM)
- Project context: `.planning/PROJECT.md` (HIGH)

---
*Feature research for: declarative tabular-to-KG ETL (bioinformatics/Translator context)*
*Researched: 2026-03-12*
