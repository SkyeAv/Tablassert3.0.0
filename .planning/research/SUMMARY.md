# Project Research Summary

**Project:** Tablassert
**Domain:** Declarative tabular-to-knowledge-graph ETL for NCATS Translator-aligned biomedical pipelines
**Researched:** 2026-03-12
**Confidence:** MEDIUM-HIGH

## Executive Summary

Tablassert is a CLI-first, declarative ETL compiler: it should turn YAML-described tabular sources into deterministic, Biolink-valid KGX artifacts with strong provenance and resolution quality controls. The research consistently points to a Python-first, file-first architecture (not graph-DB-first) with strict semantic contracts at every stage. Teams that succeed in this domain compile configs into executable plans, gate each stage with validation, and publish reproducible artifacts plus run metadata.

The recommended implementation path is opinionated: pin Biolink/LinkML/KGX versions, enforce canonical CURIE policies early, and treat identifier resolution plus provenance as release-gated quality domains rather than optional cleanup. Build around `pandas + pyarrow + duckdb` for single-node throughput, with KGX NDJSON as the canonical output and TSV/JSONL as secondary sinks. Keep orchestration in-process behind a typed CLI, and separate adapters, transform rules, resolution logic, validation gates, and exporters into hard boundaries.

Primary risk is silent semantic degradation: outputs may look structurally valid but fail Translator trust requirements due to identifier drift, weak predicate mapping, missing InfoRes lineage, or overly permissive fuzzy resolution. Mitigation is to front-load contract enforcement (Biolink-pinned validation, namespace linting, deterministic manifests), then harden with phase-gated QC, quarantine flows, and precision-first resolution policies.

## Key Findings

### Recommended Stack

The stack is mature and strongly validated for this domain: Python 3.13 with LinkML, Biolink Model, and KGX as semantic backbone; pandas/PyArrow/DuckDB as execution backbone; and bmt/sssom/pydantic/typer as quality and ergonomics layers. This combination matches Translator ecosystem reality while keeping local CLI operation fast and reproducible.

**Core technologies:**
- **Python 3.13.x**: runtime for CLI and ETL execution - aligns with project baseline and modern typing/runtime support.
- **LinkML 1.10.x**: declarative schema and validation layer - canonical for Biolink-adjacent contracts.
- **Biolink Model 4.3.x**: semantic contract - must be pinned to prevent predicate/category drift.
- **KGX 2.6.x**: exchange/validation/output layer - required path to Translator-compatible NDJSON artifacts.
- **pandas 3.0.x + PyArrow 23.0.x**: tabular IO and columnar interchange - broad connector support and fast handoffs.
- **DuckDB 1.5.x**: set-based merge/dedup/prune/QC engine - best single-node option without distributed overhead.

### Expected Features

Feature research is clear on launch scope: deliver a strict, reproducible compiler pipeline first; defer UX/platform expansion until semantic quality is stable.

**Must have (table stakes):**
- Declarative YAML ingest with strict schema/field validation.
- Biolink-aware node/edge modeling and pre-export semantic validation.
- KGX-compliant NDJSON/TSV output for Translator interoperability.
- Identifier normalization with confidence-governed resolution behavior.
- Provenance/evidence retention on edges, including source lineage.
- CLI workflow (`validate`, `build`, `report`) with deterministic artifacts.

**Should have (competitive):**
- Multi-stage entity resolution QC with policy gates and audit trail.
- Incremental diff-aware rebuilds for frequent source updates.
- Built-in integrity operations (dedupe/prune/harmonize).
- Quality scorecards and release gates for Translator readiness.

**Defer (v2+):**
- Full GUI/dashboard.
- Broad plugin SDK before core pipeline hardening.
- Real-time streaming ingestion as default operating mode.

### Architecture Approach

Architecture findings strongly favor a compiled declarative pipeline with hard stage boundaries and DuckDB-backed staging. The winning pattern is: compile config once -> transform rows deterministically -> resolve IDs -> validate semantically -> assemble/dedup/prune -> export atomically with run metadata and quarantine artifacts.

**Major components:**
1. **Control plane (CLI, orchestrator, config compiler, run manifest)** - enforces stage order and deterministic execution plans.
2. **Data plane (adapters, transform engine, resolver, validation gate)** - converts heterogeneous source rows into canonicalized, compliant candidates.
3. **Graph plane (assembly/QC/export/metrics)** - performs set-based cleanup and publishes versioned KGX artifacts plus reports.

### Critical Pitfalls

1. **KGX/Biolink contract confusion** - always validate with pinned Biolink via KGX in CI; fail on schema errors.
2. **Identifier hygiene collapse** - enforce per-column namespace/canonicalization policies with reject reports.
3. **Provenance misuse on edges** - require Translator-style source-role fields and valid `infores:*` chains.
4. **Over-aggressive entity resolution** - use precision-first thresholds, confidence bands, and review queues.
5. **Non-reproducible builds from mutable inputs** - snapshot/hash inputs and gate releases on manifest determinism.

## Implications for Roadmap

Based on the combined research, a 5-phase roadmap is the best fit.

### Phase 1: Contracts and Semantic Baseline
**Rationale:** Everything else depends on stable config semantics and Biolink/KGX correctness.
**Delivers:** Version-pinned semantic contracts, config compiler validation, KGX/Biolink CI gates.
**Addresses:** Declarative ingest foundation, Biolink-aware modeling requirements.
**Avoids:** KGX/Biolink mismatch, early semantic drift.

### Phase 2: Ingestion and Canonicalization Guardrails
**Rationale:** Source normalization and stable IDs must be in place before meaningful graph assembly.
**Delivers:** Source adapters, strict field contracts, identifier policy linting, row-to-edge lineage IDs.
**Addresses:** Declarative ingest, normalization table stakes, traceability expectations.
**Avoids:** Identifier hygiene collapse, lost row-to-edge traceability.

### Phase 3: Resolution and Quality Control Calibration
**Rationale:** Resolution quality is a core differentiator and highest scientific-risk surface.
**Delivers:** Multi-stage resolver, confidence bands, quarantine/review workflows, QC thresholds.
**Addresses:** Multi-stage resolution differentiator, release quality gates.
**Avoids:** False merges from permissive fuzzy/embedding matching.

### Phase 4: Provenance and Translator Compliance Hardening
**Rationale:** Translator readiness depends on edge-level lineage and semantic completeness, not just valid file shape.
**Delivers:** InfoRes-compliant source chains, qualifier/context mapping rules, predicate specificity checks.
**Addresses:** Provenance/evidence retention and Translator-ready packaging profile.
**Avoids:** Provenance misuse, qualifier/context loss, late-stage compliance failures.

### Phase 5: Reproducibility, Incremental Rebuilds, and Ops Hardening
**Rationale:** Performance and release confidence should be optimized only after correctness is stable.
**Delivers:** Snapshot/hash manifests, diff-aware rebuilds, partitioned validation, memory/perf budgets.
**Addresses:** Incremental rebuild differentiator, integrity operations at scale.
**Avoids:** Non-reproducible releases, memory-bound processing, unnecessary full revalidation.

### Phase Ordering Rationale

- The ordering follows architecture dependencies: contract -> ingest -> resolve -> validate/compliance -> scale/hardening.
- Groupings align with lifecycle boundaries in the recommended project structure, minimizing cross-module churn.
- This sequence directly neutralizes highest-impact pitfalls before introducing scale or UX complexity.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 3:** Resolver backend strategy (dbSERT/Node Normalizer/SSSOM interplay), threshold tuning by entity class, and review-queue operating model.
- **Phase 4:** Translator provenance edge cases (knowledge source role combinations) and qualifier requirements by association family.
- **Phase 5:** Incremental diff algorithms, partition strategy, and reproducibility policy for mutable upstream providers.

Phases with standard patterns (can likely skip `/gsd-research-phase`):
- **Phase 1:** LinkML/Biolink/KGX contract and CI validation patterns are well-documented.
- **Phase 2:** Adapter + compiler + canonicalization guardrail architecture is established and low-ambiguity.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Strong official docs plus current package metadata; version recommendations are concrete and aligned to Python 3.13. |
| Features | MEDIUM | Strong domain grounding, but differentiator prioritization depends on real user behavior and dataset mix. |
| Architecture | HIGH | Consistent patterns across Koza/KGX/Translator ecosystems and clear dependency ordering. |
| Pitfalls | MEDIUM-HIGH | Risks are well-supported by ecosystem guidance; incident frequency/severity still context-dependent. |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- **Resolution policy calibration:** Need project-specific precision/recall targets by entity class before production gating.
- **Provenance completeness policy:** Need explicit acceptance rules for missing/partial source chains per datasource type.
- **Incremental rebuild boundaries:** Need decisions on partition keys, invalidation rules, and acceptable staleness windows.
- **Security posture details:** Need explicit host allowlist/download constraints and redaction policy finalization in implementation plan.

## Sources

### Primary (HIGH confidence)
- Context7 `/websites/linkml_io_linkml` - LinkML validation and schema-driven ETL contracts.
- Context7 `/biolink/biolink-model` and official Biolink docs - semantic categories/predicates/provenance semantics.
- KGX official docs/spec/validator docs - format expectations and Biolink validation behavior.
- Koza official docs - declarative ingest and graph operation architecture patterns.
- Translator technical docs - ecosystem expectations for interoperability and semantics.
- PyPI metadata for pinned packages (`kgx`, `linkml`, `biolink-model`, `bmt`, `sssom`, `pandas`, `pyarrow`, `duckdb`, `pydantic`, `typer`).

### Secondary (MEDIUM confidence)
- Context7 `/unionai-oss/pandera` and `/frictionlessdata/frictionless-py` - data quality and tabular validation extensions.
- Node Normalizer and Reasoner Validator repositories/docs - practical integration and compliance checks.
- RTX-KG2 Translator documentation - implementation-oriented reference patterns.

### Tertiary (LOW confidence)
- None identified in current research set.

---
*Research completed: 2026-03-12*
*Ready for roadmap: yes*
