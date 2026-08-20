# Edge-count acceptance fixtures (US-007)

Offline acceptance pair for the **edge-count harness**: an agent-produced config must
reach at least `REFERENCE_EDGE_FRACTION` (0.5) of the reference config's KGX edge count
when both build the same payload against the same tiny real redb.

**Everything here is SYNTHETIC and OFFLINE** — no network fetch, no LLM call.

## Attribution (shape only)

The *shape* of the payload (a disease x organ-system association workbook with a
pleiotropic locus sheet) mirrors the PMC10766526 disease x system supplementary style.
Every cell value, disease/system/locus name pairing, and statistic is **fabricated** for
testing; nothing reproduces a real measurement from any article.

## Files

- **`payload.xlsx`** — 3 worksheets, 50 data rows total (row 0 = title, row 1 = header,
  data from row 2, so configs skip both with `row_slice: [2, auto]`):
  - `disease_system` (28 rows): disease | organ_system | beta | p_value.
  - `locus_hits` (8 rows): locus | **;-joined** diseases | beta | p_value — the
    multi-valued sheet `explode_by: ";"` turns into ~20 edges.
  - `secondary_endpoints` (14 rows): same columns as `disease_system`, disjoint pairs.
- **`agent_config.yaml`** — the improved-agent config (US-006 shape): multi-section,
  correct `sheet` + `row_slice` per section, breadth via `prioritize` and `explode_by`,
  and the statistical annotation pair (`effect_size` column + `effect_type`
  `method: value`). Covers 2 of the 3 sheets.
- **`reference_config.yaml`** — the richer reference: all 3 sheets, one section each.
- **`agent_config_poor.yaml`** — the negative control: single section over `locus_hits`
  WITHOUT `explode_by`; the joined cells never resolve, so it lands far below the
  fraction gate.

Every (disease, system) / (locus, disease) pair is unique across all sheets — edges are
keyed by (subject, predicate, object), so no pair silently collapses.

## Resolution

`tests/test_agent_edgecount.py` builds a tiny REAL redb (`tablassert.rs.build_fullmap_db`)
registering the 12 diseases (MONDO), 8 organ systems (UBERON, AnatomicalEntity), and 8
loci (HGNC) — the e2e recipe from `tests/test_e2e_smoke.py`. The configs' relative
`local` path is rewritten to the absolute fixture payload in-test.
