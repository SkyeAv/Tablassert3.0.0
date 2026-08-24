# Golden fixture: PMC11708054 (microbiome ~ tamoxifen correlations)

Offline replay pair for the US-011 eval harness. **Everything here is SYNTHETIC and
OFFLINE**: no live network fetch and no LLM call is ever made against these files. They
give the eval tests a deterministic, schema-valid config + a tiny table to score against.

## Attribution (CC-BY)

The *shape* of this fixture (an organism `correlated_with` chemical table with Spearman
rho / p-value annotations) mirrors the real ALAMV6 supplementary table from:

- **PMC11708054**: DOI [10.1128/mbio.01679-24](https://doi.org/10.1128/mbio.01679-24)
- Licensed CC-BY; cite the article DOI when reusing the real data.

The actual cell values and organism list in `source_table.csv` are **fabricated** for
testing and do NOT reproduce any real measurement from the article.

## Files

- **`ALAMV6.yaml`**: the reference *table config*: a faithful copy of the `template:`
  block documented in `docs/configuration/advanced-example.md` (excel source, full
  annotation set, lineage-glue regex). Schema-valid (`validate_section(...) is True`).
  The `local:` excel path is illustrative only; the file is not present and is never read.
- **`source_table.csv`**: a SMALL synthetic snapshot (7 rows, **headerless**) shaped like
  ALAMV6's "all correlations" sheet: column A = organism name (taxonomic string),
  column B = Spearman rho (float), column C = p_value (float). Deterministic + tiny.
- **`reference_config.yaml`**: the golden config the agent should approximate, trimmed to
  `source_table.csv`'s columns (text/CSV source; subject = column A `OrganismTaxon`;
  object = fixed literal `CHEBI:41774`; p_value = C, relationship_strength = B).
  Schema-valid.
- **`README.md`**: this file.

## Reference KGX is computed in-test (NOT committed)

No large reference KGX is committed. When a test needs the reference graph for node/edge
F1, it builds it on the fly with the e2e recipe (`tests/test_e2e_smoke.py`):

1. Build a tiny REAL redb (`tablassert.rs.build_fullmap_db`) registering the seven
   organism names as `OrganismTaxon` synonyms (→ `NCBITaxon:*`) plus `CHEBI:41774` as a
   `ChemicalEntity` (the fixed object literal must be present for edges to emit).
2. `build_and_audit(reference_config.yaml, fullmap=<tiny redb>)` writes
   `<name>_<version>.{nodes,edges}.ndjson` to an isolated workdir.
3. Load those NDJSON files with `tablassert.agent.load_kgx` and score F1 against a
   candidate build via `node_edge_f1`.

The pure F1 unit tests use small hand-made KGX dicts directly and do not build anything.
