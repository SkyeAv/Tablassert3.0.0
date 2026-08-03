
---
# Aggregate quality metrics

- PMCs assayed: **10**
- MAPPED: **10**  ·  SKIPPED: **0**  ·  MAPPED rate: **100%**
- mean best coverage: **0.974**
- predicate distribution: `associated_with`×2, `gene_associated_with_condition`×2, `actively_involved_in`×2, `increases_amount_or_activity_of`×1, `in_taxon`×1, `participates_in`×1, `expressed_in`×1
- ⚠️ generic-fallback predicate used for: PMC13161869, PMC12900646
# Tablassert agent QC assay report

State dir: `.tablassert/qc-assay`  ·  PMCs assayed: 10


---
## PMC8017771 — **MAPPED**

- **best coverage:** 0.998
- **KG:** 520 nodes / 520 edges
- **predicate:** `increases_amount_or_activity_of`
- **subject:** method=value encoding=CHEBI:9168 prioritize=None taxon=None
- **object:** method=column encoding=D prioritize=['Gene', 'Protein']
- **source:** kind=excel sheet='Supp.Table 2A_cluster-1' local=<state-dir>/downloads/PMC8017771/PMC8017771.1/NIHMS1644812-supplement-1644812_Supp_Tab2.xlsx
- **provenance:** {'repo': 'PMC', 'publication': 'PMC8017771', 'knowledge_level': 'statistical_association', 'agent_type': 'data_analysis_pipeline'}
- **config sha256:** `fb96e79edf16`

### Derived config (`configs/PMC8017771.yaml`)

```yaml
source:
  kind: excel
  local: <state-dir>/downloads/PMC8017771/PMC8017771.1/NIHMS1644812-supplement-1644812_Supp_Tab2.xlsx
  url: https://pmc-oa-opendata.s3.amazonaws.com/PMC8017771.1/NIHMS1644812-supplement-1644812_Supp_Tab2.xlsx
  sheet: "Supp.Table 2A_cluster-1"
  row_slice: [2, "auto"]
  reindex:
    - column: D
      comparator: ""
      comparison: ne
statement:
  subject:
    method: value
    encoding: "CHEBI:9168"
  predicate: increases_amount_or_activity_of
  object:
    method: column
    encoding: D
    taxon: 10090
    prioritize: [Gene, Protein]
    explode_by: ";"
    regex:
      - pattern: "\\s+"
        replacement: " "
provenance:
  repo: PMC
  publication: "PMC8017771"
  knowledge_level: statistical_association
  agent_type: data_analysis_pipeline
annotations:
  - annotation: log2_kr_ko
    method: column
    encoding: H
  - annotation: q_value_kr_ko
    method: column
    encoding: I
  - annotation: log2_kr_wt
    method: column
    encoding: J
  - annotation: q_value_kr_wt
    method: column
    encoding: K
```

### Sample edges (first 5)

```json
{"subject": "CHEBI:9168", "predicate": "biolink:increases_amount_or_activity_of", "object": "NCBIGene:14645", "primary_knowledge_source": ["infores:agent"]}
{"subject": "CHEBI:9168", "predicate": "biolink:increases_amount_or_activity_of", "object": "NCBIGene:434437", "primary_knowledge_source": ["infores:agent"]}
{"subject": "CHEBI:9168", "predicate": "biolink:increases_amount_or_activity_of", "object": "NCBIGene:23934", "primary_knowledge_source": ["infores:agent"]}
{"subject": "CHEBI:9168", "predicate": "biolink:increases_amount_or_activity_of", "object": "NCBIGene:224903", "primary_knowledge_source": ["infores:agent"]}
{"subject": "CHEBI:9168", "predicate": "biolink:increases_amount_or_activity_of", "object": "NCBIGene:12367", "primary_knowledge_source": ["infores:agent"]}
```

---
## PMC13161869 — **MAPPED**

- **best coverage:** 1.000
- **KG:** 114 nodes / 507 edges
- **predicate:** `associated_with`  ⚠️ *generic fallback*
- **subject:** method=column encoding=A prioritize=['Protein', 'Gene'] taxon=9606
- **object:** method=value encoding=MONDO:0007739 prioritize=None
- **source:** kind=excel sheet='Cap Score - Ion Level' local=<state-dir>/downloads/PMC13161869/PMC13161869.1/ACN3-13-911-s001.xlsx
- **provenance:** {'repo': 'PMC', 'publication': 'PMC13161869'}
- **config sha256:** `1aa5a8bf5ded`

### Derived config (`configs/PMC13161869.yaml`)

```yaml
template:
  provenance:
    repo: PMC
    publication: "PMC13161869"
sections:
  - source:
      kind: excel
      local: "<state-dir>/downloads/PMC13161869/PMC13161869.1/ACN3-13-911-s001.xlsx"
      url: "https://pmc-oa-opendata.s3.amazonaws.com/PMC13161869.1/ACN3-13-911-s001.xlsx"
      sheet: "Cap Score - Ion Level"
      row_slice: [1, "auto"]
      reindex:
        - column: A
          comparator: "Protein"
          comparison: ne
        - column: A
          comparator: ""
          comparison: ne
    statement:
      subject:
        method: column
        encoding: A
        prioritize: ['Protein', 'Gene']
        taxon: 9606
        regex:
          - pattern: "_HUMAN$"
            replacement: ""
          - pattern: "^AFAM$"
            replacement: "AFM"
          - pattern: "^CO4A$"
            replacement: "C4A"
          - pattern: "^HAVR2$"
            replacement: "HAVCR2"
          - pattern: "^KAIN$"
            replacement: "SERPINA4"
          - pattern: "^KPYM$"
            replacement: "PKM"
          - pattern: "^NRX1A$"
            replacement: "NRXN1"
          - pattern: "^NRX2A$"
            replacement: "NRXN2"
          - pattern: "^OSTP$"
            replacement: "SPP1"
          - pattern: "\\s+"
            replacement: " "
      predicate: associated_with
      object:
        method: value
        encoding: "MONDO:0007739"
    annotations:
      - annotation: peptide
        method: column
        encoding: B
      - annotation: ion
        method: column
        encoding: C
      - annotation: estimate_analysis1
        method: column
        encoding: D
      - annotation: p_value_analysis1
        method: column
        encoding: G
      - annotation: adjusted_p_value_analysis1
        method: column
        encoding: H
      - annotation: estimate_analysis2
        method: column
        encoding: I
      - annotation: p_value_analysis2
        method: column
        encoding: L
      - annotation: adjusted_p_value_analysis2
        method: column
        encoding: M
```

### Sample edges (first 5)

```json
{"subject": "NCBIGene:23467", "predicate": "biolink:associated_with", "object": "MONDO:0007739", "primary_knowledge_source": ["infores:agent"]}
{"subject": "NCBIGene:5345", "predicate": "biolink:associated_with", "object": "MONDO:0007739", "primary_knowledge_source": ["infores:agent"]}
{"subject": "UniProtKB:Q9H9K5", "predicate": "biolink:associated_with", "object": "MONDO:0007739", "primary_knowledge_source": ["infores:agent"]}
{"subject": "NCBIGene:7076", "predicate": "biolink:associated_with", "object": "MONDO:0007739", "primary_knowledge_source": ["infores:agent"]}
{"subject": "MGI:98863", "predicate": "biolink:associated_with", "object": "MONDO:0007739", "primary_knowledge_source": ["infores:agent"]}
```

---
## PMC12900646 — **MAPPED**

- **best coverage:** 1.000
- **KG:** 31 nodes / 288 edges
- **predicate:** `associated_with`  ⚠️ *generic fallback*
- **subject:** method=column encoding=A prioritize=['ClinicalMeasurement', 'PhenotypicFeature', 'ClinicalAttribute'] taxon=9606
- **object:** method=column encoding=B prioritize=['Cell']
- **source:** kind=excel sheet='Supp. Table 7' local=<state-dir>/downloads/PMC12900646/PMC12900646.1/41588_2025_2486_MOESM4_ESM.xlsx
- **provenance:** {'repo': 'PMC', 'publication': 'PMC12900646'}
- **config sha256:** `560d11724293`

### Derived config (`configs/PMC12900646.yaml`)

```yaml
template:
  provenance:
    repo: PMC
    publication: "PMC12900646"
sections:
  - source:
      kind: excel
      local: <state-dir>/downloads/PMC12900646/PMC12900646.1/41588_2025_2486_MOESM4_ESM.xlsx
      url: "https://pmc-oa-opendata.s3.amazonaws.com/PMC12900646.1/41588_2025_2486_MOESM4_ESM.xlsx"
      sheet: "Supp. Table 7"
      row_slice: [7, "auto"]
      reindex:
        - column: A
          comparator: ""
          comparison: ne
    statement:
      subject:
        method: column
        encoding: A
        regex:
          - {pattern: '^\s+', replacement: ''}
          - {pattern: '\s+$', replacement: ''}
          - {pattern: '_perc', replacement: ' percentage'}
          - {pattern: '_count', replacement: ' count'}
          - {pattern: '\s+', replacement: ' '}
          - {pattern: 'HLR percentage', replacement: 'reticulocyte percentage'}
          - {pattern: 'Neutrophill percentage', replacement: 'neutrophil percentage'}
          - {pattern: 'Eosinophill percentage', replacement: 'eosinophil percentage'}
          - {pattern: 'Basophill percentage', replacement: 'basophil percentage'}
          - {pattern: 'WBC count', replacement: 'white blood cell count'}
          - {pattern: 'RBC count', replacement: 'red blood cell count'}
          - {pattern: 'MCV', replacement: 'mean corpuscular volume'}
          - {pattern: 'RDW', replacement: 'red cell distribution width'}
          - {pattern: 'PDW', replacement: 'platelet distribution width'}
          - {pattern: 'MSCV', replacement: 'mean sphered cell volume'}
          - {pattern: 'Haemoglobin', replacement: 'hemoglobin'}
        prioritize: ['ClinicalMeasurement', 'PhenotypicFeature', 'ClinicalAttribute']
        taxon: 9606
      predicate: associated_with
      object:
        method: column
        encoding: B
        regex:
          - {pattern: '^\s+', replacement: ''}
          - {pattern: '\s+$', replacement: ''}
          - {pattern: '^B$', replacement: 'B cell'}
          - {pattern: '^CD4$', replacement: 'CD4-positive, alpha-beta T cell'}
          - {pattern: '^CD8$', replacement: 'CD8-positive, alpha-beta T cell'}
          - {pattern: '^CLP$', replacement: 'common lymphoid progenitor'}
          - {pattern: '^CMP$', replacement: 'common myeloid progenitor'}
          - {pattern: '^Ery$', replacement: 'erythroblast'}
          - {pattern: '^GMP-A$', replacement: 'granulocyte-monocyte progenitor cell'}
          - {pattern: '^GMP-B$', replacement: 'granulocyte-monocyte progenitor cell'}
          - {pattern: '^GMP-C$', replacement: 'granulocyte-monocyte progenitor cell'}
          - {pattern: '^HSC$', replacement: 'hematopoietic stem cell'}
          - {pattern: '^LMPP$', replacement: 'lymphoid-primed multipotent progenitor'}
          - {pattern: '^mDC$', replacement: 'myeloid dendritic cell'}
          - {pattern: '^Mega$', replacement: 'megakaryocyte'}
          - {pattern: '^Mono$', replacement: 'monocyte'}
          - {pattern: '^Neu$', replacement: 'neutrophil
```

### Sample edges (first 5)

```json
{"subject": "UMLS:C2360306", "predicate": "biolink:associated_with", "object": "UMLS:C1706982", "primary_knowledge_source": ["infores:agent"]}
{"subject": "UMLS:C1171404", "predicate": "biolink:associated_with", "object": "CL:0000556", "primary_knowledge_source": ["infores:agent"]}
{"subject": "UMLS:C1167975", "predicate": "biolink:associated_with", "object": "UMLS:C1706982", "primary_knowledge_source": ["infores:agent"]}
{"subject": "UMLS:C0427565", "predicate": "biolink:associated_with", "object": "CL:0000837", "primary_knowledge_source": ["infores:agent"]}
{"subject": "UMLS:C2360306", "predicate": "biolink:associated_with", "object": "MONDO:0005810", "primary_knowledge_source": ["infores:agent"]}
```

---
## PMC9187732 — **MAPPED**

- **best coverage:** 1.000
- **KG:** 21 nodes / 20 edges
- **predicate:** `gene_associated_with_condition`
- **subject:** method=column encoding=A prioritize=['Gene'] taxon=9606
- **object:** method=value encoding=MONDO:0004988 prioritize=None
- **source:** kind=excel sheet='Percentiles - 16p11.2' local=<state-dir>/downloads/PMC9187732/PMC9187732.1/41467_2022_30968_MOESM16_ESM.xlsx
- **provenance:** {'repo': 'PMC', 'publication': 'PMC9187732'}
- **config sha256:** `e8cb8eacc72d`

### Derived config (`configs/PMC9187732.yaml`)

```yaml
source:
  kind: excel
  local: <state-dir>/downloads/PMC9187732/PMC9187732.1/41467_2022_30968_MOESM16_ESM.xlsx
  url: https://pmc-oa-opendata.s3.amazonaws.com/PMC9187732.1/41467_2022_30968_MOESM16_ESM.xlsx
  sheet: "Percentiles - 16p11.2"
  reindex:
    - column: A
      comparator: "Gene"
      comparison: ne
statement:
  subject:
    method: column
    encoding: A
    prioritize: ['Gene']
    taxon: 9606
  predicate: gene_associated_with_condition
  object:
    method: value
    encoding: "MONDO:0004988"
annotations:
  - annotation: mean_expression
    method: column
    encoding: B
  - annotation: percentile
    method: column
    encoding: C
provenance:
  repo: PMC
  publication: "PMC9187732"
```

### Sample edges (first 5)

```json
{"subject": "NCBIGene:26470", "predicate": "biolink:gene_associated_with_condition", "object": "MONDO:0004988", "primary_knowledge_source": ["infores:agent"]}
{"subject": "NCBIGene:83723", "predicate": "biolink:gene_associated_with_condition", "object": "MONDO:0004988", "primary_knowledge_source": ["infores:agent"]}
{"subject": "NCBIGene:654483", "predicate": "biolink:gene_associated_with_condition", "object": "MONDO:0004988", "primary_knowledge_source": ["infores:agent"]}
{"subject": "NCBIGene:79008", "predicate": "biolink:gene_associated_with_condition", "object": "MONDO:0004988", "primary_knowledge_source": ["infores:agent"]}
{"subject": "NCBIGene:5531", "predicate": "biolink:gene_associated_with_condition", "object": "MONDO:0004988", "primary_knowledge_source": ["infores:agent"]}
```

---
## PMC13099431 — **MAPPED**

- **best coverage:** 0.974
- **KG:** 75 nodes / 173 edges
- **predicate:** `actively_involved_in`
- **subject:** method=column encoding=G prioritize=['Gene'] taxon=9606
- **object:** method=column encoding=A prioritize=['BiologicalProcess']
- **source:** kind=excel sheet='Supplementary Table 7' local=<state-dir>/downloads/PMC13099431/PMC13099431.1/41591_2026_4228_MOESM2_ESM.xlsx
- **provenance:** {'repo': 'PMC', 'publication': 'PMC13099431'}
- **config sha256:** `076cdc335924`

### Derived config (`configs/PMC13099431.yaml`)

```yaml
template:
  provenance:
    repo: PMC
    publication: "PMC13099431"
sections:
  - source:
      kind: excel
      local: <state-dir>/downloads/PMC13099431/PMC13099431.1/41591_2026_4228_MOESM2_ESM.xlsx
      url: "https://pmc-oa-opendata.s3.amazonaws.com/PMC13099431.1/41591_2026_4228_MOESM2_ESM.xlsx"
      sheet: "Supplementary Table 7"
      row_slice: [2, "auto"]
    statement:
      subject:
        method: column
        encoding: G
        explode_by: ";"
        prioritize: ['Gene']
        taxon: 9606
      predicate: actively_involved_in
      object:
        method: column
        encoding: A
        remove:
          - "^.*\\("
          - "\\).*$"
        prioritize: ['BiologicalProcess']
    annotations:
      - {annotation: p_value, method: column, encoding: C}
      - {annotation: adjusted_p_value, method: column, encoding: D}
      - {annotation: odds_ratio, method: column, encoding: E}
      - {annotation: combined_score, method: column, encoding: F}
```

### Sample edges (first 5)

```json
{"subject": "NCBIGene:1742", "predicate": "biolink:actively_involved_in", "object": "GO:0048813", "primary_knowledge_source": ["infores:agent"]}
{"subject": "NCBIGene:2904", "predicate": "biolink:actively_involved_in", "object": "GO:0098815", "primary_knowledge_source": ["infores:agent"]}
{"subject": "NCBIGene:1742", "predicate": "biolink:actively_involved_in", "object": "GO:0007612", "primary_knowledge_source": ["infores:agent"]}
{"subject": "NCBIGene:7532", "predicate": "biolink:actively_involved_in", "object": "GO:0006469", "primary_knowledge_source": ["infores:agent"]}
{"subject": "NCBIGene:2011", "predicate": "biolink:actively_involved_in", "object": "GO:0035088", "primary_knowledge_source": ["infores:agent"]}
```

---
## PMC13172311 — **MAPPED**

- **best coverage:** 0.994
- **KG:** 319 nodes / 318 edges
- **predicate:** `actively_involved_in`
- **subject:** method=column encoding=A prioritize=['Gene'] taxon=9606
- **object:** method=value encoding=GO:0008380 prioritize=None
- **source:** kind=excel sheet='vU1-8 KO v WT' local=<state-dir>/downloads/PMC13172311/PMC13172311.1/41467_2026_73121_MOESM5_ESM.xlsx
- **provenance:** {'repo': 'PMC', 'publication': 'PMC13172311'}
- **config sha256:** `af8d05069765`

### Derived config (`configs/PMC13172311.yaml`)

```yaml
source:
  kind: excel
  local: <state-dir>/downloads/PMC13172311/PMC13172311.1/41467_2026_73121_MOESM5_ESM.xlsx
  url: https://pmc-oa-opendata.s3.amazonaws.com/PMC13172311.1/41467_2026_73121_MOESM5_ESM.xlsx
  sheet: vU1-8 KO v WT
statement:
  subject:
    method: column
    encoding: A
    prioritize:
    - Gene
    taxon: 9606
    regex:
    - pattern: \.pdf$
      replacement: ''
    - pattern: \.[0-9]+$
      replacement: ''
    - pattern: ^ensg
      replacement: ENSG
  predicate: actively_involved_in
  object:
    method: value
    encoding: GO:0008380
provenance:
  repo: PMC
  publication: PMC13172311
```

### Sample edges (first 5)

```json
{"subject": "NCBIGene:25979", "predicate": "biolink:actively_involved_in", "object": "GO:0008380", "primary_knowledge_source": ["infores:agent"]}
{"subject": "NCBIGene:54531", "predicate": "biolink:actively_involved_in", "object": "GO:0008380", "primary_knowledge_source": ["infores:agent"]}
{"subject": "NCBIGene:91433", "predicate": "biolink:actively_involved_in", "object": "GO:0008380", "primary_knowledge_source": ["infores:agent"]}
{"subject": "NCBIGene:255057", "predicate": "biolink:actively_involved_in", "object": "GO:0008380", "primary_knowledge_source": ["infores:agent"]}
{"subject": "NCBIGene:25825", "predicate": "biolink:actively_involved_in", "object": "GO:0008380", "primary_knowledge_source": ["infores:agent"]}
```

---
## PMC12906585 — **MAPPED**

- **best coverage:** 1.000
- **KG:** 14 nodes / 1697 edges
- **predicate:** `in_taxon`
- **subject:** method=column encoding=B prioritize=['Genome'] taxon=4530
- **object:** method=value encoding=NCBITaxon:4530 prioritize=None
- **source:** kind=excel sheet='Map' local=<state-dir>/downloads/PMC12906585/PMC12906585.1/122_2026_5178_MOESM1_ESM.xlsx
- **provenance:** {'repo': 'PMC', 'publication': 'PMC12906585'}
- **config sha256:** `d80ab3fe0ca0`

### Derived config (`configs/PMC12906585.yaml`)

```yaml
template:
  provenance:
    repo: PMC
    publication: "PMC12906585"
sections:
  - source:
      kind: excel
      local: <state-dir>/downloads/PMC12906585/PMC12906585.1/122_2026_5178_MOESM1_ESM.xlsx
      url: "https://pmc-oa-opendata.s3.amazonaws.com/PMC12906585.1/122_2026_5178_MOESM1_ESM.xlsx"
      sheet: Map
    statement:
      subject:
        method: column
        encoding: B
        regex:
          - pattern: "^chr0*"
            replacement: "chromosome "
        prioritize: ['Genome']
        taxon: 4530
      predicate: in_taxon
      object:
        method: value
        encoding: "NCBITaxon:4530"
    annotations:
      - annotation: bin_id
        method: column
        encoding: A
      - annotation: start_position
        method: column
        encoding: C
      - annotation: end_position
        method: column
        encoding: D
      - annotation: length
        method: column
        encoding: E
```

### Sample edges (first 5)

```json
{"subject": "MESH:D002889", "predicate": "biolink:in_taxon", "object": "NCBITaxon:4530", "primary_knowledge_source": ["infores:agent"]}
{"subject": "MESH:D002899", "predicate": "biolink:in_taxon", "object": "NCBITaxon:4530", "primary_knowledge_source": ["infores:agent"]}
{"subject": "MESH:D002889", "predicate": "biolink:in_taxon", "object": "NCBITaxon:4530", "primary_knowledge_source": ["infores:agent"]}
{"subject": "MESH:D002893", "predicate": "biolink:in_taxon", "object": "NCBITaxon:4530", "primary_knowledge_source": ["infores:agent"]}
{"subject": "MESH:D002896", "predicate": "biolink:in_taxon", "object": "NCBITaxon:4530", "primary_knowledge_source": ["infores:agent"]}
```

---
## PMC13172025 — **MAPPED**

- **best coverage:** 0.959
- **KG:** 1360 nodes / 2770 edges
- **predicate:** `participates_in`
- **subject:** method=column encoding=L prioritize=['Gene'] taxon=9606
- **object:** method=column encoding=F prioritize=['Pathway', 'BiologicalProcess']
- **source:** kind=excel sheet='SD15' local=<state-dir>/downloads/PMC13172025/PMC13172025.1/42003_2026_10045_MOESM3_ESM.xlsx
- **provenance:** {'repo': 'PMC', 'publication': 'PMC13172025'}
- **config sha256:** `86448dd76087`

### Derived config (`configs/PMC13172025.yaml`)

```yaml
template:
  provenance:
    repo: PMC
    publication: "PMC13172025"
sections:
  - source:
      kind: excel
      local: <state-dir>/downloads/PMC13172025/PMC13172025.1/42003_2026_10045_MOESM3_ESM.xlsx
      url: "https://pmc-oa-opendata.s3.amazonaws.com/PMC13172025.1/42003_2026_10045_MOESM3_ESM.xlsx"
      sheet: "SD15"
      row_slice: [2, "auto"]
    statement:
      subject:
        method: column
        encoding: L
        prefix: "NCBIGene:"
        explode_by: "/"
        prioritize: ['Gene']
        taxon: 9606
      predicate: participates_in
      object:
        method: column
        encoding: F
        prioritize: ['Pathway', 'BiologicalProcess']
    annotations:
      - annotation: p_value
        method: column
        encoding: I
      - annotation: p_adjust
        method: column
        encoding: J
```

### Sample edges (first 5)

```json
{"subject": "NCBIGene:5579", "predicate": "biolink:participates_in", "object": "UMLS:C1513094", "primary_knowledge_source": ["infores:agent"]}
{"subject": "NCBIGene:317", "predicate": "biolink:participates_in", "object": "UMLS:C2062441", "primary_knowledge_source": ["infores:agent"]}
{"subject": "NCBIGene:3592", "predicate": "biolink:participates_in", "object": "MONDO:0004619", "primary_knowledge_source": ["infores:agent"]}
{"subject": "NCBIGene:993", "predicate": "biolink:participates_in", "object": "GO:0090398", "primary_knowledge_source": ["infores:agent"]}
{"subject": "NCBIGene:5606", "predicate": "biolink:participates_in", "object": "MONDO:0043693", "primary_knowledge_source": ["infores:agent"]}
```

---
## PMC7206184 — **MAPPED**

- **best coverage:** 0.813
- **KG:** 14703 nodes / 23584 edges
- **predicate:** `expressed_in`
- **subject:** method=column encoding=C prioritize=['Gene'] taxon=9606
- **object:** method=column encoding=A prioritize=['AnatomicalEntity', 'GrossAnatomicalStructure']
- **source:** kind=excel sheet='v68.lvedv.twas.alltissues' local=<state-dir>/downloads/PMC7206184/PMC7206184.1/41467_2020_15823_MOESM9_ESM.xlsx
- **provenance:** {'repo': 'PMC', 'publication': 'PMC7206184', 'knowledge_level': 'statistical_association', 'agent_type': 'data_analysis_pipeline'}
- **config sha256:** `d6785bc724be`

### Derived config (`configs/PMC7206184.yaml`)

```yaml
template:
  provenance:
    repo: PMC
    publication: "PMC7206184"
    knowledge_level: statistical_association
    agent_type: data_analysis_pipeline
sections:
  - source:
      kind: excel
      local: <state-dir>/downloads/PMC7206184/PMC7206184.1/41467_2020_15823_MOESM9_ESM.xlsx
      url: "https://pmc-oa-opendata.s3.amazonaws.com/PMC7206184.1/41467_2020_15823_MOESM9_ESM.xlsx"
      sheet: "v68.lvedv.twas.alltissues"
      row_slice: [1, "auto"]
    statement:
      subject:
        method: column
        encoding: C
        prioritize: ['Gene']
        taxon: 9606
      predicate: expressed_in
      object:
        method: column
        encoding: A
        prioritize: ['AnatomicalEntity', 'GrossAnatomicalStructure']
        regex:
          - pattern: "_"
            replacement: " "
    annotations:
      - annotation: twas_z
        method: column
        encoding: S
      - annotation: twas_p
        method: column
        encoding: T
```

### Sample edges (first 5)

```json
{"subject": "NCBIGene:101", "predicate": "biolink:expressed_in", "object": "UBERON:0006618", "primary_knowledge_source": ["infores:agent"]}
{"subject": "NCBIGene:404672", "predicate": "biolink:expressed_in", "object": "UBERON:0002084", "primary_knowledge_source": ["infores:agent"]}
{"subject": "NCBIGene:4927", "predicate": "biolink:expressed_in", "object": "UBERON:0002084", "primary_knowledge_source": ["infores:agent"]}
{"subject": "NCBIGene:100190938", "predicate": "biolink:expressed_in", "object": "UBERON:0002084", "primary_knowledge_source": ["infores:agent"]}
{"subject": "NCBIGene:25770", "predicate": "biolink:expressed_in", "object": "UBERON:0006618", "primary_knowledge_source": ["infores:agent"]}
```

---
## PMC11947420 — **MAPPED**

- **best coverage:** 0.999
- **KG:** 13028 nodes / 37051 edges
- **predicate:** `gene_associated_with_condition`
- **subject:** method=column encoding=H prioritize=['Gene'] taxon=9606
- **object:** method=value encoding=MONDO:0004992 prioritize=None
- **source:** kind=excel sheet='Table_S7' local=<state-dir>/downloads/PMC11947420/PMC11947420.1/mmc2.xlsx
- **provenance:** {'repo': 'PMC', 'publication': 'PMC11947420'}
- **config sha256:** `8a1c74d4efc6`

### Derived config (`configs/PMC11947420.yaml`)

```yaml
source:
  kind: excel
  local: <state-dir>/downloads/PMC11947420/PMC11947420.1/mmc2.xlsx
  url: https://pmc-oa-opendata.s3.amazonaws.com/PMC11947420.1/mmc2.xlsx
  sheet: Table_S7
  reindex:
    - column: A
      comparison: ne
      comparator: Cohort
statement:
  subject:
    method: column
    encoding: H
    prioritize:
      - Gene
    taxon: 9606
  predicate: gene_associated_with_condition
  object:
    method: value
    encoding: "MONDO:0004992"
provenance:
  repo: PMC
  publication: "PMC11947420"
```

### Sample edges (first 5)

```json
{"subject": "NCBIGene:392390", "predicate": "biolink:gene_associated_with_condition", "object": "MONDO:0004992", "primary_knowledge_source": ["infores:agent"]}
{"subject": "NCBIGene:730291", "predicate": "biolink:gene_associated_with_condition", "object": "MONDO:0004992", "primary_knowledge_source": ["infores:agent"]}
{"subject": "NCBIGene:645811", "predicate": "biolink:gene_associated_with_condition", "object": "MONDO:0004992", "primary_knowledge_source": ["infores:agent"]}
{"subject": "NCBIGene:85301", "predicate": "biolink:gene_associated_with_condition", "object": "MONDO:0004992", "primary_knowledge_source": ["infores:agent"]}
{"subject": "NCBIGene:23283", "predicate": "biolink:gene_associated_with_condition", "object": "MONDO:0004992", "primary_knowledge_source": ["infores:agent"]}
```