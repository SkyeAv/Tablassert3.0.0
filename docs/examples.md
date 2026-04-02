# Use Case Gallery

Real-world patterns for transforming tabular data into knowledge graphs with Tablassert. Each example shows a complete configuration with explanations.

## Gene-Disease Associations

Transform a gene-disease association table into KGX-compliant edges with statistical annotations.

**Data:** CSV with gene symbols, disease names, and p-values

```yaml
template:
  syntax: TC3
  status: alpha
  source:
    kind: text
    local: ./gene-disease.csv
    url: https://example.com/gene-disease.csv
    row_slice: [1, auto]
    delimiter: ","
  statement:
    subject:
      method: column
      encoding: gene_symbol
      prioritize:
        - Gene
      taxon: 9606
    predicate: associated_with
    object:
      method: column
      encoding: disease_name
      prioritize:
        - Disease
  provenance:
    repo: PMID
    publication: PMID12345678
    contributors:
      - kind: curation
        name: Your Name
        date: 01 JAN 2026
  annotations:
    - annotation: p value
      method: column
      encoding: p_value
```

**Key techniques:**

- **Taxonomic filtering** (`taxon: 9606`) restricts gene resolution to human genes
- **Category prioritization** ensures genes resolve as `biolink:Gene` and diseases as `biolink:Disease`
- **Column annotations** attach per-row p-values to each edge

---

## Drug-Target Interactions

Extract drug-target relationships from a curated interaction database.

**Data:** TSV with drug names, target genes, and interaction types

```yaml
template:
  syntax: TC3
  status: alpha
  source:
    kind: text
    local: ./drug-targets.tsv
    url: https://example.com/drug-targets.tsv
    row_slice: [1, auto]
    delimiter: "\t"
  statement:
    subject:
      method: column
      encoding: drug_name
      prioritize:
        - ChemicalEntity
        - SmallMolecule
    predicate: interacts_with
    object:
      method: column
      encoding: target_gene
      prioritize:
        - Gene
        - Protein
      taxon: 9606
  provenance:
    repo: PMID
    publication: PMID98765432
    contributors:
      - kind: curation
        name: Your Name
        date: 15 FEB 2026
  annotations:
    - annotation: interaction type
      method: column
      encoding: interaction_type
    - annotation: assay
      method: value
      encoding: "binding assay"
```

**Key techniques:**

- **Multiple prioritized categories** (`ChemicalEntity`, `SmallMolecule`) give entity resolution fallback options
- **Fixed-value annotation** (`method: value`) attaches the same assay description to all edges
- **TSV delimiter** (`delimiter: "\t"`) handles tab-separated files

---

## Microbiome-Metabolite Correlations

Extract microbe-metabolite correlations with taxonomic name cleaning.

**Data:** Excel with raw taxonomic names, correlation coefficients, and p-values

```yaml
template:
  syntax: TC3
  status: alpha
  source:
    kind: excel
    local: ./microbiome-correlations.xlsx
    url: https://example.com/microbiome-data.xlsx
    sheet: correlations
    row_slice: [2, auto]
  statement:
    subject:
      method: column
      encoding: A
      prioritize:
        - OrganismTaxon
      avoid:
        - Gene
      remove:
        - "^NA "
      regex:
        - pattern: ".*g__"
          replacement: ""
        - pattern: ";s__"
          replacement: " "
        - pattern: "sp"
          replacement: "sp. "
    predicate: correlated_with
    object:
      method: value
      encoding: CHEBI:41774
  provenance:
    repo: PMC
    publication: PMC11708054
    contributors:
      - kind: curation
        name: Your Name
        date: 01 MAR 2026
  annotations:
    - annotation: p value
      method: column
      encoding: C
    - annotation: relationship strength
      method: column
      encoding: B
    - annotation: assertion method
      method: value
      encoding: "Spearman correlation"
```

**Key techniques:**

- **Regex pipeline** cleans raw taxonomic strings (e.g., `d__Bacteria;p__Firmicutes;g__Lactobacillus` → `Lactobacillus`)
- **Avoid list** (`avoid: [Gene]`) prevents organism names from resolving to gene entities
- **Fixed-value object** (`method: value`) assigns the same metabolite CURIE to all rows
- **Excel source** with sheet name and row slicing

---

## Multi-Pathway Gene Mapping

Map genes to multiple pathways from a single source using sections.

**Data:** CSV with gene symbols and multiple pathway columns

```yaml
template:
  syntax: TC3
  source:
    kind: text
    local: ./gene-pathways.csv
    url: https://example.com/gene-pathways.csv
    row_slice: [1, auto]
    delimiter: ","
  statement:
    subject:
      method: column
      encoding: gene_symbol
      prioritize:
        - Gene
      taxon: 9606
    object:
      method: value
      encoding: PLACEHOLDER
  provenance:
    repo: PMID
    publication: PMID11223344
    contributors:
      - kind: curation
        name: Your Name
        date: 01 APR 2026

sections:
  - statement:
      predicate: participates_in
      object:
        method: column
        encoding: kegg_pathway
        prioritize:
          - Pathway
    annotations:
      - annotation: pathway database
        method: value
        encoding: "KEGG"

  - statement:
      predicate: participates_in
      object:
        method: column
        encoding: reactome_pathway
        prioritize:
          - Pathway
    annotations:
      - annotation: pathway database
        method: value
        encoding: "Reactome"
```

**Key techniques:**

- **Template + sections** avoids repeating source and provenance for each pathway column
- **Section overrides** — each section provides its own predicate and object while inheriting the shared subject and source
- **Per-section annotations** tag edges with the pathway database source

---

## Conditional Filtering with Reindex

Filter rows based on column values before entity resolution.

**Data:** CSV with gene-disease associations and significance thresholds

```yaml
template:
  syntax: TC3
  status: alpha
  source:
    kind: text
    local: ./significant-associations.csv
    url: https://example.com/associations.csv
    row_slice: [1, auto]
    delimiter: ","
    reindex:
      - column: p_value
        comparison: lt
        comparator: 0.05
      - column: sample_size
        comparison: ge
        comparator: 100
  statement:
    subject:
      method: column
      encoding: gene_symbol
      prioritize:
        - Gene
      taxon: 9606
    predicate: associated_with
    object:
      method: column
      encoding: disease_name
      prioritize:
        - Disease
  provenance:
    repo: PMID
    publication: PMID55667788
    contributors:
      - kind: curation
        name: Your Name
        date: 01 MAY 2026
  annotations:
    - annotation: p value
      method: column
      encoding: p_value
    - annotation: sample size
      method: column
      encoding: sample_size
```

**Key techniques:**

- **Reindex filtering** keeps only rows where `p_value < 0.05` AND `sample_size >= 100`
- **Comparison operators** — `lt` (less than), `ge` (greater or equal), `eq`, `ne`, `gt`, `le`
- **Multiple reindex conditions** are ANDed together

---

## Null Handling with Forward Fill

Process hierarchical data where parent values propagate down through empty cells.

**Data:** CSV with category headers followed by subcategory rows (gaps in category column)

```yaml
template:
  syntax: TC3
  status: alpha
  source:
    kind: text
    local: ./hierarchical-data.csv
    url: https://example.com/hierarchical.csv
    row_slice: [1, auto]
    delimiter: ","
  statement:
    subject:
      method: column
      encoding: category
      fill: forward
      prioritize:
        - ChemicalEntity
    predicate: subclass_of
    object:
      method: column
      encoding: subcategory
      prioritize:
        - ChemicalEntity
  provenance:
    repo: PMID
    publication: PMID99887766
    contributors:
      - kind: curation
        name: Your Name
        date: 01 JUN 2026
```

**Key techniques:**

- **Forward fill** (`fill: forward`) propagates the last non-null value downward, mapping subcategory rows to their parent category
- **Other fill strategies** — `backward`, `min`, `max`, `mean`, `zero`, `one`

---

## Next Steps

- **[Tutorial](tutorial.md)** - Step-by-step walkthrough with synthetic data
- **[Table Configuration](configuration/table.md)** - Complete field reference
- **[Advanced Example](configuration/advanced-example.md)** - Real-world configuration with annotations
- **[CLI Reference](cli.md)** - Command-line usage
