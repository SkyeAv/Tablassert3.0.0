# Advanced Example: Real-World Configuration

This page presents a real-world table configuration (ALAMV6.yaml) with annotations explaining each section.

## Overview

**Data Source:** Microbiome-chemical correlation analysis from PMC11708054

**Goal:** Extract correlations between gut microbiota and tamoxifen metabolites

**Complexity:** Excel file, complex regex for taxonomic names, statistical annotations

## Full Configuration

```yaml
template:
  syntax: TC3
  status: alpha

  # Data source: Excel file from PubMed Central
  source:
    kind: excel
    local: ./DATALAKE/ALAM.XLSX
    url: https://pmc.ncbi.nlm.nih.gov/articles/instance/11708054/bin/mbio.01679-24-s0006.xlsx
    row_slice:
      - 2  # Skip first row (header)
      - auto  # Read to end
    sheet: all correlations

  # Triple definition: Microbe correlated_with Chemical
  statement:
    subject:
      method: column
      encoding: A  # Column A contains organism names

      # Prefer organism classifications over genes
      prioritize:
        - OrganismTaxon
      avoid:
        - Gene

      # Remove invalid entries
      remove:
        - "^NA "  # Rows starting with "NA "

      # Clean taxonomic names with regex
      regex:
        # Remove genus prefix "g__"
        - pattern: ".*g__"
          replacement: ""

        # Replace species separator ";s__" with space
        - pattern: ";s__"
          replacement: " "

        # Add space after "sp" abbreviation
        - pattern: "sp"
          replacement: "sp. "

    predicate: correlated_with

    object:
      method: value
      encoding: CHEBI:41774  # All rows: 13C-tamoxifen

  # Provenance: Publication and curation info
  provenance:
    repo: PMC
    publication: PMC11708054
    contributors:
      - kind: curation
        name: Skye Lane Goetz
        date: 09 JAN 2025
        organizations:
          - Institute for Systems Biology
          - CalPoly SLO
        comment: Manual Migration From TC2 to TC3 To Test Tablassert

  # Statistical metadata as edge annotations
  annotations:
    # Fixed values
    - annotation: sample size
      method: value
      encoding: 9

    # Column values
    - annotation: p value
      method: column
      encoding: C  # Column C

    # Fixed method description
    - annotation: multiple testing correction method
      method: value
      encoding: Benjamini Hochberg

    # Column values (correlation coefficient)
    - annotation: relationship strength
      method: column
      encoding: B  # Column B (Spearman rho)

    # Fixed method
    - annotation: assertion method
      method: value
      encoding: Spearman correlation

    # Descriptive note
    - annotation: miscellaneous notes
      method: value
      encoding: Correlation analysis between microbial composition and 13C-tamoxifen abundance after FDR correction
```

## Key Techniques

### Excel Column References

Excel columns are referenced by letter:
- `encoding: A` → First column (organism names)
- `encoding: B` → Second column (correlation coefficient)
- `encoding: C` → Third column (p-value)

### Complex Regex Pipeline

The subject field uses three regex transformations in sequence:

**1. Remove genus prefix:**
```yaml
- pattern: ".*g__"
  replacement: ""
```
`"d__Bacteria;p__Firmicutes;g__Lactobacillus"` → `"Lactobacillus"`

**2. Replace species separator:**
```yaml
- pattern: ";s__"
  replacement: " "
```
`"Lactobacillus;s__rhamnosus"` → `"Lactobacillus rhamnosus"`

**3. Format species abbreviation:**
```yaml
- pattern: "sp"
  replacement: "sp. "
```
`"Lactobacillus sp"` → `"Lactobacillus sp. "`

### Taxonomic Filtering

Prevent incorrect entity resolution:

```yaml
prioritize:
  - OrganismTaxon  # Prefer organism classifications
avoid:
  - Gene  # Don't map to genes
```

Without this, "Lactobacillus" might incorrectly map to a gene with similar name.

### Mixed Annotation Methods

Combines literal values and column references:

```yaml
annotations:
  # Literal (same for all rows)
  - annotation: sample size
    method: value
    encoding: 9

  # Column (varies per row)
  - annotation: p value
    method: column
    encoding: C
```

### Subject-Predicate-Object Pattern

- **Subject:** Organism name (from column, varies per row)
- **Predicate:** `correlated_with` (fixed)
- **Object:** CHEBI:41774 (fixed CURIE for all rows)

This creates edges like:
```
Lactobacillus rhamnosus --[correlated_with]--> 13C-tamoxifen
```

## Output Example

**Nodes:**
```json
{"id":"NCBITaxon:47715","name":"Lactobacillus rhamnosus","category":["biolink:OrganismTaxon"]}
{"id":"CHEBI:41774","name":"13C-tamoxifen","category":["biolink:ChemicalEntity"]}
```

**Edges:**
```json
{
  "id":"uuid:...",
  "subject":"NCBITaxon:47715",
  "predicate":"biolink:correlated_with",
  "object":"CHEBI:41774",
  "sample size":9,
  "p value":0.001,
  "multiple testing correction method":"Benjamini Hochberg",
  "relationship strength":0.85,
  "assertion method":"Spearman correlation",
  "miscellaneous notes":"Correlation analysis between microbial composition and 13C-tamoxifen abundance after FDR correction"
}
```

## Template + Sections Example

Here's how you'd use sections if you wanted multiple predicates from the same source:

```yaml
template:
  syntax: TC3
  source: {...}  # Same source
  provenance: {...}  # Same provenance

  statement:
    subject:
      encoding: A
      prioritize: [OrganismTaxon]
      avoid: [Gene]
      regex: [...]  # Same transformations

    object:
      method: value
      encoding: CHEBI:41774

sections:
  # Section 1: Positive correlations
  - statement:
      predicate: positively_correlated_with
    reindex:
      - column: B  # Correlation coefficient
        comparison: gt
        comparator: 0

  # Section 2: Negative correlations
  - statement:
      predicate: negatively_correlated_with
    reindex:
      - column: B
        comparison: lt
        comparator: 0
```

This produces two sets of edges from one table:
1. Positive correlations (rho > 0)
2. Negative correlations (rho < 0)

## Next Steps

- **[Table Configuration Reference](table.md)** - Full field documentation
- **[Tutorial](../tutorial.md)** - Start with a simpler example
