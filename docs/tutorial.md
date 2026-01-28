# Tutorial: Your First Knowledge Graph

This tutorial walks through building a simple knowledge graph from a CSV file of gene-disease associations. You'll learn the complete workflow: creating configurations, running Tablassert, and examining the output.

**Time:** 5-10 minutes

## Prerequisites

- Tablassert installed (see [Installation](installation.md))
- Required databases: dbssert, pubmed_db, pmc_db
- Basic familiarity with YAML

## The Data

We have a CSV file with gene-disease associations:

```csv
gene_symbol,disease_name,p_value,sample_size
TP53,lung cancer,0.001,450
BRCA1,breast cancer,0.0001,1200
EGFR,colorectal cancer,0.005,680
KRAS,pancreatic cancer,0.002,320
```

**Goal:** Transform this into KGX-compliant nodes and edges.

## Step 1: Create the Data File

Save the CSV as `tutorial-data.csv`:

```bash
cat > tutorial-data.csv <<'EOF'
gene_symbol,disease_name,p_value,sample_size
TP53,lung cancer,0.001,450
BRCA1,breast cancer,0.0001,1200
EGFR,colorectal cancer,0.005,680
KRAS,pancreatic cancer,0.002,320
EOF
```

## Step 2: Create Table Configuration

Create `tutorial-table.yaml`:

```yaml
template:
  syntax: TC3
  status: alpha
  source:
    kind: text
    local: ./tutorial-data.csv
    url: https://example.com/data.csv
    row_slice:
      - 1
      - auto
    delimiter: ","
  statement:
    subject:
      method: column
      encoding: gene_symbol
      prioritize:
        - Gene
    predicate: associated_with
    object:
      method: column
      encoding: disease_name
      prioritize:
        - Disease
  provenance:
    repo: PUBMED
    publication: PMID12345678
    contributors:
      - kind: curation
        name: Tutorial Example
        date: 27 JAN 2026
        organizations:
          - Example Institute
        comment: Synthetic tutorial data
  annotations:
    - annotation: p value
      method: column
      encoding: p_value
    - annotation: sample size
      method: column
      encoding: sample_size
```

**What this does:**
- **source**: Reads CSV, skips header row (row_slice starts at 1)
- **statement**: Creates edges where genes (subject) are `associated_with` diseases (object)
- **subject/object**: Uses `column` method to read from gene_symbol and disease_name columns
- **prioritize**: Tells entity resolution to prefer Gene/Disease categories
- **annotations**: Adds p-value and sample size as edge attributes

## Step 3: Create Graph Configuration

Create `tutorial-graph.yaml`:

```yaml
syntax: GC2
name: TUTORIAL_KG
version: 1.0.0
tables:
  - ./tutorial-table.yaml
dbssert: /path/to/dbssert.duckdb
pubmed_db: /path/to/PubMed.db
pmc_db: /path/to/PMCSuppCaptions.db
```

**Important:** Replace the database paths with your actual paths.

**What this does:**
- **name/version**: Output files will be `TUTORIAL_KG_1.0.0.nodes.ndjson` and `TUTORIAL_KG_1.0.0.edges.ndjson`
- **tables**: List of table configurations to process
- **databases**: Paths to entity resolution and provenance databases

## Step 4: Run Tablassert

```bash
tablassert-cli -i tutorial-graph.yaml
```

**What happens:**
1. Loads graph configuration
2. For each table config:
   - Downloads file (or uses local)
   - Applies row slicing
   - Resolves entities (genes and diseases)
   - Validates with QC pipeline (exact → fuzzy → BERT)
   - Creates subgraph parquet file
3. Aggregates all subgraphs
4. Exports NDJSON files

## Step 5: Examine Output

**Nodes file:**

```bash
head -n 3 TUTORIAL_KG_1.0.0.nodes.ndjson
```

Example output:
```json
{"id":"HGNC:11998","name":"TP53","category":["biolink:Gene"],"taxon":"NCBITaxon:9606"}
{"id":"MONDO:0008903","name":"lung cancer","category":["biolink:Disease"]}
{"id":"HGNC:1100","name":"BRCA1","category":["biolink:Gene"],"taxon":"NCBITaxon:9606"}
```

**Edges file:**

```bash
head -n 2 TUTORIAL_KG_1.0.0.edges.ndjson
```

Example output:
```json
{"id":"uuid:...","subject":"HGNC:11998","predicate":"biolink:associated_with","object":"MONDO:0008903","p value":0.001,"sample size":450}
{"id":"uuid:...","subject":"HGNC:1100","predicate":"biolink:associated_with","object":"MONDO:0005041","p value":0.0001,"sample size":1200}
```

## Understanding the Transformation

**Input:** Text strings ("TP53", "lung cancer")

**Entity Resolution:**
- "TP53" → `HGNC:11998` (Gene)
- "lung cancer" → `MONDO:0008903` (Disease)

**Quality Control:**
- Stage 1: Exact match check
- Stage 2: Fuzzy matching (if needed)
- Stage 3: BERT semantic similarity (if needed)

**Output:** KGX-compliant nodes and edges with:
- Standardized identifiers (CURIEs)
- Biolink categories and predicates
- Provenance metadata
- Edge annotations

## What You Learned

- **Table configuration** defines data sources and transformations
- **Graph configuration** orchestrates multiple tables
- **Entity resolution** maps text to standardized identifiers
- **QC pipeline** validates mappings across three stages
- **Output** is KGX-compliant NDJSON ready for NCATS Translator

## Next Steps

- **[Configuration Reference](configuration/table.md)** - Learn all configuration options
- **[Advanced Example](configuration/advanced-example.md)** - See real-world usage with complex transformations
- **[API Reference](api/fullmap.md)** - Understand entity resolution internals
