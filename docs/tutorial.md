# Tutorial: Your First Knowledge Graph

This tutorial walks through building a simple knowledge graph from a CSV file of gene-disease associations. You'll learn the complete workflow: creating configurations, running Tablassert, and examining the output.

**Time:** 5-10 minutes

## Prerequisites

- Tablassert installed (see [Installation](installation.md))
- Required database: fullmap
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
      encoding: A
      prioritize:
        - Gene
    predicate: associated_with
    object:
      method: column
      encoding: B
      prioritize:
        - Disease
  provenance:
    repo: PMID
    publication: "12345678"
  annotations:
    - annotation: p_value
      method: column
      encoding: C
    - annotation: sample_size
      method: column
      encoding: D
```

**What this does:**
- **source**: Reads CSV, skips header row (row_slice starts at 1)
- **statement**: Creates edges where genes (subject) are `associated_with` diseases (object)
- **subject/object**: Uses `column` method to read from columns A (gene symbol) and B (disease name)
- **prioritize**: Tells entity resolution to prefer Gene/Disease categories
- **annotations**: Adds p-value (column C) and sample size (column D) as edge attributes

## Step 3: Create Graph Configuration

Create `tutorial-graph.yaml`:

```yaml
name: TUTORIAL_KG
version: 1.0.0
description: Tutorial knowledge graph built from configured tabular source data.
tables:
  - ./tutorial-table.yaml
fullmap: /path/to/fullmap
```

**Important:** Replace the database paths with your actual paths.

**What this does:**
- **name/version**: Output files will be `TUTORIAL_KG_1.0.0.nodes.ndjson`, `TUTORIAL_KG_1.0.0.edges.ndjson`, and `TUTORIAL_KG_1.0.0.RIG.yaml`
- **description**: Source-scope description written into the generated Resource Ingest Guide (required)
- **tables**: List of table configurations to process
- **fullmap**: Path to the fullmap redb file (or base directory) for entity resolution

## Step 4: Run Tablassert

```bash
tablassert build-kg tutorial-graph.yaml
```

To also run the quality-control audit stage, add `--qc` (and `--log` for verbose per-section logging):

```bash
tablassert build-kg tutorial-graph.yaml --qc --log
```

**What happens:**
1. Loads graph configuration
2. For each table config:
   - Reads the source file from disk
   - Applies row slicing
   - Resolves entities (genes and diseases)
   - Validates with the QC pipeline when `--qc` is passed (exact → fuzzy → BioBERT)
   - Creates subgraph parquet file
3. Aggregates all subgraphs
4. Exports NDJSON files and the RIG

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

Example output (numeric annotation columns are emitted as controlled-notation strings — p-values in scientific notation):
```json
{"id":"2cfea591-0f8f-33af-a7df-03da531d3359","subject":"HGNC:11998","predicate":"biolink:associated_with","object":"MONDO:0008903","p_value":"1.0000e-03","sample_size":"450"}
{"id":"7b1c9d02-5e8a-4f3b-9c1d-2a6e8f0b4d7c","subject":"HGNC:1100","predicate":"biolink:associated_with","object":"MONDO:0005041","p_value":"1.0000e-04","sample_size":"1200"}
```

**RIG file:**

```bash
cat TUTORIAL_KG_1.0.0.RIG.yaml
```

The Resource Ingest Guide records the graph's source scope (`description`), provenance (`contributions`), and a summary of its node and edge types for NCATS Translator registration.

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
