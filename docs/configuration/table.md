# Table Configuration Reference

Table configurations define how Tablassert transforms tabular data (Excel, CSV, TSV) into knowledge graph assertions.

## Purpose

A table configuration specifies:
- Data source location and format
- How to extract subject-predicate-object triples
- Entity resolution rules (taxonomic filtering, category preferences)
- Provenance metadata
- Optional edge annotations

## Template vs Sections

Table configurations support two patterns:

### Pattern 1: Template Only

Use when processing a single table with one output.

```yaml
template:
  syntax: TC3
  source: {...}
  statement: {...}
  provenance: {...}
```

### Pattern 2: Template + Sections

Use when processing variations of the same data (different columns, predicates, etc.) while sharing common configuration.

```yaml
template:
  syntax: TC3
  source: {...}  # Shared by all sections
  provenance: {...}  # Shared by all sections

sections:
  - statement:  # Section 1: Gene-Disease
      subject: {method: column, encoding: A}
      predicate: associated_with
      object: {method: column, encoding: B}

  - statement:  # Section 2: Gene-Pathway
      subject: {method: column, encoding: A}
      predicate: participates_in
      object: {method: column, encoding: C}
```

### Merge Behavior (fastmerge)

Sections inherit from template and override specific fields:

**Dictionaries:** Recursive merge, section overrides template keys
```yaml
template:
  statement:
    subject: {encoding: A}
    predicate: related_to

sections:
  - statement:
      predicate: associated_with  # Overrides, subject stays "A"
```

**Lists:** Concatenation (extends)
```yaml
template:
  statement:
    subject:
      prioritize: [Gene]

sections:
  - statement:
      subject:
        prioritize: [Protein]  # Result: [Gene, Protein]
```

**Scalars:** Section replaces template
```yaml
template:
  syntax: TC3

sections:
  - syntax: TC2  # Overrides (not recommended)
```

### Use Cases

**Single output:** Template only
```yaml
template:
  source: {kind: text, local: data.csv, url: https://example.com/data.csv}
  statement: {...}
```

**Multiple predicates, same source:**
```yaml
template:
  source: {kind: excel, local: data.xlsx, url: https://example.com/data.xlsx}
  provenance: {repo: PMC, publication: 123, contributors: [{name: Example User, date: 27 JAN 2026}]}

sections:
  - statement: {predicate: treats}
  - statement: {predicate: prevents}
```

**Multiple columns, shared provenance:**
```yaml
template:
  source: {kind: text, local: data.csv, url: https://example.com/data.csv}
  provenance: {repo: PMID, publication: 456, contributors: [{name: Example User, date: 27 JAN 2026}]}
  statement:
    subject: {method: column, encoding: A}

sections:
  - statement: {object: {method: column, encoding: B}}
  - statement: {object: {method: column, encoding: C}}
```

## Configuration Schema

### Template Metadata

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `syntax` | String | No | Configuration version. Defaults to `"TC3"`. |
| `status` | String | No | Development status. Defaults to `"alpha"`; allowed values are `"alpha"`, `"beta"`, `"primetime"`. |

### Source

Defines the data file location and format.

#### Excel Source

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `kind` | String | No | Source kind. Model default is `"excel"`, but specify it explicitly in configs. |
| `local` | Path | Yes | Local file path for caching |
| `url` | URL | Yes | Download URL (HTTP/HTTPS). Validated as reachable at parse time. |
| `sheet` | String | No | Sheet name. Defaults to `"Sheet1"`. |
| `row_slice` | List[Int\|"auto"] | No | Two-value zero-based crop bounds: `[start, stop]`. Each value may be an integer or `"auto"`. Mutually exclusive with `rows`. |
| `rows` | List[Int] | No | Zero-based row indices to keep after any `row_slice` crop. Mutually exclusive with `row_slice`. |
| `reindex` | List[Reindex] | No | Conditional row filtering |

**Example:**
```yaml
source:
  kind: excel
  local: ./data/mydata.xlsx
  url: https://example.com/data.xlsx
  sheet: "Sheet1"
  row_slice:
    - 1  # Start at the second physical row
    - auto  # Read to end
```

#### Text Source (CSV/TSV)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `kind` | String | No | Source kind. Model default is `"text"`, but specify it explicitly in configs. |
| `local` | Path | Yes | Local file path for caching |
| `url` | URL | Yes | Download URL. Validated as reachable at parse time. |
| `delimiter` | String | No | Field delimiter. Defaults to `","`. |
| `row_slice` | List[Int\|"auto"] | No | Two-value zero-based crop bounds: `[start, stop]`. Each value may be an integer or `"auto"`. Mutually exclusive with `rows`. |
| `rows` | List[Int] | No | Zero-based row indices to keep after any `row_slice` crop. Mutually exclusive with `row_slice`. |
| `reindex` | List[Reindex] | No | Conditional filtering |

**Example:**
```yaml
source:
  kind: text
  local: ./data/mydata.tsv
  url: https://example.com/data.tsv
  delimiter: "\t"
  row_slice:
    - 1
    - auto
```

#### Reindexing (Conditional Filtering)

Filter rows based on column values.

| Field | Type | Description |
|-------|------|-------------|
| `column` | String | Source column letters to evaluate (`A`-`ZZZ`) |
| `comparison` | String | Operator. Defaults to `"ne"`; allowed values are `"eq"`, `"ne"`, `"lt"`, `"le"`, `"gt"`, `"ge"`. |
| `comparator` | String\|Int\|Float | Value to compare against. Must be a string for `"eq"`/`"ne"`, or a number for `"lt"`/`"le"`/`"gt"`/`"ge"`. |

**Example:**
```yaml
reindex:
  - column: C
    comparison: lt
    comparator: 0.05  # Keep rows where column C < 0.05
```

### Statement (Triple Definition)

Defines subject-predicate-object relationships.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `subject` | NodeEncoding | Yes | Subject entity configuration |
| `predicate` | String | No | Biolink predicate. Defaults to `"related_to"`. |
| `object` | NodeEncoding | Yes | Object entity configuration |
| `qualifiers` | List[Qualifier] | No | Edge qualifiers (context) |

**Example:**
```yaml
statement:
  subject:
    method: column
    encoding: A
    prioritize: [Gene]
  predicate: treats
  object:
    method: column
    encoding: B
    prioritize: [Disease]
```

### NodeEncoding

Defines how to extract and resolve entities.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `method` | String | No | `"value"` (literal) or `"column"` (source column letters). Defaults to `"value"`. |
| `encoding` | String\|Int\|Float | Yes | Literal value or source column letters, depending on `method` |
| `taxon` | Int | No | NCBI Taxon ID for filtering (e.g., `9606` for human) |
| `prioritize` | List[String] | No | Preferred Biolink categories (must be valid `Categories` enum values such as `Gene`, `Protein`) |
| `avoid` | List[String] | No | Excluded Biolink categories (must be valid `Categories` enum values) |
| `regex` | List[Regex] | No | Pattern replacements |
| `fill` | String | No | Null-filling strategy: `"forward"`, `"backward"`, `"min"`, `"max"`, `"mean"`, `"zero"`, `"one"` |
| `remove` | List[String] | No | Strings to filter out |
| `prefix` | String | No | Add prefix to values |
| `suffix` | String | No | Add suffix to values |
| `explode_by` | String | No | Delimiter to split multi-value cells |
| `transformations` | List[Math] | No | Mathematical transformations |

#### Method: Value vs Column

**`method: value`** - Use a literal value

```yaml
subject:
  method: value
  encoding: CHEBI:41774  # All rows get this CURIE
```

**`method: column`** - Reference a source column

Source files are read without headers, so column references are Excel-style letters:
- Column A -> `"A"`
- Column B -> `"B"`
- Column AA -> `"AA"`

```yaml
subject:
  method: column
  encoding: A  # Read from column A
```

At runtime those letters are converted internally to Polars column names such as `column_1`, but those internal names are not valid configuration values.

#### Taxonomic Filtering

**`taxon: int`** - Filter entities by organism

```yaml
subject:
  method: column
  encoding: A
  taxon: 9606  # Only human genes (Homo sapiens)
```

Common taxon IDs:
- `9606` - Homo sapiens (human)
- `10090` - Mus musculus (mouse)
- `7227` - Drosophila melanogaster (fruit fly)

#### Category Prioritization

**`prioritize: list[category]`** - Prefer specific Biolink categories

```yaml
subject:
  encoding: A
  prioritize:
    - Gene
    - Protein
```

If "TP53" maps to both Gene and Protein, prefer Gene.

**`avoid: list[category]`** - Exclude specific categories

```yaml
subject:
  method: column
  encoding: A
  prioritize:
    - OrganismTaxon
  avoid:
    - Gene
```

Prevents misclassifying organism names as genes.

#### Text Transformations

**`regex: list[{pattern, replacement}]`** - Pattern-based replacements

```yaml
subject:
  encoding: A
  regex:
    - pattern: ".*g__"
      replacement: ""  # Remove genus prefix
    - pattern: ";s__"
      replacement: " "  # Replace species separator
```

Executed in order.

> **Regex dialect:** Patterns are passed directly to Polars `str.replace_all()`, which uses the Rust [`regex`](https://docs.rs/regex/) crate. Only features supported by that engine work — in particular, **capturing groups (`(...)`, `\1`) and lookarounds (`(?=...)`, `(?<=...)`, `(?!...)`, `(?<!...)` are not supported** and will raise an error at parse time. Stick to character classes, anchors (`^`, `$`), quantifiers, alternation (`a|b`), and non-capturing groups (`(?:...)`) if grouping is needed. If a transformation is too complex to express, prefer chaining several simple substitutions or capturing the residual context in a `miscellaneous notes` annotation instead.

**`remove: list[string]`** - Filter out specific strings

```yaml
subject:
  encoding: A
  remove:
    - "^NA "  # Remove rows starting with "NA "
```

Same regex constraints apply as the `regex` field — Polars-compatible patterns only, no capturing groups or lookarounds.

**`prefix` / `suffix`** - Add text

```yaml
object:
  encoding: identifier
  prefix: "CUSTOM:"  # "123" → "CUSTOM:123"
```

#### Null Handling

**`fill: string`** - Fill null values using a strategy

Available strategies:
- `"forward"` - Fill nulls with previous non-null value
- `"backward"` - Fill nulls with next non-null value
- `"min"` - Fill with column minimum
- `"max"` - Fill with column maximum
- `"mean"` - Fill with column mean
- `"zero"` - Fill with 0
- `"one"` - Fill with 1

```yaml
subject:
  method: column
  encoding: A
  fill: forward  # Propagate values down through null rows
```

```yaml
annotations:
  - annotation: expression_level
    method: column
    encoding: C
    fill: mean  # Replace nulls with column average
```

#### Multi-Value Handling

**`explode_by: string`** - Split delimited values into multiple rows

```yaml
object:
  method: column
  encoding: B
  explode_by: ";"  # "P1;P2;P3" → 3 separate edges
```

#### Mathematical Transformations

**`transformations: list[{function, arguments}]`**

Available functions: `copysign`, `pow`

Use the `"values"` token to reference column values in transformations.

### Qualifiers

Add context to edges (anatomical location, species, etc.).

| Field | Type | Description |
|-------|------|-------------|
| `qualifier` | String | Biolink qualifier from the `Qualifiers` enum (e.g., `"species_context_qualifier"`) |
| (inherits NodeEncoding) | | All NodeEncoding fields available |

**Example:**
```yaml
qualifiers:
  - qualifier: species_context_qualifier
    method: value
    encoding: NCBITaxon:9606
```

### Provenance

Required metadata about data source.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `repo` | String | No | Repository. Defaults to `"PMC"`; allowed values are `"PMC"`, `"PMID"`. |
| `publication` | String | Yes | Repository-local identifier appended to `repo:` (e.g., `"11708054"`, `"123"`). When `repo` is `"PMC"`, the value **must** start with `PMC` followed by digits (e.g., `"PMC11708054"`, `"PMC123"`). |
| `contributors` | List[Contributor] | Yes | Curation information |

**Contributor fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `kind` | String | No | Contributor role. Defaults to `"curation"`; allowed values are `"curation"`, `"validation"`, `"tool"`. |
| `name` | String | Yes | Contributor name |
| `date` | String | Yes | Date (free format) |
| `organizations` | List[String] | No | Affiliations |
| `comment` | String | No | Notes |

**Example:**
```yaml
provenance:
  repo: PMC
  publication: 11708054
  contributors:
    - kind: curation
      name: Skye Lane Goetz
      date: 09 JAN 2025
      organizations:
        - Institute for Systems Biology
        - CalPoly SLO
      comment: Migrated from TC2 to TC3
```

### Annotations

Optional edge attributes (statistical metadata, notes, etc.).

| Field | Type | Description |
|-------|------|-------------|
| `annotation` | String | Attribute name (e.g., `"p value"`, `"sample size"`). Underscores are automatically replaced with spaces at parse time. |
| (inherits Encoding) | | All Encoding fields available (method, encoding, regex, etc.) |

**Example:**
```yaml
annotations:
  - annotation: p value
    method: column
    encoding: C  # Read from column C

  - annotation: sample size
    method: value
    encoding: 450  # Literal value for all edges

  - annotation: multiple testing correction method
    method: value
    encoding: "Benjamini Hochberg"

  # Freetext catch-all for context that doesn't fit a structured field —
  # study caveats, units, post-hoc notes, anything you'd otherwise lose.
  - annotation: miscellaneous notes
    method: value
    encoding: "Values are log2 fold-change relative to vehicle control; n=3 biological replicates per arm"
```

> **Tip:** When source data carries information that can't be cleanly mapped to a structured annotation (assay-specific caveats, non-standard units, qualitative observations), add a `miscellaneous notes` annotation rather than forcing it into another field or dropping it. It accepts both `method: value` (one note for the whole table) and `method: column` (per-row notes from the source).

## Complete Example

Minimal table configuration:

```yaml
template:
  syntax: TC3
  status: alpha

  source:
    kind: text
    local: ./data.csv
    url: https://example.com/data.csv
    row_slice: [1, auto]
    delimiter: ","

  statement:
    subject:
      method: column
      encoding: A
      prioritize: [Gene]
    predicate: associated_with
    object:
      method: column
      encoding: B
      prioritize: [Disease]

  provenance:
    repo: PMID
    publication: 12345678
    contributors:
      - kind: curation
        name: Example User
        date: 27 JAN 2026

  annotations:
    - annotation: p value
      method: column
      encoding: C
```

## Next Steps

- **[Advanced Example](advanced-example.md)** - Real-world configuration with complex transformations
- **[Graph Configuration](graph.md)** - How to orchestrate multiple tables
- **[Tutorial](../tutorial.md)** - Step-by-step walkthrough
