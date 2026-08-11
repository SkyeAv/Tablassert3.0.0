# Table Configuration Reference

Table configurations define how Tablassert transforms tabular data (Excel, CSV, TSV) into knowledge-graph assertions — author one per source table to declare its source, triple mappings, entity-resolution rules, provenance, and optional edge annotations.

## Template vs Sections

Table configurations support two patterns:

### Pattern 1: Template Only

Use when processing a single table with one output.

```yaml
template:
  source: {...}
  statement: {...}
  provenance: {...}
```

### Pattern 2: Template + Sections

Use when processing variations of the same data (different columns, predicates, etc.) while sharing common configuration.

```yaml
template:
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
  statement:
    predicate: related_to

sections:
  - statement:
      predicate: treats  # Replaces the template predicate
```

## Configuration Schema

### Source

Defines the data file location and format.

#### Excel Source

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `kind` | String | No | Source kind. Model default is `"excel"`, but specify it explicitly in configs. |
| `local` | Path | Yes | Local file path the source is read from. The file must already exist here — Tablassert does not download it. |
| `url` | List[URL] | Yes | One or more source URLs recorded as provenance (emitted as the edge `source_record_urls` list and in the RIG). At least one URL is required; supply multiple to back a single section with several links. Format-validated only; not fetched. |
| `sheet` | String | No | Sheet name. Defaults to `"Sheet1"`. |
| `row_slice` | List[PositiveInt\|"auto"] | No | Two-value zero-based crop bounds: `[start, stop]`. Each value may be a positive integer or `"auto"`. Mutually exclusive with `rows`. |
| `rows` | List[PositiveInt] | No | Zero-based row indices to keep after any `row_slice` crop. Mutually exclusive with `row_slice`. |
| `reindex` | List[Reindex] | No | Conditional row filtering |

**Example:**
```yaml
source:
  kind: excel
  local: ./data/mydata.xlsx
  url:
    - https://example.com/data.xlsx
  sheet: "Sheet1"
  row_slice: [1, auto]  # Start at the second physical row, read to end
```

> **Specify `kind` explicitly.** Tablassert selects the reader purely from the declared `kind` — `excel` reads a workbook (`sheet`), `text` scans delimited text (`delimiter`); the file on disk is never inspected to infer its format. Because `kind` carries a default, a source whose `kind` is omitted or does not match the actual file is still accepted and fed to the wrong reader, surfacing only later as a read error or garbled rows. Stating `kind` explicitly makes a mis-declared source fail fast.

#### Text Source (CSV/TSV)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `kind` | String | No | Source kind. Model default is `"text"`, but specify it explicitly in configs. |
| `local` | Path | Yes | Local file path the source is read from. The file must already exist here — Tablassert does not download it. |
| `url` | List[URL] | Yes | One or more source URLs recorded as provenance (emitted as the edge `source_record_urls` list and in the RIG). At least one URL is required; supply multiple to back a single section with several links. Format-validated only; not fetched. |
| `delimiter` | String | No | Field delimiter. Defaults to `","`. |
| `row_slice` | List[PositiveInt\|"auto"] | No | Two-value zero-based crop bounds: `[start, stop]`. Each value may be a positive integer or `"auto"`. Mutually exclusive with `rows`. |
| `rows` | List[PositiveInt] | No | Zero-based row indices to keep after any `row_slice` crop. Mutually exclusive with `row_slice`. |
| `reindex` | List[Reindex] | No | Conditional filtering |

**Example:**
```yaml
source:
  kind: text
  local: ./data/mydata.tsv
  url:
    - https://example.com/data.tsv
  delimiter: "\t"
  row_slice: [1, auto]
```

#### Reindexing (Conditional Filtering)

Filter rows based on column values.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `column` | String | Yes | Source column letters to evaluate; constrained to `^[A-Z]{1,3}$` (`A`-`ZZZ`). |
| `comparison` | String | No | Operator. Defaults to `"ne"`; allowed values are `"eq"`, `"ne"`, `"lt"`, `"le"`, `"gt"`, `"ge"`. |
| `comparator` | String\|Int\|Float | Yes | Value to compare against. Must be a string for `"eq"`/`"ne"`, or a number for `"lt"`/`"le"`/`"gt"`/`"ge"`. |

**Example:**
```yaml
reindex:
  - {column: C, comparison: lt, comparator: 0.05}  # Keep rows where column C < 0.05
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
| `taxon` | PositiveInt | No | NCBI Taxon ID for filtering (e.g., `9606` for human) |
| `prioritize` | List[String] | No | Preferred Biolink categories (must be valid `Categories` enum values such as `Gene`, `Protein`) |
| `avoid` | List[String] | No | Excluded Biolink categories (must be valid `Categories` enum values) |
| `regex` | List[Regex] | No | Pattern replacements |
| `fill` | String | No | Null-filling strategy: `"forward"`, `"backward"`, `"min"`, `"max"`, `"mean"`, `"zero"`, `"one"` |
| `remove` | List[Int\|Float\|String] | No | Regex patterns to remove (replaced with empty string) |
| `prefix` | String | No | Add prefix to values |
| `suffix` | String | No | Add suffix to values |
| `explode_by` | String | No | Delimiter to split multi-value cells |
| `transformations` | List[Math] | No | Mathematical transformations |

#### Method: Value, Column, and List

**`method: value`** - Use a literal value

```yaml
subject:
  method: value
  encoding: CHEBI:41774  # All rows get this CURIE
```

**`method: column`** - Reference a source column (Excel-style letters, since sources are read without headers)

```yaml
subject:
  method: column
  encoding: A  # Read from column A
```

At runtime those letters are converted internally to Polars column names such as `column_1`, but those internal names are not valid configuration values.

**`method: list`** - A literal list of values (annotations only); emits a real JSON array for multivalued Biolink slots

```yaml
annotations:
  - annotation: has_evidence
    method: list
    encoding: ["EFO:0001", "EFO:0002"]  # Every edge carries this array
```

`method: list` is the multivalued counterpart of `method: value`: the literal list is emitted verbatim as a JSON array, so consumers iterate values instead of walking a joined string's characters (e.g. `publications.extend(edge["has_evidence"])`). It is incompatible with the scalar string ops (`regex`, `remove`, `prefix`, `suffix`, `transformations`, `fill`, `explode_by`) — encode the final values directly. `method: list` is valid on annotations only (subject/object/qualifier nodes are single entities). (The earlier annotation `delimiter` field that split an encoded scalar into a list — unrelated to the `source.delimiter` CSV/TSV separator — has been removed in favor of this explicit list method.)

> **Literal only — no per-row lists.** A list `encoding` is a literal, so every edge carries the *same* array. `method: list` therefore replaces only the literal (`method: value`) use of the removed `delimiter`; a column-based annotation that split each cell's own value (`{annotation: has_evidence, method: column, encoding: D, delimiter: "|"}`) has no direct equivalent. `explode_by` does not fill the gap — it splits one row into many rows rather than building a per-row JSON array. Handle those sources upstream (reshape so each row carries a single value, or pre-split the column before Tablassert reads it).

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
  prioritize: [Gene, Protein]
```

If "TP53" maps to both Gene and Protein, prefer Gene.

**`avoid: list[category]`** - Exclude specific categories

```yaml
subject:
  method: column
  encoding: A
  prioritize: [OrganismTaxon]
  avoid: [Gene]
```

Prevents misclassifying organism names as genes.

#### Text Transformations

**`regex: list[{pattern: Int|Float|String, replacement: Int|Float|String}]`** - Pattern-based replacements

```yaml
subject:
  encoding: A
  regex:
    - {pattern: ".*g__", replacement: ""}   # Remove genus prefix
    - {pattern: ";s__", replacement: " "}   # Replace species separator
```

Executed in order.

> **Regex dialect:** Patterns are passed directly to Polars `str.replace_all()`, which uses the Rust [`regex`](https://docs.rs/regex/) crate. Only features supported by that engine work — in particular, **backreferences (`\1`, `\2`, …) and lookarounds (`(?=...)`, `(?<=...)`, `(?!...)`, `(?<!...)`) are not supported** and will raise an error at parse time. Plain groups `(...)` and non-capturing groups `(?:...)` *are* supported. Stick to character classes, anchors (`^`, `$`), quantifiers, alternation (`a|b`), and grouping if needed. If a transformation is too complex to express, prefer chaining several simple substitutions or capturing the residual context in a `miscellaneous notes` annotation instead.

**`remove: list[regex]`** - Regex patterns to remove

```yaml
subject:
  encoding: A
  remove: ["^NA "]  # Strip leading "NA " prefix from cell text
```

Each entry is applied as a regex replace-with-empty-string on the cell text in place (rows are not dropped). Same regex constraints apply as the `regex` field — Polars-compatible patterns only, no backreferences or lookarounds.

**`prefix` / `suffix`** - Add text

```yaml
object:
  encoding: identifier
  prefix: "CUSTOM:"  # "123" → "CUSTOM:123"
```

**Output columns: `original_<col>` vs `<col>_pre_resolution`** - For every subject/object/qualifier node, the pipeline snapshots the cell value into two columns at different stages:

- `original_<col>` - the **pristine source value**, captured immediately after the column is read (or the literal is set for `method: value`) and *before* any `fill`, `explode_by`, `regex`, `remove`, `prefix`, `suffix`, or `transformations`. Emitted for both `method: value` and `method: column`. Present in final edge output.
- `<col>_pre_resolution` - the **fully-transformed value**, captured *after* all of the above, i.e. the same text that is then normalized and resolved to a CURIE. **Internal only**: used by QC (`fullmap_audit`) and as the node marker in `compile_graph`; stripped from final edge output.

Example: with `method: column`, `encoding: A`, `remove: ["^NA "]` over a cell `"NA BRCA1"`, `original_subject` is `"NA BRCA1"` while `subject_pre_resolution` is `"BRCA1"`. Annotations never emit either column.

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

Add context to edges (anatomical location, disease context, etc.).
`species_context_qualifier` is auto-derived from resolved subject/object taxon
metadata and should not be declared manually.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `qualifier` | String | Yes | Biolink qualifier from the `Qualifiers` enum (e.g., `"anatomical_context_qualifier"`) |
| (inherits NodeEncoding) | | | All NodeEncoding fields available |

**Example:**
```yaml
qualifiers:
  - {qualifier: anatomical_context_qualifier, method: value, encoding: UBERON:0000061}
```

### Provenance

Required metadata about data source.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `repo` | String | No | Repository. Defaults to `"PMC"`; allowed values are `"PMC"`, `"PMID"`. |
| `publication` | String | Yes, unless `override` is set | Repository-local identifier emitted as a CURIE. For `repo: PMC` the value **must** start with `PMC` followed by digits (e.g., `"PMC11708054"`, `"PMC123"`) and is emitted under the `PMCID:` namespace as `PMCID:PMC...`; for `repo: PMID` it is emitted as `PMID:<publication>` (e.g., `"11708054"` → `PMID:11708054`). |
| `knowledge_level` | String | No | Biolink KL/AT knowledge level of produced edges. Defaults to `"statistical_association"`; allowed values are the `KnowledgeLevels` enum (e.g., `knowledge_assertion`, `logical_entailment`, `prediction`, `statistical_association`, `text_co_occurrence`, `observation`, `not_provided`). |
| `agent_type` | String | No | Biolink KL/AT agent type responsible for produced edges. Defaults to `"data_analysis_pipeline"`; allowed values are the `AgentTypes` enum (e.g., `manual_agent`, `automated_agent`, `data_analysis_pipeline`, `computational_model`, `text_mining_agent`, `image_processing_agent`, `manual_validation_of_automated_agent`, `not_provided`). |
| `override` | Object | No | Manual provenance for non-PMC/PMID sources. When set, it replaces repo/publication-derived provenance and `publication` must be omitted. |

**Example:**
```yaml
provenance:
  repo: PMC
  publication: "PMC11708054"
  knowledge_level: statistical_association
  agent_type: data_analysis_pipeline
```

> **Default provenance trio.** When omitted, `repo` / `knowledge_level` / `agent_type` default to `PMC` / `statistical_association` / `data_analysis_pipeline`, so every produced edge is asserted as a *statistical association* generated by an automated *data analysis pipeline* under the PubMed Central namespace. That is a sensible default for GWAS/omics tables mined from a pipeline, but it is a claim about your data's semantics: if your rows are manually curated assertions, predictions, or observations, override `knowledge_level` and `agent_type` (and `repo` / `publication`) so the emitted Biolink KL/AT provenance is accurate.

#### Manual provenance override

Use `provenance.override` when a table comes from another knowledge graph or source system whose Translator provenance cannot be derived from a PMC/PMID publication. The override is wired like the other Tablassert model classes and wins over the repo/publication auto-generation for that section's upstream sources, publications, and KL/AT. The edge `primary_knowledge_source` is **not** overridable per section — it always derives from the graph-level `infores` (see [Graph](graph.md)); put manual infores CURIEs in `upstream_resource_ids`.

```yaml
provenance:
  override:
    upstream_resource_ids:
      - infores:external-source
    publications:
      - PMCID:PMC1234567
    knowledge_level: knowledge_assertion
    agent_type: manual_agent
```

Override fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `upstream_resource_ids` | List[String] | No | Manual upstream source infores CURIEs replacing the repo-derived `PMC`/`PMID` source map — the sanctioned place for manual infores. Each entry must start with `infores:`. |
| `publications` | List[String] | No | Manual publication CURIEs. Entries must currently start with `PMCID:`; PMID compatibility for manual overrides is intentionally deferred. |
| `knowledge_level` | String | No | Override-specific KL value. Defaults to `statistical_association`. |
| `agent_type` | String | No | Override-specific AT value. Defaults to `data_analysis_pipeline`. |

Tablassert emits the graph-level infores (or `infores:<graph-name>` when unset) as the Biolink-compatible `primary_knowledge_source` edge slot — a single-element list such as `["infores:multiomics-kg"]`, matching the form of `upstream_resource_ids`. The override cannot set a per-section `primary_knowledge_source`; manual infores CURIEs belong in `upstream_resource_ids`. Older `resource_id` output has been replaced so generated KGX is compatible with the Biolink edge allow-list.

### Annotations

Optional edge attributes (statistical metadata, notes, etc.).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `annotation` | String | Yes | Attribute name (e.g., `"p_value"`, `"effect_size"`). Lowercased and trimmed of leading/trailing whitespace at parse time; underscores are preserved (use snake_case). |
| (inherits Encoding) | | | All Encoding fields available (method, encoding, regex, etc.) |

Multivalued Biolink slots such as `has_evidence` or `FDA_regulatory_approvals` — whose consumers iterate the value — are declared with [`method: list`](#method-value-column-and-list), which emits a real JSON array instead of a scalar.

**Example:**
```yaml
annotations:
  - {annotation: p_value, method: column, encoding: C}                 # Read from column C
  - {annotation: adjusted_p_value, method: column, encoding: D}        # A real Association slot -> emitted on the edge
  - {annotation: supporting_study_size, method: value, encoding: 450}  # Attached to no class -> inlined supporting study (see below)
  - {annotation: multiple_testing_correction_method, method: value, encoding: "Benjamini Hochberg"}
  - {annotation: has_evidence, method: list, encoding: ["EFO:0001", "EFO:0002"]}  # Multivalued -> a JSON array

  # Descriptive name of your choice — folded into `supporting_text` on output.
  - annotation: log2fc_relative_to_vehicle_control
    method: value
    encoding: "Values are log2 fold-change relative to vehicle control; n=3 biological replicates per arm"
```

#### Allow-list and auto-folding

Annotation names fall into three groups at build time:

- **Allowed edge fields** — names on the edge allow-list: [Biolink Association](https://biolink.github.io/biolink-model/) slots, qualifier slots, and curated KGX/Tablassert edge fields (e.g. `p_value`, `adjusted_p_value`, `knowledge_level`, `primary_knowledge_source`, `supporting_text`, `publications`, `effect_size`, `effect_type`, qualifier slots like `severity_qualifier` / `disease_context_qualifier`) are written to edges verbatim.
- **Unsatisfiable slots** — names the Biolink LinkML schema declares but attaches to **no** Pydantic class: `supporting_study_size`, `sample_size`, `relationship_strength`, `statistical_significance_qualifier`, and the other `supporting_study_*` slots. A record carrying one could never validate, so their values are routed onto the edge's **inlined supporting study** (`has_supporting_studies` → `Study` → `StudyResult`, the COHD/ICEES pattern) rather than emitted as edge fields. Declaring one is legal and loses nothing, but Tablassert emits a `BiolinkRelocationWarning` naming where the value went. This set is derived from the *installed* `biolink-model`, so a slot leaves it automatically once a release attaches it.
- **Tablassert pipeline fields** — `upstream_resource_ids`, `source_record_urls`.

Any other annotation name is treated as **supporting context**. At the end of `compile_graph`, tablassert sweeps the edge columns: for each non-allow-listed name it emits `"name: value"` entries into the edge's `supporting_text` (a `list[str]`), then drops the original column. Behavior worth knowing:

- **Pick descriptive names.** Whatever string you choose becomes the prefix in `supporting_text`, so `log2fc_relative_to_vehicle_control` reads as `"log2fc_relative_to_vehicle_control: 1.4"` on the edge. Avoid generic names like `notes` or `value`.
- **Null and blank cells produce no entry.** Whitespace-only values are treated as blank.
- **Existing `supporting_text` is preserved.** If an annotation named `supporting_text` is already on the edge (Biolink-native slot), folded entries are appended to it rather than replacing it. Scalar values are coerced to a single-element list first.
- **Ordering is stable.** Folded entries are sorted alphabetically by column name.

This means nothing in your source data is silently dropped: context that doesn't map to a structured Biolink slot travels along inside `supporting_text` instead.

In addition to user-declared annotations, every edge automatically carries `extracted_from_row_number`, a 1-based index into the original source table (matching Excel-style row numbering). It is not declared as an annotation — tablassert emits it internally so each edge always carries its source-row provenance. Together with the sheet name it identifies the edge's **inlined supporting study** (`has_supporting_studies`), where it is carried alongside any relocated unsatisfiable slots; neither is folded into `supporting_text`.

## Next Steps

- **[Advanced Example](advanced-example.md)** - Real-world configuration with complex transformations
- **[Graph Configuration](graph.md)** - How to orchestrate multiple tables
- **[Tutorial](../tutorial.md)** - Step-by-step walkthrough
