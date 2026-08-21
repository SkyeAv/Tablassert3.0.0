# Graph Configuration Reference

Graph configurations orchestrate one or more [table configurations](table.md) into a single knowledge-graph build: author one to produce KGX output with `tablassert build-kg` (see the [CLI reference](../cli.md#build-kg)). To check a single table config on its own, use `tablassert validate <table.yaml> --schema table`.

## Purpose

A graph configuration file specifies:

- Output knowledge graph name and version
- List of table configurations to process
- Database location for entity resolution
- The **required `rig:` section**: all Resource Ingest Guide (RIG) metadata emitted as `<name>_<version>.RIG.yaml`

QC auditing and verbose logging are controlled at build time via the `build-kg --qc` and `build-kg --log` flags: they are **not** graph-config fields.

## Schema

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | String | Knowledge graph name (used in output filenames and the RIG) |
| `version` | String | Knowledge graph version (used in output filenames) |
| `tables` | List[Path] | Paths to table configuration YAML files |
| `fullmap` | Path | Path to the fullmap redb file, or a base directory containing it |
| `rig` | RIGConfig | Resource Ingest Guide metadata (see below) |

The legacy top-level RIG fields (`description`, `contributions`, `ui_explanation`, `infores`) are **rejected** with a migration pointer; they now live under `rig:`.

### The `rig:` section

The `rig:` section carries every human-authored RIG fact. Its shape mirrors the released [RIG schema](https://github.com/biolink/resource-ingest-guide-schema), so the generated `.RIG.yaml` is always schema-shaped. The generator derives only mechanical facts from the build (generated artifact file entries, observed edge/node type summaries) and **validates the complete document before writing anything**, so a build never leaves behind an invalid or incomplete RIG.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `rig.name` | String | No | RIG display name; defaults to `<name> v<version> Resource Ingest Guide` |
| `rig.source_info` | Object | Yes | Information about the source being ingested (see below) |
| `rig.ingest_info` | Object | Yes | Rationale and scope of the ingest (see below) |
| `rig.target_info` | Object | No | Target-level `future_considerations` and `additional_notes` (edge/node type summaries are always generated) |
| `rig.ui_explanation` | String | No | Per-edge-type UI explanation **prefix**; the built-in Tablassert explanation is always appended after it |
| `rig.provenance_info` | Object | Yes | Contributor statements and provenance artifacts |
| `rig.supporting_data_source_info` | List[Object] | No | Upstream data sources for data-derived graphs (each needs `infores_id`, `terms_of_use_info`, and `relevant_files`) |
| `rig.artifact_base_url` | String | Yes | Public URL prefix for the generated KGX artifacts; each `.nodes.ndjson`/`.edges.ndjson` name is appended to build RIG `relevant_files` locations |
| `rig.artifact_base_path` | Path | Yes | Output directory the generated artifacts (and the RIG) are written into; created when missing |

#### `rig.source_info`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `infores_id` | `infores:` CURIE | Yes | Infores of the source this graph ingests; also emitted as the edge `primary_knowledge_source` and node `provided_by`. There is no implicit derivation from the graph name |
| `name` | String | No | Human-readable source name |
| `description` | String | No | What the source contains and how its knowledge is produced |
| `citations` | List[String] | No | PMIDs, DOIs, URLs, or free-text citations |
| `terms_of_use_info` | Object | Yes | At least one of `terms_of_use_url`, `terms_of_use_description`, `license_name`, `license_url` must carry a real assessment |
| `data_access_locations` | List[String] | Yes | Where the upstream source data is accessed; each entry must contain an http(s) or file URL |
| `data_provision_mechanisms` | List[Enum] | No | `file_download`, `api_endpoint`, `database_dump`, `other` |
| `data_formats` | List[Enum] | No | `tsv`, `csv`, `xml`, `json`, `yaml`, `obo`, `protobuff`, `kgx`, `mysql`, `postgresql`, `sqlite`, `other` |
| `data_versioning_and_releases` | String | No | How the source versions/releases its data |
| `source_status` | Enum | Yes | `maintained_regular_updates`, `maintained_as_needed_updates`, `not_maintained`, `unknown` |
| `additional_notes` | List[String] | No | Anything not captured by dedicated fields |

#### `rig.ingest_info`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ingest_categories` | List[Enum] | No | Defaults to `[translator_knowledge_creator]`; also `primary_knowledge_provider`, `aggregation_provider`, `aggregation_interpreter`, `supporting_data_provider`, `ontology_provider`, `node_property_only_provider`, `other` |
| `utility` | String | Yes | Why the source is ingested and its utility for Translator use cases |
| `scope` | String | Yes | High-level narrative of what is included and excluded |
| `relevant_files` | List[Object] | No | **Upstream** source files: `file_name`, `location` (URL), optional `description`. Entries are cross-checked against the table configs' source URLs/local files, and an entry matching no configured source fails the build |
| `included_content` | List[Object] | No | Upstream `file_name` / `included_records` / optional `fields_used` entries |
| `filtered_content` | List[Object] | No | `file_name` / `filtered_records` / `rationale` entries |
| `future_considerations` | List[Object] | No | `category` (`edge_content`, `node_property_content`, `edge_property_content`, `other`), `consideration`, optional `relevant_files` |
| `additional_notes` | List[String] | No | Extra ingest notes |

The generator **prepends** two `relevant_files` entries and two `included_content` entries for the generated artifacts (`<name>_<version>.nodes.ndjson` and `.edges.ndjson`) with their exact output names, the composed `artifact_base_url` locations, and observed record counts/fields.

#### `rig.provenance_info`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `contributions` | List[String] | Yes | Who contributed and how (e.g. `"Name - code author, data modeling"`) |
| `artifacts` | List[String] | No | Links/descriptions of external provenance artifacts (tickets, surveys, repos) |

### What the generator derives

Everything under `target_info.edge_type_info` and `target_info.node_type_info` is computed from the **final emitted KGX files** after deduplication:

- **Edge types** (one per observed predicate): subject/object categories resolved from the emitted nodes, list-valued `knowledge_level`/`agent_type`, role-separated `primary_knowledge_sources` / `supporting_data_sources` / `aggregator_knowledge_sources` from each edge's `sources` retrieval provenance, observed `edge_properties`, qualifier shapes (enumerated literal values or identifier prefixes for CURIE-valued qualifiers), and `source_files` taken from the upstream `source_record_urls` (never the output filenames).
- **Node types**: observed categories and the identifier prefixes actually emitted (`source_identifier_types`); categories with prefix-less identifiers get a factual free-text entry.
- **UI explanation**: `rig.ui_explanation` (when set) followed by the built-in Tablassert explanation; the default text is always present.

### Built-in RIG validation

Before the `.RIG.yaml` is written, the build audits the assembled document and fails with `[rig-validation-failed]` (writing nothing) when:

- any required field is missing/empty or outside the RIG schema's enums;
- the generated artifact entries are missing or their locations disagree with `artifact_base_url`;
- edge/node summaries disagree with the observed graph (categories, predicates, KL/AT values, infores CURIEs);
- a configured upstream `relevant_files` entry matches no table source file or URL.

This is what makes every emitted RIG PR-worthy by construction: placeholders and silent gaps fail the build instead of shipping.

## Minimal Example

```yaml
name: MY_GRAPH
version: 1.0.0
tables:
  - ./my-table.yaml
fullmap: /data/fullmap
rig:
  source_info:
    infores_id: infores:my-graph
    terms_of_use_info:
      terms_of_use_url: https://example.org/terms
    data_access_locations:
      - My source downloads - https://example.org/downloads
    source_status: maintained_regular_updates
  ingest_info:
    utility: Why this content matters for Translator queries.
    scope: What this graph includes and excludes.
  provenance_info:
    contributions:
      - "Author Name - code author, data modeling"
  artifact_base_url: https://example.org/my-graph
  artifact_base_path: ./published/my-graph
```

## Multi-Table Example

```yaml
name: MULTIOMICS_KG
version: UNSTABLE
tables:
  - /configs/gene-disease-associations.yaml
  - /configs/drug-targets.yaml
  - /configs/protein-interactions.yaml
fullmap: /databases/fullmap
rig:
  source_info:
    infores_id: infores:multiomics-kg
    name: Multiomics supplementary tables
    description: Multi-omics associations mined from curated supplementary tables.
    citations:
      - https://doi.org/10.3389/fsysb.2025.1544432
    terms_of_use_info:
      terms_of_use_url: https://pmc.ncbi.nlm.nih.gov/about/copyright/
      terms_of_use_description: PubMed Central open-access subset; individual article licenses apply.
    data_access_locations:
      - PubMed Central - https://pmc.ncbi.nlm.nih.gov/
    data_provision_mechanisms:
      - file_download
    data_formats:
      - kgx
    source_status: maintained_as_needed_updates
  ingest_info:
    ingest_categories:
      - translator_knowledge_creator
    utility: Statistical associations supporting hypothesis generation for gene-disease and drug-target queries.
    scope: Gene-disease, drug-target, and protein-interaction assertions from the three configured tables.
    relevant_files:
      - file_name: gene-disease.tsv
        location: https://pmc.ncbi.nlm.nih.gov/articles/instance/example/bin/gene-disease.tsv
        description: Gene-disease association table.
  ui_explanation: Microbiome-host associations derived from multi-omics supplementary tables.
  provenance_info:
    contributions:
      - "Author Name - code author, data modeling"
    artifacts:
      - "Ingest ticket: https://github.com/NCATSTranslator/Data-Ingest-Coordination-Working-Group/issues/1"
  artifact_base_url: https://example.org/multiomics-kg
  artifact_base_path: /published/multiomics-kg
```

## Processing Flow

When you run `tablassert build-kg graph.yaml`:

1. **Load graph configuration** - Parse YAML, validate schema (including the full `rig:` section)
2. **Load table configurations** - Parse each YAML in `tables`
3. **Extract sections** - Expand templates into per-section `Tcode` instances
4. **Collect instructions (per section):**
   - Read the source file from disk (`source.local`)
   - Apply transformations and resolve entities using `fullmap`
   - Validate with the QC audit when `build-kg --qc` is passed
5. **Build subgraphs** - Compile each section's resolved data into a parquet file
6. **Compile graph** - Aggregate all subgraph parquets, export `{name}_{version}.nodes.ndjson` / `.edges.ndjson` into `rig.artifact_base_path`, summarize the final graph, audit the RIG document, and write `{name}_{version}.RIG.yaml`

## Output Files

Given this configuration:

```yaml
name: EXAMPLE_KG
version: 2.0.0
rig:
  artifact_base_path: ./published/example-kg
  # ...
```

Produces (inside `./published/example-kg/`):

- `EXAMPLE_KG_2.0.0.nodes.ndjson`
- `EXAMPLE_KG_2.0.0.edges.ndjson`
- `EXAMPLE_KG_2.0.0.RIG.yaml`

## PR-readiness for Translator Ingests

When the generated RIG will back a PR to [`NCATSTranslator/translator-ingests`](https://github.com/NCATSTranslator/translator-ingests):

- Use an **infores that is registered** (or being registered) in the [information resource registry](https://github.com/biolink/information-resource-registry).
- Replace any `file://` artifact base with the **public https location** where the KGX files will be served.
- Fill `terms_of_use_info` with the source's actual license/terms assessment, and `data_versioning_and_releases` with how the upstream source releases data.
- Describe upstream source files in `rig.ingest_info.relevant_files` / `included_content` / `filtered_content`: the generator cross-checks them against your table configs but the semantics are yours.
- Give every edge type's provenance real contributors under `rig.provenance_info.contributions`.

## Next Steps

- **[Table Configuration](table.md)** - Learn how to define table transformations
- **[Fullmap](../fullmap.md)** - Entity-resolution database build and schema
- **[Tutorial](../tutorial.md)** - Complete example walkthrough
