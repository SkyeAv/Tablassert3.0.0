# Changelog

All notable changes to this project are documented in this file.

## Unreleased

### Breaking Changes
- Renamed CLI commands to a consistent verb-noun scheme and retired the legacy `datassert` naming throughout: `build` → `build-graph`, `validate` → `validate-table` (`build-fullmap` is unchanged — it already fit the scheme). The `Graph` config field `datassert` is renamed to `fullmap` (also renaming the `resolve_many()` keyword argument from `datassert` to `fullmap`); existing graph YAML configs must rename their `datassert:` key to `fullmap:`. This is a pure naming migration. `build-fullmap`'s default `--output`/`--cache` directories moved from `./datassert/` to `./fullmap/`. (Note: the embedded fullmap database format was independently changed — see the redb schema item below — so existing `fullmap.redb` files **do** need to be rebuilt with `tablassert build-fullmap`.)
- Removed the `syntax` key entirely from both configuration formats. Graph configs no longer accept `syntax: GC2`/`GC3` and table configs no longer accept `syntax: TC4`; any `syntax:` field is now rejected (`extra = "forbid"`).
- Removed the `status` template-metadata field from table configurations.
- Removed the `provenance.contributors` block from table configurations. `Provenance` now carries only `repo`, `publication`, `knowledge_level`, and `agent_type`; curation-style attribution moves to the graph-level RIG `contributions` field (below).
- Removed `qc` and `log` from the graph configuration. Both are now `build-graph` CLI flags (`--qc/-q` and `--log/-l`); graph configs that set `qc:` or `log:` are rejected.
- Removed `pubmed_db` and `pmc_db` from graph configuration and removed the downstream PubMed metadata and PMC caption enrichment steps from graph builds.
- **Entity resolution was rewritten onto an embedded [redb](https://github.com/cberner/redb) database.** The DuckDB-shard store built by the external `datassert` Go CLI is replaced by a single embedded redb file built in-process by Tablassert's own Rust extension (`tablassert.rs`) — no external tool or install step is required. The on-disk fullmap schema is now `tablassert.fullmap.v3`; v1/v2 databases are rejected at lookup time and must be rebuilt with `tablassert build-fullmap`.
- **The QC runtime migrated from ONNX Runtime to PyTorch / sentence-transformers (BioBERT).** The `qc` extra now installs `torch`, `sentence-transformers`, `rapidfuzz`, `scikit-learn`, and `numpy`. The `qc-cuda` extra and all ONNX/CUDA provider selection were removed; `fullmap_audit()` no longer takes a `provider` argument and runs on the sentence-transformers backend. QC thresholds were tightened: the fuzzy stage now passes on `fuzz.ratio >= 70 OR partial_token_sort_ratio >= 80` (previously 20/30), and the BioBERT cosine-similarity stage on `>= 0.5` (previously 0.2). The cached model moved from `.tablassert/onnx/` to `.tablassert/biobert/`.
- **Removed automatic source-file downloading.** A table's `source.url` is no longer fetched into `source.local`; the input file must already exist at `source.local`. `source.url` is retained as provenance metadata — it is emitted as the edge `source_record_urls` column and recorded in the generated RIG. The parse-time URL-reachability check (`httpx.head`) and its `.cachassert/` disk cache were removed.
- Consolidated the four scattered hidden working directories (`.storassert/`, `.logassert/`, `.onnxassert/`, and the previously-documented-but-removed `.cachassert/`) into a single parent `.tablassert/` with three intuitively-named subdirectories: `.tablassert/store/` (intermediate parquet, `utils.STORE`), `.tablassert/log/` (loguru sink, `log.LOGASSERT`), and `.tablassert/biobert/` (cached BioBERT model, `qc.MODEL`). All three paths are now derived from a single `utils.BASE = Path("./.tablassert")` constant. The loguru sink file was renamed from `logassert.log` to `tablassert.log`. Any external tooling or log tailers pointed at the old paths need to be updated; the on-disk content is auto-regenerated on the next run (parquet is reproducible, the BioBERT model re-downloads if `.tablassert/biobert/` is not pre-populated).
- Removed runtime dependencies that are no longer used: `onnxruntime`, `diskcache`, `httpx`, `pyexcel`, `orjson`, `duckdb`, and `polars_hash`.

### Added
- **Rust/PyO3 extension (`tablassert.rs`)** replacing the external `datassert` tool. Public functions include `build_fullmap_db`, `lookup_fullmap_terms`, `hydrate_prefixes`/`hydrate_categories`/`hydrate_sources`/`hydrate_curies`, `dedup_ndjson`, `namespace_uuid`, and `fullmap_source_version`. Published as maturin wheels (CPython 3.11–3.14, Linux + macOS) through the PyPI workflow.
- **`tablassert build-fullmap`** now downloads NCATS Translator BABEL class/synonym files from RENCI (resumable, range-request downloads) and builds the embedded fullmap redb database in-process, with a live progress display.
- **Resource Ingest Guide (RIG) generation.** `build-graph` now also emits `<name>_<version>.RIG.yaml` alongside the nodes/edges NDJSON. A new required graph field `description` and optional `contributions` and `ui_explanation` fields populate the RIG (with sensible defaults for `contributions`/`ui_explanation`).
- **`build-graph` flags:** `--release/-r` (emit a slim, significant-only graph by dropping `biolink:not_significant` edges before resolution), `--qc/-q` (run the QC audit stage), and `--log/-l` (verbose per-section logging).
- **`build-fullmap` shorthand flags:** every `build-fullmap` option now has a short alias — `--output/-o`, `--cache/-c`, `--version/-v`, and `--threads/-t` — matching the shorthand style already used by `build-graph`.
- **Structured, coded errors** (`tablassert.errors`): `TablassertError`, `TablassertValidationError`, `GraphValidationError`, `SectionValidationError`, `QcRuntimeMissingError`, and `BabelDownloadError`, each carrying a stable error code that resolves to a documentation URL.
- Experimental Biolink feature support: per-section knowledge-level / agent-type provenance (`provenance.knowledge_level`, `provenance.agent_type`), auto-derived Biolink association edge categories (`EdgeCategories`), and a Biolink-compliant edge-column allow-list. Annotation columns not on the allow-list are folded into the edge `supporting_text` list (as `"name: value"` entries) rather than dropped.
- New edge provenance columns: `upstream_resource_ids` (from the Biolink information-resource mapping) and `source_record_urls` (list column).
- Google-style docstrings on all functions; `pytest-cov` coverage configuration; and an expanded end-to-end build-pipeline test suite.

### Changed
- The fullmap build is now parallelized with rayon: class/synonym files are processed in parallel through an in-memory sharded pipeline (equivalents map → dimension/CURIE-id assignment → term aggregation → a single write transaction), replacing the old per-row database lookups and the stage-database/temp-file copy. `--threads` now defaults to ~90% of available CPUs when unset (previously effectively single-threaded). The redb stores six tables: `records`, `prefixes`, `categories`, `sources`, `curies`, and `meta`.
- TCode operation ordering was optimized to accelerate graph build times.
- The CLI now uses a Rich-based progress interface across all pipeline stages.

### Removed
- The external `datassert` Go CLI dependency and the sharded DuckDB entity-resolution store.
- The `equivalents` and `term_records` redb staging tables (and the stage-database / temp-file build path).
- `namespace_uuid`, `basespace`, and `samphash` from `tablassert.utils` (`namespace_uuid` is now provided by the Rust extension; `utils` exposes `BASE`, `STORE`, and `mkhash`).
- The Playwright/httpx-based `downloader` module.
- The `.onnxassert/` and `.cachassert/` working directories.

### Documentation
- Comprehensive 8.0.0 documentation overhaul reconciling every surface (README, `llms.txt`, CONTRIBUTING, CITATION, and the full MkDocs site) against the codebase: removed the retired `syntax`/`status`/`contributors` config keys and the ONNX/CUDA QC documentation, documented the redb v3 schema and the new `build-graph` flags and RIG output, and corrected every shipped YAML example so it validates against the current schema.
- Upgraded the documentation site from the plain `readthedocs` theme to Material for MkDocs.
- Removed the Docker documentation (the project no longer publishes a container image).
- Added a CI test that validates the shipped example configurations against the Pydantic models and guards against reintroducing removed 8.0.0 config keys.

## 7.5.2 - 2026-07-01

### Changed
- `sig()` in `lib.py` now emits a third significance label, `"INCONCLUSIVE"`, for p-values that fall between the existing significance `cutoff` (default `0.05`, inclusive) and a new `threshold` parameter (default `0.10`, exclusive). A p-value `p` is now mapped as: null → `"UNSURE"`; `p <= cutoff` → `"YES"`; `cutoff < p < threshold` → `"INCONCLUSIVE"`; `p >= threshold` → `"NO"`. The upper bound is exclusive so `0.10` (and the `0.1` `NO` cases in the existing tests) continue to map to `"NO"`. `threshold` is added to `sig()`'s signature alongside `cutoff`; the function remains wired into `Tcode.collect()` at its default arguments, so builds are unaffected unless a caller overrides the new bound.

### Added
- One regression test in `test_lib.py` (`test_sig_marks_inconclusive_band`) asserting all four bands in a single frame: a value at/below cutoff (`YES`), a value in the inconclusive range (`INCONCLUSIVE`), and a value at the exclusive threshold (`NO`).

## 7.5.1 - 2026-07-01

### Changed
- Numeric annotation columns are now coerced and emitted as controlled-notation strings in NDJSON output instead of raw values. Two new pipeline steps wired into `Tcode.collect()` (`lib.py`): `clean_numeric()` lazily casts matching columns to `Float64` with `strict=False` (non-numeric entries drop to null), and `format_numeric()` renders them as strings — p-value columns (any name containing `"p value"`, case-insensitive) in scientific notation (`{:.4e}`), and `relationship strength` / `sample size` in decimal general format (`{:.4g}`, ≥4 significant figures). Non-matching columns are left untouched, and nulls are subsequently dropped by `strip_nulls()`. `math_op()` now also casts with `strict=False` so it tolerates residual junk in numeric annotation columns. `format_numeric()` formats via numpy-backed batch conversion rather than `map_elements` for throughput.
- Removed dead `pl.Config(set_fmt_float=...)` and `pl.Config(float_precision=...)` context managers from `compile_graph()` (`lib.py`); they were no-ops for NDJSON serialization (`write_ndjson` emits raw f64 via serde shortest-repr and ignores float display options), and the `fmt`/`precision` parameters of `compile_graph()` were removed alongside them.
- Fixed an off-by-one in the build/validate progress bar so each section loop now shows the configuration currently being processed instead of the last-completed one. `PipelineProgress.section_loop()` (`progress.py`) previously returned a single `advance(info)` callback that set the description and ticked the completed counter together, called after each item's work — so while section *K* ran the bar still displayed section *K−1*. It now returns a `(start, advance)` pair: `start(info)` updates the description to the in-flight item without incrementing, and `advance()` ticks the counter afterwards (so the counter never claims an in-flight item is complete). All five call sites in `cli.py` (TCode build, Collect, Subgraph, Graph, Validate) were updated to `start(...)` before the work and `advance()` after; the long-running Collect and Subgraph stages continue to show the full `format_section_oneline()` summary (including the `CONFIG` name) of the in-flight section.

### Added
- Fifteen regression tests in `test_lib.py` covering `numeric_columns()` detection (p-value substring, exact-name match, case-insensitivity), `clean_numeric()` (parse/coerce numeric and scientific notation, null out non-numeric junk, leave non-matching columns untouched, noop, idempotent on Float64), `format_numeric()` (scientific notation for p-value, decimal general format for relationship strength/sample size, null preservation, floating-point-noise cleaning, noop), null-stripped NDJSON rows, `compile_graph()` NDJSON emission, and `sig()` operating over a cleaned Float64 p-value column.
- Three regression tests in `tests/test_progress.py` pinning the new two-callback contract: `start` shows the in-flight item with the counter still at zero, `advance` ticks the counter without altering the description, and a `start`/`advance` cycle keeps the description synced to the current item rather than the previous one.

## 7.5.0 - 2026-07-01

### Changed
- Publication CURIEs in `compile_subgraph()` (`lib.py`) now use the `PMCID:` namespace prefix for PubMed Central sources. A `repo: PMC` section with `publication: PMC11708054` is emitted as `PMCID:PMC11708054` (previously `PMC:PMC11708054`); non-PMC repos such as `PMID` are unaffected and continue to emit `<repo>:<publication>` (e.g., `PMID:11708054`). The `repository` edge column is unchanged and still records the raw `repo` value. Extracted via a new `publication_curie()` helper.

### Added
- New `<col> table literal value` edge column for subject, object, and qualifier nodes encoded with `method: column`. Unlike the existing `original <col>` column (which snapshots the value *after* all `fill`/`explode_by`/`regex`/`remove`/`prefix`/`suffix`/`transformations`), `<col> table literal value` captures the pristine source-cell value *before* any transformation. Emitted only for column-encoded nodes; annotations and `method: value` nodes are unaffected. Implemented via a `table_literal` flag on `Tcode.encoding()`, enabled by `Tcode.node()`.
- Four regression tests in `test_lib.py`: `publication_curie()` for PMC and PMID namespaces, and two `Tcode` tcode-inspection tests covering presence/ordering of the table-literal column for column encodings and its absence for value encodings.

### Documentation
- Comprehensive accuracy pass across the API, configuration, and Docker documentation, reconciling every page against the current codebase. Highlights: corrected invalid examples that would not load (`syntax: TC2`; `publication` integers and missing `PMC` prefixes; a non-existent `Qualifiers` member; `reindex` placed at section level; a subject missing `method: column`), fixed wrong field types (`rows`/`row_slice`/`taxon` → `PositiveInt`, `remove` → regex patterns), corrected the QC fuzzy thresholds (`fuzz.ratio >= 20 OR partial_token_sort_ratio >= 30`), removed a non-existent `uuid:` prefix from `utils.md` return examples, fixed the `resolve_many()` parameter order and added the original-column-capture and optional QC-audit pipeline steps, corrected graph-config path resolution (CWD, not config-relative) and processing-flow ordering, documented the strict QC GPU no-fallback behavior and the `.cachassert/` working directory, and aligned `Categories` enum member names (`GENE`/`PROTEIN`) and Docker CI triggers with the source.

## 7.4.14 - 2026-06-30

### Changes
- Extended `sig()` in `lib.py` to select a p-value column by fuzzy matching rather than requiring an exact `"p value"` name. All schema columns whose names contain the substring `"p value"` are now considered candidates; `fuzz.ratio` (rapidfuzz) scores each against the literal `"p value"` and the highest-scoring column is used to compute the `"significant"` output. An exact `"p value"` column scores 100 and is always preferred; columns like `"adjusted p value"` or `"log p value"` are used only when no exact match is present. If no column contains the substring the function continues to emit `"UNSURE"` for all rows.
- Added five regression tests in `test_lib.py` covering: exact-match preference, non-exact fallback, closest-match selection among multiple non-exact candidates, no-p-value column (UNSURE), and null value handling.

## 7.4.13 - 2026-06-30

### Changes
- Removed the datassert prevalidation failure for unresolved `statement.subject` / `statement.object` literal encodings under `method: value`. Graph builds no longer abort during `Tcode.model_validate(...)` for cases like `"Incertae Sedis XI"`; unresolved literal values are now allowed through config validation so downstream runtime handling can decide whether they map or get filtered.
- Added a regression test at the `Tcode.model_validate(...)` layer covering an unresolved `method: value` subject encoding, matching the build-time validation path reported in the field.

## 7.4.12 - 2026-06-29

### Changes
- Expanded the placeholder-term filter regex in `distinct()` (`fullmap.py`) to drop additional non-informative terms during entity resolution. The `bad` pattern now also excludes `not applicable`, `p value`, `variable`, `result`, `exposure`, `expression`, and `symbol` alongside the existing `none`, `nan`, `na`, `null`, and `unknown`, preventing these generic column-header-like values from being sent through resolution and producing spurious CURIE mappings.

## 7.4.10 - 2026-05-29

### Changes
- Enforced explicit `biolink:` namespace prefix on predicates and qualifiers emitted by `compile_subgraph()` in `lib.py`. Both `self.statement.predicate` and `x.qualifier` (in the qualifier loop) are now prefixed via `add("biolink:", ...)`, ensuring all output edges carry fully-qualified Biolink CURIEs rather than bare predicate/qualifier names.

## 7.4.9 - 2026-05-26

### Bug Fixes
- Fixed `OSError: Too many open files` during subgraph build at large scales (700+ sections). `with_mesh()` and `with_captions()` in `lib.py` opened a `sqlite_utils.Database` per section but never closed the underlying SQLite connection, leaving FD release to GC. In the tight sequential `compile_subgraph` loop the leaked FDs accumulated past the OS soft limit, causing the next `to_store()` → `df.write_parquet()` (which polars 1.39 routes through `sink_parquet`) to fail opening its target `.storassert/*.parquet`. Both functions now wrap their query bodies in `try:` / `finally: db.conn.close()`.

## 7.4.8 - 2026-05-12

### Changes
- Expanded `fullmap_audit()` failure logging in `qc.py` to include the underlying score values that caused each rejection. Failed CURIE log lines now carry `FUZZ_RATIO`, `FUZZ_PARTIAL`, and (when the BERT stage ran) `BERT_SIMILARITY` alongside the existing `STORE`/`CONFIG`/`COL`/`ORIGINAL`/`PREFERRED`/`CURIE` fields, making it easier to diagnose why a term was dropped.
- Attached the per-row fuzzy and BERT scores as columns on the pending frame before masking, and switched the intermediate `pl.concat()` calls to `how="diagonal"` so the score columns survive concatenation with the already-passed rows.

## 7.4.7 - 2026-05-11

### Changes
- Added `Provenance.is_valid_pmc_id` model validator in `models.py` that enforces `publication` starts with `PMC` followed by digits when `repo` is `PMC` (`Repositories.PUBMED_CENTRAL`). The constraint was previously documented in 7.3.6 but only now enforced at parse time.

## 7.4.6 - 2026-05-11

### Changes
- Relaxed `BaseSource.is_real_url` validator in `models.py` to ignore `403 Forbidden` responses from `httpx.head()`. Some upstreams reject anonymous `HEAD` probes with 403 even though the URL itself is well-formed and reachable, so 403 no longer fails config validation.

## 7.4.5 - 2026-05-11

### Changes
- Cached `BaseSource.is_real_url` validator results to a `diskcache.Cache` at `.cachassert/` in `models.py`, so repeated config parses skip redundant `httpx.head()` round-trips against unchanged URLs.
- Increased `is_real_url` `httpx.head()` timeout from 5.0s to 15.0s to further reduce spurious validation failures against slow upstreams.
- Added `diskcache>=5.6.3` runtime dependency.

## 7.4.4 - 2026-05-11

### Changes
- Relaxed `BaseSource.is_real_url` validator in `models.py` to only raise on 4xx responses from `httpx.head()`. Servers that return 5xx or other non-2xx statuses to `HEAD` requests no longer fail config validation, since the URL itself is still well-formed and reachable.
- Increased `is_real_url` `httpx.head()` timeout from 3.0s to 5.0s to reduce spurious validation failures against slow upstreams.

## 7.4.3 - 2026-05-11

### Bug Fixes
- Fixed `compile_graph()` in `lib.py` stripping the final version segment from output paths. `Path(f"./{name}_{version}")` treated the trailing `.N` of a semver version (e.g. `.3` in `7.4.3`) as a suffix, so `with_suffix(".edges.ndjson.temp")` replaced the version segment instead of appending. Base path now carries a `.tmp` sentinel suffix (`Path(f"./{name}_{version}.tmp")`) that `with_suffix()` replaces, preserving the full version in emitted filenames. Temp suffixes also shortened from `.temp` to `.tmp` for consistency.

## 7.4.2 - 2026-05-07

### Changes
- Added Pydantic field and model validators to `models.py` that enforce configuration correctness at parse time: `url` fields are verified reachable via `httpx.head()`, `rows` and `row_slice` are mutually exclusive, `Reindex.comparator` type must match its `comparison` operator (`eq`/`ne` require `str`, numeric operators require `int`/`float`), `encoding` values under `method: column` must be Excel-style letters (`A`–`ZZZ`), `Regex` pattern/replacement strings are validated against the Polars regex engine, `remove` entries are validated as Polars-compatible regex, and `annotation` names have underscores replaced with spaces.
- Changed `rows` and `row_slice` element type from `NonNegativeInt` to `PositiveInt` in `BaseSource`.

## 7.4.1 - 2026-05-05

### Bug Fixes
- Fixed `AttributeError: 'str' object has no attribute 'value'` raised by `format_section_oneline()` in `progress.py` during the BUILDING TCODE stage. The `Section` model sets `use_enum_values=True`, so `Tcode.status` is already a plain string — removed the stale `.value` access.

## 7.4.0 - 2026-05-05

### Changes
- Renamed CLI commands for brevity: `build-knowledge-graph` → `build`, `verify-table-configuration-syntax` → `validate`. Version display moved from `tablassert version` subcommand to `tablassert --version` flag.
- Added `qc` parameter to `resolve_many()` for optional QC auditing during standalone batch resolution. ONNX Runtime provider is auto-detected via `get_qc_provider()`.
- Added `has_qc_runtime()` helper to `qc.py` for ONNX Runtime detection.
- Added `empty_matches()` helper to `fullmap.py` for empty result fallback.
- Added `DownloadReceipt` dataclass, `DownloadError`/`DownloadValidationError` exception classes, and `classify()`/`validate_download()`/`modernize_xls()` to `downloader.py`.
- Updated log format to include timestamps: `{time:YYYY-MM-DD HH:mm:ss}`.

### Bug Fixes
- Fixed tutorial table configuration using header names as `encoding` values instead of Excel column letters (`A`, `B`, `C`, `D`).

### Documentation
- Updated all documentation to reflect renamed CLI commands.
- Fixed tutorial and example YAML configurations to use Excel column letter references (`A`, `B`, `C`, `D`) for `method: column` encodings instead of header names, matching the headerless source reading behavior.
- Fixed `encoding` values in `docs/examples/` gallery configurations.
- Updated `resolve_many()` API reference with new `qc` parameter and auto-detected QC provider.
- Fixed CITATION.cff version (7.2.2 → 7.4.0).
- Fixed CONTRIBUTING.md lazy-loaded package list (`typer` → `cyclopts`, added missing packages).

## 7.3.6 - 2026-04-29

### Documentation
- Documented that `publication` must start with `PMC` followed by digits when `repo` is `"PMC"`.

## 7.3.5 - 2026-04-29

### Documentation
- Tightened the table-configuration reference so field requirements, defaults, accepted enum values, row indexing, and column-reference examples match the strict `Section` schema and section-merging behavior implemented in `models.py`, `ingests.py`, and the runtime loader.

## 7.3.4 - 2026-04-28

### Bug Fixes
- Fixed `downloader.from_url()` failing on URLs that trigger an immediate download. The Playwright session now opens a browser context with `accept_downloads=True`, wraps `page.goto()` inside `page.expect_download()`, and tolerates the expected `net::ERR_ABORTED` navigation error that fires when the response is a download rather than a page.

### Documentation
- Documented `miscellaneous notes` as a freetext catch-all annotation in the table configuration and advanced-example pages — used for assay caveats, non-standard units, and qualitative observations that don't map cleanly to a structured field. Supports both `method: value` (constant) and `method: column` (per-row).
- Documented Polars regex constraints for the `regex` and `remove` transforms: patterns are passed to Polars `str.replace_all()` (Rust `regex` crate), so capturing groups (`(...)` / `\1`) and lookarounds (`(?=...)`, `(?<=...)`, `(?!...)`, `(?<!...)`) are not supported and will raise at parse time. Chain simple substitutions instead, or capture residual context in a `miscellaneous notes` annotation.

## 7.3.3 - 2026-04-08

### Bug Fixes
- Changed datassert shard count to 10 (`SHARDS` constant in `fullmap.py`) to correspond to the current datassert database layout.

### Documentation
- Updated shard count references across documentation and examples to reflect the current 10-shard datassert layout.
- Corrected provenance examples so `repo` carries the namespace prefix and `publication` carries the repository-local identifier.

## 7.3.2 - 2026-04-03

### Maintenance
- Updated dependencies. No API changes.

## 7.3.1 - 2026-04-03

### Changes
- Changed `resolve_many()` return type from `dict[str, list[str]]` to `list[dict[str, Any]]` — each resolved entity is now a row dictionary, produced via `to_dicts()`.
- `resolve_many()` now preserves the original input text in an `original {col}` key on each result row.

### Documentation
- Updated `resolve_many()` API reference to match the current function signature, return type, and output format.

## 7.3.0 - 2026-04-03

### New Features
- Added `resolve_many()` to `lib` module — a standalone batch entity resolution function that resolves an iterable of text strings to CURIEs without requiring manual LazyFrame setup, NLP preprocessing, or DuckDB connection management.

### Documentation
- Added detailed API reference page for `resolve_many()` covering function signature, parameters, return value, usage examples, and integration notes.

## 7.2.2 - 2026-04-01

### Bug Fixes
- Fixed Docker publish workflow failing due to mixed-case repository owner in image tags. Hardcoded lowercase `ghcr.io/skyeav/tablassert` and switched trigger to run after autotag completion.

### Maintenance
- Updated PyPI short description.

## 7.2.1 - 2026-04-01

### Maintenance
- Improved PyPI trove classifiers. No API changes.

## 7.2.0 - 2026-03-31

### New Features
- Added `tablassert version` command to display current package version.
- Added autotag GitHub Action for automated version tagging on releases.
- Added PyPI publishing GitHub Action.
- Added Docker image publishing to GitHub Container Registry (ghcr.io).

### Changes
- Sharded datassert entity-resolution database into 16 DuckDB shards for parallel querying.
- Renamed dependency from DBssert to DATASSERT throughout.
- Separated CLI logic into dedicated `cli.py` module.
- Extracted NLP normalization into dedicated `nlp.py` module for cleaner separation of concerns.
- Implemented improved parallelization model for graph compilation.
- Annotated Pydantic model fields with `Field(...)` schema metadata.
- Renamed `fullmap.version4()` to `fullmap.resolve()` for clarity.
- Updated `fullmap` ranking to prioritize case-insensitive exact matches between normalized terms and preferred names.
- Updated `fullmap` term de-duplication to keep first occurrences, improving deterministic output ordering.
- Moved MkDocs to dev-only dependencies.

### Testing
- Added basic pytest suite covering core models, enums, ingests, lib, nlp, and utils.

### Maintenance
- Improved `.gitignore` to exclude common artifacts.

## 7.0.2 - 2026-03-23

### Changes
- Updated package metadata for the 7.0.2 release.
- Added optional `log` and `column_context` controls to `fullmap.resolve()` for more configurable entity-resolution behavior.

### Bug Fixes
- Reworked entity-resolution querying to register terms directly in DuckDB instead of writing temporary parquet files, removing tempfile lifecycle issues in `fullmap` query execution.
- Isolated unmatched-entity logging into a dedicated helper and gated it behind an explicit logging flag.

### Documentation
- Updated API reference docs to match the current `resolve()` function signature and behavior.
- Corrected QC documentation to reflect the implemented fuzzy/BERT validation pipeline.
- Fixed documentation path typos for cache/store artifact directories.

## 7.0.1 - 2026-03-17

### Documentation
- Updated installation docs to reflect `pyproject.toml` extras and added `tablassert[rt]` guidance for systems without required default Polars CPU instructions.

## 7.0.0 - 2026-03-17

### New Features
- Added pre-commit hooks for code quality (ruff linting, formatting, and pyright type checking).
- Enhanced development environment with improved VSCode settings and better gitignore including direnv support.

### Changes
- Migrated dependency management from Nix to UV for improved Python toolchain integration and simpler development workflow.
- Updated GitHub Actions workflows to use UV for deployment and documentation building.
- Removed Docker installation method from documentation to align with current supported usage.
- Removed Nix-specific installation methods and dependencies from the project.
- Removed Chromium dependency as it's no longer required for the core functionality.
- Removed random callable from codebase to simplify dependencies.
- Updated directory naming conventions for better consistency throughout the project.

### Breaking Changes
- Nix is no longer supported for development and installation. Use UV-based installation instead.
- Project now requires Python 3.11+ for compatibility with UV toolchain.

### Documentation
- Completely rewrote installation documentation to reflect UV-based development environment.
- Updated CLI and configuration documentation to remove Nix-specific sections.
- Updated project README with new installation instructions.

## 6.2.1 - 2026-03-12

### Features
- Improved QC auditing with clearer stage behavior and richer failure logging context for section/config/column tracing.
- Improved entity resolution and pipeline behavior for difficult mapping cases, including additional safeguards around nulls, strings, and column-context handling.
- Added optional `pubmed_db` and `pmc_db` graph-configuration support so enrichment can be enabled only when those databases are available.

### Bug Fixes
- Fixed multiple `fullmap` correctness issues, including handling for missing taxon values and unmatched-term edge cases.
- Fixed TCode and transform-path edge cases affecting reindex/math/null-strip behavior during section compilation.
- Fixed integration issues across lazy/eager collection boundaries to reduce incorrect intermediate outputs.

### Performance
- Optimized graph compilation by skipping empty node/edge artifacts and reducing unnecessary downstream work.

### Documentation
- Corrected stale or inaccurate docs from 6.2.0 and aligned CLI, configuration, and API references with current runtime behavior.

## 6.2.0 - 2026-02-27

### New Features
- Added `tablassert verify-table-configuration-syntax <table-config.yaml>` for fast TC3 schema validation without running a full graph build.
- Added rich progress bars across pipeline stages to improve runtime visibility during large graph builds.
- Added automated Docker publishing in CI so container images are built and distributed from the docs workflow.
- Added improved progress messaging and stage-level status updates for entity mapping and build orchestration.

### Changes
- Updated the CLI interface for graph builds from `tablassert -i <graph-config.yaml>` to `tablassert build-knowledge-graph <graph-config.yaml>`.
- Swapped hashing internals to xxHash to improve throughput in high-volume processing paths.
- Updated label-rebuild startup logic so label generation begins with clearer rebuild conditions.
- Refactored AGENTS.md hierarchy into root and scoped instruction files (`docs/`, `nix/`, `lib/tablassert/`) to reduce duplication and clarify ownership.
- Revised docs and installation guidance to align with the 6.2.0 command surface and Docker workflows.

### Breaking Changes
- Graph build invocation now requires the explicit `build-knowledge-graph` subcommand; legacy direct invocation with only `-i` is no longer the primary interface.

### Bug Fixes
- Resolved assignment and small runtime issues captured in recent maintenance commits.
- Applied lint and architecture-documentation cleanup updates to reduce drift and improve maintainability.

For full commit history, run `git log --oneline` in the repository.

## 6.1.0 - Date not tagged in repository metadata

### Notes
- Baseline release prior to the 6.2.0 CLI split and verification-command additions.
