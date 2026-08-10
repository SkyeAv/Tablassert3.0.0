from __future__ import annotations

import re
import sys
import time
from collections.abc import Callable
from importlib.metadata import version as get_version
from itertools import chain
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, BinaryIO, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cyclopts

from tablassert._lazy import LazyModule
from tablassert.errors import BabelDownloadError, GraphValidationError, SectionValidationError
from tablassert.log import cat

if TYPE_CHECKING:
    import pydantic

    from tablassert.lib import Tcode
    from tablassert.models import Graph
    from tablassert.progress import PipelineProgress
else:
    pydantic = LazyModule("pydantic")

# Pipeline completion events (BUILD, VALIDATE).
logger = cat("PIPELINE")
# BABEL downloader events (reuse, restart, done, retry).
download_logger = cat("DOWNLOAD")

APP: cyclopts.App = cyclopts.App(
    version=f"tablassert {get_version('tablassert')}", help="Extract Knowledge Assertions From Tabular Data Into KGX NDJSON"
)

BABEL_BASE: str = "https://stars.renci.org/var/babel_outputs"
BABEL_VERSION: str = "2026jul22"
BABEL_CLASS_ENDPOINTS: tuple[str, ...] = ("kgx/",)
BABEL_SYNONYM_ENDPOINTS: tuple[str, ...] = ("synonyms/", "synonyms-conflated/")
BABEL_EXCLUDE_PREFIXES: tuple[str, ...] = ("Publication", "GeneProteinConflated")
BABEL_CLASS_RE: re.Pattern[str] = re.compile(r'<a href="([^"]*_nodes[^"]*\.gz)"')
BABEL_SYNONYM_RE: re.Pattern[str] = re.compile(r'<a href="([^"]+\.gz)"')


def _load_table_indexed(args: tuple[int, Path]) -> tuple[int, object]:
    """Load one table, tagged with its input index (multiprocessing worker).

    Runs in a pool subprocess, so it re-imports ``from_yaml`` locally (the
    deferred import mirrors ``build_pipeline``). The carried index lets the
    caller reassemble results in input order even though ``imap_unordered``
    yields them in completion order.

    Args:
        args: ``(index, table_path)`` pair for one table.

    Returns:
        ``(index, parsed_yaml)`` so the caller can position the result by index.
    """
    from tablassert.ingests import from_yaml

    idx, table = args
    return idx, from_yaml(table)


def _extract_sections_indexed(args: tuple[int, object, Path]) -> tuple[int, list[dict[str, Any]]]:
    """Extract sections from one loaded table, tagged with its input index (multiprocessing worker).

    Runs in a pool subprocess, so it re-imports ``to_sections`` locally (the
    deferred import mirrors ``build_pipeline``). The carried index lets the
    caller reassemble per-table section lists in input order so the flattened
    ``sections`` order is byte-identical to the old ``starmap`` result.

    Args:
        args: ``(index, parsed_yaml, table_path)`` triple for one table.

    Returns:
        ``(index, section_list)`` so the caller can position the result by index.
    """
    from tablassert.ingests import to_sections

    idx, raw, table = args
    return idx, to_sections(raw, table)  # pyright: ignore


def _load_graph(configuration_file: Path) -> Graph:
    """Load and validate the Graph config that drives a build.

    Args:
        configuration_file: Graph YAML path.

    Returns:
        The validated Graph model.

    Raises:
        GraphValidationError: If the graph fails Pydantic validation.
    """
    from tablassert.ingests import from_yaml
    from tablassert.models import Graph
    from tablassert.progress import flatten_pydantic_error

    raw: object = from_yaml(configuration_file)
    try:
        return Graph.model_validate(raw)
    except pydantic.ValidationError as e:
        raise GraphValidationError(configuration_file, flatten_pydantic_error(e)) from e


def build_pipeline(
    configuration_file: Path, progress: PipelineProgress, release: bool = False, qc: bool = False, log: bool = False, head: bool = False
) -> None:
    """Build a knowledge graph from a YAML configuration file.

    Runs the six-stage build pipeline: load tables → extract sections → build
    Tcodes → collect instructions → build subgraphs → compile graph.

    Args:
        configuration_file: Path to the graph YAML file.
        progress: Pipeline progress reporter.
        release: When ``True``, emit release-mode artifacts.
        qc: When ``True``, run quality-control audits on each section.
        log: When ``True``, enable per-section verbose logging.
        head: When ``True``, preview a random sample of up to 5 rows per section (fast schema/shape check).

    Raises:
        GraphValidationError: If the graph YAML fails Pydantic validation.
        SectionValidationError: If any section fails Pydantic validation.
    """
    from tablassert.fullmap import fullmap_db_path
    from tablassert.lib import Tcode, compile_graph, compile_subgraph
    from tablassert.progress import flatten_pydantic_error, format_section_compact
    from tablassert.utils import STORE, mkhash

    # Stage 1/6: load tables.
    progress.stage("Loading Tables")
    g: Graph = _load_graph(configuration_file)
    # imap_unordered yields in completion order, so each worker carries its input
    # index and we reassemble by index to keep raw[i] aligned with g.tables[i].
    start, advance, _ = progress.section_loop(len(g.tables), "Load")
    raw: list[object] = [None for _ in g.tables]
    with Pool() as pool:
        for idx, parsed in pool.imap_unordered(_load_table_indexed, enumerate(g.tables)):
            raw[idx] = parsed
            start(str(g.tables[idx]))
            advance()

    # Stage 2/6: extract sections.
    progress.stage("Extracting Sections")
    # Same index-carrying reassembly keeps temp[i] aligned with g.tables[i], so the
    # flattened sections order is byte-identical to the old starmap result.
    start, advance, _ = progress.section_loop(len(g.tables), "Extract")
    temp: list[list[dict[str, Any]]] = [[] for _ in g.tables]
    with Pool() as pool:
        for idx, section_list in pool.imap_unordered(_extract_sections_indexed, zip(range(len(g.tables)), raw, g.tables, strict=True)):
            temp[idx] = section_list
            start(str(g.tables[idx]))
            advance()
    sections: list[dict[str, Any]] = list(chain.from_iterable(temp))
    n: int = len(sections)

    # Stage 3/6: build Tcode.
    progress.stage("Building TCode")
    start, advance, _ = progress.section_loop(n, "TCode")
    tcode: list[Tcode] = []
    for s in sections:
        h: str = mkhash(s)
        start(f"{Path(str(s['config'])).stem} · {h[:8]}")
        # --head preview builds cache to a distinct .head.parquet so they never clobber full builds.
        store: Path = STORE / (f"{h}.head.parquet" if head else f"{h}.parquet")
        try:
            tcode.append(
                Tcode.model_validate(
                    {**s, "store": store, "log": log, "qc": qc, "release": release, "head": head, "name": g.name, "infores": g.infores}
                )
            )
        except pydantic.ValidationError as e:
            raise SectionValidationError(configuration_file, h, flatten_pydantic_error(e)) from e
        advance()

    db: Path = fullmap_db_path(g.fullmap)

    # Stage 4/6: collect instructions.
    progress.stage("Collecting Instructions")
    start, advance, sub_step = progress.section_loop(n, "Collect")
    instructions: list[Any] = []
    for x in tcode:
        start(format_section_compact(x))
        sub_step("planning")
        instructions.append(x.collect(db))
        advance()

    # Stage 5/6: build subgraphs.
    progress.stage("Building Subgraphs")
    start, advance, sub_step = progress.section_loop(n, "Subgraph")
    subgraphs: list[Path] = []
    for x, op in zip(tcode, instructions, strict=True):
        start(format_section_compact(x))
        # on_phase drives the per-op sub-step indicator (load → filter → resolve → write ...).
        subgraphs.append(op if isinstance(op, Path) else compile_subgraph(op, on_phase=sub_step))
        advance()

    # Stage 6/6: compile graph.
    progress.stage("Compiling Graph")
    start, advance, sub_step = progress.section_loop(len(subgraphs), "Graph")
    start(f"{g.name} · v{g.version}")
    # on_phase drives the phase tag (scan → normalize → write-nodes → write-edges → dedup → rig);
    # on_subgraph ticks the bar once per subgraph, so the total is len(subgraphs).
    compile_graph(
        subgraphs, g.name, g.version, g.description, g.contributions, g.ui_explanation, g.tables, g.infores, on_phase=sub_step, on_subgraph=advance
    )

    logger.info("Built graph {name} v{version}: {n} sections", name=g.name, version=g.version, n=n)


def validate_pipeline(table_configuration_file: Path, progress: PipelineProgress) -> None:
    """Validate section syntax from a YAML configuration file.

    Runs the three-stage validate pipeline: load tables → extract sections →
    validate section syntax (no execution).

    Args:
        table_configuration_file: Path to the table YAML file.
        progress: Pipeline progress reporter.

    Raises:
        SectionValidationError: If any section fails Pydantic validation.
    """
    from tablassert.ingests import from_yaml, to_sections
    from tablassert.lib import Tcode
    from tablassert.progress import flatten_pydantic_error
    from tablassert.utils import STORE, mkhash

    # Stage 1/3: load tables.
    progress.stage("Loading Tables")
    r: object = from_yaml(table_configuration_file)

    # Stage 2/3: extract sections.
    progress.stage("Extracting Sections")
    sections: list[dict[str, Any]] = to_sections(r, table_configuration_file)  # pyright: ignore
    n: int = len(sections)

    # Stage 3/3: validate section syntax.
    progress.stage("Validating Section Syntax")
    start, advance, _ = progress.section_loop(n, "Validate")
    for s in sections:
        h: str = mkhash(s)
        start(f"{Path(str(s['config'])).stem} · {h[:8]}")
        try:
            Tcode.model_validate({**s, "store": (STORE / f"{h}.parquet")})
        except pydantic.ValidationError as e:
            raise SectionValidationError(table_configuration_file, h, flatten_pydantic_error(e)) from e
        advance()

    logger.info("Validated {n} sections from {config}", n=n, config=table_configuration_file.name)


def validate_graph_pipeline(configuration_file: Path, progress: PipelineProgress) -> None:
    """Validate a graph config and every table it references (no execution).

    Runs a two-stage validate pipeline: validate the Graph model, then validate
    each referenced table's sections through the same Tcode path
    ``validate_pipeline`` uses.

    Args:
        configuration_file: Path to the graph YAML file.
        progress: Pipeline progress reporter.

    Raises:
        GraphValidationError: If the graph YAML fails Pydantic validation.
        SectionValidationError: If any referenced table section fails validation.
    """
    from tablassert.ingests import from_yaml, to_sections
    from tablassert.lib import Tcode
    from tablassert.models import Graph
    from tablassert.progress import flatten_pydantic_error
    from tablassert.utils import STORE, mkhash

    # Stage 1/2: validate the graph config.
    progress.stage("Validating Graph")
    r: object = from_yaml(configuration_file)
    try:
        g: Graph = Graph.model_validate(r)
    except pydantic.ValidationError as e:
        raise GraphValidationError(configuration_file, flatten_pydantic_error(e)) from e

    # Stage 2/2: validate every referenced table's sections.
    progress.stage("Validating Tables")
    # Expand each referenced table into sections up front so the bar total is known.
    table_sections: list[tuple[Path, dict[str, Any]]] = []
    for table in g.tables:
        raw_table: object = from_yaml(table)
        sections: list[dict[str, Any]] = to_sections(raw_table, table)  # pyright: ignore
        for section in sections:
            table_sections.append((table, section))
    n: int = len(table_sections)
    start, advance, _ = progress.section_loop(n, "Validate")
    for table, s in table_sections:
        h: str = mkhash(s)
        start(f"{Path(str(s['config'])).stem} · {h[:8]}")
        try:
            Tcode.model_validate({**s, "store": (STORE / f"{h}.parquet")})
        except pydantic.ValidationError as e:
            raise SectionValidationError(table, h, flatten_pydantic_error(e)) from e
        advance()

    logger.info("Validated graph {name}: {n} sections across {tables} tables", name=g.name, n=n, tables=len(g.tables))


def run(stages: int, fn: Any, arg: Path, **kwargs: Any) -> None:
    from tablassert.log import LOG_FORMAT, logger
    from tablassert.progress import PipelineProgress

    with PipelineProgress(total_stages=stages) as progress:
        sink_id: int = logger.add(progress.log_sink, level="INFO", format=LOG_FORMAT)
        try:
            fn(arg, progress, **kwargs)
        finally:
            logger.remove(sink_id)


def babel_urls(version: str, endpoints: tuple[str, ...], pattern: re.Pattern[str]) -> list[tuple[str, str]]:
    """Discover BABEL files using the RENCI directory-listing convention.

    Mirrors the legacy Datassert tool this replaces: fetch each endpoint's
    HTML listing, apply ``pattern`` to extract compressed file URLs, and drop
    any file whose name starts with a banned prefix.

    Args:
        version: BABEL version label inserted into the URL template.
        endpoints: Subdirectory endpoints under ``{BABEL_BASE}/{version}/``.
        pattern: Regex with one capture group selecting ``*.gz`` file paths.

    Returns:
        List of ``(lowercased_filename, absolute_url)`` tuples.
    """
    out: list[tuple[str, str]] = []
    for endpoint in endpoints:
        listing_url: str = f"{BABEL_BASE}/{version}/{endpoint}"
        request: Request = Request(listing_url, headers={"User-Agent": "tablassert"})
        with urlopen(request, timeout=60) as response:
            body: str = response.read().decode("utf-8")
        matches: list[str] = pattern.findall(body)
        for match in matches:
            filename: str = Path(match).name
            if any(filename.startswith(prefix) for prefix in BABEL_EXCLUDE_PREFIXES):
                continue
            out.append((filename.lower(), f"{listing_url}{match}"))
    return out


def download_babel_file(filename: str, url: str, destination: Path, retries: int = 5, on_progress: Callable[[int, int], None] | None = None) -> Path:
    """Spool a BABEL download to disk so large responses are resumable and never held in memory.

    Downloads to ``{filename}.part`` with HTTP Range resume support, then
    atomically renames to ``{filename}`` on success. Cached final files are
    reused without re-fetching.

    Args:
        filename: Output basename under ``destination``.
        url: Source URL.
        destination: Directory to download into (created if missing).
        retries: Maximum number of attempts before giving up.
        on_progress: Optional ``(downloaded_bytes, total_bytes)`` callback fired
            after each chunk; ``total_bytes`` is 0 when the size is unknown.
            ``None`` (default) keeps the original download behavior exactly.

    Returns:
        Path to the downloaded file.

    Raises:
        BabelDownloadError: If every retry attempt fails.
    """
    destination.mkdir(parents=True, exist_ok=True)
    final_path: Path = destination / filename
    part_path: Path = destination / f"{filename}.part"
    if final_path.is_file():
        download_logger.info("Reusing cached BABEL file: {path}", path=final_path)
        return final_path

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        offset: int = part_path.stat().st_size if part_path.exists() else 0
        headers: dict[str, str] = {"User-Agent": "tablassert"}
        if offset > 0:
            headers["Range"] = f"bytes={offset}-"
        request: Request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=300) as response:
                status: int = response.getcode()
                mode: str = "ab" if offset > 0 and status == 206 else "wb"
                if offset > 0 and status != 206:
                    download_logger.warning("Server ignored Range header (HTTP {status}); restarting download: {url}", status=status, url=url)
                # Resume base: bytes already on disk count only when appending (HTTP 206).
                base: int = offset if mode == "ab" else 0
                content_length: str | None = response.headers.get("Content-Length")
                total: int = base + int(content_length) if content_length is not None else 0
                reporter: Callable[[int], None] | None = None if on_progress is None else _byte_reporter(on_progress, base, total)
                with part_path.open(mode) as handle:
                    stream_copy(response, handle, reporter)
            part_path.replace(final_path)
            download_logger.info("Downloaded {url} -> {path}", url=url, path=final_path)
            return final_path
        except (HTTPError, OSError, URLError) as e:
            # HTTPError subclasses URLError, so it is matched in this single clause and
            # checked first for the fail-fast case. Non-retryable 4xx (e.g. a 404 from a
            # mistyped --version) fail fast instead of burning every attempt; 408/429 are
            # transient and fall through to backoff like 5xx and other network errors.
            if isinstance(e, HTTPError) and e.code not in (408, 429) and 400 <= e.code < 500:
                raise BabelDownloadError(url, attempt, e) from e
            last_error = e
            download_logger.warning(
                "Download attempt {attempt}/{retries} failed for {url}: {error}", attempt=attempt, retries=retries, url=url, error=e
            )
            # Exponential backoff (5, 10, 20, ... capped at 60s) only when another attempt
            # remains — no dead sleep after the final failed attempt before raising.
            if attempt < retries:
                time.sleep(min(60, 5 * 2 ** (attempt - 1)))
    raise BabelDownloadError(url, retries, last_error or RuntimeError("no attempts made")) from last_error


def stream_copy(source: BinaryIO, destination: BinaryIO, on_bytes: Callable[[int], None] | None = None) -> None:
    """Copy ``source`` to ``destination`` in 1 MiB chunks.

    When ``on_bytes`` is given it is called after each write with the running
    total of bytes written by THIS call; ``None`` (default) keeps the original
    copy-only behavior exactly.
    """
    written: int = 0
    while True:
        chunk: bytes = source.read(1024 * 1024)
        if not chunk:
            return
        destination.write(chunk)
        written += len(chunk)
        if on_bytes is not None:
            on_bytes(written)


def _byte_reporter(on_progress: Callable[[int, int], None], base: int, total: int) -> Callable[[int], None]:
    """Adapt ``stream_copy``'s cumulative-bytes callback to ``on_progress(downloaded, total)``.

    ``base`` is the byte count already on disk (resume offset) so the reported
    ``downloaded`` value reflects the whole file, not just this call's chunks.
    """

    def report(bytes_this_call: int) -> None:
        on_progress(base + bytes_this_call, total)

    return report


def _download_detail(downloaded: int, total: int) -> str:
    """Render the live download detail line in megabytes (1 MB = 1_000_000 bytes).

    When ``total`` is unknown (``<= 0``) only the transferred amount is shown.
    """
    if total <= 0:
        return f"{downloaded / 1_000_000:.1f} MB"
    return f"{downloaded / 1_000_000:.1f}/{total / 1_000_000:.1f} MB"


@APP.command(name="build-kg")
def build_kg(
    graph_configuration_file: Annotated[Path, cyclopts.Parameter(name=["--configuration-file", "-f"])],
    release: Annotated[bool, cyclopts.Parameter(name=["--release", "-r"], negative="")] = False,
    qc: Annotated[bool, cyclopts.Parameter(name=["--qc", "-q"], negative="")] = False,
    log: Annotated[bool, cyclopts.Parameter(name=["--log", "-l"], negative="")] = False,
    head: Annotated[bool, cyclopts.Parameter(name=["--head", "-hd"], negative="")] = False,
) -> None:
    """Build a knowledge graph from a YAML configuration file.

    The positional config is a Graph YAML that orchestrates one or more table
    configs into a single knowledge-graph build.
    """
    run(6, build_pipeline, graph_configuration_file, release=release, qc=qc, log=log, head=head)


@APP.command(name="validate")
def validate(
    configuration_file: Annotated[Path, cyclopts.Parameter(name=["--configuration-file", "-f"])],
    schema: Annotated[Literal["graph", "table"], cyclopts.Parameter(name=["--schema", "-s"])],
) -> None:
    """Validate a YAML configuration file against the graph or table config schema.

    ``--schema graph`` validates the Graph model AND every referenced table; ``--schema
    table`` validates section syntax only. The schema is selected explicitly rather than
    sniffed from the YAML, so a config is always checked against the schema you expected.
    """
    if schema == "graph":
        run(2, validate_graph_pipeline, configuration_file)
    else:
        run(3, validate_pipeline, configuration_file)


@APP.command(name="validate-kgx")
def validate_kgx_command(
    nodes: Annotated[Path, cyclopts.Parameter(name=["--nodes", "-n"])],
    edges: Annotated[Path, cyclopts.Parameter(name=["--edges", "-e"])],
    limit: Annotated[int, cyclopts.Parameter(name=["--limit"])] = 20,
) -> None:
    """Validate built KGX NDJSON against the Biolink Model.

    Constructs every node and edge as the Biolink Pydantic class named by its own
    ``category`` -- the same classes ``NCATSTranslator/translator-ingests`` builds --
    and reports failures grouped by field and error type. Exits non-zero when any
    record fails, so a build can be gated in CI.
    """
    from tablassert.biolink import validate_kgx

    report: dict[str, Any] = validate_kgx(nodes, edges, limit=limit)
    print(f"biolink-model {report['biolink_version']}", file=sys.stderr)
    for label in ("nodes", "edges"):
        section: dict[str, Any] = report[label]
        print(f"{label}: {section['valid']}/{section['total']} valid ({section['failures']} failures)", file=sys.stderr)
        for problem, count in section["problems"].items():
            print(f"  {count:>9}  {problem}", file=sys.stderr)
        for example in section["examples"][:3]:
            print(f"  e.g. {example['id']}: {', '.join(example['errors'])}", file=sys.stderr)
        logger.info(f"validate-kgx {label}: {section['valid']}/{section['total']} valid")
    if not report["ok"]:
        print("KGX output is not Biolink-compliant.", file=sys.stderr)
        raise SystemExit(1)
    print("KGX output is Biolink-compliant.", file=sys.stderr)


@APP.command(name="agent")
def agent(
    pmc_ids: Annotated[list[str], cyclopts.Parameter(allow_leading_hyphen=False)],
    *,
    fullmap: Annotated[Path, cyclopts.Parameter(name=["--fullmap", "-f"])],
    model_id: Annotated[str | None, cyclopts.Parameter(name=["--model-id", "-m"])] = None,
    api_base: Annotated[str | None, cyclopts.Parameter(name=["--api-base", "-ab"])] = None,
    api_key: Annotated[str | None, cyclopts.Parameter(name=["--api-key", "-ak"])] = None,
    max_steps: Annotated[int, cyclopts.Parameter(name=["--max-steps", "-ms"])] = 20,
    map_threshold: Annotated[float, cyclopts.Parameter(name=["--map-threshold", "-mt"])] = 0.25,
    max_improve_iters: Annotated[int, cyclopts.Parameter(name=["--max-improve-iters", "-mi"])] = 3,
    state_dir: Annotated[Path, cyclopts.Parameter(name=["--state-dir", "-sd"])] = Path(".tablassert") / "agent",
    backend: Annotated[Literal["openai", "litellm"], cyclopts.Parameter(name=["--backend", "-b"])] = "openai",
    reflexion: Annotated[bool, cyclopts.Parameter(name=["--reflexion"], negative="")] = False,
    judge_model: Annotated[str | None, cyclopts.Parameter(name=["--judge-model"])] = None,
    judge_threshold: Annotated[float | None, cyclopts.Parameter(name=["--judge-threshold"])] = None,
    local: Annotated[list[str] | None, cyclopts.Parameter(name=["--local", "-l"])] = None,
    optimize: Annotated[bool, cyclopts.Parameter(name=["--optimize", "-o"], negative="")] = False,
    instructions_file: Annotated[Path | None, cyclopts.Parameter(name=["--instructions-file"])] = None,
    instructions_out: Annotated[Path | None, cyclopts.Parameter(name=["--instructions-out"])] = None,
    max_metric_calls: Annotated[int, cyclopts.Parameter(name=["--max-metric-calls"])] = 8,
    dataset: Annotated[Path | None, cyclopts.Parameter(name=["--dataset"])] = None,
    task_model: Annotated[str | None, cyclopts.Parameter(name=["--task-model"])] = None,
    gepa_threads: Annotated[int | None, cyclopts.Parameter(name=["--gepa-threads"])] = None,
) -> None:
    """Autonomously derive, build, audit, and improve KG configs from PMC articles.

    Takes one or more PMC ids POSITIONALLY (``tablassert agent PMC11708054 [PMC...]``) and runs the
    deterministic supervisor over them. For each article the loop is: fetch the open-access
    supplementary tables -> an inner LLM agent derives a schema-gated Section config -> build_and_audit
    scores it -> a deterministic improve loop proposes/accepts edits IFF strictly better -> the config is
    accepted when coverage reaches ``--map-threshold`` or SKIPPED when the improve budget is exhausted.
    State checkpoints to ``--state-dir`` so an interrupted batch resumes, skipping finished articles.

    Model config comes from ``--model-id``/``--api-base``/``--api-key`` OR the ``TABLASSERT_AGENT_MODEL_ID``
    / ``TABLASSERT_AGENT_API_BASE`` / ``TABLASSERT_AGENT_API_KEY`` environment variables (explicit flags win).
    Secrets are NEVER hardcoded or defaulted: a missing value fails loud (exit 2) BEFORE any model is built.
    Requires the ``[agent]`` extra (``pip install tablassert[agent]``).

    Args:
        pmc_ids: One or more PMC article ids (positional).
        fullmap: Fullmap redb file or base directory (required).
        model_id: Model id (falls back to ``TABLASSERT_AGENT_MODEL_ID``).
        api_base: API base URL (falls back to ``TABLASSERT_AGENT_API_BASE``).
        api_key: API key (falls back to ``TABLASSERT_AGENT_API_KEY``).
        max_steps: Max inner-agent steps per article.
        map_threshold: Coverage an article must reach to be MAPPED.
        max_improve_iters: Max deterministic improve iterations per article.
        state_dir: Checkpoint/resume directory.
        backend: Model backend (``openai`` or ``litellm``).
        reflexion: Enable the tier-2 LLM reflexion improver (uses the same model config) for edits that
            may change predicate/source when the deterministic proposer stalls.
        judge_model: Optional model id for the semantic judge gate (uses ``--api-base``/``--api-key``);
            when set, MAPPED additionally requires the judge score to clear ``--judge-threshold``.
        judge_threshold: Semantic judge normalized-score threshold for MAPPED (default 0.5 when unset).
        local: Use a local payload instead of fetching from PMC-AWS: a single DIR (applied to every id) or
            one or more ``PMCid=DIR`` mappings (per-article). Fails loud (exit 2) if a DIR does not exist.
        optimize: Run GEPA prompt optimization over the model config and persist optimized instructions
            (instead of running the supervisor); use ``--instructions-out`` to choose the output file.
        instructions_file: Load GEPA-optimized instructions (from a prior ``--optimize`` run) for this run.
        instructions_out: Where ``--optimize`` writes optimized instructions (default
            ``<state-dir>/optimized_instructions.yaml``).
        max_metric_calls: GEPA metric-call budget for ``--optimize``.
        dataset: Optional YAML/JSON list of ``{table_summary, coverage_feedback}`` examples for ``--optimize``.
            An example may also carry ``fullmap`` (a fullmap path used to score each proposed config with
            REAL coverage) and ``head`` (default true: score a fast 5-row preview; set false for full builds).
        task_model: Optional FAST model id for GEPA's many program evaluations (GEPA best practice: a cheap
            task LM + a strong reflection LM); ``--model-id`` is the strong reflection LM. Defaults to the
            reflection LM when unset.
        gepa_threads: Optional thread count for GEPA's evaluation pool. Parallelizes the candidate LM
            forward passes only; the coverage-scoring builds stay serialized on the process-wide
            ``_GEPA_BUILD_LOCK`` (``os.chdir`` is process-global), so more threads do not speed up builds.
    """
    from tablassert import agent as agent_mod

    resolved_id, resolved_base, resolved_key = agent_mod.resolve_model_config(model_id, api_base, api_key)
    # Fail loud on any missing secret BEFORE building a model (so this path never touches smolagents).
    checks: tuple[tuple[str | None, str, str, str], ...] = (
        (resolved_id, "model_id", "model-id", agent_mod.ENV_MODEL_ID),
        (resolved_base, "api_base", "api-base", agent_mod.ENV_API_BASE),
        (resolved_key, "api_key", "api-key", agent_mod.ENV_API_KEY),
    )
    for value, which, flag, env in checks:
        if not value:
            print(f"tablassert agent: missing {which}. Set --{flag} or the {env} environment variable. Never hardcode secrets.", file=sys.stderr)
            raise SystemExit(2)

    # A normalized-score threshold outside [0, 1] (or non-finite, e.g. nan/inf) silently changes the
    # semantic gate (-1 passes every score); fail loud BEFORE any model is built.
    if judge_threshold is not None and not 0 <= judge_threshold <= 1:
        print("tablassert agent: --judge-threshold must be a finite number between 0 and 1.", file=sys.stderr)
        raise SystemExit(2)

    # A non-positive thread count would only fail deep inside dspy/ThreadPoolExecutor AFTER the models are
    # built; fail loud up front, matching the --judge-threshold pattern.
    if gepa_threads is not None and gepa_threads < 1:
        print("tablassert agent: --gepa-threads must be a positive integer.", file=sys.stderr)
        raise SystemExit(2)

    def build_model_factory() -> object:
        return agent_mod.build_model(resolved_id, resolved_base, resolved_key, backend=backend)

    # Tier-2 reflexion (optional): a prompt-callable over the same model config, built lazily per call.
    reflexion_factory: Callable[[], object] | None = None
    if reflexion:

        def _make_reflexion() -> object:
            return agent_mod.make_prompt_callable(agent_mod.build_model(resolved_id, resolved_base, resolved_key, backend=backend))

        reflexion_factory = _make_reflexion

    # Semantic judge (optional): a prompt-callable over the judge model (same api_base/api_key).
    judge: object | None = None
    if judge_model is not None:
        judge = agent_mod.make_prompt_callable(agent_mod.build_model(judge_model, resolved_base, resolved_key, backend=backend))

    # Local payload (optional, W4): a DIR for all ids, or PMCid=DIR mappings; fail loud on a missing dir.
    def parse_local(specs: list[str] | None) -> dict[str, Path] | Path | None:
        if not specs:
            return None
        if len(specs) == 1 and "=" not in specs[0]:
            single: Path = Path(specs[0])
            if not single.is_dir():
                print(f"tablassert agent: --local directory does not exist: {single}", file=sys.stderr)
                raise SystemExit(2)
            return single
        mapping: dict[str, Path] = {}
        for spec in specs:
            if "=" not in spec:
                print(f"tablassert agent: --local expects DIR or PMCid=DIR, got {spec!r}", file=sys.stderr)
                raise SystemExit(2)
            pid, _, dirstr = spec.partition("=")
            pid = pid.strip()
            dirstr = dirstr.strip()
            if not pid or not dirstr:
                print(f"tablassert agent: --local expects PMCid=DIR, got {spec!r}", file=sys.stderr)
                raise SystemExit(2)
            per_dir: Path = Path(dirstr)
            if not per_dir.is_dir():
                print(f"tablassert agent: --local directory does not exist: {per_dir}", file=sys.stderr)
                raise SystemExit(2)
            mapping[pid] = per_dir
        return mapping

    local_payload: dict[str, Path] | Path | None = parse_local(local)

    # W6 optimization path: run GEPA over the model config and persist optimized instructions; do NOT run
    # the supervisor. The reflection LM is a real dspy.LM (deferred live path); offline tests monkeypatch
    # ``run_gepa``/``make_dspy_lm`` so no model/network fires.
    if optimize:
        # Resolve the output path to ABSOLUTE up front: GEPA's parallel metric builds chdir the process cwd
        # (os.chdir is process-global), so a relative --instructions-out must be anchored to the invocation
        # cwd here, not the cwd GEPA happens to leave behind when it returns.
        out_path: Path = (instructions_out if instructions_out is not None else (state_dir / "optimized_instructions.yaml")).resolve()
        reflection_lm: object = agent_mod.make_dspy_lm(resolved_id, resolved_base, resolved_key, backend=backend)
        # GEPA best practice: a FAST task LM for the many program evaluations + the strong model for the few
        # reflection steps. --task-model selects the task LM; it defaults to the reflection LM when unset.
        task_lm: object | None = (
            agent_mod.make_dspy_lm(task_model, resolved_base, resolved_key, backend=backend, temperature=agent_mod.GEPA_TASK_TEMPERATURE)
            if task_model
            else None
        )
        gepa_dataset: list[dict[str, object]] | None = agent_mod.load_gepa_dataset(dataset) if dataset is not None else None
        gepa_result: dict[str, object] = agent_mod.run_gepa(
            seed_instructions=agent_mod.INSTRUCTIONS,
            reflection_lm=reflection_lm,
            task_lm=task_lm,
            dataset=gepa_dataset,
            max_metric_calls=max_metric_calls,
            num_threads=gepa_threads,
        )
        # A failed GEPA compile falls back to the SEED instructions with stats["error"]; do NOT persist that
        # unoptimized prompt or report success -- fail loud with a non-zero status.
        gepa_stats: object = gepa_result.get("stats")
        gepa_error: object = gepa_stats.get("error") if isinstance(gepa_stats, dict) else None
        if gepa_error:
            print(f"tablassert agent: GEPA optimization failed: {gepa_error}", file=sys.stderr)
            raise SystemExit(1)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        opt_instructions: object = gepa_result.get("optimized_instructions", agent_mod.INSTRUCTIONS)
        opt_descriptions: object = gepa_result.get("optimized_descriptions")
        agent_mod.save_optimized_instructions(out_path, str(opt_instructions), opt_descriptions if isinstance(opt_descriptions, dict) else None)
        print(f"tablassert agent: optimized instructions -> {out_path}")
        return

    # Normal run: optionally load GEPA-optimized instructions (--instructions-file).
    run_instructions: str | None = agent_mod.load_optimized_instructions(instructions_file) if instructions_file is not None else None

    result: dict[str, object] = agent_mod.run_supervisor(
        list(pmc_ids),
        fullmap=fullmap,
        build_model_factory=build_model_factory,
        map_threshold=map_threshold,
        max_improve_iters=max_improve_iters,
        max_steps=max_steps,
        state_dir=state_dir,
        reflexion_model_factory=reflexion_factory,
        judge_model=judge,
        judge_threshold=judge_threshold,
        local=local_payload,
        instructions=run_instructions,
    )

    metrics_raw: object = result.get("metrics")
    metrics: dict[str, object] = metrics_raw if isinstance(metrics_raw, dict) else {}
    records_raw: object = result.get("records")
    records: dict[str, object] = records_raw if isinstance(records_raw, dict) else {}

    def metric(key: str, default: float) -> float:
        value: object = metrics.get(key, default)
        return float(value) if isinstance(value, (int, float)) else default

    mapped: int = int(metric("mapped", 0))
    skipped: int = int(metric("skipped", 0))
    mean_best: float = metric("mean_best_coverage", 0.0)
    total_tokens: int = int(metric("total_tokens", 0))
    total_steps: int = int(metric("total_steps", 0))
    print(
        f"tablassert agent: processed {len(records)} article(s) ({mapped} mapped, {skipped} skipped); "
        f"mean best coverage {mean_best:.3f}; {total_tokens} tokens over {total_steps} steps."
    )


@APP.command(name="rebuild-agent-graph")
def rebuild_agent_graph(
    state_dir: Annotated[Path, cyclopts.Parameter(name=["--state-dir", "-sd"])] = Path(".tablassert") / "agent",
    *,
    fullmap: Annotated[Path, cyclopts.Parameter(name=["--fullmap", "-f"])],
) -> None:
    """Rebuild the shared agent graph registry from the supervisor checkpoint.

    Reconstructs ``<state-dir>/graph.yaml`` from ``<state-dir>/state.json``: every MAPPED /
    BUILT_UNMEASURED record whose best config still exists on disk becomes a ``tables`` entry
    (sorted by pmc id); stale entries (deleted configs, non-registered statuses) are pruned.
    Parallel ``tablassert agent`` runs maintain the registry incrementally; this command
    reconstructs it deterministically. Concurrency-safe: the same exclusive ``graph.yaml.lock``
    flock + atomic write the agent registration uses.

    Args:
        state_dir: Agent state directory holding ``state.json`` + ``configs/``.
        fullmap: Fullmap redb file or base directory recorded in the registry (first-wins: an
            existing registry fullmap that differs is kept with a warning).
    """
    import yaml

    from tablassert.graph_registry import rebuild_graph

    graph_path: Path = rebuild_graph(state_dir, fullmap)
    data: object = yaml.safe_load(graph_path.read_text(encoding="utf-8"))
    tables: object = data.get("tables", []) if isinstance(data, dict) else []
    count: int = len(tables) if isinstance(tables, list) else 0
    print(f"tablassert rebuild-agent-graph: wrote {graph_path} with {count} table config(s).")


def build_fullmap_pipeline(
    output: Path, progress: PipelineProgress, cache: Path = Path("./fullmap/downloads"), version: str = BABEL_VERSION, threads: int | None = None
) -> None:
    """Build an embedded fullmap redb database from BABEL outputs.

    Runs the three-stage build pipeline: discover BABEL files → download
    BABEL files → build fullmap redb database.

    Args:
        output: Path to the output redb file.
        progress: Pipeline progress reporter.
        cache: Directory for downloaded BABEL files.
        version: BABEL version label.
        threads: Optional thread count forwarded to Rust.
    """
    from tablassert import rs

    # Stage 1/3: discover BABEL files.
    progress.stage("Discovering BABEL Files")
    start, advance, sub_step = progress.section_loop(2, "Discover")
    start("class endpoints")
    sub_step("fetching listings")
    class_urls: list[tuple[str, str]] = babel_urls(version, BABEL_CLASS_ENDPOINTS, BABEL_CLASS_RE)
    advance()
    start("synonym endpoints")
    sub_step("fetching listings")
    synonym_urls: list[tuple[str, str]] = babel_urls(version, BABEL_SYNONYM_ENDPOINTS, BABEL_SYNONYM_RE)
    advance()

    # Stage 2/3: download BABEL files.
    progress.stage("Downloading BABEL Files")
    total_files: int = len(class_urls) + len(synonym_urls)
    start, advance, sub_step = progress.section_loop(total_files, "Download")

    def report_progress(downloaded: int, total: int) -> None:
        sub_step(_download_detail(downloaded, total))

    class_files: list[Path] = []
    for filename, url in class_urls:
        start(filename)
        sub_step("downloading")
        class_files.append(download_babel_file(filename, url, cache / "classes", on_progress=report_progress))
        advance()
    synonym_files: list[Path] = []
    for filename, url in synonym_urls:
        start(filename)
        sub_step("downloading")
        synonym_files.append(download_babel_file(filename, url, cache / "synonyms", on_progress=report_progress))
        advance()

    # Stage 3/3: build fullmap database.
    progress.stage("Building Fullmap Database")
    # Rust drives per-phase progress (equivalents -> synonyms -> writing) via the
    # callback; the GIL is released during the build so the bar repaints live.
    on_progress = progress.dynamic_loop("Build")
    rs.build_fullmap_db(output, class_files, synonym_files, threads=threads, progress=on_progress)
    progress.end_section_task()

    logger.info(
        "Built fullmap v{version}: {classes} classes, {synonyms} synonyms -> {output}",
        version=version,
        classes=len(class_files),
        synonyms=len(synonym_files),
        output=output,
    )


@APP.command(name="build-fullmap")
def build_fullmap(
    output: Annotated[Path, cyclopts.Parameter(name=["--output", "-o"])] = Path("./fullmap/data/fullmap.redb"),
    cache: Annotated[Path, cyclopts.Parameter(name=["--cache", "-c"])] = Path("./fullmap/downloads"),
    version: Annotated[str, cyclopts.Parameter(name=["--version", "-v"])] = BABEL_VERSION,
    threads: Annotated[int | None, cyclopts.Parameter(name=["--threads", "-t"])] = None,
) -> None:
    """Build an embedded fullmap redb database from hardcoded BABEL outputs."""
    run(3, build_fullmap_pipeline, output, cache=cache, version=version, threads=threads)
