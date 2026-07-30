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


def _load_graph(configuration_file: Path, table_config: bool, fullmap: Path) -> Graph:
    """Load and validate the Graph config that drives a build.

    By default ``configuration_file`` is a Graph YAML loaded directly. With
    ``table_config=True`` it is instead a table (Section) YAML wrapped in a
    throwaway ``TEMP_KG`` graph so a single table config can be built or tested
    without authoring a full graph config; ``contributions`` and ``ui_explanation``
    then fall back to the Graph model defaults.

    Args:
        configuration_file: Graph YAML path, or a table YAML path when ``table_config``.
        table_config: When ``True``, wrap the table YAML in a throwaway ``TEMP_KG`` graph.
        fullmap: Fullmap path for the wrapped graph when ``table_config`` is ``True``.

    Returns:
        The validated Graph model.

    Raises:
        GraphValidationError: If the (possibly wrapped) graph fails Pydantic validation.
    """
    from tablassert.ingests import from_yaml
    from tablassert.models import Graph
    from tablassert.progress import flatten_pydantic_error

    raw: object
    if table_config:
        raw = {
            "name": "TEMP_KG",
            "version": "0.0.0",
            "description": "Temporary knowledge graph generated to test a table configuration",
            "tables": [configuration_file],
            "fullmap": fullmap,
        }
    else:
        raw = from_yaml(configuration_file)
    try:
        return Graph.model_validate(raw)
    except pydantic.ValidationError as e:
        raise GraphValidationError(configuration_file, flatten_pydantic_error(e)) from e


def build_pipeline(
    configuration_file: Path,
    progress: PipelineProgress,
    release: bool = False,
    qc: bool = False,
    log: bool = False,
    head: bool = False,
    table_config: bool = False,
    fullmap: Path = Path("./fullmap"),
) -> None:
    """Build a knowledge graph from a YAML configuration file.

    Runs the six-stage build pipeline: load tables → extract sections → build
    Tcodes → collect instructions → build subgraphs → compile graph.

    Args:
        configuration_file: Path to the graph YAML file (or a table YAML
            file when ``table_config`` is ``True``).
        progress: Pipeline progress reporter.
        release: When ``True``, emit release-mode artifacts.
        qc: When ``True``, run quality-control audits on each section.
        log: When ``True``, enable per-section verbose logging.
        head: When ``True``, preview a random sample of up to 5 rows per section (fast schema/shape check).
        table_config: When ``True``, treat ``configuration_file`` as a table
            (Section) YAML and wrap it in a throwaway ``TEMP_KG`` graph.
        fullmap: Fullmap path used to wrap a table config when ``table_config`` is ``True``.

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
    g: Graph = _load_graph(configuration_file, table_config, fullmap)
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
    configuration_file: Path,
    release: Annotated[bool, cyclopts.Parameter(name=["--release", "-r"], negative="")] = False,
    qc: Annotated[bool, cyclopts.Parameter(name=["--qc", "-q"], negative="")] = False,
    log: Annotated[bool, cyclopts.Parameter(name=["--log", "-l"], negative="")] = False,
    head: Annotated[bool, cyclopts.Parameter(name=["--head", "-hd"], negative="")] = False,
    table_config: Annotated[bool, cyclopts.Parameter(name=["--table-config", "-tc"], negative="")] = False,
    fullmap: Annotated[Path, cyclopts.Parameter(name=["--fullmap", "-f"])] = Path("./fullmap"),
) -> None:
    """Build a knowledge graph from a YAML configuration file.

    By default the positional config is a Graph YAML. With ``--table-config`` it is a
    table (Section) YAML wrapped in a throwaway ``TEMP_KG`` graph (``--fullmap`` sets
    the fullmap path) so a single table config can be built or tested without
    authoring a full graph config.
    """
    run(6, build_pipeline, configuration_file, release=release, qc=qc, log=log, head=head, table_config=table_config, fullmap=fullmap)


@APP.command(name="validate")
def validate(configuration_file: Annotated[Path, cyclopts.Parameter(name=["--configuration-file", "-f"])]) -> None:
    """Validate a graph or table YAML configuration file.

    Detects the config kind from the YAML: a mapping with a top-level ``tables`` key
    is a graph config (validates the Graph model AND every referenced table); anything
    else is treated as a table config (validates section syntax only).
    """
    from tablassert.ingests import from_yaml

    loaded: object = from_yaml(configuration_file)
    if isinstance(loaded, dict) and "tables" in loaded:
        run(2, validate_graph_pipeline, configuration_file)
    else:
        run(3, validate_pipeline, configuration_file)


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

    def build_model_factory() -> object:
        return agent_mod.build_model(resolved_id, resolved_base, resolved_key, backend=backend)

    result: dict[str, object] = agent_mod.run_supervisor(
        list(pmc_ids),
        fullmap=fullmap,
        build_model_factory=build_model_factory,
        map_threshold=map_threshold,
        max_improve_iters=max_improve_iters,
        max_steps=max_steps,
        state_dir=state_dir,
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
