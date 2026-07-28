from __future__ import annotations

import re
import time
from collections.abc import Callable
from importlib.metadata import version as get_version
from itertools import chain
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, BinaryIO, Literal
from urllib.error import URLError
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


def build_pipeline(
    graph_configuration_file: Path, progress: PipelineProgress, release: bool = False, qc: bool = False, log: bool = False, head: bool = False
) -> None:
    """Build a knowledge graph from a YAML configuration file.

    Runs the six-stage build pipeline: load tables → extract sections → build
    Tcodes → collect instructions → build subgraphs → compile graph.

    Args:
        graph_configuration_file: Path to the graph YAML file.
        progress: Pipeline progress reporter.
        release: When ``True``, emit release-mode artifacts.
        qc: When ``True``, run quality-control audits on each section.
        log: When ``True``, enable per-section verbose logging.
        head: When ``True``, preview only the first 5 rows per section (fast schema/shape check).

    Raises:
        GraphValidationError: If the graph YAML fails Pydantic validation.
        SectionValidationError: If any section fails Pydantic validation.
    """
    from tablassert.fullmap import fullmap_db_path
    from tablassert.ingests import from_yaml
    from tablassert.lib import Tcode, compile_graph, compile_subgraph
    from tablassert.models import Graph
    from tablassert.progress import flatten_pydantic_error, format_section_compact
    from tablassert.utils import STORE, mkhash

    # Stage 1/6: load tables.
    progress.stage("Loading Tables")
    r: object = from_yaml(graph_configuration_file)
    try:
        g: Graph = Graph.model_validate(r)
    except pydantic.ValidationError as e:
        raise GraphValidationError(graph_configuration_file, flatten_pydantic_error(e)) from e
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
            tcode.append(Tcode.model_validate({**s, "store": store, "log": log, "qc": qc, "release": release, "head": head, "name": g.name}))
        except pydantic.ValidationError as e:
            raise SectionValidationError(graph_configuration_file, h, flatten_pydantic_error(e)) from e
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
    compile_graph(subgraphs, g.name, g.version, g.description, g.contributions, g.ui_explanation, g.tables, on_phase=sub_step, on_subgraph=advance)

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
        except (OSError, URLError) as e:
            last_error = e
            download_logger.warning(
                "Download attempt {attempt}/{retries} failed for {url}: {error}", attempt=attempt, retries=retries, url=url, error=e
            )
            time.sleep(5)
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


@APP.command(name="build-graph")
def build_graph(
    graph_configuration_file: Path,
    release: Annotated[bool, cyclopts.Parameter(name=["--release", "-r"], negative="")] = False,
    qc: Annotated[bool, cyclopts.Parameter(name=["--qc", "-q"], negative="")] = False,
    log: Annotated[bool, cyclopts.Parameter(name=["--log", "-l"], negative="")] = False,
    head: Annotated[bool, cyclopts.Parameter(name=["--head"], negative="")] = False,
) -> None:
    """Build a knowledge graph from a YAML configuration file."""
    run(6, build_pipeline, graph_configuration_file, release=release, qc=qc, log=log, head=head)


@APP.command(name="validate-table")
def validate_table(table_configuration_file: Path) -> None:
    """Validate section syntax from a YAML configuration file."""
    run(3, validate_pipeline, table_configuration_file)


@APP.command(name="schema")
def schema(
    model: Annotated[Literal["graph", "section"], cyclopts.Parameter(name=["--model", "-m"])] = "section",
    output: Annotated[Path | None, cyclopts.Parameter(name=["--output", "-o"])] = None,
) -> None:
    """Emit the JSON schema for the Graph or Section config model.

    Prints an indented (indent=2) JSON schema for config authoring, to stdout or
    to a file. Uses deferred imports so the command never pulls in lib.py/Rust at
    import time and stays free of pipeline/network dependencies.

    Args:
        model: Which config model's schema to emit.
        output: Optional destination file; prints to stdout when ``None``.
    """
    import json

    from tablassert.models import Graph, Section

    chosen: type[Graph] | type[Section] = Graph if model == "graph" else Section
    text: str = json.dumps(chosen.model_json_schema(), indent=2)
    if output is not None:
        output.write_text(text + "\n")
    else:
        print(text)


def build_fullmap_pipeline(
    output: Path,
    progress: PipelineProgress,
    cache: Path = Path("./fullmap/downloads/fullmap"),
    version: str = BABEL_VERSION,
    threads: int | None = None,
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
    cache: Annotated[Path, cyclopts.Parameter(name=["--cache", "-c"])] = Path("./fullmap/downloads/fullmap"),
    version: Annotated[str, cyclopts.Parameter(name=["--version", "-v"])] = BABEL_VERSION,
    threads: Annotated[int | None, cyclopts.Parameter(name=["--threads", "-t"])] = None,
) -> None:
    """Build an embedded fullmap redb database from hardcoded BABEL outputs."""
    run(3, build_fullmap_pipeline, output, cache=cache, version=version, threads=threads)
