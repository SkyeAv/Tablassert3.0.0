"""Targeted coverage for previously-uncovered branches in ``src/tablassert/cli.py``.

Each test executes a specific uncovered line range (noted in its docstring) using the
established offline monkeypatch patterns from ``test_cli_progress.py`` /
``test_fullmap.py``: the HTTP seam (``cli.urlopen``) is faked with ``_FakeResponse``,
network discovery (``cli.babel_urls``) and the Rust build (``rs.build_fullmap_db``) are
stubbed, and all artifacts land in ``tmp_path``. No network, no real Rust build.
"""

from __future__ import annotations

import io
import subprocess
from email.message import Message
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

import pytest
from cyclopts.exceptions import UnknownOptionError  # pyright: ignore[reportMissingImports]

from tablassert import cli, rs
from tablassert.cli import build_fullmap_pipeline, build_kg, download_babel_file, download_babel_file_aria2c, validate_graph_pipeline
from tablassert.errors import BabelDownloadError, GraphValidationError
from tablassert.ingests import to_yaml
from tablassert.progress import PipelineProgress


class _FakeResponse:
    """Offline stand-in for ``urllib``'s context-managed HTTPResponse (mirrors test_cli_progress)."""

    def __init__(self, data: bytes, status: int, content_length: str | None) -> None:
        self._stream: io.BytesIO = io.BytesIO(data)
        self._status: int = status
        self.headers: dict[str, str] = {"Content-Length": content_length} if content_length is not None else {}

    def getcode(self) -> int:
        return self._status

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


def test_validate_graph_pipeline_rejects_malformed_graph(tmp_path: Path) -> None:
    """Cover cli.py:314-315 — ``validate_graph_pipeline`` raises ``GraphValidationError``.

    A graph YAML missing required keys (``tables``/``fullmap``) fails ``Graph.model_validate``,
    hitting the ``except pydantic.ValidationError`` -> ``raise GraphValidationError`` path that
    ``validate``'s graph branch delegates to. Existing tests only reach this via ``build_pipeline``
    (``_load_graph``) or via an invalid *table*; this drives the validate-graph path directly.
    """
    graph_file: Path = tmp_path / "graph.yaml"
    to_yaml(graph_file, {"name": "TEST", "version": "1.0.0", "description": "missing tables and fullmap"})
    with pytest.raises(GraphValidationError) as exc_info:
        validate_graph_pipeline(graph_file, PipelineProgress(total_stages=2))
    assert exc_info.value.code == "graph-validation-failed"


def test_download_babel_file_reuses_cached_final(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover cli.py:408-409 — a cached final file is reused without any HTTP fetch.

    When ``{filename}`` already exists, ``download_babel_file`` logs the reuse and returns
    immediately. ``urlopen`` is rigged to fail loudly so the test proves the network seam is
    never touched on the cache-hit path.
    """
    final: Path = tmp_path / "f.gz"
    final.write_bytes(b"cached-bytes")

    def _no_network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("urlopen must not be called when the final file is cached")

    monkeypatch.setattr(cli, "urlopen", _no_network)
    out: Path = download_babel_file("f.gz", "https://example.com/f.gz", tmp_path)
    assert out == final
    assert out.read_bytes() == b"cached-bytes"


def test_download_babel_file_restarts_when_server_ignores_range(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover cli.py:423 — server answers HTTP 200 (not 206) to a Range request => warn + restart.

    A non-empty ``.part`` makes ``offset > 0`` so a ``Range`` header is sent. The fake server
    ignores it (status 200), which fires the "Server ignored Range header" warning and forces
    ``mode="wb"``: the stale partial bytes are discarded, so the final file is exactly the fresh
    payload (NOT ``old + new``). The captured warning asserts the branch ran.
    """
    (tmp_path / "f.gz.part").write_bytes(b"OLD" * 10)  # offset > 0 => Range header sent
    payload: bytes = b"NEW" * 20
    monkeypatch.setattr(cli, "urlopen", lambda request, timeout: _FakeResponse(payload, 200, str(len(payload))))

    warnings: list[str] = []

    class _RecordingLogger:
        def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
            warnings.append(message)

        def info(self, *args: Any, **kwargs: Any) -> None:
            return None

    monkeypatch.setattr(cli, "download_logger", _RecordingLogger())

    out: Path = download_babel_file("f.gz", "https://example.com/f.gz", tmp_path)
    assert any("ignored Range header" in w for w in warnings)
    assert out.read_bytes() == payload  # restarted in "wb" mode: stale .part discarded, not appended


def test_download_babel_file_raises_after_exhausting_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every network attempt fails => per-attempt warning, exponential backoff, then raise.

    WHY: ``urlopen`` always raises ``URLError``; ``time.sleep`` is stubbed to record the
    backoffs (keeping the test instant). Backoff is exponential (``5 * 2 ** (attempt-1)``,
    capped at 60s) and only fires when another attempt remains, so with ``retries=3`` attempts
    1 and 2 sleep 5s and 10s and the FINAL attempt skips the dead sleep before the loop exits
    via ``raise BabelDownloadError`` carrying the last error.
    """
    sleeps: list[float] = []
    monkeypatch.setattr(cli.time, "sleep", sleeps.append)

    def _fail(request: Any, timeout: int) -> None:
        raise URLError("boom")

    monkeypatch.setattr(cli, "urlopen", _fail)
    with pytest.raises(BabelDownloadError):
        download_babel_file("f.gz", "https://example.com/f.gz", tmp_path, retries=3)
    assert sleeps == [5, 10]  # exponential backoff; no sleep after the final attempt


def test_download_babel_file_404_raises_immediately(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-retryable 4xx (404) raises ``BabelDownloadError`` on the FIRST attempt.

    WHY: a 404/403 (e.g. a mistyped ``--version``) can never succeed on retry; burning every
    attempt with backoff wastes ~25s. The ``HTTPError`` handler re-raises immediately for
    non-retryable 4xx (anything 400-499 except the transient 408/429), so ``urlopen`` is called
    exactly once and no backoff sleeps fire before the error surfaces.
    """
    sleeps: list[float] = []
    monkeypatch.setattr(cli.time, "sleep", sleeps.append)

    attempts: list[int] = []

    def _404(request: Any, timeout: int) -> None:
        attempts.append(1)
        raise HTTPError("https://example.com/f.gz", 404, "Not Found", Message(), None)

    monkeypatch.setattr(cli, "urlopen", _404)
    with pytest.raises(BabelDownloadError):
        download_babel_file("f.gz", "https://example.com/f.gz", tmp_path, retries=5)
    assert len(attempts) == 1  # failed fast on the first attempt, no retries
    assert sleeps == []  # no backoff before raising


def test_download_babel_file_retries_transient_http_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A retryable ``HTTPError`` (5xx, or 408/429) warns, backs off, and retries.

    WHY: only non-retryable 4xx fail fast; 5xx and the transient 408/429 codes fall through to
    the SAME exponential backoff as network errors. With ``retries=3`` the retryable HTTPError
    path sleeps 5s then 10s (``5*2**0``, ``5*2**1``) and skips the sleep after the final attempt
    before raising — proving the backoff guard fires in the HTTPError branch too.
    """
    sleeps: list[float] = []
    monkeypatch.setattr(cli.time, "sleep", sleeps.append)

    def _500(request: Any, timeout: int) -> None:
        raise HTTPError("https://example.com/f.gz", 500, "Internal Server Error", Message(), None)

    monkeypatch.setattr(cli, "urlopen", _500)
    with pytest.raises(BabelDownloadError):
        download_babel_file("f.gz", "https://example.com/f.gz", tmp_path, retries=3)
    assert sleeps == [5, 10]  # exponential backoff, skipped after the final attempt


def test_download_babel_file_zero_retries_raises_without_attempt(tmp_path: Path) -> None:
    """Cover the defensive ``or RuntimeError("no attempts made")`` operand on cli.py:440.

    With ``retries=0`` the attempt loop body never runs, so ``last_error`` stays ``None`` and the
    raise falls through to the ``RuntimeError`` fallback. A real, reachable edge (empty ``range``);
    no network or sleep is involved.
    """
    with pytest.raises(BabelDownloadError):
        download_babel_file("f.gz", "https://example.com/f.gz", tmp_path, retries=0)


def test_download_babel_file_aria2c_reuses_cached_complete_without_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A complete final file is reused before binary detection or subprocess execution."""
    final: Path = tmp_path / "f.gz"
    final.write_bytes(b"cached-bytes")

    def _resolve_must_not_run() -> str:
        raise AssertionError("aria2c resolver must not run for a complete cache hit")

    def _run_must_not_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise AssertionError("subprocess.run must not run for a complete cache hit")

    monkeypatch.setattr(cli, "_resolve_aria2_binary", _resolve_must_not_run)
    monkeypatch.setattr(cli.subprocess, "run", _run_must_not_run)
    out: Path = download_babel_file_aria2c("f.gz", "https://example.com/f.gz", tmp_path)
    assert out == final
    assert out.read_bytes() == b"cached-bytes"


def test_download_babel_file_aria2c_runs_resume_retry_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """aria2c helper uses subprocess without shell and passes resume/retry flags."""
    monkeypatch.setattr(cli, "_resolve_aria2_binary", lambda: "/usr/bin/aria2c")
    commands: list[list[str]] = []

    def _fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        assert kwargs["shell"] is False
        destination = Path(command[command.index("--dir") + 1])
        filename = command[command.index("--out") + 1]
        destination.mkdir(parents=True, exist_ok=True)
        (destination / filename).write_bytes(b"downloaded")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)
    out: Path = download_babel_file_aria2c("f.gz", "https://example.com/f.gz", tmp_path, retries=7)
    assert out == tmp_path / "f.gz"
    assert out.read_bytes() == b"downloaded"
    assert len(commands) == 1
    command = commands[0]
    assert command[0] == "/usr/bin/aria2c"
    assert "--continue=true" in command
    assert "--max-tries" in command
    assert command[command.index("--max-tries") + 1] == "7"
    assert "--retry-wait" in command
    assert command[command.index("--retry-wait") + 1] == "5"
    assert "--summary-interval=0" in command
    assert "--show-console-readout=false" in command
    assert command[-1] == "https://example.com/f.gz"


def test_resolve_aria2_binary_returns_bundled_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The resolver returns the bundled ``aria2c.ARIA2C`` path as a string."""

    class FakeAria2c:
        ARIA2C = Path("/tmp/aria2c")

    monkeypatch.setattr(cli, "import_module", lambda name: FakeAria2c())
    assert cli._resolve_aria2_binary() == "/tmp/aria2c"


def test_resolve_aria2_binary_missing_export_raises_importerror(monkeypatch: pytest.MonkeyPatch) -> None:
    """A shadowed ``aria2c`` module without ``ARIA2C`` still hits the clean missing-extra path."""

    class BrokenAria2c:
        pass

    monkeypatch.setattr(cli, "import_module", lambda name: BrokenAria2c())
    with pytest.raises(ImportError, match=r"aria2c\.ARIA2C"):
        cli._resolve_aria2_binary()


def test_download_babel_file_aria2c_missing_extra_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Opting into aria2c fails loud when the ``[aria2]`` extra is unavailable."""

    def _missing_extra() -> str:
        raise ImportError("No module named 'aria2c'")

    def _run_must_not_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise AssertionError("subprocess.run must not run when the [aria2] extra is missing")

    monkeypatch.setattr(cli, "_resolve_aria2_binary", _missing_extra)
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setattr(cli.subprocess, "run", _run_must_not_run)
    with pytest.raises(BabelDownloadError) as excinfo:
        download_babel_file_aria2c("f.gz", "https://example.com/f.gz", tmp_path)
    assert "pip install tablassert[aria2]" in str(excinfo.value)


def test_download_babel_file_aria2c_missing_extra_on_macos_raises_platform_hint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The unsupported macOS path explains that bundled aria2c wheels are unavailable."""

    def _missing_extra() -> str:
        raise ImportError("No module named 'aria2c'")

    def _run_must_not_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise AssertionError("subprocess.run must not run when the [aria2] extra is unsupported")

    monkeypatch.setattr(cli, "_resolve_aria2_binary", _missing_extra)
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    monkeypatch.setattr(cli.subprocess, "run", _run_must_not_run)
    with pytest.raises(BabelDownloadError) as excinfo:
        download_babel_file_aria2c("f.gz", "https://example.com/f.gz", tmp_path)
    message = str(excinfo.value)
    assert "no macOS wheels" in message
    assert "drop --aria2c" in message


def test_download_babel_file_aria2c_zero_retries_raises_without_unlimited_aria2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``retries=0`` is rejected before aria2 can interpret it as unlimited retries."""

    def _resolve_must_not_run() -> str:
        raise AssertionError("aria2c resolver must not run when retries is invalid")

    monkeypatch.setattr(cli, "_resolve_aria2_binary", _resolve_must_not_run)
    with pytest.raises(BabelDownloadError):
        download_babel_file_aria2c("f.gz", "https://example.com/f.gz", tmp_path, retries=0)


def test_download_babel_file_aria2c_subprocess_oserror_raises_typed_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """OS errors from launching aria2c surface as ``BabelDownloadError``."""
    monkeypatch.setattr(cli, "_resolve_aria2_binary", lambda: "/usr/bin/aria2c")

    def _raise_oserror(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise OSError("exec failed")

    monkeypatch.setattr(cli.subprocess, "run", _raise_oserror)
    with pytest.raises(BabelDownloadError):
        download_babel_file_aria2c("f.gz", "https://example.com/f.gz", tmp_path)


def test_download_babel_file_aria2c_success_without_complete_file_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Even an exit-0 aria2c run must leave a final file without a resume control file."""
    monkeypatch.setattr(cli, "_resolve_aria2_binary", lambda: "/usr/bin/aria2c")

    def _fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        destination = Path(command[command.index("--dir") + 1])
        filename = command[command.index("--out") + 1]
        destination.mkdir(parents=True, exist_ok=True)
        (destination / filename).write_bytes(b"partial")
        (destination / f"{filename}.aria2").write_bytes(b"resume-state")
        return subprocess.CompletedProcess(command, 0, stdout="done", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)
    with pytest.raises(BabelDownloadError):
        download_babel_file_aria2c("f.gz", "https://example.com/f.gz", tmp_path)


def test_download_babel_file_aria2c_preserves_control_file_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An existing aria2 control file means the target is partial and must be resumed, not reused."""
    final: Path = tmp_path / "f.gz"
    control: Path = tmp_path / "f.gz.aria2"
    final.write_bytes(b"partial")
    control.write_bytes(b"resume-state")
    monkeypatch.setattr(cli, "_resolve_aria2_binary", lambda: "/usr/bin/aria2c")
    commands: list[list[str]] = []

    def _fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="dropped connection")

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)
    with pytest.raises(BabelDownloadError):
        download_babel_file_aria2c("f.gz", "https://example.com/f.gz", tmp_path)
    assert len(commands) == 1  # final + .aria2 was NOT treated as a complete cache hit
    assert final.read_bytes() == b"partial"
    assert control.read_bytes() == b"resume-state"  # failure path preserves aria2 resume metadata


def test_build_kg_command_delegates_to_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover cli.py:501 — the ``build-kg`` cyclopts command forwards to ``run(6, build_pipeline, ...)``.

    ``cli.run`` is stubbed to a recorder so the command body executes (line 501) without a real
    multi-hour build. Asserts the stage count, pipeline function, config path, and every flag are
    threaded through unchanged.
    """
    config: Path = tmp_path / "graph.yaml"
    calls: list[tuple[Any, ...]] = []

    def _fake_run(stages: int, fn: Any, arg: Path, **kwargs: Any) -> None:
        calls.append((stages, fn, arg, kwargs))

    monkeypatch.setattr(cli, "run", _fake_run)
    build_kg(config, release=True, qc=True, log=True, head=True)
    assert calls == [(6, cli.build_pipeline, config, {"release": True, "qc": True, "log": True, "head": True})]


def test_build_fullmap_command_passes_aria2c_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``build-fullmap --aria2c`` delegates the opt-in flag to ``build_fullmap_pipeline``."""
    output: Path = tmp_path / "fullmap.redb"
    cache: Path = tmp_path / "downloads"
    calls: list[tuple[Any, ...]] = []

    def _fake_run(stages: int, fn: Any, arg: Path, **kwargs: Any) -> None:
        calls.append((stages, fn, arg, kwargs))

    monkeypatch.setattr(cli, "run", _fake_run)
    cli.build_fullmap(output=output, cache=cache, version="v", threads=2, aria2c=True)
    assert calls == [(3, cli.build_fullmap_pipeline, output, {"cache": cache, "version": "v", "threads": 2, "aria2c": True})]


def test_build_fullmap_aria2c_flag_parses() -> None:
    """``build-fullmap`` accepts ``--aria2c`` and ``-a`` but no generated negative alias."""

    def parse(argv: list[str]) -> dict[str, Any]:
        fn, bound, _ = cli.APP.parse_args(argv, exit_on_error=False)
        assert fn is cli.build_fullmap
        return dict(bound.arguments)

    assert parse(["build-fullmap"]) == {}
    assert parse(["build-fullmap", "--aria2c"])["aria2c"] is True
    assert parse(["build-fullmap", "-a"])["aria2c"] is True
    with pytest.raises(UnknownOptionError):
        parse(["build-fullmap", "--no-aria2c"])


def test_build_kg_configuration_file_flag_parses(tmp_path: Path) -> None:
    """Guard: ``build-kg``'s config binds positionally AND via ``-f``/``--configuration-file``.

    ``build-kg`` previously exposed its configuration file ONLY positionally, while ``validate``
    accepted ``--configuration-file``/``-f``. This pins the now-consistent parsing (``-f`` belongs
    to the configuration file). The bound argument is keyed by the ``graph_configuration_file``
    parameter name. It also locks the removal of the deprecated ``--table-config``/``-tc`` and
    ``--fullmap`` options (parsing them now raises). cyclopts' ``parse_args`` binds tokens WITHOUT
    executing the command, so no build runs.
    """
    config: Path = tmp_path / "graph.yaml"

    def parse(argv: list[str]) -> dict[str, Any]:
        fn, bound, _ = cli.APP.parse_args(argv, exit_on_error=False)
        assert fn is build_kg
        return dict(bound.arguments)

    # Positional usage is unchanged.
    assert parse(["build-kg", str(config)])["graph_configuration_file"] == config
    # The configuration file now also binds via -f and --configuration-file (matches validate).
    assert parse(["build-kg", "-f", str(config)])["graph_configuration_file"] == config
    assert parse(["build-kg", "--configuration-file", str(config)])["graph_configuration_file"] == config
    # The removed --table-config/-tc and --fullmap options are now rejected (locks the removal).
    for removed in (["--table-config"], ["-tc"], ["--fullmap", str(config)]):
        with pytest.raises(UnknownOptionError):
            parse(["build-kg", str(config), *removed])


def test_build_fullmap_pipeline_reports_download_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover cli.py:653 — ``report_progress`` renders ``_download_detail`` into the sub-step.

    Drives the REAL ``build_fullmap_pipeline`` download stage: ``babel_urls`` is stubbed to one
    class + one synonym file, ``cli.urlopen`` serves a small payload, and the REAL
    ``download_babel_file`` fires its ``on_progress`` (= the ``report_progress`` closure) per
    chunk, executing line 653. ``_download_detail`` is wrapped (delegating to the real impl) to
    observe the ``(downloaded, total)`` values; ``rs.build_fullmap_db`` is stubbed so Stage 3 needs
    no Rust. The downloaded files and the detail calls are both asserted.
    """
    payload: bytes = b"q" * 100
    monkeypatch.setattr(cli, "urlopen", lambda request, timeout: _FakeResponse(payload, 200, str(len(payload))))

    def _fake_babel_urls(version: str, endpoints: tuple[str, ...], pattern: object) -> list[tuple[str, str]]:
        if endpoints == cli.BABEL_CLASS_ENDPOINTS:
            return [("c.gz", "https://example.com/c.gz")]
        return [("s.gz", "https://example.com/s.gz")]

    monkeypatch.setattr(cli, "babel_urls", _fake_babel_urls)

    detail_calls: list[tuple[int, int]] = []
    real_detail = cli._download_detail

    def _recording_detail(downloaded: int, total: int) -> str:
        detail_calls.append((downloaded, total))
        return real_detail(downloaded, total)

    monkeypatch.setattr(cli, "_download_detail", _recording_detail)

    built: list[tuple[Any, ...]] = []

    def _fake_build(output: Path, class_files: list[Path], synonym_files: list[Path], threads: int | None = None, progress: Any = None) -> None:
        built.append((output, class_files, synonym_files, threads))

    monkeypatch.setattr(rs, "build_fullmap_db", _fake_build)

    output: Path = tmp_path / "fullmap.redb"
    cache: Path = tmp_path / "downloads"
    build_fullmap_pipeline(output, PipelineProgress(total_stages=3), cache=cache, version="v", threads=1)

    # Line 653 fired for each downloaded file (one chunk each), ending at the full payload size.
    assert detail_calls == [(len(payload), len(payload)), (len(payload), len(payload))]
    # The real downloader spooled both files to their class/synonym cache dirs.
    assert (cache / "classes" / "c.gz").read_bytes() == payload
    assert (cache / "synonyms" / "s.gz").read_bytes() == payload
    # Stage 3 received the downloaded paths and the thread count.
    assert built == [(output, [cache / "classes" / "c.gz"], [cache / "synonyms" / "s.gz"], 1)]


def test_build_fullmap_pipeline_uses_aria2c_when_opted_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The aria2c flag switches both class and synonym loops to the aria2 helper and simple progress text."""

    def _fake_babel_urls(version: str, endpoints: tuple[str, ...], pattern: object) -> list[tuple[str, str]]:
        if endpoints == cli.BABEL_CLASS_ENDPOINTS:
            return [("c.gz", "https://example.com/c.gz")]
        return [("s.gz", "https://example.com/s.gz")]

    monkeypatch.setattr(cli, "babel_urls", _fake_babel_urls)

    def _python_download_must_not_run(*args: Any, **kwargs: Any) -> Path:
        raise AssertionError("Python downloader must not run when aria2c=True")

    monkeypatch.setattr(cli, "download_babel_file", _python_download_must_not_run)
    aria_calls: list[tuple[str, str, Path]] = []

    def _fake_aria2c(filename: str, url: str, destination: Path, retries: int = 5) -> Path:
        aria_calls.append((filename, url, destination))
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / filename
        path.write_bytes(b"downloaded")
        return path

    monkeypatch.setattr(cli, "download_babel_file_aria2c", _fake_aria2c)
    built: list[tuple[Any, ...]] = []

    def _fake_build(output: Path, class_files: list[Path], synonym_files: list[Path], threads: int | None = None, progress: Any = None) -> None:
        built.append((output, class_files, synonym_files, threads))

    monkeypatch.setattr(rs, "build_fullmap_db", _fake_build)

    class _RecordingProgress:
        def __init__(self) -> None:
            self.sub_steps: list[str] = []
            self.advances: int = 0

        def stage(self, name: str) -> None:
            pass

        def section_loop(self, total: int, label: str) -> tuple[Any, Any, Any]:
            def start(detail: str) -> None:
                pass

            def advance() -> None:
                self.advances += 1

            def sub_step(phase: str) -> None:
                self.sub_steps.append(phase)

            return start, advance, sub_step

        def dynamic_loop(self, label: str) -> Any:
            return lambda *args: None

        def end_section_task(self) -> None:
            pass

    progress = _RecordingProgress()
    output: Path = tmp_path / "fullmap.redb"
    cache: Path = tmp_path / "downloads"
    build_fullmap_pipeline(output, progress, cache=cache, version="v", threads=1, aria2c=True)  # type: ignore[arg-type]

    assert aria_calls == [("c.gz", "https://example.com/c.gz", cache / "classes"), ("s.gz", "https://example.com/s.gz", cache / "synonyms")]
    assert progress.sub_steps.count("aria2c downloading") == 2
    assert progress.advances == 4  # two discovery entries + two downloaded files
    assert built == [(output, [cache / "classes" / "c.gz"], [cache / "synonyms" / "s.gz"], 1)]
