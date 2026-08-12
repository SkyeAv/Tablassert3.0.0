"""Targeted coverage for previously-uncovered branches in ``src/tablassert/cli.py``.

Each test executes a specific uncovered line range (noted in its docstring) using the
established offline monkeypatch patterns from ``test_cli_progress.py`` /
``test_fullmap.py``: the HTTP seam (``cli.urlopen``) is faked with ``_FakeResponse``,
network discovery (``cli.babel_urls``) and the Rust build (``rs.build_fullmap_db``) are
stubbed, and all artifacts land in ``tmp_path``. No network, no real Rust build.
"""

from __future__ import annotations

import hashlib
import io
import shutil
import subprocess
import sys
import tarfile
from email.message import Message
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

import pytest
from cyclopts.exceptions import UnknownOptionError  # pyright: ignore[reportMissingImports]

from tablassert import cli, extras, rs
from tablassert.cli import build_fullmap_pipeline, build_kg, download_babel_file, download_babel_file_aria2c, validate_graph_pipeline
from tablassert.errors import BabelDownloadError, GraphValidationError, QcRuntimeMissingError
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
    # Quoted: unquoted brackets glob in zsh, so the command as printed must be runnable as-is.
    assert 'pip install "tablassert[aria2]"' in str(excinfo.value)


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
    monkeypatch.setattr(extras, "missing", lambda extra: ())
    build_kg(config, release=True, qc=True, log=True, head=True)
    assert calls == [(6, cli.build_pipeline, config, {"release": True, "qc": True, "log": True, "head": True})]


def test_build_kg_qc_without_the_extra_stops_before_the_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--qc`` without the ``[qc]`` extra fails immediately, naming the install command.

    Why this matters more than any other extras check: the QC audit is the LAST stage of the
    pipeline, running only after entity resolution has finished. Without the preflight a user
    waits out the whole build before learning scikit-learn was never installed — so the test
    asserts ``run`` was never reached, not merely that an error was raised.
    """
    config: Path = tmp_path / "graph.yaml"
    monkeypatch.setattr(cli, "run", lambda *args, **kwargs: pytest.fail("the build started without the [qc] extra"))
    monkeypatch.setattr(extras, "missing", lambda extra: ("scikit-learn", "sentence-transformers"))

    with pytest.raises(QcRuntimeMissingError) as excinfo:
        build_kg(config, qc=True)

    message: str = str(excinfo.value)
    assert "scikit-learn" in message
    assert 'pip install "tablassert[qc]"' in message


def test_build_kg_without_qc_never_probes_the_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A plain ``build-kg`` runs on a base install, extra absent or not.

    Why: QC is opt-in. Gating every build on an extra nobody asked for would break the
    base install this project promises.
    """
    config: Path = tmp_path / "graph.yaml"
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(cli, "run", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr(extras, "missing", lambda extra: pytest.fail("probed an extra for a build that never runs QC"))

    build_kg(config)
    assert len(calls) == 1


def test_build_fullmap_command_force_build_passes_aria2c_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``build-fullmap --force --aria2c`` delegates both flags to the from-scratch build.

    With the download-first default, only ``--force`` reaches ``build_fullmap_pipeline``; the
    default path is covered by :func:`test_build_fullmap_command_defaults_to_prebuilt_download`.
    """
    output: Path = tmp_path / "fullmap.redb"
    cache: Path = tmp_path / "downloads"
    calls: list[tuple[Any, ...]] = []

    def _fake_run(stages: int, fn: Any, arg: Path, **kwargs: Any) -> None:
        calls.append((stages, fn, arg, kwargs))

    monkeypatch.setattr(cli, "run", _fake_run)
    monkeypatch.setattr(extras, "missing", lambda extra: ())
    cli.build_fullmap(output=output, cache=cache, version="v", threads=2, aria2c=True, force=True)
    assert calls == [(3, cli.build_fullmap_pipeline, output, {"cache": cache, "version": "v", "threads": 2, "aria2c": True})]


def test_build_fullmap_aria2c_without_the_extra_stops_before_downloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--aria2c`` without the ``[aria2]`` extra exits 2 before any download starts.

    Why: the flag is otherwise resolved on the FIRST download, after BABEL URL discovery has
    already hit the network. A flag that cannot work should cost nothing.
    """
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setattr(cli, "run", lambda *args, **kwargs: pytest.fail("the download started without the [aria2] extra"))
    monkeypatch.setattr(extras, "missing", lambda extra: ("aria2",))

    with pytest.raises(SystemExit) as excinfo:
        cli.build_fullmap(output=tmp_path / "fullmap.redb", aria2c=True)

    assert excinfo.value.code == 2
    assert 'pip install "tablassert[aria2]"' in capsys.readouterr().err


def test_build_fullmap_aria2c_is_not_checked_when_the_db_already_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An existing DB still short-circuits, even with ``--aria2c`` and no ``[aria2]`` extra.

    Why: that path downloads nothing, so the flag is moot. Failing a command that was going to
    be a no-op would turn a harmless leftover flag into an error.
    """
    output: Path = tmp_path / "fullmap.redb"
    output.write_bytes(b"existing-db")
    monkeypatch.setattr(cli, "run", lambda *args, **kwargs: pytest.fail("an existing DB must short-circuit"))
    monkeypatch.setattr(extras, "missing", lambda extra: ("aria2",))

    cli.build_fullmap(output=output, aria2c=True)  # must not raise
    assert "already present" in capsys.readouterr().err


def test_build_fullmap_aria2c_on_macos_says_drop_the_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """On macOS the preflight says to drop ``--aria2c``, never to install the extra.

    Why: the ``aria2`` distribution publishes no macOS wheels, so telling a mac user to install
    the extra is a dead end — the fix there is the default Python downloader.
    """
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    monkeypatch.setattr(cli, "run", lambda *args, **kwargs: pytest.fail("the download started on an unsupported platform"))
    monkeypatch.setattr(extras, "missing", lambda extra: ("aria2",))

    with pytest.raises(SystemExit) as excinfo:
        cli.build_fullmap(output=tmp_path / "fullmap.redb", aria2c=True)

    assert excinfo.value.code == 2
    message: str = capsys.readouterr().err
    assert "drop --aria2c" in message
    assert "tablassert[aria2]" not in message


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


# --- prebuilt fullmap downloader (download-first default; --force rebuilds from BABEL) ---


def _write_archive(filename: str, url: str, destination: Path, on_progress: object = None) -> Path:
    """Fake ``download_babel_file`` that just materializes the archive on disk (no network)."""
    destination.mkdir(parents=True, exist_ok=True)
    path: Path = destination / filename
    path.write_bytes(b"archive-bytes")
    return path


def _build_tiny_tar_zst(path: Path, members: dict[str, bytes]) -> None:
    """Create a tiny ``.tar.zst`` hermetically (native 3.14+ tarfile, else the ``zstd`` binary)."""
    if sys.version_info >= (3, 14):
        with tarfile.open(path, "w|zst") as tar:
            for name, data in members.items():
                info: tarfile.TarInfo = tarfile.TarInfo(name=name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        return
    zstd: str | None = shutil.which("zstd")
    if zstd is None:
        pytest.skip("no native zstd tarfile and no zstd binary to build a test archive")
    tar_path: Path = path.with_suffix(".tar")
    with tarfile.open(tar_path, "w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    subprocess.run([zstd, "-q", "-f", "-o", str(path), str(tar_path)], check=True)


def test_prebuilt_fullmap_urls_derive_installed_version() -> None:
    """The release directory is the INSTALLED Tablassert version, never a hardcoded literal.

    WHY: the RENCI layout is ``.../fullmap/<tablassert-version>/``; pinning a literal would
    break on every release. The URLs must embed the live installed-package version plus the
    BABEL snapshot, and expose the fixed ``fullmap.tar.zst`` / ``sha256sum.txt`` filenames.
    """
    from importlib.metadata import version as get_version

    release: str = get_version("tablassert")
    archive: str
    checksum: str
    archive, checksum = cli._prebuilt_fullmap_urls("2026jul22")
    assert archive == f"https://stars.renci.org/var/babel_outputs/2026jul22/fullmap/{release}/fullmap.tar.zst"
    assert checksum == f"https://stars.renci.org/var/babel_outputs/2026jul22/fullmap/{release}/sha256sum.txt"
    # The version slot is exactly the live installed version (equality above pins position);
    # an old release stamp like 8.1.0 must never appear in that slot.
    assert "8.1.0" not in archive


def test_fetch_prebuilt_sha256_parses_fullmap_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``fullmap.tar.zst`` digest is extracted from sha256sum.txt; other lines ignored."""
    digest: str = "0" * 64
    body: str = f"aaaa  other.gz\n{digest}  fullmap.tar.zst\nbbbb  fullmap.s0.redb\n"
    monkeypatch.setattr(cli, "urlopen", lambda request, timeout: _FakeResponse(body.encode(), 200, str(len(body))))
    assert cli._fetch_prebuilt_sha256("https://example.com/sha256sum.txt") == digest


def test_fetch_prebuilt_sha256_handles_binary_mode_star(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Binary-mode sha256sum prefixes the filename with ``*``; still matched."""
    digest: str = "a" * 64
    body: str = f"{digest} *fullmap.tar.zst\n"
    monkeypatch.setattr(cli, "urlopen", lambda request, timeout: _FakeResponse(body.encode(), 200, str(len(body))))
    assert cli._fetch_prebuilt_sha256("https://example.com/sha256sum.txt") == digest


def test_fetch_prebuilt_sha256_returns_none_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing/unreadable checksum file returns ``None`` (best-effort), not an error."""

    def _404(request: Any, timeout: int) -> None:
        raise HTTPError("https://example.com/sha256sum.txt", 404, "Not Found", Message(), None)

    monkeypatch.setattr(cli, "urlopen", _404)
    assert cli._fetch_prebuilt_sha256("https://example.com/sha256sum.txt") is None


def test_fetch_prebuilt_sha256_returns_none_when_no_fullmap_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A checksum file without a ``fullmap.tar.zst`` line returns ``None`` (no match to verify)."""
    body: str = "aaaa  other.gz\nbbbb  fullmap.s0.redb\n"  # no fullmap.tar.zst entry
    monkeypatch.setattr(cli, "urlopen", lambda request, timeout: _FakeResponse(body.encode(), 200, str(len(body))))
    assert cli._fetch_prebuilt_sha256("https://example.com/sha256sum.txt") is None


def test_fetch_prebuilt_fullmap_downloads_verifies_and_extracts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end orchestration: download archive -> verify checksum -> extract beside output.

    The download, checksum fetch, and zstd extraction are faked so the test asserts the
    ORCHESTRATION: primary + shards land beside ``output`` named after its stem, the verify
    branch actually ran against the downloaded bytes, and the archive is removed after extraction.
    """
    archive_bytes: bytes = b"pretend-archive"
    digest: str = hashlib.sha256(archive_bytes).hexdigest()

    def _fake_download(filename: str, url: str, destination: Path, on_progress: object = None) -> Path:
        destination.mkdir(parents=True, exist_ok=True)
        path: Path = destination / filename
        path.write_bytes(archive_bytes)
        if on_progress is not None:
            on_progress(len(archive_bytes), len(archive_bytes))  # type: ignore[operator]
        return path

    monkeypatch.setattr(cli, "download_babel_file", _fake_download)
    monkeypatch.setattr(cli, "_fetch_prebuilt_sha256", lambda url: digest)

    def _fake_extract(archive: Path, dest: Path, on_phase: object) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "fullmap.redb").write_bytes(b"PRIMARY")
        (dest / "fullmap.s0.redb").write_bytes(b"SHARD0")
        (dest / "fullmap.s1.redb").write_bytes(b"SHARD1")

    monkeypatch.setattr(cli, "_extract_zst_tar", _fake_extract)

    output: Path = tmp_path / "data" / "fullmap.redb"
    cli.fetch_prebuilt_fullmap(output, PipelineProgress(total_stages=2), version="2026jul22")

    assert output.read_bytes() == b"PRIMARY"
    assert (output.parent / "fullmap.s0.redb").read_bytes() == b"SHARD0"
    assert (output.parent / "fullmap.s1.redb").read_bytes() == b"SHARD1"
    assert not (output.parent / "fullmap.tar.zst").exists()  # archive removed after extraction


def test_fetch_prebuilt_fullmap_checksum_mismatch_unlinks_and_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A checksum mismatch deletes the corrupt archive and raises so the build falls back.

    WHY: extracting a corrupt 46 GB archive would waste disk/time and produce a broken DB.
    The mismatch path fails loud (PrebuiltFullmapUnavailable) AND removes the bad archive so
    the next attempt re-downloads instead of re-verifying garbage.
    """
    monkeypatch.setattr(cli, "download_babel_file", _write_archive)
    monkeypatch.setattr(cli, "_fetch_prebuilt_sha256", lambda url: "f" * 64)  # never matches
    monkeypatch.setattr(cli, "_extract_zst_tar", lambda *a, **k: pytest.fail("extract must not run on a checksum mismatch"))

    output: Path = tmp_path / "fullmap.redb"
    with pytest.raises(cli.PrebuiltFullmapUnavailable, match="checksum mismatch"):
        cli.fetch_prebuilt_fullmap(output, PipelineProgress(total_stages=2), version="v")
    assert not (output.parent / "fullmap.tar.zst").exists()  # corrupt archive removed


def test_fetch_prebuilt_fullmap_missing_checksum_proceeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When RENCI publishes no sha256sum.txt, extraction proceeds (warned, not blocked)."""
    monkeypatch.setattr(cli, "download_babel_file", _write_archive)
    monkeypatch.setattr(cli, "_fetch_prebuilt_sha256", lambda url: None)
    extracted: dict[str, bool] = {"ran": False}

    def _fake_extract(archive: Path, dest: Path, on_phase: object) -> None:
        extracted["ran"] = True
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "fullmap.redb").write_bytes(b"PRIMARY")

    monkeypatch.setattr(cli, "_extract_zst_tar", _fake_extract)
    output: Path = tmp_path / "fullmap.redb"
    cli.fetch_prebuilt_fullmap(output, PipelineProgress(total_stages=2), version="v")
    assert extracted["ran"] is True
    assert output.read_bytes() == b"PRIMARY"


def test_fetch_prebuilt_fullmap_download_failure_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed archive download surfaces as PrebuiltFullmapUnavailable (-> build fallback)."""

    def _fail(filename: str, url: str, destination: Path, on_progress: object = None) -> Path:
        raise BabelDownloadError(url, 1, RuntimeError("network down"))

    monkeypatch.setattr(cli, "download_babel_file", _fail)
    monkeypatch.setattr(cli, "_fetch_prebuilt_sha256", lambda url: None)
    output: Path = tmp_path / "fullmap.redb"
    with pytest.raises(cli.PrebuiltFullmapUnavailable, match="download failed"):
        cli.fetch_prebuilt_fullmap(output, PipelineProgress(total_stages=2), version="v")


def test_fetch_prebuilt_fullmap_aria2c_uses_shared_helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``aria2c=True`` routes the archive through ``download_babel_file_aria2c`` (PR #80's seam).

    WHY: the 46 GB prebuilt is the ideal aria2 use case, and aria2 discovery is moving into
    the optional ``[aria2]`` extra (PR #80). Pinning that the prebuilt path calls the SAME
    shared helper the build uses (not ``shutil.which`` directly) means it inherits #80's
    bundled binary automatically; the Python downloader must NOT run in this mode.
    """

    def _python_must_not_run(*args: Any, **kwargs: Any) -> Path:
        raise AssertionError("Python downloader must not run when aria2c=True")

    monkeypatch.setattr(cli, "download_babel_file", _python_must_not_run)
    aria_calls: list[tuple[str, str, Path]] = []

    def _fake_aria2c(filename: str, url: str, destination: Path, retries: int = 5) -> Path:
        aria_calls.append((filename, url, destination))
        destination.mkdir(parents=True, exist_ok=True)
        (destination / filename).write_bytes(b"archive-bytes")
        return destination / filename

    monkeypatch.setattr(cli, "download_babel_file_aria2c", _fake_aria2c)
    monkeypatch.setattr(cli, "_fetch_prebuilt_sha256", lambda url: None)

    def _fake_extract(archive: Path, dest: Path, on_phase: object) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "fullmap.redb").write_bytes(b"PRIMARY")

    monkeypatch.setattr(cli, "_extract_zst_tar", _fake_extract)

    output: Path = tmp_path / "data" / "fullmap.redb"
    cli.fetch_prebuilt_fullmap(output, PipelineProgress(total_stages=2), version="2026jul22", aria2c=True)

    assert len(aria_calls) == 1
    assert aria_calls[0][0] == "fullmap.tar.zst"
    assert aria_calls[0][1].endswith("/fullmap.tar.zst")
    assert aria_calls[0][2] == output.parent


def test_fetch_prebuilt_fullmap_custom_output_name_renames_shards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A custom ``--output`` stem renames the extracted primary + shards to match it.

    WHY: the read path derives shard names from the primary stem (``<stem>.s<N>.redb``), so a
    prebuilt tarball of ``fullmap.redb`` / ``fullmap.sN.redb`` must be renamed when
    ``--output`` is e.g. ``mydb.redb`` or lookups would not find the shards.
    """
    monkeypatch.setattr(cli, "download_babel_file", _write_archive)
    monkeypatch.setattr(cli, "_fetch_prebuilt_sha256", lambda url: None)

    def _fake_extract(archive: Path, dest: Path, on_phase: object) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "fullmap.redb").write_bytes(b"PRIMARY")
        (dest / "fullmap.s0.redb").write_bytes(b"SHARD0")
        (dest / "fullmap.s15.redb").write_bytes(b"SHARD15")

    monkeypatch.setattr(cli, "_extract_zst_tar", _fake_extract)
    output: Path = tmp_path / "store" / "mydb.redb"
    cli.fetch_prebuilt_fullmap(output, PipelineProgress(total_stages=2), version="v")

    assert output.read_bytes() == b"PRIMARY"
    assert (output.parent / "mydb.s0.redb").read_bytes() == b"SHARD0"
    assert (output.parent / "mydb.s15.redb").read_bytes() == b"SHARD15"
    # the original fullmap.* names did NOT survive the rename
    assert not (output.parent / "fullmap.redb").exists()
    assert not (output.parent / "fullmap.s0.redb").exists()


def test_fetch_prebuilt_fullmap_raises_when_archive_has_no_primary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An archive with only shards (no primary .redb) fails loud instead of silently passing."""
    monkeypatch.setattr(cli, "download_babel_file", _write_archive)
    monkeypatch.setattr(cli, "_fetch_prebuilt_sha256", lambda url: None)
    monkeypatch.setattr(
        cli,
        "_extract_zst_tar",
        lambda archive, dest, on_phase: (dest.mkdir(parents=True, exist_ok=True), (dest / "fullmap.s0.redb").write_bytes(b"SHARD")),
    )
    output: Path = tmp_path / "fullmap.redb"
    with pytest.raises(cli.PrebuiltFullmapUnavailable, match=r"no primary \.redb"):
        cli.fetch_prebuilt_fullmap(output, PipelineProgress(total_stages=2), version="v")


def test_extract_zst_tar_extracts_real_tiny_archive(tmp_path: Path) -> None:
    """A real tiny ``.tar.zst`` round-trips through ``_extract_zst_tar`` (native path on 3.14+).

    WHY: the production archive is a 46 GB ``.tar.zst`` we cannot fetch in tests; this builds a
    tiny one and proves the streaming extraction + data filter land members on disk, including
    a nested file (parent dirs created automatically).
    """
    archive: Path = tmp_path / "fullmap.tar.zst"
    _build_tiny_tar_zst(archive, {"fullmap.redb": b"PRIMARY", "fullmap.s0.redb": b"SHARD0", "nested/x.txt": b"hi"})
    dest: Path = tmp_path / "out"
    phases: list[str] = []
    cli._extract_zst_tar(archive, dest, on_phase=phases.append)
    assert (dest / "fullmap.redb").read_bytes() == b"PRIMARY"
    assert (dest / "fullmap.s0.redb").read_bytes() == b"SHARD0"
    assert (dest / "nested" / "x.txt").read_bytes() == b"hi"
    assert any("extracting" in p for p in phases)


def test_extract_zst_tar_falls_back_to_zstd_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When native zstd is unavailable, the ``zstd`` binary streams the archive into tarfile.

    WHY: Python 3.11-3.13 lack native tarfile zstd, so the binary fallback is the only path
    there. On 3.14+ we force the native attempt to raise ``CompressionError`` to exercise the
    fallback; skipped when no ``zstd`` binary is installed.
    """
    zstd: str | None = shutil.which("zstd")
    if zstd is None:
        pytest.skip("no zstd binary to exercise the fallback path")
    archive: Path = tmp_path / "fullmap.tar.zst"
    _build_tiny_tar_zst(archive, {"fullmap.redb": b"PRIMARY"})
    real_open = cli.tarfile.open

    def _force_native_failure(*args: Any, **kwargs: Any) -> Any:
        mode: str = args[1] if len(args) > 1 else str(kwargs.get("mode", ""))
        if "zst" in mode:
            raise cli.tarfile.CompressionError("forced: simulate a pre-3.14 runtime")
        return real_open(*args, **kwargs)

    monkeypatch.setattr(cli.tarfile, "open", _force_native_failure)
    dest: Path = tmp_path / "out"
    cli._extract_zst_tar(archive, dest, on_phase=lambda p: None)
    assert (dest / "fullmap.redb").read_bytes() == b"PRIMARY"


def _force_native_zst_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``tarfile.open`` reject the ``r|zst`` mode so the zstd-binary path runs on any Python."""
    real_open = cli.tarfile.open

    def _fake_open(*args: Any, **kwargs: Any) -> Any:
        mode: str = args[1] if len(args) > 1 else str(kwargs.get("mode", ""))
        if "zst" in mode:
            raise cli.tarfile.CompressionError("forced: simulate a pre-3.14 runtime")
        return real_open(*args, **kwargs)

    monkeypatch.setattr(cli.tarfile, "open", _fake_open)


def test_extract_zst_tar_raises_when_no_zstd_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No native zstd AND no zstd binary => PrebuiltFullmapUnavailable (-> build fallback).

    WHY: a pre-3.14 Python without the zstd CLI cannot extract the archive, so the default
    from-scratch BABEL build is the correct fallback.
    """
    archive: Path = tmp_path / "fullmap.tar.zst"
    archive.write_bytes(b"not-used")  # never reaches the archive before the binary check
    _force_native_zst_failure(monkeypatch)
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    with pytest.raises(cli.PrebuiltFullmapUnavailable, match="`zstd` executable was not found"):
        cli._extract_zst_tar(archive, tmp_path / "out", on_phase=lambda p: None)


def test_extract_zst_tar_raises_when_zstd_binary_fails_to_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A zstd binary present but failing to exec => PrebuiltFullmapUnavailable (defensive OSError)."""
    archive: Path = tmp_path / "fullmap.tar.zst"
    archive.write_bytes(b"x")
    _force_native_zst_failure(monkeypatch)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/zstd")
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(OSError("exec failed")))
    with pytest.raises(cli.PrebuiltFullmapUnavailable, match="could not start zstd"):
        cli._extract_zst_tar(archive, tmp_path / "out", on_phase=lambda p: None)


def test_extract_zst_tar_raises_on_corrupt_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A corrupt (non-zstd) archive => tarfile read error => PrebuiltFullmapUnavailable (-> build fallback).

    WHY: a truncated/corrupt download must fail loud rather than silently extract garbage. zstd
    decompresses nothing, tarfile hits an empty/invalid stream, and the fallback read-error path fires.
    """
    zstd: str | None = shutil.which("zstd")
    if zstd is None:
        pytest.skip("no zstd binary to exercise the corrupt-archive path")
    archive: Path = tmp_path / "fullmap.tar.zst"
    archive.write_bytes(b"this is definitely not a zstd stream")
    _force_native_zst_failure(monkeypatch)
    with pytest.raises(cli.PrebuiltFullmapUnavailable, match="failed to extract prebuilt archive"):
        cli._extract_zst_tar(archive, tmp_path / "out", on_phase=lambda p: None)


def test_extract_zst_tar_raises_on_nonzero_zstd_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-zero zstd exit AFTER a valid tar stream still fails loud (returncode guard).

    WHY: zstd can write a complete, valid stream and still exit non-zero (e.g. trailing garbage
    after the frame). The returncode guard refuses to trust such output. Native is forced off and
    a fake Popen serves a real (uncompressed) tar stream while reporting exit 2.
    """
    tar_buf: io.BytesIO = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w|") as tar:
        info: tarfile.TarInfo = tarfile.TarInfo(name="fullmap.redb")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"P"))
    tar_buf.seek(0)

    class _FakeProc:
        def __init__(self) -> None:
            self.stdout = tar_buf
            self.stderr = io.BytesIO(b"")
            self.returncode = 2

        def wait(self) -> int:
            return 2

    _force_native_zst_failure(monkeypatch)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/zstd")
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *a, **k: _FakeProc())
    archive: Path = tmp_path / "fullmap.tar.zst"
    archive.write_bytes(b"ignored")  # fake Popen ignores the archive path
    with pytest.raises(cli.PrebuiltFullmapUnavailable, match="zstd exited with status 2"):
        cli._extract_zst_tar(archive, tmp_path / "out", on_phase=lambda p: None)


def test_build_fullmap_command_defaults_to_prebuilt_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Default (no ``--force``) tries the prebuilt download FIRST, not a from-scratch build.

    WHY: 'download from here first (before building)' is the requested default. The command
    calls ``run(2, fetch_prebuilt_fullmap, ...)`` and returns on success WITHOUT invoking the
    3-stage build. (``output`` absent => no skip-if-exists short-circuit.)
    """
    calls: list[tuple[Any, ...]] = []

    def _fake_run(stages: int, fn: Any, arg: Path, **kwargs: Any) -> None:
        calls.append((stages, fn, arg, kwargs))

    monkeypatch.setattr(cli, "run", _fake_run)
    monkeypatch.setattr(extras, "missing", lambda extra: ())  # --aria2c preflight: report [aria2] as installed
    output: Path = tmp_path / "fullmap.redb"  # does not exist
    cli.build_fullmap(output=output, version="v", aria2c=True)
    assert calls == [(2, cli.fetch_prebuilt_fullmap, output, {"version": "v", "aria2c": True})]


def test_build_fullmap_command_skips_when_output_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """A complete existing DB short-circuits everything (no ``run`` call) unless ``--force``."""
    output: Path = tmp_path / "fullmap.redb"
    output.write_bytes(b"existing-db")

    def _run_must_not_run(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("run must not be called when output already exists")

    monkeypatch.setattr(cli, "run", _run_must_not_run)
    cli.build_fullmap(output=output, version="v")
    assert "already present" in capsys.readouterr().err


def test_build_fullmap_command_falls_back_to_build_on_prebuilt_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the prebuilt fetch raises, the command falls back to a from-scratch build.

    WHY: a version with no published prebuilt (or a download/extract failure) must not abort;
    it builds from BABEL. Both ``run`` calls fire: the fetch (which raises) then the build.
    """
    calls: list[tuple[Any, ...]] = []
    state: dict[str, int] = {"n": 0}

    def _fake_run(stages: int, fn: Any, arg: Path, **kwargs: Any) -> None:
        calls.append((stages, fn, arg, kwargs))
        state["n"] += 1
        if state["n"] == 1:
            raise cli.PrebuiltFullmapUnavailable("no prebuilt for this version")

    monkeypatch.setattr(cli, "run", _fake_run)
    monkeypatch.setattr(extras, "missing", lambda extra: ())  # --aria2c preflight: report [aria2] as installed
    output: Path = tmp_path / "fullmap.redb"  # absent
    cache: Path = tmp_path / "c"
    cli.build_fullmap(output=output, cache=cache, version="v", threads=4, aria2c=True)
    assert len(calls) == 2
    assert calls[0][0] == 2
    assert calls[0][1] is cli.fetch_prebuilt_fullmap
    assert calls[1] == (3, cli.build_fullmap_pipeline, output, {"cache": cache, "version": "v", "threads": 4, "aria2c": True})


def test_build_fullmap_force_flag_parses() -> None:
    """``--force``/``-f`` parse True; no generated ``--no-force`` alias (``negative=""``)."""

    def parse(argv: list[str]) -> dict[str, Any]:
        fn, bound, _ = cli.APP.parse_args(argv, exit_on_error=False)
        assert fn is cli.build_fullmap
        return dict(bound.arguments)

    assert parse(["build-fullmap"]) == {}  # defaults are absent when no flag is passed
    assert parse(["build-fullmap", "--force"])["force"] is True
    assert parse(["build-fullmap", "-f"])["force"] is True
    with pytest.raises(UnknownOptionError):
        parse(["build-fullmap", "--no-force"])
