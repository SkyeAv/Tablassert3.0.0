"""Order-preservation guards for the per-file Stage 1/2 progress workers (US-010).

``build_pipeline`` Stages 1-2 switched from ``Pool.map``/``Pool.starmap`` (which return
results in INPUT order) to ``Pool.imap_unordered`` (which yields them in COMPLETION order)
to get per-file progress ticks. These tests pin that the index-carrying workers plus
by-index reassembly keep ``raw[i]``/``temp[i]`` aligned with ``g.tables[i]`` — so the
flattened ``sections`` order (and thus the downstream Tcode build + NDJSON write order)
is byte-identical to the old code, regardless of the order ``imap_unordered`` yields in.

Offline and fast: tiny YAML table configs under ``tmp_path``; table ``i`` yields two
sections stamped with ``marker: i`` so ordering is directly observable.
"""

from __future__ import annotations

import io
from itertools import chain
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import pytest

from tablassert import cli
from tablassert.cli import _download_detail, _extract_sections_indexed, _load_table_indexed, download_babel_file, stream_copy
from tablassert.ingests import from_yaml, to_sections, to_yaml


def _write_tables(root: Path, count: int) -> list[Path]:
    """Write ``count`` tiny table configs; table ``i`` yields 2 sections stamped ``marker: i``."""
    tables: list[Path] = []
    for i in range(count):
        table: Path = root / f"table_{i}.yaml"
        to_yaml(table, {"template": {"marker": i}, "sections": [{"s": 0}, {"s": 1}]})
        tables.append(table)
    return tables


def test_load_table_indexed_reassembly_preserves_input_order(tmp_path: Path) -> None:
    """Stage 1: by-index reassembly yields ``raw`` in table order even if completion order varies."""
    tables: list[Path] = _write_tables(tmp_path, 5)
    # Compute the worker results, then feed them in REVERSED (out-of-order) completion order.
    results: list[tuple[int, object]] = [_load_table_indexed((idx, table)) for idx, table in enumerate(tables)]
    raw: list[object] = [None for _ in tables]
    for idx, parsed in reversed(results):
        raw[idx] = parsed
    assert raw == [from_yaml(table) for table in tables]


def test_extract_sections_indexed_reassembly_preserves_input_order(tmp_path: Path) -> None:
    """Stage 2: by-index reassembly keeps flattened sections in table order, identical to starmap+chain."""
    tables: list[Path] = _write_tables(tmp_path, 4)
    raw: list[object] = [from_yaml(table) for table in tables]
    # Ground truth: the OLD starmap+chain result (sequential, input order).
    expected: list[dict[str, Any]] = list(chain.from_iterable(to_sections(r, table) for r, table in zip(raw, tables, strict=True)))  # pyright: ignore
    # New behavior: worker results fed in REVERSED completion order, reassembled by index, flattened.
    results: list[tuple[int, list[dict[str, Any]]]] = [
        _extract_sections_indexed((idx, r, table)) for idx, (r, table) in enumerate(zip(raw, tables, strict=True))
    ]
    temp: list[list[dict[str, Any]]] = [[] for _ in tables]
    for idx, section_list in reversed(results):
        temp[idx] = section_list
    sections: list[dict[str, Any]] = list(chain.from_iterable(temp))
    assert sections == expected
    # The flattened ``config`` stamps follow table order (2 sections per table): t0,t0,t1,t1,...
    assert [s["config"] for s in sections] == [table for table in tables for _ in range(2)]


def test_imap_unordered_reassembly_preserves_input_order(tmp_path: Path) -> None:
    """End-to-end through the real ``Pool.imap_unordered`` path used by build_pipeline Stages 1-2."""
    tables: list[Path] = _write_tables(tmp_path, 6)
    raw: list[object] = [None for _ in tables]
    with Pool() as pool:
        for idx, parsed in pool.imap_unordered(_load_table_indexed, enumerate(tables)):
            raw[idx] = parsed
    assert raw == [from_yaml(table) for table in tables]

    temp: list[list[dict[str, Any]]] = [[] for _ in tables]
    with Pool() as pool:
        for idx, section_list in pool.imap_unordered(_extract_sections_indexed, zip(range(len(tables)), raw, tables, strict=True)):
            temp[idx] = section_list
    expected: list[dict[str, Any]] = list(chain.from_iterable(to_sections(from_yaml(table), table) for table in tables))  # pyright: ignore
    assert list(chain.from_iterable(temp)) == expected


class _FakeResponse:
    """Offline stand-in for ``urllib``'s context-managed HTTPResponse."""

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


def test_stream_copy_reports_cumulative_bytes() -> None:
    """on_bytes fires after each chunk with a monotonic running total ending at the source size."""
    payload: bytes = b"x" * (2 * 1024 * 1024 + 512)  # spans 3 one-MiB chunks
    destination: io.BytesIO = io.BytesIO()
    calls: list[int] = []
    stream_copy(io.BytesIO(payload), destination, on_bytes=calls.append)
    assert destination.getvalue() == payload  # bytes identical
    assert calls == sorted(calls)  # monotonic non-decreasing
    assert calls[-1] == len(payload)  # ends at total source size
    assert len(calls) == 3  # one callback per chunk written


def test_stream_copy_without_callback_copies_identically() -> None:
    """on_bytes=None (default) preserves the original copy-only behavior exactly."""
    payload: bytes = b"abcdef" * 300_000
    destination: io.BytesIO = io.BytesIO()
    stream_copy(io.BytesIO(payload), destination)
    assert destination.getvalue() == payload


def test_download_detail_formats_megabytes() -> None:
    """Detail line shows transferred/total MB, dropping the denominator when total is unknown."""
    assert _download_detail(1_500_000, 3_000_000) == "1.5/3.0 MB"
    assert _download_detail(1_500_000, 0) == "1.5 MB"  # unknown total
    assert _download_detail(300_000, -1) == "0.3 MB"  # total <= 0 => no denominator


def test_download_babel_file_reports_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """on_progress receives (downloaded, total) with total from Content-Length on HTTP 200."""
    payload: bytes = b"z" * (2 * 1024 * 1024 + 10)
    monkeypatch.setattr(cli, "urlopen", lambda request, timeout: _FakeResponse(payload, 200, str(len(payload))))
    events: list[tuple[int, int]] = []
    out: Path = download_babel_file("f.gz", "https://example.com/f.gz", tmp_path, on_progress=lambda d, t: events.append((d, t)))
    assert out.read_bytes() == payload  # downloaded bytes identical
    downloaded: list[int] = [d for d, _ in events]
    assert downloaded == sorted(downloaded)  # monotonic
    assert all(total == len(payload) for _, total in events)  # total known from Content-Length
    assert events[-1] == (len(payload), len(payload))  # finishes at full size


def test_download_babel_file_resume_progress_includes_offset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On HTTP 206 resume, progress base includes the existing .part offset and total = offset + Content-Length."""
    existing: bytes = b"a" * 100
    remaining: bytes = b"b" * 50
    (tmp_path / "f.gz.part").write_bytes(existing)
    monkeypatch.setattr(cli, "urlopen", lambda request, timeout: _FakeResponse(remaining, 206, str(len(remaining))))
    events: list[tuple[int, int]] = []
    out: Path = download_babel_file("f.gz", "https://example.com/f.gz", tmp_path, on_progress=lambda d, t: events.append((d, t)))
    assert out.read_bytes() == existing + remaining  # appended, bytes identical
    assert events[-1] == (150, 150)  # downloaded = offset(100)+remaining(50); total = offset+Content-Length
