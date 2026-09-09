"""Opt-in benchmark harness for streaming source-file hashing.

WHY: content-aware Stage 3 keys prevent stale cached builds, but the source file must
be read cheaply enough that correctness does not become a build bottleneck. This
harness measures the Rust XXH64 pass against a same-file chunked read floor and
measures the actual Stage-3-style key derivation overhead against the former
config-only ``mkhash(section)`` key.

Usage (opt-in; this module is skipped before test collection by default)::

    TABLASSERT_BENCH=1 uv run pytest tests/bench_file_hash_bench.py -s -n 0 -q

The one-gigabyte CSV is generated in pytest's session temporary directory using a
seeded, repeatable row pattern. Generation is streamed in 8 MiB blocks and the
benchmark performs one run of each measurement.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("TABLASSERT_BENCH") != "1", reason="benchmark harness; opt in with TABLASSERT_BENCH=1")

if os.environ.get("TABLASSERT_BENCH") != "1":
    pytest.skip("benchmark harness; opt in with TABLASSERT_BENCH=1", allow_module_level=True)

_SEED = 6006
_GIB = 1 << 30
_CHUNK_BYTES = 8 * 1024 * 1024
_FIXTURE_BYTES = 4 * 1024 * 1024


def _seeded_csv_block() -> bytes:
    """Return one deterministic 8 MiB block made from a seeded CSV row."""
    import random

    rng = random.Random(_SEED)
    fields = [f"field_{index}_{rng.randrange(1_000_000_000)}" for index in range(8)]
    row = ("subject,predicate,value," + ",".join(fields) + "\n").encode()
    return (row * ((_CHUNK_BYTES // len(row)) + 1))[:_CHUNK_BYTES]


def _write_seeded_csv(path: Path, size: int) -> int:
    """Stream exactly ``size`` bytes of a repeatable CSV-shaped artifact."""
    block = _seeded_csv_block()
    written = 0
    with path.open("wb") as handle:
        while written < size:
            chunk_size = min(len(block), size - written)
            handle.write(block[:chunk_size])
            written += chunk_size
    return written


@pytest.fixture(scope="session")
def benchmark_files(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Create the large throughput artifact and the small Stage 3 fixture once."""
    root = tmp_path_factory.mktemp("file_hash_bench")
    large = root / "seeded_1gib.csv"
    fixture = root / "stage3_fixture.csv"
    assert _write_seeded_csv(large, _GIB) == _GIB
    assert _write_seeded_csv(fixture, _FIXTURE_BYTES) == _FIXTURE_BYTES
    return large, fixture


def _time_chunked_read(path: Path) -> tuple[float, int]:
    """Read a file in the hashing primitive's chunk size and return time and bytes."""
    started = time.perf_counter()
    read_bytes = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            read_bytes += len(chunk)
    return time.perf_counter() - started, read_bytes


def _gib_per_second(size: int, elapsed: float) -> float:
    """Convert a byte count and wall time into binary GiB/s."""
    return size / _GIB / elapsed


def _stage3_key_timings(source: Path) -> tuple[float, float, list[str], list[str]]:
    """Measure content-aware Stage 3 keys against config-only keys."""
    from tablassert.cli import _section_store_key_for_build
    from tablassert.utils import mkhash

    sections = [{"config": Path("table.yaml"), "marker": f"section-{index}", "source": {"local": str(source)}} for index in range(8)]
    memo: dict[tuple[Path, int, int], str] = {}
    started = time.perf_counter()
    content_keys = [_section_store_key_for_build(section, Path("graph.yaml"), memo)[0] for section in sections]
    content_elapsed = time.perf_counter() - started

    started = time.perf_counter()
    config_keys = [mkhash(section) for section in sections]
    config_elapsed = time.perf_counter() - started
    return content_elapsed, config_elapsed, content_keys, config_keys


def test_bench_file_hash(benchmark_files: tuple[Path, Path]) -> None:
    """Measure file hashing, the raw-read floor, and Stage 3 key derivation.

    WHY: a measured one-run result makes the performance tradeoff visible without
    imposing a multi-gigabyte workload on normal CI, which skips this module entirely.
    """
    from tablassert import rs

    large, fixture = benchmark_files
    size = large.stat().st_size
    assert size == _GIB, "benchmark artifact must be exactly 1 GiB"

    started = time.perf_counter()
    digest = rs.xxh64_file(str(large))
    hash_elapsed = time.perf_counter() - started
    assert large.stat().st_size == size
    assert len(digest) == 16
    assert all(character in "0123456789abcdef" for character in digest)

    read_elapsed, read_bytes = _time_chunked_read(large)
    assert read_bytes == size, "pure-read floor must exercise the complete benchmark artifact"

    content_elapsed, config_elapsed, content_keys, config_keys = _stage3_key_timings(fixture)
    assert len(content_keys) == len(config_keys) == 8
    assert all(len(key) == 16 for key in content_keys + config_keys)
    assert content_keys != config_keys, "content-aware keys must differ from config-only keys"
    delta_ms_per_file = (content_elapsed - config_elapsed) * 1000 / len(content_keys)

    print(
        f"\nxxh64_file: size={size:,} bytes ({size / _GIB:.3f} GiB) digest={digest} "
        f"wall_time={hash_elapsed:.3f}s throughput={_gib_per_second(size, hash_elapsed):.3f} GiB/s"
    )
    print(
        f"pure read: bytes={read_bytes:,} ({read_bytes / _GIB:.3f} GiB) "
        f"wall_time={read_elapsed:.3f}s throughput={_gib_per_second(read_bytes, read_elapsed):.3f} GiB/s"
    )
    print(
        f"stage 3 keys: sections={len(content_keys)} shared_files=1 content_aware={content_elapsed * 1000:.3f}ms "
        f"config_only_mkhash={config_elapsed * 1000:.3f}ms delta={delta_ms_per_file:.3f}ms/file"
    )
