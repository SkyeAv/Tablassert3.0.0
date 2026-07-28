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

from itertools import chain
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from tablassert.cli import _extract_sections_indexed, _load_table_indexed
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
