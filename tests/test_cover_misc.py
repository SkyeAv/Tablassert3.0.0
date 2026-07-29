"""Coverage tests for small uncovered branches across rig, coerce, ingests, progress."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from tablassert.coerce import STUDY_SIZE_COUNT_PATTERN, STUDY_SIZE_SUFFIX_PATTERN, study_size_target
from tablassert.ingests import fastmerge
from tablassert.progress import _truncate
from tablassert.rig import clean_values, normalize_biolink_category, rig_edge_type_info


def test_normalize_biolink_category_rejects_non_string_and_empty() -> None:
    """rig.py:84 — non-string or empty input returns ``None`` (the guard branch)."""
    assert normalize_biolink_category(None) is None
    assert normalize_biolink_category(123) is None
    assert normalize_biolink_category("") is None


def test_clean_values_skips_none_items() -> None:
    """rig.py:118 — ``None`` items inside a nested list are skipped via ``continue``."""
    assert clean_values([[None, "gene"]]) == ["gene"]


def test_clean_values_skips_empty_and_placeholder_tokens() -> None:
    """rig.py:121 — empty strings and na/nan/null/none placeholders are dropped."""
    assert clean_values(["", "   ", "NA", "nan", "null", "none", "real"]) == ["real"]


def test_rig_edge_type_info_empty_without_expected_columns(tmp_path: Path) -> None:
    """rig.py:156 — returns ``[]`` when none of the expected edge columns exist."""
    lf: pl.LazyFrame = pl.LazyFrame({"unrelated": [1, 2]})
    assert rig_edge_type_info(lf, tmp_path / "edges.tsv", None) == []


def test_study_size_suffix_pattern_is_shadowed_by_count() -> None:
    """coerce.py:331 — the SUFFIX return is unreachable: SUFFIX is a strict subset of COUNT.

    Every ``unit<_SEP>n`` name matches ``STUDY_SIZE_COUNT_PATTERN`` first (coerce.py:326),
    so ``study_size_target`` returns before reaching the SUFFIX branch. This test documents
    that the SUFFIX pattern itself still matches (defensive duplicate), but line 331 is
    dead code that cannot be exercised through the public function.
    """
    name: str = "samples_n"
    assert STUDY_SIZE_SUFFIX_PATTERN.search(name) is not None
    assert STUDY_SIZE_COUNT_PATTERN.search(name) is not None
    assert study_size_target(name) == "supporting_study_size"


def test_fastmerge_returns_b_on_scalar_collision() -> None:
    """ingests.py:42 — top-level scalar/scalar (non dict-dict, non list-list) returns ``b``."""
    assert fastmerge(1, 2) == 2  # pyright: ignore[reportArgumentType]  # deliberate scalar to hit the fallback branch
    assert fastmerge("a", "b") == "b"  # pyright: ignore[reportArgumentType]  # deliberate scalar to hit the fallback branch
    assert fastmerge({"x": 1}, [1, 2]) == [1, 2]


def test_truncate_collapses_to_ellipsis_when_max_width_at_most_one() -> None:
    """progress.py:68 — over-long input with ``max_width <= 1`` yields a lone ellipsis."""
    assert _truncate("abcdef", 1) == "…"
    assert _truncate("abcdef", 0) == "…"
