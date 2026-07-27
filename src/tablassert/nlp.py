from __future__ import annotations

from typing import TYPE_CHECKING

from tablassert._lazy import LazyModule

if TYPE_CHECKING:
    import polars as pl
else:
    pl = LazyModule("polars")


def level_one(lf: pl.LazyFrame, col: str) -> pl.LazyFrame:
    """Strip whitespace and lowercase a text column (level-one normalization).

    Args:
        lf: Source LazyFrame.
        col: Name of the column to normalize in place.

    Returns:
        LazyFrame with the column cast to string, trimmed, and lowercased.
    """
    expr: pl.Expr = pl.col(col).cast(pl.String).str.strip_chars().str.to_lowercase()
    return lf.with_columns(expr.alias(col))


def level_two(
    lf: pl.LazyFrame,
    col: str,  # pyright: ignore
    regex: str = r"\W+",
    tag: str = "_two",
) -> pl.LazyFrame:
    """Remove non-word characters from a text column (level-two normalization).

    The cleaned values are written to a new column named ``f"{col}{tag}"`` rather
    than overwriting the source column.

    Args:
        lf: Source LazyFrame.
        col: Name of the source column to clean.
        regex: Pattern whose matches are removed. Defaults to runs of non-word
            characters (``\\W+``).
        tag: Suffix appended to ``col`` to form the output column name.

    Returns:
        LazyFrame with the new tagged column added.
    """
    expr: pl.Expr = pl.col(col).str.replace_all(regex, "")
    col: str = col + tag
    return lf.with_columns(expr.alias(col))
