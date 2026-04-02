from __future__ import annotations

from typing import TYPE_CHECKING

import lazy_loader as Lazy

if TYPE_CHECKING:
    import polars as pl
else:
    pl = Lazy.load("polars")


def level_one(lf: pl.LazyFrame, col: str) -> pl.LazyFrame:
    # ? Level One Text Processing
    expr: pl.Expr = pl.col(col).cast(pl.String).str.strip_chars().str.to_lowercase()
    return lf.with_columns(expr.alias(col))


def level_two(
    lf: pl.LazyFrame,
    col: str,  # pyright: ignore
    regex: str = r"\W+",
    tag: str = " two",
) -> pl.LazyFrame:
    # ? Level Two Text Processing
    expr: pl.Expr = pl.col(col).str.replace_all(regex, "")
    col: str = col + tag
    return lf.with_columns(expr.alias(col))
