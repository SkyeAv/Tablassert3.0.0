# Changelog

The canonical release history lives in the repository root at [`CHANGELOG.md`](https://github.com/SkyeAv/Tablassert/blob/main/CHANGELOG.md).

## Current Release Notes

## 7.4.2 - 2026-05-07

### Changes

- Added Pydantic field and model validators to `models.py` that catch misconfigurations at parse time: unreachable `url` fields, mutually exclusive `rows`/`row_slice`, mismatched `Reindex` comparator types, non–Excel-style column letters under `method: column`, Polars-incompatible `regex`/`remove` patterns, and `annotation` name normalization (underscores → spaces).
- Changed `rows` and `row_slice` element type from `NonNegativeInt` to `PositiveInt`.

## 7.4.1 - 2026-05-05

### Bug Fixes

- Fixed crash in the BUILDING TCODE progress display caused by `format_section_oneline()` calling `.value` on `Tcode.status`, which is a plain string under `use_enum_values=True`.

For older releases and the full project history, open the root `CHANGELOG.md` in the repository.
