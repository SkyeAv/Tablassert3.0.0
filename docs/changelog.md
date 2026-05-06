# Changelog

The canonical release history lives in the repository root at [`CHANGELOG.md`](https://github.com/SkyeAv/Tablassert/blob/main/CHANGELOG.md).

## Current Release Notes

## 7.4.1 - 2026-05-05

### Bug Fixes

- Fixed crash in the BUILDING TCODE progress display caused by `format_section_oneline()` calling `.value` on `Tcode.status`, which is a plain string under `use_enum_values=True`.

For older releases and the full project history, open the root `CHANGELOG.md` in the repository.
