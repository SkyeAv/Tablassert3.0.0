# Changelog

The canonical release history lives in the repository root at [`CHANGELOG.md`](https://github.com/SkyeAv/Tablassert/blob/main/CHANGELOG.md).

## Current Release Notes

## 7.4.3 - 2026-05-11

### Bug Fixes

- Fixed `compile_graph()` emitting output paths with the trailing semver segment stripped (e.g. `tablassert_7.4` instead of `tablassert_7.4.3`). `Path.with_suffix()` treated the version's final `.N` as the path suffix; the base path now carries a `.tmp` sentinel that gets replaced instead. Temp suffixes also normalized from `.temp` to `.tmp`.

For older releases and the full project history, open the root `CHANGELOG.md` in the repository.
