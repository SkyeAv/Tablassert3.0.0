# Changelog

The canonical release history lives in the repository root at [`CHANGELOG.md`](https://github.com/SkyeAv/Tablassert/blob/main/CHANGELOG.md).

## Current Release Notes

## 7.4.5 - 2026-05-11

### Changes

- Cached `url` field validator results in `BaseSource` to a `diskcache.Cache` stored at `.cachassert/`, so repeated config parses skip redundant `httpx.head()` round-trips against unchanged URLs.
- Increased the `httpx.head()` timeout in the `url` validator from 5.0s to 15.0s to further tolerate slower upstreams.
- Added `diskcache` as a runtime dependency.

For older releases and the full project history, open the root `CHANGELOG.md` in the repository.
