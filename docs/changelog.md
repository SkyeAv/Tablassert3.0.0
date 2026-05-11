# Changelog

The canonical release history lives in the repository root at [`CHANGELOG.md`](https://github.com/SkyeAv/Tablassert/blob/main/CHANGELOG.md).

## Current Release Notes

## 7.4.6 - 2026-05-11

### Changes

- Relaxed the `url` field validator in `BaseSource` to ignore `403 Forbidden` responses from `httpx.head()`. Some upstreams reject anonymous `HEAD` probes with 403 even though the URL itself is well-formed and reachable, so 403 no longer fails config validation.

For older releases and the full project history, open the root `CHANGELOG.md` in the repository.
