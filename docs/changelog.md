# Changelog

The canonical release history lives in the repository root at [`CHANGELOG.md`](https://github.com/SkyeAv/Tablassert/blob/main/CHANGELOG.md).

## Current Release Notes

## 7.4.4 - 2026-05-11

### Changes

- Relaxed the `url` field validator in `BaseSource` to only fail on 4xx responses from `httpx.head()`. Sources whose servers return 5xx or other non-2xx statuses to `HEAD` requests now pass config validation, since the URL itself is still well-formed.
- Increased the `httpx.head()` timeout in the `url` validator from 3.0s to 5.0s to tolerate slower upstreams.

For older releases and the full project history, open the root `CHANGELOG.md` in the repository.
