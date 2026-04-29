# Changelog

The canonical release history lives in the repository root at [`CHANGELOG.md`](https://github.com/SkyeAv/Tablassert/blob/main/CHANGELOG.md).

## Current Release Notes

### 7.3.4 - 2026-04-28

- `downloader.from_url()` now handles URLs that respond with an immediate download instead of a navigable page — the Playwright session uses a download-aware browser context and tolerates the expected `net::ERR_ABORTED` navigation error.
- Table configuration docs now describe `miscellaneous notes` as a freetext catch-all annotation for source context that doesn't map cleanly to a structured field.
- Regex transform documentation now spells out the Polars `str.replace_all()` constraints — no capturing groups or lookarounds — so authors know to chain simple substitutions or fall back to `miscellaneous notes`.

For older releases and the full project history, open the root `CHANGELOG.md` in the repository.
