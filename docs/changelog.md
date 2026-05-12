# Changelog

The canonical release history lives in the repository root at [`CHANGELOG.md`](https://github.com/SkyeAv/Tablassert/blob/main/CHANGELOG.md).

## Current Release Notes

## 7.4.8 - 2026-05-12

### Changes

- Expanded `fullmap_audit()` failure logging in `qc.py` so each rejected CURIE log line now carries the underlying `FUZZ_RATIO`, `FUZZ_PARTIAL`, and (when the BERT stage ran) `BERT_SIMILARITY` score values, making it easier to diagnose why a term was dropped during QC.

For older releases and the full project history, open the root `CHANGELOG.md` in the repository.
