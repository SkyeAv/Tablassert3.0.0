# Changelog

The canonical release history lives in the repository root at [`CHANGELOG.md`](https://github.com/SkyeAv/Tablassert/blob/main/CHANGELOG.md).

## Current Release Notes

## 7.5.2 - 2026-07-01

### Changed
- `sig()` in `lib.py` now emits a third significance label, `"INCONCLUSIVE"`, for p-values that fall between the existing significance `cutoff` (default `0.05`, inclusive) and a new `threshold` parameter (default `0.10`, exclusive). A p-value `p` is now mapped as: null → `"UNSURE"`; `p <= cutoff` → `"YES"`; `cutoff < p < threshold` → `"INCONCLUSIVE"`; `p >= threshold` → `"NO"`. The upper bound is exclusive so `0.10` (and the `0.1` `NO` cases in the existing tests) continue to map to `"NO"`. `threshold` is added to `sig()`'s signature alongside `cutoff`; the function remains wired into `Tcode.collect()` at its default arguments, so builds are unaffected unless a caller overrides the new bound.

### Added
- One regression test in `test_lib.py` (`test_sig_marks_inconclusive_band`) asserting all four bands in a single frame: a value at/below cutoff (`YES`), a value in the inconclusive range (`INCONCLUSIVE`), and a value at the exclusive threshold (`NO`).
