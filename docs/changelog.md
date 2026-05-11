# Changelog

The canonical release history lives in the repository root at [`CHANGELOG.md`](https://github.com/SkyeAv/Tablassert/blob/main/CHANGELOG.md).

## Current Release Notes

## 7.4.7 - 2026-05-11

### Changes

- Added an `is_valid_pmc_id` model validator on `Provenance` that enforces `publication` starts with `PMC` followed by digits when `repo` is `PMC`. The constraint was previously documented in 7.3.6 but only now enforced at parse time.

For older releases and the full project history, open the root `CHANGELOG.md` in the repository.
