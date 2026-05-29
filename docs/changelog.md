# Changelog

The canonical release history lives in the repository root at [`CHANGELOG.md`](https://github.com/SkyeAv/Tablassert/blob/main/CHANGELOG.md).

## Current Release Notes

## 7.4.10 - 2026-05-29

### Changes

- Enforced explicit `biolink:` namespace prefix on predicates and qualifiers emitted by `compile_subgraph()` in `lib.py`. Both `self.statement.predicate` and `x.qualifier` (in the qualifier loop) are now prefixed via `add("biolink:", ...)`, ensuring all output edges carry fully-qualified Biolink CURIEs rather than bare predicate/qualifier names.
