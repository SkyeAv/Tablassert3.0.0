"""In-repo benchmark harness for the Rust `rs.dedup_ndjson` dedup passes.

WHY: US-002 rewrites the merge fold for speed and needs reproducible before/after
numbers without relying on throwaway /tmp scripts. This harness regenerates the four
merge-mode benchmark corpora -- byte-identical to the original `/tmp/merge_bench`
generators (`gen.py` / `gen_dup.py`, seed 42) -- and times `rs.dedup_ndjson` in merge
mode. US-006 adds scenarios E and N so the two OTHER passes that share the rewritten
canonical serializer are measured too (the audit found the default `error` mode and the
node path had no before/after number at all).

Usage (opt-in; collected nowhere by default, so it never slows the suite):

    TABLASSERT_BENCH=1 uv run pytest tests/bench_merge_bench.py -s -n 0 -q

`-n 0` keeps a single process (the datasets are session-scoped; xdist would rebuild
them in every worker), `-s` shows the timings. Generation writes ~1.7 GB of NDJSON
into pytest's basetemp and takes a couple of minutes before any timing starts.

`TABLASSERT_BENCH_ONLY=<comma-separated scenario names>` restricts which corpora get
generated AND timed -- the baseline half of a before/after comparison only needs the
scenarios under test, and skipping the rest saves ~1.1 GB of generation:

    TABLASSERT_BENCH=1 TABLASSERT_BENCH_ONLY=e,n uv run pytest tests/bench_merge_bench.py -s -n 0 -q

    scenario  pass / mode                            shape                                baseline (pre US-002)
    A         merge                                  200k triples x 3 rows, 20 cases      6.71s
    B         merge                                  5k triples x 20 rows, 100 cases      20.69s
    C         merge                                  1k triples x 100 rows, 50 cases      37.45s
    DUP       merge                                  100k byte-identical rows             0.59s
    E         default `error` mode, edges            598k lines (2% exact repeats)        see US-006 brief
    N         node dedup (`is_edges=False`)          600k lines (300k unique)             see US-006 brief
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("TABLASSERT_BENCH") != "1", reason="benchmark harness; opt in with TABLASSERT_BENCH=1")

_SEED = 42
_DOMAIN = "infores:multiomicskg"
_UUID_FIELDS = ["subject", "predicate", "object"]
# (n_triples, rows_per_triple, n_cases) -- exactly the shapes behind /tmp/merge_bench's
# edges_a/b/c.ndjson (verified byte-identical against the originals, seed 42).
_SCENARIOS: dict[str, tuple[int, int, int]] = {"a": (200_000, 3, 20), "b": (5_000, 20, 100), "c": (1_000, 100, 50)}
# Scenario E (default `error` mode). `source_record_urls` is unique per source row, so the
# three divergent rows of a triple derive three DISTINCT ids and the pass never aborts with
# `uuid-fields-not-a-key` -- the discriminating field that error's own message recommends.
_ERROR_UUID_FIELDS = ["subject", "predicate", "object", "source_record_urls"]
_ERROR_TRIPLES = 195_000
_ERROR_CASES = 20
# Re-emit every Nth triple's first row byte-identically, so duplicate suppression is real
# work (~2% of the stream) instead of the pass only ever seeing fresh ids.
_ERROR_DUPLICATE_EVERY = 15
# Scenario N (node dedup): `1 + node_i % 3` copies of each unique node -> 600k lines over
# 300k distinct records, so `record_if_new` suppresses half the stream by byte equality.
_NODE_COUNT = 300_000
_ALL_SCENARIOS: tuple[str, ...] = ("a", "b", "c", "dup", "e", "n")


def _edge(rng: random.Random, triple_i: int, row_j: int, n_cases: int) -> dict[str, object]:
    """One synthetic edge; verbatim port of /tmp/merge_bench/gen.py's `edge`."""
    return {
        "subject": f"MONDO:{triple_i:07d}",
        "predicate": "biolink:associated_with",
        "object": f"NCBIGene:{triple_i % 20000:07d}",
        "p_value": f"{rng.random() * 0.05:.3e}",
        "effect_size": round(rng.random(), 4),
        "supporting_text": f"Study row text variant {row_j} for association {triple_i}.",
        "publications": [f"PMCID:PMC{(triple_i * 7 + k) % 12_000_000:08d}" for k in range(row_j % 4 + 1)],
        "sources": [{"resource_id": "infores:multiomicskg", "resource_role": "primary_knowledge_source"}],
        "source_record_urls": [f"https://example.org/table/{triple_i}#row{row_j}"],
        "has_supporting_studies": [f"STUDY:{(triple_i + row_j) % 5000:05d}"],
        "number_of_cases": n_cases,
        "supporting_case_ids": [f"CASE:{triple_i:06d}:{row_j:03d}:{c:04d}" for c in range(n_cases)],
    }


def _generate_scenario(path: Path, n_triples: int, rows_per_triple: int, n_cases: int) -> int:
    """Reproduce /tmp/merge_bench/gen.py byte-for-byte (seed 42); return the line count."""
    rng = random.Random(_SEED)
    lines = 0
    with path.open("w") as handle:
        for triple_i in range(n_triples):
            for row_j in range(rows_per_triple):
                handle.write(json.dumps(_edge(rng, triple_i, row_j, n_cases if row_j else max(1, n_cases // 2))) + "\n")
                lines += 1
    return lines


def _generate_error_edges(path: Path) -> int:
    """Scenario E corpus: default-`error`-mode edges plus exact-duplicate rows.

    Reuses `_edge` verbatim (same seed, same 20-case shape as scenario A) so each line is
    the realistic ~700-byte record that `finalize_record` canonicalizes once per line on
    this pass -- the cost US-003 removed the clone from, and the reason this scenario
    exists. Returns the line count (598,000 = 195k x 3 + 13k repeats).
    """
    rng = random.Random(_SEED)
    lines = 0
    with path.open("w") as handle:
        for triple_i in range(_ERROR_TRIPLES):
            first_row = ""
            for row_j in range(3):
                line = json.dumps(_edge(rng, triple_i, row_j, _ERROR_CASES if row_j else max(1, _ERROR_CASES // 2))) + "\n"
                if row_j == 0:
                    first_row = line
                handle.write(line)
                lines += 1
            if triple_i % _ERROR_DUPLICATE_EVERY == 0:
                # Byte-identical repeat of a row already written: same derived id AND same
                # content hash -> `EdgeVerdict::Duplicate`, suppressed without aborting.
                handle.write(first_row)
                lines += 1
    return lines


def _node(rng: random.Random, node_i: int) -> dict[str, object]:
    """One synthetic KGX node, in the post-coercion shape the node pass actually sees.

    No empty/null-like values: `strip_nulls` would drop them and the corpus would no longer
    measure the bytes it claims to.
    """
    return {
        "id": f"MONDO:{node_i:07d}",
        "name": f"Disease term number {node_i}",
        "category": ["biolink:Disease" if node_i % 2 else "biolink:PhenotypicFeature"],
        "description": f"Synthetic node description variant {rng.randrange(97)} for term {node_i}.",
        "synonym": [f"alias {node_i} {k}" for k in range(1 + node_i % 3)],
        "xrefs": [f"UMLS:C{(node_i * 3) % 2_000_000:07d}", f"DOID:{node_i % 20_000}"],
        "provided_by": [_DOMAIN],
    }


def _generate_nodes(path: Path) -> int:
    """Scenario N corpus: node lines with heavy exact duplication (600k lines, 300k unique).

    Duplicates are written adjacent to their original, which is the common real shape
    (records arrive grouped by source table) and keeps generation streaming -- no corpus is
    ever held in memory. Returns the line count.
    """
    rng = random.Random(_SEED)
    lines = 0
    with path.open("w") as handle:
        for node_i in range(_NODE_COUNT):
            line = json.dumps(_node(rng, node_i)) + "\n"
            for _ in range(1 + node_i % 3):
                handle.write(line)
                lines += 1
    return lines


def _generate_duplicates(path: Path) -> int:
    """Reproduce /tmp/merge_bench/gen_dup.py byte-for-byte; return the line count."""
    row: dict[str, object] = {
        "subject": "MONDO:0000001",
        "predicate": "biolink:associated_with",
        "object": "NCBIGene:0000001",
        "p_value": "1e-5",
        "effect_size": 0.5,
        "supporting_text": "same",
        "publications": ["PMCID:PMC1"],
        "sources": [{"resource_id": "infores:x", "resource_role": "primary_knowledge_source"}],
        "number_of_cases": 50,
        "supporting_case_ids": [f"CASE:{c:04d}" for c in range(50)],
    }
    line = json.dumps(row) + "\n"
    with path.open("w") as handle:
        for _ in range(100_000):
            handle.write(line)
    return 100_000


def _selected_scenarios() -> tuple[str, ...]:
    """Scenario names to generate and time: `TABLASSERT_BENCH_ONLY`, or all six.

    Fails loudly on an unknown name -- a typo must not silently time nothing and report a
    green benchmark run.
    """
    raw = os.environ.get("TABLASSERT_BENCH_ONLY", "")
    if not raw.strip():
        return _ALL_SCENARIOS
    names = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    unknown = [name for name in names if name not in _ALL_SCENARIOS]
    if unknown:
        raise ValueError(f"TABLASSERT_BENCH_ONLY names unknown scenario(s) {unknown}; expected a subset of {list(_ALL_SCENARIOS)}")
    return names


@pytest.fixture(scope="session")
def bench_datasets(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple[Path, int]]:
    """Generate the selected corpora once per session (skipped entirely unless opted in).

    Each generator seeds its own `random.Random(_SEED)`, so a corpus is byte-identical no
    matter which subset was requested or which worktree generated it -- that is what lets a
    baseline checkout and the branch checkout be compared on the same input (verified by
    checksum in the US-006 run).
    """
    root = tmp_path_factory.mktemp("merge_bench")
    selected = _selected_scenarios()
    datasets: dict[str, tuple[Path, int]] = {}
    for name in _ALL_SCENARIOS:
        if name not in selected:
            continue
        if name in _SCENARIOS:
            path = root / f"edges_{name}.ndjson"
            datasets[name] = (path, _generate_scenario(path, *_SCENARIOS[name]))
        elif name == "dup":
            path = root / "edges_dup.ndjson"
            datasets[name] = (path, _generate_duplicates(path))
        elif name == "e":
            path = root / "edges_error_mode.ndjson"
            datasets[name] = (path, _generate_error_edges(path))
        else:  # "n"
            path = root / "nodes_dedup.ndjson"
            datasets[name] = (path, _generate_nodes(path))
    return datasets


def _time_dedup(
    name: str, datasets: dict[str, tuple[Path, int]], tmp_path: Path, *, is_edges: bool, uuid_fields: list[str] | None, on_collision: str | None
) -> None:
    """Time one `rs.dedup_ndjson` run on a generated corpus and report it.

    The returned pair is `(merged divergent records, conflicting scalar fields)` in merge
    mode and `(0, 0)` on the other two passes, so the mode is printed beside it to keep all
    six scenarios' lines comparable in one table. Output lines are counted OUTSIDE the timed
    region: they are the non-vacuity evidence that duplicates were really suppressed.
    """
    from tablassert import rs

    if name not in datasets:
        pytest.skip(f"scenario {name} was not generated (TABLASSERT_BENCH_ONLY={os.environ.get('TABLASSERT_BENCH_ONLY', '')!r})")
    p_in, line_count = datasets[name]
    p_out = tmp_path / f"deduped_{name}.ndjson"
    start = time.perf_counter()
    merged, conflicts = rs.dedup_ndjson(p_in, p_out, is_edges, _DOMAIN, uuid_fields, on_collision)
    elapsed = time.perf_counter() - start
    assert p_out.stat().st_size > 0, f"scenario {name} produced an empty output"
    with p_out.open("rb") as handle:
        written = sum(1 for _ in handle)
    mode = "nodes" if not is_edges else f"edges/{on_collision or 'error'}"
    print(
        f"\nbench {name}: mode={mode} lines={line_count:,} written={written:,} merged={merged:,} conflicts={conflicts:,} "
        f"elapsed={elapsed:.3f}s ({line_count / elapsed:,.0f} lines/s)"
    )


def test_bench_scenario_a(bench_datasets: dict[str, tuple[Path, int]], tmp_path: Path) -> None:
    """Scenario A: 200k triples x 3 rows, 20 cases. Baseline pre-US-002: 6.71s."""
    _time_dedup("a", bench_datasets, tmp_path, is_edges=True, uuid_fields=_UUID_FIELDS, on_collision="merge")


def test_bench_scenario_b(bench_datasets: dict[str, tuple[Path, int]], tmp_path: Path) -> None:
    """Scenario B: 5k triples x 20 rows, 100 cases. Baseline pre-US-002: 20.69s."""
    _time_dedup("b", bench_datasets, tmp_path, is_edges=True, uuid_fields=_UUID_FIELDS, on_collision="merge")


def test_bench_scenario_c(bench_datasets: dict[str, tuple[Path, int]], tmp_path: Path) -> None:
    """Scenario C: 1k triples x 100 rows, 50 cases. Baseline pre-US-002: 37.45s."""
    _time_dedup("c", bench_datasets, tmp_path, is_edges=True, uuid_fields=_UUID_FIELDS, on_collision="merge")


def test_bench_scenario_dup(bench_datasets: dict[str, tuple[Path, int]], tmp_path: Path) -> None:
    """Scenario DUP: 100k byte-identical rows (pure suppression path). Baseline: 0.59s."""
    _time_dedup("dup", bench_datasets, tmp_path, is_edges=True, uuid_fields=_UUID_FIELDS, on_collision="merge")


def test_bench_scenario_e(bench_datasets: dict[str, tuple[Path, int]], tmp_path: Path) -> None:
    """Scenario E: default `error` mode over 598k edge lines (13k of them exact repeats).

    WHY: US-003's allocation-free `canonical_json_bytes` is called once per line by
    `finalize_record`'s content hash on the DEFAULT mode too -- not just inside the merge
    fold -- so the rewrite lands on every production edge build. The merge-only A-DUP
    scenarios never measured that path, which is exactly the gap this scenario closes.
    """
    _time_dedup("e", bench_datasets, tmp_path, is_edges=True, uuid_fields=_ERROR_UUID_FIELDS, on_collision=None)


def test_bench_scenario_n(bench_datasets: dict[str, tuple[Path, int]], tmp_path: Path) -> None:
    """Scenario N: node dedup over 600k node lines (300k unique, so half are suppressed).

    WHY: the node pass is the other leg REQ-PERF-4/DoD-8 asks for. Measured, not assumed,
    it turns out NOT to reach `canonical_json_bytes` at all: `finalize_record` returns
    before the content hash when `is_edges` is false (`rust/src/ndjson.rs`) and
    `record_if_new` keys on `emitted_json_bytes`. So N is the regression guard for the pass
    US-003 leaves untouched and its before/after delta is expected to be run-to-run noise,
    while E is the leg that actually exercises the rewritten serializer.
    """
    _time_dedup("n", bench_datasets, tmp_path, is_edges=False, uuid_fields=None, on_collision=None)
