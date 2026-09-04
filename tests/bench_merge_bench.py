"""In-repo benchmark harness for the `uuid_on_collision: merge` edge pass.

WHY: US-002 rewrites the merge fold for speed and needs reproducible before/after
numbers without relying on throwaway /tmp scripts. This harness regenerates the four
benchmark corpora -- byte-identical to the original `/tmp/merge_bench` generators
(`gen.py` / `gen_dup.py`, seed 42) -- and times `rs.dedup_ndjson` in merge mode.

Usage (opt-in; collected nowhere by default, so it never slows the suite):

    TABLASSERT_BENCH=1 uv run pytest tests/bench_merge_bench.py -s -n 0 -q

`-n 0` keeps a single process (the datasets are session-scoped; xdist would rebuild
them in every worker), `-s` shows the timings. Generation writes ~1.1 GB of NDJSON
into pytest's basetemp and takes roughly a minute before any timing starts.

    scenario  shape                                baseline (pre US-002)
    A         200k triples x 3 rows, 20 cases      6.71s
    B         5k triples x 20 rows, 100 cases      20.69s
    C         1k triples x 100 rows, 50 cases      37.45s
    DUP       100k byte-identical rows             0.59s
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


@pytest.fixture(scope="session")
def bench_datasets(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple[Path, int]]:
    """Generate all four corpora once per session (skipped entirely unless opted in)."""
    root = tmp_path_factory.mktemp("merge_bench")
    datasets: dict[str, tuple[Path, int]] = {}
    for name, (n_triples, rows_per_triple, n_cases) in _SCENARIOS.items():
        path = root / f"edges_{name}.ndjson"
        datasets[name] = (path, _generate_scenario(path, n_triples, rows_per_triple, n_cases))
    dup_path = root / "edges_dup.ndjson"
    datasets["dup"] = (dup_path, _generate_duplicates(dup_path))
    return datasets


def _time_merge(name: str, datasets: dict[str, tuple[Path, int]], tmp_path: Path) -> None:
    """Time one `rs.dedup_ndjson` merge-mode run on a generated corpus and report it."""
    from tablassert import rs

    p_in, line_count = datasets[name]
    p_out = tmp_path / f"merged_{name}.ndjson"
    start = time.perf_counter()
    merged, conflicts = rs.dedup_ndjson(p_in, p_out, True, _DOMAIN, _UUID_FIELDS, "merge")
    elapsed = time.perf_counter() - start
    assert p_out.stat().st_size > 0, f"scenario {name} produced an empty merge output"
    print(
        f"\nbench {name}: lines={line_count:,} merged={merged:,} conflicts={conflicts:,} elapsed={elapsed:.3f}s ({line_count / elapsed:,.0f} lines/s)"
    )


def test_bench_scenario_a(bench_datasets: dict[str, tuple[Path, int]], tmp_path: Path) -> None:
    """Scenario A: 200k triples x 3 rows, 20 cases. Baseline pre-US-002: 6.71s."""
    _time_merge("a", bench_datasets, tmp_path)


def test_bench_scenario_b(bench_datasets: dict[str, tuple[Path, int]], tmp_path: Path) -> None:
    """Scenario B: 5k triples x 20 rows, 100 cases. Baseline pre-US-002: 20.69s."""
    _time_merge("b", bench_datasets, tmp_path)


def test_bench_scenario_c(bench_datasets: dict[str, tuple[Path, int]], tmp_path: Path) -> None:
    """Scenario C: 1k triples x 100 rows, 50 cases. Baseline pre-US-002: 37.45s."""
    _time_merge("c", bench_datasets, tmp_path)


def test_bench_scenario_dup(bench_datasets: dict[str, tuple[Path, int]], tmp_path: Path) -> None:
    """Scenario DUP: 100k byte-identical rows (pure suppression path). Baseline: 0.59s."""
    _time_merge("dup", bench_datasets, tmp_path)
