"""Tests for US-009: outer DETERMINISTIC supervisor + monotonic improve loop + checkpoint/resume.

The supervisor is PLAIN PYTHON over agentic decisions: the inner ``CodeAgent`` (driven OFFLINE by a
``FakeModel``) only produces the initial config, and the improve loop is deterministic
(``propose_config_edit`` -> ``build_and_audit`` -> accept IFF strictly better). Every test is offline
+ fast: a tiny REAL ``rs.build_fullmap_db`` redb (``brca1`` -> HGNC:1100, ``mapk1`` -> HGNC:6871), a
small text table, ``fetch_pmc_tables`` monkeypatched to return the fixture table (no network), and an
autouse fixture disabling HuggingFace telemetry so ``agent.run`` never blocks on the network. The whole
module skips cleanly when the ``[agent]`` extra is absent (``importorskip("smolagents")``).
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
import yaml

from tablassert import rs
from tablassert.agent import ConfigRecord, SupervisorState, load_state, make_fake_model, run_supervisor, save_state

pytest.importorskip("smolagents")


@pytest.fixture(autouse=True)
def _offline_no_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ``agent.run`` hermetically offline by disabling HuggingFace telemetry.

    Why: a real ``CodeAgent.run`` fires huggingface_hub telemetry that BLOCKS on a network call
    (observed: the run hangs indefinitely at zero CPU when telemetry is enabled). The supervisor
    calls ``agent.run`` once per pmc, so every test here needs the standard opt-out vars. Scoped to
    this file (NOT global conftest) so the QC suite's own HuggingFace model loading is unaffected.
    """
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("DO_NOT_TRACK", "1")


# --------------------------------------------------------------------------- #
# Offline fixtures: tiny REAL redb + small text tables + a valid Section config
# --------------------------------------------------------------------------- #


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def _synonym_row(curie: str, preferred_name: str, names: list[str], category: str) -> dict[str, Any]:
    return {"curie": curie, "preferred_name": preferred_name, "names": names, "types": [category], "taxa": ["NCBITaxon:9606"]}


def _class_row(curie: str, equivalents: list[str]) -> dict[str, Any]:
    return {"id": curie, "equivalent_identifiers": [{"identifier": x} for x in equivalents]}


@pytest.fixture
def fullmap_db(tmp_path: Path) -> Path:
    """A tiny REAL fullmap redb: ``brca1`` -> HGNC:1100, ``mapk1`` -> HGNC:6871."""
    root: Path = tmp_path / "fullmap"
    root.mkdir(parents=True, exist_ok=True)
    classes: Path = _write_jsonl(root / "classes.ndjson", [_class_row("HGNC:1100", ["NCBIGene:672"])])
    synonyms: Path = _write_jsonl(
        root / "synonyms.ndjson",
        [_synonym_row("HGNC:1100", "BRCA1", ["BRCA1", "brca1"], "Gene"), _synonym_row("HGNC:6871", "MAPK1", ["MAPK1", "mapk1"], "Gene")],
    )
    output: Path = root / "data" / "fullmap.redb"
    rs.build_fullmap_db(output, [classes], [synonyms], threads=2)
    return output


def _write_table(tmp_path: Path, name: str, text: str) -> Path:
    table: Path = tmp_path / name
    table.write_text(text)
    return table


def _column_cfg(table: Path) -> dict[str, Any]:
    """A valid merged Section config: subject=column A, object=column B, PMC provenance."""
    return {
        "source": {"kind": "text", "local": str(table), "url": "https://e.com/d.tsv", "delimiter": "\t"},
        "statement": {
            "subject": {"method": "column", "encoding": "A"},
            "predicate": "associated_with",
            "object": {"method": "column", "encoding": "B"},
        },
        "provenance": {"repo": "PMC", "publication": "PMC1"},
    }


def _patch_fetch(monkeypatch: pytest.MonkeyPatch, table: Path) -> list[str]:
    """Monkeypatch ``fetch_pmc_tables`` to return ``[table]`` and record the pmc ids requested."""
    calls: list[str] = []

    def fake_fetch(pmc_id: str, outdir: Path, *, timeout: int = 120) -> list[Path]:  # pyright: ignore[reportUnusedParameter]
        calls.append(pmc_id)
        return [table]

    monkeypatch.setattr("tablassert.agent.fetch_pmc_tables", fake_fetch)
    return calls


# --------------------------------------------------------------------------- #
# Supervisor behavior
# --------------------------------------------------------------------------- #


def test_supervisor_happy_path_mapped(tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One pmc whose FakeModel returns a fully-resolving config reaches MAPPED + checkpoints state."""
    table: Path = _write_table(tmp_path, "good.tsv", "brca1\tmapk1\nbrca1\tmapk1\n")
    _patch_fetch(monkeypatch, table)
    good_yaml: str = yaml.safe_dump(_column_cfg(table), sort_keys=False)
    state_dir: Path = tmp_path / "state"

    result = run_supervisor(
        ["PMC1"],
        fullmap=fullmap_db,
        build_model_factory=lambda: make_fake_model(final_yaml=good_yaml),
        map_threshold=0.8,
        state_dir=state_dir,
        workdir=tmp_path / "w",
    )

    records: dict[str, ConfigRecord] = result["records"]  # pyright: ignore[reportAssignmentType]
    rec: ConfigRecord = records["PMC1"]
    assert rec.status == "MAPPED"
    assert rec.coverage_history, "coverage_history should be non-empty"
    assert rec.coverage_history[-1] >= 0.8
    assert rec.best_config_path is not None
    assert Path(rec.best_config_path).is_file()
    assert (state_dir / "state.json").is_file()

    reloaded: SupervisorState | None = load_state(state_dir)
    assert reloaded is not None
    assert reloaded.records["PMC1"].status == "MAPPED"


def test_supervisor_improve_loop_accepts_better(tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A first config below threshold is genuinely improved by propose_config_edit and accepted.

    The table's subject column carries ``g__brca1`` (unresolved raw -> coverage 0.5); the proposer
    detects the ``g__`` lineage glue, adds a regex strip, and the edited config resolves -> 1.0. The
    improve loop must ACCEPT this strictly-better edit (history strictly increases) and reach MAPPED.
    """
    table: Path = _write_table(tmp_path, "glue.tsv", "g__brca1\tmapk1\ng__brca1\tmapk1\n")
    _patch_fetch(monkeypatch, table)
    first_yaml: str = yaml.safe_dump(_column_cfg(table), sort_keys=False)  # no regex -> coverage 0.5

    result = run_supervisor(
        ["PMC1"],
        fullmap=fullmap_db,
        build_model_factory=lambda: make_fake_model(final_yaml=first_yaml),
        map_threshold=1.0,
        max_improve_iters=3,
        state_dir=tmp_path / "state",
        workdir=tmp_path / "w",
    )

    rec: ConfigRecord = result["records"]["PMC1"]  # pyright: ignore[reportIndexIssue]
    assert rec.status == "MAPPED"
    assert len(rec.coverage_history) >= 2, "the improve loop should have accepted at least one edit"
    assert rec.coverage_history[0] < 1.0, "the initial config is below threshold"
    assert rec.coverage_history[-1] >= 1.0, "an accepted edit lifts coverage to the threshold"
    assert rec.last_edits, "the accepted edit records a rationale"
    assert rec.best_coverage == max(rec.coverage_history)


def test_supervisor_monotonic_no_regression(tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """coverage_history is non-decreasing: an edit is recorded as best IFF strictly better."""
    table: Path = _write_table(tmp_path, "glue.tsv", "g__brca1\tmapk1\ng__brca1\tmapk1\n")
    _patch_fetch(monkeypatch, table)
    first_yaml: str = yaml.safe_dump(_column_cfg(table), sort_keys=False)

    result = run_supervisor(
        ["PMC1"],
        fullmap=fullmap_db,
        build_model_factory=lambda: make_fake_model(final_yaml=first_yaml),
        map_threshold=1.0,
        max_improve_iters=3,
        state_dir=tmp_path / "state",
        workdir=tmp_path / "w",
    )

    records: dict[str, ConfigRecord] = result["records"]  # pyright: ignore[reportAssignmentType]
    history: list[float] = records["PMC1"].coverage_history
    assert history, "coverage_history should be non-empty"
    assert all(later >= earlier for earlier, later in pairwise(history)), f"history regressed: {history}"


def test_supervisor_budget_exhaustion_skipped(tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unimprovable config below an unreachable threshold with no budget -> SKIPPED with a reason."""
    table: Path = _write_table(tmp_path, "bad.tsv", "brca1\tzzznotreal\nbrca1\tzzznotreal\n")  # object unresolved -> 0.5
    _patch_fetch(monkeypatch, table)
    bad_yaml: str = yaml.safe_dump(_column_cfg(table), sort_keys=False)

    result = run_supervisor(
        ["PMC1"],
        fullmap=fullmap_db,
        build_model_factory=lambda: make_fake_model(final_yaml=bad_yaml),
        map_threshold=1.0,
        max_improve_iters=0,  # no improve budget
        state_dir=tmp_path / "state",
        workdir=tmp_path / "w",
    )

    rec: ConfigRecord = result["records"]["PMC1"]  # pyright: ignore[reportIndexIssue]
    assert rec.status == "SKIPPED"
    assert rec.notes.startswith("SKIPPED")
    assert rec.best_coverage < 1.0


def test_supervisor_batch_isolation(tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One pmc whose fetch raises is SKIPPED; the other still maps. No exception escapes the batch."""
    table: Path = _write_table(tmp_path, "good.tsv", "brca1\tmapk1\nbrca1\tmapk1\n")
    good_yaml: str = yaml.safe_dump(_column_cfg(table), sort_keys=False)

    def fake_fetch(pmc_id: str, outdir: Path, *, timeout: int = 120) -> list[Path]:  # pyright: ignore[reportUnusedParameter]
        if pmc_id == "PMCBAD":
            raise FileNotFoundError("no supplementary tables for PMCBAD")
        return [table]

    monkeypatch.setattr("tablassert.agent.fetch_pmc_tables", fake_fetch)

    result = run_supervisor(
        ["PMCGOOD", "PMCBAD"],
        fullmap=fullmap_db,
        build_model_factory=lambda: make_fake_model(final_yaml=good_yaml),
        map_threshold=0.8,
        state_dir=tmp_path / "state",
        workdir=tmp_path / "w",
    )

    records: dict[str, ConfigRecord] = result["records"]  # pyright: ignore[reportAssignmentType]
    assert set(records) == {"PMCGOOD", "PMCBAD"}
    assert records["PMCGOOD"].status == "MAPPED"
    assert records["PMCBAD"].status == "SKIPPED"
    assert records["PMCBAD"].notes.startswith("SKIPPED")
    metrics: dict[str, object] = result["metrics"]  # pyright: ignore[reportAssignmentType]
    assert metrics["mapped"] == 1
    assert metrics["skipped"] == 1


def test_supervisor_resume_skips_done(tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second run over [A,B] with the same state_dir skips the already-MAPPED A and processes B."""
    table: Path = _write_table(tmp_path, "good.tsv", "brca1\tmapk1\nbrca1\tmapk1\n")
    calls: list[str] = _patch_fetch(monkeypatch, table)
    good_yaml: str = yaml.safe_dump(_column_cfg(table), sort_keys=False)
    state_dir: Path = tmp_path / "state"

    def factory() -> object:
        return make_fake_model(final_yaml=good_yaml)

    first = run_supervisor(["PMCA"], fullmap=fullmap_db, build_model_factory=factory, map_threshold=0.8, state_dir=state_dir, workdir=tmp_path / "w")
    assert first["records"]["PMCA"].status == "MAPPED"  # pyright: ignore[reportIndexIssue]
    first_attempts: int = first["records"]["PMCA"].attempts  # pyright: ignore[reportIndexIssue]

    second = run_supervisor(
        ["PMCA", "PMCB"], fullmap=fullmap_db, build_model_factory=factory, map_threshold=0.8, state_dir=state_dir, workdir=tmp_path / "w"
    )
    records: dict[str, ConfigRecord] = second["records"]  # pyright: ignore[reportAssignmentType]
    assert records["PMCA"].status == "MAPPED"
    assert records["PMCA"].attempts == first_attempts, "PMCA must NOT be reprocessed on resume"
    assert records["PMCB"].status == "MAPPED"
    assert calls.count("PMCA") == 1, "fetch must not be called again for the already-MAPPED PMCA"
    assert calls.count("PMCB") == 1


def test_state_roundtrip_atomic(tmp_path: Path) -> None:
    """save_state -> load_state round-trips exactly, and the atomic write leaves no .tmp behind."""
    state_dir: Path = tmp_path / "state"
    original: SupervisorState = SupervisorState(
        pmc_ids=["PMC1", "PMC2"],
        records={
            "PMC1": ConfigRecord(pmc_id="PMC1", status="MAPPED", coverage_history=[0.5, 1.0], best_coverage=1.0, attempts=2, last_edits="edit"),
            "PMC2": ConfigRecord(pmc_id="PMC2", status="SKIPPED", notes="SKIPPED: budget", qc_pass_rate=None),
        },
        metrics={"mapped": 1, "skipped": 1, "mean_best_coverage": 0.5},
    )

    save_state(state_dir, original)
    assert (state_dir / "state.json").is_file()
    assert not (state_dir / "state.json.tmp").exists(), "the atomic write must not leave a .tmp behind"

    loaded: SupervisorState | None = load_state(state_dir)
    assert loaded is not None
    assert loaded.pmc_ids == original.pmc_ids
    assert loaded.records["PMC1"].status == "MAPPED"
    assert loaded.records["PMC1"].coverage_history == [0.5, 1.0]
    assert loaded.records["PMC1"].best_coverage == 1.0
    assert loaded.records["PMC1"].attempts == 2
    assert loaded.records["PMC2"].status == "SKIPPED"
    assert loaded.records["PMC2"].qc_pass_rate is None
    assert loaded.metrics["mapped"] == 1
