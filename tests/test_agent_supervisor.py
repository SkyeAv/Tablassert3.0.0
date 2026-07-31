"""Tests for US-009: outer DETERMINISTIC supervisor + monotonic improve loop + checkpoint/resume.

The supervisor is PLAIN PYTHON over agentic decisions: the inner ``CodeAgent`` (driven OFFLINE by a
``FakeModel``) only produces the initial config, and the improve loop is deterministic
(``propose_config_edit`` -> ``build_and_audit`` -> accept IFF strictly better). Every test is offline
+ fast: a tiny REAL ``rs.build_fullmap_db`` redb (``brca1`` -> HGNC:1100, ``mapk1`` -> HGNC:6871), a
small text table, ``fetch_pmc_article`` monkeypatched to return the fixture table (no network), and an
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
    """Monkeypatch ``fetch_pmc_article`` to return ``[table]`` and record the pmc ids requested."""
    calls: list[str] = []

    def fake_fetch(pmc_id: str, outdir: Path, *, timeout: int = 120) -> list[Path]:  # pyright: ignore[reportUnusedParameter]
        calls.append(pmc_id)
        return [table]

    monkeypatch.setattr("tablassert.agent.fetch_pmc_article", fake_fetch)
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

    monkeypatch.setattr("tablassert.agent.fetch_pmc_article", fake_fetch)

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


def test_supervisor_fetch_no_table_skipped(tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fetch that fails fast with 'No supplementary tables' marks the record SKIPPED (batch advances)."""
    import tablassert.agent as agent_mod

    def fake_fetch(pmc_id: str, outdir: Path, *, timeout: int = 120) -> list[Path]:  # pyright: ignore[reportUnusedParameter]
        raise FileNotFoundError("No supplementary tables found for PMC1.")

    monkeypatch.setattr(agent_mod, "fetch_pmc_article", fake_fetch)
    result = run_supervisor(
        ["PMC1"],
        fullmap=fullmap_db,
        build_model_factory=lambda: make_fake_model(),
        map_threshold=0.8,
        state_dir=tmp_path / "state",
        workdir=tmp_path / "w",
    )
    rec: ConfigRecord = result["records"]["PMC1"]  # pyright: ignore[reportIndexIssue]
    assert rec.status == "SKIPPED"
    assert "No supplementary tables" in rec.notes


def test_supervisor_task_lists_all_tables_and_main_text(tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The agent task lists EVERY candidate table, wires the main-text path, and mentions source.sheet."""
    import tablassert.agent as agent_mod

    t1: Path = _write_table(tmp_path, "s1.tsv", "brca1\tmapk1\n")
    t2: Path = _write_table(tmp_path, "s2.tsv", "brca1\tmapk1\n")
    xml: Path = tmp_path / "PMC1.1.xml"
    xml.write_text("<article/>")
    good_yaml: str = yaml.safe_dump(_column_cfg(t1), sort_keys=False)

    def fake_fetch(pmc_id: str, outdir: Path, *, timeout: int = 120) -> list[Path]:  # pyright: ignore[reportUnusedParameter]
        return [xml, t1, t2]

    monkeypatch.setattr(agent_mod, "fetch_pmc_article", fake_fetch)

    captured: dict[str, str] = {}
    real_build_agent = agent_mod.build_agent

    def spy_build_agent(*args: object, **kwargs: object) -> object:
        agent = real_build_agent(*args, **kwargs)

        class _Spy:
            def run(self, task: str) -> object:
                captured["task"] = task
                return agent.run(task)  # pyright: ignore[reportAttributeAccessIssue]

        return _Spy()

    monkeypatch.setattr(agent_mod, "build_agent", spy_build_agent)

    result = run_supervisor(
        ["PMC1"],
        fullmap=fullmap_db,
        build_model_factory=lambda: make_fake_model(final_yaml=good_yaml),
        map_threshold=0.8,
        state_dir=tmp_path / "state",
        workdir=tmp_path / "w",
    )
    task: str = captured["task"]
    assert "pmc_article_context" in task  # main text wired in
    assert str(t1) in task  # ALL candidate tables listed
    assert str(t2) in task
    assert "source.sheet" in task  # sheet guidance present
    assert result["records"]["PMC1"].status == "MAPPED"  # pyright: ignore[reportIndexIssue]


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


def test_supervisor_breaks_after_rejected_edit(tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression (review fix 3): the improve loop BREAKS once a deterministic edit is rejected.

    propose_config_edit + build_and_audit are DETERMINISTIC, so after a rejected edit (current_config
    unchanged) every further iteration would propose the IDENTICAL edit and reject again, burning up to
    max_improve_iters full real builds with no possible progress. The loop now breaks on rejection. With
    max_improve_iters=5 and an always-rejected edit, build_and_audit runs exactly twice (the initial build
    + ONE rejected improve), not 1 + 5 = 6.
    """
    import tablassert.agent as agent_mod

    table: Path = _write_table(tmp_path, "d.tsv", "brca1\tmapk1\n")
    _patch_fetch(monkeypatch, table)
    good_yaml: str = yaml.safe_dump(_column_cfg(table))
    calls: dict[str, int] = {"build": 0}

    def fake_build(
        config_yaml: str,
        *,
        fullmap: Path,
        name: str = "agent",
        version: str = "0.0.1",
        qc: bool = False,
        head: bool = False,
        workdir: Path | None = None,
    ) -> dict[str, Any]:  # pyright: ignore[reportUnusedParameter]
        calls["build"] += 1
        return {
            "ok": True,
            "coverage_pct": 0.5,
            "qc_pass_rate": None,
            "errors": [],
            "error_codes": [],
            "kgx_path": None,
            "edges_path": None,
            "node_count": 1,
            "edge_count": 1,
            "unresolved": ["x"],
        }

    monkeypatch.setattr(agent_mod, "build_and_audit", fake_build)
    monkeypatch.setattr(
        agent_mod,
        "map_coverage",
        lambda *a, **k: {
            "overall": 0.5,
            "measured": True,
            "per_column": {"subject": {"coverage": 0.5, "total": 1, "resolved": 0, "unresolved": ["g__x"], "method": "column"}},
            "unresolved": ["g__x"],
        },
    )
    monkeypatch.setattr(agent_mod, "propose_config_edit", lambda cfg, rep: (good_yaml, "proposed edit"))

    result: dict[str, Any] = run_supervisor(
        ["PMC1"],
        fullmap=fullmap_db,
        build_model_factory=lambda: make_fake_model(final_yaml=good_yaml),
        map_threshold=0.8,
        max_improve_iters=5,
        state_dir=tmp_path / "state",
        workdir=tmp_path / "w",
    )
    assert calls["build"] == 2  # initial build + ONE rejected improve, then break (not 1 + 5)
    assert result["records"]["PMC1"].status == "SKIPPED"  # 0.5 < 0.8 and never improved


def test_supervisor_built_unmeasured_is_non_failure(tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """W5: a config that BUILDS (ok=True) but whose coverage is UNMEASURABLE -> BUILT_UNMEASURED.

    ``build_and_audit`` is stubbed to report a successful build with ``measured=False`` (e.g. an
    unreproducible source frame). The supervisor must record the TERMINAL non-failure BUILT_UNMEASURED,
    write the best config, and NOT count it as a SKIPPED failure; the metrics report ``built_unmeasured``.
    """
    import tablassert.agent as agent_mod

    table: Path = _write_table(tmp_path, "d.tsv", "brca1\tmapk1\n")
    _patch_fetch(monkeypatch, table)
    good_yaml: str = yaml.safe_dump(_column_cfg(table))

    def fake_build(
        config_yaml: str,
        *,
        fullmap: Path,
        name: str = "agent",
        version: str = "0.0.1",
        qc: bool = False,
        head: bool = False,
        workdir: Path | None = None,
    ) -> dict[str, Any]:  # pyright: ignore[reportUnusedParameter]
        return {
            "ok": True,
            "coverage_pct": 0.0,
            "measured": False,
            "qc_pass_rate": None,
            "errors": ["coverage unmeasurable: could not reproduce the source frame (treated as 0.0, not a perfect score)"],
            "error_codes": [],
            "kgx_path": None,
            "edges_path": None,
            "node_count": 1,
            "edge_count": 1,
            "unresolved": [],
        }

    monkeypatch.setattr(agent_mod, "build_and_audit", fake_build)
    monkeypatch.setattr(agent_mod, "map_coverage", lambda *a, **k: {"overall": 0.0, "measured": False, "per_column": {}, "unresolved": []})
    monkeypatch.setattr(agent_mod, "propose_config_edit", lambda cfg, rep: (good_yaml, "no safe edit"))

    result: dict[str, Any] = run_supervisor(
        ["PMC1"],
        fullmap=fullmap_db,
        build_model_factory=lambda: make_fake_model(final_yaml=good_yaml),
        map_threshold=0.8,
        max_improve_iters=3,
        state_dir=tmp_path / "state",
        workdir=tmp_path / "w",
    )
    rec: ConfigRecord = result["records"]["PMC1"]  # pyright: ignore[reportIndexIssue]
    assert rec.status == "BUILT_UNMEASURED"  # NOT SKIPPED, NOT MAPPED
    assert rec.notes.startswith("BUILT_UNMEASURED")
    assert rec.best_config_path is not None
    assert Path(rec.best_config_path).is_file()  # the best config is still written
    metrics: dict[str, object] = result["metrics"]  # pyright: ignore[reportAssignmentType]
    assert metrics["built_unmeasured"] == 1
    assert metrics["mapped"] == 0
    assert metrics["skipped"] == 0


def test_supervisor_resume_skips_built_unmeasured(tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """W5: BUILT_UNMEASURED is TERMINAL — a resume over the same id does not reprocess it (no re-fetch)."""
    import tablassert.agent as agent_mod

    table: Path = _write_table(tmp_path, "d.tsv", "brca1\tmapk1\n")
    calls: list[str] = _patch_fetch(monkeypatch, table)
    good_yaml: str = yaml.safe_dump(_column_cfg(table))
    state_dir: Path = tmp_path / "state"

    monkeypatch.setattr(
        agent_mod,
        "build_and_audit",
        lambda *a, **k: {
            "ok": True,
            "coverage_pct": 0.0,
            "measured": False,
            "qc_pass_rate": None,
            "errors": [],
            "error_codes": [],
            "kgx_path": None,
            "edges_path": None,
            "node_count": 1,
            "edge_count": 1,
            "unresolved": [],
        },
    )
    monkeypatch.setattr(agent_mod, "map_coverage", lambda *a, **k: {"overall": 0.0, "measured": False, "per_column": {}, "unresolved": []})
    monkeypatch.setattr(agent_mod, "propose_config_edit", lambda cfg, rep: (good_yaml, "no safe edit"))

    def factory() -> object:
        return make_fake_model(final_yaml=good_yaml)

    first = run_supervisor(["PMC1"], fullmap=fullmap_db, build_model_factory=factory, map_threshold=0.8, state_dir=state_dir, workdir=tmp_path / "w")
    assert first["records"]["PMC1"].status == "BUILT_UNMEASURED"  # pyright: ignore[reportIndexIssue]

    second = run_supervisor(["PMC1"], fullmap=fullmap_db, build_model_factory=factory, map_threshold=0.8, state_dir=state_dir, workdir=tmp_path / "w")
    assert second["records"]["PMC1"].status == "BUILT_UNMEASURED"  # pyright: ignore[reportIndexIssue]
    assert calls.count("PMC1") == 1, "fetch must not be called again for the terminal BUILT_UNMEASURED record"


# --------------------------------------------------------------------------- #
# W1: tier-2 LLM reflexion + semantic judge gate
# --------------------------------------------------------------------------- #


def _judge_lines(score: int) -> str:
    """A pointwise judge response scoring every dimension ``score`` (0-3)."""
    dims: list[str] = [
        "schema_validity",
        "coverage_appropriateness",
        "qc_pass",
        "predicate_category_appropriateness",
        "provenance_completeness",
        "efficiency",
        "tool_call_cleanliness",
    ]
    return "\n".join(f"{d}: {score}" for d in dims)


def _patch_build_sequence(monkeypatch: pytest.MonkeyPatch, coverages: list[float]) -> None:
    """Stub ``build_and_audit`` to return successive ``coverage_pct`` values (fast; no real build).

    Used by the gate/reflexion wiring tests where build FIDELITY is not the point. Once ``coverages`` is
    exhausted the last value repeats. ``measured`` is True and ``ok`` True so the MAPPED/gate path runs.
    """
    import tablassert.agent as agent_mod

    idx: dict[str, int] = {"i": 0}

    def fake_build(
        config_yaml: str,
        *,
        fullmap: Path,
        name: str = "agent",
        version: str = "0.0.1",
        qc: bool = False,
        head: bool = False,
        workdir: Path | None = None,
    ) -> dict[str, Any]:  # pyright: ignore[reportUnusedParameter]
        cov: float = coverages[idx["i"]] if idx["i"] < len(coverages) else coverages[-1]
        idx["i"] += 1
        return {
            "ok": True,
            "coverage_pct": cov,
            "measured": True,
            "qc_pass_rate": None,
            "errors": [],
            "error_codes": [],
            "kgx_path": None,
            "edges_path": None,
            "node_count": 1,
            "edge_count": 1,
            "unresolved": [],
        }

    monkeypatch.setattr(agent_mod, "build_and_audit", fake_build)


def test_supervisor_tier2_reflexion_on_stall(tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When tier-1 deterministic candidates stall, tier-2 LLM reflexion supplies a distinct, better config.

    The object column is unresolvable (coverage 0.5) and the deterministic proposer is stubbed to yield
    nothing; the reflexion model returns a config that makes the object a fixed literal (vacuous -> 1.0),
    which the supervisor accepts -> MAPPED, with the tier-2 rationale recorded.
    """
    import tablassert.agent as agent_mod

    table: Path = _write_table(tmp_path, "d.tsv", "brca1\tzzznotreal\nbrca1\tzzznotreal\n")
    _patch_fetch(monkeypatch, table)
    first_yaml: str = yaml.safe_dump(_column_cfg(table))

    monkeypatch.setattr(agent_mod, "propose_config_candidates", lambda cfg, rep: [])  # tier 1 stalls
    # Builds: initial 0.5, then the tier-2 head + full builds both 1.0 (the reflexion fix).
    _patch_build_sequence(monkeypatch, [0.5, 1.0, 1.0])

    fixed: dict[str, Any] = _column_cfg(table)
    fixed["statement"]["object"] = {"method": "value", "encoding": "CHEBI:41774"}  # vacuous -> coverage 1.0
    fixed_yaml: str = yaml.safe_dump(fixed, sort_keys=False)

    result = run_supervisor(
        ["PMC1"],
        fullmap=fullmap_db,
        build_model_factory=lambda: make_fake_model(final_yaml=first_yaml),
        map_threshold=1.0,
        max_improve_iters=2,
        state_dir=tmp_path / "state",
        workdir=tmp_path / "w",
        reflexion_model_factory=lambda: lambda prompt: fixed_yaml,
    )
    rec: ConfigRecord = result["records"]["PMC1"]  # pyright: ignore[reportIndexIssue]
    assert rec.status == "MAPPED"
    assert rec.last_edits == "tier-2 LLM reflexion edit"
    assert rec.coverage_history[-1] >= 1.0


@pytest.mark.parametrize(
    "full_overrides",
    [
        {"coverage_pct": 0.2},  # full build scored LOWER than the prior best (0.3)
        {"ok": False, "coverage_pct": 0.9},  # full build FAILED despite an optimistic head score
    ],
    ids=["lower-coverage", "failed-build"],
)
def test_supervisor_improve_rejects_unconfirmed_full_build(
    full_overrides: dict[str, Any], tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CodeRabbit: a confirming full build that fails or scores <= the prior best is NOT committed.

    The 5-row head sample optimistically scores 0.8 > 0.3, but the subsequent full build either scores
    lower (0.2) or fails (ok=False). The supervisor must reject it: coverage_history stays monotonic
    ([0.3]), best_coverage is preserved, and the article is SKIPPED rather than regressed.
    """
    import tablassert.agent as agent_mod

    table: Path = _write_table(tmp_path, "d.tsv", "brca1\tmapk1\n")
    _patch_fetch(monkeypatch, table)
    good_yaml: str = yaml.safe_dump(_column_cfg(table))

    monkeypatch.setattr(agent_mod, "propose_config_candidates", lambda cfg, rep: [(good_yaml, "edit")])
    monkeypatch.setattr(agent_mod, "map_coverage", lambda *a, **k: {"overall": 0.3, "measured": True, "per_column": {}, "unresolved": []})

    base: dict[str, Any] = {
        "ok": True,
        "coverage_pct": 0.0,
        "measured": True,
        "qc_pass_rate": None,
        "errors": [],
        "error_codes": [],
        "kgx_path": None,
        "edges_path": None,
        "node_count": 1,
        "edge_count": 1,
        "unresolved": [],
    }
    reports: list[dict[str, Any]] = [{**base, "coverage_pct": 0.3}, {**base, "coverage_pct": 0.8}, {**base, **full_overrides}]
    idx: dict[str, int] = {"i": 0}

    def fake_build(
        config_yaml: str,
        *,
        fullmap: Path,
        name: str = "agent",
        version: str = "0.0.1",
        qc: bool = False,
        head: bool = False,
        workdir: Path | None = None,
    ) -> dict[str, Any]:  # pyright: ignore[reportUnusedParameter]
        report: dict[str, Any] = reports[idx["i"]] if idx["i"] < len(reports) else reports[-1]
        idx["i"] += 1
        return report

    monkeypatch.setattr(agent_mod, "build_and_audit", fake_build)

    result = run_supervisor(
        ["PMC1"],
        fullmap=fullmap_db,
        build_model_factory=lambda: make_fake_model(final_yaml=good_yaml),
        map_threshold=0.9,
        max_improve_iters=3,
        state_dir=tmp_path / "state",
        workdir=tmp_path / "w",
    )
    rec: ConfigRecord = result["records"]["PMC1"]  # pyright: ignore[reportIndexIssue]
    assert rec.coverage_history == [0.3]  # the rejected full build is NOT appended -> monotonic
    assert rec.best_coverage == 0.3
    assert rec.status == "SKIPPED"


def test_supervisor_tier2_rejects_unconfirmed_full_build(tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CodeRabbit: a tier-2 reflexion full build that does not beat the prior best is NOT committed.

    Tier 1 stalls (no candidates); tier 2 proposes a config whose head sample scores 0.8 > 0.3 but whose
    full build regresses to 0.2. The supervisor must reject it: coverage_history stays [0.3], best_coverage
    is preserved, the tier-2 rationale is NOT recorded, and the article is SKIPPED.
    """
    import tablassert.agent as agent_mod

    table: Path = _write_table(tmp_path, "d.tsv", "brca1\tmapk1\n")
    _patch_fetch(monkeypatch, table)
    good_yaml: str = yaml.safe_dump(_column_cfg(table))

    monkeypatch.setattr(agent_mod, "propose_config_candidates", lambda cfg, rep: [])  # tier 1 stalls
    monkeypatch.setattr(agent_mod, "llm_propose_config_edit", lambda cfg, rep, task, model=None: good_yaml)  # tier 2 proposes
    monkeypatch.setattr(agent_mod, "map_coverage", lambda *a, **k: {"overall": 0.3, "measured": True, "per_column": {}, "unresolved": []})
    _patch_build_sequence(monkeypatch, [0.3, 0.8, 0.2])  # initial 0.3, tier-2 head 0.8, tier-2 full 0.2 (regresses)

    result = run_supervisor(
        ["PMC1"],
        fullmap=fullmap_db,
        build_model_factory=lambda: make_fake_model(final_yaml=good_yaml),
        map_threshold=0.9,
        max_improve_iters=2,
        state_dir=tmp_path / "state",
        workdir=tmp_path / "w",
        reflexion_model_factory=lambda: lambda prompt: good_yaml,
    )
    rec: ConfigRecord = result["records"]["PMC1"]  # pyright: ignore[reportIndexIssue]
    assert rec.coverage_history == [0.3]  # the rejected tier-2 full build is NOT appended -> monotonic
    assert rec.best_coverage == 0.3
    assert rec.last_edits != "tier-2 LLM reflexion edit"  # the rejected edit is NOT recorded
    assert rec.status == "SKIPPED"


def test_supervisor_semantic_gate_blocks_low_score(tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Coverage reaches threshold but a configured judge scores below judge_threshold -> SKIPPED (semantic gate)."""
    table: Path = _write_table(tmp_path, "good.tsv", "brca1\tmapk1\nbrca1\tmapk1\n")
    _patch_fetch(monkeypatch, table)
    good_yaml: str = yaml.safe_dump(_column_cfg(table))  # coverage 1.0
    _patch_build_sequence(monkeypatch, [1.0])

    result = run_supervisor(
        ["PMC1"],
        fullmap=fullmap_db,
        build_model_factory=lambda: make_fake_model(final_yaml=good_yaml),
        map_threshold=0.8,
        state_dir=tmp_path / "state",
        workdir=tmp_path / "w",
        judge_model=lambda prompt: _judge_lines(1),  # normalized ~0.33
        judge_threshold=0.9,
    )
    rec: ConfigRecord = result["records"]["PMC1"]  # pyright: ignore[reportIndexIssue]
    assert rec.status == "SKIPPED"
    assert "semantic gate" in rec.notes


def test_supervisor_semantic_gate_passes_high_score(tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Coverage reaches threshold AND the judge score clears judge_threshold -> MAPPED."""
    table: Path = _write_table(tmp_path, "good.tsv", "brca1\tmapk1\nbrca1\tmapk1\n")
    _patch_fetch(monkeypatch, table)
    good_yaml: str = yaml.safe_dump(_column_cfg(table))
    _patch_build_sequence(monkeypatch, [1.0])

    result = run_supervisor(
        ["PMC1"],
        fullmap=fullmap_db,
        build_model_factory=lambda: make_fake_model(final_yaml=good_yaml),
        map_threshold=0.8,
        state_dir=tmp_path / "state",
        workdir=tmp_path / "w",
        judge_model=lambda prompt: _judge_lines(3),  # normalized 1.0
        judge_threshold=0.5,
    )
    assert result["records"]["PMC1"].status == "MAPPED"  # pyright: ignore[reportIndexIssue]


def test_supervisor_no_judge_coverage_only(tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a judge model, coverage ALONE gates MAPPED (the offline heuristic judge is advisory only)."""
    table: Path = _write_table(tmp_path, "good.tsv", "brca1\tmapk1\nbrca1\tmapk1\n")
    _patch_fetch(monkeypatch, table)
    good_yaml: str = yaml.safe_dump(_column_cfg(table))
    _patch_build_sequence(monkeypatch, [1.0])

    result = run_supervisor(
        ["PMC1"],
        fullmap=fullmap_db,
        build_model_factory=lambda: make_fake_model(final_yaml=good_yaml),
        map_threshold=0.8,
        state_dir=tmp_path / "state",
        workdir=tmp_path / "w",
    )  # no judge_model
    assert result["records"]["PMC1"].status == "MAPPED"  # pyright: ignore[reportIndexIssue]


# --------------------------------------------------------------------------- #
# W2: head intermediate builds + full final build; multi-iteration improve loop
# --------------------------------------------------------------------------- #


def test_supervisor_head_intermediate_full_final(tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Intermediate improve builds use head=True; the accepted config gets a FULL build (head=False)."""
    import tablassert.agent as agent_mod

    table: Path = _write_table(tmp_path, "glue.tsv", "g__brca1\tmapk1\ng__brca1\tmapk1\n")
    _patch_fetch(monkeypatch, table)
    first_yaml: str = yaml.safe_dump(_column_cfg(table))  # 0.5 -> full edit (regex strip) -> 1.0

    head_flags: list[bool] = []
    real_build = agent_mod.build_and_audit

    def spy_build(
        config_yaml: str,
        *,
        fullmap: Path,
        name: str = "agent",
        version: str = "0.0.1",
        qc: bool = False,
        head: bool = False,
        workdir: Path | None = None,
    ) -> dict[str, Any]:
        head_flags.append(head)
        return real_build(config_yaml, fullmap=fullmap, name=name, version=version, qc=qc, head=head, workdir=workdir)

    monkeypatch.setattr(agent_mod, "build_and_audit", spy_build)

    result = run_supervisor(
        ["PMC1"],
        fullmap=fullmap_db,
        build_model_factory=lambda: make_fake_model(final_yaml=first_yaml),
        map_threshold=1.0,
        max_improve_iters=3,
        state_dir=tmp_path / "state",
        workdir=tmp_path / "w",
    )
    assert result["records"]["PMC1"].status == "MAPPED"  # pyright: ignore[reportIndexIssue]
    assert head_flags[0] is False  # initial build is full
    assert any(head_flags)  # at least one head intermediate scoring build
    assert head_flags[-1] is False  # the accepted config's persisted build is full


def test_supervisor_loop_iterates_while_improving(tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The improve loop runs >1 iteration when successive candidates keep improving coverage."""
    import tablassert.agent as agent_mod

    table: Path = _write_table(tmp_path, "d.tsv", "brca1\tmapk1\n")
    _patch_fetch(monkeypatch, table)
    good_yaml: str = yaml.safe_dump(_column_cfg(table))

    monkeypatch.setattr(agent_mod, "propose_config_candidates", lambda cfg, rep: [(good_yaml, "edit")])
    coverages: list[float] = [0.2, 0.5, 0.5, 0.9, 0.9]  # initial, then (head, full) per accepted iter
    idx: dict[str, int] = {"i": 0}

    def fake_build(
        config_yaml: str,
        *,
        fullmap: Path,
        name: str = "agent",
        version: str = "0.0.1",
        qc: bool = False,
        head: bool = False,
        workdir: Path | None = None,
    ) -> dict[str, Any]:  # pyright: ignore[reportUnusedParameter]
        cov: float = coverages[idx["i"]] if idx["i"] < len(coverages) else 0.9
        idx["i"] += 1
        return {
            "ok": True,
            "coverage_pct": cov,
            "measured": True,
            "qc_pass_rate": None,
            "errors": [],
            "error_codes": [],
            "kgx_path": None,
            "edges_path": None,
            "node_count": 1,
            "edge_count": 1,
            "unresolved": [],
        }

    monkeypatch.setattr(agent_mod, "build_and_audit", fake_build)

    result = run_supervisor(
        ["PMC1"],
        fullmap=fullmap_db,
        build_model_factory=lambda: make_fake_model(final_yaml=good_yaml),
        map_threshold=0.8,
        max_improve_iters=5,
        state_dir=tmp_path / "state",
        workdir=tmp_path / "w",
    )
    rec: ConfigRecord = result["records"]["PMC1"]  # pyright: ignore[reportIndexIssue]
    assert rec.status == "MAPPED"
    assert rec.coverage_history == [0.2, 0.5, 0.9]  # initial + 2 accepted improvements => >1 iteration


# --------------------------------------------------------------------------- #
# W4: local-payload input (no PMC-AWS fetch)
# --------------------------------------------------------------------------- #


def test_supervisor_local_payload_no_network(tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """W4: a local payload (a bare DIR) runs the same pipeline with NO fetch -> MAPPED; task flags local payload."""
    import tablassert.agent as agent_mod

    payload: Path = tmp_path / "payload"
    payload.mkdir()
    table: Path = payload / "s1.tsv"
    table.write_text("brca1\tmapk1\nbrca1\tmapk1\n")

    def boom_fetch(pmc_id: str, outdir: Path, *, timeout: int = 120) -> list[Path]:  # pyright: ignore[reportUnusedParameter]
        raise AssertionError("fetch_pmc_article must NOT be called for a local payload")

    monkeypatch.setattr(agent_mod, "fetch_pmc_article", boom_fetch)

    captured: dict[str, str] = {}
    real_build_agent = agent_mod.build_agent

    def spy_build_agent(*args: object, **kwargs: object) -> object:
        agent = real_build_agent(*args, **kwargs)

        class _Spy:
            def run(self, task: str) -> object:
                captured["task"] = task
                return agent.run(task)  # pyright: ignore[reportAttributeAccessIssue]

        return _Spy()

    monkeypatch.setattr(agent_mod, "build_agent", spy_build_agent)

    good_yaml: str = yaml.safe_dump(_column_cfg(table))
    result = run_supervisor(
        ["PMC1"],
        fullmap=fullmap_db,
        build_model_factory=lambda: make_fake_model(final_yaml=good_yaml),
        map_threshold=0.8,
        state_dir=tmp_path / "state",
        workdir=tmp_path / "w",
        local=payload,  # a bare Path applies to every id
    )
    assert result["records"]["PMC1"].status == "MAPPED"  # pyright: ignore[reportIndexIssue]
    assert str(table) in captured["task"]
    assert "local payload" in captured["task"]  # no fabricated S3 link for local files


def test_supervisor_local_payload_per_id_mapping(tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """W4: a {pmc_id: dir} mapping selects the per-article local payload; unmapped ids still fetch."""
    import tablassert.agent as agent_mod

    payload: Path = tmp_path / "payload"
    payload.mkdir()
    table: Path = payload / "s1.tsv"
    table.write_text("brca1\tmapk1\nbrca1\tmapk1\n")

    fetched: list[str] = []

    def fake_fetch(pmc_id: str, outdir: Path, *, timeout: int = 120) -> list[Path]:  # pyright: ignore[reportUnusedParameter]
        fetched.append(pmc_id)
        return [table]

    monkeypatch.setattr(agent_mod, "fetch_pmc_article", fake_fetch)

    good_yaml: str = yaml.safe_dump(_column_cfg(table))
    result = run_supervisor(
        ["PMCLOCAL", "PMCFETCH"],
        fullmap=fullmap_db,
        build_model_factory=lambda: make_fake_model(final_yaml=good_yaml),
        map_threshold=0.8,
        state_dir=tmp_path / "state",
        workdir=tmp_path / "w",
        local={"PMCLOCAL": payload},  # only PMCLOCAL is local; PMCFETCH falls back to fetch
    )
    records: dict[str, ConfigRecord] = result["records"]  # pyright: ignore[reportAssignmentType]
    assert records["PMCLOCAL"].status == "MAPPED"
    assert records["PMCFETCH"].status == "MAPPED"
    assert fetched == ["PMCFETCH"]  # fetch used ONLY for the id without a local payload


def test_supervisor_local_payload_no_table_skipped(tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """W4: a local payload with no data table is SKIPPED (fail-fast), and the batch advances."""
    import tablassert.agent as agent_mod

    payload: Path = tmp_path / "payload"
    payload.mkdir()
    (payload / "notes.txt").write_text("no table here")  # not a data table

    monkeypatch.setattr(agent_mod, "fetch_pmc_article", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not fetch")))

    result = run_supervisor(
        ["PMC1"],
        fullmap=fullmap_db,
        build_model_factory=lambda: make_fake_model(),
        map_threshold=0.8,
        state_dir=tmp_path / "state",
        workdir=tmp_path / "w",
        local=payload,
    )
    rec: ConfigRecord = result["records"]["PMC1"]  # pyright: ignore[reportIndexIssue]
    assert rec.status == "SKIPPED"
