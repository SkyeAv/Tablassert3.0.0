"""Coverage tests for the UNCOVERED branches of ``src/tablassert/agent.py``.

Targets the propose/config-edit helpers (``_looks_taxonomic`` / ``_column_unresolved`` /
``_statement_nodes`` / ``_edit_node`` / ``propose_config_edit``), the ``make_step_callback``
context-trim guard, the ``make_tools`` ``get_fullmap`` closure, the ``load_state`` non-mapping
branch, and the ``run_supervisor`` validate-gate SKIPPED + improve-loop coverage-failure paths.

The PURE helper tests run in the base environment (no ``[agent]`` extra). The tests that drive a
real ``CodeAgent``/smolagents ``Tool`` call ``pytest.importorskip("smolagents")`` INSIDE the test so
this module still COLLECTS without the extra (module-top imports stay stdlib/project-only). Patterns
(fixtures, ``fetch_pmc_article`` monkeypatch, ``make_fake_model``, HF-telemetry-off autouse fixture)
are copied from ``tests/test_agent_supervisor.py`` / ``tests/test_agent_assembly.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from tablassert import rs
from tablassert.agent import (
    ConfigRecord,
    _column_unresolved,
    _edit_node,
    _looks_taxonomic,
    _statement_nodes,
    load_state,
    make_fake_model,
    make_step_callback,
    make_tools,
    propose_config_edit,
    run_supervisor,
)


@pytest.fixture(autouse=True)
def _offline_no_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ``agent.run`` hermetically offline by disabling HuggingFace telemetry.

    Why: a real ``CodeAgent.run`` fires huggingface_hub telemetry that BLOCKS on a network call
    (observed: the run hangs indefinitely at zero CPU when telemetry is enabled). Scoped to this
    file (NOT global conftest) so the QC suite's own HuggingFace model loading is unaffected.
    """
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("DO_NOT_TRACK", "1")


# --------------------------------------------------------------------------- #
# Offline fixtures: tiny REAL redb + small text tables + a valid Section config
# (copied from tests/test_agent_supervisor.py)
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
        "source": {"kind": "text", "local": str(table), "url": ["https://e.com/d.tsv"], "delimiter": "\t"},
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
# PURE helper tests (base env; no importorskip)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("term", ["Escherichia sp", "Escherichia sp."])
def test_looks_taxonomic_species_marker(term: str) -> None:
    """Covers agent.py:1150 — the ``" sp" in term or "sp." in term`` species-marker return True.

    These terms carry NO lineage separator (so the earlier ``return True`` at :1148 is skipped) but
    DO carry an ``sp``/``sp.`` marker, so the second branch is the one that returns True.
    """
    assert _looks_taxonomic(term) is True


def test_column_unresolved_non_list_returns_empty() -> None:
    """Covers agent.py:1212 — a COLUMN node whose ``unresolved`` is NOT a list returns ``[]``.

    The entry is a dict with a column method (passes the ``_is_column_method`` guard) but its
    ``unresolved`` field is a str, so the ``not isinstance(unresolved, list)`` branch returns ``[]``.
    """
    assert _column_unresolved({"method": "column", "unresolved": "not-a-list"}) == []


def test_statement_nodes_qualifiers() -> None:
    """Covers agent.py:1227-1231 — the qualifier loop pairing named qualifier dicts with their column.

    A statement with a ``qualifiers`` list exercises the loop body: a dict qualifier WITH a string
    ``qualifier`` name is paired and appended (1227-1231); a non-dict entry and a dict without a
    string name are skipped by the two isinstance guards.
    """
    statement: dict[str, object] = {
        "subject": {"method": "column", "encoding": "A"},
        "qualifiers": [{"qualifier": "q1", "method": "column"}, "not-a-dict", {"method": "column"}],
    }
    nodes = _statement_nodes(statement)
    assert ("subject", {"method": "column", "encoding": "A"}) in nodes
    assert ("q1", {"qualifier": "q1", "method": "column"}) in nodes
    # the non-dict entry and the dict lacking a string qualifier name add nothing
    assert [name for name, _ in nodes] == ["subject", "q1"]


def test_edit_node_exclude_regex_hint() -> None:
    """Covers agent.py:1269-1270 — a report-level ``exclude_regex`` hint is extended onto the node.

    The unresolved term is neither taxonomic nor noise and there are no prefix hints, so the ONLY
    heuristic that fires is the ``hint_regex`` branch: it sets ``fired = True`` (1269) and appends
    the ``excluded regex`` rationale (1270).
    """
    node: dict[str, object] = {}
    rationale = _edit_node("subject", node, ["someunresolvedterm"], [], ["^OMIM"])
    assert rationale is not None
    assert "excluded regex" in rationale
    assert node["exclude_regex"] == ["^OMIM"]


def test_propose_config_not_mapping() -> None:
    """Covers agent.py:1303 — config YAML that parses to a non-mapping returns the 'did not parse' note.

    ``"[1, 2, 3]"`` parses to a list (not a dict), so ``propose_config_edit`` returns the original
    YAML unchanged with the 'config did not parse to a mapping' rationale instead of editing.
    """
    edited, rationale = propose_config_edit("[1, 2, 3]", {"per_column": {}})
    assert edited == "[1, 2, 3]"
    assert "did not parse to a mapping" in rationale


def test_propose_edit_fails_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers agent.py:1334 — a changed edit that FAILS ``validate_section`` returns the original.

    ``validate_section`` is monkeypatched to return False so the (genuinely changed) taxonomic edit
    is rejected at the re-validation gate, returning the ORIGINAL config with the 'failed schema
    validation' rationale.
    """
    monkeypatch.setattr("tablassert.agent.validate_section", lambda *args, **kwargs: False)
    config: dict[str, Any] = {
        "source": {"kind": "text", "local": "./d.tsv", "url": ["https://e.com/d.tsv"], "delimiter": "\t"},
        "statement": {
            "subject": {"method": "column", "encoding": "A"},
            "predicate": "associated_with",
            "object": {"method": "value", "encoding": "CHEBI:41774"},
        },
        "provenance": {"repo": "PMC", "publication": "PMC1"},
    }
    report: dict[str, Any] = {
        "per_column": {"subject": {"coverage": 0.0, "total": 1, "resolved": 0, "unresolved": ["g__Bacteroides"], "method": "column"}}
    }
    original: str = yaml.safe_dump(config, sort_keys=False)
    edited, rationale = propose_config_edit(config, report)
    assert edited == original
    assert "failed schema validation" in rationale


def test_propose_exception_returns_original(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers agent.py:1336-1337 — an internal exception is caught and the original is returned.

    ``_merge_first_section`` is monkeypatched to raise so the proposer's blanket ``except Exception``
    handler fires, returning the original config with the 'propose_config_edit error' note (the
    proposer must NEVER raise).
    """

    def boom(cfg: dict[str, object]) -> dict[str, object]:  # pyright: ignore[reportUnusedParameter]
        raise RuntimeError("synthetic internal failure")

    monkeypatch.setattr("tablassert.agent._merge_first_section", boom)
    config: dict[str, Any] = {"statement": {"subject": {"method": "column", "encoding": "A"}}}
    edited, rationale = propose_config_edit(config, {"per_column": {}})
    assert edited == yaml.safe_dump(config, sort_keys=False)
    assert "propose_config_edit error" in rationale
    assert "synthetic internal failure" in rationale


@dataclass(frozen=True)
class _FrozenStep:
    """An immutable step type: assigning ``observations`` raises (a differing memory shape)."""

    observations: str


def test_step_callback_trim_exception_swallowed() -> None:
    """Covers agent.py:1588-1589 — a context-trim failure is swallowed and never breaks the run.

    The oldest step in ``agent.memory.steps`` is an IMMUTABLE step whose ``observations`` is >4000
    chars, so the trim's ``old.observations = ...`` assignment raises; the ``except Exception: pass``
    guard swallows it and the callback still completes (``metrics["steps"]`` increments).
    """
    metrics: dict[str, object] = {}
    cb = make_step_callback(metrics)
    old = _FrozenStep(observations="x" * 5000)
    agent = SimpleNamespace(memory=SimpleNamespace(steps=[old, SimpleNamespace(observations="a"), SimpleNamespace(observations="b")]))
    step = SimpleNamespace(error=None, tool_calls=[], observations="ok", token_usage=None)

    cb(step, agent)  # must NOT raise

    assert metrics["steps"] == 1
    assert old.observations == "x" * 5000  # the immutable observation was left untouched


def test_load_state_non_mapping_returns_none(tmp_path: Path) -> None:
    """Covers agent.py:1947 — a ``state.json`` that is valid JSON but NOT a mapping yields ``None``.

    The file exists and parses (a JSON list), so ``load_state`` passes the ``is_file`` check but hits
    the ``not isinstance(data, dict)`` branch and returns ``None`` (fresh run) rather than raising.
    """
    state_dir: Path = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "state.json").write_text("[1, 2, 3]")
    assert load_state(state_dir) is None


# --------------------------------------------------------------------------- #
# smolagents-dependent tests (require the [agent] extra; skip cleanly when absent)
# --------------------------------------------------------------------------- #


def test_make_tools_get_fullmap_closure(tmp_path: Path, fullmap_db: Path) -> None:
    """Covers agent.py:1813 — the ``get_fullmap`` closure returns the bound fullmap path.

    ``make_tools`` binds ``fullmap`` via the nested ``get_fullmap`` closure; invoking the returned
    ``map_coverage`` tool's ``forward`` calls ``get_fullmap()`` (the ``return fullmap`` line) to feed
    the real coverage measurement, which resolves both genes in the tiny redb to full coverage.
    """
    pytest.importorskip("smolagents")
    tools: list[Any] = make_tools(fullmap=fullmap_db, name="agent", version="0.0.1")
    assert len(tools) == 6
    by_name: dict[str, Any] = {tool.name: tool for tool in tools}

    table: Path = _write_table(tmp_path, "good.tsv", "brca1\tmapk1\nbrca1\tmapk1\n")
    config_yaml: str = yaml.safe_dump(_column_cfg(table), sort_keys=False)
    report: dict[str, Any] = json.loads(by_name["map_coverage"].forward(config_yaml))
    assert "per_column" in report
    assert report["per_column"]["subject"]["resolved"] == report["per_column"]["subject"]["total"]


def test_supervisor_invalid_final_answer_skipped(tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers agent.py:2079-2082 — an agent final answer that fails the validate gate -> SKIPPED.

    ``build_agent`` is monkeypatched to a stub whose ``run`` returns a non-config string, so the
    supervisor's post-run ``validate_table_config(config)`` gate fails and the record is marked SKIPPED
    with the 'failed the validate_table_config gate' note, checkpointed, and skipped (batch advances).
    """
    pytest.importorskip("smolagents")
    import tablassert.agent as agent_mod

    table: Path = _write_table(tmp_path, "good.tsv", "brca1\tmapk1\n")
    _patch_fetch(monkeypatch, table)

    def stub_build_agent(*args: object, **kwargs: object) -> object:  # pyright: ignore[reportUnusedParameter]
        class _Stub:
            def run(self, task: str) -> object:  # pyright: ignore[reportUnusedParameter]
                return "definitely not a config"

        return _Stub()

    monkeypatch.setattr(agent_mod, "build_agent", stub_build_agent)

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
    assert "validate_table_config gate" in rec.notes


def test_supervisor_coverage_failure_fallback(tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers agent.py:2104-2105 — a ``map_coverage`` failure in the improve loop falls back safely.

    ``map_coverage`` is monkeypatched to raise inside the improve loop, so the ``except Exception``
    handler substitutes the empty ``{"per_column": {}, "unresolved": []}`` report (2104-2105) and the
    attempt continues: the deterministic edit is rejected (no improvement) and the record ends SKIPPED.
    """
    pytest.importorskip("smolagents")
    import tablassert.agent as agent_mod

    table: Path = _write_table(tmp_path, "d.tsv", "brca1\tmapk1\n")
    _patch_fetch(monkeypatch, table)
    good_yaml: str = yaml.safe_dump(_column_cfg(table), sort_keys=False)
    calls: dict[str, int] = {"mapcov": 0}

    def fake_build(
        config_yaml: str, *, fullmap: Path, name: str = "agent", version: str = "0.0.1", qc: bool = False, workdir: Path | None = None
    ) -> dict[str, Any]:  # pyright: ignore[reportUnusedParameter]
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

    def boom_coverage(*args: object, **kwargs: object) -> dict[str, object]:  # pyright: ignore[reportUnusedParameter]
        calls["mapcov"] += 1
        raise RuntimeError("coverage backend down")

    monkeypatch.setattr(agent_mod, "build_and_audit", fake_build)
    monkeypatch.setattr(agent_mod, "map_coverage", boom_coverage)
    monkeypatch.setattr(agent_mod, "propose_config_edit", lambda cfg, rep: (good_yaml, "proposed edit"))

    result = run_supervisor(
        ["PMC1"],
        fullmap=fullmap_db,
        build_model_factory=lambda: make_fake_model(final_yaml=good_yaml),
        map_threshold=0.8,
        max_improve_iters=3,
        state_dir=tmp_path / "state",
        workdir=tmp_path / "w",
    )
    assert calls["mapcov"] >= 1  # the improve loop did invoke map_coverage (and swallowed its failure)
    rec: ConfigRecord = result["records"]["PMC1"]  # pyright: ignore[reportIndexIssue]
    assert rec.status == "SKIPPED"  # 0.5 < 0.8 and the rejected edit never improved coverage
