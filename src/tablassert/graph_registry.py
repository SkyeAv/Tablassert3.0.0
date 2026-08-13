"""Concurrency-safe SHARED graph registry for parallel ``tablassert agent`` runs.

Several agent processes pointed at the SAME ``--state-dir`` each self-register their successful
builds (status ``MAPPED`` or ``BUILT_UNMEASURED``) into ONE aggregate ``<state-dir>/graph.yaml``
that ``tablassert build-kg -f <state-dir>/graph.yaml`` then builds as a whole.

Concurrency safety: an EXCLUSIVE ``fcntl.flock`` on the ``<state-dir>/graph.yaml.lock`` sidecar
serializes every read-modify-write, and the write itself is atomic (a tmp file in the same
directory + ``os.replace`` — the same pattern as ``save_state``). A corrupt registry (YAML parse
error, not a mapping, or failing ``Graph.model_validate``) is quarantined to
``graph.yaml.corrupt-<UTC timestamp>`` and rebuilt fresh, so unattended parallel runs self-heal
instead of wedging. Stdlib only — no new dependencies.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pydantic
import yaml

from tablassert.log import cat
from tablassert.models import Graph

logger = cat("REGISTRY")

GRAPH_YAML: str = "graph.yaml"
LOCK_NAME: str = f"{GRAPH_YAML}.lock"
TMP_NAME: str = f"{GRAPH_YAML}.tmp"
CORRUPT_PREFIX: str = f"{GRAPH_YAML}.corrupt-"
GRAPH_NAME: str = "tablassert-agent"
GRAPH_VERSION: str = "1"
#: Record statuses whose best config self-registers (both are SUCCESSFUL builds).
REGISTERED_STATUSES: frozenset[str] = frozenset({"MAPPED", "BUILT_UNMEASURED"})


def _registry_rig(state_dir: Path) -> dict[str, Any]:
    """The honest ``rig:`` block for the aggregate agent registry graph.

    Every fact here is mechanical: the agent only mines PubMed Central
    open-access supplementary tables, so source terms/access describe PMC, and
    the artifact bases point at the state directory itself (a ``file://`` base
    is a valid unpublished URI; swap it for a public https base before sending
    the generated RIG anywhere).
    """
    resolved: str = str(state_dir.resolve())
    return {
        "source_info": {
            "infores_id": "infores:tablassert-agent",
            "name": "PubMed Central open-access supplementary tables",
            "description": "Aggregate of tabular associations mined from PubMed Central open-access supplementary files by the Tablassert agent.",
            "terms_of_use_info": {
                "terms_of_use_url": "https://pmc.ncbi.nlm.nih.gov/about/copyright/",
                "terms_of_use_description": "PubMed Central open-access subset; individual article licenses apply.",
            },
            "data_access_locations": ["PubMed Central - https://pmc.ncbi.nlm.nih.gov/"],
            "source_status": "unknown",
        },
        "ingest_info": {
            "utility": "Aggregates agent-derived tabular knowledge assertions for Translator-style querying.",
            "scope": "All agent-built table configs registered under this state directory.",
        },
        "provenance_info": {"contributions": ["Tablassert agent: automated config derivation and build"]},
        "artifact_base_url": f"file://{resolved}",
        "artifact_base_path": resolved,
    }


@contextlib.contextmanager
def _registry_lock(state_dir: Path) -> Iterator[None]:
    """Hold an EXCLUSIVE ``flock`` on ``<state_dir>/graph.yaml.lock`` (created if missing).

    The sidecar lock file is never deleted, so concurrent processes always contend on the same
    inode even while ``graph.yaml`` itself is atomically replaced underneath them.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path: Path = state_dir / LOCK_NAME
    with lock_path.open("a") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _fresh_doc(state_dir: Path) -> dict[str, Any]:
    """A brand-new registry document (``_apply_fullmap`` fills the first-wins ``fullmap``)."""
    return {"name": GRAPH_NAME, "version": GRAPH_VERSION, "tables": [], "rig": _registry_rig(state_dir)}


def _quarantine(state_dir: Path, reason: str) -> Path:
    """Rename a corrupt ``graph.yaml`` aside to ``graph.yaml.corrupt-<UTC timestamp>`` and warn.

    The corrupt bytes are preserved for forensics; the caller continues against a fresh document,
    which is what lets unattended parallel runs self-heal.
    """
    stamp: str = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    quarantined: Path = state_dir / f"{CORRUPT_PREFIX}{stamp}"
    os.replace(state_dir / GRAPH_YAML, quarantined)
    logger.warning("graph registry: quarantined corrupt {name} -> {quarantined} ({reason})", name=GRAPH_YAML, quarantined=quarantined, reason=reason)
    return quarantined


def _load_registry(state_dir: Path) -> dict[str, Any]:
    """Load the existing registry document; absent -> fresh, corrupt -> quarantine + fresh.

    Corrupt means: a YAML parse error, a top level that is not a mapping, or a document failing
    ``Graph.model_validate``. Validation on load guarantees every mutation below starts from a
    document the pipelines would accept.
    """
    path: Path = state_dir / GRAPH_YAML
    if not path.is_file():
        return _fresh_doc(state_dir)
    try:
        data: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        _quarantine(state_dir, f"YAML parse error: {exc}")
        return _fresh_doc(state_dir)
    if not isinstance(data, dict):
        _quarantine(state_dir, f"top level is not a mapping (got {type(data).__name__})")
        return _fresh_doc(state_dir)
    try:
        Graph.model_validate(data)
    except pydantic.ValidationError as exc:
        _quarantine(state_dir, f"fails Graph.model_validate ({len(exc.errors())} error(s))")
        return _fresh_doc(state_dir)
    return data


def _apply_fullmap(doc: dict[str, Any], fullmap: Path) -> None:
    """Apply the FIRST-WINS fullmap rule: set when absent; keep + warn when present and different."""
    resolved: str = str(fullmap.resolve())
    existing: object = doc.get("fullmap")
    if existing is None:
        doc["fullmap"] = resolved
        return
    if str(existing) != resolved:
        logger.warning(
            "graph registry: keeping existing fullmap {existing}; caller requested {requested} (fullmap is first-wins)",
            existing=str(existing),
            requested=resolved,
        )


def _write_registry(state_dir: Path, doc: dict[str, Any]) -> Path:
    """Validate and atomically persist the registry (tmp write + ``os.replace`` under the held lock)."""
    Graph.model_validate(doc)  # final gate: never persist a document the pipelines would reject
    target: Path = state_dir / GRAPH_YAML
    tmp: Path = state_dir / TMP_NAME
    tmp.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    os.replace(tmp, target)
    return target


def register_build(state_dir: Path, pmc_id: str, config_path: Path, fullmap: Path) -> Path:
    """Upsert one successful agent build into the shared ``<state_dir>/graph.yaml`` registry.

    Drops any existing ``tables`` entry whose basename stem equals ``pmc_id`` (a re-run REPLACES
    the prior entry), then appends the ABSOLUTE config path — the pipelines resolve ``Graph.tables``
    against the CWD, and saved configs already carry absolute ``source.local``, so absolute entries
    make ``build-kg`` work from any CWD. The fullmap is first-wins and the write is atomic under
    the exclusive sidecar lock, so concurrent registrations serialize and never lose entries.

    Returns the registry path.
    """
    absolute: str = str(config_path.resolve())
    with _registry_lock(state_dir):
        doc: dict[str, Any] = _load_registry(state_dir)
        tables: list[Any] = doc.get("tables", [])
        doc["tables"] = [entry for entry in tables if Path(str(entry)).stem != pmc_id] + [absolute]
        _apply_fullmap(doc, fullmap)
        target: Path = _write_registry(state_dir, doc)
    logger.info("graph registry: registered {pmc} -> {config} in {target}", pmc=pmc_id, config=absolute, target=target)
    return target


def rebuild_graph(state_dir: Path, fullmap: Path) -> Path:
    """Reconstruct the registry ``tables`` from ``state.json`` records, pruning stale entries.

    Keeps every record whose status is in ``REGISTERED_STATUSES`` and whose ``best_config_path``
    still exists on disk (sorted by pmc id); every other entry — SKIPPED records, deleted configs,
    stale leftovers — is pruned. Same exclusive lock + atomic write as ``register_build``; the
    fullmap is first-wins. Returns the registry path.
    """
    from tablassert.agent import load_state  # deferred: tablassert.agent imports this module at top level

    with _registry_lock(state_dir):
        doc: dict[str, Any] = _load_registry(state_dir)
        try:
            state = load_state(state_dir)
        except (OSError, ValueError) as exc:  # ValueError covers json.JSONDecodeError + UnicodeDecodeError
            # The recovery command must not die on the very damage it is meant to fix: warn and
            # rebuild an empty registry instead of raising a raw traceback.
            logger.warning("graph registry: unreadable state.json ({error}); rebuilding an empty registry", error=exc)
            state = None
        tables: list[str] = []
        if state is not None:
            for pmc_id in sorted(state.records):
                record = state.records[pmc_id]
                if record.status not in REGISTERED_STATUSES or record.best_config_path is None:
                    continue
                best: Path = Path(record.best_config_path)
                if best.is_file():
                    tables.append(str(best.resolve()))
        doc["tables"] = tables
        _apply_fullmap(doc, fullmap)
        target: Path = _write_registry(state_dir, doc)
    logger.info("graph registry: rebuilt {target} from state.json ({count} table config(s))", target=target, count=len(tables))
    return target
