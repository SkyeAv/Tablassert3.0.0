from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from tablassert.log import cat

logger = cat("QC")

# Human-readable phrasing for each assertion, used by format_violations.
_MESSAGES: dict[str, str] = {
    "file-missing": "file not found",
    "malformed-lines": "empty or malformed JSON lines",
    "whitespace-values": "values with leading/trailing whitespace",
    "duplicate-node-ids": "duplicate node ids",
    "undeclared-nodes": "nodes referenced by edges but not declared in the nodes file",
    "isolated-nodes": "declared nodes participating in no edge",
}


@dataclass
class StudyViolation:
    """One failed study assertion over the final KGX NDJSON files.

    Attributes:
        check: Assertion key into ``_MESSAGES`` (e.g. ``"duplicate-node-ids"``).
        label: Which file the violation belongs to (``"nodes"`` or ``"edges"``).
        count: Exact number of offending records/values.
        examples: Capped list of example offenders for the stderr summary.
    """

    check: str
    label: str
    count: int
    examples: list[str] = field(default_factory=list)


@dataclass
class _FileScan:
    """Accumulated facts from one streamed pass over an NDJSON file."""

    ids: set[str]
    duplicate_ids: Counter[str]
    whitespace: Counter[str]
    malformed: int
    missing: bool
    path: Path


def _scan_ndjson(path: Path, *, edge: bool) -> _FileScan:
    """Stream one NDJSON file, collecting the facts the study assertions need.

    Args:
        path: Path to a ``.nodes.ndjson`` or ``.edges.ndjson`` file.
        edge: ``True`` to collect referenced ids from ``subject``/``object``;
            ``False`` to collect declared node ``id``s and track duplicates.

    Returns:
        A :class:`_FileScan`; ``missing`` is set (and nothing else) when the
        file does not exist, so a typo'd path can never read as a clean pass.
    """
    scan: _FileScan = _FileScan(set(), Counter(), Counter(), 0, not path.is_file(), path)
    if scan.missing:
        return scan
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                scan.malformed += 1
                continue
            try:
                record: object = json.loads(line)
            except json.JSONDecodeError:
                scan.malformed += 1
                continue
            if not isinstance(record, dict):
                scan.malformed += 1
                continue
            for key, value in record.items():
                if isinstance(value, str) and value != value.strip():
                    scan.whitespace[key] += 1
            if edge:
                for role in ("subject", "object"):
                    ident: object = record.get(role)
                    if isinstance(ident, str):
                        scan.ids.add(ident)
            else:
                ident = record.get("id")
                if isinstance(ident, str):
                    # Rust dedup only removes byte-identical lines, so a repeated id
                    # here means two nodes share an id with different content.
                    if ident in scan.ids:
                        scan.duplicate_ids[ident] += 1
                    scan.ids.add(ident)
    return scan


def study_kgx(nodes_path: Path, edges_path: Path, *, example_limit: int = 10) -> list[StudyViolation]:
    """Assert over the final KGX NDJSON files, in the spirit of studyKGtsvs.pl.

    Streams both files once each and checks: duplicate node ids, nodes referenced
    by edges but never declared (``undeclared``), declared nodes participating in
    no edge (``isolated``), empty/malformed lines, and string values carrying
    leading/trailing whitespace. Every check is an assertion -- the caller decides
    whether violations fail the build.

    Args:
        nodes_path: Path to ``<name>_<version>.nodes.ndjson``.
        edges_path: Path to ``<name>_<version>.edges.ndjson``.
        example_limit: Maximum number of examples retained per violation.

    Returns:
        A list of :class:`StudyViolation`; empty when every assertion passes.
    """
    nodes: _FileScan = _scan_ndjson(nodes_path, edge=False)
    edges: _FileScan = _scan_ndjson(edges_path, edge=True)

    violations: list[StudyViolation] = []
    for label, scan in (("nodes", nodes), ("edges", edges)):
        if scan.missing:
            violations.append(StudyViolation("file-missing", label, 1, [str(scan.path)]))
            continue
        if scan.malformed:
            violations.append(StudyViolation("malformed-lines", label, scan.malformed))
        if scan.whitespace:
            examples: list[str] = [f"{field_name} ({n})" for field_name, n in scan.whitespace.most_common(example_limit)]
            violations.append(StudyViolation("whitespace-values", label, sum(scan.whitespace.values()), examples))
    if nodes.duplicate_ids:
        examples = [ident for ident, _ in nodes.duplicate_ids.most_common(example_limit)]
        violations.append(StudyViolation("duplicate-node-ids", "nodes", len(nodes.duplicate_ids), examples))
    if not nodes.missing and not edges.missing:
        undeclared: list[str] = sorted(edges.ids - nodes.ids)
        if undeclared:
            violations.append(StudyViolation("undeclared-nodes", "edges", len(undeclared), undeclared[:example_limit]))
        isolated: list[str] = sorted(nodes.ids - edges.ids)
        if isolated:
            violations.append(StudyViolation("isolated-nodes", "nodes", len(isolated), isolated[:example_limit]))
    return violations


def format_violations(violations: list[StudyViolation]) -> str:
    """Render violations as a human-readable, one-line-per-assertion summary.

    Args:
        violations: The failed assertions returned by :func:`study_kgx`.

    Returns:
        Newline-joined summary lines, e.g.
        ``nodes: 3 duplicate node ids (e.g. HGNC:5, HGNC:6)``.
    """
    lines: list[str] = []
    for violation in violations:
        line: str = f"{violation.label}: {violation.count} {_MESSAGES[violation.check]}"
        if violation.examples:
            line += f" (e.g. {', '.join(violation.examples)})"
        lines.append(line)
    return "\n".join(lines)
