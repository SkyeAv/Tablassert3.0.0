from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID


def test_namespace_uuid_returns_uuid() -> None:
    """namespace_uuid returns a UUID shaped string."""
    from tablassert import rs

    result: str = rs.namespace_uuid("domain", ["a", "b"])
    assert isinstance(result, str)
    UUID(result)  # valid UUID shape


def test_dedup_ndjson_deduplicates_nodes(tmp_path: Path) -> None:
    """dedup_ndjson strips null like values and deduplicates node lines."""
    from tablassert import rs

    p_in: Path = tmp_path / "nodes.ndjson.tmp"
    p_out: Path = tmp_path / "nodes.ndjson"
    p_in.write_text('{"id":"A","drop":"NA"}\n{"id":"A","drop":"NA"}\n{}\n')

    rs.dedup_ndjson(p_in, p_out, False, "TABLASSERT")

    assert p_out.read_text() == '{"id":"A"}\n'


def test_dedup_ndjson_labels_edges(tmp_path: Path) -> None:
    """dedup_ndjson labels edges with a UUID shaped id."""
    from tablassert import rs

    p_in: Path = tmp_path / "edges.ndjson.tmp"
    p_out: Path = tmp_path / "edges.ndjson"
    p_in.write_text('{"subject":"A","object":"B","predicate":"r"}\n')

    rs.dedup_ndjson(p_in, p_out, True, "TABLASSERT")

    row: dict = json.loads(p_out.read_text())
    UUID(row["id"])  # edge id is a valid UUID
