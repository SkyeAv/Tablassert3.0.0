from __future__ import annotations

import json
import random
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any
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


def test_dedup_ndjson_skips_blank_lines(tmp_path: Path) -> None:
    """dedup_ndjson tolerates empty and whitespace-only lines."""
    from tablassert import rs

    p_in: Path = tmp_path / "nodes.ndjson.tmp"
    p_out: Path = tmp_path / "nodes.ndjson"
    p_in.write_text('\n  \n{"id":"A"}\n\n{"id":"A"}\n   \n')

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


# ---------------------------------------------------------------------------
# Pure-Python reference of the CURRENT rust merge-mode semantics (frozen oracle
# for the US-002 rewrite). Mirrors `strip_nulls` / `uuid_for_json_object` /
# `merge_records` / `dedup_edges_merge` from `rust/src/{json,uuid,ndjson}.rs`.
# Intentionally simple (lists, not hash sets) because correctness against the
# rust output matters more than speed here.
# ---------------------------------------------------------------------------

_BAD_TOKENS = frozenset({"", "na", "nan", "null", "none"})
_MISSING: Any = object()
_NIL_NAMESPACE = uuid.UUID(int=0)


def _canonical_json_bytes(value: Any) -> bytes:
    """Mirror of rust `canonical_json_bytes`: keys sorted recursively, compact bytes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _strip_nulls(record: dict[str, Any]) -> dict[str, Any]:
    """Mirror of rust `strip_nulls` (json.rs): drop absent/null-like values only.

    Deliberately NOT Python truthiness -- `0`, `0.0` and `false` are meaningful values
    and stay; nested dicts recurse (an emptied nested dict survives as `{}`), and list
    items recurse only when they are dicts.
    """

    def is_present(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, bool | int | float):
            return True
        if isinstance(value, str):
            return value != ""
        return len(value) > 0  # lists and dicts

    def passes_bad_check(value: Any) -> bool:
        return not (isinstance(value, str) and value.strip().lower() in _BAD_TOKENS)

    def transform(value: Any) -> Any:
        if isinstance(value, list):
            return [_strip_nulls(item) if isinstance(item, dict) else item for item in value]
        if isinstance(value, dict):
            return _strip_nulls(value)
        return value

    return {key: transform(value) for key, value in record.items() if is_present(value) and passes_bad_check(value)}


def _uuid_part(value: Any) -> str | None:
    """Mirror of rust `uuid_part`: normalize one field value to its hashable form."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, str):
        return value if value else None
    return _canonical_json_bytes(value).decode("utf-8")


def _edge_uuid(domain: str, record: dict[str, Any], fields: list[str]) -> str:
    """Mirror of rust `uuid_for_json_object` over declared `uuid_fields`.

    v3 UUIDs: namespaced by `uuid3(NIL, domain)`, over the length-prefixed
    `<byte-len>:<part>` join of sorted key/value parts; entries whose value normalizes
    to nothing (null, empty) drop out key included.
    """
    namespace = uuid.uuid3(_NIL_NAMESPACE, domain)
    keys: list[str] = sorted({key for key in fields if key in record})
    parts: list[str] = []
    for key in keys:
        part = _uuid_part(record[key])
        if part is not None:
            parts.append(key)
            parts.append(part)
    joined = "".join(f"{len(part.encode('utf-8'))}:{part}" for part in parts)
    return str(uuid.uuid3(namespace, joined))


def _merge_records(stored: dict[str, Any], incoming: dict[str, Any]) -> int:
    """Mirror of rust `merge_records`: fold `incoming` into `stored` field-wise.

    List fields union by canonical bytes and sort ONLY when both sides carry an array
    (a list copied from a later record keeps its order); stored-side duplicates survive,
    incoming-side ones collapse. Scalars are first-wins, each difference counts one
    conflict; a field only on `incoming` is copied, not a conflict. `id` is untouched.
    `number_of_cases` is recomputed to the union length of `supporting_case_ids` when the
    merged record carries the carrier list and either side carried a count, and that
    superseded divergence is decremented out of the conflict total (both sides present,
    unequal, not both arrays -- the exact mirror of the rust loop condition).
    """
    conflicts = 0
    stored_cases: Any = stored.get("number_of_cases", _MISSING)
    incoming_cases: Any = incoming.get("number_of_cases", _MISSING)
    for key, incoming_value in incoming.items():
        if key == "id":
            continue
        if key not in stored:
            stored[key] = incoming_value
            continue
        stored_value = stored[key]
        if isinstance(stored_value, list) and isinstance(incoming_value, list):
            seen: list[bytes] = [_canonical_json_bytes(item) for item in stored_value]
            for item in incoming_value:
                item_bytes = _canonical_json_bytes(item)
                if item_bytes not in seen:
                    seen.append(item_bytes)
                    stored_value.append(item)
            stored_value.sort(key=_canonical_json_bytes)
        elif stored_value != incoming_value:
            conflicts += 1
    union = stored.get("supporting_case_ids")
    if isinstance(union, list) and (stored_cases is not _MISSING or incoming_cases is not _MISSING):
        if (
            stored_cases is not _MISSING
            and incoming_cases is not _MISSING
            and stored_cases != incoming_cases
            and not (isinstance(stored_cases, list) and isinstance(incoming_cases, list))
        ):
            conflicts -= 1
        stored["number_of_cases"] = len(union)
    return conflicts


def _swap_remove(record: dict[str, Any], key: str) -> None:
    """Mirror of serde_json `Map::remove` under `preserve_order`: indexmap `swap_remove`.

    The LAST key fills the vacated slot -- removal does NOT preserve the order of the
    remaining keys. This is part of the pinned emit semantics of `strip_internal_edge_
    fields`, so the reference must reproduce it byte-for-byte.
    """
    if key not in record:
        return
    keys = list(record.keys())
    if keys[-1] == key:
        del record[key]
        return
    rebuilt: dict[str, Any] = {}
    for existing in keys:
        if existing == key:
            rebuilt[keys[-1]] = record[keys[-1]]
        elif existing != keys[-1]:
            rebuilt[existing] = record[existing]
    record.clear()
    record.update(rebuilt)


def _merge_reference(lines: Iterable[str], domain: str, fields: list[str]) -> tuple[bytes, int, int]:
    """Drive the CURRENT rust merge pass over NDJSON lines, entirely in Python.

    Mirror of `dedup_edges_merge`: skip blank lines and empty objects, strip nulls,
    label each edge, suppress exact content repeats WITHOUT counting them, fold divergent
    same-id records first-wins, then emit one compact line per id in FIRST-SEEN order with
    `supporting_case_ids` stripped. Returns `(output bytes, merged, scalar_conflicts)`.
    """
    records: dict[str, tuple[list[bytes], dict[str, Any]]] = {}
    order: list[str] = []
    merged = 0
    conflicts = 0
    for line in lines:
        if not line.strip():
            continue
        cleaned = _strip_nulls(json.loads(line))
        if not cleaned:
            continue
        content = _canonical_json_bytes(cleaned)
        cleaned["id"] = _edge_uuid(domain, cleaned, fields)
        slot = records.get(cleaned["id"])
        if slot is None:
            records[cleaned["id"]] = ([content], cleaned)
            order.append(cleaned["id"])
            continue
        hashes, stored = slot
        if content in hashes:
            continue
        merged += 1
        conflicts += _merge_records(stored, cleaned)
        hashes.append(content)
    chunks: list[bytes] = []
    for edge_id in order:
        _, value = records[edge_id]
        _swap_remove(value, "supporting_case_ids")
        chunks.append(json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n")
    return b"".join(chunks), merged, conflicts


_SUBJECTS = [f"MONDO:{index:03d}" for index in range(6)]
_PREDICATES = ["biolink:related_to", "biolink:associated_with"]
_OBJECTS = [f"NCBIGene:{index:03d}" for index in range(4)]
_TAGS = ["tag:a", "tag:b", "tag:c", "tag:d", "tag:e"]
_CASE_IDS = [f"case:{index}" for index in range(8)]
_P_VALUES = ["1e-5", "0.01", "0.99"]


def _fuzz_source(rng: random.Random) -> dict[str, str]:
    identity, role = ("infores:one", "primary_knowledge_source") if rng.randrange(2) == 0 else ("infores:two", "aggregator_knowledge_source")
    if rng.randrange(2) == 0:
        return {"resource_id": identity, "resource_role": role}
    return {"resource_role": role, "resource_id": identity}  # same source, different key order


def _fuzz_record(rng: random.Random, group: int, record_index: int) -> dict[str, Any]:
    record: dict[str, Any] = {}
    # Identity triple in random insertion order: same id, divergent bytes -- the fold, not
    # the id derivation, decides the outcome. (Int scalars only: float formatting differs
    # between rust and python serializers and is irrelevant to merge semantics.)
    triple: list[tuple[str, str]] = [("subject", rng.choice(_SUBJECTS)), ("predicate", rng.choice(_PREDICATES)), ("object", rng.choice(_OBJECTS))]
    rng.shuffle(triple)
    record.update(triple)
    record["p_value"] = rng.choice(_P_VALUES)
    record["effect_size"] = rng.randrange(4)
    if rng.randrange(4) == 0:
        record["negated"] = rng.randrange(2) == 0
    if rng.randrange(2) == 0:
        # Drawn WITH replacement: a record may repeat an item inside its own array.
        record["tags"] = [rng.choice(_TAGS) for _ in range(rng.randrange(4))]
    if rng.randrange(2) == 0:
        record["sources"] = [_fuzz_source(rng) for _ in range(1 + rng.randrange(2))]
    if rng.randrange(3) == 0:
        record["mode"] = "solo" if rng.randrange(2) == 0 else ["solo", "extra"]  # scalar-vs-array conflict
    if record_index > 0 and (group % 3 == 0 or rng.randrange(2) == 0):
        record["late"] = f"late:{group}"  # field only later records carry
    if rng.randrange(2) == 0:
        case_ids = [rng.choice(_CASE_IDS) for _ in range(1 + rng.randrange(4))]
        record["supporting_case_ids"] = case_ids
        if rng.randrange(4):
            # Deliberately wrong sometimes: the recompute supersedes the count and must
            # also undo the scalar conflict the divergence would otherwise have counted.
            record["number_of_cases"] = len(case_ids) if rng.randrange(3) == 0 else len(case_ids) + 1 + rng.randrange(3)
    return record


def _fuzz_dataset(rng: random.Random) -> list[str]:
    records: list[dict[str, Any]] = []
    for group in range(36):  # divergent same-id groups of 1-7 records
        for record_index in range(1 + rng.randrange(7)):
            records.append(_fuzz_record(rng, group, record_index))
    rng.shuffle(records)
    lines = [json.dumps(record) for record in records]
    unique = len(lines)
    for _ in range(10):  # exact byte repeats -> content-hash suppression, no counters
        lines.append(lines[rng.randrange(unique)])
    rng.shuffle(lines)
    stream: list[str] = []
    for line in lines:
        if rng.randrange(14) == 0:
            stream.append("")  # blank lines are legal stream noise
        if rng.randrange(20) == 0:
            stream.append("   ")
        if rng.randrange(16) == 0:
            stream.append("{}")  # empty objects are skipped
        stream.append(line)
    return stream


def test_dedup_edges_merge_matches_python_reference(tmp_path: Path) -> None:
    """Merge mode must byte-match an independent Python port of its own semantics.

    WHY: US-002 will rewrite the rust merge fold for speed. This test pins the CURRENT
    semantics from a SECOND implementation: a seeded randomized dataset (divergent same-id
    groups, list unions over object and scalar items, key-order variants, late fields,
    scalar-vs-array conflicts, exact repeats, empty objects, blank lines) is deduped by
    `rs.dedup_ndjson(..., on_collision="merge")` and by the pure-Python reference above.
    The emitted bytes -- first-seen id order, folded records, `supporting_case_ids`
    stripped -- plus the `(merged, scalar_conflicts)` counters must agree exactly, so any
    semantic drift in the rewrite fails here.
    """
    from tablassert import rs

    rng = random.Random(1337)
    lines = _fuzz_dataset(rng)
    domain = "infores:multiomicskg"
    fields = ["subject", "predicate", "object"]
    p_in: Path = tmp_path / "edges.ndjson.tmp"
    p_out: Path = tmp_path / "edges.ndjson"
    p_in.write_text("\n".join(lines) + "\n")

    merged, conflicts = rs.dedup_ndjson(p_in, p_out, True, domain, fields, "merge")
    expected, expected_merged, expected_conflicts = _merge_reference(lines, domain, fields)

    # Non-vacuity: the seeded dataset must actually exercise the fold.
    assert merged > 0, "seeded dataset must actually fold records"
    assert conflicts > 0, "seeded dataset must produce scalar conflicts"
    actual = p_out.read_bytes()
    assert b"supporting_case_ids" not in actual
    assert actual == expected
    assert (merged, conflicts) == (expected_merged, expected_conflicts)
