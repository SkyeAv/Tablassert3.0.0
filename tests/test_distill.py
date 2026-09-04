"""``--distill`` capture: the ChatML NDJSON recorder, the generate-wrapping model, and export.

The recorder tests are PURE Python (base env, no ``importorskip``): ``tablassert.distill`` is
zero-dependency by design and serializes duck-typed messages. The wrapping-model tests drive the
real ``make_distilling_model`` over ``make_fake_model`` and so skip cleanly without the ``[agent]``
extra; the export test skips without ``[distill]``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tablassert import distill
from tablassert.agent import distill_dir, make_distilling_model, make_fake_model


class _Msg:
    """ChatMessage-shaped duck type (``role``/``content``/optional ``token_usage``)."""

    def __init__(self, role: object, content: object, token_usage: object = None) -> None:
        self.role = role
        self.content = content
        self.token_usage = token_usage


class _Role:
    """Enum-shaped role (MessageRole carries a ``value``)."""

    def __init__(self, value: str) -> None:
        self.value = value


class _Usage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


def _read_records(path: Path) -> list[dict[str, Any]]:
    """Parse an NDJSON file into a list of records, asserting strict one-object-per-line."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_serialize_message_accepts_dicts_and_chatmessage_shapes() -> None:
    """Both plain dicts and ChatMessage-shaped objects serialize to ChatML role/content.

    Why: the wrapper sees smolagents ``ChatMessage`` objects, but tests and defensive callers may
    hand in dicts; both must land in the same ``{"role", "content"}`` shape Studio auto-detects.
    """
    assert distill.serialize_message({"role": "user", "content": "hi"}) == {"role": "user", "content": "hi"}
    assert distill.serialize_message(_Msg("assistant", "<code>...</code>")) == {"role": "assistant", "content": "<code>...</code>"}


def test_serialize_message_unwraps_enum_roles_and_stringifies_content() -> None:
    """A ``MessageRole`` enum serializes to its plain value; non-string content is stringified.

    Why: ``str(MessageRole.USER)`` would leak ``"MessageRole.USER"`` into the training corpus,
    and multimodal part-lists would serialize as Python reprs — both poison a chat template.
    """
    record: dict[str, str] = distill.serialize_message(_Msg(_Role("system"), ["part-a", "part-b"]))
    assert record["role"] == "system"
    assert isinstance(record["content"], str)


def test_serialize_token_usage_extracts_counts_or_none() -> None:
    """Token usage rides along when the response carries it, else the field is ``None``."""
    assert distill.serialize_token_usage(_Msg("assistant", "x", _Usage(10, 5))) == {"input_tokens": 10, "output_tokens": 5}
    assert distill.serialize_token_usage(_Msg("assistant", "x")) is None
    assert distill.serialize_token_usage(None) is None


def test_recorder_writes_complete_chatml_records(tmp_path: Path) -> None:
    """Each record is one JSON line: full messages + appended assistant response + metadata.

    Why: Unsloth Studio auto-maps a top-level ``messages`` column on JSONL upload ONLY when every
    line is a standalone object (no outer array), and the assistant reply must be the final
    message for the line to be a complete SFT example.
    """
    path: Path = tmp_path / "nested" / distill.RECORDS_FILENAME  # parent created lazily
    recorder = distill.DistillRecorder(path)
    recorder.record(
        "agent",
        [_Msg(_Role("system"), "You derive configs."), _Msg("user", "Derive PMC1.")],
        _Msg(_Role("assistant"), "<code>final_answer(...)</code>", _Usage(100, 20)),
        pmc_id="PMC1",
        model_id="big-model",
    )

    records = _read_records(path)
    assert len(records) == 1
    record = records[0]
    assert [m["role"] for m in record["messages"]] == ["system", "user", "assistant"]
    assert record["messages"][-1]["content"] == "<code>final_answer(...)</code>"
    assert record["purpose"] == "agent"
    assert record["pmc_id"] == "PMC1"
    assert record["model_id"] == "big-model"
    assert record["call_index"] == 0
    assert record["token_usage"] == {"input_tokens": 100, "output_tokens": 20}
    assert isinstance(record["timestamp"], str)


def test_recorder_appends_across_invocations(tmp_path: Path) -> None:
    """A SECOND recorder over the same file appends — the dataset accumulates run over run.

    Why: the fine-tuning corpus is built by repeated ``tablassert agent --distill`` invocations;
    each constructs a fresh ``DistillRecorder``, so append (never truncate) is the load-bearing
    behavior. ``call_index`` restarts per invocation; ``timestamp`` disambiguates runs.
    """
    path: Path = tmp_path / distill.RECORDS_FILENAME
    first = distill.DistillRecorder(path)
    first.record("agent", [_Msg("user", "run one")], _Msg("assistant", "a"), pmc_id="PMC1")
    second = distill.DistillRecorder(path)  # a later invocation
    second.record("judge", [_Msg("user", "run two")], _Msg("assistant", "b"))

    records = _read_records(path)
    assert len(records) == 2
    assert [r["purpose"] for r in records] == ["agent", "judge"]
    assert [r["call_index"] for r in records] == [0, 0]  # per-invocation counter


def test_recorder_without_response_still_records(tmp_path: Path) -> None:
    """A call with no response object records the prompt messages alone (best-effort)."""
    path: Path = tmp_path / distill.RECORDS_FILENAME
    distill.DistillRecorder(path).record("reflexion", [_Msg("user", "propose an edit")])
    (record,) = _read_records(path)
    assert [m["role"] for m in record["messages"]] == ["user"]
    assert record["token_usage"] is None


def test_recorder_never_raises_into_the_run(tmp_path: Path) -> None:
    """An unwritable sink (path IS a directory) logs and swallows instead of breaking a batch.

    Why: recording is observability, not pipeline logic — a distillation failure must never skip
    an article mid-supervisor-run.
    """
    recorder = distill.DistillRecorder(tmp_path)  # opening a directory for append fails
    recorder.record("agent", [_Msg("user", "x")], _Msg("assistant", "y"))  # must not raise


def test_distill_dir_is_a_pure_path_helper() -> None:
    """``distill_dir(root)`` is ``root/distill`` with no I/O, matching the sibling helpers."""
    root: Path = Path(".tablassert") / "agent"
    assert distill_dir(root) == root / "distill"
    assert not distill_dir(root).exists()  # pure: nothing created


def test_distilling_model_records_every_generate_call(tmp_path: Path) -> None:
    """The wrapper delegates to the real model and appends one tagged record per call.

    Why: wrapping ``generate`` is the single capture seam — the inner agent, judge, and reflexion
    all route through it. The wrapped model's ``model_id`` is recovered for the record, and the
    response passes through untouched (the run must behave exactly as if unwrapped).
    """
    pytest.importorskip("smolagents")
    recorder = distill.DistillRecorder(tmp_path / distill.RECORDS_FILENAME)
    wrapped = make_distilling_model(make_fake_model(), recorder, purpose="agent", meta={"pmc_id": "PMC7"})

    response = wrapped.generate([_Msg(_Role("user"), "derive a config")])  # pyright: ignore[reportAttributeAccessIssue]
    assert "final_answer" in str(getattr(response, "content", ""))  # the fake's answer passes through
    wrapped.generate([_Msg(_Role("user"), "second call")])  # pyright: ignore[reportAttributeAccessIssue]

    records = _read_records(recorder.path)
    assert len(records) == 2
    for index, record in enumerate(records):
        assert record["purpose"] == "agent"
        assert record["pmc_id"] == "PMC7"
        assert record["call_index"] == index
        assert record["messages"][-1]["role"] == "assistant"
    assert "final_answer" in records[0]["messages"][-1]["content"]


def test_distilling_model_survives_a_raising_recorder(tmp_path: Path) -> None:
    """A recorder that raises does not change the wrapped model's behavior."""

    class ExplodingRecorder:
        def record(self, *args: object, **kwargs: object) -> None:
            raise OSError("disk full")

    pytest.importorskip("smolagents")
    wrapped = make_distilling_model(make_fake_model(), ExplodingRecorder(), purpose="agent")
    response = wrapped.generate([_Msg("user", "still works")])  # pyright: ignore[reportAttributeAccessIssue]
    assert "final_answer" in str(getattr(response, "content", ""))


def test_distill_export_writes_an_hf_dataset(tmp_path: Path) -> None:
    """``distill-export`` turns recorded NDJSON into a ``save_to_disk`` dataset.

    Why: the raw NDJSON already loads in Studio; this command exists for ``datasets``-native
    workflows, and its output must round-trip through ``load_from_disk``.
    """
    pytest.importorskip("datasets")
    from datasets import load_from_disk  # pyright: ignore[reportMissingImports]

    from tablassert.cli import distill_export

    ndjson_dir: Path = tmp_path / "distill"
    recorder = distill.DistillRecorder(ndjson_dir / distill.RECORDS_FILENAME)
    recorder.record("agent", [_Msg("user", "derive"), _Msg("assistant", "thinking")], _Msg("assistant", "done"), pmc_id="PMC1")
    out: Path = tmp_path / "hf-dataset"

    distill_export(distill_dir=ndjson_dir, out=out)

    dataset = load_from_disk(str(out))
    assert len(dataset) == 1
    assert dataset[0]["purpose"] == "agent"
    assert dataset[0]["messages"][-1]["content"] == "done"


def test_distill_export_fails_loud_on_an_empty_dir(tmp_path: Path) -> None:
    """No recorded NDJSON means exit 2 naming the fix — not a cryptic datasets error."""
    from tablassert.cli import distill_export

    with pytest.raises(SystemExit) as exc_info:
        distill_export(distill_dir=tmp_path / "empty", out=tmp_path / "out")
    assert exc_info.value.code == 2
