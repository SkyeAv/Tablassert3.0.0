"""Distillation capture: record agent LLM traffic as an HF-ready ChatML NDJSON dataset.

When ``tablassert agent --distill`` is set, every ``model.generate`` call made during the run
(inner CodeAgent turns, the semantic judge, tier-2 reflexion) is appended as ONE JSON object per
line to ``<state_dir>/distill/records.ndjson``. The schema is ChatML — a top-level ``messages``
list of ``{"role", "content"}`` dicts — which Unsloth Studio auto-detects on JSONL upload and
``datasets.load_dataset("json", ...)`` loads directly; metadata keys (``purpose``, ``pmc_id``,
``model_id``, ``call_index``, ``timestamp``, ``token_usage``) ride alongside as extra columns for
filtering (e.g. keeping only MAPPED runs). Strict JSONL: no outer array, no commas between lines.

This module is ZERO-dependency by design — recording must work in any install that can run the
agent, and must NEVER break a batch: every write is guarded and failures only log.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from tablassert.log import cat

logger = cat("AGENT")

#: Default dataset filename inside ``distill_dir(state_dir)``; append-only across runs so a
#: fine-tuning corpus accumulates over many invocations.
RECORDS_FILENAME: Final[str] = "records.ndjson"


def _role_name(role: object) -> str:
    """Coerce a message role (``MessageRole`` enum or plain string) to its plain name."""
    value: object = getattr(role, "value", role)
    return str(value)


def serialize_message(message: object) -> dict[str, str]:
    """Serialize one smolagents ``ChatMessage`` (or ``{"role", "content"}`` dict) to ChatML.

    All access is guarded so an unexpected message shape degrades to a best-effort dict rather
    than raising into the run. Non-string content (e.g. multimodal part lists) is stringified.
    """
    if isinstance(message, dict):
        role: object = message.get("role", "user")
        content: object = message.get("content", "")
    else:
        role = getattr(message, "role", "user")
        content = getattr(message, "content", "")
    return {"role": _role_name(role), "content": content if isinstance(content, str) else str(content)}


def serialize_messages(messages: object) -> list[dict[str, str]]:
    """Serialize a ``model.generate`` message list to a ChatML ``messages`` column."""
    if not isinstance(messages, (list, tuple)):
        return []
    return [serialize_message(message) for message in messages]


def serialize_token_usage(response: object) -> dict[str, int] | None:
    """Extract ``{"input_tokens", "output_tokens"}`` from a response's ``token_usage``, if any."""
    usage: object = getattr(response, "token_usage", None)
    if usage is None:
        return None
    input_tokens: object = getattr(usage, "input_tokens", None)
    output_tokens: object = getattr(usage, "output_tokens", None)
    if not isinstance(input_tokens, (int, float)) and not isinstance(output_tokens, (int, float)):
        return None
    return {
        "input_tokens": int(input_tokens) if isinstance(input_tokens, (int, float)) else 0,
        "output_tokens": int(output_tokens) if isinstance(output_tokens, (int, float)) else 0,
    }


class DistillRecorder:
    """Append-only NDJSON sink for distillation records; never raises into the run.

    ``record(purpose, messages, response, **meta)`` serializes the conversation, APPENDS the
    assistant response as the final message (so each line is a complete ChatML training
    example), and writes one JSON line. The parent directory is created lazily on first write.
    ``call_index`` counts records written by THIS recorder (per invocation), letting downstream
    filtering keep only each run's final (most complete) record.
    """

    def __init__(self, path: Path) -> None:
        self.path: Path = Path(path)
        self.call_index: int = 0

    def record(self, purpose: str, messages: object, response: object = None, **meta: Any) -> None:
        """Append one record; any serialization or I/O failure is logged and swallowed."""
        try:
            conversation: list[dict[str, str]] = serialize_messages(messages)
            if response is not None:
                conversation.append(serialize_message(response))
            record: dict[str, object] = {
                "messages": conversation,
                "purpose": purpose,
                "call_index": self.call_index,
                "timestamp": datetime.now(UTC).isoformat(),
                "token_usage": serialize_token_usage(response),
            }
            record.update(meta)
            line: str = json.dumps(record, ensure_ascii=False, default=str)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            self.call_index += 1
        except Exception as exc:  # recording must never break a batch
            logger.warning(f"distill: failed to record {purpose} call to {self.path}: {exc}")
