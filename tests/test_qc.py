from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz import process as rf_process
from sklearn.metrics import pairwise

import tablassert.qc as qc


def test_fullmap_audit_suppresses_logs(monkeypatch: Any) -> None:
    """fullmap_audit suppresses failed QC logs when disabled."""
    messages: list[str] = []

    class DummyLogger:
        def info(self, message: str, *args: Any, **kwargs: Any) -> None:
            messages.append(message.format(*args, **kwargs) if kwargs else message)

    class DummyBioBERT:
        def encode(self, values: list[str]) -> object:
            return np.array([[0.0], [1.0]])

    def fake_cpdist(left: list[str], right: list[str], scorer: Any) -> object:
        return np.array([0.0])

    def fake_cosine_similarity(left: object, right: object) -> object:
        return np.array([[0.1]])

    monkeypatch.setattr(qc, "logger", DummyLogger())
    monkeypatch.setattr(qc, "get_biobert", lambda: DummyBioBERT())
    monkeypatch.setattr(rf_process, "cpdist", fake_cpdist)
    monkeypatch.setattr(pairwise, "cosine_similarity", fake_cosine_similarity)

    lf: pl.LazyFrame = pl.DataFrame({"subject": ["FOO:1"], "subject_pre_resolution": ["foo"], "subject_name": ["bar"]}).lazy()
    result: pl.DataFrame = qc.fullmap_audit(lf, "subject", "store123", "config.yaml", log=False).collect()

    assert result.height == 0
    assert messages == []


def test_fullmap_audit_logs_failures(monkeypatch: Any) -> None:
    """fullmap_audit logs failed QC rows when enabled."""
    messages: list[str] = []

    class DummyLogger:
        def info(self, message: str, *args: Any, **kwargs: Any) -> None:
            messages.append(message.format(*args, **kwargs) if kwargs else message)

    class DummyBioBERT:
        def encode(self, values: list[str]) -> object:
            return np.array([[0.0], [1.0]])

    def fake_cpdist(left: list[str], right: list[str], scorer: Any) -> object:
        assert scorer in {fuzz.ratio, fuzz.partial_token_sort_ratio}
        return np.array([0.0])

    def fake_cosine_similarity(left: object, right: object) -> object:
        return np.array([[0.1]])

    monkeypatch.setattr(qc, "logger", DummyLogger())
    monkeypatch.setattr(qc, "get_biobert", lambda: DummyBioBERT())
    monkeypatch.setattr(rf_process, "cpdist", fake_cpdist)
    monkeypatch.setattr(pairwise, "cosine_similarity", fake_cosine_similarity)

    lf: pl.LazyFrame = pl.DataFrame({"subject": ["FOO:1"], "subject_pre_resolution": ["foo"], "subject_name": ["bar"]}).lazy()
    result: pl.DataFrame = qc.fullmap_audit(lf, "subject", "store123", "config.yaml", log=True).collect()

    assert result.height == 0
    assert len(messages) == 1
    assert "QC rejected" in messages[0]
    assert "hash=store123" in messages[0]
    assert "config=config.yaml" in messages[0]
    assert "fuzz=" in messages[0]
    assert "bert=" in messages[0]


def test_fullmap_audit_log_score_values(monkeypatch: Any) -> None:
    """fullmap_audit log message contains expected score values."""
    messages: list[str] = []

    class DummyLogger:
        def info(self, message: str, *args: Any, **kwargs: Any) -> None:
            messages.append(message.format(*args, **kwargs) if kwargs else message)

    class DummyBioBERT:
        def encode(self, values: list[str]) -> object:
            return np.array([[0.0], [1.0]])

    def fake_cpdist(left: list[str], right: list[str], scorer: Any) -> object:
        return np.array([5.0]) if scorer == fuzz.ratio else np.array([8.0])

    def fake_cosine_similarity(left: object, right: object) -> object:
        return np.array([[0.05]])

    monkeypatch.setattr(qc, "logger", DummyLogger())
    monkeypatch.setattr(qc, "get_biobert", lambda: DummyBioBERT())
    monkeypatch.setattr(rf_process, "cpdist", fake_cpdist)
    monkeypatch.setattr(pairwise, "cosine_similarity", fake_cosine_similarity)

    lf: pl.LazyFrame = pl.DataFrame({"subject": ["X:1"], "subject_pre_resolution": ["foo"], "subject_name": ["bar"]}).lazy()
    result: pl.DataFrame = qc.fullmap_audit(lf, "subject", "store456", "cfg.yaml", log=True).collect()

    assert result.height == 0
    assert len(messages) == 1
    assert "fuzz=8.0" in messages[0]
    assert "bert=0.05" in messages[0]
