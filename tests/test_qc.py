from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import polars as pl
import pytest
from rapidfuzz import fuzz
from rapidfuzz import process as rf_process
from sklearn.metrics import pairwise

import tablassert.qc as qc


# ? QC Provider Defaults To CPU Runtime
def test_get_qc_provider_prefers_cpu_runtime(monkeypatch: Any) -> None:
    monkeypatch.setattr(qc, "has_qc_runtime", lambda name: name == "onnxruntime")

    provider: tuple[str, Optional[dict[str, object]]] = qc.get_qc_provider()

    assert provider == (qc.CPU_PROVIDER, None)


# ? QC Provider Prefers CUDA Runtime When Available
def test_get_qc_provider_prefers_cuda_runtime(monkeypatch: Any) -> None:
    class DummyOrt:
        def get_available_providers(self) -> list[str]:
            return [qc.CUDA_PROVIDER, qc.CPU_PROVIDER]

    monkeypatch.setattr(qc, "ort", DummyOrt())
    monkeypatch.setattr(qc, "has_qc_runtime", lambda name: name == "onnxruntime-gpu")

    provider: tuple[str, Optional[dict[str, object]]] = qc.get_qc_provider()

    assert provider == (qc.CUDA_PROVIDER, {"device_id": 0})


# ? QC CUDA Path Hard-Fails Without CUDA Provider
def test_get_qc_provider_hardfails_when_cuda_unavailable(monkeypatch: Any) -> None:
    class DummyOrt:
        def get_available_providers(self) -> list[str]:
            return [qc.CPU_PROVIDER]

    monkeypatch.setattr(qc, "ort", DummyOrt())
    monkeypatch.setattr(qc, "has_qc_runtime", lambda name: name == "onnxruntime-gpu")

    with pytest.raises(RuntimeError, match="will not fall back to CPU"):
        qc.get_qc_provider()


# ? BioBERT Threads Provider Options Into SentenceTransformer
def test_get_biobert_threads_provider_options(monkeypatch: Any, tmp_path: Path) -> None:
    class DummySessionOptions:
        def __init__(self) -> None:
            self.graph_optimization_level: object = None

    class DummyGraphOptimizationLevel:
        ORT_ENABLE_ALL: str = "all"

    class DummyOrt:
        SessionOptions = DummySessionOptions
        GraphOptimizationLevel = DummyGraphOptimizationLevel

    captured: list[dict[str, object]] = []

    class DummySentenceTransformer:
        def __init__(self, model_name: str, backend: str, model_kwargs: dict[str, object]) -> None:
            captured.append(model_kwargs)

        def save(self, model: Path) -> None:
            return None

    monkeypatch.setattr(qc, "ort", DummyOrt())
    monkeypatch.setattr(
        qc, "sentence_transformers", type("DummyST", (), {"SentenceTransformer": DummySentenceTransformer})
    )
    monkeypatch.setattr(qc, "get_qc_provider", lambda provider=None: (qc.CUDA_PROVIDER, {"device_id": 0}))
    monkeypatch.setattr(qc, "MODEL", tmp_path / ".onnxassert")
    monkeypatch.setattr(qc, "BIOBERT", {})

    qc.get_biobert("cuda")

    assert captured[0]["provider"] == qc.CUDA_PROVIDER
    assert captured[0]["provider_options"] == {"device_id": 0}


# ? fullmap_audit Suppresses Failed QC Logs When Disabled
def test_fullmap_audit_suppresses_logs(monkeypatch: Any) -> None:
    messages: list[str] = []

    class DummyLogger:
        def info(self, message: str) -> None:
            messages.append(message)

    class DummyBioBERT:
        def encode(self, values: list[str]) -> object:
            return np.array([[0.0], [1.0]])

    def fake_cpdist(left: list[str], right: list[str], scorer: Any) -> object:
        return np.array([0.0])

    def fake_cosine_similarity(left: object, right: object) -> object:
        return np.array([[0.1]])

    monkeypatch.setattr(qc, "logger", DummyLogger())
    monkeypatch.setattr(qc, "get_biobert", lambda provider=None: DummyBioBERT())
    monkeypatch.setattr(rf_process, "cpdist", fake_cpdist)
    monkeypatch.setattr(pairwise, "cosine_similarity", fake_cosine_similarity)

    lf: pl.LazyFrame = pl.DataFrame(
        {"subject": ["MONDO:1"], "original subject": ["foo"], "subject name": ["bar"]}
    ).lazy()
    result: pl.DataFrame = qc.fullmap_audit(lf, "subject", "store123", "config.yaml", log=False).collect()

    assert result.height == 0
    assert messages == []


# ? fullmap_audit Logs Failed QC Rows When Enabled
def test_fullmap_audit_logs_failures(monkeypatch: Any) -> None:
    messages: list[str] = []

    class DummyLogger:
        def info(self, message: str) -> None:
            messages.append(message)

    class DummyBioBERT:
        def encode(self, values: list[str]) -> object:
            return np.array([[0.0], [1.0]])

    def fake_cpdist(left: list[str], right: list[str], scorer: Any) -> object:
        assert scorer in {fuzz.ratio, fuzz.partial_token_sort_ratio}
        return np.array([0.0])

    def fake_cosine_similarity(left: object, right: object) -> object:
        return np.array([[0.1]])

    monkeypatch.setattr(qc, "logger", DummyLogger())
    monkeypatch.setattr(qc, "get_biobert", lambda provider=None: DummyBioBERT())
    monkeypatch.setattr(rf_process, "cpdist", fake_cpdist)
    monkeypatch.setattr(pairwise, "cosine_similarity", fake_cosine_similarity)

    lf: pl.LazyFrame = pl.DataFrame(
        {"subject": ["MONDO:1"], "original subject": ["foo"], "subject name": ["bar"]}
    ).lazy()
    result: pl.DataFrame = qc.fullmap_audit(lf, "subject", "store123", "config.yaml", log=True).collect()

    assert result.height == 0
    assert len(messages) == 1
    assert "FAILED" in messages[0]
    assert "STORE: store123" in messages[0]
    assert "CONFIG: config.yaml" in messages[0]


# ? GPU Runtime Can Be Forced To CPU
@pytest.mark.gpu
@pytest.mark.network
def test_get_biobert_cpu_on_gpu_runtime(monkeypatch: Any) -> None:
    if not qc.has_qc_runtime("onnxruntime-gpu"):
        pytest.skip("onnxruntime-gpu is not installed")
    if qc.CUDA_PROVIDER not in qc.ort.get_available_providers():  # pyright: ignore
        pytest.skip("CUDAExecutionProvider is unavailable")

    monkeypatch.setattr(qc, "BIOBERT", {})
    model: object = qc.get_biobert("cpu")
    embeddings: object = model.encode(["BRCA1", "TP53"])  # pyright: ignore

    assert len(embeddings) == 2  # pyright: ignore


# ? GPU Runtime Executes On CUDA When Requested
@pytest.mark.gpu
@pytest.mark.network
def test_get_biobert_cuda_runtime(monkeypatch: Any) -> None:
    if not qc.has_qc_runtime("onnxruntime-gpu"):
        pytest.skip("onnxruntime-gpu is not installed")
    if qc.CUDA_PROVIDER not in qc.ort.get_available_providers():  # pyright: ignore
        pytest.skip("CUDAExecutionProvider is unavailable")

    monkeypatch.setattr(qc, "BIOBERT", {})
    model: object = qc.get_biobert("cuda")
    embeddings: object = model.encode(["BRCA1", "TP53"])  # pyright: ignore

    assert len(embeddings) == 2  # pyright: ignore
