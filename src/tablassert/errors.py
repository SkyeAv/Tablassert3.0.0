from __future__ import annotations

from pathlib import Path
from typing import Literal

DOCS_URL: str = "https://tablassert.readthedocs.io/errors/"

TablassertErrorCodes = Literal[
    "qc-runtime-missing",
    "qc-cuda-package-missing",
    "qc-cuda-provider-unavailable",
    "qc-cuda-no-cpu-fallback",
    "graph-validation-failed",
    "section-validation-failed",
    "babel-download-failed",
    "config-rows-and-row-slice-conflict",
    "comparison-bad-comparator-type",
    "comparison-nonnumeric-comparator",
    "encoding-bad-excel-column",
    "regex-bad-pattern",
    "regex-bad-replacement",
    "encoding-bad-remove-entry",
    "provenance-bad-pmc-id",
]


class _Coded:
    # ? Mixin: stable kebab-case slug + auto-appended docs URL on str().
    message: str
    code: "TablassertErrorCodes"

    def __str__(self) -> str:
        return f"{self.message}\n\nFor further information visit {DOCS_URL}{self.code}"


class TablassertError(_Coded, RuntimeError):
    def __init__(self, message: str, *, code: "TablassertErrorCodes") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class TablassertValidationError(_Coded, ValueError):
    # ! Inherits ValueError so Pydantic v2 still wraps it inside ValidationError.
    def __init__(self, message: str, *, code: "TablassertErrorCodes") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class QcRuntimeMissingError(TablassertError):
    def __init__(self) -> None:
        super().__init__("QC requires optional runtime dependencies. Install tablassert[qc] or tablassert[qc-cuda].", code="qc-runtime-missing")


class QcCudaPackageMissingError(TablassertError):
    def __init__(self) -> None:
        super().__init__("QC requested CUDA but onnxruntime-gpu is not installed. Install tablassert[qc-cuda].", code="qc-cuda-package-missing")


class QcCudaProviderUnavailableError(TablassertError):
    def __init__(self, *, requested: bool) -> None:
        verb: str = "requested" if requested else "detected"
        super().__init__(
            f"QC {verb} onnxruntime-gpu but CUDAExecutionProvider is unavailable. Verify the CUDA/cuDNN environment for tablassert[qc-cuda].",
            code="qc-cuda-provider-unavailable",
        )


class QcCudaNoCpuFallbackError(TablassertError):
    def __init__(self) -> None:
        super().__init__(
            "Detected onnxruntime-gpu but CUDAExecutionProvider is unavailable. "
            "Tablassert will not fall back to CPU from qc-cuda. "
            "Install tablassert[qc] or fix the CUDA/cuDNN environment.",
            code="qc-cuda-no-cpu-fallback",
        )


class GraphValidationError(TablassertError):
    def __init__(self, config: Path, detail: str) -> None:
        super().__init__(f"Graph config validation failed: {config}\n{detail}", code="graph-validation-failed")


class SectionValidationError(TablassertError):
    def __init__(self, config: Path, section_hash: str, detail: str) -> None:
        super().__init__(f"Section validation failed: {config} (hash {section_hash[:8]})\n{detail}", code="section-validation-failed")


class BabelDownloadError(TablassertError):
    def __init__(self, url: str, retries: int, last_error: BaseException) -> None:
        super().__init__(
            f"BABEL download failed after {retries} attempts: {url} (last error: {last_error}). Check network connectivity or pin a different BABEL version.",
            code="babel-download-failed",
        )
