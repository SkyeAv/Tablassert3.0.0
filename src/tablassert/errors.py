from __future__ import annotations

from pathlib import Path
from typing import Literal

DOCS_URL: str = "https://tablassert.readthedocs.io/errors/"

TablassertErrorCodes = Literal[
    "qc-runtime-missing",
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
    "graph-bad-infores",
    "override-bad-publication",
    "override-bad-upstream-infores",
    "provenance-bad-pmc-id",
    "provenance-missing-publication",
    "provenance-publication-and-override",
    "encoding-list-requires-list",
    "encoding-list-incompatible-ops",
    "encoding-list-annotation-only",
    "qualifier-auto-derived",
    "qualifier-bad-value",
    "qualifier-unsatisfiable",
]


class _Coded:
    """Mixin providing a stable kebab-case slug and an auto-appended docs URL on ``str()``."""

    message: str
    code: TablassertErrorCodes

    def __str__(self) -> str:
        return f"{self.message}\n\nFor further information visit {DOCS_URL}{self.code}"


class TablassertError(_Coded, RuntimeError):
    def __init__(self, message: str, *, code: TablassertErrorCodes) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class TablassertValidationError(_Coded, ValueError):
    """Base class for validation failures raised through Pydantic v2.

    Notes:
        Inherits ``ValueError`` so Pydantic v2 still wraps it inside ``ValidationError``.
    """

    def __init__(self, message: str, *, code: TablassertErrorCodes) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class BiolinkRelocationWarning(UserWarning):
    """An annotation is valid but will not land on the edge under its own name.

    Distinct from a deprecation: nothing is wrong with the config and nothing is lost. The value is
    relocated -- onto the inlined ``StudyResult`` for a slot Biolink attaches to no class, or into
    ``supporting_text`` for a name that is not an association slot at all. Its own category so
    callers can silence or assert on relocations without touching the deprecation scaffold.
    """


class QcRuntimeMissingError(TablassertError):
    def __init__(self) -> None:
        super().__init__("QC requires optional runtime dependencies. Install tablassert[qc].", code="qc-runtime-missing")


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
