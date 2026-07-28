from __future__ import annotations

from pathlib import Path

from tablassert.errors import DOCS_URL, BabelDownloadError, GraphValidationError, QcRuntimeMissingError, SectionValidationError


def test_qc_runtime_missing_error_code_and_docs_url() -> None:
    """Guard: the QC-runtime failure carries a stable slug and a docs link.

    A user who hits this mid-build needs the `qc-runtime-missing` code and a docs
    URL pointing straight at the fix (install tablassert[qc]), not a bare traceback.
    """
    err: QcRuntimeMissingError = QcRuntimeMissingError()
    assert err.code == "qc-runtime-missing"
    assert str(err).endswith(DOCS_URL + "qc-runtime-missing")


def test_graph_validation_error_code_and_docs_url() -> None:
    """Guard: a rejected graph config carries a stable slug and a docs link.

    The `graph-validation-failed` code plus docs URL is what tells a user their graph
    YAML is invalid before a multi-hour build ever starts.
    """
    err: GraphValidationError = GraphValidationError(Path("graph.yaml"), "tables field is required")
    assert err.code == "graph-validation-failed"
    assert str(err).endswith(DOCS_URL + "graph-validation-failed")


def test_section_validation_error_code_and_docs_url() -> None:
    """Guard: a rejected table section carries a stable slug and a docs link.

    The `section-validation-failed` code plus docs URL is what tells a user which table
    section is invalid before a multi-hour build ever starts.
    """
    err: SectionValidationError = SectionValidationError(Path("table.yaml"), "0123456789abcdef", "source field is required")
    assert err.code == "section-validation-failed"
    assert str(err).endswith(DOCS_URL + "section-validation-failed")


def test_babel_download_error_code_and_docs_url() -> None:
    """Guard: an exhausted BABEL download carries a stable slug and a docs link.

    The `babel-download-failed` code plus docs URL is what tells a user a network fetch
    gave up after retries, and where to read about pinning a different BABEL version.
    """
    err: BabelDownloadError = BabelDownloadError("https://stars.renci.org/var/babel_outputs/x.gz", 5, RuntimeError("network down"))
    assert err.code == "babel-download-failed"
    assert str(err).endswith(DOCS_URL + "babel-download-failed")
