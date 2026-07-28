"""Offline unit tests for US-002 ``fetch_pmc_tables`` (PMC ``s3://pmc-oa-opendata``).

Everything here runs with NO network and NO ``[agent]`` extra: the PURE parsers are
exercised directly, and ``fetch_pmc_tables`` runs against the two mocked I/O seams
(``_http_get_text`` / ``_http_get_bytes``) routed by URL. Canned fixtures mirror the
live-verified bucket layout: a list-objects-v2 listing (XML *and* JSON), a JATS
``<supplementary-material>`` with one ``.xlsx`` table + one ``.jpg`` (drop-wins), and
CC-BY vs non-OA ``.json`` metadata.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tablassert.agent import (
    fetch_pmc_tables,
    is_open_access,
    is_table_file,
    normalize_pmc_id,
    public_url,
    table_files_from_jats,
    version_prefixes_from_listing,
)

# A realistic list-objects-v2 XML body. The echoed request ``<Prefix>PMC11708054.``
# (no trailing slash) must be filtered out; only the ``CommonPrefixes`` entry survives.
LISTING_XML: str = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
    "<Name>pmc-oa-opendata</Name>"
    "<Prefix>PMC11708054.</Prefix>"
    "<KeyCount>1</KeyCount>"
    "<CommonPrefixes><Prefix>PMC11708054.1/</Prefix></CommonPrefixes>"
    "</ListBucketResult>"
)
LISTING_JSON: str = '{"CommonPrefixes": [{"Prefix": "PMC11708054.1/"}]}'

# One supplementary-material carrying a real table (.xlsx, "Table S1") AND an image
# (.jpg) that shares the label: the image must be dropped (extension drop wins).
JATS_XML: str = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<article xmlns:xlink="http://www.w3.org/1999/xlink" article-type="research-article">'
    "<body><sec>"
    '<supplementary-material id="sm1">'
    "<label>Table S1</label>"
    "<caption><title>Supplementary table for the article.</title></caption>"
    '<media xlink:href="mbio.01679-24-s0006.xlsx"/>'
    '<media xlink:href="fig1.jpg"/>'
    "</supplementary-material>"
    "</sec></body>"
    "</article>"
)
JATS_NO_SM: str = '<?xml version="1.0"?><article><body><p>No supplementary material here.</p></body></article>'

METADATA_OA: str = '{"is_pmc_openaccess": true, "license_code": "CC-BY"}'
METADATA_NON_OA: str = '{"is_pmc_openaccess": false}'

XLSX_BYTES: bytes = b"FAKEXLSX"


def _patch_http(monkeypatch: pytest.MonkeyPatch, *, listing: str = LISTING_XML, metadata: str = METADATA_OA, jats: str = JATS_XML) -> None:
    """Route the two mocked HTTP seams by URL so ``fetch_pmc_tables`` runs offline.

    WHY: ``_http_get_text``/``_http_get_bytes`` are the SINGLE I/O seam. Routing by URL
    substring lets one fake serve the listing, the ``.json`` metadata and the ``.xml``
    JATS for any version prefix without hard-coding the full S3 URL, so each test only
    overrides the one fixture it cares about (empty listing / non-OA / no tables).
    """

    def get_text(url: str, *, timeout: int = 120) -> str:
        if "list-type=2" in url:
            return listing
        if url.endswith(".json"):
            return metadata
        if url.endswith(".xml"):
            return jats
        raise AssertionError(f"unexpected text url: {url}")

    def get_bytes(url: str, *, timeout: int = 120) -> bytes:
        assert url.endswith(".xlsx")
        return XLSX_BYTES

    monkeypatch.setattr("tablassert.agent._http_get_text", get_text)
    monkeypatch.setattr("tablassert.agent._http_get_bytes", get_bytes)


@pytest.mark.parametrize("raw", ["PMC11708054", "11708054", "pmc11708054", " PMC11708054 "])
def test_normalize_pmc_id_accepts_common_forms(raw: str) -> None:
    """A bare number, either case prefix, and surrounding whitespace all normalize.

    WHY: callers paste PMC ids in inconsistent shapes; the S3 prefix needs exactly
    ``PMC<n>`` so every common form must collapse to the canonical id.
    """
    assert normalize_pmc_id(raw) == "PMC11708054"


@pytest.mark.parametrize("raw", ["abc", "", "PMC"])
def test_normalize_pmc_id_rejects_garbage(raw: str) -> None:
    """No remaining digits -> a loud ``ValueError`` instead of a bogus S3 prefix."""
    with pytest.raises(ValueError, match="Invalid PMC id"):
        normalize_pmc_id(raw)


def test_version_prefixes_from_listing_xml() -> None:
    """XML listing -> the single version prefix; the echoed request prefix is dropped."""
    assert version_prefixes_from_listing(LISTING_XML) == ["PMC11708054.1/"]


def test_version_prefixes_from_listing_json() -> None:
    """The JSON variant some clients return parses to the same prefix list."""
    assert version_prefixes_from_listing(LISTING_JSON) == ["PMC11708054.1/"]


@pytest.mark.parametrize("listing", ["<<garbage>>", "", "{not valid json"])
def test_version_prefixes_from_listing_bad(listing: str) -> None:
    """Garbage / empty / malformed bodies yield ``[]`` (never a raise)."""
    assert version_prefixes_from_listing(listing) == []


@pytest.mark.parametrize(
    ("filename", "label", "expected"),
    [
        ("a.xlsx", None, True),
        ("a.csv", None, True),
        ("a.tsv", None, True),
        ("fig.jpg", "Table 1", False),  # DROP extension wins over a "Table" label
        ("data.bin", "Table S2", True),  # label-only match for an unknown extension
        ("data.bin", None, False),
        ("A.XLSX", None, True),  # case-insensitive extension
    ],
)
def test_is_table_file(filename: str, label: str | None, expected: bool) -> None:
    """Extension drop > table extension > ``Table`` label, case-insensitively."""
    assert is_table_file(filename, label) is expected


def test_table_files_from_jats_realistic() -> None:
    """Only the ``.xlsx`` survives; the label-sharing ``.jpg`` is dropped."""
    assert table_files_from_jats(JATS_XML) == [{"href": "mbio.01679-24-s0006.xlsx", "label": "Table S1", "is_table": True}]


def test_table_files_from_jats_no_supplementary() -> None:
    """No ``<supplementary-material>`` -> ``[]``."""
    assert table_files_from_jats(JATS_NO_SM) == []


def test_table_files_from_jats_malformed() -> None:
    """Malformed XML returns ``[]`` rather than raising into the caller."""
    assert table_files_from_jats("<not xml") == []


def test_is_open_access_cc_by_json() -> None:
    """CC-BY metadata string is open access."""
    assert is_open_access(METADATA_OA) is True


def test_is_open_access_non_oa() -> None:
    """``is_pmc_openaccess: false`` with no CC license is NOT open access."""
    assert is_open_access(METADATA_NON_OA) is False


def test_is_open_access_dict_input() -> None:
    """A pre-parsed dict is accepted on the same code path."""
    assert is_open_access({"is_pmc_openaccess": True, "license_code": "CC-BY"}) is True


def test_is_open_access_bad_json() -> None:
    """An unparseable metadata string is treated as non-OA (False), never a raise."""
    assert is_open_access("{not json") is False


def test_public_url() -> None:
    """Trailing/leading slashes are normalized into a single public HTTPS URL."""
    assert public_url("PMC11708054.1/", "f.xlsx") == "https://pmc-oa-opendata.s3.amazonaws.com/PMC11708054.1/f.xlsx"


def test_fetch_pmc_tables_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """OA article with one table -> the file is really written under ``outdir/<prefix>``.

    WHY: this is the end-to-end contract (list -> metadata -> JATS -> download) with
    the network mocked. Asserting the bytes on disk proves the download path wired the
    routed ``_http_get_bytes`` output to the correct local destination.
    """
    _patch_http(monkeypatch)
    outdir: Path = tmp_path / "tables"
    result: list[Path] = fetch_pmc_tables("PMC11708054", outdir)
    expected: Path = outdir / "PMC11708054.1" / "mbio.01679-24-s0006.xlsx"
    assert result == [expected]
    assert expected.is_file()
    assert expected.read_bytes() == XLSX_BYTES


def test_fetch_pmc_tables_empty_listing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No version prefixes -> ``FileNotFoundError`` (not OA or wrong id)."""
    empty: str = '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Prefix>PMC999.</Prefix></ListBucketResult>'
    _patch_http(monkeypatch, listing=empty)
    with pytest.raises(FileNotFoundError, match="No PMC open-access versions"):
        fetch_pmc_tables("PMC999", tmp_path)


def test_fetch_pmc_tables_non_oa(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Readable metadata with no CC license -> ``PermissionError`` even if a table exists."""
    _patch_http(monkeypatch, metadata=METADATA_NON_OA)
    with pytest.raises(PermissionError, match="not open access"):
        fetch_pmc_tables("PMC11708054", tmp_path)


def test_fetch_pmc_tables_no_tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """OA article whose JATS has no supplementary tables -> ``FileNotFoundError``."""
    _patch_http(monkeypatch, jats=JATS_NO_SM)
    with pytest.raises(FileNotFoundError, match="No supplementary tables"):
        fetch_pmc_tables("PMC11708054", tmp_path)


def test_fetch_pmc_tables_bad_id(tmp_path: Path) -> None:
    """A bad id raises ``ValueError`` before any HTTP is attempted (no mock needed)."""
    with pytest.raises(ValueError, match="Invalid PMC id"):
        fetch_pmc_tables("not-an-id", tmp_path)
