"""Offline unit tests for US-002 ``fetch_pmc_article`` (PMC ``s3://pmc-oa-opendata``).

Everything here runs with NO network and NO ``[agent]`` extra: the PURE parsers are
exercised directly, and ``fetch_pmc_article`` runs against the two mocked I/O seams
(``_http_get_text`` / ``_http_get_bytes``) routed by URL. The fetch flow is a fail-fast
ladder: list version prefixes -> pick the LATEST -> check OA metadata -> enumerate the
version's objects -> confirm a table exists -> download ONLY the useful files (main text
``.xml/.nxml/.txt/.pdf``, ``.json`` metadata, and data tables; never images/``.docx``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tablassert.agent import (
    candidate_tables,
    fetch_pmc_article,
    fetch_pmc_tables,
    is_open_access,
    is_table_file,
    is_useful_file,
    latest_version_prefix,
    normalize_pmc_id,
    object_keys_from_listing,
    public_url,
    version_prefixes_from_listing,
)

# A realistic list-objects-v2 XML body for VERSION prefixes. The echoed request
# ``<Prefix>PMC11708054.`` (no trailing slash) must be filtered out; only the
# ``CommonPrefixes`` entry survives.
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

METADATA_OA: str = '{"is_pmc_openaccess": true, "license_code": "CC-BY"}'
METADATA_NON_OA: str = '{"is_pmc_openaccess": false}'

# The full object inventory of PMC11708054.1/ (mirrors the live bucket): main text
# (.json/.pdf/.txt/.xml), two data tables (.xlsx), one binary .docx and one .jpg figure.
OBJECT_KEYS: list[str] = [
    "PMC11708054.1/PMC11708054.1.json",
    "PMC11708054.1/PMC11708054.1.pdf",
    "PMC11708054.1/PMC11708054.1.txt",
    "PMC11708054.1/PMC11708054.1.xml",
    "PMC11708054.1/mbio.01679-24-s0001.docx",
    "PMC11708054.1/mbio.01679-24-s0002.xlsx",
    "PMC11708054.1/mbio.01679-24-s0003.xlsx",
    "PMC11708054.1/mbio.01679-24.f001.jpg",
]
# Only the useful files are downloaded (main text + metadata + tables; NOT .docx/.jpg).
USEFUL_NAMES: list[str] = [
    "PMC11708054.1.json",
    "PMC11708054.1.pdf",
    "PMC11708054.1.txt",
    "PMC11708054.1.xml",
    "mbio.01679-24-s0002.xlsx",
    "mbio.01679-24-s0003.xlsx",
]


def _object_listing(keys: list[str]) -> str:
    """Build a list-objects-v2 XML body whose ``<Contents><Key>`` entries are ``keys``."""
    contents: str = "".join(f"<Contents><Key>{key}</Key><Size>1</Size></Contents>" for key in keys)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        "<Name>pmc-oa-opendata</Name>"
        f"{contents}"
        "</ListBucketResult>"
    )


def _patch_http(
    monkeypatch: pytest.MonkeyPatch, *, version_listing: str = LISTING_XML, object_listing: str | None = None, metadata: str = METADATA_OA
) -> tuple[list[str], list[str]]:
    """Route the two mocked HTTP seams by URL; return (requested_text_urls, downloaded_urls).

    WHY: ``_http_get_text``/``_http_get_bytes`` are the SINGLE I/O seam. The version listing
    is the ``list-type=2`` call WITH ``delimiter=``; the object listing is ``list-type=2``
    WITHOUT a delimiter; ``.json`` is the metadata. Recording the URLs lets tests assert the
    fail-fast ORDER (e.g. a non-OA id never reaches the object listing nor downloads bytes).
    """
    requested: list[str] = []
    downloaded: list[str] = []
    objects: str = object_listing if object_listing is not None else _object_listing(OBJECT_KEYS)

    def get_text(url: str, *, timeout: int = 120) -> str:
        requested.append(url)
        if "list-type=2" in url and "delimiter=" in url:
            return version_listing
        if "list-type=2" in url:
            return objects
        if url.endswith(".json"):
            return metadata
        raise AssertionError(f"unexpected text url: {url}")

    def get_bytes(url: str, *, timeout: int = 120) -> bytes:
        downloaded.append(url)
        return b"FAKEBYTES"

    monkeypatch.setattr("tablassert.agent._http_get_text", get_text)
    monkeypatch.setattr("tablassert.agent._http_get_bytes", get_bytes)
    return requested, downloaded


# --------------------------------------------------------------------------- #
# normalize_pmc_id / version_prefixes_from_listing / is_table_file (unchanged)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("raw", ["PMC11708054", "11708054", "pmc11708054", " PMC11708054 "])
def test_normalize_pmc_id_accepts_common_forms(raw: str) -> None:
    """A bare number, either case prefix, and surrounding whitespace all normalize."""
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


# --------------------------------------------------------------------------- #
# New pure helpers: latest_version_prefix / object_keys_from_listing /
# is_useful_file / candidate_tables
# --------------------------------------------------------------------------- #


def test_latest_version_prefix_numeric() -> None:
    """Numeric compare: ``.10`` beats ``.2`` (a string sort would get this wrong)."""
    assert latest_version_prefix(["PMC1.1/", "PMC1.10/", "PMC1.2/"]) == "PMC1.10/"


def test_latest_version_prefix_single() -> None:
    """A single prefix is returned unchanged."""
    assert latest_version_prefix(["PMC11708054.1/"]) == "PMC11708054.1/"


def test_object_keys_from_listing_xml() -> None:
    """XML ``<Contents><Key>`` -> sorted unique keys."""
    keys: list[str] = object_keys_from_listing(_object_listing(["p/b.xlsx", "p/a.xml"]))
    assert keys == ["p/a.xml", "p/b.xlsx"]


def test_object_keys_from_listing_json() -> None:
    """The JSON variant parses to the same key list."""
    assert object_keys_from_listing('{"Contents": [{"Key": "p/a.xml"}, {"Key": "p/b.xlsx"}]}') == ["p/a.xml", "p/b.xlsx"]


def test_object_keys_from_listing_drops_folder_markers() -> None:
    """Keys ending in ``/`` (folder markers) and empty keys are dropped."""
    body: str = '{"Contents": [{"Key": "p/"}, {"Key": ""}, {"Key": "p/a.xml"}]}'
    assert object_keys_from_listing(body) == ["p/a.xml"]


@pytest.mark.parametrize("listing", ["<<garbage>>", "", "{not valid json"])
def test_object_keys_from_listing_bad(listing: str) -> None:
    """Garbage / empty / malformed bodies yield ``[]`` (never a raise)."""
    assert object_keys_from_listing(listing) == []


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("PMC1.1.xml", True),
        ("PMC1.1.nxml", True),
        ("PMC1.1.txt", True),
        ("PMC1.1.pdf", True),  # kept as MAIN TEXT (not a table)
        ("PMC1.1.json", True),
        ("s.xlsx", True),
        ("s.csv", True),
        ("s.tsv", True),
        ("fig.jpg", False),
        ("fig.png", False),
        ("suppl.docx", False),
        ("blob.bin", False),
    ],
)
def test_is_useful_file(filename: str, expected: bool) -> None:
    """Main text + metadata + data tables are useful; binary media is not."""
    assert is_useful_file(filename) is expected


def test_candidate_tables_returns_all_tables(tmp_path: Path) -> None:
    """Every table-extension path is returned (not just the first), in order."""
    files: list[Path] = [tmp_path / "a.xlsx", tmp_path / "b.jpg", tmp_path / "c.csv"]
    assert candidate_tables(files) == [tmp_path / "a.xlsx", tmp_path / "c.csv"]


def test_candidate_tables_none_raises(tmp_path: Path) -> None:
    """No table among the files -> ``FileNotFoundError`` (fail-fast when a fetch yields no tables)."""
    with pytest.raises(FileNotFoundError, match="No supplementary table"):
        candidate_tables([tmp_path / "a.xml", tmp_path / "b.jpg"])


# --------------------------------------------------------------------------- #
# fetch_pmc_article (mocked HTTP) — happy path + fail-fast ladder
# --------------------------------------------------------------------------- #


def test_fetch_pmc_article_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """OA article -> only the useful files land on disk under ``outdir/<key>`` (no .docx/.jpg)."""
    _, downloaded = _patch_http(monkeypatch)
    outdir: Path = tmp_path / "out"
    result: list[Path] = fetch_pmc_article("PMC11708054", outdir)

    assert sorted(p.name for p in result) == sorted(USEFUL_NAMES)
    # every returned path really exists under the version subdir
    for path in result:
        assert path.is_file()
        assert path.parent.name == "PMC11708054.1"
    # binary media were never downloaded
    assert not any(url.endswith((".jpg", ".docx")) for url in downloaded)
    # the useful files were
    assert sum(1 for url in downloaded if url.endswith((".xml", ".txt", ".pdf", ".json", ".xlsx"))) == len(USEFUL_NAMES)


def test_fetch_pmc_tables_wrapper_returns_only_tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The backward-compat wrapper filters the full payload down to data tables."""
    _patch_http(monkeypatch)
    result: list[Path] = fetch_pmc_tables("PMC11708054", tmp_path / "out")
    assert sorted(p.name for p in result) == ["mbio.01679-24-s0002.xlsx", "mbio.01679-24-s0003.xlsx"]


def test_fetch_pmc_article_latest_version_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With two versions, ONLY the latest (``PMC1.2``) is listed/downloaded; ``PMC1.1`` is ignored."""
    version_listing: str = (
        '<?xml version="1.0"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        "<CommonPrefixes><Prefix>PMC1.1/</Prefix></CommonPrefixes>"
        "<CommonPrefixes><Prefix>PMC1.2/</Prefix></CommonPrefixes>"
        "</ListBucketResult>"
    )
    objects: str = _object_listing(["PMC1.2/PMC1.2.xml", "PMC1.2/t.xlsx"])
    requested, downloaded = _patch_http(monkeypatch, version_listing=version_listing, object_listing=objects)

    result: list[Path] = fetch_pmc_article("PMC1", tmp_path / "out")
    assert sorted(p.name for p in result) == ["PMC1.2.xml", "t.xlsx"]
    # the object listing + metadata targeted PMC1.2, and nothing ever touched PMC1.1
    assert any("prefix=PMC1.2/" in url for url in requested)
    assert not any("PMC1.1" in url for url in requested)
    assert all("PMC1.2/" in url for url in downloaded)


def test_fetch_pmc_article_non_oa_fails_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-OA metadata -> ``PermissionError`` BEFORE the object listing and BEFORE any byte download."""
    requested, downloaded = _patch_http(monkeypatch, metadata=METADATA_NON_OA)
    with pytest.raises(PermissionError, match="not open access"):
        fetch_pmc_article("PMC11708054", tmp_path / "out")
    # fail-fast order: no object listing (list-type=2 without delimiter) and no bytes
    assert not any("list-type=2" in url and "delimiter" not in url for url in requested)
    assert downloaded == []


def test_fetch_pmc_article_empty_object_listing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A version with no objects -> ``FileNotFoundError`` (no files)."""
    _patch_http(monkeypatch, object_listing=_object_listing([]))
    with pytest.raises(FileNotFoundError, match="No files found"):
        fetch_pmc_article("PMC11708054", tmp_path / "out")


def test_fetch_pmc_article_no_tables_fails_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """OA article with main text + a figure but NO table -> ``FileNotFoundError`` before any download."""
    keys: list[str] = [
        "PMC11708054.1/PMC11708054.1.xml",
        "PMC11708054.1/PMC11708054.1.txt",
        "PMC11708054.1/PMC11708054.1.json",
        "PMC11708054.1/fig.jpg",
    ]
    _, downloaded = _patch_http(monkeypatch, object_listing=_object_listing(keys))
    with pytest.raises(FileNotFoundError, match="No supplementary tables"):
        fetch_pmc_article("PMC11708054", tmp_path / "out")
    assert downloaded == []  # the table gate precedes every byte download


def test_fetch_pmc_article_empty_version_listing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No version prefixes -> ``FileNotFoundError`` (not OA or wrong id)."""
    empty: str = '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Prefix>PMC999.</Prefix></ListBucketResult>'
    _patch_http(monkeypatch, version_listing=empty)
    with pytest.raises(FileNotFoundError, match="No PMC open-access versions"):
        fetch_pmc_article("PMC999", tmp_path / "out")


def test_fetch_pmc_article_bad_id(tmp_path: Path) -> None:
    """A bad id raises ``ValueError`` before any HTTP is attempted (no mock needed)."""
    with pytest.raises(ValueError, match="Invalid PMC id"):
        fetch_pmc_article("not-an-id", tmp_path)
