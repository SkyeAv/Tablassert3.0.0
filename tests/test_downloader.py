from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tablassert.downloader import DownloadError, DownloadReceipt, classify, direct, from_url, resolve, temp_path


def fake_xlsx_bytes() -> bytes:
    return b"PK\x03\x04fake-xlsx"


# ? classify Returns direct For Known File Extensions
@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/data.xlsx",
        "https://example.com/data.XLSX",
        "https://example.com/data.xls",
        "https://example.com/data.csv",
        "https://example.com/data.tsv",
        "https://example.com/data.txt",
    ],
)
def test_classify_direct_extensions(url: str) -> None:
    assert classify(url) == "direct"


# ? classify Returns Host Adapter Keys For Known Domains
def test_classify_pmc() -> None:
    url: str = "https://pmc.ncbi.nlm.nih.gov/articles/instance/9499989/bin/mmc2.xlsx"
    assert classify(url) == "pmc"


def test_classify_springer() -> None:
    url: str = "https://static-content.springer.com/esm/art%3A10.1038%2Fs41591-024-03038-y/MediaObjects/41591_2024_3038_MOESM2_ESM.xlsx"
    assert classify(url) == "springer"


def test_classify_figshare() -> None:
    assert classify("https://ndownloader.figstatic.com/files/47044431") == "figshare"


# ? classify Returns browser For Unknown Domains Without File Extensions
def test_classify_unknown() -> None:
    assert classify("https://unknown.example.com/page") == "browser"


# ? resolve Returns None For Direct URLs
def test_resolve_direct_url() -> None:
    assert resolve("https://example.com/data.xlsx") is None


# ? resolve Passes Through Figshare File URLs
def test_resolve_figshare_file() -> None:
    url: str = "https://ndownloader.figstatic.com/files/47044431"
    assert resolve(url) == url


# ? resolve Returns None For Non Figshare URLs
def test_resolve_non_figshare() -> None:
    assert resolve("https://pmc.ncbi.nlm.nih.gov/articles/instance/9499989/bin/mmc2.xlsx") is None


# ? direct Writes Response Content To File
def test_direct_writes_file(tmp_path: Path) -> None:
    out: Path = tmp_path / "test.csv"
    mock_response: MagicMock = MagicMock()
    mock_response.content = b"a,b\n1,2\n"
    mock_response.url = "https://example.com/data.csv"
    mock_response.headers = {"content-type": "text/csv"}
    mock_response.raise_for_status = MagicMock()

    with patch("tablassert.downloader.httpx") as mock_httpx:
        mock_client: MagicMock = MagicMock()
        mock_client.get.return_value = mock_response
        mock_httpx.Client.return_value.__enter__.return_value = mock_client

        result: DownloadReceipt = direct("https://example.com/data.csv", out, 30_000)

    assert result.final_url == "https://example.com/data.csv"
    assert out.read_bytes() == b"a,b\n1,2\n"


# ? direct Raises On HTTP Error
def test_direct_raises_on_error(tmp_path: Path) -> None:
    out: Path = tmp_path / "test.csv"
    with patch("tablassert.downloader.httpx") as mock_httpx:
        mock_client: MagicMock = MagicMock()
        mock_client.get.side_effect = Exception("connection refused")
        mock_httpx.Client.return_value.__enter__.return_value = mock_client

        with pytest.raises(Exception, match="connection refused"):
            direct("https://example.com/data.csv", out, 30_000)


# ? from_url Returns Immediately If File Exists
def test_from_url_skips_existing(tmp_path: Path) -> None:
    existing: Path = tmp_path / "data.csv"
    existing.write_text("already here")
    result: Path = from_url("https://example.com/data.csv", existing)
    assert result == existing
    assert existing.read_text() == "already here"


# ? from_url Raises DownloadError After Retries Exhausted
def test_from_url_raises_after_retries(tmp_path: Path) -> None:
    out: Path = tmp_path / "data.csv"
    with patch("tablassert.downloader.browser", side_effect=RuntimeError("fail")):
        with pytest.raises(DownloadError, match="FAILED DOWNLOAD"):
            from_url("https://unknown.example.com/page", out, retries=2)


# ? from_url Logs Strategy On Invocation
def test_from_url_logs_strategy(tmp_path: Path) -> None:
    out: Path = tmp_path / "data.xlsx"
    with patch("tablassert.downloader.direct") as mock_direct:

        def write_file(url: str, p: Path, timeout: int) -> DownloadReceipt:
            p.write_bytes(fake_xlsx_bytes())
            return DownloadReceipt(final_url=url, content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        mock_direct.side_effect = write_file
        with patch("tablassert.downloader.logger") as mock_logger:
            from_url("https://example.com/data.xlsx", out, retries=1)

        mock_logger.info.assert_any_call(
            "DOWNLOAD | STRATEGY: direct | URL: https://example.com/data.xlsx | CONFIG: None | HASH: None"
        )


# ? from_url Threads Config Name And Hash Into Error Message
def test_from_url_error_includes_context(tmp_path: Path) -> None:
    out: Path = tmp_path / "data.csv"
    with patch("tablassert.downloader.browser", side_effect=RuntimeError("fail")):
        with pytest.raises(DownloadError, match="CONFIG: myconfig") as exc_info:
            from_url("https://unknown.example.com/page", out, config_name="myconfig", section_hash="abc123", retries=1)

    assert "HASH: abc123" in str(exc_info.value)


# ? from_url Uses Direct Strategy For File Extension URLs
def test_from_url_uses_direct_for_extensions(tmp_path: Path) -> None:
    out: Path = tmp_path / "data.xlsx"

    def write_file(url: str, p: Path, timeout: int) -> DownloadReceipt:
        p.write_bytes(fake_xlsx_bytes())
        return DownloadReceipt(final_url=url, content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with patch("tablassert.downloader.direct", side_effect=write_file) as mock_direct:
        from_url("https://example.com/data.xlsx", out, retries=1)

    mock_direct.assert_called_once()


# ? from_url Falls Back To Browser When Direct Fails For Host Adapter
def test_from_url_host_adapter_fallback(tmp_path: Path) -> None:
    out: Path = tmp_path / "data.xlsx"

    def browser_success(website: str, p: Path, timeout: int) -> None:
        p.write_bytes(fake_xlsx_bytes())

    def direct_html(url: str, p: Path, timeout: int) -> DownloadReceipt:
        p.write_text("<!doctype html><html><body>blocked</body></html>")
        return DownloadReceipt(final_url=url, content_type="text/html")

    with patch("tablassert.downloader.direct", side_effect=direct_html):
        with patch("tablassert.downloader.browser", side_effect=browser_success) as mock_browser:
            result: Path = from_url(
                "https://pmc.ncbi.nlm.nih.gov/articles/instance/12345/bin/data.xlsx", out, retries=1
            )

    mock_browser.assert_called_once()
    assert result.is_file()


# ? from_url Rejects Html Payload Served As Spreadsheet
def test_from_url_rejects_html_payload_and_cleans_up(tmp_path: Path) -> None:
    out: Path = tmp_path / "data.xlsx"

    def direct_html(url: str, p: Path, timeout: int) -> DownloadReceipt:
        p.write_text("<!doctype html><html><body>challenge</body></html>")
        return DownloadReceipt(final_url=url, content_type="text/html")

    with patch("tablassert.downloader.direct", side_effect=direct_html):
        with pytest.raises(DownloadError, match="html content-type"):
            from_url("https://example.com/data.xlsx", out, retries=1)

    assert not out.exists()
    assert not temp_path(out).exists()


# ? from_url Rejects Anti Bot Text Payloads
def test_from_url_rejects_challenge_payload_and_cleans_up(tmp_path: Path) -> None:
    out: Path = tmp_path / "data.csv"

    def direct_challenge(url: str, p: Path, timeout: int) -> DownloadReceipt:
        p.write_text("Access denied. Please enable JavaScript to continue.")
        return DownloadReceipt(final_url=url, content_type="text/plain")

    with patch("tablassert.downloader.direct", side_effect=direct_challenge):
        with pytest.raises(DownloadError, match="challenge payload"):
            from_url("https://example.com/data.csv", out, retries=1)

    assert not out.exists()
    assert not temp_path(out).exists()


# ? from_url Cleans Browser Candidate On Invalid Download
def test_from_url_cleans_browser_candidate_on_invalid_payload(tmp_path: Path) -> None:
    out: Path = tmp_path / "data.csv"

    def browser_html(website: str, p: Path, timeout: int) -> None:
        p.write_text("<!doctype html><html><body>access denied</body></html>")

    with patch("tablassert.downloader.browser", side_effect=browser_html):
        with pytest.raises(DownloadError, match="html payload"):
            from_url("https://unknown.example.com/page", out, retries=1)

    assert not out.exists()
    assert not temp_path(out).exists()


# ? Network Test Downloads Small File
@pytest.mark.network
def test_from_url_downloads_small_file(tmp_path: Path) -> None:
    url: str = "https://github.com/octocat/Hello-World/archive/refs/heads/master.zip"
    out: Path = tmp_path / "hello.zip"
    result: Path = from_url(url, out, timeout=30_000, retries=2)
    assert result == out
    assert out.is_file()
    assert out.stat().st_size > 0
