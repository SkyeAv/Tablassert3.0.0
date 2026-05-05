from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Optional
from urllib.parse import urlparse

import lazy_loader as Lazy

from tablassert.log import logger

if TYPE_CHECKING:
    import httpx
    import pyexcel
else:
    httpx = Lazy.load("httpx")
    pyexcel = Lazy.load("pyexcel")


DIRECT_EXTENSIONS: tuple[str, ...] = (".xlsx", ".xls", ".csv", ".tsv", ".txt")
HTML_CONTENT_TYPES: tuple[str, ...] = ("text/html", "application/xhtml+xml")
CHALLENGE_MARKERS: tuple[str, ...] = (
    "captcha",
    "cloudflare",
    "access denied",
    "enable javascript",
    "forbidden",
    "bot check",
)
OLE_MAGIC: bytes = bytes.fromhex("D0CF11E0A1B11AE1")


@dataclass
class DownloadReceipt:
    final_url: str
    content_type: Optional[str] = None
    content_disposition: Optional[str] = None


def modernize_xls(p: Path) -> Path:
    xlsx: Path = p.with_suffix(".xlsx")
    pyexcel.save_book_as(file_name=str(p), dest_file_name=str(xlsx))
    return xlsx


class DownloadError(RuntimeError):
    pass


class DownloadValidationError(DownloadError):
    pass


def classify(url: str) -> str:
    parsed: urlparse = urlparse(url)  # pyright: ignore
    host: str = parsed.hostname or ""  # pyright: ignore
    path: str = parsed.path.lower()  # pyright: ignore

    if host.endswith("pmc.ncbi.nlm.nih.gov"):
        return "pmc"
    if host.endswith("springer.com"):
        return "springer"
    if host.endswith("figstatic.com") or host.endswith("figshare.com"):
        return "figshare"

    if any(path.endswith(ext) for ext in DIRECT_EXTENSIONS):
        return "direct"

    return "browser"


def resolve(url: str) -> Optional[str]:
    host: str = urlparse(url).hostname or ""
    path: str = urlparse(url).path

    if host.endswith("figstatic.com") and "/files/" in path:
        return url

    return None


def temp_path(p: Path) -> Path:
    return p.with_name(f".{p.name}.download")


def cleanup(p: Path) -> None:
    if p.exists():
        p.unlink()


def content_type_is_html(content_type: Optional[str]) -> bool:
    if content_type is None:
        return False

    value: str = content_type.split(";", 1)[0].strip().lower()
    return value in HTML_CONTENT_TYPES


def sample_text(sample: bytes) -> str:
    return sample.decode("utf-8", errors="ignore").strip().lower()


def looks_like_html(sample: bytes) -> bool:
    text: str = sample_text(sample)
    return text.startswith("<!doctype html") or text.startswith("<html") or "<body" in text or "<head" in text


def looks_like_challenge(sample: bytes) -> bool:
    text: str = sample_text(sample)
    return any(marker in text for marker in CHALLENGE_MARKERS)


def reject_download(p: Path, reason: str) -> None:
    cleanup(p)
    raise DownloadValidationError(reason)


def validate_download(p: Path, receipt: Optional[DownloadReceipt] = None) -> None:
    if not p.is_file():
        raise DownloadValidationError(f"missing artifact at {p}")

    size: int = p.stat().st_size
    if size == 0:
        reject_download(p, f"empty artifact at {p}")

    with p.open("rb") as fh:
        sample: bytes = fh.read(4096)

    suffix: str = p.suffix.lower()
    if receipt and content_type_is_html(receipt.content_type):
        reject_download(p, f"html content-type for {suffix or 'unknown'} artifact")

    if looks_like_html(sample):
        reject_download(p, f"html payload detected for {suffix or 'unknown'} artifact")

    if looks_like_challenge(sample):
        reject_download(p, f"challenge payload detected for {suffix or 'unknown'} artifact")

    if suffix == ".xlsx" and not sample.startswith(b"PK\x03\x04"):
        reject_download(p, "xlsx signature mismatch")

    if suffix == ".xls" and not sample.startswith(OLE_MAGIC):
        reject_download(p, "xls signature mismatch")


def direct(url: str, p: Path, timeout: int) -> DownloadReceipt:
    with httpx.Client(timeout=timeout / 1000, follow_redirects=True) as client:
        response: httpx.Response = client.get(url)
        response.raise_for_status()
        p.write_bytes(response.content)
        return DownloadReceipt(
            final_url=str(response.url),
            content_type=response.headers.get("content-type"),
            content_disposition=response.headers.get("content-disposition"),
        )


def browser(website: str, p: Path, timeout: int) -> None:
    from playwright.async_api import async_playwright

    async def run() -> None:
        async with async_playwright() as pw:
            br = await pw.chromium.launch(headless=True)
            ctx = await br.new_context(accept_downloads=True)
            try:
                page = await ctx.new_page()
                async with page.expect_download(timeout=timeout) as info:
                    try:
                        await page.goto(website, wait_until="load", timeout=timeout)
                    except Exception as e:
                        msg: str = str(e)
                        if "net::ERR_ABORTED" not in msg and "Download is starting" not in msg:
                            raise
                download = await info.value
                await download.save_as(p)
            finally:
                await ctx.close()
                await br.close()

    asyncio.run(run())


def from_url(
    website: str,
    p: Path,
    config_name: Optional[str] = None,
    section_hash: Optional[str] = None,
    timeout: int = 60_000,
    retries: int = 3,
) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.is_file():
        return p

    strategy: str = classify(website)
    logger.info(f"download | strategy={strategy} | url={website} | config={config_name} | hash={section_hash}")

    resolved: Optional[str] = resolve(website)
    url: str = resolved if resolved is not None else website

    last: Optional[Exception] = None
    for attempt in range(retries):
        candidate: Path = temp_path(p)
        cleanup(candidate)
        try:
            if strategy == "direct":
                receipt: DownloadReceipt = direct(url, candidate, timeout)
                validate_download(candidate, receipt)
                candidate.replace(p)
                if p.is_file():
                    return p

            elif strategy in ("pmc", "springer", "figshare"):
                try:
                    receipt = direct(url, candidate, timeout)
                    validate_download(candidate, receipt)
                    candidate.replace(p)
                    if p.is_file():
                        return p
                except Exception as e:
                    cleanup(candidate)
                    logger.info(f"download | host adapter fell back to browser | url={url} | reason={e}")

            browser(website, candidate, timeout)
            validate_download(candidate)
            candidate.replace(p)
            if p.is_file():
                return p

        except Exception as e:
            last = e
            cleanup(candidate)
            logger.warning(f"download | attempt={attempt + 1}/{retries} | error={e}")
            if attempt < retries - 1:
                sleep(2**attempt)

    raise DownloadError(
        f"01 | FAILED DOWNLOAD | URL: {website} | LOCAL: {p} | CONFIG: {config_name} | HASH: {section_hash} | LAST: {last!r}"
    ) from last
