from __future__ import annotations

from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Optional

import lazy_loader as Lazy
from playwright.sync_api import sync_playwright

if TYPE_CHECKING:
    import pyexcel
else:
    pyexcel = Lazy.load("pyexcel")


def modernize_xls(p: Path) -> Path:
    xlsx: Path = p.with_suffix(".xlsx")
    pyexcel.save_book_as(file_name=str(p), dest_file_name=str(xlsx))
    return xlsx


def from_url(website: str, p: Path, timeout: int = 60_000, retries: int = 3) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.is_file():
        return p

    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                context = browser.new_context(accept_downloads=True)

                page = context.new_page()
                with page.expect_download(timeout=timeout) as info:
                    try:
                        page.goto(website, wait_until="load", timeout=timeout)
                    except Exception as e:
                        if "net::ERR_ABORTED" not in str(e):
                            raise

                download = info.value
                download.save_as(p)

                context.close()
                browser.close()

            return p

        except Exception as e:
            last = e
            if attempt < retries - 1:
                sleep(2**attempt)

    raise RuntimeError(f"01 | Download Failed After {retries} Attempts: {last}")
