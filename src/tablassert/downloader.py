from pathlib import Path
from time import sleep
from typing import Optional

import pyexcel
from playwright.sync_api import sync_playwright


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
        page = browser.new_page()
        page.goto(website, wait_until="networkidle", timeout=timeout)
        with page.expect_download(timeout=timeout) as info:
          download = info.value
          download.save_as(p)
        browser.close()
      return p
    except Exception as e:
      last = e
      if attempt < retries - 1:
        sleep(2**attempt)

  raise RuntimeError(f"01 | Download Failed After {retries} Attempts: {last}")
