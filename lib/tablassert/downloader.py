from playwright.sync_api import sync_playwright
from pathlib import Path
from os import environ
import pyexcel

CHROMIUM: str = environ.get("CHROMIUM_PATH")

def modernize_xls(p: Path) -> Path:
  xlsx: Path = p.with_suffix(".xlsx")
  pyexcel.save_book_as(file_name=str(p), dest_file_name=str(xlsx))
  return xlsx

def from_url(website: str, p: Path, timeout: int = 10_000) -> Path:
  try:
    p.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
      browser = pw.chromium.launch(
        headless=True,
        executable_path=CHROMIUM,
        args=["--no-sandbox"]
      )

      page: object = browser.new_page()
      page.goto(website, wait_until="networkidle")
      with page.expect_download(timeout=timeout) as info:
        download = info.value
        download.save_as(p)

    return p

  finally:
    browser.close()
