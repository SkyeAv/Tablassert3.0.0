"""Optional ``[agent]`` extra: an autonomous KGX knowledge-graph builder.

This module hosts a smolagents ``CodeAgent`` pipeline that autonomously builds
and audits KGX knowledge graphs from PubMed Central articles. It is part of the
OPTIONAL ``[agent]`` extra, so ``smolagents`` and ``dspy`` are imported LAZILY
(via :class:`tablassert._lazy.LazyModule`) and the base package never requires
them at import time. Install the extra with ``pip install tablassert[agent]``.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.request import Request, urlopen

from tablassert._lazy import LazyModule
from tablassert.log import cat

if TYPE_CHECKING:
    import dspy  # pyright: ignore[reportMissingImports,reportUnusedImport]
    import polars as pl  # pyright: ignore[reportUnusedImport]
    import smolagents  # pyright: ignore[reportMissingImports,reportUnusedImport]
else:
    dspy = LazyModule("dspy")
    pl = LazyModule("polars")
    smolagents = LazyModule("smolagents")

AGENT_EXTRA: str = "pip install tablassert[agent]"

logger = cat("AGENT")


def _require(name: str) -> None:
    """Import an optional dependency or raise a loud, actionable ImportError."""
    try:
        import_module(name)
    except ImportError as exc:
        raise ImportError(f"tablassert agent features require the '{name}' package. Install with {AGENT_EXTRA}.") from exc


def is_lazy() -> bool:
    """Confirm the module loaded without eagerly importing any optional extra.

    A tiny public sentinel tests use to assert ``tablassert.agent`` imported
    cleanly in the base environment; always ``True`` because reaching this call
    already proves no top-level optional import was forced.
    """
    return True


# --------------------------------------------------------------------------- #
# US-002: PMC open-access download (s3://pmc-oa-opendata) + table identification
#
# The OLD ``s3://pmc-open-access`` bucket + FTP ``oa_file_list.csv`` + per-article
# ``.tar.gz`` are DEAD. The live bucket is ``pmc-oa-opendata`` (us-east-1,
# world-readable, FREE, unsigned / no credentials, NOT requester-pays): one prefix
# per article-version ``PMC<n>.<version>/`` holding ``.xml`` (JATS), ``.pdf``,
# ``.txt``, ``.json`` (metadata) and loose media/supplementary files (no tar.gz).
#
# Design: PURE parsers (no I/O, unit-tested directly) are split from two thin I/O
# wrappers (``_http_get_text`` / ``_http_get_bytes``) that are the SINGLE seam tests
# monkeypatch. Only stdlib is used here, so none of this needs the ``[agent]`` extra.
# --------------------------------------------------------------------------- #

PMC_BUCKET: str = "pmc-oa-opendata"
PMC_HTTPS_BASE: str = "https://pmc-oa-opendata.s3.amazonaws.com"
PMC_S3API_BASE: str = "https://pmc-oa-opendata.s3.us-east-1.amazonaws.com"
TABLE_EXTENSIONS: frozenset[str] = frozenset({".xlsx", ".xls", ".csv", ".tsv"})
DROP_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".pdf", ".gif", ".docx"})

_XLINK_HREF: str = "{http://www.w3.org/1999/xlink}href"


def _localname(tag: str) -> str:
    """Strip an ``{namespace}`` prefix from an ElementTree tag (namespace-tolerant)."""
    return tag.rsplit("}", 1)[-1]


def normalize_pmc_id(pmc_id: str) -> str:
    """Normalize ``PMC11708054`` / ``11708054`` / ``pmc11708054`` / padded -> ``PMC11708054``.

    Raises ``ValueError`` when no numeric id remains after stripping an optional
    leading ``PMC`` so callers get an actionable error instead of a bogus S3 prefix.
    """
    cleaned: str = pmc_id.strip()
    digits: str = cleaned[3:] if cleaned[:3].upper() == "PMC" else cleaned
    if not (digits.isascii() and digits.isdigit()):
        raise ValueError(f"Invalid PMC id: {pmc_id!r}; expected a numeric PMC id like 'PMC11708054'.")
    return f"PMC{digits}"


def version_prefixes_from_listing(listing: str) -> list[str]:
    """Parse an S3 list-objects-v2 response into sorted unique version prefixes.

    Tolerates BOTH the XML default (``<CommonPrefixes><Prefix>PMC<n>.<v>/</Prefix>``)
    and a JSON variant (``{"CommonPrefixes":[{"Prefix":...}]}``). Only ``Prefix``
    texts ending in ``/`` are kept, which drops the echoed request ``<Prefix>`` that
    lacks a trailing slash. Returns ``[]`` on empty input or any parse error.
    """
    stripped: str = listing.strip()
    if not stripped:
        return []
    if stripped.startswith("{"):
        try:
            data: dict[str, object] = json.loads(stripped)
        except json.JSONDecodeError:
            return []
        prefixes: list[str] = []
        common: object = data.get("CommonPrefixes")
        if isinstance(common, list):
            for entry in common:
                if isinstance(entry, dict):
                    value: object = entry.get("Prefix")
                    if isinstance(value, str) and value.endswith("/"):
                        prefixes.append(value)
        return sorted(set(prefixes))
    try:
        root: ET.Element = ET.fromstring(stripped)
    except ET.ParseError:
        return []
    found: list[str] = []
    for el in root.iter():
        text: str | None = el.text
        if _localname(el.tag) == "Prefix" and text is not None and text.endswith("/"):
            found.append(text)
    return sorted(set(found))


def is_table_file(filename: str, label: str | None = None) -> bool:
    """Decide whether a supplementary file is a data table.

    Extension DROP wins over a ``Table`` label (a ``fig.jpg`` labeled "Table 1" is
    still dropped). Otherwise a table extension wins, then a ``Table`` label.
    Case-insensitive on both extension and label.
    """
    ext: str = Path(filename).suffix.lower()
    if ext in DROP_EXTENSIONS:
        return False
    if ext in TABLE_EXTENSIONS:
        return True
    return bool(label and "table" in label.lower())


def _nearest_label(scope: ET.Element) -> str | None:
    """Return the first descendant ``<label>`` text within ``scope`` (or ``None``)."""
    for el in scope.iter():
        if _localname(el.tag) == "label" and el.text:
            return el.text.strip()
    return None


def table_files_from_jats(xml_text: str) -> list[dict[str, object]]:
    """Extract supplementary table files from JATS XML (namespace-tolerant).

    Walks every ``<supplementary-material>``, reads each descendant ``<media>`` href
    (``xlink:href`` then plain ``href``) plus the nearest ``<label>``, and returns
    ``{"href", "label", "is_table"}`` entries filtered to ``is_table`` True. Returns
    ``[]`` on malformed XML or when there is no supplementary material.
    """
    try:
        root: ET.Element = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    tables: list[dict[str, object]] = []
    for el in root.iter():
        if _localname(el.tag) != "supplementary-material":
            continue
        label: str | None = _nearest_label(el)
        for media in el.iter():
            if _localname(media.tag) != "media":
                continue
            href: str | None = media.get(_XLINK_HREF) or media.get("href")
            if not href:
                continue
            is_table: bool = is_table_file(href, label)
            if is_table:
                tables.append({"href": href, "label": label, "is_table": True})
    return tables


def public_url(prefix: str, filename: str) -> str:
    """Build the public HTTPS URL for a file under a version prefix."""
    return f"{PMC_HTTPS_BASE}/{prefix.strip('/')}/{filename.lstrip('/')}"


def is_open_access(metadata: str | dict[str, object]) -> bool:
    """Return True iff article metadata shows open access (OA flag or a CC license)."""
    data: dict[str, object]
    if isinstance(metadata, str):
        try:
            data = json.loads(metadata)
        except json.JSONDecodeError:
            return False
    else:
        data = metadata
    if not isinstance(data, dict):
        return False
    if data.get("is_pmc_openaccess"):
        return True
    return str(data.get("license_code", "")).upper().startswith("CC")


def _http_get_text(url: str, *, timeout: int = 120) -> str:
    """GET a URL and return decoded text (the single I/O seam tests monkeypatch)."""
    with urlopen(Request(url, headers={"User-Agent": "tablassert"}), timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _http_get_bytes(url: str, *, timeout: int = 120) -> bytes:
    """GET a URL and return raw bytes (the single I/O seam tests monkeypatch)."""
    with urlopen(Request(url, headers={"User-Agent": "tablassert"}), timeout=timeout) as resp:
        return resp.read()


def fetch_pmc_tables(pmc_id: str, outdir: Path, *, timeout: int = 120) -> list[Path]:
    """Download every supplementary table for a PMC article from ``s3://pmc-oa-opendata``.

    Lists the article's version prefixes, confirms open access via the ``.json``
    metadata, identifies table files from the JATS ``.xml``, and downloads them to
    ``outdir/<prefix>/<filename>``. Raises ``ValueError`` (bad id),
    ``FileNotFoundError`` (no OA versions / no tables) or ``PermissionError``
    (metadata readable but not CC-licensed). S3 only; never scrapes the PMC website.
    """
    pmc: str = normalize_pmc_id(pmc_id)
    outdir.mkdir(parents=True, exist_ok=True)

    listing: str = _http_get_text(f"{PMC_S3API_BASE}?list-type=2&prefix={pmc}.&delimiter=/", timeout=timeout)
    prefixes: list[str] = version_prefixes_from_listing(listing)
    if not prefixes:
        raise FileNotFoundError(f"No PMC open-access versions found for {pmc}; it may not be open access or the id is wrong.")

    any_oa: bool = False
    metadata_read: bool = False
    candidates: list[tuple[str, str]] = []
    for prefix in prefixes:
        stem: str = prefix.strip("/")
        try:
            meta: str = _http_get_text(public_url(prefix, f"{stem}.json"), timeout=timeout)
            metadata_read = True
            if is_open_access(meta):
                any_oa = True
        except Exception:  # network-shaped metadata gaps must not hard-fail; license treated as unknown
            logger.warning("Could not read metadata for {prefix}; treating license as unknown", prefix=prefix)
        try:
            xml_text: str = _http_get_text(public_url(prefix, f"{stem}.xml"), timeout=timeout)
        except Exception:  # an unreadable version is skipped, not fatal while other versions may succeed
            logger.warning("Could not read JATS XML for {prefix}; skipping", prefix=prefix)
            continue
        for table in table_files_from_jats(xml_text):
            candidates.append((prefix, str(table["href"])))

    if metadata_read and not any_oa:
        raise PermissionError(f"{pmc} is not open access (no CC license in metadata).")
    if not metadata_read:
        logger.warning("Metadata unreadable for all versions of {pmc}; proceeding without license confirmation", pmc=pmc)
    if not candidates:
        raise FileNotFoundError(f"No supplementary tables found for {pmc}.")

    downloaded: list[Path] = []
    for prefix, href in candidates:
        dest: Path = outdir / prefix.strip("/") / Path(href).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(_http_get_bytes(public_url(prefix, href), timeout=timeout))
        downloaded.append(dest)

    logger.info("Fetched {n} tables for {pmc} from s3://{bucket} (CC-BY; cite the article DOI)", n=len(downloaded), pmc=pmc, bucket=PMC_BUCKET)
    return downloaded
