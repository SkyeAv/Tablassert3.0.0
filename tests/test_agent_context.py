"""Offline unit tests for the PMC JATS context tooling (US-002/US-008).

Covers the PURE JATS parsers (``supplementary_materials_from_jats``, ``parse_jats_summary``,
``_clean_abstract``), the data-fenced ``pmc_article_context`` renderer, and the lazily-built
``pmc_article_context`` smolagents tool. No network; the tool test skips without the ``[agent]``
extra (``importorskip("smolagents")``).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from tablassert.agent import (
    DATA_FENCE_BEGIN,
    DATA_FENCE_END,
    DATA_GUARDRAIL,
    _clean_abstract,
    make_pmc_article_context_tool,
    parse_jats_summary,
    pmc_article_context,
    supplementary_materials_from_jats,
)

# A compact but realistic JATS article: title, journal, an abstract with a glued
# "ABSTRACT" heading, two body sections, and two supplementary materials — a .docx
# labeled "Supplemental material" (NOT a table) and a .xlsx labeled "Table S1" (a table).
JATS_XML: str = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<article xmlns:xlink="http://www.w3.org/1999/xlink" article-type="research-article" dtd-version="1.4">'
    "<front>"
    "<journal-meta><journal-title>mBio</journal-title></journal-meta>"
    "<article-meta>"
    "<title-group><article-title>Gut microbiota and tamoxifen</article-title></title-group>"
    "<abstract><title>ABSTRACT</title><p>Tamoxifen is a drug.</p></abstract>"
    "</article-meta>"
    "</front>"
    "<body>"
    "<sec><title>INTRODUCTION</title><p>intro</p></sec>"
    "<sec><title>RESULTS</title><p>results</p></sec>"
    "</body>"
    "<back>"
    '<supplementary-material id="sm1">'
    "<label>Supplemental material</label>"
    "<caption><title>Supplemental figures.</title></caption>"
    '<media xlink:href="mbio-s0001.docx"/>'
    "</supplementary-material>"
    '<supplementary-material id="sm2">'
    "<label>Table S1</label>"
    "<caption><title>Taxonomic classification.</title></caption>"
    '<media xlink:href="mbio-s0002.xlsx"/>'
    "</supplementary-material>"
    "</back>"
    "</article>"
)
JATS_NO_SM: str = '<?xml version="1.0"?><article><body><p>No supplementary material here.</p></body></article>'


# --------------------------------------------------------------------------- #
# supplementary_materials_from_jats
# --------------------------------------------------------------------------- #


def test_supplementary_materials_realistic() -> None:
    """Both materials are returned with label/caption; the .docx is NOT a table, the .xlsx IS."""
    materials: list[dict[str, object]] = supplementary_materials_from_jats(JATS_XML)
    assert materials == [
        {"href": "mbio-s0001.docx", "label": "Supplemental material", "caption": "Supplemental figures.", "is_table": False},
        {"href": "mbio-s0002.xlsx", "label": "Table S1", "caption": "Taxonomic classification.", "is_table": True},
    ]


def test_supplementary_materials_label_only_table() -> None:
    """An unknown extension carrying a 'Table' label is still flagged is_table (label-aware)."""
    xml: str = (
        '<article xmlns:xlink="http://www.w3.org/1999/xlink"><supplementary-material>'
        "<label>Table S9</label>"
        '<media xlink:href="data.bin"/>'
        "</supplementary-material></article>"
    )
    materials: list[dict[str, object]] = supplementary_materials_from_jats(xml)
    assert materials[0]["is_table"] is True


def test_supplementary_materials_none() -> None:
    """No <supplementary-material> -> []."""
    assert supplementary_materials_from_jats(JATS_NO_SM) == []


def test_supplementary_materials_malformed() -> None:
    """Malformed XML -> [] rather than raising."""
    assert supplementary_materials_from_jats("<not xml") == []


# --------------------------------------------------------------------------- #
# parse_jats_summary / _clean_abstract
# --------------------------------------------------------------------------- #


def test_parse_jats_summary_fields() -> None:
    """Title, journal, abstract (heading stripped) and the body section outline are extracted."""
    summary: dict[str, object] = parse_jats_summary(JATS_XML)
    assert summary["title"] == "Gut microbiota and tamoxifen"
    assert summary["journal"] == "mBio"
    assert summary["abstract"] == "Tamoxifen is a drug."  # leading "ABSTRACT" stripped
    assert summary["sections"] == ["INTRODUCTION", "RESULTS"]


def test_parse_jats_summary_malformed() -> None:
    """Malformed XML -> all-empty summary (never a raise)."""
    assert parse_jats_summary("<not xml") == {"title": "", "journal": "", "abstract": "", "sections": []}


def test_clean_abstract_strips_heading() -> None:
    """A glued 'ABSTRACT' heading is removed; other text is preserved and collapsed."""
    el: ET.Element = ET.fromstring("<abstract><title>ABSTRACT</title><p>Foo   bar</p></abstract>")
    assert _clean_abstract(el) == "Foo bar"


def test_clean_abstract_no_heading() -> None:
    """Without a known heading the collapsed text is returned unchanged."""
    el: ET.Element = ET.fromstring("<abstract><p>Plain abstract text.</p></abstract>")
    assert _clean_abstract(el) == "Plain abstract text."


# --------------------------------------------------------------------------- #
# pmc_article_context (data-fenced renderer)
# --------------------------------------------------------------------------- #


def test_pmc_article_context_xml(tmp_path: Path) -> None:
    """An .xml renders a fenced, structured summary with the title + table manifest."""
    xml_path: Path = tmp_path / "PMC1.1.xml"
    xml_path.write_text(JATS_XML, encoding="utf-8")
    out: str = pmc_article_context(xml_path)
    assert DATA_GUARDRAIL in out
    assert DATA_FENCE_BEGIN in out
    assert DATA_FENCE_END in out
    assert "title: Gut microbiota and tamoxifen" in out
    assert "journal: mBio" in out
    assert "  - INTRODUCTION" in out
    assert "Table S1" in out
    assert "mbio-s0002.xlsx" in out
    assert "is_table: True" in out
    assert "is_table: False" in out


def test_pmc_article_context_txt_excerpt(tmp_path: Path) -> None:
    """A .txt renders a fenced excerpt; long text is truncated with a note."""
    txt_path: Path = tmp_path / "PMC1.1.txt"
    txt_path.write_text("x" * 100, encoding="utf-8")
    out: str = pmc_article_context(txt_path, max_chars=10)
    assert DATA_FENCE_BEGIN in out
    assert "... (truncated)" in out
    assert "x" * 10 in out
    assert "x" * 11 not in out  # only max_chars of body retained


def _minimal_pdf(text: str) -> bytes:
    """Build a tiny valid single-page PDF whose content stream renders ``text`` (extractable by pdfminer)."""
    content: bytes = f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET".encode()
    objs: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out: bytes = b"%PDF-1.4\n"
    offsets: list[int] = []
    for i, obj in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_pos: int = len(out)
    out += b"xref\n0 " + str(len(objs) + 1).encode() + b"\n"
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += b"trailer\n<< /Size " + str(len(objs) + 1).encode() + b" /Root 1 0 R >>\nstartxref\n" + str(xref_pos).encode() + b"\n%%EOF"
    return out


def test_pmc_article_context_pdf_renders_excerpt(tmp_path: Path) -> None:
    """W4: a .pdf main text is extracted (pdfminer.six) into a data-fenced excerpt (skip if engine absent)."""
    pytest.importorskip("pdfminer")
    pdf_path: Path = tmp_path / "PMC1.1.pdf"
    pdf_path.write_bytes(_minimal_pdf("tamoxifen gut microbiota"))
    out: str = pmc_article_context(pdf_path)
    assert DATA_GUARDRAIL in out
    assert DATA_FENCE_BEGIN in out
    assert DATA_FENCE_END in out
    assert "tamoxifen" in out  # the extracted text rides inside the fence


def test_pmc_article_context_pdf_missing_engine_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """W4: when pdfminer.six is unavailable, a .pdf raises a clear ValueError naming the install path."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("pdfminer"):
            raise ImportError("No module named 'pdfminer'")
        return real_import(name, *args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    pdf_path: Path = tmp_path / "PMC1.1.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 garbage")
    with pytest.raises(ValueError, match=r"pdfminer\.six"):
        pmc_article_context(pdf_path)


def test_pmc_article_context_missing(tmp_path: Path) -> None:
    """A missing path -> FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Article file not found"):
        pmc_article_context(tmp_path / "nope.xml")


# --------------------------------------------------------------------------- #
# make_pmc_article_context_tool (needs the [agent] extra)
# --------------------------------------------------------------------------- #


def test_make_pmc_article_context_tool(tmp_path: Path) -> None:
    """The lazy tool exposes the right name/inputs and forwards to pmc_article_context."""
    pytest.importorskip("smolagents")
    tool = make_pmc_article_context_tool()
    assert tool.name == "pmc_article_context"
    assert "source" in tool.inputs
    assert tool.output_type == "string"

    xml_path: Path = tmp_path / "PMC1.1.xml"
    xml_path.write_text(JATS_XML, encoding="utf-8")
    out: str = tool.forward(str(xml_path))
    assert DATA_FENCE_BEGIN in out
    assert "Table S1" in out
