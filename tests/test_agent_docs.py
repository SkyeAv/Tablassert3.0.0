"""Tests for US-012 docs: docs/agent.md exists, is wired into the mkdocs nav, and is safe.

Pure base-environment tests (no ``[agent]`` extra). They guard the documentation contract: the agent
page exists and is navigable, documents the NEW PMC bucket (marking the dead one as deprecated),
references the real CLI surface, and contains NO hardcoded secrets.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT: Path = Path(__file__).parent.parent
DOC: Path = ROOT / "docs" / "agent.md"
MKDOCS: Path = ROOT / "mkdocs.yml"


def test_agent_doc_exists_and_in_nav() -> None:
    """docs/agent.md exists and the mkdocs nav references it (the page is reachable)."""
    assert DOC.is_file(), "docs/agent.md is missing"
    nav: str = MKDOCS.read_text()
    assert "agent.md" in nav, "mkdocs.yml nav does not reference agent.md"


def test_agent_doc_documents_new_bucket_and_marks_old_dead() -> None:
    """The new pmc-oa-opendata bucket is documented; the dead pmc-open-access is marked deprecated."""
    text: str = DOC.read_text()
    assert "pmc-oa-opendata" in text, "docs must document the new pmc-oa-opendata bucket"
    assert "pmc-open-access" in text, "docs should mention the old bucket to warn it is dead"
    assert any(word in text.lower() for word in ("deprecated", "dead", "removed")), "old bucket must be marked dead/deprecated"


def test_agent_doc_references_cli_surface() -> None:
    """The documented command + flags match the real CLI (tablassert agent + key flags + env vars)."""
    text: str = DOC.read_text()
    assert "tablassert agent" in text
    for flag in ("--configuration-file", "--map-threshold", "--max-improve-iters", "--state-dir", "--min-rows"):
        assert flag in text, f"docs missing CLI flag {flag}"
    for env in ("TABLASSERT_AGENT_MODEL_ID", "TABLASSERT_AGENT_API_BASE", "TABLASSERT_AGENT_API_KEY"):
        assert env in text, f"docs missing env var {env}"


def test_agent_doc_has_no_hardcoded_secret() -> None:
    """No real-looking API key is committed (placeholders like sk-*** / YOUR are fine)."""
    text: str = DOC.read_text()
    # A real key would be sk- followed by a long run of alphanumerics; placeholders use *** or YOUR.
    real_key: re.Match[str] | None = re.search(r"sk-[A-Za-z0-9]{20,}", text)
    assert real_key is None, f"docs appear to contain a hardcoded secret: {real_key.group(0) if real_key else ''}"
    assert "never" in text.lower(), "docs should state secrets are never hardcoded"
    assert "hardcode" in text.lower(), "docs should state secrets are never hardcoded"


def test_agent_doc_covers_pipeline_and_eval() -> None:
    """The page documents the pipeline, the prompt-injection defense, and the eval/optimization loop."""
    text: str = DOC.read_text()
    for topic in ("ReAct", "GEPA", "Pareto", "Reflexion", "checkpoint", "<<<PMC_DATA_BEGIN>>>", "final_answer_checks"):
        assert topic in text, f"docs missing topic {topic}"
