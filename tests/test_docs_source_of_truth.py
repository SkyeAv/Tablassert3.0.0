"""Source-of-truth guards for documentation YAML configs and the documentation index lists.

``tests/test_docs_examples.py`` guards the YAML files shipped under ``docs/examples/``, but the
configs readers copy first live in prose: the README quick start, the tutorial listings, and the
use-case gallery. Those blocks were never validated, so the README quick start drifted -- it
declared ``source.url`` as a bare string against ``models.BaseSource.url: list[HttpUrl]`` -- and
nothing failed. This module closes that gap and the matching navigation gap (a page added to
``mkdocs.yml`` nav without being linked from the README or the docs index).

Every expectation is derived from live source rather than copied out of the docs:

* YAML blocks are lifted from the Markdown itself and classified by shape, so a new complete
  config anywhere in ``README.md`` / ``docs/**/*.md`` is guarded automatically and an abbreviated
  or partial snippet is skipped by rule -- there is no hand-maintained include/skip list.
* Validation runs the exact paths the CLI runs: ``ingests.to_sections`` -> ``Tcode.model_validate``
  for a table (what ``tablassert validate --schema table`` does) and ``models.Graph.model_validate``
  for a graph (what ``--schema graph`` does first).
* The index lists are checked against the live ``mkdocs.yml`` nav and ``site_url``.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pytest
import yaml
from yaml import CSafeLoader

from tablassert.errors import BiolinkRelocationWarning, UnpairedEffectAnnotationWarning
from tablassert.ingests import to_sections
from tablassert.lib import Tcode
from tablassert.models import Graph

# Repository root (parent of tests/).
ROOT: Path = Path(__file__).resolve().parent.parent
DOCS: Path = ROOT / "docs"
README: Path = ROOT / "README.md"
MKDOCS: Path = ROOT / "mkdocs.yml"

# The docs index page whose "Documentation Sections" list must mirror the mkdocs nav. The README
# carries the same list in published-site-URL style; both are the site's Home surface, which is why
# the nav Home entry is not linked from either.
INDEX: Path = DOCS / "index.md"
# `Tcode` requires a `store` path but never touches it during validation, so one stable name keeps
# the parametrization deterministic and xdist-safe (no worker writes or reads it).
STORE: Path = ROOT / ".tablassert" / "store" / "docs-source-of-truth.parquet"

ConfigKind = Literal["graph", "table"]

# Keys whose presence makes a parsed block a COMPLETE config for one live schema. Mirrors the
# required fields of `models.Section` (source/statement/provenance) and `models.Graph`.
TABLE_KEYS: frozenset[str] = frozenset({"provenance", "source", "statement"})
GRAPH_KEYS: frozenset[str] = frozenset({"fullmap", "name", "rig", "tables", "version"})
# Elision markers. A block carrying one is an excerpt by construction and cannot validate.
PLACEHOLDER_TOKENS: tuple[str, ...] = ("{...}", "...")

FENCED_YAML: re.Pattern[str] = re.compile(r"^```(?:yaml|yml)[^\S\n]*\n(.*?)^```[^\S\n]*$", re.MULTILINE | re.DOTALL)
HEADING: re.Pattern[str] = re.compile(r"^(#+)[ \t]+(.*?)[ \t]*$", re.MULTILINE)
MARKDOWN_LINK: re.Pattern[str] = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")


@dataclass(frozen=True)
class YamlSnippet:
    """One fenced YAML block lifted out of a documentation page, with its classified schema kind."""

    page: Path
    line: int
    block: str
    kind: ConfigKind | None

    @property
    def location(self) -> str:
        """Return a ``path:line`` pointer for test ids and failure messages.

        Returns:
            The block's opening-fence location relative to the repository root.
        """
        return f"{self.page.relative_to(ROOT)}:{self.line}"

    def parse(self) -> Any:
        """Parse the block fresh on every call.

        Returns:
            The parsed YAML object.

        Re-parsing instead of caching is deliberate: ``ingests.to_sections`` stamps ``config`` into
        the template dict it is handed, so a cached parse would leak mutation between tests (and
        between the two workers that collect the same parametrization under xdist).
        """
        return yaml.load(self.block, Loader=CSafeLoader)


def _mkdocs() -> dict[str, Any]:
    """Return the parsed live ``mkdocs.yml`` -- the nav and site_url source of truth.

    Returns:
        The mkdocs configuration as a mapping.
    """
    parsed: Any = yaml.load(MKDOCS.read_text(encoding="utf-8"), Loader=CSafeLoader)
    assert isinstance(parsed, dict), f"{MKDOCS.name} did not parse to a mapping"
    return parsed


def _excluded_docs_dirs() -> frozenset[str]:
    """Return ``docs/`` subdirectory names mkdocs excludes from the published site.

    ``mkdocs.yml:exclude_docs`` keeps gitignored local scratch notes out of the site; unpublished
    notes are not documentation a reader can copy from, so they are not schema-guarded here.

    Returns:
        Directory names (without the trailing slash mkdocs writes).
    """
    raw: str = str(_mkdocs().get("exclude_docs") or "")
    return frozenset(line.strip().removesuffix("/") for line in raw.splitlines() if line.strip().endswith("/"))


def _markdown_pages() -> list[Path]:
    """Return the README plus every published Markdown page under ``docs/``, in a stable order.

    Returns:
        Sorted documentation pages, README first.
    """
    excluded: frozenset[str] = _excluded_docs_dirs()
    published: list[Path] = [path for path in sorted(DOCS.rglob("*.md")) if path.relative_to(DOCS).parts[0] not in excluded]
    return [README, *published]


def _classify(block: str, parsed: Any) -> ConfigKind | None:
    """Return which live schema a block is a COMPLETE config for, or ``None`` when it must be skipped.

    Rule-based on purpose. The docs abbreviate with ``{...}`` / ``...`` and show partial fragments
    (a lone ``source:``, a template/section merge illustration); neither can validate, and neither
    may be excluded by a copied list -- so a block is guarded only when its parsed shape carries
    every key the corresponding live model requires.

    Args:
        block: Raw YAML text of the fenced block.
        parsed: The same block parsed with the safe loader.

    Returns:
        ``"table"``, ``"graph"``, or ``None`` to skip.
    """
    if any(token in block for token in PLACEHOLDER_TOKENS) or not isinstance(parsed, dict):
        return None
    template: Any = parsed.get("template")
    top: set[str] = set(parsed)
    declared: set[str] = top | (set(template) if isinstance(template, dict) else set())
    if TABLE_KEYS.issubset(declared):
        return "table"
    if GRAPH_KEYS.issubset(top):
        return "graph"
    return None


def _snippet(page: Path, line: int, block: str) -> YamlSnippet:
    """Build one classified documentation YAML snippet.

    Args:
        page: Markdown page the block came from.
        line: 1-based line of the block's opening fence.
        block: Raw YAML text of the block.

    Returns:
        The classified snippet. A block that does not parse classifies as ``None`` (skipped) so a
        docs typo cannot crash collection for the whole module; ``test_yaml_snippets_parse_as_yaml``
        reports it per block instead.
    """
    try:
        parsed: Any = yaml.load(block, Loader=CSafeLoader)
    except yaml.YAMLError:
        parsed = None
    return YamlSnippet(page=page, line=line, block=block, kind=_classify(block, parsed))


def _yaml_snippets(pages: list[Path] | None = None) -> list[YamlSnippet]:
    """Extract and classify every fenced YAML block in the documentation corpus.

    Args:
        pages: Pages to scan; defaults to the whole published corpus.

    Returns:
        Snippets in page order (README first), each carrying its classified kind.
    """
    snippets: list[YamlSnippet] = []
    for page in _markdown_pages() if pages is None else pages:
        text: str = page.read_text(encoding="utf-8")
        for match in FENCED_YAML.finditer(text):
            snippets.append(_snippet(page, text.count("\n", 0, match.start()) + 1, match.group(1)))
    return snippets


def _snippets_in(page: Path, complete_only: bool = False) -> list[YamlSnippet]:
    """Return one page's snippets, optionally narrowed to the ones the classifier guards.

    Args:
        page: Markdown page to scan.
        complete_only: When True, drop skipped fragments and placeholders.

    Returns:
        Snippets in document order.
    """
    found: list[YamlSnippet] = _yaml_snippets([page])
    return [snippet for snippet in found if snippet.kind is not None] if complete_only else found


def _complete_snippets() -> list[YamlSnippet]:
    """Return every documentation YAML block the classifier calls a complete table or graph config.

    Returns:
        Guarded snippets in corpus order.
    """
    return [snippet for snippet in _yaml_snippets() if snippet.kind is not None]


def _validate(snippet: YamlSnippet) -> None:
    """Validate one complete snippet through the same code path ``tablassert validate`` uses.

    Args:
        snippet: A snippet whose ``kind`` is ``"table"`` or ``"graph"``.

    Raises:
        AssertionError: If a table snippet expands to zero sections.
    """
    parsed: Any = snippet.parse()
    # Only the two known relocation/effect notices are silenced: the tutorial and the gallery
    # deliberately show `study_size` and non-slot annotations, and the live models WARN (never
    # fail) about where such a value lands. Everything else stays audible so new drift surfaces.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BiolinkRelocationWarning)
        warnings.simplefilter("ignore", UnpairedEffectAnnotationWarning)
        if snippet.kind == "graph":
            Graph.model_validate(parsed)
            return
        sections: list[dict[str, Any]] = to_sections(parsed, Path(f"{snippet.location}.yaml"))  # pyright: ignore
        assert sections, f"{snippet.location} expanded to zero sections"
        for section in sections:
            Tcode.model_validate({**section, "store": STORE})


def _section_range(text: str, heading: str) -> tuple[int, int]:
    """Return the character span of the Markdown section opened by ``heading``.

    Args:
        text: Full Markdown source.
        heading: Heading text without its leading hashes.

    Returns:
        ``(start, end)`` offsets covering the section body, ending at the next heading of the same
        or higher level (or end of file).

    Raises:
        AssertionError: If no heading with that exact text exists -- an anchor that silently stops
            matching would make every pin below vacuous.
    """
    headings: list[re.Match[str]] = list(HEADING.finditer(text))
    for index, match in enumerate(headings):
        if match.group(2) != heading:
            continue
        level: int = len(match.group(1))
        end: int = len(text)
        for later in headings[index + 1 :]:
            if len(later.group(1)) <= level:
                end = later.start()
                break
        return match.end(), end
    raise AssertionError(f"heading {heading!r} not found; the section anchor is stale")


def _list_links(page: Path, heading: str) -> dict[str, str]:
    """Return ``{link text: target}`` for every Markdown link in one documentation index list.

    Args:
        page: Markdown page holding the list.
        heading: ``##`` heading the list lives under.

    Returns:
        Link text mapped to its target, in document order.
    """
    text: str = page.read_text(encoding="utf-8")
    start, end = _section_range(text, heading)
    return {match.group(1): match.group(2) for match in MARKDOWN_LINK.finditer(text, start, end)}


def _first_leaf(target: Any) -> str | None:
    """Return the first leaf page under a nav target (a page string or a nested section list).

    Args:
        target: A mkdocs nav value.

    Returns:
        The docs-relative page path, or ``None`` when the target holds no page.
    """
    if isinstance(target, str):
        return target
    if isinstance(target, list):
        for child in target:
            leaf: str | None = _first_leaf(next(iter(child.values()))) if isinstance(child, dict) else _first_leaf(child)
            if leaf is not None:
                return leaf
    return None


def _nav_pages() -> list[tuple[str, str]]:
    """Return ``(nav title, docs-relative page)`` for every top-level mkdocs nav entry.

    A section resolves to its first leaf page, which is what both index lists already link
    (``Configuration`` -> ``configuration/graph.md``). The Home entry is dropped: the README and
    ``docs/index.md`` ARE the Home surface and link the site root rather than themselves.

    Returns:
        Nav pages in ``mkdocs.yml`` order.
    """
    nav: Any = _mkdocs().get("nav")
    assert isinstance(nav, list), "mkdocs.yml nav is missing or not a list"
    assert nav, "mkdocs.yml nav is empty"
    pages: list[tuple[str, str]] = []
    for entry in nav:
        assert isinstance(entry, dict), f"unsupported mkdocs nav entry: {entry!r}"
        assert len(entry) == 1, f"mkdocs nav entry must carry exactly one title: {entry!r}"
        title, target = next(iter(entry.items()))
        leaf: str | None = _first_leaf(target)
        if leaf is not None and leaf != "index.md":
            pages.append((str(title), leaf))
    assert pages, "mkdocs.yml nav lists no page beyond Home"
    return pages


def _site_base() -> str:
    """Return the live published-site base URL, slash-terminated.

    Returns:
        ``mkdocs.yml``'s ``site_url``, so the guard never hardcodes the documentation host.
    """
    base: str = str(_mkdocs()["site_url"])
    return base if base.endswith("/") else f"{base}/"


def _published_url(page: str) -> str:
    """Return the published-site URL mkdocs serves ``page`` at -- the README's existing link style.

    Args:
        page: Docs-relative page path such as ``configuration/graph.md``.

    Returns:
        Absolute URL built from the live ``mkdocs.yml`` ``site_url``.
    """
    return f"{_site_base()}{page.removesuffix('.md')}/"


def _assert_nav_parity(page: Path, heading: str) -> None:
    """Assert one documentation index list links every top-level nav page at the right target.

    Args:
        page: Markdown page holding the index list.
        heading: ``##`` heading the list lives under.

    Raises:
        AssertionError: Naming the missing nav page, or the wrong target for a linked one.
    """
    links: dict[str, str] = _list_links(page, heading)
    for title, nav_page in _nav_pages():
        assert title in links, f"{page.relative_to(ROOT)} '## {heading}' is missing the nav page {title!r} ({nav_page})"
        expected: str = _published_url(nav_page) if page == README else nav_page
        assert links[title] == expected, f"{page.relative_to(ROOT)} links {title!r} to {links[title]!r}, expected {expected!r}"


@pytest.mark.parametrize("snippet", _yaml_snippets(), ids=lambda snippet: snippet.location)
def test_yaml_snippets_parse_as_yaml(snippet: YamlSnippet) -> None:
    """Every fenced YAML block in the documentation corpus is syntactically valid YAML.

    Runs over ALL blocks, including the fragments the schema guard skips, because an unparseable
    block is invisible to every other check here: it classifies as ``None`` and would otherwise
    drift unnoticed (and, if parsed at collection time, take the whole module down with it).

    Args:
        snippet: One documentation YAML block.
    """
    try:
        parsed: Any = snippet.parse()
    except yaml.YAMLError as error:
        pytest.fail(f"{snippet.location} is not valid YAML: {error}")
    assert snippet.block.strip(), f"{snippet.location} is an empty YAML block"
    assert parsed is not None, f"{snippet.location} parsed to nothing"


@pytest.mark.parametrize("snippet", _complete_snippets(), ids=lambda snippet: f"{snippet.location}-{snippet.kind}")
def test_yaml_snippets_validate_against_live_schema(snippet: YamlSnippet) -> None:
    """Every complete documentation config validates exactly as ``tablassert validate`` would.

    This is the guard that catches a quick start or gallery entry drifting away from the live
    Pydantic models: the block a reader copies is checked through the same ``to_sections`` ->
    ``Tcode`` path (table) or ``Graph.model_validate`` path (graph) the CLI uses.

    Args:
        snippet: A classified complete table or graph config lifted from the docs.
    """
    _validate(snippet)


def test_yaml_snippets_classifier_selects_complete_configs() -> None:
    """The classifier picks up the configs readers actually copy -- pinned against under-selection.

    A classifier that quietly skips the README quick start would make the guard above vacuous, so
    the selection is pinned from source: the README quick start and the tutorial each contribute a
    table/graph pair, every use-case gallery block is a complete table config, and none of those
    pages has a block the classifier drops.
    """
    assert _complete_snippets(), "no complete YAML config found in the corpus; the fenced-block regex broke"
    for page, why in ((README, "quick start"), (DOCS / "tutorial.md", "tutorial pair"), (DOCS / "examples.md", "use case gallery")):
        blocks: list[YamlSnippet] = _snippets_in(page)
        complete: list[YamlSnippet] = _snippets_in(page, complete_only=True)
        assert complete, f"{page.relative_to(ROOT)} {why}: classifier selected no complete config"
        assert len(complete) == len(blocks), f"{page.relative_to(ROOT)} {why}: classifier skipped {[s.location for s in blocks if s.kind is None]}"
    assert [snippet.kind for snippet in _snippets_in(README, complete_only=True)] == ["table", "graph"], (
        "README quick start must be one table then one graph config"
    )
    assert [snippet.kind for snippet in _snippets_in(DOCS / "tutorial.md", complete_only=True)] == ["table", "graph"], (
        "tutorial must be one table then one graph config"
    )
    assert {snippet.kind for snippet in _snippets_in(DOCS / "examples.md", complete_only=True)} == {"table"}, (
        "every gallery block is a complete table config"
    )


def test_yaml_snippets_classifier_skips_fragments() -> None:
    """Abbreviated and partial blocks are skipped by rule -- pinned against over-selection.

    Over-selection would fail CI on excerpts that are not configs at all, so two source-derived
    pins hold the line: every block in the corpus that elides with ``{...}`` / ``...`` is skipped
    (and such blocks exist, so the pin is not vacuous), and every block inside the table
    reference's merge-behavior section is skipped because those illustrate ``template``/``sections``
    merging rather than declaring a standalone config.
    """
    elided: list[YamlSnippet] = [s for s in _yaml_snippets() if any(token in s.block for token in PLACEHOLDER_TOKENS)]
    assert elided, "corpus no longer elides a YAML block with `{...}`/`...`; this pin went vacuous"
    assert all(s.kind is None for s in elided), f"classifier selected an elided block: {[s.location for s in elided if s.kind is not None]}"

    table_reference: Path = DOCS / "configuration" / "table.md"
    text: str = table_reference.read_text(encoding="utf-8")
    start, end = _section_range(text, "Merge Behavior (fastmerge)")
    first_line: int = text.count("\n", 0, start) + 1
    last_line: int = text.count("\n", 0, end) + 1
    merge_blocks: list[YamlSnippet] = [s for s in _snippets_in(table_reference) if first_line <= s.line < last_line]
    assert merge_blocks, "the merge-behavior section has no YAML block; this pin went vacuous"
    assert all(s.kind is None for s in merge_blocks), (
        f"classifier selected a merge fragment: {[s.location for s in merge_blocks if s.kind is not None]}"
    )


def test_readme_documentation_links_cover_every_nav_page() -> None:
    """The README documentation list links every top-level mkdocs nav page at its published URL.

    Both the nav and the expected target come from live source (``mkdocs.yml`` nav order plus
    ``site_url``), so a newly published page is orphaned in CI until the README links it -- the
    drift that left ``Fullmap`` unlinked here.
    """
    _assert_nav_parity(README, "Documentation")


def test_index_documentation_links_cover_every_nav_page() -> None:
    """``docs/index.md``'s Documentation Sections list reaches the same nav parity, repo-relative.

    The docs index renders inside the site, so its targets are docs-relative page paths rather than
    absolute URLs; the required set is still derived from the live nav.
    """
    _assert_nav_parity(INDEX, "Documentation Sections")


def test_readme_documentation_links_use_published_site_urls() -> None:
    """Every README documentation index link is a published-site URL, not a repo-relative path.

    The README renders on PyPI and GitHub, where ``installation.md`` is a dead link; pinning the
    host to the live ``mkdocs.yml`` ``site_url`` keeps the list usable off-repository and stops a
    contributor from "fixing" a link into a relative path.
    """
    base: str = _site_base()
    links: dict[str, str] = _list_links(README, "Documentation")
    assert links, "README '## Documentation' list is empty"
    broken: list[str] = [f"{text} -> {target}" for text, target in links.items() if not target.startswith(base)]
    assert not broken, f"README documentation links must be published-site URLs under {base}: {broken}"
