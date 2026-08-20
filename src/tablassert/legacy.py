"""Load legacy YAML configs without silently losing data to duplicate mapping keys.

WHY: PyYAML's default mapping construction keeps only the LAST occurrence of a duplicate
key, silently dropping everything earlier occurrences contributed. Legacy human-curated
table configs accumulated such duplicates over years, and a plain ``yaml.safe_load`` would
ingest them with data lost and no signal. This loader DETECTS every duplicate, MERGES the
occurrences (dict+dict deep-merged, list+list extended, later value wins otherwise), and
fires :class:`~tablassert.errors.LegacyDuplicateKeyWarning` per duplicate so curators can
fix the source file.

Safety stance mirrors :func:`tablassert.ingests.from_yaml` — configs are untrusted, so
parsing runs on a ``yaml.SafeLoader`` subclass: only the safe tags are resolved and no
arbitrary Python objects are constructed. The pure-Python loader is used because the
duplicate hook replaces the mapping constructor; legacy files are small, so the C loader's
speed buys nothing here.
"""

from __future__ import annotations

import collections.abc
import glob
import re
import warnings
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, cast
from urllib.parse import urlparse

import pydantic
import yaml
from yaml import SafeLoader
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode, Node

from tablassert.errors import LegacyDuplicateKeyWarning, LegacySourceUnresolvedError, LegacyUnsupportedSyntaxError
from tablassert.ingests import fastmerge, to_sections
from tablassert.models import Reindex, Section


class _LegacyDuplicateMergingSafeLoader(SafeLoader):
    """SafeLoader that merges duplicate mapping keys instead of dropping them.

    See the module docstring for WHY. Merge semantics are delegated to
    :func:`tablassert.ingests.fastmerge` so this loader and the production
    section-merge path can never drift apart.
    """

    def construct_mapping(self, node: Node, deep: bool = False) -> dict[Any, Any]:
        """Build one mapping, warning on and merging every duplicated key.

        Args:
            node: The mapping node being constructed.
            deep: Accepted for ``SafeConstructor`` signature compatibility; values are
                always constructed eagerly (see below), which subsumes it.

        Returns:
            The mapping with duplicate keys merged instead of overwritten.

        Note:
            Values are constructed with ``deep=True``: PyYAML's lazy two-phase
            construction returns EMPTY placeholders for nested containers whose fills are
            deferred, so merging them (or deep-copying them) would freeze the emptiness and
            lose every item once the deferred fills land on the orphaned originals.
        """
        if not isinstance(node, MappingNode):
            raise ConstructorError(None, None, f"expected a mapping node, but found {node.tag}", node.start_mark)
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key: Any = self.construct_object(key_node, deep=True)
            if not isinstance(key, collections.abc.Hashable):
                raise ConstructorError("while constructing a mapping", node.start_mark, "found unhashable key", key_node.start_mark)
            value: Any = self.construct_object(value_node, deep=True)
            if key in mapping:
                warnings.warn(
                    f"Duplicate key `{key}` at {key_node.start_mark.name}:{key_node.start_mark.line + 1} — merged with the "
                    "earlier value instead of dropping it (PyYAML silently keeps only the last occurrence of a duplicate key).",
                    LegacyDuplicateKeyWarning,
                    stacklevel=2,
                )
                # deepcopy: the earlier value may be shared with another node through a YAML alias;
                # fastmerge mutates in place, so merging into the shared object would silently
                # rewrite the aliased original too.
                mapping[key] = fastmerge(deepcopy(mapping[key]), value)
            else:
                mapping[key] = value
        return mapping


def load_legacy_yaml(p: Path) -> object:
    """Read a legacy YAML config, merging duplicate mapping keys instead of dropping them.

    Duplicate-key occurrences fire :class:`~tablassert.errors.LegacyDuplicateKeyWarning`
    naming the key and its line; the merged result is returned regardless.

    Args:
        p: Path to the legacy YAML file.

    Returns:
        Parsed YAML content (typically a ``dict``).
    """
    with p.open("r") as f:
        # SafeLoader (never Loader/CLoader): legacy configs are untrusted; SafeLoader resolves
        # only the safe tags and refuses arbitrary-object construction (same stance as
        # ingests.from_yaml's CSafeLoader).
        return yaml.load(f, Loader=_LegacyDuplicateMergingSafeLoader)


# --- US-003: legacy -> v12 conversion ---------------------------------------- #

#: The only keys a v12 section carries. Anything else in a legacy template or section entry
#: has no v12 home, so it is rejected instead of silently dropped.
SECTION_FIELDS: frozenset[str] = frozenset({"source", "statement", "provenance", "annotations"})

#: Legacy spellings of the removed ``relationship_strength`` annotation, case- and
#: separator-insensitive (space, underscore, dash, or none). The rewrite RENAMES the
#: annotation to the canonical ``effect_size`` slot and keeps every other encoding field
#: untouched; the ``effect_size``/``effect_type`` pairing policy is deliberately NOT
#: re-implemented here — unpaired halves pass through to ``Section`` validation, which
#: drops them with an :class:`~tablassert.errors.UnpairedEffectAnnotationWarning`.
_RELATIONSHIP_STRENGTH: re.Pattern[str] = re.compile(r"^relationship[\s_.\-]*strength$", re.IGNORECASE)

#: A payload directory shaped like ``PMC<n>.<v>`` (the ``s3://pmc-oa-opendata`` article-version
#: prefix a fetch writes under). Only a resolved payload under such a directory reveals its real
#: S3 object; anything else keeps the legacy url verbatim rather than guessing one.
_PAYLOAD_STEM: re.Pattern[str] = re.compile(r"^PMC\d+\.\d+$")


def _unsupported(path: Path, detail: str) -> LegacyUnsupportedSyntaxError:
    return LegacyUnsupportedSyntaxError(path, detail)


def _check_block_keys(path: Path, where: str, block: dict[Any, Any]) -> None:
    """Reject any key the v12 section shape cannot express (no partial silent writes)."""
    unknown: list[str] = sorted(str(key) for key in block if key not in SECTION_FIELDS)
    if unknown:
        raise _unsupported(path, f"`{where}` declares unsupported key(s) {unknown}; a v12 section carries only {sorted(SECTION_FIELDS)}")


def _publication(section: dict[str, Any]) -> str | None:
    """The section's ``provenance.publication`` when declared as a string, else ``None``."""
    provenance: object = section.get("provenance")
    if isinstance(provenance, dict):
        publication: object = provenance.get("publication")
        if isinstance(publication, str) and publication.strip():
            return publication
    return None


def _rewrite_effect_aliases(path: Path, index: int, section: dict[str, Any]) -> None:
    """Rename legacy ``relationship strength`` annotations to ``effect_size`` in place.

    Only the ``annotation`` name is rewritten; ``method``/``encoding`` and every other
    encoding field pass through untouched. The pairing policy lives on ``Section``.
    """
    annotations: object = section.get("annotations")
    if annotations is None:
        return
    if not isinstance(annotations, list):
        raise _unsupported(path, f"section {index}: `annotations` must hold a list, found {type(annotations).__name__}")
    for entry in annotations:
        if not isinstance(entry, dict):
            raise _unsupported(path, f"section {index}: every `annotations` entry must hold a mapping, found {type(entry).__name__}")
        name: object = entry.get("annotation")
        if isinstance(name, str) and _RELATIONSHIP_STRENGTH.match(name):
            entry["annotation"] = "effect_size"


def _check_reindex(path: Path, index: int, source: dict[str, Any]) -> None:
    """Validate every ``source.reindex`` entry against ``models.Reindex`` without mutating it.

    Valid blocks pass through unchanged; an entry the v12 ``Reindex`` model cannot express
    fails the WHOLE conversion instead of leaking a half-translated filter into the output.
    """
    reindex: object = source.get("reindex")
    if reindex is None:
        return
    if not isinstance(reindex, list):
        raise _unsupported(path, f"section {index}: `source.reindex` must hold a list, found {type(reindex).__name__}")
    for entry in reindex:
        if not isinstance(entry, dict):
            raise _unsupported(path, f"section {index}: every `source.reindex` entry must hold a mapping, found {type(entry).__name__}")
        try:
            Reindex.model_validate(entry)
        except pydantic.ValidationError as e:
            problems: str = "; ".join(f"{'.'.join(str(part) for part in err['loc']) or 'entry'}: {err['msg']}" for err in e.errors())
            raise _unsupported(path, f"section {index}: `source.reindex` entry {entry} is not expressible in v12 ({problems})") from e


def _match_payload(downloads: Path, name: str, publication: str | None) -> Path | None:
    """Find ``name`` under ``downloads`` (recursive), preferring the article's own directory.

    A downloads parent can hold many articles, and generic payload names (``media-1.xlsx``)
    collide across them, so a match under a directory named after the section's publication
    wins over an elsewhere match; ties break deterministically on the sorted path.
    """
    matches: list[Path] = sorted(downloads.rglob(glob.escape(name)))
    if not matches:
        return None
    if publication:
        wanted: str = publication.strip().upper()
        for match in matches:
            if any(parent.name.upper() == wanted for parent in match.parents):
                return match
    return matches[0]


def _match_candidates(downloads: Path | None, candidates: list[str], publication: str | None, tried: list[str]) -> Path | None:
    """Match the candidate basenames against ``downloads`` in order; the first hit wins.

    Every attempt is recorded in ``tried`` so an unresolved source can name ALL of the
    basenames/locations that were tried, not just the last one.
    """
    if downloads is None:
        return None
    for candidate in candidates:
        tried.append(f"basename {candidate!r} under {downloads} (recursive)")
        matched: Path | None = _match_payload(downloads, candidate, publication)
        if matched is not None:
            return matched
    return None


def _fetch_payload(path: Path, local: str, downloads: Path | None, candidates: list[str], publication: str | None, tried: list[str]) -> Path | None:
    """Fetch the article payload through the agent's PMC downloader, then retry the match
    against EVERY candidate basename (the fetched objects carry the url basenames, which
    usually differ from the human ``local`` alias).

    ``tablassert.agent`` is imported LAZILY here (never at module scope): conversion must stay
    importable in the base environment, and the fetch path is the only piece that needs the
    agent module. Fetch failures leave the source unresolved, so they surface as
    ``legacy-source-unresolved`` with the cause chained.
    """
    if downloads is None:
        tried.append("fetch=True needs a downloads directory to fetch into (got None)")
        return None
    if publication is None:
        tried.append("fetch needs `provenance.publication` to know which article to download (none declared)")
        return None
    # Below the degenerate-input guards: a fetch=True call that can never fetch must not
    # import tablassert.agent (legacy conversion stays importable in the base environment).
    from tablassert.agent import fetch_pmc_article, normalize_pmc_id  # lazy: keep legacy import agent-free

    try:
        pmc: str = normalize_pmc_id(publication)
    except ValueError as e:
        tried.append(f"`provenance.publication` {publication!r} is not a PMC id ({e})")
        return None
    target: Path = downloads / pmc
    tried.append(f"PMC open-access fetch of {pmc} into {target}")
    try:
        fetch_pmc_article(pmc, target)
    except (ValueError, OSError) as e:
        raise LegacySourceUnresolvedError(path, local, tried) from e
    return _match_candidates(downloads, candidates, publication, tried)


def _url_basenames(path: Path, index: int, source: dict[str, Any]) -> list[str]:
    """The basename of every ``source.url`` entry, in declared order.

    Legacy ``url`` entries point at the REAL payload objects while ``local`` holds a human
    alias that rarely exists on disk, so the url basenames are the fallback resolution
    candidates after the local basename. A malformed ``url`` is an unsupported construct:
    v12 carries ``url`` as a list of URL strings, so anything else can neither be resolved
    against the downloads directory nor expressed in the output.
    """
    url: object = source.get("url")
    if url is None:
        return []
    if not isinstance(url, list):
        raise _unsupported(path, f"section {index}: `source.url` must hold a list, found {type(url).__name__}")
    basenames: list[str] = []
    for entry in url:
        if not isinstance(entry, str):
            raise _unsupported(path, f"section {index}: every `source.url` entry must hold a URL string, found {type(entry).__name__}")
        # urlparse first: a query string or fragment is not part of the payload's filename.
        name: str = PurePosixPath(urlparse(entry).path).name
        if name:
            basenames.append(name)
    return basenames


def _resolve_source(path: Path, index: int, section: dict[str, Any], downloads: Path | None, fetch: bool) -> None:
    """Point ``source.local`` at the real downloaded payload and repair ``source.url``.

    The legacy ``local`` path (``./DATALAKE/...``) is never kept, since it no longer exists:
    resolution tries candidate basenames against the downloads directory IN ORDER — the local
    basename first, then each ``source.url`` entry's basename (the urls hold the real payload
    filenames) — and the first recursive match wins, still preferring a hit under the
    section's own publication directory. The url becomes the real S3 object only when the
    resolved payload reveals its ``PMC<n>.<v>`` stem; otherwise the legacy url is kept
    verbatim (never fabricated).
    """
    source: object = section.get("source")
    if source is None:
        return  # Section validation owns missing-source failures
    if not isinstance(source, dict):
        raise _unsupported(path, f"section {index}: `source` must hold a mapping, found {type(source).__name__}")
    _check_reindex(path, index, source)
    local: object = source.get("local")
    if local is None:
        return  # Section validation owns missing-local failures
    if not isinstance(local, str):
        raise _unsupported(path, f"section {index}: `source.local` must hold a path string, found {type(local).__name__}")
    candidates: list[str] = []
    for candidate in [PurePosixPath(local).name, *_url_basenames(path, index, source)]:
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    publication: str | None = _publication(section)
    tried: list[str] = []
    matched: Path | None = _match_candidates(downloads, candidates, publication, tried)
    if matched is None and fetch:
        matched = _fetch_payload(path, local, downloads, candidates, publication, tried)
    if matched is None:
        if downloads is None:
            tried.append("no downloads directory supplied")
        raise LegacySourceUnresolvedError(path, local, tried)
    resolved: Path = matched.resolve()
    source["local"] = str(resolved)
    stem: str = resolved.parent.name
    if _PAYLOAD_STEM.match(stem):
        from tablassert.agent import public_url  # lazy: keep legacy import agent-free

        source["url"] = [public_url(stem, resolved.name)]


def _dedupe_list_entries(value: Any) -> None:
    """Drop exact-duplicate entries from every list in place, preserving first-seen order.

    WHY: the template-over-section overlay EXTENDS list-valued keys (``fastmerge``), so an
    entry declared IDENTICALLY on the template and a section (e.g. the same qualifier or
    annotation twice) lands twice in the merged section. Distinct entries and their order
    are kept, non-list values pass through untouched, and the walk is bottom-up, so entries
    are compared after their own nested lists were deduped. The dedup lives in the CONVERTER
    only: ``fastmerge`` keeps its plain extend semantics on the production path.
    """
    if isinstance(value, dict):
        for item in value.values():
            _dedupe_list_entries(item)
    elif isinstance(value, list):
        unique: list[Any] = []
        for item in value:
            _dedupe_list_entries(item)
            if not any(item == kept for kept in unique):
                unique.append(item)
        value[:] = unique


def _merge_duplicate_qualifiers(path: Path, index: int, section: dict[str, Any]) -> None:
    """Merge same-key ``statement.qualifiers`` entries so each key survives exactly once.

    WHY: ``Section`` rejects a qualifier key declared twice (``qualifier-duplicated``), but
    the overlay EXTENDS the template's qualifiers with the section's — the MIN1 pattern, a
    generic template qualifier re-declared per section with a more specific encoding. Such a
    key cannot stay a pair in v12: the section's entry WINS the first-seen slot of the key
    (fastmerge's later-wins collision semantics — the section re-declares the key to
    specialize it, and the template's generic entry stays on sections that declare no own
    entry). Exact duplicates collapse to one entry silently; distinct keys keep their
    entries and order untouched, and the literal/vocabulary checks stay on ``Section``.
    """
    statement: object = section.get("statement")
    if not isinstance(statement, dict):
        return
    qualifiers: object = statement.get("qualifiers")
    if qualifiers is None:
        return
    if not isinstance(qualifiers, list):
        raise _unsupported(path, f"section {index}: `statement.qualifiers` must hold a list, found {type(qualifiers).__name__}")
    for position, entry in enumerate(qualifiers):
        if not isinstance(entry, dict):
            raise _unsupported(path, f"section {index}: every `statement.qualifiers` entry must hold a mapping, found {type(entry).__name__}")
        key: object = entry.get("qualifier")
        if not isinstance(key, str) or not key.strip():
            raise _unsupported(path, f"section {index}: `statement.qualifiers[{position}].qualifier` must hold a non-empty string")
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for entry in qualifiers:
        key = str(entry["qualifier"])
        if key not in merged:
            order.append(key)
        # exact duplicates collapse; a differing same-key entry wins its first-seen slot
        merged[key] = entry
    statement["qualifiers"] = [merged[key] for key in order]


def _section_label(index: int, section: dict[str, Any]) -> str:
    """Identify one expanded section in a diagnostic: its index, plus ``source.local`` when declared."""
    source: object = section.get("source")
    local: object = source.get("local") if isinstance(source, dict) else None
    if isinstance(local, str) and local.strip():
        return f"section {index} (source.local {local!r})"
    return f"section {index}"


def convert_legacy(path: Path, downloads: Path | None, fetch: bool = False) -> dict[str, Any]:
    """Convert one legacy table config into the v12 ``{template, sections}`` shape.

    Expansion REUSES :func:`tablassert.ingests.to_sections` (``fastmerge`` of template over
    each section), so legacy merge semantics and production merge semantics can never drift;
    a template-only file yields exactly one section. Each expanded section is then fully
    specified: exact-duplicate list entries from the overlay dropped, same-key qualifier
    entries merged (each qualifier key survives once), annotation aliases renamed,
    ``reindex`` validated, ``source.local`` resolved against the downloads payload (local
    basename first, then each ``source.url`` basename) and ``source.url`` repaired. A
    provenance shared by every section is hoisted into ``template`` (the v12 idiom); divergent
    provenances stay on their sections. The result serializes via :func:`tablassert.ingests.to_yaml` and every
    section constructs through the :class:`tablassert.models.Section` models.

    Args:
        path: Path to the legacy YAML config.
        downloads: Directory holding downloaded article payloads (``PMC<n>/PMC<n>.<v>/...``);
            ``None`` when no payload directory exists.
        fetch: When no local match exists, download the article payload from PMC open access
            (via ``provenance.publication``) into ``downloads`` before failing.

    Returns:
        The v12 config dict ``{"template": ..., "sections": [...]}``.

    Raises:
        LegacyUnsupportedSyntaxError: The file uses a construct v12 cannot express.
        LegacySourceUnresolvedError: A ``source.local`` stayed unmapped onto a payload.
    """
    raw: object = load_legacy_yaml(path)
    if not isinstance(raw, dict):
        raise _unsupported(path, f"the file must hold a mapping, found {type(raw).__name__}")
    unknown: list[str] = sorted(str(key) for key in raw if key not in ("template", "sections"))
    if unknown:
        raise _unsupported(path, f"unknown top-level key(s) {unknown}; only `template` and `sections` are supported")
    template: object = raw.get("template", {})
    if not isinstance(template, dict):
        raise _unsupported(path, f"`template` must hold a mapping, found {type(template).__name__}")
    _check_block_keys(path, "template", template)
    entries: object = raw.get("sections", [{}])
    if not isinstance(entries, list):
        raise _unsupported(path, f"`sections` must hold a list, found {type(entries).__name__}")
    if not entries:
        raise _unsupported(path, "`sections` is an empty list; a legacy config without sections omits the key")
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise _unsupported(path, f"sections[{position}] must hold a mapping, found {type(entry).__name__}")
        _check_block_keys(path, f"sections[{position}]", entry)

    sections: list[dict[str, Any]] = []
    # cast: ``to_sections`` returns one dict per section, but its annotation nests one list too deep.
    expanded: list[dict[str, Any]] = cast("list[dict[str, Any]]", to_sections(raw, path))
    for index, merged in enumerate(expanded):
        merged.pop("config", None)  # stamped by to_sections; not a v12 config field
        _dedupe_list_entries(merged)
        _merge_duplicate_qualifiers(path, index, merged)
        _rewrite_effect_aliases(path, index, merged)
        _resolve_source(path, index, merged, downloads, fetch)
        # Enforce the docstring guarantee: EVERY expanded section must construct through the
        # Section models, so a gap the overlay left (e.g. a template missing source/statement)
        # fails the conversion here instead of writing a .v12.yaml that fails downstream.
        try:
            Section.model_validate(merged)
        except pydantic.ValidationError as e:
            problems: str = "; ".join(f"{'.'.join(str(part) for part in err['loc']) or 'section'}: {err['msg']}" for err in e.errors())
            raise _unsupported(path, f"{_section_label(index, merged)} fails `Section` validation: {problems}") from e
        sections.append(merged)

    provenances: list[object] = [section.get("provenance") for section in sections]
    first: object = provenances[0]
    if isinstance(first, dict) and all(provenance == first for provenance in provenances):
        for section in sections:
            section.pop("provenance", None)
        return {"template": {"provenance": first}, "sections": sections}
    return {"template": {}, "sections": sections}
