from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tablassert.cli import schema
from tablassert.ingests import from_yaml
from tablassert.models import DEFAULT_RIG_UI_EXPLANATION, Section

# Tcode (lib.py) extends Section with per-section runtime-only fields that are injected by the
# pipeline, never authored in YAML. The exported Section schema must not leak them to config authors.
TCODE_RUNTIME_FIELDS: tuple[str, ...] = ("config", "store", "log", "qc", "release", "head", "name")


def _emit(capsys: Any, **kwargs: Any) -> dict[str, Any]:
    """Run the ``schema`` command directly and parse its stdout as JSON.

    Mirrors the repo convention (test_progress.py) of invoking command/pipeline
    functions in-process and capturing stdout with ``capsys`` rather than spawning
    a subprocess.
    """
    schema(**kwargs)
    captured: Any = capsys.readouterr()
    parsed: dict[str, Any] = json.loads(captured.out)
    return parsed


def test_schema_section_stdout_is_valid_json_with_core_fields_and_defaults(capsys: Any) -> None:
    """The default ``schema`` invocation emits the Section contract as valid JSON.

    Why: ``tablassert schema`` is the authoring contract for table configs. It must
    print parseable JSON that exposes exactly the four fields authors set
    (source/statement/provenance/annotations) and surfaces defaults so authors can
    tell what is optional. Pinning the required set guards against accidental
    runtime-field leakage into the author-facing surface.
    """
    parsed: dict[str, Any] = _emit(capsys)
    properties: dict[str, Any] = parsed["properties"]
    assert set(properties) == {"source", "statement", "provenance", "annotations"}
    assert parsed["required"] == ["source", "statement", "provenance"]
    # Defaults are surfaced: annotations is optional (default null) and the nested
    # Statement.predicate carries a default, so authors see what they may omit.
    assert properties["annotations"]["default"] is None
    assert "default" in parsed["$defs"]["Statement"]["properties"]["predicate"]


def test_schema_graph_stdout_is_valid_json_with_core_fields_and_defaults(capsys: Any) -> None:
    """``schema --model graph`` emits the full Graph contract including RIG defaults.

    Why: graph authors need the complete Graph field set, and the Resource Ingest
    Guide defaults (contributions/ui_explanation) must be visible so authors know
    what metadata is injected for them. ``ui_explanation`` is a literal default and
    is serialized verbatim; ``contributions`` is a declared array-of-string field.
    """
    parsed: dict[str, Any] = _emit(capsys, model="graph")
    properties: dict[str, Any] = parsed["properties"]
    assert {"name", "version", "description", "contributions", "ui_explanation", "tables", "fullmap"} <= set(properties)
    assert properties["ui_explanation"]["default"] == DEFAULT_RIG_UI_EXPLANATION
    assert properties["contributions"]["items"]["type"] == "string"


def test_schema_section_excludes_tcode_runtime_fields(capsys: Any) -> None:
    """The Section schema never exposes Tcode's pipeline-injected runtime fields.

    Why: config/store/log/qc/release/head/name are added by ``build_pipeline`` at
    runtime (lib.py:558) and are never valid in authored YAML. If they leaked into
    the exported schema, authors would be told to supply fields they cannot. This
    asserts they are absent from Section's own properties, absent as ``$defs``
    definition names, and (strongest) absent as quoted property keys anywhere in the
    serialized schema.
    """
    parsed: dict[str, Any] = _emit(capsys)
    for field in TCODE_RUNTIME_FIELDS:
        assert field not in parsed["properties"]
    # Absent as definition names across the whole schema.
    definition_names: set[str] = set(parsed.get("$defs", {})) | set(parsed["properties"])
    for field in TCODE_RUNTIME_FIELDS:
        assert field not in definition_names
    # Absent as quoted property keys anywhere in the serialized document.
    blob: str = json.dumps(parsed)
    for field in TCODE_RUNTIME_FIELDS:
        assert f'"{field}"' not in blob


def test_schema_output_flag_writes_file(capsys: Any, tmp_path: Path) -> None:
    """``--output`` writes the identical, indented schema to disk instead of stdout.

    Why: tooling (editors, CI validators) consumes the schema from a file. The file
    content must be byte-for-byte the same deterministic schema printed to stdout
    (indent=2), so both paths stay interchangeable.
    """
    out: Path = tmp_path / "section.json"
    schema(output=out)
    assert capsys.readouterr().out == ""
    text: str = out.read_text()
    parsed: dict[str, Any] = json.loads(text)
    assert parsed == Section.model_json_schema()
    # indent=2 is deterministic and human-diffable.
    assert text == json.dumps(Section.model_json_schema(), indent=2) + "\n"


def test_schema_section_describes_known_good_fixture(capsys: Any, fixtures_path: Path) -> None:
    """The exported Section schema describes a real, valid section fixture.

    Why: this is the required conformance proof without taking a ``jsonschema``
    dependency — every top-level key of a known-good fixture must be a declared
    schema property, and the fixture must round-trip through ``Section.model_validate``.
    Together these show the schema actually matches what valid configs look like.
    """
    parsed: dict[str, Any] = _emit(capsys)
    fixture: dict[str, Any] = from_yaml(fixtures_path / "minimal_section.yaml")  # pyright: ignore
    for key in fixture:
        assert key in parsed["properties"]
    validated: Section = Section.model_validate(fixture)
    assert isinstance(validated, Section)
