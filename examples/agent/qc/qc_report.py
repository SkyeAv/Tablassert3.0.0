"""QC assay report: read the agent's state + derived configs + built KGX for a batch of PMCs and emit a
markdown report for manual review (per-PMC config + predicate/encodings/provenance + coverage + KG sample
+ aggregate quality metrics)."""

import contextlib
import hashlib
import json
import re
import sys
from pathlib import Path

import yaml

STATE_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".tablassert/qc-assay")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else STATE_DIR / "QC_REPORT.md"

# predicates that are "specific" vs generic fallbacks (for a heuristic appropriateness flag)
GENERIC_PREDICATES = {"associated_with", "related_to", "biolink:associated_with", "biolink:related_to"}

# absolute-path roots that indicate a machine-specific local filesystem path (never a public URL)
_LOCAL_ROOTS = r"(?:home|Users|tmp|root|var|mnt|srv|opt|private)"


def redact_paths(text: str) -> str:
    """Redact absolute local filesystem paths before they reach the report.

    Paths under ``STATE_DIR`` are normalized to ``<state-dir>/...``; any other absolute local path
    becomes the stable ``<local-path>`` placeholder. Public URLs are untouched (the lookbehind
    rejects a match preceded by word chars, e.g. the host of ``https://host/...``).
    """
    out = text.replace(str(STATE_DIR.resolve()), "<state-dir>")
    return re.sub(rf"(?<![\w@%+./-])/{_LOCAL_ROOTS}/[^\s'\"`)\]>]+", "<local-path>", out)


def load_state() -> dict:
    return json.loads((STATE_DIR / "state.json").read_text())


def load_config(pmc: str) -> tuple[str, dict] | None:
    for name in (f"{pmc}.yaml", f"{pmc}.derived.yaml"):
        p = STATE_DIR / "configs" / name
        if p.is_file():
            try:
                return p.read_text(), yaml.safe_load(p.read_text())
            except Exception:
                return p.read_text(), {}
    return None


def kg_counts(pmc: str) -> tuple[int, int]:
    bdir = STATE_DIR / "builds" / pmc
    n = e = 0
    np_, ep = bdir / "agent_0.0.1.nodes.ndjson", bdir / "agent_0.0.1.edges.ndjson"
    if np_.is_file():
        n = sum(1 for line in np_.open() if line.strip())
    if ep.is_file():
        e = sum(1 for line in ep.open() if line.strip())
    return n, e


def sample_edges(pmc: str, k: int = 5) -> list[dict]:
    ep = STATE_DIR / "builds" / pmc / "agent_0.0.1.edges.ndjson"
    if not ep.is_file():
        return []
    out = []
    with ep.open() as fh:
        for line in fh:
            if line.strip():
                with contextlib.suppress(Exception):
                    out.append(json.loads(line))
            if len(out) >= k:
                break
    return out


def first_section(cfg: dict) -> dict:
    # multi-section: {template, sections: [...]}
    secs = cfg.get("sections")
    if secs:
        return secs[0] or {}
    # single section nested in template: {template: {source, statement, provenance}}
    tmpl = cfg.get("template") or {}
    if tmpl.get("statement") or tmpl.get("source"):
        return tmpl
    # top-level single section: {source, statement, provenance}
    if cfg.get("statement") or cfg.get("source"):
        return cfg
    return {}


def main() -> None:
    state = load_state()
    records = state.get("records", {})
    lines: list[str] = []
    lines.append("# Tablassert agent QC assay report\n")
    lines.append(f"State dir: `{redact_paths(str(STATE_DIR))}`  ·  PMCs assayed: {len(records)}\n")

    mapped = skipped = 0
    coverages: list[float] = []
    predicate_counts: dict[str, int] = {}
    generic_predicate_pmc: list[str] = []
    error_pmc: list[tuple[str, str]] = []

    for pmc, rec in records.items():
        status = rec.get("status", "?")
        cov = float(rec.get("best_coverage", 0.0) or 0.0)
        notes = rec.get("notes", "") or ""
        if status == "MAPPED":
            mapped += 1
        elif status == "SKIPPED":
            skipped += 1
        coverages.append(cov)
        if notes and "SKIPPED" in notes:
            error_pmc.append((pmc, redact_paths(notes[:160])))

        loaded = load_config(pmc)
        cfg_text, cfg = loaded if loaded else ("", {})
        sec = first_section(cfg) if cfg else {}
        stmt = sec.get("statement") or {}
        pred = stmt.get("predicate", "?")
        predicate_counts[pred] = predicate_counts.get(pred, 0) + 1
        if pred in GENERIC_PREDICATES:
            generic_predicate_pmc.append(pmc)
        subj = stmt.get("subject") or {}
        obj = stmt.get("object") or {}
        src = sec.get("source") or {}
        prov = (cfg.get("template") or {}).get("provenance") or cfg.get("provenance") or {}
        n, e = kg_counts(pmc)

        lines.append(f"\n---\n## {pmc} — **{status}**\n")
        lines.append(f"- **best coverage:** {cov:.3f}")
        lines.append(f"- **KG:** {n} nodes / {e} edges")
        lines.append(f"- **predicate:** `{pred}`" + ("  ⚠️ *generic fallback*" if pred in GENERIC_PREDICATES else ""))
        lines.append(
            f"- **subject:** method={subj.get('method')} encoding={subj.get('encoding')} "
            f"prioritize={subj.get('prioritize')} taxon={subj.get('taxon')}"
        )
        lines.append(f"- **object:** method={obj.get('method')} encoding={obj.get('encoding')} prioritize={obj.get('prioritize')}")
        lines.append(f"- **source:** kind={src.get('kind')} sheet={src.get('sheet')!r} local={redact_paths(str(src.get('local')))}")
        lines.append(f"- **provenance:** {prov}")
        # Shared with QC_REVIEW.md so a future report/review mismatch against the configs is detectable.
        cfg_sha = hashlib.sha256(cfg_text.encode()).hexdigest()[:12] if cfg_text else "-"
        lines.append(f"- **config sha256:** `{cfg_sha}`")
        if notes:
            lines.append(f"- **notes:** {redact_paths(notes[:200])}")
        lines.append(f"\n### Derived config (`configs/{pmc}.yaml`)\n")
        lines.append("```yaml")
        lines.append(redact_paths(cfg_text.strip())[:3000] if cfg_text else "(no config produced)")
        lines.append("```")
        edges = sample_edges(pmc, 5)
        if edges:
            lines.append(f"\n### Sample edges (first {len(edges)})\n")
            lines.append("```json")
            for ed in edges:
                compact = {k: ed.get(k) for k in ("subject", "predicate", "object", "relation", "primary_knowledge_source") if k in ed}
                lines.append(json.dumps(compact))
            lines.append("```")

    # aggregate
    avg_cov = sum(coverages) / len(coverages) if coverages else 0.0
    agg = ["\n---\n# Aggregate quality metrics\n"]
    agg.append(f"- PMCs assayed: **{len(records)}**")
    agg.append(
        f"- MAPPED: **{mapped}**  ·  SKIPPED: **{skipped}**  ·  MAPPED rate: **{mapped / len(records) * 100:.0f}%**" if records else "- no records"
    )
    agg.append(f"- mean best coverage: **{avg_cov:.3f}**")
    agg.append("- predicate distribution: " + ", ".join(f"`{p}`\u00d7{c}" for p, c in sorted(predicate_counts.items(), key=lambda kv: -kv[1])))
    if generic_predicate_pmc:
        agg.append(f"- ⚠️ generic-fallback predicate used for: {', '.join(generic_predicate_pmc)}")
    if error_pmc:
        agg.append("- SKIPPED reasons:")
        for pmc, note in error_pmc:
            agg.append(f"  - {pmc}: {note}")
    lines = agg + lines

    OUT.write_text("\n".join(lines))
    print(f"QC report -> {OUT}")
    print(f"MAPPED={mapped} SKIPPED={skipped} mean_cov={avg_cov:.3f}")
    print("predicates:", predicate_counts)
    if generic_predicate_pmc:
        print("generic-fallback predicate PMCs:", generic_predicate_pmc)


if __name__ == "__main__":
    main()
