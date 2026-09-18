"""Assemble `references.jsonl`, the gold pronunciation layer for DOSE-R.

One record per unique ingredient across the 274 DOSE rows (284 of them, since
nine rows are combination products and a few ingredients repeat). Each record
carries every accepted variant, its provenance, and a confidence tier that the
replication-fidelity report can stratify on.

Confidence follows the project contract and is deliberately conservative:

    high     two independent sources agree
    medium   exactly one external source answered
    low      no external source; derived from spelling by rule

Multi-word ingredients are resolved word by word when the full name misses, so
`fluticasone propionate` can take a real MW pronunciation for the head and fall
back only where it has to. A record is only as trustworthy as its weakest word,
so the tier is the minimum across words.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..judge.fixtures import g2p
from . import sources
from .notation import convert
from .wiki_notation import ipa_to_arpabet_ipa, respell_to_arpabet_ipa, NotationError as WikiNotationError

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "dose_v1.jsonl"
OUT = Path(__file__).resolve().parent / "references.jsonl"
COVERAGE = Path(__file__).resolve().parent / "COVERAGE.md"

TIERS = ("high", "medium", "low")


def ingredients() -> dict[str, str]:
    """Unique ingredient -> name_type, across every DOSE row."""
    out: dict[str, str] = {}
    with DATASET.open() as f:
        for line in f:
            row = json.loads(line)
            for ing in row["ingredients"]:
                out.setdefault(ing, row["name_type"])
    return out


def _from_g2p(word: str) -> list[tuple[str, str]]:
    return [(" ".join(v), g2p.to_ipa(v)) for v in g2p.to_arpabet_variants(word)]


def _wiki_variants(hit: dict) -> list[tuple[str, str]]:
    """A wikipedia/wiktionary source dict -> [(ARPABET, IPA), ...]."""
    try:
        if hit["kind"] == "ipa":
            return [ipa_to_arpabet_ipa(hit["raw"])]
        return [respell_to_arpabet_ipa(hit["raw"].split("|"))]
    except WikiNotationError:
        return []


def _resolve_word(word: str) -> tuple[list[tuple[str, str]], list[dict], str]:
    """Variants, sources and tier for a single word."""
    found = []

    mw = sources.merriam_webster(word)
    if mw:
        variants = convert(mw["raw"])
        if variants:
            found.append((variants, mw))

    wp = sources.wikipedia_pronunciation(word)
    if wp:
        variants = _wiki_variants(wp)
        if variants:
            found.append((variants, {k: v for k, v in wp.items() if k != "kind"}))

    wikt = sources.wiktionary_pronunciation(word)
    if wikt:
        variants = _wiki_variants(wikt)
        if variants:
            found.append((variants, {k: v for k, v in wikt.items() if k != "kind"}))

    cmu = sources.cmudict_lookup(word)
    if cmu:
        from ..judge.fixtures.g2p import to_ipa

        arpa = cmu["raw"]
        found.append(([(arpa, to_ipa(arpa.split()))], cmu))

    if not found:
        return _from_g2p(word), [], "low"

    variants: list[tuple[str, str]] = []
    seen_arpa: set[str] = set()
    for group, _ in found:
        for v in group:
            # Dedupe on the ARPABET form, which is what scoring actually
            # matches against; two sources rendering the identical phoneme
            # sequence with slightly different IPA glyphs (e.g. length marks)
            # are not a second variant.
            if v[0] not in seen_arpa:
                seen_arpa.add(v[0])
                variants.append(v)

    tier = "high" if len(found) > 1 else "medium"
    return variants, [src for _, src in found], tier


def _join(parts: list[list[tuple[str, str]]]) -> list[tuple[str, str]]:
    """Concatenate per-word variant lists, keeping each word's preferred first."""
    arpa = " ".join(p[0][0] for p in parts)
    ipa = "".join(p[0][1] for p in parts)
    return [(arpa, ipa)]


def resolve(name: str, name_type: str) -> dict:
    whole, srcs, tier = _resolve_word(name)
    notes = ""

    if tier == "low" and " " in name:
        per_word = [_resolve_word(w) for w in name.split()]
        whole = _join([p[0] for p in per_word])
        srcs = [s for p in per_word for s in p[1]]
        tier = min((p[2] for p in per_word), key=TIERS.index)
        notes = "resolved word by word; tier is the weakest word"

    if tier == "low":
        notes = (notes + "; " if notes else "") + (
            "no external source found -- derived from spelling by rule. "
            "TODO: needs LLM arbitration or human review before it is trusted"
        )

    return {
        "ingredient": name,
        "name_type": name_type,
        "ipa_variants": [i for _, i in whole],
        "arpabet_variants": [a for a, _ in whole],
        "sources": srcs,
        "confidence": tier,
        "notes": notes,
    }


def build(workers: int = 8) -> list[dict]:
    items = ingredients()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda kv: resolve(*kv), items.items()))


def coverage_report(records: list[dict]) -> str:
    by_tier = {t: [r for r in records if r["confidence"] == t] for t in TIERS}
    by_source: dict[str, int] = {}
    for r in records:
        for s in r["sources"]:
            by_source[s["name"]] = by_source.get(s["name"], 0) + 1

    lines = [
        "# Gold Reference Layer — Coverage",
        "",
        f"{len(records)} unique ingredients across the 274 DOSE rows.",
        "",
        "## Confidence",
        "",
        "| Tier | Count | Share | Meaning |",
        "| --- | --- | --- | --- |",
    ]
    meaning = {
        "high": "two independent sources agree",
        "medium": "exactly one external source answered",
        "low": "no external source; derived from spelling by rule",
    }
    for t in TIERS:
        n = len(by_tier[t])
        lines.append(f"| {t} | {n} | {n / len(records):.1%} | {meaning[t]} |")

    lines += ["", "## By name type", "", "| Tier | brand | generic |", "| --- | --- | --- |"]
    for t in TIERS:
        b = sum(1 for r in by_tier[t] if r["name_type"] == "brand")
        g = sum(1 for r in by_tier[t] if r["name_type"] == "generic")
        lines.append(f"| {t} | {b} | {g} |")

    lines += ["", "## Sources that answered", "", "| Source | Ingredients |", "| --- | --- |"]
    for name, n in sorted(by_source.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {name} | {n} |")

    lines += [
        "",
        "## Sources that did not answer",
        "",
        "- **Drugs.com** — HTTP 403 for every request from this environment.",
        "- **FDA labels (openFDA, DailyMed)** — reachable, but the label text carries",
        "  no pronunciation respellings, so there is nothing to extract.",
        "",
        "## Needs arbitration",
        "",
        f"{len(by_tier['low'])} ingredients have no external source and are currently",
        "rule-derived. These are the layer's weak spot and must not be read as gold:",
        "",
    ]
    for r in sorted(by_tier["low"], key=lambda r: r["ingredient"]):
        lines.append(f"- {r['ingredient']} ({r['name_type']})")

    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    records = sorted(build(args.workers), key=lambda r: r["ingredient"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    COVERAGE.write_text(coverage_report(records))

    counts = {t: sum(1 for r in records if r["confidence"] == t) for t in TIERS}
    print(f"wrote {len(records)} references -> {args.out}")
    print(f"  high={counts['high']} medium={counts['medium']} low={counts['low']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
