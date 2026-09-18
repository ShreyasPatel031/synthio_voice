"""Build `data/reference_audio/AUDIO_COVERAGE.md` from the current manifest.

Coverage is reported two ways for Merriam-Webster because its manifest holds
two different things under one source: a "full" clip pronounces the whole
ingredient, a "component" clip pronounces only one word of a multi-word
generic (see `audio_sources.py`). Only "full" clips satisfy what DOSE scores
-- the name as it appears in the carrier sentence -- so they are what this
report calls "MW audio" everywhere it matters for scoring; "any coverage"
(full + component) is reported alongside as the superset.

Full-name coverage itself is computed generically across however many
sources the manifest carries (`FULL_COVERAGE_SOURCES` below), not just two,
so a fourth source needs only a label added there.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dose_r.references import audio_manifest

DATASET = ROOT / "data" / "dose_v1.jsonl"
REFERENCES = ROOT / "dose_r" / "references" / "references.jsonl"
OUT = ROOT / "data" / "reference_audio" / "AUDIO_COVERAGE.md"

# source -> display label, in the order the headline table lists them.
FULL_COVERAGE_SOURCES = {
    "drugs.com": "Drugs.com",
    "merriam-webster": "Merriam-Webster (full name)",
    "umich": "UMich",
}


def ingredient_types() -> dict[str, str]:
    with REFERENCES.open() as f:
        return {r["ingredient"]: r["name_type"] for r in (json.loads(l) for l in f if l.strip())}


def confidence_tiers() -> dict[str, str]:
    with REFERENCES.open() as f:
        return {r["ingredient"]: r["confidence"] for r in (json.loads(l) for l in f if l.strip())}


def pct(n: int, d: int) -> str:
    return f"{n / d:.0%}" if d else "0%"


def table(rows: list[tuple], header: tuple) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return lines


def build() -> str:
    records = audio_manifest.load()
    types = ingredient_types()
    tiers = confidence_tiers()
    all_ingredients = set(types)

    dc = {r["ingredient"] for r in records if r["source"] == "drugs.com"}
    mw_full = {r["ingredient"] for r in records if r["source"] == "merriam-webster" and r["coverage"] == "full"}
    mw_component = {r["ingredient"] for r in records if r["source"] == "merriam-webster" and r["coverage"] == "component"}
    mw_any = mw_full | mw_component

    full_coverage = {
        "drugs.com": dc,
        "merriam-webster": mw_full,
        "umich": {r["ingredient"] for r in records if r["source"] == "umich"},
    }
    source_labels = {s: FULL_COVERAGE_SOURCES[s] for s in full_coverage}

    union = set().union(*full_coverage.values())
    hit_counts = Counter(i for s in full_coverage.values() for i in s)
    cross_checkable = {i for i, n in hit_counts.items() if n >= 2}
    all_sources = {i for i, n in hit_counts.items() if n == len(full_coverage)}
    only = {
        s: names - set().union(*(o for other, o in full_coverage.items() if other != s))
        for s, names in full_coverage.items()
    }
    none = all_ingredients - union

    dupes = audio_manifest.duplicate_groups(records)
    partial_flags = [
        r for r in records if r["status"] == "flagged" and r["coverage"] == "component"
    ]

    sources_prose = ", ".join(source_labels.values())
    lines = [
        "# Reference Audio -- Coverage",
        "",
        "Human-recorded pronunciation audio for the 284 unique ingredients across",
        f"the 274 DOSE rows, from three sources: {sources_prose}.",
        "Merriam-Webster's Medical API; Drugs.com collected by hand -- see",
        "`data/collected/HANDOFF_AUDIO_COLLECTION.md`, drugs.com 403s this environment",
        "on every request; UMich's student pronunciation page via the Wayback Machine,",
        "since the live site 403s behind a Cloudflare challenge. \"Coverage\" below",
        "means a clip that pronounces the *whole* ingredient name, since that is what",
        "DOSE scores; Merriam-Webster's word-level partial clips are reported",
        "separately.",
        "",
        "## Headline",
        "",
        *table(
            [
                ("Total unique ingredients", len(all_ingredients), "100%"),
                *[
                    (f"{label} audio", len(full_coverage[s]), pct(len(full_coverage[s]), len(all_ingredients)))
                    for s, label in source_labels.items()
                ],
                ("Union -- any audio", len(union), pct(len(union), len(all_ingredients))),
                ("2+ sources (cross-checkable)", len(cross_checkable), pct(len(cross_checkable), len(all_ingredients))),
                ("All 3 sources", len(all_sources), pct(len(all_sources), len(all_ingredients))),
                *[
                    (f"Only {label}", len(only[s]), pct(len(only[s]), len(all_ingredients)))
                    for s, label in source_labels.items()
                ],
                ("No audio anywhere", len(none), pct(len(none), len(all_ingredients))),
            ],
            ("Metric", "Count", "Share"),
        ),
        "",
        "## By name type",
        "",
        *table(
            [
                (
                    label,
                    sum(1 for i in group if types[i] == "brand"),
                    sum(1 for i in group if types[i] == "generic"),
                    len(group),
                )
                for label, group in [
                    ("2+ sources", cross_checkable),
                    *[(f"Only {source_labels[s]}", only[s]) for s in full_coverage],
                    ("No audio anywhere", none),
                ]
            ],
            ("Group", "brand", "generic", "total"),
        ),
        "",
        "## By reference-layer confidence tier",
        "",
        "Cross-referencing audio coverage against the phoneme reference layer's own",
        "confidence tier (`dose_r/references/COVERAGE.md`): where the two disagree --",
        "audio present but tier `low`, or tier `high` but no audio -- is exactly where",
        "independent verification helps most.",
        "",
        *table(
            [
                (
                    tier,
                    sum(1 for i in cross_checkable if tiers.get(i) == tier),
                    sum(1 for i in (union - cross_checkable) if tiers.get(i) == tier),
                    sum(1 for i in none if tiers.get(i) == tier),
                )
                for tier in ("high", "medium", "low")
            ],
            ("Tier", "2+ sources", "1 source", "no audio"),
        ),
        "",
        "## Cross-source agreement",
        "",
        "Every ingredient with audio from 2 or more sources, with each source's",
        "clip duration -- a rough plausibility check, not a substitute for an ear",
        "check. UMich's clips run consistently longer than the other sources' for",
        "the same name (roughly 1.3x-2x), which reads as a slower, more deliberate",
        "teaching-recording pace rather than a name mismatch: `audio_verify`'s",
        "syllable-outlier check, which flags a clip disproportionate to *its own*",
        "batch, raised nothing for UMich because the lengthening is uniform across",
        "its whole batch, not isolated to one name.",
        "",
        *table(
            [
                (
                    ing,
                    ", ".join(
                        f"{source_labels[s]} {r['duration_s']:.3f}s"
                        for s in full_coverage
                        if ing in full_coverage[s]
                        for r in records
                        if r["ingredient"] == ing and r["source"] == s and r["coverage"] == "full"
                    ),
                )
                for ing in sorted(cross_checkable)
            ],
            ("Ingredient", "Durations by source"),
        ),
        "",
        "## Merriam-Webster: reconciling row counts",
        "",
        f"The manifest carries {sum(1 for r in records if r['source']=='merriam-webster')} "
        f"Merriam-Webster rows for {len(mw_any)} unique ingredients -- more rows than",
        "ingredients because two multi-word generics only ever resolved word-by-word",
        "and get one row per word that answered (`dimethyl fumarate`: 2 rows;",
        "`formoterol fumarate dihydrate`: 3 rows -- 3 extra rows over 89 ingredients",
        "= 92).",
        "",
        f"Of those {len(mw_any)}, {len(mw_full)} carry a clip for the *whole* name",
        f"(`coverage: full`, unflagged) and {len(mw_component)} carry only a",
        "single word of a multi-word generic (`coverage: component`, flagged as",
        "partial -- listed below). This report counts only the "
        f"{len(mw_full)} full clips as \"Merriam-Webster audio\" for scoring purposes,",
        "since a component clip does not say the name DOSE asks about.",
        "",
        "This does not exactly reproduce an expected external count of 87: 82 (full",
        "only) and 89 (full + any component) bracket it, and 87 falls between the",
        "two. The most likely reading is that 87 counts most but not all of the 7",
        "component-only ingredients as \"available\" -- a judgment call this report",
        "did not have grounds to make one way or the other without a live",
        "`MW_MEDICAL_KEY` re-check (not set in this environment) or the original",
        "count's own criteria. All 7 component-only ingredients are listed below so",
        "a human can decide.",
        "",
        "Component-only Merriam-Webster ingredients (partial, not full-name, audio):",
        "",
        *[
            f"- `{ing}` -- only "
            + ", ".join(
                sorted(
                    r["query"]
                    for r in records
                    if r["ingredient"] == ing and r["coverage"] == "component"
                )
            )
            + " answered"
            for ing in sorted(mw_component)
        ],
        "",
        "## Flagged clips",
        "",
        "### Possible name mismatch (syllable-outlier check)",
        "",
        "Duration-per-syllable outliers relative to their own source's batch median",
        "(see `audio_verify.syllable_outliers`) -- candidates for a human ear check,",
        "not discarded. Every one so far is a brand-name clip running long, consistent",
        "with (but not proof of) a brand page's audio actually pronouncing its",
        "generic, the failure mode confirmed possible for Anktiva in the handoff doc.",
        "",
        *table(
            [
                (source_labels[r["source"]], r["ingredient"], r["duration_s"], r["flags"][-1])
                for r in sorted(
                    (
                        r
                        for r in records
                        if r["status"] == "flagged"
                        and any("possible name mismatch" in f for f in r["flags"])
                    ),
                    key=lambda r: (r["source"], r["ingredient"]),
                )
            ],
            ("Source", "Ingredient", "Duration (s)", "Flag"),
        ),
        "",
        "### Partial coverage (Merriam-Webster, word-level only)",
        "",
        "Flagged because the clip pronounces one word of a multi-word generic name,",
        "not the whole ingredient -- listed in full in the reconciliation section",
        "above.",
        "",
        *table(
            [
                (r["ingredient"], r["query"], r["duration_s"])
                for r in sorted(partial_flags, key=lambda r: (r["ingredient"], r["query"]))
            ],
            ("Ingredient", "Word covered", "Duration (s)"),
        ),
        "",
        "## Duplicate audio across ingredients",
        "",
        "Byte-identical clips shared by more than one ingredient -- either a genuine",
        "shared headword or a bug; made visible either way rather than silently",
        "inflating coverage.",
        "",
    ]
    if dupes:
        lines += table(
            [(h[:12], ", ".join(f"`{i}`" for i in ings)) for h, ings in dupes.items()],
            ("sha256", "Ingredients"),
        )
        lines += [
            "",
            "`dimethyl fumarate` and `formoterol fumarate dihydrate` share the word",
            "\"fumarate\" -- both resolved to the same Merriam-Webster audio file for that",
            "one shared word, which is the correct behavior, not a bug.",
        ]
    else:
        lines.append("None found.")

    lines += [
        "",
        "## What is still missing",
        "",
        f"{len(none)} ingredients ({pct(len(none), len(all_ingredients))}) have no audio",
        "from any source: "
        f"{sum(1 for i in none if types[i]=='generic')} generic, "
        f"{sum(1 for i in none if types[i]=='brand')} brand. The gap skews generic --",
        "coined INN names are exactly what neither a general dictionary, a consumer",
        "drug-information site, nor an older pharmacy-school teaching list (UMich's,",
        "which barely overlaps DOSE's newer names) reliably records. See",
        "`data/collected/HANDOFF_AUDIO_COLLECTION.md` for sources tried and the",
        "paid/licensed options (USP Dictionary of USAN, a citable MedlinePlus key,",
        "a Drugs.com data license) that would close the rest.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.write_text(build())
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
