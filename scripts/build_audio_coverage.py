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
import re
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
    "clincalc": "ClinCalc (full name)",
    "nci": "NCI Dictionary of Cancer Terms",
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

    mw_full = {r["ingredient"] for r in records if r["source"] == "merriam-webster" and r["coverage"] == "full"}
    mw_component = {r["ingredient"] for r in records if r["source"] == "merriam-webster" and r["coverage"] == "component"}
    mw_any = mw_full | mw_component

    cc_full = {r["ingredient"] for r in records if r["source"] == "clincalc" and r["coverage"] == "full"}
    cc_component = {r["ingredient"] for r in records if r["source"] == "clincalc" and r["coverage"] == "component"}
    cc_any = cc_full | cc_component
    cc_rows = [r for r in records if r["source"] == "clincalc"]

    cc_urls = {r["source_url"] for r in cc_rows}
    cc_paired = {
        r["ingredient"]
        for r in cc_rows
        if r["respelling"] and (m := re.search(r"\((https://[^)]+)\)$", r["respelling"]))
        and m.group(1) in cc_urls
    }

    full_coverage = {
        "drugs.com": {r["ingredient"] for r in records if r["source"] == "drugs.com"},
        "merriam-webster": mw_full,
        "umich": {r["ingredient"] for r in records if r["source"] == "umich"},
        "clincalc": cc_full,
        "nci": {r["ingredient"] for r in records if r["source"] == "nci"},
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
    n_sources = len(source_labels)
    lines = [
        "# Reference Audio -- Coverage",
        "",
        "Human-recorded pronunciation audio for the 284 unique ingredients across",
        f"the 274 DOSE rows, from {n_sources} sources: {sources_prose}.",
        "Merriam-Webster's Medical API; Drugs.com collected by hand -- see",
        "`data/collected/HANDOFF_AUDIO_COLLECTION.md`, drugs.com 403s this environment",
        "on every request; UMich's student pronunciation page via the Wayback Machine,",
        "since the live site 403s behind a Cloudflare challenge; ClinCalc's Top 250",
        "Drugs pronunciation pages, fetched live -- the only source that records a",
        "generic name and a brand name as two separate clips instead of one page's",
        "one recording; the NCI Dictionary of Cancer Terms's own backing JSON API",
        "(`webapis.cancer.gov/glossary/v1/`), a real hosted government recording per",
        "term, fetched live. \"Coverage\" below means a clip that pronounces the",
        "*whole* ingredient name, since that is what DOSE scores; Merriam-Webster's",
        "and ClinCalc's word-/name-level partial clips are reported separately.",
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
                (f"All {n_sources} sources", len(all_sources), pct(len(all_sources), len(all_ingredients))),
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
        "## ClinCalc: reconciling row counts",
        "",
        f"The manifest carries {len(cc_rows)} ClinCalc rows for {len(cc_any)} unique",
        f"ingredients. Of those, {len(cc_full)} carry a clip that is the *only* name",
        "in its audio block (`coverage: full`) and "
        f"{len(cc_component)} share a block's one",
        "recording with one or more other names -- a combination product's several",
        "generic components, or several brands sold for one generic, e.g. ClinCalc's",
        "one `ibuprofen` brand clip says \"Advil, Motrin\" together, not either alone",
        "(`coverage: component`, flagged as partial, listed below).",
        "",
        f"{len(cc_paired)} of the {len(cc_any)} carry a `respelling` note pointing to",
        "a second ClinCalc clip on the same page for the paired generic or brand name,",
        "where that paired name is itself another DOSE ingredient with its own row --",
        "e.g. `atorvastatin`'s row notes ClinCalc's separate `Lipitor` clip, and",
        "`Lipitor`'s row notes the `atorvastatin` one back. Every ClinCalc row carries",
        "such a note when the page has both a generic and a brand block, whether or",
        "not the other name happens to be its own DOSE ingredient -- see the",
        "manifest's `respelling` field for the rest.",
        "",
        "Component-only ClinCalc ingredients (partial, not full-name, audio):",
        "",
        *[
            f"- `{ing}` -- shares its clip with "
            + ", ".join(
                sorted(
                    {
                        n
                        for r in cc_rows
                        if r["ingredient"] == ing and r["coverage"] == "component"
                        for n in r["headword"].split("; ")
                        if n.lower() != ing.lower()
                    }
                )
            )
            for ing in sorted(cc_component)
        ],
        "",
        "## ClinCalc cross-check of previously flagged clips",
        "",
        "ClinCalc is the first source that records a brand and a generic name as two",
        "separate clips, so where it covers a flagged ingredient its own clip (or a",
        "same-source sibling, like a clean single-word Merriam-Webster clip) gives an",
        "unambiguous duration to compare the flagged clip's duration against, instead",
        "of only a within-batch syllable estimate.",
        "",
        "- `Advair` (Drugs.com, 1.666s, flagged) -- ClinCalc's dedicated single-name",
        "  `Advair` brand clip runs 2.214s and Merriam-Webster's unflagged `Advair`",
        "  clip runs 1.857s. Drugs.com's duration sits at or below both independent",
        "  clean-word recordings, not anywhere near ClinCalc's own combined",
        "  `Fluticasone; salmeterol` clip (2.893s) a mispronunciation as the generic",
        "  would have to resemble. **Resolved**: the flag was a within-Drugs.com-batch",
        "  artifact; the clip's duration is consistent with genuinely saying \"Advair\".",
        "- `Motrin` (Drugs.com, 1.625s, flagged) -- ClinCalc never recorded `Motrin`",
        "  alone (its ibuprofen page's one brand clip says \"Advil, Motrin\" together,",
        "  2.736s), but Merriam-Webster's unflagged, unambiguous single-word `Motrin`",
        "  clip runs only 0.605s. Drugs.com's `Motrin` (1.625s) is also 2.2x its own",
        "  `Advil` clip (0.734s) despite both being two-syllable brand names recorded",
        "  in the same batch. **Confirmed suspicious**: nothing here contradicts the",
        "  original flag, and the size of the gap from Merriam-Webster's clean word",
        "  makes a wrong-name clip (most likely the generic, \"ibuprofen\") more likely",
        "  than a slow reading of \"Motrin\" alone.",
        "- `fluticasone propionate` (Drugs.com, 1.278s, flagged) -- ClinCalc's clean,",
        "  unflagged single-name generic clip, headed \"Fluticasone (inhaled)\", runs",
        "  1.149s -- 11% off Drugs.com's duration for what both would then be the same",
        "  bare word. **Resolved**: consistent with Drugs.com's clip pronouncing only",
        "  the base name \"fluticasone\" and omitting the \"propionate\" salt, not with a",
        "  different drug; the flag's syllable estimate over-counted using the full",
        "  ingredient name's syllables against a clip that likely never spoke them all.",
        "- `formoterol fumarate dihydrate` (Drugs.com, 1.026s, flagged) -- ClinCalc's",
        "  clean standalone `Formoterol` clip runs 1.848s and Merriam-Webster's clean",
        "  `formoterol` word-clip runs 1.300s; Drugs.com's 1.026s is the *shortest* of",
        "  the three bare-\"formoterol\" measurements, not the longest a wrong, longer",
        "  name would produce. **Resolved**: same reading as `fluticasone propionate`",
        "  -- a clip of the base generic name only, not a name mismatch.",
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
            "one shared word, which is the correct behavior, not a bug. Likewise every",
            "ClinCalc pair here (`Advil`/`Motrin`, `Metformin`/`sitagliptin`, and the rest)",
            "shares one page's one `coverage: component` clip that names both -- also",
            "correct, not a bug; see the ClinCalc reconciliation section above.",
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
        "drug-information site, an older pharmacy-school teaching list (UMich's,",
        "which barely overlaps DOSE's newer names), a commonly-prescribed-drugs",
        "pronunciation page (ClinCalc's, which skews the same way), nor a cancer-",
        "specific dictionary (NCI's, whose real gain was cross-checking names other",
        "sources already had, not covering brand-new non-oncology names) reliably",
        "records. See",
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
