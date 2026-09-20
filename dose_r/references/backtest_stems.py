"""Back-test the USAN stem engine against real, sourced pronunciations.

There is no way to check an unsourced guess against ground truth -- by
definition it has none. What we CAN do is take every generic in
`references.jsonl` that already has a real external source (confidence
`high` or `medium`) and whose name happens to end in a stem this project
recognizes, throw away its real source, run only the stem engine and the
plain `g2p.py` fallback against it blind, and score both against the real
pronunciation with the judge's own phonetic-distance scorer.

This produces a genuine, measured error rate for the method on names it was
never shown the answer for. It is not proof that any individual UNSOURCED
guess elsewhere in the low tier is correct -- it is a defensible confidence
statement about the tier as a whole, and a per-stem number honest enough to
drop a stem that does not actually help.

Run: `python3 -m dose_r.references.backtest_stems`. Its numbers are mirrored
by hand into `usan_stems.BACKTEST_RESULTS` (and the notes `build.py` writes
into `references.jsonl`) after each run -- re-run this and update that
table together if a stem's rule changes.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..judge.fixtures import g2p
from ..judge.phonetic_scorer import score_against_reference
from ..judge.references import Reference, Variant
from . import usan_stems

REFERENCES = Path(__file__).resolve().parent / "references.jsonl"
REPORT = Path(__file__).resolve().parent / "STEM_BACKTEST.md"
SOURCED_TIERS = {"high", "medium"}

# `empagliflozin`'s sourced row is a scraping bug, not a real reference: its
# single Wikipedia hit resolved to the BRAND name "Jardiance"'s pronunciation
# page, not the generic. Scoring either engine against "Jardiance" measures
# nothing about `-gliflozin` and would silently corrupt the aggregate, so it
# is excluded here with this note rather than quietly included or fixed --
# fixing `build.py`'s source resolution is a separate task.
KNOWN_BAD_SOURCE = {
    "empagliflozin": "sourced reference is the brand name's (Jardiance) pronunciation, not the generic's",
}


def make_reference(row: dict) -> Reference:
    variants = tuple(
        Variant(i, tuple(arp.split()), ipa)
        for i, (arp, ipa) in enumerate(zip(row["arpabet_variants"], row["ipa_variants"]))
    )
    return Reference(
        ingredient=row["ingredient"],
        name_type=row["name_type"],
        variants=variants,
        confidence=row["confidence"],
    )


def load_sourced_generics() -> list[dict]:
    rows = [json.loads(line) for line in REFERENCES.open()]
    return [
        r
        for r in rows
        if r["name_type"] == "generic"
        and r["confidence"] in SOURCED_TIERS
        and r["ingredient"].lower() not in KNOWN_BAD_SOURCE
    ]


def backtest() -> dict:
    """{"per_stem": {suffix: [{ingredient, g2p_error, stem_error, improvement}]}}"""
    per_stem: dict[str, list[dict]] = {}
    for row in load_sourced_generics():
        name = row["ingredient"]
        stem = usan_stems.match_stem(name)
        if stem is None:
            continue

        reference = make_reference(row)
        g2p_hyp = g2p.to_arpabet_variants(name)[0]
        stem_hyp = usan_stems.to_arpabet_variants(name)[0]

        g2p_error = score_against_reference(g2p_hyp, reference).normalized_error
        stem_error = score_against_reference(stem_hyp, reference).normalized_error

        per_stem.setdefault(stem.suffix, []).append(
            {
                "ingredient": name,
                "g2p_error": g2p_error,
                "stem_error": stem_error,
                "improvement": g2p_error - stem_error,
            }
        )
    return per_stem


def summarize(per_stem: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for suffix, examples in sorted(per_stem.items()):
        n = len(examples)
        avg_g2p = sum(e["g2p_error"] for e in examples) / n
        avg_stem = sum(e["stem_error"] for e in examples) / n
        avg_improvement = sum(e["improvement"] for e in examples) / n
        rows.append(
            {
                "suffix": suffix,
                "n": n,
                "avg_g2p_error": avg_g2p,
                "avg_stem_error": avg_stem,
                "avg_improvement": avg_improvement,
                "examples": examples,
            }
        )
    return rows


def render_report(summary: list[dict]) -> str:
    tested_suffixes = {r["suffix"] for r in summary}
    untested = sorted(s for s, stem in usan_stems.STEMS.items() if s not in tested_suffixes)
    kept = [r for r in summary if r["avg_improvement"] > 0]
    dropped = [r for r in summary if r["avg_improvement"] <= 0]

    lines = [
        "# USAN Stem Engine -- Back-test Results",
        "",
        "Method: for every generic in `references.jsonl` with a real external",
        "source (confidence `high`/`medium`) whose name ends in a recognized",
        "USAN/INN stem, its real source is ignored and both the stem engine",
        "(`usan_stems.py`) and the plain generic fallback (`g2p.py`) are scored",
        "blind against the real, sourced pronunciation, using the judge's own",
        "phonetic-distance scorer (`dose_r/judge/distance.py` +",
        "`phonetic_scorer.py`) -- the identical PEU metric a TTS system is",
        "scored with. A positive improvement means the stem engine produced a",
        "lower (better) normalized PEU error than plain `g2p.py`.",
        "",
        f"`{KNOWN_BAD_SOURCE and list(KNOWN_BAD_SOURCE)[0]}` excluded: "
        f"{list(KNOWN_BAD_SOURCE.values())[0]}.",
        "",
        "## Per-stem results",
        "",
        "| Stem | n | avg g2p PEU | avg stem PEU | avg improvement | verdict |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in summary:
        verdict = "kept" if r["avg_improvement"] > 0 else "DROPPED -- no measured benefit"
        lines.append(
            f"| -{r['suffix']} | {r['n']} | {r['avg_g2p_error']:.4f} | "
            f"{r['avg_stem_error']:.4f} | {r['avg_improvement']:+.4f} | {verdict} |"
        )

    if kept:
        n_total = sum(r["n"] for r in kept)
        agg_g2p = sum(r["avg_g2p_error"] * r["n"] for r in kept) / n_total
        agg_stem = sum(r["avg_stem_error"] * r["n"] for r in kept) / n_total
        lines += [
            "",
            "## Aggregate, kept stems only",
            "",
            f"{n_total} sourced examples across {len(kept)} stems.",
            f"Plain `g2p.py` average normalized error: {agg_g2p:.4f} PEU.",
            f"Stem engine average normalized error: {agg_stem:.4f} PEU.",
            f"Average improvement: {agg_g2p - agg_stem:+.4f} PEU "
            f"({(agg_g2p - agg_stem) / agg_g2p:.0%} reduction).",
        ]

    lines += ["", "## Examples", ""]
    for r in summary:
        lines.append(f"### -{r['suffix']}")
        lines.append("")
        lines.append("| Ingredient | g2p PEU | stem PEU | improvement |")
        lines.append("| --- | --- | --- | --- |")
        for e in r["examples"]:
            lines.append(
                f"| {e['ingredient']} | {e['g2p_error']:.4f} | "
                f"{e['stem_error']:.4f} | {e['improvement']:+.4f} |"
            )
        lines.append("")

    if dropped:
        lines += ["## Dropped stems", ""]
        for r in dropped:
            lines.append(
                f"- `-{r['suffix']}`: average improvement {r['avg_improvement']:+.4f} PEU "
                f"over {r['n']} example(s) -- the stem engine did not measurably help "
                "(or measurably hurt), so it is not applied by `build.py` even though "
                "it stays documented in `usan_stems.py` for reference."
            )
        lines.append("")

    lines += [
        "## Untested stems",
        "",
        "No generic in this dataset ends in these stems with a real external",
        "source, so there is nothing to back-test them against. They stay in",
        "`usan_stems.py`, documented and cited, and are applied by `build.py`",
        "when they match -- but their confidence notes say plainly that they",
        "are unverified in this dataset, not silently treated the same as a",
        "back-tested stem.",
        "",
    ]
    for s in untested:
        lines.append(f"- `-{s}`: {usan_stems.STEMS[s].citation}")
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    per_stem = backtest()
    summary = summarize(per_stem)
    REPORT.write_text(render_report(summary))
    print(f"wrote {REPORT}")
    for r in summary:
        print(f"  -{r['suffix']}: n={r['n']} improvement={r['avg_improvement']:+.4f} PEU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
