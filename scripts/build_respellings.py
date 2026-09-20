"""Build homogenized dictionary respellings from references.jsonl.

Offline: no fetches. Reads the existing citation snapshot, picks one source
per ingredient (USAN/AMA > DailyMed > NCI > MW > Gemini), writes a single
hyphenated CAPS respelling.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dose_r.references.respelling import homogenize_record, is_canonical

DEFAULT_IN = ROOT / "dose_r" / "references" / "references.jsonl"
DEFAULT_OUT = ROOT / "dose_r" / "references" / "respellings.jsonl"
DEFAULT_MD = ROOT / "dose_r" / "references" / "RESPELLINGS.md"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def coverage_markdown(rows: list[dict]) -> str:
    n = len(rows)
    sourced = [r for r in rows if r.get("respelling")]
    unsourced = [r for r in rows if not r.get("respelling")]
    inferred = [r for r in sourced if r.get("stress_inferred")]
    by_source = Counter(r["source"] for r in sourced)
    dropped = Counter(
        name for r in rows for name in r.get("dropped_sources") or []
    )
    lines = [
        "# Dictionary respellings — homogenized",
        "",
        "One ASCII hyphenated respelling per ingredient. One source.",
        "USAN/AMA first, then DailyMed, NCI, Merriam-Webster medical, Gemini.",
        "Wikipedia IPA, Wiktionary IPA, and CMUdict are never selected.",
        "",
        f"Canonical form: `toe-fa-SYE-ti-nib` (unstressed lowercase, exactly",
        f"one ALL-CAPS primary per word).",
        "",
        "## Coverage",
        "",
        f"| | Count | Share of {n} |",
        f"| --- | ---: | ---: |",
        f"| Canonical dictionary respelling | {len(sourced)} | {100 * len(sourced) / n:.1f}% |",
        f"| No source (cannot invent) | {len(unsourced)} | {100 * len(unsourced) / n:.1f}% |",
        f"| Stress inferred (source unmarked) | {len(inferred)} | {100 * len(inferred) / n:.1f}% |",
        "",
        "## Primary source (exactly one per sourced name)",
        "",
        "| Source | Ingredients |",
        "| --- | ---: |",
    ]
    for name, count in by_source.most_common():
        lines.append(f"| {name} | {count} |")
    lines += [
        "",
        "## Dropped (not used as the primary)",
        "",
        "| Source | Citations dropped |",
        "| --- | ---: |",
    ]
    for name, count in dropped.most_common():
        lines.append(f"| {name} | {count} |")
    if unsourced:
        lines += ["", "## Unsourced (left empty on purpose)", ""]
        for r in unsourced:
            lines.append(f"- {r['ingredient']}")
    if inferred:
        lines += [
            "",
            "## Stress inferred from first syllable",
            "",
            "Source wrote no primary-stress mark. First syllable capitalized.",
            "",
        ]
        for r in inferred:
            lines.append(
                f"- `{r['ingredient']}` ← `{r['source_raw']}` → `{r['respelling']}`"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = ap.parse_args()

    records = load_jsonl(args.src)
    rows = [homogenize_record(r) for r in records]
    bad = [
        r for r in rows
        if r.get("respelling") and not is_canonical(r["respelling"])
    ]
    if bad:
        raise SystemExit(
            "non-canonical: "
            + ", ".join(f"{r['ingredient']}={r['respelling']!r}" for r in bad[:20])
        )
    write_jsonl(args.out, rows)
    args.md.write_text(coverage_markdown(rows))
    sourced = sum(1 for r in rows if r.get("respelling"))
    print(f"wrote {args.out} ({sourced}/{len(rows)} sourced)")
    print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
