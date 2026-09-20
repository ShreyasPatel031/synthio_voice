"""Generate a development-only stand-in for the Workstream 1a reference layer.

Run:  python -m dose_r.judge.fixtures.build_synthetic

Writes `dose_r/judge/fixtures/synthetic_references.jsonl`, one record per unique
ingredient in the DOSE test set, in the exact schema Workstream 1a is
committed to. Every record is `"confidence": "low"` and carries a note saying it
is machine-guessed, so that a synthetic file cannot be mistaken for gold if it
is ever pointed at by accident.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..dataset import unique_ingredients
from .g2p import to_arpabet_variants, to_ipa

OUT_PATH = Path(__file__).resolve().parent / "synthetic_references.jsonl"
NOTE = (
    "SYNTHETIC. Machine-guessed by dose_r.judge.fixtures.g2p for development "
    "only. Not a pronunciation authority; do not report scores against this."
)


def build(names: list[str]) -> list[dict]:
    records = []
    for name in names:
        arpabet = to_arpabet_variants(name)
        records.append(
            {
                "ingredient": name,
                "name_type": "generic",
                "ipa_variants": [to_ipa(v) for v in arpabet],
                "arpabet_variants": [" ".join(v) for v in arpabet],
                "sources": [{"name": "synthetic-g2p", "raw": name, "url": ""}],
                "confidence": "low",
                "notes": NOTE,
            }
        )
    return records


def main() -> int:
    names = unique_ingredients()
    records = build(names)
    with OUT_PATH.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    variants = sum(len(r["arpabet_variants"]) for r in records)
    print(f"wrote {len(records)} synthetic references ({variants} variants) -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
