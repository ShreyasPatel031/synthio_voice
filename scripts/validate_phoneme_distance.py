#!/usr/bin/env python
"""Validate Path 3 (phoneme_distance) BEFORE trusting it, mirroring the
discipline speech_similarity.py was held to: a human-ceiling check (does a
real human's OWN recording score close to a real pronunciation?) and a
discrimination check (does a clearly WRONG word score much worse?).

References.jsonl is read directly from the other branch (not yet merged
onto this one) rather than assumed stable -- see the module docstring in
dose_r/scoring/phoneme_distance.py for why this metric exists at all.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dose_r.references.reference_clips import available_clips  # noqa: E402
from dose_r.scoring.phoneme_distance import best_phoneme_distance  # noqa: E402
from dose_r.scoring.phoneme_model import transcribe_phonemes  # noqa: E402

REFERENCES_PATH = Path("/tmp/claude-0/-home-user-synthio-voice/78dc55a7-ba43-5659-8df2-f9bf7234f4f0/scratchpad/references.jsonl")


def main() -> int:
    refs = {json.loads(l)["ingredient"].lower(): json.loads(l)
            for l in REFERENCES_PATH.read_text().splitlines() if l.strip()}
    clips = available_clips()

    both = [(name, clip) for name, clip in clips.items()
            if name.lower() in refs and refs[name.lower()]["ipa_variants"]]
    print(f"items with human audio AND dictionary IPA: {len(both)}")

    print("\n=== HUMAN CEILING: decode each human clip, score vs its own dictionary IPA ===")
    rates = []
    worst = []
    for name, clip in both:
        try:
            decoded = transcribe_phonemes(clip.path)
            result = best_phoneme_distance(decoded, refs[name.lower()]["ipa_variants"])
            rates.append(result["rate"])
            worst.append((name, result["rate"], decoded, result["best_variant"]))
        except Exception as exc:
            print(f"  SKIP {name}: {exc}")

    rates.sort()
    n = len(rates)
    print(f"n={n}")
    print(f"mean rate={statistics.mean(rates):.4f}  median={statistics.median(rates):.4f}")
    print(f"p10={rates[int(n*0.1)]:.4f}  p90={rates[int(n*0.9)]:.4f}  max={rates[-1]:.4f}")

    worst.sort(key=lambda r: -r[1])
    print("\n=== worst 10 human-ceiling rates (highest distance for a REAL human recording) ===")
    for name, rate, decoded, variant in worst[:10]:
        print(f"  {name:20s} rate={rate:.3f}  decoded={decoded!r}  vs={variant!r}")

    print("\n=== DISCRIMINATION: correct pair vs deliberately mismatched pair ===")
    sample = [n for n, _ in both[:8]]
    mismatch_rates = []
    for i, name in enumerate(sample):
        wrong_name = sample[(i + 1) % len(sample)]
        clip = clips[name] if name in clips else clips[next(c for c in clips if c.lower() == name)]
        decoded = transcribe_phonemes(clip.path)
        result = best_phoneme_distance(decoded, refs[wrong_name.lower()]["ipa_variants"])
        mismatch_rates.append(result["rate"])
        print(f"  {name:16s} audio vs {wrong_name:16s} IPA -> rate={result['rate']:.3f}")

    print(f"\ncorrect-pair mean rate:   {statistics.mean(rates):.4f}")
    print(f"mismatched-pair mean rate: {statistics.mean(mismatch_rates):.4f}")
    print(f"margin: {statistics.mean(mismatch_rates) - statistics.mean(rates):.4f}")

    out = {"human_ceiling_rates": rates, "mismatch_rates": mismatch_rates,
           "worst_10": worst[:10]}
    (REPO_ROOT / "runs" / "phoneme-distance-validation-v1.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
