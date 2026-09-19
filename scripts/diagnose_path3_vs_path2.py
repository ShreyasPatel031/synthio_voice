#!/usr/bin/env python
"""Diagnose why Path 3 disagrees with Path 2 (the project's validated ground
truth) and find which distance-metric variant best reproduces Path 2.

Path 2 (speech-similarity-v4) has been validated repeatedly and directly
against the project owner's ear across many items this session and is
treated as ground truth. Path 3 exists only to reach the items Path 2
cannot score at all (no human recording). So the right question is not
"which metric is theoretically nicer" but "which Path 3 variant best
reproduces Path 2's judgments on the 171 items where both can be
computed" -- and specifically, which one gets the known-bad items
(Adquey, Voranigo, Vyloy -- all confirmed horrible by ear) down at the
bottom where Path 2 puts them.

Recomputes variants OFFLINE from the decoded phonemes already stored in
Path 3's results metadata -- no audio decoding, no model loading beyond
panphon, so this is fast to iterate on.

Variants tested:
  weighted_rate     -- current: weighted_feature_edit_distance / ref_len
  weighted_raw      -- weighted_feature_edit_distance, unnormalized
  weighted_sqrt     -- weighted / sqrt(ref_len), a middle ground
  feature_rate/raw  -- panphon's unweighted feature edit distance
  leven_rate/raw    -- plain Levenshtein over IPA segments (classic PER)
  dolgo_rate/raw    -- coarse sound-class (Dolgopolsky) distance

The length-normalization question is the crux: Voranigo's raw weighted
distance (14.0) is almost identical to Adquey's (15.5), but Voranigo's
reference is 10 segments vs Adquey's 4, so dividing by length rescues
Voranigo's score (3.18) while Adquey stays at 0.0 -- even though BOTH are
confirmed horrible by ear. Normalizing by length is standard for phoneme
error rate, but it may be exactly the wrong choice for "is this drug name
recognizably right", where a fixed amount of mangling ruins a word
regardless of how long the word is.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dose_r.scoring.phoneme_distance import _get_panphon, normalize_phonemes  # noqa: E402

# Confirmed by the project owner listening to the actual audio this session.
EAR_VERDICTS = {
    "adquey": "BAD", "voranigo": "BAD", "vyloy": "BAD",
    "chantix": "GOOD",
    "aripiprazole": "OK", "acoramidis": "OK",
    "esomeprazole": "OK", "eliquis": "OK",
}


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def main() -> int:
    ft, dist = _get_panphon()

    p2 = {json.loads(l)["item_id"]: json.loads(l)
          for l in (REPO_ROOT / "runs/gemini-flash-tts-v1-speech-similarity-v4/results.jsonl").read_text().splitlines() if l.strip()}
    p3 = {json.loads(l)["item_id"]: json.loads(l)
          for l in (REPO_ROOT / "runs/gemini-flash-tts-v1-phoneme-distance-v1/results.jsonl").read_text().splitlines() if l.strip()}

    fns = {
        "weighted": dist.weighted_feature_edit_distance,
        "feature": dist.feature_edit_distance,
        "leven": dist.fast_levenshtein_distance,
        "dolgo": dist.dolgo_prime_distance,
    }
    variants: dict[str, dict[str, float]] = {}

    for item_id, r3 in p3.items():
        s3 = r3.get("score")
        if not s3 or not s3.get("scoreable"):
            continue
        meta = s3["metadata"]
        if meta.get("suffix_coverage_gap_caveat"):
            continue  # known data gap, excluded from this diagnosis
        decoded = normalize_phonemes(meta.get("decoded_phonemes", ""))
        if not decoded:
            continue
        # recompute against ALL variants, taking the best (min) per metric
        all_ipa = list(meta.get("all_variant_rates", {}).keys()) or [meta["best_matching_ipa_variant"]]
        row: dict[str, float] = {}
        for fname, fn in fns.items():
            best_raw, best_len = None, 1
            for v in all_ipa:
                rn = normalize_phonemes(v)
                d = fn(decoded, rn)
                n = len(ft.ipa_segs(rn)) or 1
                if best_raw is None or d / n < best_raw / best_len:
                    best_raw, best_len = d, n
            row[f"{fname}_raw"] = float(best_raw)
            row[f"{fname}_rate"] = best_raw / best_len
            row[f"{fname}_sqrt"] = best_raw / math.sqrt(best_len)
        variants[item_id] = row

    overlap = [i for i in variants if i in p2 and p2[i].get("score") and p2[i]["score"].get("scoreable")]
    print(f"items with BOTH Path 2 and Path 3 scores (suffix-gap excluded): {len(overlap)}\n")

    p2_scores = [p2[i]["score"]["score"] for i in overlap]
    print("Spearman correlation vs Path 2 (ground truth). Distances are")
    print("inverted (higher distance = worse), so MORE NEGATIVE = better match.\n")
    metric_names = sorted(next(iter(variants.values())).keys())
    ranked = []
    for m in metric_names:
        vals = [variants[i][m] for i in overlap]
        rho = spearman(vals, p2_scores)
        ranked.append((rho, m))
    for rho, m in sorted(ranked):
        print(f"  {m:18s} rho={rho:+.4f}")

    print("\n\nWhere the ear-verified items land under each variant")
    print("(percentile within the corpus: 0 = worst-scoring, 100 = best;")
    print("BAD items should be near 0, GOOD near 100):\n")
    hdr = f"{'metric':18s}" + "".join(f"{n[:9]:>11s}" for n in EAR_VERDICTS)
    print(hdr)
    print(f"{'(ear verdict)':18s}" + "".join(f"{v:>11s}" for v in EAR_VERDICTS.values()))
    print("-" * len(hdr))
    for m in metric_names:
        vals = sorted(variants[i][m] for i in overlap)
        cells = ""
        for name in EAR_VERDICTS:
            if name not in variants:
                cells += f"{'--':>11s}"
                continue
            v = variants[name][m]
            pct = 100.0 * (1.0 - sum(1 for x in vals if x < v) / len(vals))
            cells += f"{pct:>10.0f}%"
        print(f"{m:18s}{cells}")

    (REPO_ROOT / "runs" / "path3-metric-diagnosis.json").write_text(
        json.dumps({"variants": variants, "overlap": overlap}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
