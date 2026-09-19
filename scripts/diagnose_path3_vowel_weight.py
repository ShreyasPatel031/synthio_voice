#!/usr/bin/env python
"""Sweep vowel vs consonant edit costs against Path 2.

The CTC decoder's vowel confusions are large. If Path 2's ranking is mostly
carried by consonant identity, down-weighting vowels should raise correlation.
Path 2 stays the target.
"""
from __future__ import annotations

import json
import math
import sys
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dose_r.scoring.phoneme_distance import _get_panphon, normalize_phonemes  # noqa: E402

EAR = ["adquey", "voranigo", "vyloy", "chantix"]


def spearman(xs, ys):
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


def is_vowel(seg: str, ft) -> bool:
    # panphon: a segment is a vowel if syl is '+'. Multi-char tokens: vowel
    # if every panphon piece is a vowel, or the first one is.
    parts = ft.ipa_segs(seg) or [seg]
    vecs = []
    for p in parts:
        v = ft.segment_to_vector(p)
        if not v:
            return False
        vecs.append(v)
    names = ft.names
    if "syl" not in names:
        return False
    idx = names.index("syl")
    return all(v[idx] == "+" for v in vecs)


def levenshtein(a, b, sub, indel_a, indel_b):
    n, m = len(a), len(b)
    dp = [0.0] * (m + 1)
    for j in range(1, m + 1):
        dp[j] = dp[j - 1] + indel_b(b[j - 1])
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], dp[0] + indel_a(a[i - 1])
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = min(
                dp[j] + indel_a(a[i - 1]),
                dp[j - 1] + indel_b(b[j - 1]),
                prev + sub(a[i - 1], b[j - 1]),
            )
            prev = cur
    return dp[m]


def cons_only(tokens, vowel):
    return [t for t in tokens if not vowel(t)]


def main() -> int:
    ft, dist = _get_panphon()

    @lru_cache(maxsize=4096)
    def vowel(seg: str) -> bool:
        return is_vowel(seg, ft)

    p2 = {json.loads(l)["item_id"]: json.loads(l)
          for l in (REPO_ROOT / "runs/gemini-flash-tts-v1-speech-similarity-v4/results.jsonl").read_text().splitlines() if l.strip()}
    p3 = {json.loads(l)["item_id"]: json.loads(l)
          for l in (REPO_ROOT / "runs/gemini-flash-tts-v1-phoneme-distance-v1/results.jsonl").read_text().splitlines() if l.strip()}

    # panphon segments, both sides
    data = {}
    for item_id, r3 in p3.items():
        s3 = r3.get("score")
        if not s3 or not s3.get("scoreable") or s3["metadata"].get("suffix_coverage_gap_caveat"):
            continue
        if not p2.get(item_id, {}).get("score", {}).get("scoreable"):
            continue
        decoded = normalize_phonemes(s3["metadata"].get("decoded_phonemes") or "")
        if not decoded:
            continue
        variants = list(s3["metadata"].get("all_variant_rates", {}).keys()) or [s3["metadata"]["best_matching_ipa_variant"]]
        hyps = ft.ipa_segs(decoded)
        refs = [ft.ipa_segs(normalize_phonemes(v)) for v in variants]
        data[item_id] = (hyps, refs, p2[item_id]["score"]["score"], p3[item_id]["score"]["score"])

    ids = list(data)
    p2s = [data[i][2] for i in ids]
    print(f"n={len(ids)} path3 rho={spearman([data[i][3] for i in ids], p2s):+.4f}")

    def best_rate(item_id, cost_fn):
        hyps, refs, _, _ = data[item_id]
        best = None
        for ref in refs:
            if not ref:
                continue
            d = cost_fn(hyps, ref)
            rate = d / len(ref)
            if best is None or rate < best:
                best = rate
        return best if best is not None else 0.0

    weights = [0.0, 0.15, 0.25, 0.5, 1.0]
    for w in weights:
        def sub(x, y, w=w):
            if x == y:
                return 0.0
            # vowel-vowel substitutions are cheap; anything involving a
            # consonant is full cost
            if vowel(x) and vowel(y):
                return w
            return 1.0
        def indel(t, w=w):
            return w if vowel(t) else 1.0
        rates = [best_rate(i, lambda h, r, sub=sub, indel=indel: levenshtein(h, r, sub, indel, indel)) for i in ids]
        rho = spearman(rates, p2s)
        print(f"vowel_cost={w:.2f} rho={rho:+.4f}")
        for name in EAR:
            vals = sorted(rates)
            v = rates[ids.index(name)]
            pct = 100.0 * sum(1 for x in vals if x < v) / len(vals)
            print(f"    {name:12s} rate={v:.3f} worst-pct={pct:.0f}")

    # consonant sequence only
    rates = []
    for i in ids:
        hyps, refs, _, _ = data[i]
        h = cons_only(hyps, vowel)
        best = None
        for ref in refs:
            r = cons_only(ref, vowel)
            if not r:
                continue
            d = levenshtein(h, r, lambda x, y: 0.0 if x == y else 1.0, lambda t: 1.0, lambda t: 1.0)
            rate = d / len(r)
            if best is None or rate < best:
                best = rate
        rates.append(best or 0.0)
    print(f"consonants-only rho={spearman(rates, p2s):+.4f}")
    for name in EAR:
        v = rates[ids.index(name)]
        pct = 100.0 * sum(1 for x in rates if x < v) / len(rates)
        print(f"    {name:12s} rate={v:.3f} worst-pct={pct:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
