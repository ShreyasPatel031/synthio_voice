#!/usr/bin/env python
"""Test alignment variants that ignore span-bleed insertions.

Path 2's frame matching can ignore a little extra audio on either side of
the drug. Path 3's full-string edit distance cannot. These variants are
computed from stored decodes; Path 2 remains the regression target.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/workspace/.cache/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/.cache/huggingface")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dose_r.scoring.phoneme_distance import normalize_phonemes  # noqa: E402
from dose_r.scoring.phoneme_model import MODEL_ID  # noqa: E402

EAR = ["adquey", "voranigo", "vyloy", "chantix", "aripiprazole",
       "acoramidis", "esomeprazole", "eliquis", "talquetamab"]


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


def vocab():
    from transformers import Wav2Vec2Processor
    return set(Wav2Vec2Processor.from_pretrained(MODEL_ID).tokenizer.get_vocab())


def longest_match(ipa: str, symbols: set[str]) -> list[str]:
    i, out = 0, []
    while i < len(ipa):
        match = None
        for j in range(len(ipa), i, -1):
            if ipa[i:j] in symbols:
                match = ipa[i:j]
                break
        if match is None:
            i += 1
        else:
            out.append(match)
            i += len(match)
    return [t for t in out if t in symbols]


def expand_rhotic(tokens, symbols):
    out = []
    for t in tokens:
        if len(t) > 1 and t.endswith("ɹ") and t[:-1] in symbols:
            out.extend([t[:-1], "ɹ"])
        else:
            out.append(t)
    return out


def lcs(a, b):
    n, m = len(a), len(b)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = cur
    return dp[m]


def semi_global(hyp, ref):
    """Edit distance, free insertions at the ends of hyp only.
    Returns (distance, matched window)."""
    n, m = len(hyp), len(ref)
    INF = 10 ** 9
    # dp[i][j] = cost to align hyp[:i] to ref[:j], free to skip hyp prefix
    prev = [INF] * (m + 1)
    prev[0] = 0
    for i in range(1, n + 1):
        cur = [0] * (m + 1)  # j=0: skipped this hyp token (leading or we'll min later)
        cur[0] = 0
        for j in range(1, m + 1):
            sub = prev[j - 1] + (0 if hyp[i - 1] == ref[j - 1] else 1)
            delete_ref = cur[j - 1] + 1  # missing a reference phone
            insert_hyp = prev[j] + 1     # extra hyp phone in the middle
            cur[j] = min(sub, delete_ref, insert_hyp)
        prev = cur
    # free trailing hyp insertions: min over all i is already in the last row
    # because insert cost is paid... trailing free means we should NOT pay for
    # unmatched hyp suffix. Recompute with free suffix by taking min along the
    # last column? Standard semi-global: init first row/col 0 and read min of
    # last row. Let me redo properly.
    return None


def semi_global_dist(hyp, ref) -> float:
    """Distance of ref as a subsequence-window inside hyp.
    Leading and trailing hyp tokens are free. Internal insertions cost 1.
    Substitutions and deletions cost 1.
    """
    n, m = len(hyp), len(ref)
    if m == 0:
        return 0.0
    INF = 1e9
    # dp[j] after consuming some hyp prefix, aligned to ref[:j]
    # start: no ref consumed, cost 0, and we can skip hyp freely before start
    dp = [INF] * (m + 1)
    dp[0] = 0.0
    for i in range(n):
        ndp = [INF] * (m + 1)
        # skipping this hyp token: free if we haven't started (dp[0]), else insertion
        # We track "started" implicitly: once j>0, skips cost 1, and after the
        # ref is complete, skips are free. Do three regions.
        ndp[0] = 0.0  # still in the free prefix (never started)
        for j in range(1, m + 1):
            # substitution / match, consuming hyp[i] and ref[j-1]
            sub = dp[j - 1] + (0.0 if hyp[i] == ref[j - 1] else 1.0)
            # insertion of hyp[i] while inside the alignment
            ins = dp[j] + 1.0
            # deletion of ref phone, not consuming hyp (stay on same hyp by
            # applying after the loop). Handle deletions separately.
            ndp[j] = min(sub, ins)
        # deletions: a reference phone with no hyp token
        for j in range(1, m + 1):
            ndp[j] = min(ndp[j], ndp[j - 1] + 1.0)
        # once ref is complete, further hyp tokens are free
        ndp[m] = min(ndp[m], dp[m])
        dp = ndp
    return dp[m]


def main() -> int:
    symbols = vocab()
    p2 = {json.loads(l)["item_id"]: json.loads(l)
          for l in (REPO_ROOT / "runs/gemini-flash-tts-v1-speech-similarity-v4/results.jsonl").read_text().splitlines() if l.strip()}
    p3 = {json.loads(l)["item_id"]: json.loads(l)
          for l in (REPO_ROOT / "runs/gemini-flash-tts-v1-phoneme-distance-v1/results.jsonl").read_text().splitlines() if l.strip()}
    rows = {}
    for item_id, r3 in p3.items():
        s3 = r3.get("score")
        if not s3 or not s3.get("scoreable") or s3["metadata"].get("suffix_coverage_gap_caveat"):
            continue
        decoded = s3["metadata"].get("decoded_phonemes") or ""
        if not decoded.strip():
            continue
        hyp = expand_rhotic(decoded.split(), symbols)
        variants = list(s3["metadata"].get("all_variant_rates", {}).keys()) or [s3["metadata"]["best_matching_ipa_variant"]]
        best = None
        for v in variants:
            ref = expand_rhotic(longest_match(normalize_phonemes(v), symbols), symbols)
            if not ref:
                continue
            rec = {
                "lcs_recall": lcs(hyp, ref) / len(ref),
                "semi": semi_global_dist(hyp, ref) / len(ref),
                "semi_raw": semi_global_dist(hyp, ref),
                "len_ratio": len(hyp) / len(ref),
            }
            if best is None or rec["semi"] < best["semi"]:
                best = rec
                best["hyp"] = hyp
                best["ref"] = ref
        if best:
            rows[item_id] = best

    overlap = [i for i in rows if p2.get(i, {}).get("score", {}).get("scoreable")]
    p2s = [p2[i]["score"]["score"] for i in overlap]
    print(f"overlap {len(overlap)}")
    for k, sign in (("lcs_recall", +1), ("semi", -1), ("semi_raw", -1), ("len_ratio", 0)):
        rho = spearman([rows[i][k] for i in overlap], p2s)
        print(f"  {k:12s} rho={rho:+.4f}")
    cur = spearman([p3[i]["score"]["score"] for i in overlap], p2s)
    print(f"  {'path3':12s} rho={cur:+.4f}")

    print("\nkey items")
    for name in EAR:
        r = rows.get(name)
        if not r:
            continue
        print(f"\n{name} p2={p2[name]['score']['score']:.3f} p3={p3[name]['score']['score']:.3f}")
        print(f"  hyp {r['hyp']}")
        print(f"  ref {r['ref']}")
        for k in ("lcs_recall", "semi", "semi_raw", "len_ratio"):
            vals = [rows[i][k] for i in overlap]
            v = r[k]
            # for distances, low is bad; for recall, low is bad
            if k == "lcs_recall":
                pct = 100.0 * sum(1 for x in vals if x < v) / len(vals)
                print(f"  {k:12s} {v:.3f}  best-pct {pct:.0f}")
            else:
                pct = 100.0 * sum(1 for x in vals if x < v) / len(vals)
                print(f"  {k:12s} {v:.3f}  worst-pct {pct:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
