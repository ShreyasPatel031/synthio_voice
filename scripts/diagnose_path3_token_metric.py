#!/usr/bin/env python
"""Retest Path 3 distance using the CTC model's own tokens, not panphon's.

Path 2 is ground truth. Prior metric sweeps all segmented with panphon,
which splits diphthongs (oʊ -> o + ʊ) and affricates (tʃ -> t + ʃ) that
facebook/wav2vec2-lv-60-espeak-cv-ft emits as single vocabulary items.
A wrong diphthong then costs half an edit. This script segments dictionary
IPA with longest-match against that vocabulary and edits at token level.
"""
from __future__ import annotations

import json
import math
import os
import sys
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/workspace/.cache/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/.cache/huggingface")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dose_r.scoring.phoneme_distance import _get_panphon, normalize_phonemes  # noqa: E402
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


@lru_cache(maxsize=1)
def vocab() -> set[str]:
    from transformers import Wav2Vec2Processor
    proc = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    return set(proc.tokenizer.get_vocab())


def longest_match(ipa: str, symbols: set[str]) -> list[str]:
    i, out = 0, []
    while i < len(ipa):
        match = None
        for j in range(len(ipa), i, -1):
            if ipa[i:j] in symbols:
                match = ipa[i:j]
                break
        if match is None:
            out.append(ipa[i])
            i += 1
        else:
            out.append(match)
            i += len(match)
    return out


def expand_rhotic(tokens: list[str], symbols: set[str]) -> list[str]:
    """espeak packs some r-colored vowels as one symbol (ɔːɹ). Split them
    so they line up with dictionary IPA, which writes the vowel and ɹ apart.
    """
    out = []
    for t in tokens:
        if len(t) > 1 and t.endswith("ɹ") and t[:-1] in symbols:
            out.extend([t[:-1], "ɹ"])
        else:
            out.append(t)
    return out


def levenshtein(a: list[str], b: list[str], sub_cost) -> float:
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            ins = dp[j] + 1.0
            delete = dp[j - 1] + 1.0
            sub = prev + sub_cost(a[i - 1], b[j - 1])
            prev = cur
            dp[j] = min(ins, delete, sub)
    return dp[m]


def main() -> int:
    symbols = vocab()
    ft, dist = _get_panphon()

    @lru_cache(maxsize=20000)
    def feat_cost(x: str, y: str) -> float:
        if x == y:
            return 0.0
        # panphon cost of the two tokens as whole strings, scaled so a
        # totally unrelated substitution is about 1 (its max per-phone
        # weighted cost is ~8-ish; cap so one token can't dwarf an indel).
        raw = dist.weighted_feature_edit_distance(x, y)
        return min(1.0, raw / 4.0)

    def unit_cost(x, y):
        return 0.0 if x == y else 1.0

    p2 = {json.loads(l)["item_id"]: json.loads(l)
          for l in (REPO_ROOT / "runs/gemini-flash-tts-v1-speech-similarity-v4/results.jsonl").read_text().splitlines() if l.strip()}
    p3 = {json.loads(l)["item_id"]: json.loads(l)
          for l in (REPO_ROOT / "runs/gemini-flash-tts-v1-phoneme-distance-v1/results.jsonl").read_text().splitlines() if l.strip()}

    rows = {}
    unk = 0
    for item_id, r3 in p3.items():
        s3 = r3.get("score")
        if not s3 or not s3.get("scoreable"):
            continue
        meta = s3["metadata"]
        if meta.get("suffix_coverage_gap_caveat"):
            continue
        decoded = meta.get("decoded_phonemes") or ""
        if not decoded.strip():
            continue
        hyp = [t for t in decoded.split() if t]
        hyp_x = expand_rhotic(hyp, symbols)
        variants = list(meta.get("all_variant_rates", {}).keys()) or [meta["best_matching_ipa_variant"]]
        best = {}
        for v in variants:
            ref_raw = longest_match(normalize_phonemes(v), symbols)
            if any(t not in symbols for t in ref_raw):
                unk += 1
            ref = [t for t in ref_raw if t in symbols] or ref_raw
            ref_x = expand_rhotic(ref, symbols)
            cands = {
                "tok_leven_rate": levenshtein(hyp, ref, unit_cost) / max(1, len(ref)),
                "tok_leven_raw": levenshtein(hyp, ref, unit_cost),
                "tok_feat_rate": levenshtein(hyp, ref, feat_cost) / max(1, len(ref)),
                "x_leven_rate": levenshtein(hyp_x, ref_x, unit_cost) / max(1, len(ref_x)),
                "x_feat_rate": levenshtein(hyp_x, ref_x, feat_cost) / max(1, len(ref_x)),
                "x_leven_raw": levenshtein(hyp_x, ref_x, unit_cost),
            }
            if not best:
                best = cands
                best["_ref"] = ref
                best["_ref_x"] = ref_x
            else:
                for k, val in cands.items():
                    if val < best[k]:
                        best[k] = val
                        if k == "x_leven_rate":
                            best["_ref_x"] = ref_x
                        if k == "tok_leven_rate":
                            best["_ref"] = ref
        best["_hyp"] = hyp
        best["_hyp_x"] = hyp_x
        rows[item_id] = best

    overlap = [i for i in rows if i in p2 and p2[i].get("score") and p2[i]["score"].get("scoreable")]
    p2s = [p2[i]["score"]["score"] for i in overlap]
    print(f"overlap {len(overlap)} unknown-symbol segmentations touched {unk}")
    keys = ["tok_leven_rate", "tok_leven_raw", "tok_feat_rate", "x_leven_rate", "x_feat_rate", "x_leven_raw"]
    print("\nSpearman vs Path 2 (more negative is better):")
    for k in keys:
        rho = spearman([rows[i][k] for i in overlap], p2s)
        print(f"  {k:18s} rho={rho:+.4f}")
    # current path 3 score (higher=better) so flip the comparison note
    rho_cur = spearman([p3[i]["score"]["score"] for i in overlap], p2s)
    print(f"  {'path3_current':18s} rho={rho_cur:+.4f}  (higher=better, so positive is good)")

    print("\nkey items (percentile: for distances 0=worst; for path2/path3 100=best)")
    for name in EAR:
        if name not in rows:
            print(name, "missing")
            continue
        r = rows[name]
        print(f"\n{name}  p2={p2[name]['score']['score']:.3f} p3={p3[name]['score']['score']:.3f}")
        print(f"  hyp   {r['_hyp']}")
        print(f"  hyp_x {r['_hyp_x']}")
        print(f"  ref   {r['_ref']}")
        print(f"  ref_x {r['_ref_x']}")
        for k in keys:
            vals = sorted(rows[i][k] for i in overlap)
            v = r[k]
            pct_worst = 100.0 * sum(1 for x in vals if x < v) / len(vals)
            print(f"  {k:18s} {v:.3f}  worst-pct {pct_worst:.0f}")

    out = {i: {k: rows[i][k] for k in keys} for i in overlap}
    (REPO_ROOT / "runs" / "path3-token-metric-diagnosis.json").write_text(json.dumps(out))
    print("\nwrote runs/path3-token-metric-diagnosis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
