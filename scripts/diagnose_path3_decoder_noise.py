#!/usr/bin/env python
"""Decisive test of WHY Path 3 tracks Path 2 only weakly (rho~-0.50).

Hypothesis: the bottleneck is the CTC phoneme decoder's own noise, not the
distance metric. Evidence so far:
  - Every metric variant (weighted/feature/levenshtein/dolgo, raw/rate/sqrt)
    lands at rho -0.45..-0.50 vs Path 2 -- the choice barely matters.
  - Decoding a HUMAN recording (a correct pronunciation by definition) and
    scoring it against its own dictionary IPA already gives mean rate 0.88
    (scripts/validate_phoneme_distance.py). Gemini's corpus mean rate is
    ~0.99. So the noise floor is ~0.88 and the entire Gemini error signal
    on top of it is ~0.11 -- terrible signal-to-noise.
  - The decoder visibly garbles things: "ø" (not an English phoneme) for
    talquetamab, a spurious hard "ɡ" in Adquey, a doubled "ɹɹ" in Voranigo,
    "s ai5 ts. əɜ" for Zaiidra.

The test: compare Gemini's decoded phonemes against the HUMAN recording's
decoded phonemes (same decoder on both sides) instead of against the
written dictionary IPA (different "channel"). If the decoder's systematic
biases are the problem, they should partially CANCEL when both sides pass
through the same decoder, and correlation with Path 2 should jump.

This is a diagnostic, not a proposed scorer -- comparing against a human
recording obviously defeats Path 3's whole purpose (covering items with NO
human recording). But it isolates the cause: if same-channel comparison
tracks Path 2 much better, the problem is decoder-vs-written-IPA channel
mismatch and/or decoder noise, and the fix is a better recognizer or a
noise model -- not a different edit-distance formula.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dose_r.references.reference_clips import available_clips  # noqa: E402
from dose_r.scoring.phoneme_distance import _get_panphon, normalize_phonemes  # noqa: E402
from dose_r.scoring.phoneme_model import transcribe_phonemes  # noqa: E402

CACHE = REPO_ROOT / "runs" / "human-ref-decoded-phonemes.json"


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


def main() -> int:
    ft, dist = _get_panphon()
    clips = available_clips()

    if CACHE.exists():
        human_decoded = json.loads(CACHE.read_text())
        print(f"loaded {len(human_decoded)} cached human decodes")
    else:
        human_decoded = {}
        for i, (name, clip) in enumerate(clips.items(), 1):
            try:
                human_decoded[name.lower()] = transcribe_phonemes(clip.path)
            except Exception as exc:
                print(f"  SKIP {name}: {exc}")
            if i % 40 == 0:
                print(f"  decoded {i}/{len(clips)}", flush=True)
        CACHE.write_text(json.dumps(human_decoded, indent=2))
        print(f"cached {len(human_decoded)} human decodes")

    p2 = {json.loads(l)["item_id"]: json.loads(l)
          for l in (REPO_ROOT / "runs/gemini-flash-tts-v1-speech-similarity-v4/results.jsonl").read_text().splitlines() if l.strip()}
    p3 = {json.loads(l)["item_id"]: json.loads(l)
          for l in (REPO_ROOT / "runs/gemini-flash-tts-v1-phoneme-distance-v1/results.jsonl").read_text().splitlines() if l.strip()}

    import dose_r.dataset as dataset
    drug_of = {i.item_id: i.drug for i in dataset.load_items()}

    rows = {}
    for item_id, r3 in p3.items():
        s3 = r3.get("score")
        s2 = p2.get(item_id, {}).get("score")
        if not (s3 and s3.get("scoreable") and s2 and s2.get("scoreable")):
            continue
        if s3["metadata"].get("suffix_coverage_gap_caveat"):
            continue
        gem = normalize_phonemes(s3["metadata"].get("decoded_phonemes", ""))
        hum_raw = human_decoded.get(drug_of.get(item_id, "").lower())
        if not gem or not hum_raw:
            continue
        hum = normalize_phonemes(hum_raw)
        ipa_variants = list(s3["metadata"].get("all_variant_rates", {}).keys()) or [s3["metadata"]["best_matching_ipa_variant"]]

        # A: Gemini decode vs written dictionary IPA (what Path 3 does today)
        best_a = min(dist.weighted_feature_edit_distance(gem, normalize_phonemes(v))
                     / (len(ft.ipa_segs(normalize_phonemes(v))) or 1) for v in ipa_variants)
        # B: Gemini decode vs HUMAN decode -- same decoder both sides
        n_h = len(ft.ipa_segs(hum)) or 1
        best_b = dist.weighted_feature_edit_distance(gem, hum) / n_h

        rows[item_id] = {"vs_dict": best_a, "vs_human_decode": best_b,
                         "path2": s2["score"]}

    ids = list(rows)
    p2s = [rows[i]["path2"] for i in ids]
    print(f"\nitems compared: {len(ids)}\n")
    for key in ("vs_dict", "vs_human_decode"):
        vals = [rows[i][key] for i in ids]
        print(f"  {key:18s} spearman vs Path 2 = {spearman(vals, p2s):+.4f}")

    print("\near-verified items, percentile (0=worst, BAD should be low, OK/GOOD high):")
    ear = {"adquey": "BAD", "voranigo": "BAD", "vyloy": "BAD", "chantix": "GOOD",
           "aripiprazole": "OK", "acoramidis": "OK", "esomeprazole": "OK", "eliquis": "OK"}
    hdr = f"{'metric':18s}" + "".join(f"{n[:9]:>11s}" for n in ear)
    print(hdr)
    print(f"{'(ear verdict)':18s}" + "".join(f"{v:>11s}" for v in ear.values()))
    for key in ("vs_dict", "vs_human_decode"):
        vals = sorted(rows[i][key] for i in ids)
        cells = ""
        for name in ear:
            if name not in rows:
                cells += f"{'--':>11s}"
                continue
            v = rows[name][key]
            pct = 100.0 * (1.0 - sum(1 for x in vals if x < v) / len(vals))
            cells += f"{pct:>10.0f}%"
        print(f"{key:18s}{cells}")

    (REPO_ROOT / "runs" / "path3-decoder-noise-diagnosis.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
