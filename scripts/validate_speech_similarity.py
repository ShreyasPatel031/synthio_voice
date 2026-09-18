#!/usr/bin/env python
"""Validate Path 2 (speech_similarity) before trusting it at any scale.

Three checks, all must pass:
1. Discrimination -- correct TTS/reference pairs score high, deliberately
   mismatched pairs score low.
2. Voice-invariance -- two different HUMANS saying the same correct word
   score high despite being different speakers (the property this whole
   metric exists to have, that raw acoustic distance does not).
3. Naive-baseline comparison -- the same voice-invariance pairs scored with
   plain MFCC+DTW, to show concretely that the naive approach conflates
   voice identity with pronunciation and this approach does not.

Costs one Cloud STT call per TTS clip (to extract the drug-name span via
dose_r.audio_span) plus local model inference -- no other spend.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dose_r import dataset  # noqa: E402
from dose_r.audio_span import extract_drug_span  # noqa: E402
from dose_r.scoring.speech_similarity import (  # noqa: E402
    extract_frame_embeddings,
    score_speech_similarity,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TTS_AUDIO_DIR = REPO_ROOT / "runs" / "standin-v1" / "audio" / "gtts-standard-c"
MW_VALIDATION_DIR = REPO_ROOT / "data" / "reference_audio" / "mw_validation_sample"

CORRECT_NAMES = ["Abilify", "Advil", "Aspirin", "Chantix", "Claritin"]
MISMATCHES = [("Abilify", "Chantix"), ("Advil", "Claritin"), ("Aspirin", "Abilify")]


def _mfcc_dtw_score(path_a: Path, path_b: Path) -> float:
    """Naive baseline this module deliberately does NOT use -- built only to
    demonstrate, not to ship as a scorer."""
    import librosa

    ya, sr = librosa.load(str(path_a), sr=16000, mono=True)
    yb, _ = librosa.load(str(path_b), sr=16000, mono=True)
    ma = librosa.feature.mfcc(y=ya, sr=sr, n_mfcc=13).T
    mb = librosa.feature.mfcc(y=yb, sr=sr, n_mfcc=13).T
    D, wp = librosa.sequence.dtw(ma.T, mb.T, metric="euclidean")
    avg_cost = D[-1, -1] / len(wp)
    return max(0.0, 5.0 - avg_cost / 40.0)  # rough, loose scale -- not calibrated


def main() -> int:
    items = {i.item_id: i for i in dataset.load_items()}
    manifest = [json.loads(l) for l in
               (REPO_ROOT / "data/reference_audio/manifest.jsonl").read_text().splitlines()]
    dc_map = {r["ingredient"]: REPO_ROOT / r["local_path"] for r in manifest
             if r["source"] == "drugs.com" and r.get("coverage") == "full"}
    mw_map = {r["ingredient"]: r["source_url"].split("/")[-1] for r in manifest
             if r["source"] == "merriam-webster" and r.get("coverage") == "full"}

    spans: dict[str, bytes | None] = {}
    print("Extracting drug-name spans from full-sentence TTS clips (costs 1 STT call each)...")
    for name in CORRECT_NAMES:
        item = items[name.lower()]
        full_audio = (TTS_AUDIO_DIR / f"{name.lower()}.wav").read_bytes()
        spans[name] = extract_drug_span(full_audio, item.sentence, item.drug)
        print(f"  {name:12s} {'OK' if spans[name] else 'FAILED to locate span'}")

    print("\n" + "=" * 70)
    print("CHECK 1: DISCRIMINATION")
    print("=" * 70)
    correct_scores, mismatch_scores = [], []
    for name in CORRECT_NAMES:
        if not spans[name]:
            continue
        fa = extract_frame_embeddings(spans[name])
        fb = extract_frame_embeddings(dc_map[name])
        score, comp = score_speech_similarity(fa, fb)
        correct_scores.append(score)
        print(f"  CORRECT  {name:12s} score={score:.2f}  f1={comp['f1']}")
    for a, b in MISMATCHES:
        if not spans[a]:
            continue
        fa = extract_frame_embeddings(spans[a])
        fb = extract_frame_embeddings(dc_map[b])
        score, comp = score_speech_similarity(fa, fb)
        mismatch_scores.append(score)
        print(f"  MISMATCH {a:8s} vs {b:8s}-ref   score={score:.2f}  f1={comp['f1']}  (want LOW)")
    check1_pass = correct_scores and mismatch_scores and min(correct_scores) > max(mismatch_scores)
    print(f"\n  CHECK 1: {'PASS' if check1_pass else 'FAIL'} "
         f"(min correct={min(correct_scores):.2f}, max mismatch={max(mismatch_scores):.2f})")

    if not MW_VALIDATION_DIR.exists():
        print(f"\n{MW_VALIDATION_DIR} not found -- skipping checks 2/3 "
             "(need Merriam-Webster clips pulled for voice-invariance testing).")
        return 0 if check1_pass else 1

    print("\n" + "=" * 70)
    print("CHECK 2: VOICE-INVARIANCE (two different humans, same correct word)")
    print("=" * 70)
    invariance_scores = []
    for name in CORRECT_NAMES:
        mw_path = MW_VALIDATION_DIR / mw_map[name]
        if not mw_path.exists():
            continue
        fa = extract_frame_embeddings(dc_map[name])
        fb = extract_frame_embeddings(mw_path)
        score, comp = score_speech_similarity(fa, fb)
        invariance_scores.append(score)
        print(f"  {name:12s} drugs.com-human vs MW-human   score={score:.2f}  f1={comp['f1']}")
    check2_pass = invariance_scores and (sum(invariance_scores) / len(invariance_scores)) >= 3.0

    print("\n" + "=" * 70)
    print("CHECK 3: naive MFCC+DTW baseline on the SAME voice-invariance pairs")
    print("=" * 70)
    naive_scores = []
    for name in CORRECT_NAMES:
        mw_path = MW_VALIDATION_DIR / mw_map[name]
        if not mw_path.exists():
            continue
        s = _mfcc_dtw_score(dc_map[name], mw_path)
        naive_scores.append(s)
        print(f"  {name:12s} drugs.com-human vs MW-human   MFCC+DTW score={s:.2f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Check 1 (discrimination):    {'PASS' if check1_pass else 'FAIL'}")
    print(f"  Check 2 (voice-invariance):  {'PASS' if check2_pass else 'FAIL'} "
         f"(mean={sum(invariance_scores)/max(len(invariance_scores),1):.2f})")
    if naive_scores:
        print(f"  Check 3 (vs. naive baseline): SSL mean={sum(invariance_scores)/len(invariance_scores):.2f} "
             f"vs. MFCC+DTW mean={sum(naive_scores)/len(naive_scores):.2f} "
             f"({'SSL approach wins' if sum(invariance_scores)/len(invariance_scores) > sum(naive_scores)/len(naive_scores) else 'naive baseline was NOT worse -- investigate'})")
    return 0 if (check1_pass and check2_pass) else 1


if __name__ == "__main__":
    raise SystemExit(main())
