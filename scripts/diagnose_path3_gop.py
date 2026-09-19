#!/usr/bin/env python
"""Test Goodness-of-Pronunciation (GOP) as Path 3's scoring mechanism,
against Path 2 (ground truth) on the items where both can be computed.

Why GOP instead of free phoneme decoding
------------------------------------------
Path 3 currently decodes the candidate's audio into a phoneme string with
no constraints, then edit-distances that string against dictionary IPA.
Two diagnostics showed that is the wrong tool:
  - Every distance-metric variant tracks Path 2 at rho -0.45..-0.50
    (scripts/diagnose_path3_vs_path2.py) -- the formula is not the
    bottleneck.
  - Comparing Gemini's decode against a HUMAN's decode of the same word
    (same decoder both sides, so systematic bias would cancel) made
    correlation slightly WORSE, -0.462 vs -0.485
    (scripts/diagnose_path3_decoder_noise.py). So the decoder's error is
    random per-utterance noise, which compounds rather than cancels.

Free decoding is an open-ended guess: the model must pick among ~392
phoneme symbols with nothing to anchor it, and on coined drug names it
visibly fails (emitting "ø", a non-English phoneme, for talquetamab).

GOP is the standard approach in the mispronunciation-detection / CAPT
literature and is a far more constrained question: given that this audio
is SUPPOSED to be phoneme sequence X (from the dictionary), how well do
the acoustics actually support X? No open-vocabulary guessing. We already
run CTC forced alignment for span extraction, and
`torchaudio.functional.forced_align` returns per-frame log-probabilities
along the aligned path, so the core signal is nearly free.

Two variants computed here:
  gop_mean  -- mean per-frame log-prob along the forced-aligned path.
               Higher (closer to 0) = acoustics support the expected
               phonemes well.
  gop_ratio -- the classic GOP: mean over frames of
               log P(aligned phoneme) - log P(best competing phoneme).
               Penalizes frames where some OTHER phoneme fits much better,
               which is exactly the mispronunciation signal and is more
               robust to overall audio quality/loudness than gop_mean.

Best score across the item's accepted IPA variants is taken, as elsewhere.
"""

from __future__ import annotations

import json
import math
import sys
from io import BytesIO
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dose_r import dataset  # noqa: E402
from dose_r.forced_align import extract_drug_span_forced_align  # noqa: E402
from dose_r.references.ipa_references import ipa_variants_for  # noqa: E402
from dose_r.scoring.phoneme_distance import _get_panphon, normalize_phonemes  # noqa: E402
from dose_r.scoring.phoneme_model import _get_model  # noqa: E402

RUN_DIR = REPO_ROOT / "runs" / "gemini-flash-tts-v1"
_TARGET_SR = 16_000


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


def gop_for_variant(span_bytes: bytes, ipa: str) -> tuple[float, float] | None:
    """(gop_mean, gop_ratio) for one expected IPA sequence, or None if the
    sequence can't be mapped into the model's vocabulary at all.
    """
    import librosa
    import torchaudio

    torch, processor, model = _get_model()
    ft, _ = _get_panphon()
    vocab = processor.tokenizer.get_vocab()
    blank_id = processor.tokenizer.pad_token_id

    segs = [s for s in ft.ipa_segs(normalize_phonemes(ipa)) if s in vocab]
    if not segs:
        return None
    target_ids = [vocab[s] for s in segs]

    audio, _ = librosa.load(BytesIO(span_bytes), sr=_TARGET_SR, mono=True)
    inputs = processor(audio, sampling_rate=_TARGET_SR, return_tensors="pt")
    with torch.no_grad():
        log_probs = torch.log_softmax(model(inputs.input_values).logits, dim=-1)

    if log_probs.shape[1] < len(target_ids):
        return None  # audio too short to contain the expected sequence

    targets = torch.tensor([target_ids], dtype=torch.int32)
    aligned, scores = torchaudio.functional.forced_align(
        log_probs, targets,
        torch.tensor([log_probs.shape[1]]), torch.tensor([len(target_ids)]),
        blank=blank_id,
    )
    frame_labels = aligned[0]
    path_logp = scores[0]

    # Only frames that emit a real phoneme (not blank) carry pronunciation
    # evidence; blank frames are the model declining to commit and would
    # otherwise dominate the average on a peaky CTC model.
    mask = frame_labels != blank_id
    if mask.sum() == 0:
        return None

    gop_mean = float(path_logp[mask].mean())

    # Classic GOP ratio: how much better does the BEST competing phoneme fit
    # each aligned frame? Near 0 = nothing fits better = well pronounced.
    best_any = log_probs[0].max(dim=-1).values
    gop_ratio = float((path_logp[mask] - best_any[mask]).mean())
    return gop_mean, gop_ratio


def main() -> int:
    p2 = {json.loads(l)["item_id"]: json.loads(l)
          for l in (REPO_ROOT / "runs/gemini-flash-tts-v1-speech-similarity-v4/results.jsonl").read_text().splitlines() if l.strip()}
    p3 = {json.loads(l)["item_id"]: json.loads(l)
          for l in (REPO_ROOT / "runs/gemini-flash-tts-v1-phoneme-distance-v1/results.jsonl").read_text().splitlines() if l.strip()}
    recs = {json.loads(l)["item_id"]: json.loads(l)
            for l in (RUN_DIR / "results.jsonl").read_text().splitlines() if l.strip()}
    items_by_id = {i.item_id: i for i in dataset.load_items()}

    todo = [i for i in p3
            if p3[i].get("score") and p3[i]["score"].get("scoreable")
            and p2.get(i, {}).get("score") and p2[i]["score"].get("scoreable")
            and not p3[i]["score"]["metadata"].get("suffix_coverage_gap_caveat")]
    print(f"items to score with GOP: {len(todo)}")

    rows = {}
    for n, item_id in enumerate(todo, 1):
        item = items_by_id[item_id]
        variants = ipa_variants_for(item.drug)
        if not variants:
            continue
        try:
            audio = (RUN_DIR / recs[item_id]["audio_path"]).read_bytes()
            span = extract_drug_span_forced_align(audio, item.sentence, item.drug)
            if span is None:
                continue
            results = [r for r in (gop_for_variant(span, v) for v in variants) if r]
            if not results:
                continue
        except Exception as exc:
            print(f"  SKIP {item_id}: {exc}")
            continue
        rows[item_id] = {
            "gop_mean": max(r[0] for r in results),
            "gop_ratio": max(r[1] for r in results),
            "path2": p2[item_id]["score"]["score"],
            "path3_current": p3[item_id]["score"]["score"],
        }
        if n % 25 == 0:
            print(f"  {n}/{len(todo)}", flush=True)

    ids = list(rows)
    p2s = [rows[i]["path2"] for i in ids]
    print(f"\nscored: {len(ids)}\n")
    print("Spearman vs Path 2 (ground truth). GOP is higher=better, so")
    print("POSITIVE and large is good here (unlike the distance metrics).")
    for key in ("gop_mean", "gop_ratio", "path3_current"):
        vals = [rows[i][key] for i in ids]
        print(f"  {key:16s} rho={spearman(vals, p2s):+.4f}")

    ear = {"adquey": "BAD", "voranigo": "BAD", "vyloy": "BAD", "chantix": "GOOD",
           "aripiprazole": "OK", "acoramidis": "OK", "esomeprazole": "OK", "eliquis": "OK"}
    print("\near-verified items, percentile (0=worst; BAD low, OK/GOOD high):")
    hdr = f"{'metric':16s}" + "".join(f"{n[:9]:>11s}" for n in ear)
    print(hdr)
    print(f"{'(ear verdict)':16s}" + "".join(f"{v:>11s}" for v in ear.values()))
    for key in ("gop_mean", "gop_ratio", "path3_current"):
        vals = sorted(rows[i][key] for i in ids)
        cells = ""
        for name in ear:
            if name not in rows:
                cells += f"{'--':>11s}"
                continue
            v = rows[name][key]
            pct = 100.0 * (sum(1 for x in vals if x < v) / len(vals))
            cells += f"{pct:>10.0f}%"
        print(f"{key:16s}{cells}")

    (REPO_ROOT / "runs" / "path3-gop-diagnosis.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
