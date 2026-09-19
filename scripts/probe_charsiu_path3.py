#!/usr/bin/env python
"""Decode the 9 probe spans with an English phoneme CTC model and
compare to Path 2 on the same clips.
"""
from __future__ import annotations

import math
import os
import sys
from io import BytesIO
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/workspace/.cache/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/.cache/huggingface")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dose_r.references.ipa_references import ipa_variants_for  # noqa: E402
from dose_r.scoring.phoneme_distance import best_phoneme_distance, rate_to_score  # noqa: E402

MODEL = "vitouphy/wav2vec2-xls-r-300m-timit-phoneme"
NAMES = ["chantix", "adquey", "voranigo", "vyloy", "aripiprazole",
         "esomeprazole", "eliquis", "acoramidis", "talquetamab"]
# Path 2 on these same resynthesized spans
P2 = {
    "chantix": 3.272, "adquey": 2.286, "voranigo": 3.327, "vyloy": 1.546,
    "aripiprazole": 2.635, "esomeprazole": 2.666, "eliquis": 2.600,
    "acoramidis": 2.346, "talquetamab": 2.967,
}
SPAN = REPO_ROOT / "runs" / "path3-gop-probe" / "spans"


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
    import librosa
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    print("loading", MODEL, flush=True)
    proc = Wav2Vec2Processor.from_pretrained(MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL)
    model.eval()
    vocab = proc.tokenizer.get_vocab()
    print("vocab", len(vocab), "sample", list(vocab)[:30])

    scores, p2s = [], []
    for name in NAMES:
        audio, _ = librosa.load(BytesIO((SPAN / f"{name}.wav").read_bytes()), sr=16000, mono=True)
        inputs = proc(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        text = proc.batch_decode(torch.argmax(logits, dim=-1))[0]
        variants = ipa_variants_for(name)
        try:
            dist = best_phoneme_distance(text, variants)
            score = rate_to_score(dist["rate"])
        except Exception as exc:
            score = None
            dist = {"error": str(exc)}
        print(f"{name:16s} p2={P2[name]:.3f} score={score} decode={text!r}")
        if score is not None:
            scores.append(score)
            p2s.append(P2[name])
    if len(scores) == len(NAMES):
        print("spearman vs path2", round(spearman(scores, p2s), 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
