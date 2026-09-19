#!/usr/bin/env python
"""GOP with the English phoneme CTC model, on the 40 paired spans."""
from __future__ import annotations

import json
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
from dose_r.scoring.phoneme_distance import normalize_phonemes  # noqa: E402
from scripts.probe_timit_paired import MODEL, SPAN, longest  # noqa: E402

EAR = ["chantix", "adquey", "voranigo", "vyloy", "aripiprazole",
       "esomeprazole", "eliquis", "acoramidis", "talquetamab"]


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


def gop(log_probs, ids, blank, torch, torchaudio):
    if log_probs.shape[0] < len(ids) or not ids:
        return None
    targets = torch.tensor([ids], dtype=torch.int32)
    lp = log_probs.unsqueeze(0)
    aligned, scores = torchaudio.functional.forced_align(
        lp, targets,
        torch.tensor([lp.shape[1]]), torch.tensor([len(ids)]),
        blank=blank,
    )
    labels = aligned[0]
    path = scores[0]
    best = lp[0].max(dim=-1).values
    emit = labels != blank
    if int(emit.sum()) == 0:
        return None
    phone = []
    i, n = 0, int(labels.shape[0])
    while i < n:
        tok = int(labels[i])
        j = i + 1
        while j < n and int(labels[j]) == tok:
            j += 1
        if tok != blank:
            phone.append(float((path[i:j] - best[i:j]).mean()))
        i = j
    if not phone:
        return None
    return {
        "ratio": float((path[emit] - best[emit]).mean()),
        "phone_min": min(phone),
        "phone_mean": sum(phone) / len(phone),
    }


def main() -> int:
    import librosa
    import torch
    import torchaudio
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    paired = json.loads((REPO_ROOT / "runs/path3-timit-probe/paired.json").read_text())
    rows = paired["rows"]
    proc = Wav2Vec2Processor.from_pretrained(MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL)
    model.eval()
    vocab = proc.tokenizer.get_vocab()
    symbols = set(vocab)
    blank = proc.tokenizer.pad_token_id

    scored = []
    for item_id, row in rows.items():
        if "path2" not in row:
            continue
        span = SPAN / f"{item_id}.wav"
        if not span.exists():
            continue
        audio, _ = librosa.load(BytesIO(span.read_bytes()), sr=16000, mono=True)
        inputs = proc(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            log_probs = torch.log_softmax(model(inputs.input_values).logits, dim=-1)[0]
        best = None
        for v in ipa_variants_for(item_id):
            toks = longest(normalize_phonemes(v), symbols)
            ids = [vocab[t] for t in toks if t in vocab]
            g = gop(log_probs, ids, blank, torch, torchaudio)
            if g and (best is None or g["ratio"] > best["ratio"]):
                best = g
                best["tokens"] = toks
        if not best:
            continue
        best["path2"] = row["path2"]
        best["rate"] = row.get("rate")
        scored.append((item_id, best))
        if item_id in EAR:
            print(f"{item_id:16s} p2={best['path2']:.3f} gop={best['ratio']:.3f} min={best['phone_min']:.3f} rate={best['rate']}")

    print(f"\nn={len(scored)}")
    p2 = [b["path2"] for _, b in scored]
    for key in ("ratio", "phone_min", "phone_mean"):
        rho = spearman([b[key] for _, b in scored], p2)
        print(f"  gop {key:12s} rho={rho:+.4f}")
    if any(b.get("rate") is not None for _, b in scored):
        rho = spearman([b["rate"] for _, b in scored], p2)
        print(f"  {'edit rate':16s} rho={rho:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
