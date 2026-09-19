#!/usr/bin/env python
"""Token-level distance using the TIMIT/IPA CTC vocab, vs Path 2 on the
same 9 resynthesized spans.
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
from dose_r.scoring.phoneme_distance import normalize_phonemes  # noqa: E402

MODEL = "vitouphy/wav2vec2-xls-r-300m-timit-phoneme"
NAMES = ["chantix", "adquey", "voranigo", "vyloy", "aripiprazole",
         "esomeprazole", "eliquis", "acoramidis", "talquetamab"]
P2 = {
    "chantix": 3.272, "adquey": 2.286, "voranigo": 3.327, "vyloy": 1.546,
    "aripiprazole": 2.635, "esomeprazole": 2.666, "eliquis": 2.600,
    "acoramidis": 2.346, "talquetamab": 2.967,
}
SPAN = REPO_ROOT / "runs" / "path3-gop-probe" / "spans"

# Dictionary IPA symbols -> this model's symbols.
_MAP = {
    "tʃ": "ʧ", "dʒ": "ʤ", "ɡ": "g",
    "iː": "i", "uː": "u", "ɑː": "ɑ", "ɔː": "ɔ", "ɜː": "ɝ", "eː": "eɪ", "oː": "oʊ",
    "ɚ": "ɝ", "ɝ": "ɝ", "ɨ": "ɪ", "ᵻ": "ɪ", "ɐ": "ə",
    "r": "ɹ",
}


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


def longest(s, symbols):
    # apply multi-char map first by trying mapped symbols in the matcher
    i, out = 0, []
    while i < len(s):
        match = None
        mapped = None
        for j in range(len(s), i, -1):
            piece = s[i:j]
            piece = _MAP.get(piece, piece)
            if piece in symbols:
                match = s[i:j]
                mapped = piece
                break
        if mapped is None:
            i += 1
        else:
            out.append(mapped)
            i += len(match)
    return out


def leven(a, b):
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (0 if a[i - 1] == b[j - 1] else 1))
            prev = cur
    return dp[m]


def ctc_tokens(ids, proc, blank):
    out, prev = [], None
    skip = {"<pad>", "<s>", "</s>", "<unk>", "|", " ", ""}
    for i in ids:
        if i != prev and i != blank:
            tok = proc.tokenizer.convert_ids_to_tokens(int(i))
            if tok not in skip:
                out.append(tok)
        prev = i
    return out


def main() -> int:
    import librosa
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    proc = Wav2Vec2Processor.from_pretrained(MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL)
    model.eval()
    symbols = set(proc.tokenizer.get_vocab())
    blank = proc.tokenizer.pad_token_id
    rates, p2s = [], []
    for name in NAMES:
        audio, _ = librosa.load(BytesIO((SPAN / f"{name}.wav").read_bytes()), sr=16000, mono=True)
        inputs = proc(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        hyp = ctc_tokens(torch.argmax(logits, dim=-1)[0].tolist(), proc, blank)
        best = None
        best_ref = None
        for v in ipa_variants_for(name):
            ref = longest(normalize_phonemes(v), symbols)
            if not ref:
                continue
            rate = leven(hyp, ref) / len(ref)
            if best is None or rate < best:
                best, best_ref = rate, ref
        score = 5 * max(0.0, 1 - (best or 1))
        print(f"{name:16s} p2={P2[name]:.3f} rate={best:.3f} score={score:.2f}")
        print(f"  hyp {hyp}")
        print(f"  ref {best_ref}")
        rates.append(best)
        p2s.append(P2[name])
    print("spearman rate vs path2 (neg better)", round(spearman(rates, p2s), 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
