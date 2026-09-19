#!/usr/bin/env python
"""Resynthesize the ear-checked Gemini items and score GOP against Path 2.

Original candidate WAVs were gitignored and did not transfer. These clips
are a fresh Kore synthesis of the same sentences, so stored Path 2 numbers
are only a reference, not a paired score. GOP here uses the CTC vocabulary's
own tokens (longest match), not panphon segments.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from io import BytesIO
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/workspace/.cache/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/.cache/huggingface")

if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON"):
    raw = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_B64", "")
    if raw:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"] = base64.b64decode(raw).decode()

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dose_r import dataset  # noqa: E402
from dose_r.adapters.gemini_tts import GeminiTTSAdapter  # noqa: E402
from dose_r.config import ALL_SYSTEMS  # noqa: E402
from dose_r.forced_align import extract_drug_span_forced_align  # noqa: E402
from dose_r.references.ipa_references import ipa_variants_for  # noqa: E402
from dose_r.scoring.phoneme_distance import (  # noqa: E402
    _get_panphon, best_phoneme_distance, normalize_phonemes, rate_to_score,
)
from dose_r.scoring.phoneme_model import MODEL_ID, _get_model, transcribe_phonemes  # noqa: E402

NAMES = ["chantix", "adquey", "voranigo", "vyloy", "aripiprazole",
         "esomeprazole", "eliquis", "acoramidis", "talquetamab"]
OUT = REPO_ROOT / "runs" / "path3-gop-probe" / "audio"
_TARGET_SR = 16_000


def longest_match(ipa: str, symbols: set[str]) -> list[str]:
    i, out = 0, []
    norm = normalize_phonemes(ipa)
    while i < len(norm):
        match = None
        for j in range(len(norm), i, -1):
            if norm[i:j] in symbols:
                match = norm[i:j]
                break
        if match is None:
            i += 1
        else:
            out.append(match)
            i += len(match)
    return out


def gop(span: bytes, tokens: list[str], vocab: dict, blank_id: int):
    import librosa
    import torchaudio

    torch, processor, model = _get_model()
    if not tokens:
        return None
    ids = [vocab[t] for t in tokens]
    audio, _ = librosa.load(BytesIO(span), sr=_TARGET_SR, mono=True)
    inputs = processor(audio, sampling_rate=_TARGET_SR, return_tensors="pt")
    with torch.no_grad():
        log_probs = torch.log_softmax(model(inputs.input_values).logits, dim=-1)
    if log_probs.shape[1] < len(ids):
        return None
    targets = torch.tensor([ids], dtype=torch.int32)
    aligned, scores = torchaudio.functional.forced_align(
        log_probs, targets,
        torch.tensor([log_probs.shape[1]]), torch.tensor([len(ids)]),
        blank=blank_id,
    )
    labels = aligned[0]
    path = scores[0]
    best_any = log_probs[0].max(dim=-1).values
    phone_gops = []
    i = 0
    n = labels.shape[0]
    while i < n:
        tok = int(labels[i])
        j = i + 1
        while j < n and int(labels[j]) == tok:
            j += 1
        if tok != blank_id:
            sl = slice(i, j)
            phone_gops.append(float((path[sl] - best_any[sl]).mean()))
        i = j
    if not phone_gops:
        return None
    emit = labels != blank_id
    return {
        "gop_ratio": float((path[emit] - best_any[emit]).mean()),
        "gop_mean": float(path[emit].mean()),
        "phone_min": min(phone_gops),
        "phone_mean": sum(phone_gops) / len(phone_gops),
        "n_phones": len(phone_gops),
    }


def main() -> int:
    p2 = {json.loads(l)["item_id"]: json.loads(l)["score"]["score"]
          for l in (REPO_ROOT / "runs/gemini-flash-tts-v1-speech-similarity-v4/results.jsonl").read_text().splitlines() if l.strip()}
    items = {i.item_id: i for i in dataset.load_items()}
    spec = next(v for v in ALL_SYSTEMS.values() if v.tier == "gemini-tts")
    adapter = GeminiTTSAdapter(spec, speaker="Kore")
    OUT.mkdir(parents=True, exist_ok=True)

    print("loading ctc model", flush=True)
    torch, processor, model = _get_model()
    vocab = processor.tokenizer.get_vocab()
    symbols = set(vocab)
    blank = processor.tokenizer.pad_token_id

    for name in NAMES:
        item = items[name]
        path = OUT / f"{name}.wav"
        if not path.exists():
            print(f"synth {name}", flush=True)
            wav, meta = adapter._synthesize(item.sentence)
            path.write_bytes(wav)
        else:
            wav = path.read_bytes()
        span = extract_drug_span_forced_align(wav, item.sentence, item.drug)
        if span is None:
            print(name, "NO SPAN")
            continue
        decoded = transcribe_phonemes(span)
        variants = ipa_variants_for(item.drug)
        dist = best_phoneme_distance(decoded, variants)
        gops = []
        for v in variants:
            toks = longest_match(v, symbols)
            g = gop(span, toks, vocab, blank)
            if g:
                g["tokens"] = toks
                gops.append(g)
        best = max(gops, key=lambda g: g["gop_ratio"]) if gops else None
        print(f"\n{name} stored_p2={p2.get(name)} old_formula={rate_to_score(dist['rate'])}")
        print(f"  decoded {decoded}")
        print(f"  dict    {variants}")
        if best:
            print(f"  tokens  {best['tokens']}")
            print(f"  gop_ratio={best['gop_ratio']:.3f} phone_mean={best['phone_mean']:.3f} phone_min={best['phone_min']:.3f} gop_mean={best['gop_mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
