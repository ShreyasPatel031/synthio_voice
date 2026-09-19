#!/usr/bin/env python
"""Smoke-test GOP and IPA-to-vocab mapping on ear-checked items.

Path 2 is the regression target. This does not treat Path 3 as a peer.
"""
from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dose_r import dataset  # noqa: E402
from dose_r.forced_align import extract_drug_span_forced_align  # noqa: E402
from dose_r.references.ipa_references import ipa_variants_for  # noqa: E402
from dose_r.scoring.phoneme_distance import _get_panphon, normalize_phonemes  # noqa: E402
from dose_r.scoring.phoneme_model import _get_model, phonemize_word  # noqa: E402

RUN_DIR = REPO_ROOT / "runs" / "gemini-flash-tts-v1"
_TARGET_SR = 16_000
NAMES = [
    "chantix", "adquey", "voranigo", "vyloy",
    "aripiprazole", "esomeprazole", "eliquis", "acoramidis", "talquetamab",
]


def gop_detail(span_bytes: bytes, ipa: str, vocab: dict, blank_id: int):
    import librosa
    import torchaudio

    torch, processor, model = _get_model()
    ft, _ = _get_panphon()
    segs = list(ft.ipa_segs(normalize_phonemes(ipa)))
    kept = [s for s in segs if s in vocab]
    dropped = [s for s in segs if s not in vocab]
    if not kept:
        return {"segs": segs, "kept": kept, "dropped": dropped, "error": "no invok"}

    target_ids = [vocab[s] for s in kept]
    audio, _ = librosa.load(BytesIO(span_bytes), sr=_TARGET_SR, mono=True)
    inputs = processor(audio, sampling_rate=_TARGET_SR, return_tensors="pt")
    with torch.no_grad():
        log_probs = torch.log_softmax(model(inputs.input_values).logits, dim=-1)
    if log_probs.shape[1] < len(target_ids):
        return {"segs": segs, "kept": kept, "dropped": dropped, "error": "audio short"}

    targets = torch.tensor([target_ids], dtype=torch.int32)
    aligned, scores = torchaudio.functional.forced_align(
        log_probs, targets,
        torch.tensor([log_probs.shape[1]]), torch.tensor([len(target_ids)]),
        blank=blank_id,
    )
    frame_labels = aligned[0]
    path_logp = scores[0]
    mask = frame_labels != blank_id
    if int(mask.sum()) == 0:
        return {"segs": segs, "kept": kept, "dropped": dropped, "error": "all blank"}
    best_any = log_probs[0].max(dim=-1).values
    return {
        "segs": segs,
        "kept": kept,
        "dropped": dropped,
        "gop_mean": float(path_logp[mask].mean()),
        "gop_ratio": float((path_logp[mask] - best_any[mask]).mean()),
        "n_emit": int(mask.sum()),
    }


def main() -> int:
    p2 = {json.loads(l)["item_id"]: json.loads(l)
          for l in (REPO_ROOT / "runs/gemini-flash-tts-v1-speech-similarity-v4/results.jsonl").read_text().splitlines() if l.strip()}
    p3 = {json.loads(l)["item_id"]: json.loads(l)
          for l in (REPO_ROOT / "runs/gemini-flash-tts-v1-phoneme-distance-v1/results.jsonl").read_text().splitlines() if l.strip()}
    recs = {json.loads(l)["item_id"]: json.loads(l)
            for l in (RUN_DIR / "results.jsonl").read_text().splitlines() if l.strip()}
    items = {i.item_id: i for i in dataset.load_items()}

    print("loading model...", flush=True)
    torch, processor, model = _get_model()
    vocab = processor.tokenizer.get_vocab()
    blank_id = processor.tokenizer.pad_token_id
    print(f"vocab size {len(vocab)}", flush=True)

    # Which dictionary symbols are actually in the CTC vocab?
    interesting = ["oʊ", "aʊ", "ɑː", "ɑ", "o", "ʊ", "ɔː", "ɔ", "ɹ", "r", "iː", "i", "ɡ", "g", "æ", "ə", "ɪ"]
    print("vocab membership:", {s: (s in vocab) for s in interesting})

    for name in NAMES:
        item = items[name]
        variants = ipa_variants_for(item.drug)
        audio = (RUN_DIR / recs[name]["audio_path"]).read_bytes()
        span = extract_drug_span_forced_align(audio, item.sentence, item.drug)
        espeak = phonemize_word(item.drug)
        print("\n==", name, "p2", p2[name]["score"]["score"], "p3", p3[name]["score"]["score"])
        print("  espeak-of-spelling:", espeak)
        print("  espeak in vocab:", [(s, s in vocab) for s in espeak])
        if span is None:
            print("  NO SPAN")
            continue
        best = None
        for v in variants:
            d = gop_detail(span, v, vocab, blank_id)
            print(f"  ipa {v}")
            print(f"    segs={d.get('segs')} dropped={d.get('dropped')} gop_mean={d.get('gop_mean')} gop_ratio={d.get('gop_ratio')} err={d.get('error')}")
            if d.get("gop_ratio") is not None and (best is None or d["gop_ratio"] > best):
                best = d["gop_ratio"]
        print("  best gop_ratio", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
