#!/usr/bin/env python
"""Path 2 (wavlm F1) on the freshly synthesized probe clips, same audio GOP saw."""
from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/workspace/.cache/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/.cache/huggingface")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dose_r import dataset  # noqa: E402
from dose_r.forced_align import extract_drug_span_forced_align  # noqa: E402
from dose_r.references.reference_clips import available_clips  # noqa: E402
from dose_r.scoring.phoneme_model import _get_model  # noqa: E402

NAMES = ["chantix", "adquey", "voranigo", "vyloy", "aripiprazole",
         "esomeprazole", "eliquis", "acoramidis", "talquetamab"]
AUDIO = REPO_ROOT / "runs" / "path3-gop-probe" / "audio"
SPAN = REPO_ROOT / "runs" / "path3-gop-probe" / "spans"


def main() -> int:
    items = {i.item_id: i for i in dataset.load_items()}
    SPAN.mkdir(parents=True, exist_ok=True)
    print("extract spans", flush=True)
    _get_model()
    for name in NAMES:
        item = items[name]
        out = SPAN / f"{name}.wav"
        if out.exists():
            continue
        span = extract_drug_span_forced_align((AUDIO / f"{name}.wav").read_bytes(), item.sentence, item.drug)
        if span is None:
            print(name, "NO SPAN")
            continue
        out.write_bytes(span)
        print(" ", name, len(span), flush=True)
    # drop the CTC model before loading wavlm
    _get_model.cache_clear()
    gc.collect()

    from dose_r.scoring.speech_similarity import extract_frame_embeddings, score_speech_similarity
    clips = available_clips()
    print("path2", flush=True)
    for name in NAMES:
        span_path = SPAN / f"{name}.wav"
        clip = clips.get(items[name].drug)
        if not span_path.exists() or clip is None:
            print(name, "skip", bool(span_path.exists()), clip is not None)
            continue
        score, comp = score_speech_similarity(
            extract_frame_embeddings(span_path.read_bytes()),
            extract_frame_embeddings(clip.path),
        )
        print(f"{name:16s} path2={score:.3f} f1={comp['f1']:.3f} src={clip.source}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
