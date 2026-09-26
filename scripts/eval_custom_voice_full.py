#!/usr/bin/env python3
"""Synth every DoSE sentence with a custom_voice ckpt, then official CTC score."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DOSE = ROOT / "data" / "dose_v1.jsonl"
MAX_NEW = 384
MAX_WAV = 2_000_000


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def main() -> None:
    ckpt = Path(sys.argv[1])
    speaker = sys.argv[2]
    out = Path(sys.argv[3])
    condition = sys.argv[4]
    out.mkdir(parents=True, exist_ok=True)
    items = []
    seen = set()
    for line in DOSE.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = slug(row["name"])
        if key in seen:
            continue
        seen.add(key)
        items.append((key, row["sentence"]))
    todo = [it for it in items if not (out / f"{it[0]}.wav").exists()]
    print(f"synth {len(todo)}/{len(items)} {ckpt} speaker={speaker}", flush=True)
    if todo:
        model = Qwen3TTSModel.from_pretrained(
            str(ckpt), device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        bs = 16
        i = 0
        while i < len(todo):
            batch = todo[i : i + bs]
            texts = [t for _, t in batch]
            try:
                torch.manual_seed(0)
                wavs, sr = model.generate_custom_voice(
                    text=texts, language=["English"] * len(texts),
                    speaker=speaker, do_sample=False, max_new_tokens=MAX_NEW,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                print("OOM batch", bs, flush=True)
                continue
            for (key, _), wav in zip(batch, wavs):
                dest = out / f"{key}.wav"
                sf.write(dest, wav, sr)
                if dest.stat().st_size > MAX_WAV:
                    dest.unlink()
                    print("RUNAWAY", key, flush=True)
                else:
                    print("ok", key, flush=True)
            i += len(batch)
        del model
        torch.cuda.empty_cache()
    subprocess.check_call([
        sys.executable, str(ROOT / "scripts" / "score_dose_ctc_vs_gemini_ipa.py"),
        "--wav-dir", str(out),
        "--condition", condition,
        "--scope", "full",
    ])


if __name__ == "__main__":
    main()
