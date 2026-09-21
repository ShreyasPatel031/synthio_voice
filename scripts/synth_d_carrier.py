#!/usr/bin/env python3
"""Synth DoSE carrier sentences with recipe-D custom_voice checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "runs/teacher-match-recipes/D_sft_text/checkpoint"
SPEAKER = "dose_teacher"
OUT = ROOT / "runs/listen-d-carrier"

SENTENCES = {
    "vorasidenib": "Let's initiate vorasidenib therapy for this patient to target the mutant IDH1 and IDH2 enzymes in the tumor.",
    "Advair": "We are starting the patient on Advair twice daily because combining an inhaled corticosteroid with a LABA will better control her asthma symptoms.",
    "Benadryl": "You can administer Benadryl to help relieve your child's sneezing and runny nose caused by upper respiratory allergies.",
    "Vraylar": "Because Vraylar acts as a dopamine D2 partial agonist, it should help manage the patient's schizophrenia symptoms effectively.",
}


def latest_ckpt() -> Path:
    epochs = sorted(CKPT.glob("checkpoint-epoch-*"))
    if not epochs:
        raise SystemExit(f"no D checkpoints under {CKPT}")
    return epochs[-1]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = latest_ckpt()
    print("ckpt", ckpt, flush=True)
    model = Qwen3TTSModel.from_pretrained(
        str(ckpt),
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    meta = []
    for name, sentence in SENTENCES.items():
        slug = name.lower()
        dest = OUT / f"{slug}__d_carrier.wav"
        wavs, sr = model.generate_custom_voice(
            text=sentence, language="English", speaker=SPEAKER
        )
        sf.write(dest, wavs[0], sr)
        meta.append({"ingredient": name, "sentence": sentence, "wav": str(dest), "sr": sr})
        print("ok", name, dest, flush=True)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    main()
