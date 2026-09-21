#!/usr/bin/env python3
"""Same 4 Gemini sentences, 1 pass, LoRA on all talker attn+MLP.

Last-4-layer FT lifted dose (0.53→0.65) but dropped confirm.
LoRA lets every layer move a little without full-model smash.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dose_r.forced_align import extract_drug_span_forced_align
from dose_r.references.reference_clips import available_clips_all
from dose_r.scoring.candidate_eval import score_against_best_reference

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
FT_SCRIPTS = Path("/home/shreyaspatel/Qwen3-TTS/finetuning")
if not (FT_SCRIPTS / "sft_12hz.py").exists():
    FT_SCRIPTS = Path.home() / "Qwen3-TTS" / "finetuning"

DRUG = "vorasidenib"
SPEAKER = "dose_sent_lora"
WORK = ROOT / "runs" / "ft-vora-lora"
SENT_TTS = ROOT / "runs" / "vora-sent-cloud-vs-gemini" / "tts"
CODED_REUSE = ROOT / "runs" / "ft-vora-last-layers" / "train_coded.jsonl"
MAX_WAV_BYTES = 2_000_000

SENTENCES = [
    {"id": "dose", "text": "Let's initiate vorasidenib therapy for this patient to target the mutant IDH1 and IDH2 enzymes in the tumor."},
    {"id": "prescribed", "text": "The oncologist prescribed vorasidenib for the patient's IDH-mutant glioma."},
    {"id": "confirm", "text": "Please confirm the vorasidenib dose before the next clinic visit."},
    {"id": "monitor", "text": "Patients taking vorasidenib need monitoring for liver enzyme elevation."},
]


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    repeats = 8
    WORK.mkdir(parents=True, exist_ok=True)
    raw, coded, ckpt_root = WORK / "train_raw.jsonl", WORK / "train_coded.jsonl", WORK / "ckpt"
    if CODED_REUSE.exists():
        shutil.copy(CODED_REUSE, coded)
        print(f"reused {CODED_REUSE}", flush=True)
    else:
        n = 0
        with raw.open("w") as f:
            for item in SENTENCES:
                wav = SENT_TTS / f"{item['id']}__gemini.wav"
                rec = {"audio": str(wav.resolve()), "text": item["text"], "ref_audio": str(wav.resolve())}
                for _ in range(repeats):
                    f.write(json.dumps(rec) + "\n")
                    n += 1
        print(f"train rows={n}", flush=True)
        _run([
            sys.executable, str(FT_SCRIPTS / "prepare_data.py"),
            "--device", "cuda:0",
            "--tokenizer_model_path", "Qwen/Qwen3-TTS-Tokenizer-12Hz",
            "--input_jsonl", str(raw), "--output_jsonl", str(coded),
        ])
    if ckpt_root.exists():
        shutil.rmtree(ckpt_root)
    ckpt_root.mkdir(parents=True)
    _run([
        sys.executable, str(FT_SCRIPTS / "sft_12hz_lora.py"),
        "--init_model_path", MODEL_ID,
        "--output_model_path", str(ckpt_root),
        "--train_jsonl", str(coded),
        "--batch_size", "1",
        "--lr", "2e-5",
        "--num_epochs", "1",
        "--speaker_name", SPEAKER,
        "--lora-r", "8",
        "--lora-alpha", "16",
    ])
    ckpt = next(ckpt_root.glob("checkpoint-epoch-*"))
    clips: dict = {}
    for ing, clist in available_clips_all().items():
        clips.setdefault(ing.lower(), []).extend(clist)
    human = clips[DRUG]
    model = Qwen3TTSModel.from_pretrained(
        str(ckpt), device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    per = {}
    for item in SENTENCES:
        scores = []
        for d in range(3):
            torch.manual_seed(9000 + d * 13 + sum(map(ord, item["id"])))
            wavs, sr = model.generate_custom_voice(
                text=item["text"], language="English", speaker=SPEAKER,
                temperature=0.3, top_p=0.85, do_sample=True,
            )
            dest = WORK / "eval" / f"{item['id']}_d{d}.wav"
            dest.parent.mkdir(parents=True, exist_ok=True)
            sf.write(dest, wavs[0], sr)
            if dest.stat().st_size > MAX_WAV_BYTES:
                scores.append(0.0)
                continue
            span = extract_drug_span_forced_align(dest.read_bytes(), item["text"], DRUG)
            scores.append(0.0 if span is None else float(score_against_best_reference(span, human).best_f1))
        per[item["id"]] = {
            "draws": [round(s, 4) for s in scores],
            "mean": round(sum(scores) / len(scores), 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
        }
        print(f"  {item['id']}: {per[item['id']]}", flush=True)
    means = [v["mean"] for v in per.values()]
    out = {
        "per_sentence": per,
        "mean_of_means": round(sum(means) / len(means), 4),
        "baseline_full_ft_ep0": {"dose": 0.5259, "prescribed": 0.7318, "confirm": 0.6084, "monitor": 0.742, "mean": 0.652},
        "last4_layers": {"dose": 0.65, "prescribed": 0.709, "confirm": 0.5362, "monitor": 0.6982, "mean": 0.6483},
        "success": all(v["mean"] >= 0.70 for v in per.values()),
    }
    (WORK / "summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
