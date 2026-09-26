#!/usr/bin/env python3
"""SFT Base on ONLY the failing sentences (dose+confirm) + isolated teacher.

Heavy repeats, 1-2 epochs, eval all 4 sentences with 3 draws.
Avoids continuing from a custom_voice ckpt (speaker_encoder crash).
"""

from __future__ import annotations

import argparse
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
def _ft_scripts() -> Path:
    candidates = [
        Path.home() / "Qwen3-TTS" / "finetuning",
        Path("/home/shreyaspatel/Qwen3-TTS/finetuning"),
        Path("/home/ubuntu/Qwen3-TTS/finetuning"),
    ]
    for p in candidates:
        try:
            if (p / "sft_12hz.py").exists():
                return p
        except PermissionError:
            continue
    raise SystemExit("Qwen3-TTS/finetuning/sft_12hz.py not found")


FT_SCRIPTS = _ft_scripts()

DRUG = "vorasidenib"
SPEAKER = "dose_hard"
WORK = ROOT / "runs" / "ft-vora-hard-only"
TEACHER_DIR = ROOT / "runs" / "ft-vora-sent-hillclimb" / "teachers"
ISO_WAV = ROOT / "data" / "gold_gemini_ipa" / "wavs" / "vorasidenib.wav"

EVAL = [
    {"id": "dose", "text": "Let's initiate vorasidenib therapy for this patient to target the mutant IDH1 and IDH2 enzymes in the tumor."},
    {"id": "prescribed", "text": "The oncologist prescribed vorasidenib for the patient's IDH-mutant glioma."},
    {"id": "confirm", "text": "Please confirm the vorasidenib dose before the next clinic visit."},
    {"id": "monitor", "text": "Patients taking vorasidenib need monitoring for liver enzyme elevation."},
]
TRAIN_IDS = {"dose", "confirm"}


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def load_model(path) -> Qwen3TTSModel:
    return Qwen3TTSModel.from_pretrained(
        str(path), device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa"
    )


def free(m) -> None:
    del m
    torch.cuda.empty_cache()


def ok(per: dict) -> bool:
    return all(v["mean"] >= 0.70 for v in per.values()) and all(
        v["min"] >= 0.62 for v in per.values()
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=12)
    ap.add_argument("--lr", type=float, default=8e-6)
    ap.add_argument("--n-draws", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.25)
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    raw = WORK / "train_raw.jsonl"
    coded = WORK / "train_coded.jsonl"
    ckpt_root = WORK / "ckpt"

    n = 0
    with raw.open("w") as f:
        for item in EVAL:
            if item["id"] not in TRAIN_IDS:
                continue
            wav = TEACHER_DIR / f"{item['id']}__gemini.wav"
            rec = {"audio": str(wav.resolve()), "text": item["text"], "ref_audio": str(wav.resolve())}
            for _ in range(args.repeats):
                f.write(json.dumps(rec) + "\n")
                n += 1
        if ISO_WAV.exists():
            rec = {"audio": str(ISO_WAV.resolve()), "text": DRUG, "ref_audio": str(ISO_WAV.resolve())}
            for _ in range(args.repeats):
                f.write(json.dumps(rec) + "\n")
                n += 1
    print(f"train rows={n}", flush=True)
    _run(
        [
            sys.executable,
            str(FT_SCRIPTS / "prepare_data.py"),
            "--device",
            "cuda:0",
            "--tokenizer_model_path",
            "Qwen/Qwen3-TTS-Tokenizer-12Hz",
            "--input_jsonl",
            str(raw),
            "--output_jsonl",
            str(coded),
        ]
    )
    if ckpt_root.exists():
        shutil.rmtree(ckpt_root)
    ckpt_root.mkdir(parents=True)
    _run(
        [
            sys.executable,
            str(FT_SCRIPTS / "sft_12hz.py"),
            "--init_model_path",
            MODEL_ID,
            "--output_model_path",
            str(ckpt_root),
            "--train_jsonl",
            str(coded),
            "--batch_size",
            "1",
            "--lr",
            str(args.lr),
            "--num_epochs",
            str(args.epochs),
            "--speaker_name",
            SPEAKER,
        ]
    )

    clips = {}
    for ing, clist in available_clips_all().items():
        clips.setdefault(ing.lower(), []).extend(clist)
    human = clips[DRUG]

    history = []
    best = None
    epochs = sorted(ckpt_root.glob("checkpoint-epoch-*"), key=lambda p: int(p.name.split("-")[-1]))
    for ckpt in epochs:
        ep = int(ckpt.name.split("-")[-1])
        print(f"=== eval epoch {ep} ===", flush=True)
        model = load_model(ckpt)
        per = {}
        for item in EVAL:
            scores = []
            for d in range(args.n_draws):
                seed = 5000 + d * 41 + sum(map(ord, item["id"])) % 1009
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                wavs, sr = model.generate_custom_voice(
                    text=item["text"],
                    language="English",
                    speaker=SPEAKER,
                    temperature=args.temperature,
                    top_p=0.8,
                    do_sample=True,
                )
                dest = WORK / "eval" / f"ep{ep}_{item['id']}_d{d}.wav"
                dest.parent.mkdir(parents=True, exist_ok=True)
                sf.write(dest, wavs[0], sr)
                span = extract_drug_span_forced_align(dest.read_bytes(), item["text"], DRUG)
                scores.append(
                    0.0
                    if span is None
                    else float(score_against_best_reference(span, human).best_f1)
                )
            per[item["id"]] = {
                "draws": [round(s, 4) for s in scores],
                "mean": round(sum(scores) / len(scores), 4),
                "min": round(min(scores), 4),
                "max": round(max(scores), 4),
            }
            print(f"  {item['id']}: {per[item['id']]}", flush=True)
        free(model)
        means = [v["mean"] for v in per.values()]
        stats = {
            "epoch": ep,
            "per_sentence": per,
            "mean_of_means": round(sum(means) / len(means), 4),
            "min_of_means": round(min(means), 4),
            "min_of_mins": round(min(v["min"] for v in per.values()), 4),
            "success": ok(per),
            "ckpt": str(ckpt),
        }
        history.append(stats)
        print(
            f"epoch {ep}: mean={stats['mean_of_means']} min_mean={stats['min_of_means']} success={stats['success']}",
            flush=True,
        )
        if best is None or stats["min_of_means"] > best["min_of_means"]:
            best = stats
            dest = WORK / "best_checkpoint"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(ckpt, dest)

    for ckpt in epochs:
        if best and Path(best["ckpt"]).name != ckpt.name:
            shutil.rmtree(ckpt)
            print("freed", ckpt.name, flush=True)

    summary = {"best": best, "history": history, "success": bool(best and best.get("success"))}
    (WORK / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k != "history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
