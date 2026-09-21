#!/usr/bin/env python3
"""1-epoch sentence FT from Base, aimed at dose/confirm without later-epoch collapse.

Winner so far: 4 Gemini sentences, 1 epoch, custom_voice. Dose/confirm still fail.
This run keeps 1 epoch and changes only the train set:
  - all 12 Gemini sentences (more contexts for the name)
  - extra copies of dose + confirm
  - CTC-cut Gemini word spans as extra rows (text=vorasidenib)

Eval is text-only custom_voice, 3 draws, vs sentence-FT ep0.
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
FT_SCRIPTS = Path("/home/shreyaspatel/Qwen3-TTS/finetuning")
if not (FT_SCRIPTS / "sft_12hz.py").exists():
    FT_SCRIPTS = Path.home() / "Qwen3-TTS" / "finetuning"

DRUG = "vorasidenib"
SPEAKER = "dose_sent_e1"
WORK = ROOT / "runs" / "ft-vora-one-epoch"
TEACHER_DIR = ROOT / "runs" / "ft-vora-sent-hillclimb" / "teachers"
SPAN_DIR = ROOT / "runs" / "vora-sent-cloud-vs-gemini" / "tts"
EVAL_IDS = ["dose", "prescribed", "confirm", "monitor"]
HARD_IDS = {"dose", "confirm"}

ALL_SENTENCES = [
    {"id": "dose", "text": "Let's initiate vorasidenib therapy for this patient to target the mutant IDH1 and IDH2 enzymes in the tumor."},
    {"id": "prescribed", "text": "The oncologist prescribed vorasidenib for the patient's IDH-mutant glioma."},
    {"id": "confirm", "text": "Please confirm the vorasidenib dose before the next clinic visit."},
    {"id": "monitor", "text": "Patients taking vorasidenib need monitoring for liver enzyme elevation."},
    {"id": "start", "text": "We will start vorasidenib this week if labs remain stable."},
    {"id": "discuss", "text": "I want to discuss the risks and benefits of vorasidenib with you today."},
    {"id": "oral", "text": "Vorasidenib is taken by mouth once daily with or without food."},
    {"id": "switch", "text": "If side effects worsen we may hold vorasidenib and reassess."},
    {"id": "idh", "text": "Because the tumor carries an IDH mutation, vorasidenib is a reasonable option."},
    {"id": "pharmacy", "text": "Please counsel the patient on how to store and take vorasidenib correctly."},
    {"id": "followup", "text": "At follow-up we will review imaging and decide whether to continue vorasidenib."},
    {"id": "combo", "text": "Do not combine vorasidenib with strong CYP inducers without checking interactions."},
]


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def _span_wav(item: dict) -> Path | None:
    existing = SPAN_DIR / f"{item['id']}__gemini_span.wav"
    if existing.exists():
        return existing
    full = TEACHER_DIR / f"{item['id']}__gemini.wav"
    if not full.exists():
        return None
    span = extract_drug_span_forced_align(full.read_bytes(), item["text"], DRUG)
    if span is None:
        return None
    dest = WORK / "spans" / f"{item['id']}_span.wav"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(span)
    return dest


def build_train(raw: Path, repeats: int, hard_mult: int) -> int:
    raw.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with raw.open("w") as f:
        for item in ALL_SENTENCES:
            wav = TEACHER_DIR / f"{item['id']}__gemini.wav"
            if not wav.exists():
                raise SystemExit(f"missing {wav}")
            rec = {
                "audio": str(wav.resolve()),
                "text": item["text"],
                "ref_audio": str(wav.resolve()),
            }
            r = repeats * (hard_mult if item["id"] in HARD_IDS else 1)
            for _ in range(r):
                f.write(json.dumps(rec) + "\n")
                n += 1
            span = _span_wav(item)
            if span is not None:
                rec_w = {
                    "audio": str(span.resolve()),
                    "text": DRUG,
                    "ref_audio": str(span.resolve()),
                }
                wr = repeats * (hard_mult if item["id"] in HARD_IDS else 1)
                for _ in range(wr):
                    f.write(json.dumps(rec_w) + "\n")
                    n += 1
    return n


def load_model(path: Path) -> Qwen3TTSModel:
    return Qwen3TTSModel.from_pretrained(
        str(path),
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )


def eval_ckpt(ckpt: Path, *, n_draws: int, temperature: float, human) -> dict:
    model = load_model(ckpt)
    per = {}
    for item in ALL_SENTENCES:
        if item["id"] not in EVAL_IDS:
            continue
        scores = []
        for draw in range(n_draws):
            seed = 7000 + draw * 31 + (sum(map(ord, item["id"])) % 1009)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            wavs, sr = model.generate_custom_voice(
                text=item["text"],
                language="English",
                speaker=SPEAKER,
                temperature=temperature,
                top_p=0.85,
                do_sample=True,
            )
            dest = WORK / "eval" / f"{item['id']}_d{draw}.wav"
            dest.parent.mkdir(parents=True, exist_ok=True)
            sf.write(dest, wavs[0], sr)
            span = extract_drug_span_forced_align(dest.read_bytes(), item["text"], DRUG)
            if span is None:
                scores.append(0.0)
                continue
            span_path = dest.with_name(dest.stem + "_span.wav")
            span_path.write_bytes(span)
            scores.append(float(score_against_best_reference(span, human).best_f1))
        per[item["id"]] = {
            "draws": [round(s, 4) for s in scores],
            "mean": round(sum(scores) / len(scores), 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
        }
        print(f"  {item['id']}: {per[item['id']]}", flush=True)
    del model
    torch.cuda.empty_cache()
    means = [v["mean"] for v in per.values()]
    return {
        "per_sentence": per,
        "mean_of_means": round(sum(means) / len(means), 4),
        "min_of_means": round(min(means), 4),
        "min_of_mins": round(min(v["min"] for v in per.values()), 4),
    }


def ok(stats: dict) -> bool:
    return all(v["mean"] >= 0.70 for v in stats["per_sentence"].values()) and all(
        v["min"] >= 0.62 for v in stats["per_sentence"].values()
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--hard-mult", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--n-draws", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.3)
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    raw = WORK / "train_raw.jsonl"
    coded = WORK / "train_coded.jsonl"
    ckpt_root = WORK / "ckpt"

    if coded.exists() and coded.stat().st_size > 100:
        n = sum(1 for _ in coded.open() if _.strip())
        print(f"reuse coded rows={n}", flush=True)
    else:
        n = build_train(raw, args.repeats, args.hard_mult)
        print(f"train rows={n} epochs=1 lr={args.lr} batch_size={args.batch_size}", flush=True)
        if coded.exists():
            coded.unlink()
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
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--num_epochs",
            "1",
            "--speaker_name",
            SPEAKER,
        ]
    )
    ckpt = next(ckpt_root.glob("checkpoint-epoch-*"))
    clips: dict = {}
    for ing, clist in available_clips_all().items():
        clips.setdefault(ing.lower(), []).extend(clist)
    print("=== eval epoch 0 vs sent_ft_ep0 ===", flush=True)
    stats = eval_ckpt(
        ckpt, n_draws=args.n_draws, temperature=args.temperature, human=clips[DRUG]
    )
    stats["success"] = ok(stats)
    stats["baseline_sent_ft_ep0"] = {
        "dose": 0.5259,
        "prescribed": 0.7318,
        "confirm": 0.6084,
        "monitor": 0.742,
        "mean": 0.652,
    }
    (WORK / "summary.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
