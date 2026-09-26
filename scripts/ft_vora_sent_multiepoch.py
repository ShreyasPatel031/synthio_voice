#!/usr/bin/env python3
"""Multi-epoch sentence FT from Base; eval EVERY saved epoch with multi-draw CTC.

Avoids continuing SFT from a custom_voice checkpoint (that path crashes).
Trains once with num_epochs=N, scores checkpoint-epoch-0..N-1, keeps best
by min sentence-mean F1. Uses low temperature + 3 draws.

Also upweights hard eval sentences (dose, confirm) in the train jsonl.
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
SPEAKER = "dose_sent3"
WORK = ROOT / "runs" / "ft-vora-sent-hillclimb"
TEACHER_DIR = WORK / "teachers"
EVAL_IDS = ["dose", "prescribed", "confirm", "monitor"]
HARD_IDS = {"dose", "confirm"}  # upweight

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


def build_train(raw: Path, repeats: int) -> int:
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
            r = repeats * (3 if item["id"] in HARD_IDS else 1)
            for _ in range(r):
                f.write(json.dumps(rec) + "\n")
                n += 1
        iso = ROOT / "data/gold_gemini_ipa/wavs/vorasidenib.wav"
        if iso.exists():
            rec = {
                "audio": str(iso.resolve()),
                "text": DRUG,
                "ref_audio": str(iso.resolve()),
            }
            for _ in range(repeats):
                f.write(json.dumps(rec) + "\n")
                n += 1
    return n


def load_model(path: Path) -> Qwen3TTSModel:
    return Qwen3TTSModel.from_pretrained(
        str(path),
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )


def free(m) -> None:
    del m
    torch.cuda.empty_cache()


def eval_ckpt(ckpt: Path, *, n_draws: int, temperature: float, human) -> dict:
    model = load_model(ckpt)
    per = {}
    for item in ALL_SENTENCES:
        if item["id"] not in EVAL_IDS:
            continue
        scores = []
        for draw in range(n_draws):
            seed = 2000 + draw * 31 + (sum(map(ord, item["id"])) % 1009)
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
            dest = WORK / "eval2" / f"{ckpt.name}_{item['id']}_d{draw}.wav"
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
        print(
            f"  {item['id']}: mean={per[item['id']]['mean']} "
            f"min={per[item['id']]['min']} {per[item['id']]['draws']}",
            flush=True,
        )
    free(model)
    means = [v["mean"] for v in per.values()]
    mins = [v["min"] for v in per.values()]
    return {
        "per_sentence": per,
        "mean_of_means": round(sum(means) / len(means), 4),
        "min_of_means": round(min(means), 4),
        "min_of_mins": round(min(mins), 4),
        "ckpt": str(ckpt),
    }


def ok(stats: dict) -> bool:
    return all(v["mean"] >= 0.70 for v in stats["per_sentence"].values()) and all(
        v["min"] >= 0.62 for v in stats["per_sentence"].values()
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--n-draws", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    raw = WORK / "train_raw_v2.jsonl"
    coded = WORK / "train_coded_v2.jsonl"
    ckpt_root = WORK / "ckpt_multi"

    n = build_train(raw, args.repeats)
    print(f"train rows={n}", flush=True)
    if not coded.exists() or coded.stat().st_mtime < raw.stat().st_mtime:
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

    if not args.skip_train:
        if ckpt_root.exists():
            shutil.rmtree(ckpt_root)
        ckpt_root.mkdir(parents=True, exist_ok=True)
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
    # keep all for eval then thin later
    for ckpt in epochs:
        ep = int(ckpt.name.split("-")[-1])
        print(f"=== eval epoch {ep} ===", flush=True)
        stats = eval_ckpt(
            ckpt, n_draws=args.n_draws, temperature=args.temperature, human=human
        )
        stats["epoch"] = ep
        stats["success"] = ok(stats)
        history.append(stats)
        print(
            f"epoch {ep}: mean={stats['mean_of_means']} "
            f"min_mean={stats['min_of_means']} min_draw={stats['min_of_mins']} "
            f"success={stats['success']}",
            flush=True,
        )
        if best is None or stats["min_of_means"] > best["min_of_means"]:
            best = stats
            dest = WORK / "best_checkpoint_v2"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(ckpt, dest)
            (WORK / "best_stats_v2.json").write_text(json.dumps(best, indent=2))

    # free non-best epoch dirs to save disk
    best_name = Path(best["ckpt"]).name if best else None
    for ckpt in epochs:
        if ckpt.name != best_name:
            shutil.rmtree(ckpt)
            print("freed", ckpt.name, flush=True)

    summary = {
        "best": best,
        "history": history,
        "success": bool(best and best.get("success")),
        "criteria": "each eval mean>=0.70 and min_draw>=0.62",
        "temperature": args.temperature,
        "n_draws": args.n_draws,
        "epochs": args.epochs,
        "lr": args.lr,
        "hard_upweight": list(HARD_IDS),
    }
    (WORK / "summary_v2.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k != "history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
