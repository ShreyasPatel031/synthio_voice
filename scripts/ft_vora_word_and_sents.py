#!/usr/bin/env python3
"""Clean A/B: four Gemini sentences + isolated word, ONE pass, from scratch.

Matches the winning sentence-FT setup (4 lines, 8 repeats, lr 2e-5, batch 1)
and only adds the isolated vorasidenib clip the same number of times.
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
SPEAKER = "dose_word_and_sent"
WORK = ROOT / "runs" / "ft-vora-word-and-sents"
SENT_TTS = ROOT / "runs" / "vora-sent-cloud-vs-gemini" / "tts"
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


def isolated_wav() -> Path:
    for p in (
        ROOT / "runs/gemini31-ipa-vs-cloud/tts/vorasidenib.sidecar.wav",
        ROOT / "runs/verify-google-ipa/tts/vorasidenib.ipa.wav",
        ROOT / "data/gold_gemini_ipa/wavs/vorasidenib.wav",
    ):
        if p.exists():
            return p
    raise SystemExit("no isolated vorasidenib wav")


def build_train(raw: Path, repeats: int) -> dict:
    iso = isolated_wav()
    n_sent = n_word = 0
    raw.parent.mkdir(parents=True, exist_ok=True)
    with raw.open("w") as f:
        for item in SENTENCES:
            wav = SENT_TTS / f"{item['id']}__gemini.wav"
            if not wav.exists():
                raise SystemExit(f"missing {wav}")
            rec = {"audio": str(wav.resolve()), "text": item["text"], "ref_audio": str(wav.resolve())}
            for _ in range(repeats):
                f.write(json.dumps(rec) + "\n")
                n_sent += 1
        rec = {"audio": str(iso.resolve()), "text": DRUG, "ref_audio": str(iso.resolve())}
        for _ in range(repeats):
            f.write(json.dumps(rec) + "\n")
            n_word += 1
    return {"isolated": str(iso), "n_sentence_rows": n_sent, "n_word_rows": n_word, "n": n_sent + n_word}


def main() -> None:
    repeats = 8
    lr = 2e-5
    n_draws = 3
    temperature = 0.3
    WORK.mkdir(parents=True, exist_ok=True)
    raw, coded, ckpt_root = WORK / "train_raw.jsonl", WORK / "train_coded.jsonl", WORK / "ckpt"
    meta = build_train(raw, repeats)
    print(json.dumps(meta, indent=2), flush=True)
    if coded.exists():
        coded.unlink()
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
        sys.executable, str(FT_SCRIPTS / "sft_12hz.py"),
        "--init_model_path", MODEL_ID,
        "--output_model_path", str(ckpt_root),
        "--train_jsonl", str(coded),
        "--batch_size", "1",
        "--lr", str(lr),
        "--num_epochs", "1",
        "--speaker_name", SPEAKER,
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
        for d in range(n_draws):
            torch.manual_seed(8000 + d * 17 + sum(map(ord, item["id"])))
            wavs, sr = model.generate_custom_voice(
                text=item["text"], language="English", speaker=SPEAKER,
                temperature=temperature, top_p=0.85, do_sample=True,
            )
            dest = WORK / "eval" / f"{item['id']}_d{d}.wav"
            dest.parent.mkdir(parents=True, exist_ok=True)
            sf.write(dest, wavs[0], sr)
            if dest.stat().st_size > MAX_WAV_BYTES:
                print(f"  RUNAWAY {item['id']} d{d} bytes={dest.stat().st_size}", flush=True)
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
        "baseline_sent_only_ep0": {"dose": 0.5259, "prescribed": 0.7318, "confirm": 0.6084, "monitor": 0.742, "mean": 0.652},
        "train": meta,
        "success": all(v["mean"] >= 0.70 for v in per.values()),
    }
    (WORK / "summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
