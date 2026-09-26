#!/usr/bin/env python3
"""LoRA the hardest base-Qwen names, a few at a time, from Base each round.

Hardest = lowest CTC F1 on runs/dose-ctc-vs-gemini-ipa/pron-gemini-base
(official gold, data/gold_gemini_ipa). This script only reads that gold.

Each round trains one epoch of LoRA on the current bottom-K isolated gold
wavs (spelling as text, never IPA). Then it resynthesizes those DoSE
sentences plus a few names the base already said well, and CTC-scores them
the same way. Next round grows K only if the trained names' mean went up
and the easy names did not fall apart.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dose_r.forced_align import extract_drug_span_forced_align
from dose_r.scoring.speech_similarity import extract_frame_embeddings, speech_bertscore

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
FT_SCRIPTS = Path("/home/shreyaspatel/Qwen3-TTS/finetuning")
if not (FT_SCRIPTS / "sft_12hz_lora.py").exists():
    FT_SCRIPTS = Path.home() / "Qwen3-TTS" / "finetuning"

GOLD_MANIFEST = ROOT / "data" / "gold_gemini_ipa" / "manifest.jsonl"
BASE_SCORES = ROOT / "runs" / "dose-ctc-vs-gemini-ipa" / "pron-gemini-base" / "scores.jsonl"
WORK = ROOT / "runs" / "ft-lora-hard-words"
MAX_WAV_BYTES = 2_000_000
ROUNDS = (8, 16, 24)
N_CONTROLS = 4


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def load_gold() -> dict[str, dict]:
    out = {}
    for line in GOLD_MANIFEST.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["ingredient"].lower()] = rec
    return out


def load_base_rows() -> list[dict]:
    rows = []
    for line in BASE_SCORES.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("gemini_f1") is None:
            continue
        rows.append(rec)
    rows.sort(key=lambda r: r["gemini_f1"])
    return rows


def unique_by_drug(rows: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for rec in rows:
        key = rec["drug"].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def wav_bytes(path: Path) -> bytes:
    audio, sr = sf.read(path, dtype="int16", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1).astype("int16")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())
    return buf.getvalue()


def score_sentence(wav: Path, sentence: str, spoken: str, gold: Path) -> float | None:
    span = extract_drug_span_forced_align(wav_bytes(wav), sentence, spoken)
    if span is None:
        span = extract_drug_span_forced_align(wav_bytes(wav), sentence, spoken.split()[-1])
    if span is None:
        return None
    emb_c = extract_frame_embeddings(span)
    emb_g = extract_frame_embeddings(gold.read_bytes())
    return float(speech_bertscore(emb_c, emb_g)["f1"])


def train_round(gold: dict, names: list[str], round_dir: Path, speaker: str) -> Path:
    raw = round_dir / "train_raw.jsonl"
    coded = round_dir / "train_coded.jsonl"
    ckpt_root = round_dir / "ckpt"
    round_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with raw.open("w") as f:
        for name in names:
            rec = gold[name.lower()]
            audio = str((ROOT / rec["audio"]).resolve())
            row = {"audio": audio, "text": rec["spoken_text"], "ref_audio": audio}
            for _ in range(16):
                f.write(json.dumps(row) + "\n")
                n += 1
    print(f"train rows={n} names={len(names)}", flush=True)
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
        sys.executable, str(FT_SCRIPTS / "sft_12hz_lora.py"),
        "--init_model_path", MODEL_ID,
        "--output_model_path", str(ckpt_root),
        "--train_jsonl", str(coded),
        "--batch_size", "8",
        "--lr", "2e-5",
        "--num_epochs", "1",
        "--speaker_name", speaker,
        "--lora-r", "8",
        "--lora-alpha", "16",
    ])
    return next(ckpt_root.glob("checkpoint-epoch-*"))


def eval_rows(model, speaker: str, rows: list[dict], out_dir: Path, gold: dict) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    scored = []
    for rec in rows:
        drug = rec["drug"]
        rec_gold = gold.get(drug.lower())
        gold_path = ROOT / rec_gold["audio"] if rec_gold else None
        dest = out_dir / f"{rec['dose_id']}.wav"
        torch.manual_seed(0)
        wavs, sr = model.generate_custom_voice(
            text=rec["sentence"],
            language="English",
            speaker=speaker,
            do_sample=False,
        )
        sf.write(dest, wavs[0], sr)
        f1 = None
        err = None
        if dest.stat().st_size > MAX_WAV_BYTES:
            err = "runaway"
        elif gold_path is None or not gold_path.exists():
            err = "no_gold"
        else:
            try:
                f1 = score_sentence(dest, rec["sentence"], rec.get("spoken") or drug, gold_path)
            except Exception as exc:
                err = str(exc)
        scored.append({
            "drug": drug,
            "base_f1": rec["gemini_f1"],
            "lora_f1": None if f1 is None else round(f1, 4),
            "delta": None if f1 is None else round(f1 - rec["gemini_f1"], 4),
            "error": err,
        })
        print(f"  {drug}: base {rec['gemini_f1']:.3f} -> {scored[-1]['lora_f1']} ({scored[-1]['delta']})", flush=True)
    return scored


def mean_of(rows: list[dict], key: str) -> float | None:
    xs = [r[key] for r in rows if r.get(key) is not None]
    return None if not xs else round(sum(xs) / len(xs), 4)


def main() -> None:
    gold = load_gold()
    ranked = unique_by_drug(load_base_rows())
    controls = list(reversed(ranked[-N_CONTROLS:]))
    control_keys = {r["drug"].lower() for r in controls}
    pool = [r for r in ranked if r["drug"].lower() not in control_keys and r["drug"].lower() in gold]
    history = []
    prev_train_mean = None
    base_control_mean = mean_of([{"x": r["gemini_f1"]} for r in controls], "x")

    for i, k in enumerate(ROUNDS):
        train_rows = pool[:k]
        names = [r["drug"] for r in train_rows]
        missing = [n for n in names if n.lower() not in gold]
        if missing:
            raise SystemExit(f"not in gold manifest: {missing}")
        speaker = f"dose_lora_k{k}"
        round_dir = WORK / f"round-k{k}"
        print(f"=== round {i+1} train {k}: {names} ===", flush=True)
        ckpt = train_round(gold, names, round_dir, speaker)
        model = Qwen3TTSModel.from_pretrained(
            str(ckpt), device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        print("trained", flush=True)
        trained = eval_rows(model, speaker, train_rows, round_dir / "eval-trained", gold)
        print("controls", flush=True)
        held = eval_rows(model, speaker, controls, round_dir / "eval-controls", gold)
        del model
        torch.cuda.empty_cache()
        block = {
            "k": k,
            "trained_mean_base": mean_of(trained, "base_f1"),
            "trained_mean_lora": mean_of(trained, "lora_f1"),
            "control_mean_base": mean_of(held, "base_f1"),
            "control_mean_lora": mean_of(held, "lora_f1"),
            "trained": trained,
            "controls": held,
        }
        history.append(block)
        (WORK / "curve.json").write_text(json.dumps(history, indent=2))
        print(json.dumps({k2: block[k2] for k2 in block if k2 not in ("trained", "controls")}, indent=2), flush=True)
        got = block["trained_mean_lora"]
        if got is None:
            print("stop: no scores", flush=True)
            break
        if prev_train_mean is not None and got <= prev_train_mean:
            print(f"stop: trained mean {got} did not beat previous {prev_train_mean}", flush=True)
            break
        if block["trained_mean_base"] is not None and got <= block["trained_mean_base"]:
            print("stop: did not beat the untouched base on these names", flush=True)
            break
        ctrl = block["control_mean_lora"]
        if ctrl is not None and base_control_mean is not None and ctrl < base_control_mean - 0.05:
            print(f"stop: easy names fell {base_control_mean} -> {ctrl}", flush=True)
            break
        prev_train_mean = got

    print("=== curve ===", flush=True)
    print(json.dumps(history, indent=2), flush=True)


if __name__ == "__main__":
    main()
