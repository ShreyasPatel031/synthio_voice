#!/usr/bin/env python3
"""Selective word+sentence fine-tune, worst Qwen-vs-Gemini names only.

What worked before: one full-model pass on a gold word clip plus Gemini
sentence audio (not LoRA, not many epochs, not the whole list).

Start at the lowest CTC F1 versus data/gold_gemini_ipa. Each round adds
one still-failing name, retrains from Base, and stops when that name
reaches 0.70, a new name does not improve, or easy names drop.
Gold files are only read. Sentence teachers are new wavs under runs/.
"""

from __future__ import annotations

import base64
import io
import json
import shutil
import subprocess
import sys
import time
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
FT = Path("/home/shreyaspatel/Qwen3-TTS/finetuning")
if not (FT / "sft_12hz.py").exists():
    FT = Path.home() / "Qwen3-TTS" / "finetuning"

GOLD_MANIFEST = ROOT / "data" / "gold_gemini_ipa" / "manifest.jsonl"
BASE_SCORES = ROOT / "runs" / "dose-ctc-vs-gemini-ipa" / "pron-gemini-base" / "scores.jsonl"
WORK = ROOT / "runs" / "ft-mix-selective"
PASS = 0.70
# Only names this far from the Gemini benchmark. Not the whole list.
MAX_BASE_F1 = 0.50
MAX_IN_SET = 3
ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"
PROJECT = "project-amer-scs-sandbox"
GEMINI_MODEL = "gemini-3.1-flash-tts-preview"
GEMINI_VOICE = "Kore"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


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


def load_gold() -> dict[str, dict]:
    out = {}
    for line in GOLD_MANIFEST.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["ingredient"].lower()] = rec
    return out


def load_ranked() -> list[dict]:
    rows = []
    seen = set()
    for line in BASE_SCORES.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("gemini_f1") is None:
            continue
        key = rec["drug"].lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append(rec)
    rows.sort(key=lambda r: r["gemini_f1"])
    return rows


def gemini_sentence(sentence: str, drug: str, ipa: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 500:
        print("sentence teacher exists", dest.name, flush=True)
        return
    import requests

    tok = subprocess.check_output(
        ["/snap/bin/gcloud", "auth", "print-access-token"], text=True
    ).strip()
    prompt = (
        "Read this clinical sentence naturally. "
        f"Pronounce {drug} using this IPA exactly: /{ipa}/. "
        "Do not spell the name letter by letter."
    )
    body = {
        "input": {
            "text": sentence,
            "prompt": prompt,
            "customPronunciations": {
                "pronunciations": [{
                    "phrase": drug,
                    "phoneticEncoding": "PHONETIC_ENCODING_IPA",
                    "pronunciation": ipa,
                }]
            },
        },
        "voice": {"languageCode": "en-US", "name": GEMINI_VOICE, "modelName": GEMINI_MODEL},
        "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": 24000},
    }
    resp = requests.post(
        ENDPOINT,
        headers={
            "Authorization": f"Bearer {tok}",
            "Content-Type": "application/json",
            "x-goog-user-project": PROJECT,
        },
        json=body,
        timeout=120,
    )
    if resp.status_code != 200:
        raise SystemExit(f"gemini {resp.status_code}: {resp.text[:400]}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(base64.b64decode(resp.json()["audioContent"]))
    print("sentence teacher", drug, dest.stat().st_size, flush=True)


def train(gold: dict, rows: list[dict], round_dir: Path, speaker: str) -> Path:
    teachers = round_dir / "teachers"
    raw, coded, ckpt_root = round_dir / "train_raw.jsonl", round_dir / "train_coded.jsonl", round_dir / "ckpt"
    round_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with raw.open("w") as f:
        for rec in rows:
            g = gold[rec["drug"].lower()]
            word = str((ROOT / g["audio"]).resolve())
            sent = teachers / f"{rec['drug'].lower().replace(' ', '-')}.wav"
            gemini_sentence(rec["sentence"], rec.get("spoken") or rec["drug"], g["ipa"], sent)
            for _ in range(8):
                f.write(json.dumps({"audio": word, "text": g["spoken_text"], "ref_audio": word}) + "\n")
                f.write(json.dumps({"audio": str(sent.resolve()), "text": rec["sentence"], "ref_audio": str(sent.resolve())}) + "\n")
                n += 2
    print(f"train rows={n} (word+sentence x8)", flush=True)
    if coded.exists():
        coded.unlink()
    _run([
        sys.executable, str(FT / "prepare_data.py"),
        "--device", "cuda:0",
        "--tokenizer_model_path", "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        "--input_jsonl", str(raw), "--output_jsonl", str(coded),
    ])
    if ckpt_root.exists():
        shutil.rmtree(ckpt_root)
    ckpt_root.mkdir(parents=True)
    _run([
        sys.executable, str(FT / "sft_12hz.py"),
        "--init_model_path", MODEL_ID,
        "--output_model_path", str(ckpt_root),
        "--train_jsonl", str(coded),
        "--batch_size", "1",
        "--lr", "2e-5",
        "--num_epochs", "1",
        "--speaker_name", speaker,
    ])
    return next(ckpt_root.glob("checkpoint-epoch-*"))


def score_one(model, speaker: str, rec: dict, gold: dict, dest: Path) -> float | None:
    g = gold.get(rec["drug"].lower())
    if g is None:
        return None
    torch.manual_seed(0)
    wavs, sr = model.generate_custom_voice(
        text=rec["sentence"], language="English", speaker=speaker, do_sample=False,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dest, wavs[0], sr)
    if dest.stat().st_size > 2_000_000:
        return None
    gold_path = ROOT / g["audio"]
    spoken = rec.get("spoken") or rec["drug"]
    span = extract_drug_span_forced_align(wav_bytes(dest), rec["sentence"], spoken)
    if span is None:
        return None
    return float(speech_bertscore(
        extract_frame_embeddings(span),
        extract_frame_embeddings(gold_path.read_bytes()),
    )["f1"])


def mean(xs: list[float]) -> float | None:
    return None if not xs else round(sum(xs) / len(xs), 4)


def main() -> None:
    gold = load_gold()
    ranked = load_ranked()
    controls = list(reversed(ranked[-4:]))
    pool = [r for r in ranked if r["gemini_f1"] < MAX_BASE_F1 and r["drug"].lower() in gold]
    print(
        f"pool={len(pool)} worst={[r['drug'] for r in pool[:6]]} pass={PASS} max_in_set={MAX_IN_SET}",
        flush=True,
    )
    base_ctrl = mean([r["gemini_f1"] for r in controls])
    history = []
    active: list[dict] = []
    for step in range(MAX_IN_SET):
        nxt = next((r for r in pool if r["drug"].lower() not in {a["drug"].lower() for a in active}), None)
        if nxt is None:
            print("stop: no names left under the divergence cutoff", flush=True)
            break
        active.append(nxt)
        speaker = f"dose_mix_{len(active)}"
        round_dir = WORK / f"round-{len(active)}"
        names = [r["drug"] for r in active]
        print(f"=== round {step+1} full FT 1 pass, word+sentence: {names} ===", flush=True)
        ckpt = train(gold, active, round_dir, speaker)
        model = Qwen3TTSModel.from_pretrained(
            str(ckpt), device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        trained = []
        for rec in active:
            f1 = score_one(model, speaker, rec, gold, round_dir / "eval" / f"{rec['drug'].lower().replace(' ', '-')}.wav")
            row = {
                "drug": rec["drug"],
                "base_f1": rec["gemini_f1"],
                "ft_f1": None if f1 is None else round(f1, 4),
                "delta": None if f1 is None else round(f1 - rec["gemini_f1"], 4),
                "pass": f1 is not None and f1 >= PASS,
            }
            trained.append(row)
            print(f"  {rec['drug']}: base {rec['gemini_f1']:.3f} -> {row['ft_f1']}", flush=True)
        held = []
        for rec in controls:
            f1 = score_one(model, speaker, rec, gold, round_dir / "eval" / f"ctrl-{rec['drug'].lower().replace(' ', '-')}.wav")
            held.append({
                "drug": rec["drug"],
                "base_f1": rec["gemini_f1"],
                "ft_f1": None if f1 is None else round(f1, 4),
            })
            print(f"  control {rec['drug']}: base {rec['gemini_f1']:.3f} -> {held[-1]['ft_f1']}", flush=True)
        del model
        torch.cuda.empty_cache()
        block = {
            "names": names,
            "trained_mean_base": mean([r["base_f1"] for r in trained]),
            "trained_mean_ft": mean([r["ft_f1"] for r in trained if r["ft_f1"] is not None]),
            "control_mean_base": base_ctrl,
            "control_mean_ft": mean([r["ft_f1"] for r in held if r["ft_f1"] is not None]),
            "trained": trained,
            "controls": held,
        }
        history.append(block)
        (WORK / "curve.json").write_text(json.dumps(history, indent=2))
        print(json.dumps({k: block[k] for k in block if k not in ("trained", "controls")}, indent=2), flush=True)
        new = trained[-1]
        ctrl = block["control_mean_ft"]
        if new["ft_f1"] is None or new["ft_f1"] <= new["base_f1"]:
            print(f"stop: {new['drug']} did not beat the base", flush=True)
            break
        if ctrl is not None and base_ctrl is not None and ctrl < base_ctrl - 0.04:
            print(f"stop: easy names fell {base_ctrl} -> {ctrl}", flush=True)
            break
        if all(r["pass"] for r in trained):
            print("stop: trained names cleared 0.70", flush=True)
            break
        time.sleep(1)
    print("=== curve ===", flush=True)
    print(json.dumps(history, indent=2), flush=True)


if __name__ == "__main__":
    main()
