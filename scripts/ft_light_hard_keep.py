#!/usr/bin/env python3
"""Light last-4-layer SFT from Base. Two packs, 3 epochs, probe each epoch.

A: 16 names Base misses (lowest official CTC vs gold).
B: those 16 plus 8 names Base already says, using Base's own sentence
   wav as the keep teacher so they do not get overwritten.

lr=2e-6 (Qwen's README default). 4 repeats. Isolated gold word for hard
names. Do not edit data/gold_gemini_ipa. Do not continue from LoRA.
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
FT = Path("/home/shreyaspatel/Qwen3-TTS/finetuning")
try:
    _ft_ok = (FT / "sft_12hz_last_layers.py").exists()
except PermissionError:
    _ft_ok = False
if not _ft_ok:
    FT = Path.home() / "Qwen3-TTS" / "finetuning"

GOLD_MANIFEST = ROOT / "data" / "gold_gemini_ipa" / "manifest.jsonl"
BASE_SCORES = ROOT / "runs" / "dose-ctc-vs-gemini-ipa" / "pron-gemini-base-full" / "scores.jsonl"
BASE_WAV = ROOT / "runs" / "oss-eval" / "pron-gemini-base"
WORK = ROOT / "runs" / "ft-light-hard-keep"
MAX_WAV = 2_000_000
LR = "2e-6"
EPOCHS = 3
REPEATS = 4
UNFREEZE = 4

HARD = [
    "acoramidis", "atogepant", "Lipitor", "Veppanu", "xanomeline",
    "Palynziq", "Revtorpyk", "Meibo", "Claritin", "Bysanti",
    "Obicetrapib", "apremilast", "Cobenfy", "Zycubo", "Qulipta", "aripiprazole",
]
KEEP = [
    "acetaminophen", "omeprazole", "diazepam", "fluticasone propionate",
    "Retatrutide", "Ozempic", "semaglutide", "Tecfidera",
]
HELDOUT = ["Zevaskyn", "gepotidacin", "Xeljanz", "Vyvgart", "Advair", "Eliquis"]


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def slug(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


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


def load_base_rows() -> dict[str, dict]:
    out = {}
    for line in BASE_SCORES.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("gemini_f1") is None:
            continue
        out.setdefault(rec["drug"].lower(), rec)
    return out


def find_base_wav(rec: dict) -> Path | None:
    for key in (slug(rec["drug"]), slug(rec.get("product") or ""), rec.get("dose_id") or ""):
        if not key:
            continue
        p = BASE_WAV / f"{key}.wav"
        if p.exists() and p.stat().st_size > 500:
            return p
    return None


def write_train(path: Path, gold: dict, base_rows: dict, hard: list[str], keep: list[str]) -> int:
    n = 0
    with path.open("w") as f:
        for name in hard:
            g = gold[name.lower()]
            audio = str((ROOT / g["audio"]).resolve())
            row = {"audio": audio, "text": g["spoken_text"], "ref_audio": audio}
            for _ in range(REPEATS):
                f.write(json.dumps(row) + "\n")
                n += 1
        for name in keep:
            rec = base_rows[name.lower()]
            wav = find_base_wav(rec)
            if wav is None:
                raise SystemExit(f"no Base sentence wav for keep name {name}")
            row = {"audio": str(wav.resolve()), "text": rec["sentence"], "ref_audio": str(wav.resolve())}
            for _ in range(REPEATS):
                f.write(json.dumps(row) + "\n")
                n += 1
    return n


def train(raw: Path, coded: Path, ckpt: Path, speaker: str, batch_size: int = 1, epochs: int | None = None) -> None:
    n_ep = EPOCHS if epochs is None else epochs
    if coded.exists():
        coded.unlink()
    _run([
        sys.executable, str(FT / "prepare_data.py"),
        "--device", "cuda:0",
        "--tokenizer_model_path", "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        "--input_jsonl", str(raw), "--output_jsonl", str(coded),
    ])
    if ckpt.exists():
        shutil.rmtree(ckpt)
    ckpt.mkdir(parents=True)
    _run([
        sys.executable, str(FT / "sft_12hz_last_layers.py"),
        "--init_model_path", MODEL_ID,
        "--output_model_path", str(ckpt),
        "--train_jsonl", str(coded),
        "--batch_size", str(batch_size),
        "--lr", LR,
        "--num_epochs", str(n_ep),
        "--speaker_name", speaker,
        "--unfreeze-last-n", str(UNFREEZE),
    ])


def score_rec(model, speaker: str, rec: dict, gold: dict, dest: Path) -> float | None:
    g = gold.get(rec["drug"].lower())
    if g is None:
        return None
    torch.manual_seed(0)
    wavs, sr = model.generate_custom_voice(
        text=rec["sentence"], language="English", speaker=speaker, do_sample=False,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dest, wavs[0], sr)
    if dest.stat().st_size > MAX_WAV:
        return None
    gold_path = ROOT / g["audio"]
    span = extract_drug_span_forced_align(wav_bytes(dest), rec["sentence"], rec.get("spoken") or rec["drug"])
    if span is None:
        return None
    return float(speech_bertscore(
        extract_frame_embeddings(span),
        extract_frame_embeddings(gold_path.read_bytes()),
    )["f1"])


def mean(xs: list[float]) -> float | None:
    return None if not xs else round(sum(xs) / len(xs), 4)


def probe(ckpt: Path, speaker: str, names: list[str], tag: str, gold: dict, base_rows: dict) -> dict:
    model = Qwen3TTSModel.from_pretrained(
        str(ckpt), device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    rows = []
    for name in names:
        rec = base_rows[name.lower()]
        f1 = score_rec(model, speaker, rec, gold, WORK / "probe" / tag / f"{slug(name)}.wav")
        rows.append({
            "drug": rec["drug"],
            "base_f1": rec["gemini_f1"],
            "ft_f1": None if f1 is None else round(f1, 4),
            "delta": None if f1 is None else round(f1 - rec["gemini_f1"], 4),
        })
        print(f"  {tag} {rec['drug']}: {rec['gemini_f1']:.3f} -> {rows[-1]['ft_f1']}", flush=True)
    del model
    torch.cuda.empty_cache()
    ft = [r["ft_f1"] for r in rows if r["ft_f1"] is not None]
    base = [r["base_f1"] for r in rows]
    return {
        "tag": tag,
        "mean_base": mean(base),
        "mean_ft": mean(ft),
        "n_up": sum(1 for r in rows if r["delta"] is not None and r["delta"] > 0.01),
        "n_down": sum(1 for r in rows if r["delta"] is not None and r["delta"] < -0.01),
        "rows": rows,
    }


def run_variant(name: str, speaker: str, hard: list[str], keep: list[str], gold: dict, base_rows: dict) -> list[dict]:
    vdir = WORK / name
    vdir.mkdir(parents=True, exist_ok=True)
    raw, coded, ckpt = vdir / "train_raw.jsonl", vdir / "train_coded.jsonl", vdir / "ckpt"
    n = write_train(raw, gold, base_rows, hard, keep)
    print(f"=== {name} speaker={speaker} rows={n} hard={len(hard)} keep={len(keep)} lr={LR} last-{UNFREEZE} epochs={EPOCHS} ===", flush=True)
    train(raw, coded, ckpt, speaker)
    curve = []
    prev = None
    for ep in range(EPOCHS):
        c = ckpt / f"checkpoint-epoch-{ep}"
        if not c.exists():
            print("missing", c, flush=True)
            break
        print(f"--- probe {name} epoch {ep} ---", flush=True)
        hard_p = probe(c, speaker, hard, f"{name}-ep{ep}-hard", gold, base_rows)
        keep_p = probe(c, speaker, keep, f"{name}-ep{ep}-keep", gold, base_rows) if keep else {"mean_ft": None, "mean_base": None, "n_down": 0, "rows": []}
        hold_p = probe(c, speaker, [h for h in HELDOUT if h.lower() in base_rows], f"{name}-ep{ep}-hold", gold, base_rows)
        block = {
            "variant": name,
            "epoch": ep,
            "ckpt": str(c),
            "hard": {k: hard_p[k] for k in ("mean_base", "mean_ft", "n_up", "n_down")},
            "keep": {k: keep_p[k] for k in ("mean_base", "mean_ft", "n_down")} if keep else None,
            "heldout": {k: hold_p[k] for k in ("mean_base", "mean_ft", "n_up", "n_down")},
            "hard_rows": hard_p["rows"],
            "keep_rows": keep_p.get("rows"),
            "heldout_rows": hold_p["rows"],
        }
        curve.append(block)
        (vdir / "curve.json").write_text(json.dumps(curve, indent=2))
        print(json.dumps({k: block[k] for k in block if "rows" not in k}, indent=2), flush=True)
        score = hard_p["mean_ft"]
        keep_drop = 0.0
        if keep and keep_p["mean_ft"] is not None and keep_p["mean_base"] is not None:
            keep_drop = keep_p["mean_base"] - keep_p["mean_ft"]
        hold_drop = 0.0
        if hold_p["mean_ft"] is not None and hold_p["mean_base"] is not None:
            hold_drop = hold_p["mean_base"] - hold_p["mean_ft"]
        if prev is not None and score is not None and score < prev - 0.01:
            print(f"stop {name}: hard mean fell {prev} -> {score}", flush=True)
            break
        if keep_drop > 0.03:
            print(f"stop {name}: keep names dropped {keep_drop:.3f}", flush=True)
            break
        if hold_drop > 0.03:
            print(f"stop {name}: held-out dropped {hold_drop:.3f}", flush=True)
            break
        prev = score
    return curve


def main() -> None:
    gold = load_gold()
    base_rows = load_base_rows()
    for n in HARD + KEEP + HELDOUT:
        if n.lower() not in base_rows:
            raise SystemExit(f"missing base score {n}")
        if n.lower() not in gold:
            raise SystemExit(f"missing gold {n}")
    WORK.mkdir(parents=True, exist_ok=True)
    all_curve = []
    all_curve.extend(run_variant("A-hard", "dose_light_a", HARD, [], gold, base_rows))
    all_curve.extend(run_variant("B-hard-keep", "dose_light_b", HARD, KEEP, gold, base_rows))
    (WORK / "curve.json").write_text(json.dumps(all_curve, indent=2))
    print("=== all curve ===", flush=True)
    print(json.dumps([{k: b[k] for k in b if "rows" not in k} for b in all_curve], indent=2), flush=True)


if __name__ == "__main__":
    main()
