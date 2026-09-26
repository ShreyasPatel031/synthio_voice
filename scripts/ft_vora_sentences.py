#!/usr/bin/env python3
"""Fine-tune Qwen 1.7B on vorasidenib *sentences* (Gemini+IPA teachers).

Hypothesis: isolated-word D FT fails in-sentence (CTC span human 0.623).
Training on full clinical sentences with the better in-sentence teacher
(Gemini 3.1 + source IPA) should recover CTC-cut Path-2 on the DoSE carrier.

Train: sentence text → Gemini sentence wav; ref_audio = same wav.
Infer: custom_voice speaker=dose_sent on DoSE carrier; CTC-cut vorasidenib.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
FT_SCRIPTS = Path("/home/shreyaspatel/Qwen3-TTS/finetuning")
if not (FT_SCRIPTS / "sft_12hz.py").exists():
    FT_SCRIPTS = Path.home() / "Qwen3-TTS" / "finetuning"

SENT_TTS = ROOT / "runs" / "vora-sent-cloud-vs-gemini" / "tts"
WORK = ROOT / "runs" / "ft-vora-sentences"
DRUG = "vorasidenib"
SPEAKER = "dose_sent"
DOSE_SENTENCE = (
    "Let's initiate vorasidenib therapy for this patient to target "
    "the mutant IDH1 and IDH2 enzymes in the tumor."
)

SENTENCES = [
    {
        "id": "dose",
        "text": DOSE_SENTENCE,
        "audio": SENT_TTS / "dose__gemini.wav",
    },
    {
        "id": "prescribed",
        "text": "The oncologist prescribed vorasidenib for the patient's IDH-mutant glioma.",
        "audio": SENT_TTS / "prescribed__gemini.wav",
    },
    {
        "id": "confirm",
        "text": "Please confirm the vorasidenib dose before the next clinic visit.",
        "audio": SENT_TTS / "confirm__gemini.wav",
    },
    {
        "id": "monitor",
        "text": "Patients taking vorasidenib need monitoring for liver enzyme elevation.",
        "audio": SENT_TTS / "monitor__gemini.wav",
    },
]


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def _wav_bytes(path: Path) -> bytes:
    import soundfile as sf

    audio, sr = sf.read(path, dtype="int16", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype("int16")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())
    return buf.getvalue()


def build_train(raw: Path, repeats: int, include_isolated: bool) -> int:
    raw.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with raw.open("w") as f:
        for item in SENTENCES:
            audio = item["audio"]
            if not audio.exists():
                raise SystemExit(f"missing teacher wav: {audio}")
            rec = {
                "audio": str(audio.resolve()),
                "text": item["text"],
                "ref_audio": str(audio.resolve()),
            }
            for _ in range(repeats):
                f.write(json.dumps(rec) + "\n")
                n += 1
        if include_isolated:
            iso = ROOT / "data/gold_gemini_ipa/wavs/vorasidenib.wav"
            # Prefer Gemini isolated if present; else Cloud IPA teacher
            gem_iso = None
            for p in (
                ROOT / "runs/gemini31-ipa-vs-cloud/tts/vorasidenib.sidecar.wav",
                ROOT / "runs/verify-google-ipa/tts/vorasidenib.ipa.wav",
            ):
                if p.exists():
                    gem_iso = p
                    break
            target = gem_iso or iso
            rec = {
                "audio": str(target.resolve()),
                "text": DRUG,
                "ref_audio": str(target.resolve()),
            }
            for _ in range(repeats):
                f.write(json.dumps(rec) + "\n")
                n += 1
    return n


def prepare_and_train(
    raw: Path,
    coded: Path,
    ckpt_root: Path,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
) -> Path:
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
            str(batch_size),
            "--lr",
            str(lr),
            "--num_epochs",
            str(epochs),
            "--speaker_name",
            SPEAKER,
        ]
    )
    epochs_dirs = sorted(ckpt_root.glob("checkpoint-epoch-*"))
    # keep first, mid, last only
    keep = set()
    if epochs_dirs:
        keep.add(epochs_dirs[0].name)
        keep.add(epochs_dirs[len(epochs_dirs) // 2].name)
        keep.add(epochs_dirs[-1].name)
    for d in epochs_dirs:
        if d.name not in keep:
            shutil.rmtree(d)
    return sorted(ckpt_root.glob("checkpoint-epoch-*"))[-1]


def synth_carrier(ckpt: Path, out_wav: Path) -> None:
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    model = Qwen3TTSModel.from_pretrained(
        str(ckpt),
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    wavs, sr = model.generate_custom_voice(
        text=DOSE_SENTENCE, language="English", speaker=SPEAKER
    )
    sf.write(out_wav, wavs[0], sr)
    print("wrote", out_wav, flush=True)
    del model
    torch.cuda.empty_cache()


def score_carrier(carrier_wav: Path, isolated_ref: Path | None) -> dict:
    sys.path.insert(0, str(ROOT))
    from dose_r.forced_align import extract_drug_span_forced_align
    from dose_r.references.reference_clips import available_clips_all
    from dose_r.scoring.candidate_eval import score_against_best_reference
    from dose_r.scoring.speech_similarity import extract_frame_embeddings, speech_bertscore

    clips = {}
    for ing, clist in available_clips_all().items():
        clips.setdefault(ing.lower(), []).extend(clist)
    human = clips[DRUG]
    teacher = ROOT / "data/gold_gemini_ipa/wavs/vorasidenib.wav"
    # also score vs the Gemini dose sentence span teacher if available
    gem_full = SENT_TTS / "dose__gemini.wav"
    gem_span_path = SENT_TTS / "dose__gemini_span.wav"

    raw = carrier_wav.read_bytes()
    span = extract_drug_span_forced_align(raw, DOSE_SENTENCE, DRUG)
    if span is None:
        raise SystemExit("CTC align failed on FT carrier")
    span_path = carrier_wav.parent / "vora_ft_sent_carrier_span.wav"
    span_path.write_bytes(span)

    emb_t = extract_frame_embeddings(teacher.read_bytes())
    best = score_against_best_reference(span, human)
    t_f1 = float(speech_bertscore(extract_frame_embeddings(span), emb_t)["f1"])
    out = {
        "cond": "ft_sentence_carrier_ctc",
        "human_f1": round(best.best_f1, 4),
        "teacher_isolated_f1": round(t_f1, 4),
        "span_wav": str(span_path),
        "carrier_wav": str(carrier_wav),
    }
    if gem_span_path.exists():
        g_f1 = float(
            speech_bertscore(
                extract_frame_embeddings(span),
                extract_frame_embeddings(gem_span_path.read_bytes()),
            )["f1"]
        )
        out["teacher_gemini_span_f1"] = round(g_f1, 4)
    elif gem_full.exists():
        gspan = extract_drug_span_forced_align(gem_full.read_bytes(), DOSE_SENTENCE, DRUG)
        if gspan is not None:
            gem_span_path.write_bytes(gspan)
            g_f1 = float(
                speech_bertscore(
                    extract_frame_embeddings(span),
                    extract_frame_embeddings(gspan),
                )["f1"]
            )
            out["teacher_gemini_span_f1"] = round(g_f1, 4)

    # baseline numbers from prior D isolated FT (hardcoded for comparison file)
    out["prior_d_isolated_human"] = 0.7901
    out["prior_d_carrier_ctc_human"] = 0.6229
    out["prior_gemini_dose_ctc_human"] = 0.7495
    out["delta_vs_prior_d_carrier"] = round(out["human_f1"] - 0.6229, 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--include-isolated", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--work", default=str(WORK))
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    raw = work / "train_raw.jsonl"
    coded = work / "train_with_codes.jsonl"
    ckpt_root = work / "checkpoint"

    n = build_train(raw, args.repeats, args.include_isolated)
    print(f"train rows={n} include_isolated={args.include_isolated}", flush=True)
    meta = {
        "epochs": args.epochs,
        "repeats": args.repeats,
        "lr": args.lr,
        "speaker": SPEAKER,
        "teachers": "gemini-3.1 + IPA sentence wavs",
        "n_rows": n,
        "include_isolated": args.include_isolated,
    }
    (work / "meta.json").write_text(json.dumps(meta, indent=2))

    if not args.skip_train:
        last = prepare_and_train(
            raw,
            coded,
            ckpt_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
    else:
        last = sorted(ckpt_root.glob("checkpoint-epoch-*"))[-1]
    print("ckpt", last, flush=True)

    carrier = work / "vora_ft_sent_carrier.wav"
    synth_carrier(last, carrier)
    scores = score_carrier(carrier, None)
    (work / "carrier_scores.json").write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2), flush=True)


if __name__ == "__main__":
    main()
