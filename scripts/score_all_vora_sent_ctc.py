#!/usr/bin/env python3
"""CTC-cut Path-2 on all 4 vorasidenib sentences for:
  - Gemini+IPA teacher (already synthesized)
  - Cloud+IPA teacher (already synthesized)
  - Prior D isolated FT (custom_voice dose_teacher)
  - Sentence FT epoch 0 (custom_voice dose_sent)
"""

from __future__ import annotations

import json
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
from dose_r.scoring.speech_similarity import extract_frame_embeddings, speech_bertscore

DRUG = "vorasidenib"
TTS_TEACHERS = ROOT / "runs/vora-sent-cloud-vs-gemini/tts"
OUT = ROOT / "runs/ft-vora-sentences/all-sent-ctc"
SENT_CKPT = ROOT / "runs/ft-vora-sentences/checkpoint/checkpoint-epoch-0"
D_CKPT = ROOT / "runs/teacher-match-recipes/D_sft_text/checkpoint/checkpoint-epoch-9"
# D kept epochs 0,4,9 — use last trained

SENTENCES = [
    {
        "id": "dose",
        "text": "Let's initiate vorasidenib therapy for this patient to target the mutant IDH1 and IDH2 enzymes in the tumor.",
    },
    {
        "id": "prescribed",
        "text": "The oncologist prescribed vorasidenib for the patient's IDH-mutant glioma.",
    },
    {
        "id": "confirm",
        "text": "Please confirm the vorasidenib dose before the next clinic visit.",
    },
    {
        "id": "monitor",
        "text": "Patients taking vorasidenib need monitoring for liver enzyme elevation.",
    },
]


def load_model(path: Path) -> Qwen3TTSModel:
    return Qwen3TTSModel.from_pretrained(
        str(path),
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )


def free(model) -> None:
    del model
    torch.cuda.empty_cache()


def synth_all(ckpt: Path, speaker: str, tag: str) -> None:
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(ckpt)
    for item in SENTENCES:
        dest = out / f"{item['id']}.wav"
        if dest.exists() and dest.stat().st_size > 500:
            print("skip", tag, item["id"], flush=True)
            continue
        wavs, sr = model.generate_custom_voice(
            text=item["text"], language="English", speaker=speaker
        )
        sf.write(dest, wavs[0], sr)
        print("ok", tag, item["id"], flush=True)
    free(model)


def score_arm(tag: str, wav_dir: Path, human, teacher_emb) -> list[dict]:
    rows = []
    for item in SENTENCES:
        sid = item["id"]
        path = wav_dir / f"{sid}.wav"
        # teachers live as sid__cloud.wav / sid__gemini.wav
        if not path.exists():
            alt = wav_dir / f"{sid}__{tag}.wav"
            path = alt if alt.exists() else path
        if not path.exists():
            rows.append({"id": sid, "arm": tag, "error": "missing"})
            continue
        raw = path.read_bytes()
        span = extract_drug_span_forced_align(raw, item["text"], DRUG)
        if span is None:
            rows.append({"id": sid, "arm": tag, "error": "align_none"})
            print(tag, sid, "align fail", flush=True)
            continue
        span_path = OUT / tag / f"{sid}_span.wav"
        span_path.parent.mkdir(parents=True, exist_ok=True)
        span_path.write_bytes(span)
        best = score_against_best_reference(span, human)
        t_f1 = float(
            speech_bertscore(extract_frame_embeddings(span), teacher_emb)["f1"]
        )
        row = {
            "id": sid,
            "arm": tag,
            "human_f1": round(best.best_f1, 4),
            "vs_isolated_teacher_f1": round(t_f1, 4),
            "sentence": item["text"],
        }
        rows.append(row)
        print(f"{tag} {sid}: human={row['human_f1']}", flush=True)
    return rows


def mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # resolve D ckpt (latest kept)
    d_ckpts = sorted(
        (ROOT / "runs/teacher-match-recipes/D_sft_text/checkpoint").glob(
            "checkpoint-epoch-*"
        )
    )
    if not d_ckpts:
        raise SystemExit("missing D checkpoint")
    d_ckpt = d_ckpts[-1]
    if not SENT_CKPT.exists():
        raise SystemExit(f"missing sentence FT epoch-0: {SENT_CKPT}")

    print("D ckpt", d_ckpt, flush=True)
    print("sent ckpt", SENT_CKPT, flush=True)

    synth_all(d_ckpt, "dose_teacher", "prior_d")
    synth_all(SENT_CKPT, "dose_sent", "sent_ft_ep0")

    # copy teacher wavs into OUT layout for scoring
    for arm in ("cloud", "gemini"):
        d = OUT / arm
        d.mkdir(parents=True, exist_ok=True)
        for item in SENTENCES:
            src = TTS_TEACHERS / f"{item['id']}__{arm}.wav"
            dest = d / f"{item['id']}.wav"
            if src.exists() and not dest.exists():
                dest.write_bytes(src.read_bytes())

    clips = {}
    for ing, clist in available_clips_all().items():
        clips.setdefault(ing.lower(), []).extend(clist)
    human = clips[DRUG]
    teacher_emb = extract_frame_embeddings(
        (ROOT / "data/gold_gemini_ipa/wavs/vorasidenib.wav").read_bytes()
    )

    all_rows = []
    for tag in ("cloud", "gemini", "prior_d", "sent_ft_ep0"):
        all_rows.extend(score_arm(tag, OUT / tag, human, teacher_emb))

    by_arm: dict[str, list] = {}
    for r in all_rows:
        if r.get("human_f1") is None:
            continue
        by_arm.setdefault(r["arm"], []).append(r)

    summary = {
        "drug": DRUG,
        "n_sentences": len(SENTENCES),
        "by_arm": {
            arm: {
                "mean_human_f1": mean([r["human_f1"] for r in rows]),
                "mean_vs_isolated_teacher": mean(
                    [r["vs_isolated_teacher_f1"] for r in rows]
                ),
                "per_sentence": {
                    r["id"]: {
                        "human_f1": r["human_f1"],
                        "vs_isolated_teacher_f1": r["vs_isolated_teacher_f1"],
                    }
                    for r in rows
                },
            }
            for arm, rows in by_arm.items()
        },
        "rows": all_rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["by_arm"], indent=2), flush=True)


if __name__ == "__main__":
    main()
