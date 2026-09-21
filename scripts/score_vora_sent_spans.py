#!/usr/bin/env python3
"""Score Cloud vs Gemini vorasidenib sentence CTC spans (run on GPU VM)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dose_r.forced_align import extract_drug_span_forced_align
from dose_r.references.reference_clips import available_clips_all
from dose_r.scoring.candidate_eval import score_against_best_reference
from dose_r.scoring.speech_similarity import extract_frame_embeddings, speech_bertscore

DRUG = "vorasidenib"
IPA = "vɔːrəˈsɪdənɪb"
TTS = ROOT / "runs/vora-sent-cloud-vs-gemini/tts"
OUT = ROOT / "runs/vora-sent-cloud-vs-gemini"
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


def mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def main() -> None:
    clips = {}
    for ing, clist in available_clips_all().items():
        clips.setdefault(ing.lower(), []).extend(clist)
    human = clips[DRUG]
    teacher_emb = extract_frame_embeddings(
        (ROOT / "data/gold_gemini_ipa/wavs/vorasidenib.wav").read_bytes()
    )

    rows = []
    for item in SENTENCES:
        sid = item["id"]
        sentence = item["text"]
        row = {"id": sid, "sentence": sentence, "ipa": IPA, "drug": DRUG}
        for arm in ("cloud", "gemini"):
            path = TTS / f"{sid}__{arm}.wav"
            raw = path.read_bytes()
            span = extract_drug_span_forced_align(raw, sentence, DRUG)
            if span is None:
                row[f"{arm}_error"] = "align_none"
                print(sid, arm, "align fail", flush=True)
                continue
            (TTS / f"{sid}__{arm}_span.wav").write_bytes(span)
            best = score_against_best_reference(span, human)
            t = float(speech_bertscore(extract_frame_embeddings(span), teacher_emb)["f1"])
            row[f"{arm}_human_f1"] = round(best.best_f1, 4)
            row[f"{arm}_teacher_f1"] = round(t, 4)
            print(
                f"{sid} {arm}: human={row[f'{arm}_human_f1']} teacher={row[f'{arm}_teacher_f1']}",
                flush=True,
            )
        ch, gh = row.get("cloud_human_f1"), row.get("gemini_human_f1")
        if ch is not None and gh is not None:
            d = gh - ch
            row["delta_human_gemini_minus_cloud"] = round(d, 4)
            row["winner_human"] = (
                "gemini" if d > 0.01 else "cloud" if d < -0.01 else "tie"
            )
        rows.append(row)

    summary = {
        "ipa": IPA,
        "n": len(rows),
        "mean_cloud_human": mean(
            [r["cloud_human_f1"] for r in rows if r.get("cloud_human_f1") is not None]
        ),
        "mean_gemini_human": mean(
            [r["gemini_human_f1"] for r in rows if r.get("gemini_human_f1") is not None]
        ),
        "mean_cloud_teacher": mean(
            [r["cloud_teacher_f1"] for r in rows if r.get("cloud_teacher_f1") is not None]
        ),
        "mean_gemini_teacher": mean(
            [r["gemini_teacher_f1"] for r in rows if r.get("gemini_teacher_f1") is not None]
        ),
        "n_gemini_wins": sum(1 for r in rows if r.get("winner_human") == "gemini"),
        "n_cloud_wins": sum(1 for r in rows if r.get("winner_human") == "cloud"),
        "n_tie": sum(1 for r in rows if r.get("winner_human") == "tie"),
        "rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
