#!/usr/bin/env python3
"""CosyVoice3 zero-shot: Gemini sentence wav as prompt for each vorasidenib sentence.

Prompt = teacher saying the same sentence (or isolated drug). Score CTC-cut
Path-2 vs human. Success: each eval sentence mean>=0.70 and min>=0.62.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dose_r.forced_align import extract_drug_span_forced_align
from dose_r.references.reference_clips import available_clips_all
from dose_r.scoring.candidate_eval import score_against_best_reference

DRUG = "vorasidenib"
WORK = ROOT / "runs" / "vora-sent-cosy"
TEACHER_DIR = ROOT / "runs" / "ft-vora-sent-hillclimb" / "teachers"
ISO_WAV = ROOT / "data" / "gold_gemini_ipa" / "wavs" / "vorasidenib.wav"
RESPELL = "vor-ah-SID-eh-nib"

EVAL = [
    {"id": "dose", "text": "Let's initiate vorasidenib therapy for this patient to target the mutant IDH1 and IDH2 enzymes in the tumor."},
    {"id": "prescribed", "text": "The oncologist prescribed vorasidenib for the patient's IDH-mutant glioma."},
    {"id": "confirm", "text": "Please confirm the vorasidenib dose before the next clinic visit."},
    {"id": "monitor", "text": "Patients taking vorasidenib need monitoring for liver enzyme elevation."},
]


def _first_speech(generator):
    chunks = []
    for piece in generator:
        chunks.append(piece["tts_speech"])
    if not chunks:
        raise RuntimeError("model returned no audio")
    return torch.cat(chunks, dim=1) if chunks[0].ndim == 2 else torch.cat(chunks, dim=0)


def load_human():
    clips = {}
    for ing, clist in available_clips_all().items():
        clips.setdefault(ing.lower(), []).extend(clist)
    return clips[DRUG]


def score_span(wav_path: Path, sentence: str, human) -> float:
    span = extract_drug_span_forced_align(wav_path.read_bytes(), sentence, DRUG)
    if span is None:
        span = extract_drug_span_forced_align(
            wav_path.read_bytes(), sentence.replace(DRUG, RESPELL), RESPELL
        )
        if span is None:
            return 0.0
    return float(score_against_best_reference(span, human).best_f1)


def ok(per: dict) -> bool:
    return all(v["mean"] >= 0.70 for v in per.values()) and all(
        v["min"] >= 0.62 for v in per.values()
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--cosyvoice-root", required=True)
    ap.add_argument("--n-draws", type=int, default=3)
    args = ap.parse_args()

    sys.path.insert(0, args.cosyvoice_root)
    sys.path.insert(0, str(Path(args.cosyvoice_root) / "third_party" / "Matcha-TTS"))
    import huggingface_hub
    if not hasattr(huggingface_hub, "cached_download"):
        huggingface_hub.cached_download = huggingface_hub.hf_hub_download
    from cosyvoice.cli.cosyvoice import AutoModel

    WORK.mkdir(parents=True, exist_ok=True)
    human = load_human()
    model = AutoModel(model_dir=args.model_dir)

    recipes = [
        ("cosy_prompt_sent_teacher", "sentence"),
    ]
    results = {}
    for recipe, mode in recipes:
        print(f"--- {recipe} ---", flush=True)
        per = {}
        for item in EVAL:
            scores = []
            if mode == "sentence":
                prompt_wav = str(TEACHER_DIR / f"{item['id']}__gemini.wav")
                prompt_transcript = item["text"]
            else:
                prompt_wav = str(ISO_WAV)
                prompt_transcript = DRUG
            # CosyVoice3 requires <|endofprompt|> in text or prompt_text.
            prompt_text = f"You are a helpful assistant.<|endofprompt|>{prompt_transcript}"
            for d in range(args.n_draws):
                dest = WORK / f"{recipe}__{item['id']}__d{d}.wav"
                try:
                    speech = _first_speech(
                        model.inference_zero_shot(
                            item["text"],
                            prompt_text,
                            prompt_wav,
                            stream=False,
                        )
                    )
                    wav = speech.squeeze().cpu().numpy()
                    sf.write(dest, wav, 24000)
                    scores.append(score_span(dest, item["text"], human))
                except Exception as exc:
                    print(f"  FAIL {item['id']} d{d}: {exc}", flush=True)
                    scores.append(0.0)
            per[item["id"]] = {
                "draws": [round(x, 4) for x in scores],
                "mean": round(sum(scores) / len(scores), 4),
                "min": round(min(scores), 4),
                "max": round(max(scores), 4),
            }
            print(
                f"  {item['id']}: mean={per[item['id']]['mean']} "
                f"min={per[item['id']]['min']} {per[item['id']]['draws']}",
                flush=True,
            )
        means = [v["mean"] for v in per.values()]
        results[recipe] = {
            "per_sentence": per,
            "mean_of_means": round(sum(means) / len(means), 4),
            "min_of_means": round(min(means), 4),
            "success": ok(per),
        }
        print(
            f"  => mean={results[recipe]['mean_of_means']} "
            f"min_mean={results[recipe]['min_of_means']} "
            f"success={results[recipe]['success']}",
            flush=True,
        )

    out = {"results": results, "any_success": any(v["success"] for v in results.values())}
    (WORK / "summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
