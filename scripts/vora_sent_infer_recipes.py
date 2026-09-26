#!/usr/bin/env python3
"""Inference recipes to get vorasidenib sentences to human F1 >= 0.70.

Tests several paths without assuming custom_voice FT alone is enough:
  A  base_clone_sent_teacher   — ceiling: clone Gemini sentence wav
  B  base_clone_iso_xvec       — Base, x-vector from isolated drug, speak sentence
  C  base_clone_iso_full       — Base, full clone isolated drug + ref_text=drug
  D  ft_custom                 — FT custom_voice speaker (if ckpt given)
  E  ft_clone_iso_xvec         — FT weights + x-vector isolated drug
  F  base_custom_instruct      — if using a CustomVoice checkpoint, instruct hint
  G  base_clone_iso_respell    — Base clone iso; text with ASCII respelling of drug

Multi-draw, low temperature. Success: each sentence mean>=0.70 and min>=0.62.
"""

from __future__ import annotations

import argparse
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

DRUG = "vorasidenib"
# Source-published / DailyMed-style ASCII respelling — NOT IPA G2P.
RESPELL = "vor-ah-SID-eh-nib"
IPA_SOURCE = "vɔːrəˈsɪdənɪb"  # only used if a source published it; for clone text we avoid IPA

MODEL_BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
WORK = ROOT / "runs" / "vora-sent-recipes"
TEACHER_DIR = ROOT / "runs" / "ft-vora-sent-hillclimb" / "teachers"
ISO_WAV = ROOT / "data" / "gold_gemini_ipa" / "wavs" / "vorasidenib.wav"

EVAL = [
    {"id": "dose", "text": "Let's initiate vorasidenib therapy for this patient to target the mutant IDH1 and IDH2 enzymes in the tumor."},
    {"id": "prescribed", "text": "The oncologist prescribed vorasidenib for the patient's IDH-mutant glioma."},
    {"id": "confirm", "text": "Please confirm the vorasidenib dose before the next clinic visit."},
    {"id": "monitor", "text": "Patients taking vorasidenib need monitoring for liver enzyme elevation."},
]


def load_model(path: str) -> Qwen3TTSModel:
    return Qwen3TTSModel.from_pretrained(
        path,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )


def free(m) -> None:
    del m
    torch.cuda.empty_cache()


def synth(model, recipe: str, item: dict, *, ft_speaker: str, temperature: float, seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    kw = dict(temperature=temperature, top_p=0.8, do_sample=True)
    text = item["text"]
    teacher = TEACHER_DIR / f"{item['id']}__gemini.wav"

    if recipe == "A_base_clone_sent_teacher":
        return model.generate_voice_clone(
            text=text,
            language="English",
            ref_audio=str(teacher),
            ref_text=text,
            **kw,
        )
    if recipe == "B_base_clone_iso_xvec":
        return model.generate_voice_clone(
            text=text,
            language="English",
            ref_audio=str(ISO_WAV),
            ref_text=DRUG,
            x_vector_only_mode=True,
            **kw,
        )
    if recipe == "C_base_clone_iso_full":
        return model.generate_voice_clone(
            text=text,
            language="English",
            ref_audio=str(ISO_WAV),
            ref_text=DRUG,
            x_vector_only_mode=False,
            **kw,
        )
    if recipe == "D_ft_custom":
        return model.generate_custom_voice(
            text=text, language="English", speaker=ft_speaker, **kw
        )
    if recipe == "E_ft_clone_iso_xvec":
        return model.generate_voice_clone(
            text=text,
            language="English",
            ref_audio=str(ISO_WAV),
            ref_text=DRUG,
            x_vector_only_mode=True,
            **kw,
        )
    if recipe == "G_base_clone_iso_respell":
        spoken = text.replace(DRUG, RESPELL).replace(DRUG.capitalize(), RESPELL)
        return model.generate_voice_clone(
            text=spoken,
            language="English",
            ref_audio=str(ISO_WAV),
            ref_text=DRUG,
            x_vector_only_mode=True,
            **kw,
        )
    if recipe == "H_base_clone_iso_respell_full":
        spoken = text.replace(DRUG, RESPELL).replace(DRUG.capitalize(), RESPELL)
        return model.generate_voice_clone(
            text=spoken,
            language="English",
            ref_audio=str(ISO_WAV),
            ref_text=DRUG,
            x_vector_only_mode=False,
            **kw,
        )
    if recipe == "I_clone_sent_teacher_respell":
        spoken = text.replace(DRUG, RESPELL).replace(DRUG.capitalize(), RESPELL)
        return model.generate_voice_clone(
            text=spoken,
            language="English",
            ref_audio=str(teacher),
            ref_text=text,
            **kw,
        )
    if recipe == "J_clone_sent_instruct":
        return model.generate_voice_clone(
            text=text,
            language="English",
            ref_audio=str(teacher),
            ref_text=text,
            instruct="Pronounce vorasidenib slowly and clearly, matching the reference.",
            **kw,
        )
    if recipe == "K_clone_sent_greedy":
        kw = dict(do_sample=False)
        return model.generate_voice_clone(
            text=text,
            language="English",
            ref_audio=str(teacher),
            ref_text=text,
            **kw,
        )
    if recipe == "F_ft_clone_sent_teacher":
        return model.generate_voice_clone(
            text=text,
            language="English",
            ref_audio=str(teacher),
            ref_text=text,
            **kw,
        )
    raise SystemExit(f"unknown recipe {recipe}")


def score_span(wav_path: Path, sentence: str, human) -> float:
    span = extract_drug_span_forced_align(wav_path.read_bytes(), sentence, DRUG)
    if span is None:
        # try with respelling in sentence for aligner
        alt = sentence.replace(DRUG, RESPELL)
        span = extract_drug_span_forced_align(wav_path.read_bytes(), alt, RESPELL)
        if span is None:
            return 0.0
    return float(score_against_best_reference(span, human).best_f1)


def ok(per: dict) -> bool:
    return all(v["mean"] >= 0.70 for v in per.values()) and all(
        v["min"] >= 0.62 for v in per.values()
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipes", nargs="+", default=[
        "A_base_clone_sent_teacher",
        "B_base_clone_iso_xvec",
        "C_base_clone_iso_full",
        "G_base_clone_iso_respell",
        "H_base_clone_iso_respell_full",
    ])
    ap.add_argument("--ft-ckpt", default="")
    ap.add_argument("--ft-speaker", default="dose_sent3")
    ap.add_argument("--n-draws", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.25)
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    clips = {}
    for ing, clist in available_clips_all().items():
        clips.setdefault(ing.lower(), []).extend(clist)
    human = clips[DRUG]

    recipes = list(args.recipes)
    if args.ft_ckpt:
        extra = ["D_ft_custom", "E_ft_clone_iso_xvec", "F_ft_clone_sent_teacher"]
        recipes.extend([r for r in extra if r not in recipes])

    results = {}
    # group by which model weights needed
    need_base = [r for r in recipes if r.startswith(("A_", "B_", "C_", "G_", "H_", "I_", "J_", "K_"))]
    need_ft = [r for r in recipes if r.startswith(("D_", "E_", "F_"))]

    def run_group(model_path: str, group: list[str]):
        if not group:
            return
        print(f"=== load {model_path} ===", flush=True)
        model = load_model(model_path)
        for recipe in group:
            print(f"--- {recipe} ---", flush=True)
            per = {}
            for item in EVAL:
                scores = []
                for d in range(args.n_draws):
                    seed = 4000 + d * 47 + sum(map(ord, item["id"] + recipe)) % 997
                    wavs, sr = synth(
                        model,
                        recipe,
                        item,
                        ft_speaker=args.ft_speaker,
                        temperature=args.temperature,
                        seed=seed,
                    )
                    dest = WORK / f"{recipe}__{item['id']}__d{d}.wav"
                    sf.write(dest, wavs[0], sr)
                    s = score_span(dest, item["text"], human)
                    scores.append(s)
                per[item["id"]] = {
                    "draws": [round(x, 4) for x in scores],
                    "mean": round(sum(scores) / len(scores), 4),
                    "min": round(min(scores), 4),
                    "max": round(max(scores), 4),
                }
                print(
                    f"  {item['id']}: mean={per[item['id']]['mean']} "
                    f"min={per[item['id']]['min']} max={per[item['id']]['max']} "
                    f"{per[item['id']]['draws']}",
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
        free(model)

    run_group(MODEL_BASE, need_base)
    if need_ft and args.ft_ckpt:
        run_group(args.ft_ckpt, need_ft)

    out = {
        "temperature": args.temperature,
        "n_draws": args.n_draws,
        "criteria": "mean>=0.70 and min>=0.62 per sentence",
        "results": results,
        "any_success": any(v["success"] for v in results.values()),
        "best": max(results.items(), key=lambda kv: kv[1]["min_of_means"])[0]
        if results
        else None,
    }
    (WORK / "summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in out if k != "results"}, indent=2), flush=True)
    for r, v in sorted(results.items(), key=lambda kv: -kv[1]["min_of_means"]):
        print(
            f"{r}: mean={v['mean_of_means']} min_mean={v['min_of_means']} success={v['success']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
