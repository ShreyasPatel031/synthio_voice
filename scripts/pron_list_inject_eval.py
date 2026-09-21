#!/usr/bin/env python3
"""Pronunciation-list fallback: inject the listed word's audio at generate time.

No fine-tune. For each eval sentence, look up the drug in a pronunciation list
(spoken form + wav path + optional source IPA), then generate with Base
voice_clone using that wav as ref. Score CTC-cut span vs:
  - human Path-2
  - Gemini gold sentence span (the in-sentence teacher)

Compares greedy vs sampled decode. Caps runaway wavs.
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
from dose_r.scoring.speech_similarity import extract_frame_embeddings, speech_bertscore

DRUG = "vorasidenib"
MODEL_BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
WORK = ROOT / "runs" / "pron-list-inject"
LIST_PATH = ROOT / "data" / "pronunciation_list.json"
SENT_TTS = ROOT / "runs" / "vora-sent-cloud-vs-gemini" / "tts"
MAX_WAV_BYTES = 2_000_000

EVAL = [
    {"id": "dose", "text": "Let's initiate vorasidenib therapy for this patient to target the mutant IDH1 and IDH2 enzymes in the tumor."},
    {"id": "prescribed", "text": "The oncologist prescribed vorasidenib for the patient's IDH-mutant glioma."},
    {"id": "confirm", "text": "Please confirm the vorasidenib dose before the next clinic visit."},
    {"id": "monitor", "text": "Patients taking vorasidenib need monitoring for liver enzyme elevation."},
]


def write_list() -> dict:
    """One entry: spelling → spoken form + source IPA + teacher wav (no G2P)."""
    iso = ROOT / "data/gold_gemini_ipa/wavs/vorasidenib.wav"
    # Prefer Gemini isolated if present; else Cloud Standard-C IPA teacher.
    for p in (
        ROOT / "runs/gemini31-ipa-vs-cloud/tts/vorasidenib.sidecar.wav",
        ROOT / "runs/verify-google-ipa/tts/vorasidenib.ipa.wav",
    ):
        if p.exists():
            iso = p
            break
    entry = {
        "spoken": "vorasidenib",
        # Source-published IPA from Cloud pack / Google — not respelling G2P.
        "ipa": "vɔːrəˈsɪdənɪb",
        "audio": str(iso.relative_to(ROOT)),
        "audio_abs": str(iso.resolve()),
        "notes": "Cloud Standard-C IPA teacher (or Gemini isolated if present). Used as inject clip.",
    }
    data = {"version": 1, "entries": {DRUG: entry, DRUG.capitalize(): entry}}
    LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    LIST_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return entry


def load_model():
    return Qwen3TTSModel.from_pretrained(
        MODEL_BASE,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )


def gemini_span(item: dict) -> bytes | None:
    p = SENT_TTS / f"{item['id']}__gemini_span.wav"
    if p.exists():
        return p.read_bytes()
    full = SENT_TTS / f"{item['id']}__gemini.wav"
    if not full.exists():
        return None
    return extract_drug_span_forced_align(full.read_bytes(), item["text"], DRUG)


def synth(model, text: str, ref_wav: str, ref_text: str, *, greedy: bool, seed: int, xvec: bool):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    kw = dict(do_sample=False) if greedy else dict(do_sample=True, temperature=0.2, top_p=0.8)
    return model.generate_voice_clone(
        text=text,
        language="English",
        ref_audio=ref_wav,
        ref_text=ref_text,
        x_vector_only_mode=xvec,
        **kw,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-draws", type=int, default=3)
    args = ap.parse_args()

    entry = write_list()
    ref_wav = entry["audio_abs"]
    print(f"list={LIST_PATH} inject={ref_wav} ipa={entry['ipa']}", flush=True)

    clips: dict = {}
    for ing, clist in available_clips_all().items():
        clips.setdefault(ing.lower(), []).extend(clist)
    human = clips[DRUG]

    WORK.mkdir(parents=True, exist_ok=True)
    model = load_model()

    recipes = [
        ("inject_word_greedy", True, False),
        ("inject_word_sample", False, False),
        ("inject_word_xvec_greedy", True, True),
        ("inject_word_xvec_sample", False, True),
    ]
    results = {}
    for name, greedy, xvec in recipes:
        print(f"--- {name} ---", flush=True)
        per = {}
        for item in EVAL:
            gspan = gemini_span(item)
            g_emb = extract_frame_embeddings(gspan) if gspan else None
            human_scores, gemini_scores = [], []
            for d in range(args.n_draws):
                seed = 9000 + d * 19 + sum(map(ord, item["id"] + name))
                wavs, sr = synth(
                    model, item["text"], ref_wav, DRUG,
                    greedy=greedy, seed=seed, xvec=xvec,
                )
                dest = WORK / f"{name}__{item['id']}__d{d}.wav"
                sf.write(dest, wavs[0], sr)
                if dest.stat().st_size > MAX_WAV_BYTES:
                    print(f"  RUNAWAY {item['id']} d{d}", flush=True)
                    human_scores.append(0.0)
                    gemini_scores.append(0.0)
                    continue
                span = extract_drug_span_forced_align(dest.read_bytes(), item["text"], DRUG)
                if span is None:
                    human_scores.append(0.0)
                    gemini_scores.append(0.0)
                    continue
                (dest.with_name(dest.stem + "_span.wav")).write_bytes(span)
                human_scores.append(float(score_against_best_reference(span, human).best_f1))
                if g_emb is not None:
                    gemini_scores.append(
                        float(speech_bertscore(extract_frame_embeddings(span), g_emb)["f1"])
                    )
                else:
                    gemini_scores.append(0.0)
            per[item["id"]] = {
                "human": {
                    "draws": [round(s, 4) for s in human_scores],
                    "mean": round(sum(human_scores) / len(human_scores), 4),
                    "max": round(max(human_scores), 4),
                },
                "vs_gemini_span": {
                    "draws": [round(s, 4) for s in gemini_scores],
                    "mean": round(sum(gemini_scores) / len(gemini_scores), 4),
                    "max": round(max(gemini_scores), 4),
                },
            }
            print(
                f"  {item['id']}: human={per[item['id']]['human']['mean']} "
                f"gemini={per[item['id']]['vs_gemini_span']['mean']} "
                f"h={per[item['id']]['human']['draws']} "
                f"g={per[item['id']]['vs_gemini_span']['draws']}",
                flush=True,
            )
        h_means = [v["human"]["mean"] for v in per.values()]
        g_means = [v["vs_gemini_span"]["mean"] for v in per.values()]
        results[name] = {
            "per_sentence": per,
            "mean_human": round(sum(h_means) / len(h_means), 4),
            "mean_vs_gemini": round(sum(g_means) / len(g_means), 4),
            "min_human": round(min(h_means), 4),
            "min_vs_gemini": round(min(g_means), 4),
        }
        print(
            f"  => human={results[name]['mean_human']} "
            f"vs_gemini={results[name]['mean_vs_gemini']}",
            flush=True,
        )

    del model
    torch.cuda.empty_cache()
    summary = {
        "note": "No FT. Pronunciation list → inject isolated word wav via voice_clone.",
        "list": str(LIST_PATH),
        "inject_wav": ref_wav,
        "ipa": entry["ipa"],
        "prior_full_sentence_clone_greedy_mean_human": 0.7048,
        "results": results,
        "best_by_mean_human": max(results.items(), key=lambda kv: kv[1]["mean_human"])[0],
        "best_by_mean_gemini": max(results.items(), key=lambda kv: kv[1]["mean_vs_gemini"])[0],
    }
    (WORK / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k != "results"}, indent=2), flush=True)
    for k, v in sorted(results.items(), key=lambda kv: -kv[1]["mean_vs_gemini"]):
        print(
            f"{k}: human={v['mean_human']} gemini={v['mean_vs_gemini']} "
            f"min_h={v['min_human']} min_g={v['min_vs_gemini']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
