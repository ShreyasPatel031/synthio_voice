#!/usr/bin/env python3
"""Official DoSE benchmark: CTC-cut Path-2 vs validated Gemini IPA gold.

For every DoSE carrier sentence:
  1. Load candidate full-sentence wav
  2. Forced-align CTC-cut the ingredient being scored
  3. SpeechBERTScore F1 vs the teacher in data/gold_gemini_ipa/wavs/

Gold is ONLY data/gold_gemini_ipa (validated IPA + Gemini 3.1 Kore wav).
Do not edit those IPA strings. Do not G2P a respelling into IPA.
Do not score against runs/gemini31-ipa-vs-cloud or Cloud Standard-C.

Usage:
  python scripts/score_dose_ctc_vs_gemini_ipa.py \\
    --wav-dir runs/oss-eval/kokoro-full-plain \\
    --condition kokoro-full-plain-vs-gemini-ipa \\
    --scope full
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import wave
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dose_r.forced_align import extract_drug_span_forced_align
from dose_r.scoring.speech_similarity import (
    extract_frame_embeddings,
    speech_bertscore,
)

# Human-vs-human floor used elsewhere in this repo (Path 2).
_HUMAN_F1_FLOOR = 0.747


def f1_to_score(f1: float) -> float:
    """1 at F1=0, 5 at human-vs-human floor, clamped."""
    return max(1.0, min(5.0, 1.0 + 4.0 * (float(f1) / _HUMAN_F1_FLOOR)))

GOLD = ROOT / "data" / "gold_gemini_ipa"
GOLD_MANIFEST = GOLD / "manifest.jsonl"
DOSE = ROOT / "data" / "dose_v1.jsonl"
HARD = ROOT / "runs" / "hard-subset-v1.json"
PASS_F1 = 0.70  # near Gemini-in-sentence band; Path-2 human band floor is 0.747


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def slug_us(name: str) -> str:
    return name.lower().replace(" ", "_")


def gemini_ipa_wav(name: str) -> Path | None:
    """Validated Gemini 3.1 Kore teacher. Never a sidecar/prompt/Cloud wav."""
    key = name.lower()
    if GOLD_MANIFEST.exists():
        for line in GOLD_MANIFEST.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("ingredient", "").lower() != key:
                continue
            rel = rec.get("audio") or ""
            p = ROOT / rel if rel and not Path(rel).is_absolute() else Path(rel)
            if p.exists() and p.stat().st_size > 500:
                return p
            break
    for s in (slug(name), slug_us(name), name.lower()):
        p = GOLD / "wavs" / f"{s}.wav"
        if p.exists() and p.stat().st_size > 500:
            return p
    return None


def wav_bytes(path: Path) -> bytes:
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


def find_candidate(wav_dir: Path, name: str, item_id: str | None) -> Path | None:
    for key in (slug(name), slug_us(name), name.lower(), (item_id or "")):
        if not key:
            continue
        for p in (wav_dir / f"{key}.wav", wav_dir / f"{key}.mp3"):
            if p.exists() and p.stat().st_size > 500:
                return p
    return None


def load_eval_items(scope: str) -> list[dict]:
    """One row per (dose sentence, ingredient) with a spoken span."""
    hard_drugs: set[str] | None = None
    if scope == "hard":
        hard = json.loads(HARD.read_text())["items"]
        hard_drugs = {v["drug"].lower() for v in hard.values()}

    items: list[dict] = []
    for line in DOSE.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ingredients = row.get("ingredients") or [row["name"]]
        spans = row.get("spans") or []
        sentence = row["sentence"]
        for i, ing in enumerate(ingredients):
            if hard_drugs is not None and ing.lower() not in hard_drugs:
                # also allow product name match for hard-subset branded items
                if row["name"].lower() not in hard_drugs:
                    continue
            # spoken casing from sentence
            idx = -1
            if i < len(spans) and isinstance(spans[i], (list, tuple)) and len(spans[i]) == 2:
                a, b = spans[i]
                spoken = sentence[a:b]
                idx = a
            else:
                idx = sentence.lower().find(ing.lower())
                if idx < 0:
                    continue
                spoken = sentence[idx : idx + len(ing)]
            items.append(
                {
                    "item_id": f"{row['id']}#{i}",
                    "dose_id": row["id"],
                    "product": row["name"],
                    "drug": ing,
                    "spoken": spoken,
                    "sentence": sentence,
                    "name_type": row.get("name_type"),
                }
            )
    return items


def score_span_vs_gemini(span: bytes, gold_path: Path) -> float:
    emb_c = extract_frame_embeddings(span)
    emb_g = extract_frame_embeddings(gold_path.read_bytes())
    return float(speech_bertscore(emb_c, emb_g)["f1"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True, type=Path)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--scope", choices=("full", "hard"), default="full")
    ap.add_argument("--save-spans", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    wav_dir = args.wav_dir
    if not wav_dir.is_absolute():
        wav_dir = ROOT / wav_dir
    out_dir = ROOT / "runs" / "dose-ctc-vs-gemini-ipa" / args.condition
    out_dir.mkdir(parents=True, exist_ok=True)
    span_dir = out_dir / "spans"
    if args.save_spans:
        span_dir.mkdir(exist_ok=True)

    items = load_eval_items(args.scope)
    if args.limit:
        items = items[: args.limit]
    print(f"items={len(items)} scope={args.scope} wav_dir={wav_dir}", flush=True)

    rows = []
    n_ok = n_miss_cand = n_miss_gold = n_align_fail = 0
    f1s: list[float] = []

    for i, item in enumerate(items):
        row = dict(item)
        row["condition"] = args.condition
        gold = gemini_ipa_wav(item["drug"])
        if gold is None:
            row.update(scoreable=False, error="no_gemini_ipa_gold")
            rows.append(row)
            n_miss_gold += 1
            continue
        row["gemini_gold"] = str(gold.relative_to(ROOT)) if gold.is_relative_to(ROOT) else str(gold)

        cand = find_candidate(wav_dir, item["product"], item["dose_id"])
        if cand is None:
            # hard-subset often keys by drug slug
            cand = find_candidate(wav_dir, item["drug"], item.get("item_id"))
        if cand is None:
            row.update(scoreable=False, error="missing_candidate_wav")
            rows.append(row)
            n_miss_cand += 1
            continue
        row["candidate_wav"] = str(cand)

        try:
            span = extract_drug_span_forced_align(
                wav_bytes(cand), item["sentence"], item["spoken"]
            )
            if span is None:
                span = extract_drug_span_forced_align(
                    wav_bytes(cand), item["sentence"], item["drug"]
                )
        except Exception as exc:
            row.update(scoreable=False, error=f"align:{exc}")
            rows.append(row)
            n_align_fail += 1
            continue
        if span is None:
            row.update(scoreable=False, error="align_none")
            rows.append(row)
            n_align_fail += 1
            continue

        if args.save_spans:
            (span_dir / f"{slug(item['drug'])}__{item['dose_id']}.wav").write_bytes(span)

        try:
            f1 = score_span_vs_gemini(span, gold)
        except Exception as exc:
            row.update(scoreable=False, error=f"score:{exc}")
            rows.append(row)
            continue

        score_15 = f1_to_score(f1)
        row.update(
            scoreable=True,
            gemini_f1=round(f1, 4),
            gemini_score_1_5=round(score_15, 3),
            pass_f1=f1 >= PASS_F1,
        )
        rows.append(row)
        n_ok += 1
        f1s.append(f1)
        if (i + 1) % 25 == 0 or i == 0:
            print(
                f"[{i+1}/{len(items)}] {item['drug']}: f1={f1:.4f} "
                f"running_mean={sum(f1s)/len(f1s):.4f}",
                flush=True,
            )

    scores_path = out_dir / "scores.jsonl"
    with scores_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    summary = {
        "condition": args.condition,
        "scope": args.scope,
        "wav_dir": str(wav_dir),
        "n_items": len(items),
        "n_scoreable": n_ok,
        "n_missing_candidate": n_miss_cand,
        "n_missing_gemini_gold": n_miss_gold,
        "n_align_fail": n_align_fail,
        "mean_gemini_f1": round(sum(f1s) / len(f1s), 4) if f1s else None,
        "median_gemini_f1": round(sorted(f1s)[len(f1s) // 2], 4) if f1s else None,
        "pass_rate_f1_ge_0_70": round(sum(1 for x in f1s if x >= PASS_F1) / len(f1s), 4)
        if f1s
        else None,
        "pass_threshold_f1": PASS_F1,
        "gold": "data/gold_gemini_ipa (validated IPA + Gemini 3.1 Kore wav). Do not edit IPA.",
        "method": "CTC forced-align drug span vs Gemini IPA gold SpeechBERTScore F1",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print("wrote", scores_path, flush=True)


if __name__ == "__main__":
    main()
