#!/usr/bin/env python3
"""Misaki pin trial for one drug: lexicon inject → sentence synth → CTC crop → F1.

Does not write chosen.json or gold IPA. Updates user_pins only with --keep.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path("/home/shreyaspatel/synthio_voice")
sys.path.insert(0, str(ROOT))
STORE = ROOT / "runs" / "misaki-iter"
VOICE = "af_heart"
SPEED = 1.0
_MISSING = object()

from dose_r.forced_align import extract_drug_span_forced_align
from dose_r.scoring.speech_similarity import extract_frame_embeddings, speech_bertscore


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def load_item(slug: str) -> dict:
    for line in (ROOT / "data" / "dose_v1.jsonl").read_text().splitlines():
        row = json.loads(line)
        for i, ing in enumerate(row.get("ingredients") or [row["name"]]):
            if slugify(ing) == slug:
                spans = row.get("spans") or []
                sentence = row["sentence"]
                if i < len(spans) and len(spans[i]) == 2:
                    a, b = spans[i]
                    spoken = sentence[a:b]
                else:
                    spoken = ing
                return {"drug": ing, "spoken": spoken, "sentence": sentence, "slug": slug}
    raise SystemExit(f"missing {slug}")


def gold_wav(slug: str, spoken: str) -> Path | None:
    names = [slug, spoken.lower().replace(" ", "_"), slug.replace("-", "_")]
    for name in names:
        for folder in (STORE / "cloud-gold", ROOT / "data" / "gold_gemini_ipa" / "wavs"):
            p = folder / f"{name}.wav"
            if p.exists() and p.stat().st_size > 500:
                return p
    return None


def lexicon_assignments(spoken: str, phones: str) -> dict[str, str]:
    words = spoken.split()
    parts = phones.split()
    if len(words) > 1 and len(parts) == len(words):
        return dict(zip(words, parts))
    return {spoken: phones}


def apply_lexicon(golds: dict, spoken: str, phones: str) -> dict:
    saved = {}
    for key, value in lexicon_assignments(spoken, phones).items():
        saved[key] = golds.get(key, _MISSING)
        golds[key] = value
    return saved


def restore_lexicon(golds: dict, saved: dict) -> None:
    for key, old in saved.items():
        if old is _MISSING:
            golds.pop(key, None)
        else:
            golds[key] = old


def main() -> int:
    import torch
    from kokoro import KPipeline

    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--phones", nargs="+", required=True)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    it = load_item(args.slug)
    gold = gold_wav(args.slug, it["spoken"])
    if gold is None:
        raise SystemExit("no gold")
    gold_emb = extract_frame_embeddings(gold.read_bytes())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=device)
    golds = pipeline.g2p.lexicon.golds
    out_dir = STORE / "bottom-up-misaki" / args.slug
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"device={device} slug={args.slug} gold={gold}", flush=True)
    for phones in args.phones:
        saved = apply_lexicon(golds, it["spoken"], phones)
        try:
            result = next(pipeline(it["sentence"], voice=VOICE, speed=SPEED))
            if result.audio is None:
                raise RuntimeError("no audio")
            wav = result.audio.detach().cpu().numpy()
            safe = re.sub(r"[^a-zA-Z0-9]+", "_", phones).strip("_")
            sent_path = out_dir / f"{safe}.sent.wav"
            sf.write(sent_path, wav, 24000)
            raw = sent_path.read_bytes()
            span = extract_drug_span_forced_align(raw, it["sentence"], it["spoken"])
            if span is None:
                span = extract_drug_span_forced_align(raw, it["sentence"], it["drug"])
            if span is None:
                print(f"NOSPAN {phones}", flush=True)
                continue
            (out_dir / f"{safe}.span.wav").write_bytes(span)
            score = float(
                speech_bertscore(extract_frame_embeddings(span), gold_emb)["f1"]
            )
        finally:
            restore_lexicon(golds, saved)

        rows.append({"phones": phones, "f1": round(score, 4)})
        print(f"{score:.4f}  {phones}", flush=True)

    rows.sort(key=lambda r: -r["f1"])
    report = {
        "slug": args.slug,
        "gold": str(gold),
        "current_baseline": 0.5472 if args.slug == "idvynso" else None,
        "rows": rows,
    }
    (out_dir / "trial.json").write_text(json.dumps(report, indent=2) + "\n")
    print("BEST", rows[0] if rows else None, flush=True)

    if args.keep and rows:
        pins_path = STORE / "user_pins.json"
        pins = json.loads(pins_path.read_text())
        best = rows[0]["phones"]
        pins[args.slug] = {
            "word": it["spoken"],
            "misaki": best,
            "source": "bottom-up-misaki",
        }
        pins_path.write_text(json.dumps(pins, indent=2, ensure_ascii=False) + "\n")
        print("WROTE pin", args.slug, best, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
