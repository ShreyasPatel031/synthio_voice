#!/usr/bin/env python3
"""Test asymmetric CTC end recovery on a small name set before changing defaults.

Compares mid50/0.15 (current) vs end75/0.15 and start25+end75/0.15.
"""
from __future__ import annotations

import io
import json
import sys
import wave
from pathlib import Path

ROOT = Path("/home/shreyaspatel/synthio_voice")
sys.path.insert(0, str(ROOT))
STORE = ROOT / "runs" / "misaki-iter"

from dose_r.scoring.speech_similarity import extract_frame_embeddings, speech_bertscore

# Import extract helper from ctc_diag_bottom
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "ctc_diag_bottom", ROOT / "scripts" / "ctc_diag_bottom.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
extract_with_params = _mod.extract_with_params
gold_path = _mod.gold_path
load_item = _mod.load_item


def f1(a: bytes, b: bytes, cache: dict, key: str) -> float:
    if key not in cache:
        cache[key] = extract_frame_embeddings(b)
    return float(speech_bertscore(extract_frame_embeddings(a), cache[key])["f1"])


def main() -> int:
    # bottom 3 + a few mid/high controls so we don't regress winners
    slugs = sys.argv[1:] or [
        "idvynso",
        "advair",
        "vorasidenib",
        "osimertinib",
        "lipitor",
        "tzield",
        "ubrelvy",
        "wegovy",
    ]
    variants = [
        ("default", 0.15, 0.5, 0.5),
        ("end75", 0.15, 0.5, 0.75),
        ("end75_gap25", 0.25, 0.5, 0.75),
        ("start25_end75", 0.15, 0.25, 0.75),
        ("no_start_end75", 0.15, 0.0, 0.75),
    ]
    cache: dict = {}
    rows = []
    for slug in slugs:
        it = load_item(slug)
        gold = gold_path(slug)
        sent = STORE / "lexicon-sent" / f"{slug}.wav"
        if not gold or not sent.exists():
            print("skip", slug, flush=True)
            continue
        gold_b = gold.read_bytes()
        sent_b = sent.read_bytes()
        gkey = str(gold)
        rec = {"slug": slug}
        print(f"\n=== {slug} ===", flush=True)
        for name, gap, ss, es in variants:
            span, meta = extract_with_params(
                sent_b, it["sentence"], it["spoken"], gap, ss, es
            )
            if span is None:
                continue
            score = f1(span, gold_b, cache, gkey)
            rec[name] = {"f1": round(score, 4), **meta}
            print(f"  {name:16} {score:.4f} crop={meta.get('crop_s')}", flush=True)
        if "default" in rec and "end75" in rec:
            rec["delta_end75"] = round(rec["end75"]["f1"] - rec["default"]["f1"], 4)
        rows.append(rec)
    out = STORE / "bottom-up-ctc" / "asymmetric-test.json"
    out.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    print("\nWROTE", out, flush=True)
    print("\nSummary delta end75 vs default:", flush=True)
    for r in rows:
        d = r.get("delta_end75")
        print(f"  {r['slug']:20} {d:+.4f}" if d is not None else f"  {r['slug']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
