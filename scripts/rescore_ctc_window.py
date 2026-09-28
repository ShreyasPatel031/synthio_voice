#!/usr/bin/env python3
"""Re-crop lexicon-sent wavs with current forced_align and rescore vs gold.

Use after CTC window changes. Merges F1 into lexicon-rank.json for --only slugs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/home/shreyaspatel/synthio_voice")
sys.path.insert(0, str(ROOT))
STORE = ROOT / "runs" / "misaki-iter"

from dose_r.forced_align import extract_drug_span_forced_align
from dose_r.scoring.speech_similarity import extract_frame_embeddings, speech_bertscore

# reuse helpers
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "ctc_diag_bottom", ROOT / "scripts" / "ctc_diag_bottom.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
gold_path = _mod.gold_path
load_item = _mod.load_item


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=())
    args = ap.parse_args()
    rank_path = STORE / "lexicon-rank.json"
    rank = json.loads(rank_path.read_text())
    by = {r["slug"]: r for r in rank["rows"]}
    slugs = list(args.only) if args.only else [r["slug"] for r in rank["rows"]]
    cache: dict = {}
    for n, slug in enumerate(slugs, 1):
        it = load_item(slug)
        gold = gold_path(slug)
        sent = STORE / "lexicon-sent" / f"{slug}.wav"
        if not gold or not sent.exists() or slug not in by:
            print("skip", slug, flush=True)
            continue
        raw = sent.read_bytes()
        span = extract_drug_span_forced_align(raw, it["sentence"], it["spoken"])
        if span is None:
            span = extract_drug_span_forced_align(raw, it["sentence"], it["drug"])
        if span is None:
            print("nospan", slug, flush=True)
            continue
        (STORE / "lexicon-span" / f"{slug}.wav").write_bytes(span)
        gkey = str(gold)
        if gkey not in cache:
            cache[gkey] = extract_frame_embeddings(gold.read_bytes())
        f1 = float(
            speech_bertscore(extract_frame_embeddings(span), cache[gkey])["f1"]
        )
        old = by[slug]["ctc_f1"]
        by[slug]["ctc_f1"] = round(f1, 4)
        by[slug]["ctc_rescored"] = "end75"
        print(f"{n}/{len(slugs)} {slug} {old:.4f} -> {f1:.4f} ({f1-old:+.3f})", flush=True)
    rows = sorted(by.values(), key=lambda r: r["ctc_f1"])
    scored = [r["ctc_f1"] for r in rows]
    rank["rows"] = rows
    rank["n"] = len(rows)
    rank["mean_f1"] = round(sum(scored) / len(scored), 4) if scored else 0.0
    rank["ctc_window"] = "start50_end75_cap0.15"
    rank_path.write_text(json.dumps(rank, indent=2) + "\n")
    print(f"WROTE {rank_path} n={rank['n']} mean={rank['mean_f1']}", flush=True)
    print("lowest:", [(r["ctc_f1"], r["slug"]) for r in rows[:8]], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
