#!/usr/bin/env python3
"""Forced-align drug spans from kokoro-full-plain sentence wavs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util

from dose_r.forced_align import extract_drug_span_forced_align

_spec = importlib.util.spec_from_file_location("unify", ROOT / "scripts" / "unify_lexicon_benchmark.py")
_unify = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_unify)
load_items = _unify.load_items

SRC = ROOT / "runs" / "oss-eval" / "kokoro-full-plain"
OUT = ROOT / "runs" / "misaki-iter" / "kokoro-plain-span"
RANK = ROOT / "runs" / "misaki-iter" / "lexicon-rank.json"


def main() -> int:
    items = {it["slug"]: it for it in load_items()}
    slugs = [r["slug"] for r in json.loads(RANK.read_text())["rows"]]
    OUT.mkdir(parents=True, exist_ok=True)
    ok = 0
    for n, slug in enumerate(slugs, 1):
        it = items.get(slug)
        wav = SRC / f"{slug}.wav"
        if it is None or not wav.exists():
            print("skip", slug, flush=True)
            continue
        raw = wav.read_bytes()
        span = extract_drug_span_forced_align(raw, it["sentence"], it["spoken"])
        if span is None:
            span = extract_drug_span_forced_align(raw, it["sentence"], it["drug"])
        if span is None:
            print("nospan", slug, flush=True)
            continue
        (OUT / f"{slug}.wav").write_bytes(span)
        ok += 1
        if n % 40 == 0:
            print(n, slug, flush=True)
    print(f"WROTE {ok} -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
