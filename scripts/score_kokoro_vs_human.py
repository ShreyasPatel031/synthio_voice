#!/usr/bin/env python3
"""WavLM F1: lexicon Kokoro sentence crop vs the preferred human clip."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dose_r.references.reference_clips import available_clips
from dose_r.scoring.speech_similarity import extract_frame_embeddings, speech_bertscore

STORE = ROOT / "runs" / "misaki-iter"
SPAN = STORE / "lexicon-span"
OUT = STORE / "lexicon-vs-human.json"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def main() -> int:
    rank = json.loads((STORE / "lexicon-rank.json").read_text())
    clips = {slug(k): v for k, v in available_clips().items()}
    rows = []
    for n, row in enumerate(rank["rows"], 1):
        s = row["slug"]
        span = SPAN / f"{s}.wav"
        clip = clips.get(s)
        if clip is None or not span.exists():
            continue
        f1 = float(
            speech_bertscore(
                extract_frame_embeddings(span.read_bytes()),
                extract_frame_embeddings(clip.path.read_bytes()),
            )["f1"]
        )
        rec = {
            "slug": s,
            "drug": row["drug"],
            "human_source": clip.source,
            "human_f1": round(f1, 4),
            "cloud_f1": row["ctc_f1"],
        }
        rows.append(rec)
        print(f"{n} {s} human={f1:.3f} cloud={row['ctc_f1']:.3f} src={clip.source}", flush=True)
    OUT.write_text(json.dumps({"n": len(rows), "rows": rows}, indent=2) + "\n")
    print(f"WROTE {OUT} n={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
