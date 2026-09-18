#!/usr/bin/env python
"""Compute the human-vs-human F1 ceiling for Path 2 (speech_similarity).

Scores every ingredient with BOTH a Drugs.com and a Merriam-Webster human
reference clip against each other, using the identical SpeechBERTScore F1
pipeline used to score TTS candidates. This is the number any TTS system's
Path 2 score should be read against -- not against 5.0, which even two real
humans don't reach (see docs/PATH2_GEMINI_VS_HUMAN.md).
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dose_r.scoring.speech_similarity import extract_frame_embeddings, speech_bertscore  # noqa: E402


def main() -> int:
    manifest_path = REPO_ROOT / "data" / "reference_audio" / "manifest.jsonl"
    recs = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
    dc = {r["ingredient"]: r["local_path"] for r in recs
         if r["source"] == "drugs.com" and r.get("coverage") == "full"}
    mw = {r["ingredient"]: r["source_url"].split("/")[-1] for r in recs
         if r["source"] == "merriam-webster" and r.get("coverage") == "full"}
    both = sorted(set(dc) & set(mw))
    print(f"dual-source ingredients: {len(both)}")

    results: list[tuple[str, float]] = []
    for i, name in enumerate(both, 1):
        try:
            fa = extract_frame_embeddings(REPO_ROOT / dc[name])
            fb = extract_frame_embeddings(REPO_ROOT / "data" / "reference_audio" / "mw" / mw[name])
            r = speech_bertscore(fa, fb)
            results.append((name, r["f1"]))
        except Exception as exc:
            print(f"  SKIP {name}: {exc}")
        if i % 20 == 0:
            print(f"  {i}/{len(both)}")

    f1s = sorted(f for _, f in results)
    print(f"\nn={len(f1s)}")
    print(f"mean f1={statistics.mean(f1s):.4f}  median={statistics.median(f1s):.4f}")
    print(f"as 0-5 score: mean={5*statistics.mean(f1s):.3f}  median={5*statistics.median(f1s):.3f}")
    print(f"p10={f1s[int(len(f1s)*0.1)]:.4f}  p90={f1s[int(len(f1s)*0.9)]:.4f}")

    out_path = REPO_ROOT / "runs" / "human-ceiling-v1.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "pairs": results, "n": len(f1s),
        "mean_f1": statistics.mean(f1s), "median_f1": statistics.median(f1s),
        "p10_f1": f1s[int(len(f1s) * 0.1)], "p90_f1": f1s[int(len(f1s) * 0.9)],
    }, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
