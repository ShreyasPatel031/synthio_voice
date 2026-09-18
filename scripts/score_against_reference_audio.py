#!/usr/bin/env python
"""Ground the ASR round-trip scores in real human reference audio.

Two things happen here that the spelling-based ASR round-trip scorer
(`runs/asr-roundtrip-v1-full`) could not do on its own:

1. Establish, per ingredient, whether Cloud STT can even recognize a HUMAN
   saying the name correctly. This turns the "ASR has its own drug-name bias"
   caveat from a guess into a measured per-item fact.
2. Re-score every synthesized clip against the reference recognizer's own
   transcript instead of the spelled name, for the subset of ingredients that
   have a committed reference clip (Drugs.com only in this repo -- see
   dose_r.references.reference_clips).

Usage:
    python scripts/score_against_reference_audio.py \\
        --synth-run runs/asr-roundtrip-v1-full --out runs/reference-grounded-v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dose_r import auth, report  # noqa: E402
from dose_r.references import audio_grounded, reference_clips  # noqa: E402


def transcribe_all_references(out_path: Path) -> dict[str, audio_grounded.ReferenceTranscript]:
    """Transcribe every available reference clip once, cached to disk."""
    clips = reference_clips.available_clips()
    cached: dict[str, audio_grounded.ReferenceTranscript] = {}

    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            cached[d["ingredient"]] = audio_grounded.ReferenceTranscript(**d)

    headers = auth.auth_headers()
    todo = [c for ing, c in clips.items() if ing not in cached]
    print(f"reference clips available: {len(clips)}  cached: {len(cached)}  to transcribe: {len(todo)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as sink:
        for n, clip in enumerate(todo, start=1):
            try:
                t = audio_grounded.transcribe_reference_clip(clip, headers)
            except Exception as exc:
                print(f"  [{n}/{len(todo)}] {clip.ingredient}: ERROR {exc}")
                continue
            cached[clip.ingredient] = t
            sink.write(json.dumps(t.__dict__) + "\n")
            sink.flush()
            if n % 25 == 0 or n == len(todo):
                print(f"  [{n}/{len(todo)}] {clip.ingredient} -> "
                      f"{t.recognized!r} recognizable={t.asr_recognizable}")
    return cached


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth-run", default="runs/asr-roundtrip-v1-full")
    ap.add_argument("--out", default="runs/reference-grounded-v1")
    args = ap.parse_args()

    out_dir = Path(args.out)
    transcripts_path = out_dir / "reference_transcripts.jsonl"
    references = transcribe_all_references(transcripts_path)

    recognizable = [t for t in references.values() if t.recognized is not None]
    baseline_ok = [t for t in recognizable if t.asr_recognizable]
    print(f"\n=== ASR baseline on HUMAN reference audio ===")
    print(f"  {len(references)} clips transcribed, {len(references) - len(recognizable)} produced no transcript")
    print(f"  {len(baseline_ok)}/{len(recognizable)} = "
          f"{100*len(baseline_ok)/max(len(recognizable),1):.1f}% recognized correctly "
          f"when a HUMAN says the name")

    synth_records = report.load_records(Path(args.synth_run) / "results.jsonl")
    grounded_records = []
    for rec in synth_records:
        ing = rec["drug"]
        ref = references.get(ing)
        score = rec.get("score") or {}
        synth_span = (score.get("metadata") or {}).get("recognized_span")
        out_rec = dict(rec)
        if ref is None or synth_span is None:
            out_rec["grounded"] = {"scoreable": False, "reason": "no reference clip or no synth transcript"}
        else:
            out_rec["grounded"] = audio_grounded.score_against_reference(ref, synth_span)
        grounded_records.append(out_rec)

    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    with results_path.open("w") as f:
        for r in grounded_records:
            f.write(json.dumps(r) + "\n")

    # --- report -------------------------------------------------------
    scoreable = [r for r in grounded_records if r["grounded"].get("scoreable")]
    print(f"\n{len(scoreable)}/{len(grounded_records)} synth records have a reference-grounded score "
          f"({len(references)} unique ingredients covered)")

    by_system: dict[str, list[dict]] = {}
    for r in scoreable:
        by_system.setdefault(r["system_id"], []).append(r)

    print(f"\n=== Pass rate vs REFERENCE AUDIO (all covered items) ===")
    print(f"{'system':26s} {'n':>5s} {'pass%':>7s}   vs   {'restricted to ASR-recognizable refs':>36s} {'n':>5s} {'pass%':>7s}")
    for sysid, rows in sorted(by_system.items()):
        n = len(rows)
        passed = sum(1 for r in rows if r["grounded"]["passed_vs_reference"])
        restricted = [r for r in rows if r["grounded"]["reference_asr_recognizable"]]
        rn = len(restricted)
        rp = sum(1 for r in restricted if r["grounded"]["passed_vs_reference"])
        print(f"{sysid:26s} {n:5d} {100*passed/n:6.1f}%   {'':36s} {rn:5d} "
              f"{100*rp/max(rn,1):6.1f}%")

    # side-by-side with the original spelling-based score, same subset
    print(f"\n=== Same subset, spelling-based score (original scorer) for comparison ===")
    for sysid, rows in sorted(by_system.items()):
        n = len(rows)
        orig_pass = sum(1 for r in rows if (r.get("score") or {}).get("passed"))
        print(f"{sysid:26s} {n:5d} {100*orig_pass/n:6.1f}%")

    summary = {
        "reference_clips_available": len(references),
        "reference_clips_asr_recognizable": len(baseline_ok),
        "reference_clips_no_transcript": len(references) - len(recognizable),
        "synth_records_grounded": len(scoreable),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nartifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
