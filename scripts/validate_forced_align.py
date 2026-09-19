#!/usr/bin/env python
"""Validate `forced_align.extract_drug_span_forced_align` on the 4 items the
user specifically flagged (esomeprazole, talquetamab, Vyloy, Eliquis) before
any full-corpus rerun -- per this project's "validate before scaling"
discipline. Prints the extracted span's timing and writes each clip to the
scratchpad so they can be listened to directly.
"""

from __future__ import annotations

import json
import sys
import wave
from io import BytesIO
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dose_r import dataset  # noqa: E402
from dose_r.forced_align import extract_drug_span_forced_align  # noqa: E402

OUT_DIR = Path("/tmp/claude-0/-home-user-synthio-voice/78dc55a7-ba43-5659-8df2-f9bf7234f4f0/scratchpad")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_DIR = REPO_ROOT / "runs" / "gemini-flash-tts-v1"


def main() -> int:
    recs = [json.loads(l) for l in (RUN_DIR / "results.jsonl").read_text().splitlines() if l.strip()]
    items_by_id = {i.item_id: i for i in dataset.load_items()}

    targets = ["eliquis", "esomeprazole", "talquetamab", "vyloy"]
    for t in targets:
        rec = next((r for r in recs if r["item_id"] == t), None)
        if rec is None:
            print(f"{t}: NOT FOUND in run")
            continue
        item = items_by_id.get(t)
        audio_bytes = (RUN_DIR / rec["audio_path"]).read_bytes()

        print(f"\n=== {t} ===")
        print(f"sentence: {item.sentence}")
        print(f"drug: {item.drug}")

        span = extract_drug_span_forced_align(audio_bytes, item.sentence, item.drug)
        if span is None:
            print("  -> None (could not align)")
            continue

        with wave.open(BytesIO(span), "rb") as w:
            dur = w.getnframes() / w.getframerate()
        print(f"  -> {len(span)} bytes, {dur:.3f}s")

        out_path = OUT_DIR / f"{t}_FORCED_ALIGN_v2.wav"
        out_path.write_bytes(span)
        print(f"  wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
