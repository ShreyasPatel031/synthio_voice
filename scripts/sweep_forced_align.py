#!/usr/bin/env python
"""Extraction-only sweep of `forced_align.extract_drug_span_forced_align` over
all 274 Gemini Flash TTS items -- no scoring, just: does it produce a span,
how long is it, and does anything raise. Run before committing to a full
re-score pass, per this project's "validate before scaling" discipline.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
import wave
from io import BytesIO
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dose_r import dataset  # noqa: E402
from dose_r.forced_align import extract_drug_span_forced_align  # noqa: E402

RUN_DIR = REPO_ROOT / "runs" / "gemini-flash-tts-v1"


def main() -> int:
    recs = [json.loads(l) for l in (RUN_DIR / "results.jsonl").read_text().splitlines() if l.strip()]
    items_by_id = {i.item_id: i for i in dataset.load_items()}

    ok, none_count, err_count = 0, 0, 0
    durations = []
    errors = []
    t0 = time.perf_counter()
    for n, rec in enumerate(recs, 1):
        item = items_by_id.get(rec["item_id"])
        if item is None or not rec["synthesis"]["ok"] or not rec.get("audio_path"):
            continue
        audio_bytes = (RUN_DIR / rec["audio_path"]).read_bytes()
        try:
            span = extract_drug_span_forced_align(audio_bytes, item.sentence, item.drug)
        except Exception as exc:
            err_count += 1
            errors.append((rec["item_id"], f"{type(exc).__name__}: {exc}"))
            continue
        if span is None:
            none_count += 1
            continue
        with wave.open(BytesIO(span), "rb") as w:
            durations.append(w.getnframes() / w.getframerate())
        ok += 1
        if n % 50 == 0:
            print(f"  {n}/{len(recs)}  ok={ok} none={none_count} err={err_count}", flush=True)

    print(f"\ndone in {time.perf_counter()-t0:.1f}s")
    print(f"ok={ok}  none={none_count}  err={err_count}  total_considered={ok+none_count+err_count}")
    if durations:
        durations.sort()
        n = len(durations)
        print(f"span duration: min={durations[0]:.3f} p10={durations[int(n*0.1)]:.3f} "
              f"median={durations[n//2]:.3f} p90={durations[int(n*0.9)]:.3f} max={durations[-1]:.3f}")
    if errors:
        print("\nerrors:")
        for item_id, msg in errors[:20]:
            print(f"  {item_id}: {msg}")

    out = {"ok": ok, "none": none_count, "err": err_count, "durations": durations, "errors": errors}
    (REPO_ROOT / "runs" / "forced-align-sweep-v1.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
