#!/usr/bin/env python
"""Extraction-only sweep recording each item's forced-align span duration and
its drug's target phoneme count, to find outliers BEFORE trusting the full
corpus rescore -- specifically: items with very few phonemes (the "Advair"
failure mode: CTC peakiness truncates short/fast words worst) and items with
an unusually long duration-per-phoneme (the midpoint-boundary fix could, in
principle, over-extend into a real pause on the other side of a gap).
"""

from __future__ import annotations

import json
import sys
import time
import wave
from io import BytesIO
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dose_r import dataset  # noqa: E402
from dose_r import forced_align  # noqa: E402

RUN_DIR = REPO_ROOT / "runs" / "gemini-flash-tts-v1"


def main() -> int:
    recs = [json.loads(l) for l in (RUN_DIR / "results.jsonl").read_text().splitlines() if l.strip()]
    items_by_id = {i.item_id: i for i in dataset.load_items()}

    torch, processor, model = forced_align.phoneme_model._get_model() if False else (None, None, None)
    # (import lazily via the module's own _get_model to reuse its cache)
    from dose_r.scoring import phoneme_model
    torch, processor, model = phoneme_model._get_model()
    vocab = processor.tokenizer.get_vocab()

    rows = []
    t0 = time.perf_counter()
    for n, rec in enumerate(recs, 1):
        item = items_by_id.get(rec["item_id"])
        if item is None or not rec["synthesis"]["ok"] or not rec.get("audio_path"):
            continue
        audio_bytes = (RUN_DIR / rec["audio_path"]).read_bytes()

        word_idx = forced_align._locate_drug_word_indices(item.sentence, item.drug)
        if word_idx is None:
            continue
        target_ids, word_of_token = forced_align._build_target_sequence(item.sentence, vocab)
        n_phon = sum(1 for w in word_of_token if w in word_idx)

        try:
            span = forced_align.extract_drug_span_forced_align(audio_bytes, item.sentence, item.drug)
        except Exception as exc:
            rows.append({"item_id": rec["item_id"], "n_phon": n_phon, "dur": None, "err": str(exc)})
            continue
        if span is None:
            rows.append({"item_id": rec["item_id"], "n_phon": n_phon, "dur": None, "err": "none"})
            continue
        with wave.open(BytesIO(span), "rb") as w:
            dur = w.getnframes() / w.getframerate()
        rows.append({"item_id": rec["item_id"], "n_phon": n_phon, "dur": dur,
                     "dur_per_phon": round(dur / n_phon, 4) if n_phon else None})
        if n % 50 == 0:
            print(f"  {n}/{len(recs)}", flush=True)

    print(f"done in {time.perf_counter()-t0:.1f}s, {len(rows)} rows")
    (REPO_ROOT / "runs" / "forced-align-durations-v1.json").write_text(json.dumps(rows, indent=2))

    ok_rows = [r for r in rows if r["dur"] is not None]
    print(f"\n=== fewest phonemes (highest truncation risk) ===")
    for r in sorted(ok_rows, key=lambda r: r["n_phon"])[:15]:
        print(f"  {r['item_id']:25s} n_phon={r['n_phon']:3d}  dur={r['dur']:.3f}  dur/phon={r['dur_per_phon']:.3f}")

    print(f"\n=== lowest dur/phon (fastest -- most truncation-suspicious) ===")
    for r in sorted(ok_rows, key=lambda r: r["dur_per_phon"])[:15]:
        print(f"  {r['item_id']:25s} n_phon={r['n_phon']:3d}  dur={r['dur']:.3f}  dur/phon={r['dur_per_phon']:.3f}")

    print(f"\n=== highest dur/phon (slowest -- possible over-extension) ===")
    for r in sorted(ok_rows, key=lambda r: -r["dur_per_phon"])[:15]:
        print(f"  {r['item_id']:25s} n_phon={r['n_phon']:3d}  dur={r['dur']:.3f}  dur/phon={r['dur_per_phon']:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
