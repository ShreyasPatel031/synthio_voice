#!/usr/bin/env python
"""Score the cached audio of a completed run with a different scorer, without
re-synthesizing anything.

Motivation: `runs/standin-v1` already has 274 items x 4 systems of real,
already-paid-for audio on disk (`audio/<system_id>/<item_id>.wav`,
`audio_path` in each `results.jsonl` record). Swapping in a real pronunciation
scorer (e.g. `asr-roundtrip`) is a scoring-only pass over that audio -- it
must not re-hit the TTS APIs (extra spend, extra latency, and it would leave
the accuracy numbers no longer comparable to the same synthesized bytes the
run already measured latency/cost for).

Writes a new run directory (default `<run_id>-<scorer_id>`) with its own
`results.jsonl`, `manifest.json`, `report.txt` and `summary.json`, built from
the source run's cached audio and synthesis metadata. The source run
directory is never modified.

Examples
--------
  # score a 20-item slice of two systems first (sanity check before spending
  # on the full 274 x 4)
  python scripts/score_existing_run.py --run-id standin-v1 --scorer asr-roundtrip \\
      --systems gtts-standard-c gtts-chirp3hd-achernar --limit 20

  # full rescore
  python scripts/score_existing_run.py --run-id standin-v1 --scorer asr-roundtrip
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dose_r import __version__, config, dataset, report  # noqa: E402
from dose_r.adapters.base import SynthesisResult  # noqa: E402
from dose_r.scoring import build_scorer  # noqa: E402


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _record_to_synthesis_result(rec: dict, audio: bytes | None) -> SynthesisResult:
    """Reconstruct just enough of SynthesisResult for `Scorer.score()` from a
    logged record. Retry/backoff fields are provenance-only here since we are
    not re-synthesizing.
    """
    s = rec["synthesis"]
    return SynthesisResult(
        system_id=rec["system_id"], item_id=rec["item_id"], ok=s["ok"],
        audio=audio, audio_format=s.get("audio_format", "wav"),
        sample_rate_hz=s.get("sample_rate_hz"),
        ttfa_ms=s.get("ttfa_ms"), total_ms=s.get("total_ms"),
        streaming=s.get("streaming", False),
        billable_chars=s.get("billable_chars", 0), cost_usd=s.get("cost_usd"),
        cost_estimated=s.get("cost_estimated", True),
        price_verified=s.get("price_verified", False),
        attempts=s.get("attempts", 1), error=s.get("error"),
        metadata=s.get("metadata", {}),
    )


def _score_one(scorer, item, syn_result, *, max_attempts: int, retry_delay_s: float) -> dict:
    """Score with a couple of retries against transient scorer-side failures
    (e.g. a Cloud STT hiccup) -- distinct from re-synthesis retries, this is
    just "call the judge again", never "resynthesize".
    """
    last = None
    for attempt in range(1, max_attempts + 1):
        last = scorer.score(item, syn_result)
        if last.scoreable or attempt == max_attempts:
            return last.to_record()
        time.sleep(retry_delay_s)
    return last.to_record()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True, help="source run to rescore")
    ap.add_argument("--scorer", required=True, help="registered scorer name")
    ap.add_argument("--systems", nargs="+", help="subset of system ids (default: all in the run)")
    ap.add_argument("--limit", type=int, help="first N cached records per system (smoke tests)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out-run-id", help="output run id (default: '<run-id>-<scorer_id>')")
    ap.add_argument("--score-retries", type=int, default=2,
                    help="retries on a transient (scoreable=False) scorer failure")
    args = ap.parse_args()

    src_dir = config.RUNS_DIR / args.run_id
    src_results = src_dir / "results.jsonl"
    src_manifest = json.loads((src_dir / "manifest.json").read_text())
    if not src_results.exists():
        ap.error(f"{src_results} not found")

    scorer = build_scorer(args.scorer)
    out_run_id = args.out_run_id or f"{args.run_id}-{scorer.scorer_id}"
    out_dir = config.RUNS_DIR / out_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_results_path = out_dir / "results.jsonl"

    items_by_id = {i.item_id: i for i in dataset.load_items()}

    records = _load_jsonl(src_results)
    if args.systems:
        records = [r for r in records if r["system_id"] in args.systems]
    if args.limit:
        # first N per system, preserving the source file's per-system ordering
        capped: list[dict] = []
        seen: dict[str, int] = {}
        for r in records:
            n = seen.get(r["system_id"], 0)
            if n < args.limit:
                capped.append(r)
                seen[r["system_id"]] = n + 1
        records = capped

    # Resume: skip (system_id, item_id) pairs already scored in a prior partial run.
    already: set[tuple[str, str]] = set()
    if out_results_path.exists():
        for r in _load_jsonl(out_results_path):
            already.add((r["system_id"], r["item_id"]))
    todo = [r for r in records if (r["system_id"], r["item_id"]) not in already]

    print(f"source run   : {args.run_id}  ({len(records)} cached records selected)")
    print(f"output run   : {out_run_id}  ({len(already)} already scored, {len(todo)} to do)")
    print(f"scorer       : {scorer.scorer_id} (measures_pronunciation={scorer.measures_pronunciation})")

    def _work(rec: dict):
        item = items_by_id.get(rec["item_id"])
        audio = None
        if rec.get("audio_path"):
            audio_path = src_dir / rec["audio_path"]
            if audio_path.exists():
                audio = audio_path.read_bytes()
        if item is None:
            score_record = None
        else:
            syn_result = _record_to_synthesis_result(rec, audio)
            score_record = _score_one(scorer, item, syn_result,
                                       max_attempts=1 + args.score_retries,
                                       retry_delay_s=1.5)
        out_rec = dict(rec)
        out_rec["run_id"] = out_run_id
        out_rec["source_run_id"] = args.run_id
        out_rec["score"] = score_record
        return out_rec

    n_done = 0
    t0 = time.perf_counter()
    with out_results_path.open("a") as sink, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(_work, rec): rec for rec in todo}
        for fut in as_completed(futures):
            out_rec = fut.result()
            sink.write(json.dumps(out_rec) + "\n")
            sink.flush()
            n_done += 1
            if n_done % 25 == 0 or n_done == len(todo):
                print(f"  scored {n_done}/{len(todo)}", flush=True)

    manifest = {
        "run_id": out_run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dose_r_version": __version__,
        "source_run_id": args.run_id,
        "source_manifest": src_manifest,
        "systems": sorted({r["system_id"] for r in records}),
        "scorer": scorer.scorer_id,
        "scorer_measures_pronunciation": scorer.measures_pronunciation,
        "n_items": len(records),
        "gcp_project": config.GCP_PROJECT,
        "notes": (
            f"Rescoring pass over cached audio from run {args.run_id!r} -- no "
            "re-synthesis, no new TTS spend. Latency/cost figures below are "
            "copied from the source run's synthesis calls, not remeasured."
        ),
        "status": "complete",
        "wall_clock_s": round(time.perf_counter() - t0, 2),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    all_records = _load_jsonl(out_results_path)
    summaries = report.summarize(all_records)
    text = report.render_text(summaries, manifest)
    (out_dir / "report.txt").write_text(text + "\n")
    (out_dir / "summary.json").write_text(
        json.dumps({k: v.to_dict() for k, v in summaries.items()}, indent=2)
    )

    print()
    print(text)
    print()
    print(f"wrote {out_results_path}, {out_dir / 'report.txt'}, {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
