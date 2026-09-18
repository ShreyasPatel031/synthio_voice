#!/usr/bin/env python
"""Score cached synthesized audio with the audio-LLM panel judge, across one
or more previous runs, into one combined report.

This does NOT synthesize anything -- every system it scores must already have
cached audio from a prior `run_benchmark.py` run. That keeps synthesis spend
(TTS) and judging spend (LLM calls) as two separate, independently-repeatable
steps: re-scoring with a different judge model never re-pays for audio.

Usage:
    python scripts/score_with_llm_panel.py \\
        --source runs/standin-v1 \\
        --source runs/gemini-flash-tts-v1 \\
        --judge-model gemini-2.5-flash \\
        --out runs/llm-panel-flash-v1
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dose_r import report  # noqa: E402
from dose_r.adapters.base import SynthesisResult  # noqa: E402
from dose_r.dataset import DoseItem  # noqa: E402
from dose_r.scoring.llm_panel import AudioLLMPanelScorer  # noqa: E402


def load_source_records(source_dir: Path) -> list[dict]:
    records = report.load_records(source_dir / "results.jsonl")
    for r in records:
        r["_source_dir"] = str(source_dir)
    return records


def already_scored(out_path: Path) -> set[tuple[str, str]]:
    if not out_path.exists():
        return set()
    done = set()
    for line in out_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("score") and rec["score"].get("scoreable") is not None:
            done.add((rec["system_id"], rec["item_id"]))
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", required=True,
                    help="a prior run directory with results.jsonl + audio/; repeatable")
    ap.add_argument("--judge-model", default="gemini-2.5-flash")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--limit", type=int, help="cap total records (smoke tests)")
    args = ap.parse_args()

    all_records: list[dict] = []
    for src in args.source:
        recs = load_source_records(Path(src))
        print(f"  loaded {len(recs)} records from {src}")
        all_records.extend(recs)
    if args.limit:
        all_records = all_records[: args.limit]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"

    done = already_scored(results_path)
    if done:
        print(f"  resuming: {len(done)} already scored")
    todo = [r for r in all_records if (r["system_id"], r["item_id"]) not in done]
    print(f"  to score: {len(todo)} of {len(all_records)}")

    scorer = AudioLLMPanelScorer(judge_models=(args.judge_model,))

    def score_one(rec: dict) -> dict:
        item = DoseItem(drug=rec["drug"], name_type=rec["name_type"],
                        sentence=rec["sentence"], drug_raw=rec.get("drug_raw", rec["drug"]))
        synth = rec["synthesis"]
        audio_path = Path(rec["_source_dir"]) / rec["audio_path"] if rec.get("audio_path") else None
        audio_bytes = audio_path.read_bytes() if audio_path and audio_path.exists() else None

        result = SynthesisResult(
            system_id=rec["system_id"], item_id=rec["item_id"], ok=synth["ok"] and audio_bytes is not None,
            audio=audio_bytes, audio_format=synth.get("audio_format", "wav"),
            sample_rate_hz=synth.get("sample_rate_hz"),
            error=synth.get("error") or ("cached audio missing on disk" if synth["ok"] and not audio_bytes else None),
        )
        score = scorer.score(item, result)
        out = {k: v for k, v in rec.items() if not k.startswith("_")}
        out["score"] = score.to_record()
        out["judge_model"] = args.judge_model
        return out

    n_done = 0
    with results_path.open("a") as sink, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(score_one, r): r for r in todo}
        for fut in as_completed(futures):
            rec = futures[fut]
            try:
                out = fut.result()
            except Exception as exc:
                out = dict(rec)
                out.pop("_source_dir", None)
                out["score"] = {"scoreable": False, "error": f"scorer crashed: {exc}"}
            sink.write(json.dumps(out) + "\n")
            sink.flush()
            n_done += 1
            if n_done % 50 == 0 or n_done == len(todo):
                print(f"  {n_done}/{len(todo)}")

    usage = scorer.usage_summary()
    (out_dir / "judge_usage.json").write_text(json.dumps(usage, indent=2))
    print(f"\ntoken usage: {usage}")

    records = report.load_records(results_path)
    summaries = report.summarize(records)
    manifest = {"run_id": out_dir.name, "scorer": "llm-panel-v1",
               "scorer_measures_pronunciation": True, "judge_model": args.judge_model,
               "sources": args.source}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    text = report.render_text(summaries, manifest)
    print("\n" + text)
    (out_dir / "report.txt").write_text(text)
    (out_dir / "summary.json").write_text(
        json.dumps({k: v.to_dict() for k, v in summaries.items()}, indent=2)
    )
    print(f"\nartifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
