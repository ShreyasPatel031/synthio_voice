#!/usr/bin/env python
"""Run the DOSE-R benchmark harness.

Examples
--------
  # offline harness self-test, no network or spend
  python scripts/run_benchmark.py --systems mock-perfect mock-truncated --limit 20

  # full cheap-tier stand-in benchmark over all 274 items
  python scripts/run_benchmark.py --tier cheap
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dose_r import config, dataset, report  # noqa: E402
from dose_r.adapters import available_systems  # noqa: E402
from dose_r.runner import BenchmarkRunner, RunConfig  # noqa: E402
from dose_r.scoring import build_scorer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--systems", nargs="+", help="explicit system ids")
    g.add_argument("--tier", choices=["cheap", "control", "all"])
    ap.add_argument("--scorer", default="standin")
    ap.add_argument("--limit", type=int, help="first N items (smoke tests)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--run-id")
    ap.add_argument("--no-audio", action="store_true", help="skip persisting audio")
    ap.add_argument("--notes", default="")
    ap.add_argument("--list", action="store_true", help="list systems and exit")
    args = ap.parse_args()

    if args.list:
        for s in available_systems():
            spec = config.ALL_SYSTEMS[s]
            print(f"  {s:26s} tier={spec.tier:10s} {spec.notes}")
        return 0

    if args.tier == "cheap":
        systems = list(config.CHEAP_TIER)
    elif args.tier == "control":
        systems = list(config.MOCK_TIER)
    elif args.tier == "all":
        systems = available_systems()
    else:
        systems = args.systems

    unknown = [s for s in systems if s not in config.ALL_SYSTEMS]
    if unknown:
        ap.error(f"unknown systems: {unknown}")

    items = dataset.load_items()
    scorer = build_scorer(args.scorer)

    cfg = RunConfig(
        systems=systems, concurrency=args.concurrency, limit=args.limit,
        save_audio=not args.no_audio, notes=args.notes,
        **({"run_id": args.run_id} if args.run_id else {}),
    )
    runner = BenchmarkRunner(cfg, scorer=scorer)

    print(f"run_id   : {cfg.run_id}")
    print(f"systems  : {', '.join(systems)}")
    print(f"items    : {args.limit or len(items)} of {len(items)}")
    print(f"scorer   : {scorer.scorer_id} "
          f"(measures_pronunciation={scorer.measures_pronunciation})")
    print()

    results_path = runner.run(items, progress=lambda m: print(f"  {m}", flush=True))

    records = report.load_records(results_path)
    manifest = json.loads((runner.run_dir / "manifest.json").read_text())
    summaries = report.summarize(records)
    text = report.render_text(summaries, manifest)

    print("\n" + text)
    (runner.run_dir / "report.txt").write_text(text)
    (runner.run_dir / "summary.json").write_text(
        json.dumps({k: v.to_dict() for k, v in summaries.items()}, indent=2)
    )
    print(f"\nartifacts: {runner.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
