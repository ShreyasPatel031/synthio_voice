"""Build the replication-fidelity report for one or more judged TTS runs.

Usage:
    python -m dose_r.report.build_report --run rxpronounce=artifacts/rxpronounce/rows.jsonl ...

Each `--run system_id=path` points at a `rows.jsonl` produced by
`dose_r.judge.pipeline.write_artifacts`. **No such file exists for a real
system in this repo** (see FIDELITY.md) -- this command is here so the report
can be produced the moment one does, without writing any of this analysis
under time pressure after a paid run finally lands.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..strata.build_strata import OUT as STRATA_PATH
from . import fidelity as fid
from . import power

ROOT = Path(__file__).resolve().parents[2]
PUBLISHED_PATH = Path(__file__).resolve().parent / "published_leaderboard.json"


def load_published(path: Path = PUBLISHED_PATH) -> dict[str, float]:
    data = json.loads(path.read_text())
    return {k: v["overall_pass_rate"] for k, v in data["systems"].items()}


def build(run_paths: dict[str, Path], strata_path: Path = STRATA_PATH) -> dict:
    strata = fid.load_strata(strata_path)
    runs = {system_id: fid.load_rows_jsonl(path) for system_id, path in run_paths.items()}
    published = load_published()

    overall = {sid: fid.pass_rate(rows).rate for sid, rows in runs.items()}
    overall = {k: v for k, v in overall.items() if v is not None}

    by_system = {}
    for sid, rows in runs.items():
        by_system[sid] = {
            "overall": fid.pass_rate(rows).to_dict(),
            "by_era": {k: v.to_dict() for k, v in fid.stratified_pass_rate(rows, strata, "era").items()},
            "by_difficulty": {
                k: v.to_dict() for k, v in fid.stratified_pass_rate(rows, strata, "difficulty").items()
            },
            "by_name_type": _by_name_type(rows),
            "reference_confidence_attribution": fid.attribution_by_reference_confidence(rows, strata),
        }

    n_total = len(strata)
    n_low_confidence = sum(1 for s in strata.values() if s.get("era_confidence") == "low")

    return {
        "systems": by_system,
        "gap_vs_published": fid.gap_table(overall, published),
        "rank_correlation": fid.rank_correlation(overall, published),
        "power_analysis": {
            "sampling_only_n274": power.sampling_power(n_total).__dict__,
            "effective_sample_size": power.effective_sample_size(n_total, n_low_confidence),
            "reference_uncertainty_sensitivity": power.propagate_reference_uncertainty(
                n_total, n_low_confidence, p_hat=sum(overall.values()) / len(overall) if overall else 0.7
            ),
        },
        "not_run_on_real_data": True,
    }


def _by_name_type(rows: dict[str, fid.RowResult]) -> dict:
    buckets: dict[str, list[fid.RowResult]] = {}
    for r in rows.values():
        buckets.setdefault(r.name_type, []).append(r)
    return {k: fid.group_stat(v).to_dict() for k, v in buckets.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run", action="append", default=[], metavar="system_id=path",
        help="repeatable; one judged rows.jsonl per system",
    )
    ap.add_argument("--strata", type=Path, default=STRATA_PATH)
    ap.add_argument("--out", type=Path, default=Path("artifacts") / "fidelity_report.json")
    args = ap.parse_args(argv)

    if not args.run:
        print(
            "no --run given: no judged system output exists in this repo yet "
            "(see dose_r/report/FIDELITY.md). Nothing to report.",
            file=sys.stderr,
        )
        return 1

    run_paths = {}
    for spec in args.run:
        system_id, _, path = spec.partition("=")
        if not path:
            print(f"bad --run spec {spec!r}, expected system_id=path", file=sys.stderr)
            return 2
        run_paths[system_id] = Path(path)

    report = build(run_paths, args.strata)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote fidelity report -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
