"""Command line for the DOSE-R judge.

    python -m dose_r.judge.cli score     --hypotheses runs/system-a.jsonl
    python -m dose_r.judge.cli anchors   --n 50
    python -m dose_r.judge.cli neighbors --top 30
    python -m dose_r.judge.cli tripwire  --baseline a.json --candidate b.json

Every subcommand defaults to the SYNTHETIC reference fixture, because the real
reference layer is Workstream 1a's and does not exist yet. Pass `--references
dose_r/references/references.jsonl` once it does; results against the fixture
are for wiring and regression only and must not be reported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import calibration, pipeline
from .confusability import ConfusabilityIndex
from .dataset import load_rows, load_spans
from .fixtures.build_synthetic import OUT_PATH as SYNTHETIC_REFERENCES
from .references import ReferenceSet


def _references(args) -> ReferenceSet:
    path = Path(args.references)
    references = ReferenceSet.load(path)
    if path == SYNTHETIC_REFERENCES:
        print(
            "WARNING: using the synthetic development fixture. Scores from this "
            "run are not reportable.\n"
        )
    return references


def cmd_score(args) -> int:
    references = _references(args)
    rows = load_rows()
    if args.hypotheses:
        hypotheses = pipeline.load_hypotheses(args.hypotheses)
    elif args.simulate is not None:
        hypotheses = pipeline.simulated_hypotheses(rows, references, quality=args.simulate)
    else:
        hypotheses = pipeline.reference_hypotheses(rows, references)
    result = pipeline.run(hypotheses, references, rows, aggregator=args.aggregator)
    out = pipeline.write_artifacts(result, args.out)

    overall = result.agreement["overall"]
    confusable = [c for c in result.confusability.values() if c.confusable]
    narrow = [c for c in result.confusability.values() if c.narrow]
    print(f"spans scored: {overall['n']}  skipped: {len(result.skipped)}")
    print(f"rows scored:  {len(result.rows)} (aggregator={args.aggregator})")
    print(
        f"pass rate     phonetic {overall['phonetic_pass_rate']:.1%}  "
        f"panel {overall['panel_pass_rate']:.1%}"
    )
    print(
        f"agreement     exact {overall['exact_agreement']:.1%}  "
        f"within-1 {overall['within_one']:.1%}  "
        f"pass kappa {overall['pass_kappa']:.3f}"
    )
    print(f"flagged for manual listen-through: {overall['flagged']}")
    print(f"confusable: {len(confusable)}  narrow margin: {len(narrow)}")
    for stratum, stats in result.agreement["by_stratum"].items():
        print(
            f"  {stratum:22s} n={stats['n']:4d} exact={stats['exact_agreement']:6.1%} "
            f"kappa={stats['pass_kappa']:+.3f} flagged={stats['flagged']}"
        )
    print(f"wrote {out}/")
    return 0


def cmd_anchors(args) -> int:
    references = _references(args)
    spans = [s for s in load_spans() if s.ingredient in references]
    index = ConfusabilityIndex(references, sorted({s.ingredient for s in spans}))
    profile = calibration.difficulty_profile(references, index)
    anchors = calibration.sample_anchor_set(spans, profile, n=args.n, seed=args.seed)
    summary = calibration.anchor_set_summary(anchors, profile)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for span in anchors:
            d = profile[span.ingredient.lower()]
            f.write(
                json.dumps(
                    {
                        "item_id": span.item_id,
                        "row_id": span.row_id,
                        "ingredient": span.ingredient,
                        "name_type": span.name_type,
                        "is_combination": span.is_combination,
                        "difficulty": d.band,
                        "phoneme_length": d.phoneme_length,
                        "nearest_distance": round(d.nearest_distance, 4),
                        "reference_confidence": d.confidence,
                        "sentence": span.sentence,
                    }
                )
                + "\n"
            )
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    return 0


def cmd_neighbors(args) -> int:
    references = _references(args)
    ingredients = sorted({s.ingredient for s in load_spans() if s.ingredient in references})
    index = ConfusabilityIndex(references, ingredients)
    graph = index.neighbor_graph(k=1)
    pairs = sorted(
        ((v[0].distance, k, v[0].ingredient) for k, v in graph.items() if v)
    )
    print(f"{index.size} ingredients; closest confusable pairs:")
    for distance, a, b in pairs[: args.top]:
        print(f"  {distance:6.3f}  {a}  ~  {b}")
    return 0


def cmd_tripwire(args) -> int:
    baseline = {k: tuple(v) for k, v in json.loads(Path(args.baseline).read_text()).items()}
    candidate = {k: tuple(v) for k, v in json.loads(Path(args.candidate).read_text()).items()}
    result = calibration.tripwire(baseline, candidate)
    print(json.dumps(result.to_dict(), indent=2))
    return 1 if result.flagged else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dose_r.judge", description=__doc__)
    ap.add_argument("--references", default=str(SYNTHETIC_REFERENCES))
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("score", help="run both scorers end to end")
    s.add_argument("--hypotheses", help="JSONL of system renderings; omit for the gold baseline")
    s.add_argument(
        "--simulate",
        type=float,
        nargs="?",
        const=1.0,
        help="score a synthetic system instead; higher values inject more errors",
    )
    s.add_argument("--aggregator", default="worst", choices=["worst", "mean", "best"])
    s.add_argument("--out", default="artifacts")
    s.set_defaults(func=cmd_score)

    s = sub.add_parser("anchors", help="sample the human-verification anchor set")
    s.add_argument("--n", type=int, default=calibration.ANCHOR_TARGET)
    s.add_argument("--seed", default="dose-r-anchor-v1")
    s.add_argument("--out", default="artifacts/anchor_set.jsonl")
    s.set_defaults(func=cmd_anchors)

    s = sub.add_parser("neighbors", help="intrinsically confusable name pairs in DOSE")
    s.add_argument("--top", type=int, default=25)
    s.set_defaults(func=cmd_neighbors)

    s = sub.add_parser("tripwire", help="check score and confusability moved together")
    s.add_argument("--baseline", required=True)
    s.add_argument("--candidate", required=True)
    s.set_defaults(func=cmd_tripwire)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
