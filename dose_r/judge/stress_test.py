"""Known-good / known-bad separation. Calibration step 4, runnable today.

A judge that cannot cleanly separate deliberately garbled speech from the
reference pronunciation is not worth calibrating, so this runs before any tuning
and needs no human data: the reference layer supplies the gold, and the
perturbations supply the garbage.

The classes are deliberately graded rather than binary. "Reference" and "garbled
beyond recognition" is an easy separation that any metric passes. The interesting
cases are the near-gold classes (an accepted alternate variant, a stress shift, a
single voicing slip) which MUST pass, and the near-bad classes (one dropped
syllable, one whole wrong phoneme) which must not. A judge that passes the easy
split and fails the graded one is a judge that only measures gross intelligibility.

Run:  python -m dose_r.judge.stress_test
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .confusability import ConfusabilityIndex
from .dataset import unique_ingredients
from .phonemes import FEATURES, is_vowel, strip_stress, substitution_cost
from .phonetic_scorer import PASS_SCORE, PhoneticScorer
from .references import Reference, ReferenceSet
from .syllables import syllabify

# Expected verdict per class. `True` = must pass (>= 4), `False` = must fail.
EXPECTATIONS = {
    "reference": True,
    "alternate_variant": True,
    "stress_shift": True,
    "voicing_slip": True,
    "vowel_merger": True,
    "swap_adjacent": False,
    "whole_phoneme_wrong": False,
    "dropped_syllable": False,
    "heavy_substitution": False,
    "scrambled": False,
    "other_drug": False,
}
GOOD_CLASSES = tuple(k for k, v in EXPECTATIONS.items() if v)
BAD_CLASSES = tuple(k for k, v in EXPECTATIONS.items() if not v)

VOICING_PAIRS = {
    "P": "B", "B": "P", "T": "D", "D": "T", "K": "G", "G": "K",
    "F": "V", "V": "F", "S": "Z", "Z": "S", "TH": "DH", "DH": "TH",
    "CH": "JH", "JH": "CH", "SH": "ZH", "ZH": "SH",
}
MERGER_PAIRS = {"AA": "AO", "AO": "AA", "IH": "IY", "IY": "IH", "UH": "UW", "UW": "UH"}


def _restress(phoneme: str, level: int) -> str:
    return f"{strip_stress(phoneme)}{level}" if is_vowel(phoneme) else phoneme


def _farthest(phoneme: str) -> str:
    bare = strip_stress(phoneme)
    target = max(FEATURES, key=lambda p: substitution_cost(bare, p))
    return f"{target}0" if is_vowel(target) else target


def _stress_shift(seq: list[str]) -> list[str] | None:
    nuclei = [i for i, p in enumerate(seq) if is_vowel(p)]
    if len(nuclei) < 2:
        return None
    primary = next((i for i in nuclei if p_stress(seq[i]) == 1), nuclei[0])
    target = nuclei[(nuclei.index(primary) + 1) % len(nuclei)]
    out = list(seq)
    out[primary] = _restress(out[primary], 0)
    out[target] = _restress(out[target], 1)
    return out


def p_stress(phoneme: str) -> int:
    return int(phoneme[-1]) if phoneme[-1].isdigit() else -1


def _first_where(seq: list[str], table: dict[str, str]) -> list[str] | None:
    for i, p in enumerate(seq):
        bare = strip_stress(p)
        if bare in table:
            out = list(seq)
            out[i] = (
                f"{table[bare]}{p[-1]}" if p[-1].isdigit() else table[bare]
            )
            return out
    return None


def perturb(
    reference: Reference, kind: str, rng: random.Random, neighbor: Reference | None
) -> list[str] | None:
    """Build one hypothesis of the named class, or None if inapplicable."""
    seq = list(reference.preferred.arpabet)

    if kind == "reference":
        return seq
    if kind == "alternate_variant":
        return list(reference.variants[1].arpabet) if len(reference.variants) > 1 else None
    if kind == "stress_shift":
        return _stress_shift(seq)
    if kind == "voicing_slip":
        return _first_where(seq, VOICING_PAIRS)
    if kind == "vowel_merger":
        return _first_where(seq, MERGER_PAIRS)
    if kind == "swap_adjacent":
        if len(seq) < 4:
            return None
        i = rng.randrange(1, len(seq) - 2)
        seq[i], seq[i + 1] = seq[i + 1], seq[i]
        return seq
    if kind == "whole_phoneme_wrong":
        i = rng.randrange(len(seq))
        seq[i] = _farthest(seq[i])
        return seq
    if kind == "dropped_syllable":
        syllables = syllabify(seq)
        if len(syllables) < 2:
            return None
        victim = syllables[rng.randrange(len(syllables))]
        return seq[: victim.start] + seq[victim.end :]
    if kind == "heavy_substitution":
        n = max(2, round(0.30 * len(seq)))
        for i in rng.sample(range(len(seq)), min(n, len(seq))):
            seq[i] = _farthest(seq[i])
        return seq
    if kind == "scrambled":
        rng.shuffle(seq)
        return seq
    if kind == "other_drug":
        return list(neighbor.preferred.arpabet) if neighbor else None
    raise ValueError(f"unknown perturbation {kind!r}")


@dataclass(frozen=True)
class Trial:
    ingredient: str
    kind: str
    score: int
    normalized_error: float
    margin: float
    expected_pass: bool

    @property
    def passed(self) -> bool:
        return self.score >= PASS_SCORE

    @property
    def correct(self) -> bool:
        return self.passed == self.expected_pass


def run(
    references: ReferenceSet,
    ingredients: list[str] | None = None,
    seed: int = 20260917,
    with_confusability: bool = True,
) -> list[Trial]:
    scorer = PhoneticScorer(references)
    index = ConfusabilityIndex(references, ingredients)
    names = ingredients if ingredients is not None else [r.ingredient for r in references]
    graph = index.neighbor_graph(k=1)

    trials = []
    for name in names:
        reference = references.require(name)
        neighbors = graph.get(reference.ingredient)
        neighbor = references.get(neighbors[0].ingredient) if neighbors else None
        rng = random.Random(f"{seed}:{reference.key}")
        for kind in EXPECTATIONS:
            hyp = perturb(reference, kind, rng, neighbor)
            if hyp is None:
                continue
            scored = scorer.score(name, hyp)
            margin = (
                index.report(name, hyp, k=1).margin if with_confusability else float("nan")
            )
            trials.append(
                Trial(reference.ingredient, kind, scored.score, scored.normalized_error,
                      margin, EXPECTATIONS[kind])
            )
    return trials


def _auc(good: list[float], bad: list[float]) -> float:
    """P(a known-bad rendering scores worse than a known-good one), ties at 0.5.

    Computed on the continuous normalised error rather than the 0-5 score, so it
    measures the separation the metric achieves independently of where the
    thresholds happen to sit.
    """
    wins = sum(
        (b > g) + 0.5 * (b == g) for g in good for b in bad
    )
    return wins / (len(good) * len(bad))


def summarize(trials: list[Trial]) -> dict:
    by_kind: dict[str, list[Trial]] = defaultdict(list)
    for t in trials:
        by_kind[t.kind].append(t)

    good = [t.normalized_error for t in trials if t.expected_pass]
    bad = [t.normalized_error for t in trials if not t.expected_pass]
    good_scores = [t.score for t in trials if t.expected_pass]
    bad_scores = [t.score for t in trials if not t.expected_pass]

    per_class = {}
    for kind, group in by_kind.items():
        per_class[kind] = {
            "n": len(group),
            "expected_pass": EXPECTATIONS[kind],
            "pass_rate": sum(t.passed for t in group) / len(group),
            "correct_rate": sum(t.correct for t in group) / len(group),
            "mean_score": sum(t.score for t in group) / len(group),
            "mean_error": sum(t.normalized_error for t in group) / len(group),
            "confusable_rate": sum(t.margin <= 0 for t in group) / len(group),
        }

    return {
        "n_trials": len(trials),
        "separation_auc": _auc(good, bad),
        "good": {
            "n": len(good),
            "pass_rate": sum(s >= PASS_SCORE for s in good_scores) / len(good_scores),
            "mean_score": sum(good_scores) / len(good_scores),
            "max_error": max(good),
        },
        "bad": {
            "n": len(bad),
            "pass_rate": sum(s >= PASS_SCORE for s in bad_scores) / len(bad_scores),
            "mean_score": sum(bad_scores) / len(bad_scores),
            "min_error": min(bad),
        },
        "error_gap": min(bad) - max(good),
        "overall_correct_rate": sum(t.correct for t in trials) / len(trials),
        "by_class": dict(sorted(per_class.items(), key=lambda kv: -EXPECTATIONS[kv[0]])),
    }


def main() -> int:
    import argparse

    from .fixtures.build_synthetic import OUT_PATH as SYNTHETIC

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--references", default=str(SYNTHETIC))
    ap.add_argument("--out", default="artifacts/stress_test.json")
    args = ap.parse_args()

    references = ReferenceSet.load(args.references)
    trials = run(references, ingredients=[
        i for i in unique_ingredients() if i in references
    ])
    report = summarize(trials)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"references: {args.references}")
    print(f"trials: {report['n_trials']}  AUC: {report['separation_auc']:.4f}")
    print(
        f"known-good pass {report['good']['pass_rate']:.1%}  "
        f"known-bad pass {report['bad']['pass_rate']:.1%}  "
        f"error gap {report['error_gap']:+.3f}"
    )
    for kind, stats in report["by_class"].items():
        want = "pass" if stats["expected_pass"] else "fail"
        print(
            f"  {kind:22s} n={stats['n']:4d} want={want:4s} "
            f"pass={stats['pass_rate']:6.1%} correct={stats['correct_rate']:6.1%} "
            f"mean={stats['mean_score']:.2f} confusable={stats['confusable_rate']:5.1%}"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
