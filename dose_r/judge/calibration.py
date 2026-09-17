"""Calibration harness: anchor sampler, overfitting tripwire, tuning seam.

The calibration plan is locked and only part of it can run today:

  1. A human-verified anchor set of 40-60 names, stratified by brand/generic and
     by difficulty, is the only true ground truth available. `sample_anchor_set`
     picks it. BUILT AND RUNNABLE.
  2. Tune the primary score's weighting against that anchor set, never against
     DOSE's nine published numbers -- fitting nine aggregate percentages is
     curve-fitting to noise. `tune_weights` is the entry point. BLOCKED on (1).
  3. DOSE's leaderboard is an out-of-sample check computed after the judge is
     locked. Not implemented here on purpose: there is nothing to build, and a
     convenient function would invite using it early.
  4. Known-good / known-bad separation. See `stress_test.py`. RUNNABLE NOW, and
     run: a judge that cannot separate garbage from gold is not worth
     calibrating.
  5. The overfitting tripwire: primary score and confusability must move
     together. `tripwire` implements it. RUNNABLE NOW.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass

from .confusability import ConfusabilityIndex
from .dataset import Span
from .references import ReferenceSet

ANCHOR_TARGET = 50
ANCHOR_BOUNDS = (40, 60)
DIFFICULTY_BANDS = ("easy", "medium", "hard")
CONFIDENCE_PENALTY = {"high": 0.0, "medium": 0.5, "low": 1.0}

# A system whose primary score rose while its confusability margin did not is
# not more reliable, it is more agreeable to this judge. Any share of such items
# above this fraction of the items that improved gets the run flagged rather
# than certified.
TRIPWIRE_SHARE = 0.20


@dataclass(frozen=True)
class Difficulty:
    ingredient: str
    phoneme_length: int
    nearest_distance: float
    confidence: str
    score: float
    band: str


def _rank_fraction(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    for rank, i in enumerate(order):
        out[i] = rank / max(len(values) - 1, 1)
    return out


def difficulty_profile(
    references: ReferenceSet, index: ConfusabilityIndex
) -> dict[str, Difficulty]:
    """Rank every ingredient by how hard it is to say and to tell apart.

    Three inputs, all available without any human listening: phoneme count (long
    names accumulate more chances to fail), distance to the nearest other drug
    in the set (a name with a close neighbour is the one where an error is
    dangerous), and the reference layer's own confidence (a name whose correct
    pronunciation is itself contested is hard by definition).
    """
    refs = list(references)
    graph = index.neighbor_graph(k=1)

    lengths = [len(r.preferred.arpabet) for r in refs]
    nearest = [
        graph[r.ingredient][0].distance if graph.get(r.ingredient) else float("inf")
        for r in refs
    ]
    length_rank = _rank_fraction([float(x) for x in lengths])
    # Inverted: a SMALL distance to the nearest other drug means HIGH difficulty.
    crowding_rank = [1.0 - x for x in _rank_fraction(nearest)]

    profile = {}
    for i, ref in enumerate(refs):
        score = (
            0.40 * length_rank[i]
            + 0.40 * crowding_rank[i]
            + 0.20 * CONFIDENCE_PENALTY[ref.confidence]
        )
        profile[ref.key] = Difficulty(
            ingredient=ref.ingredient,
            phoneme_length=lengths[i],
            nearest_distance=nearest[i],
            confidence=ref.confidence,
            score=score,
            band="",
        )

    ordered = sorted(profile.values(), key=lambda d: d.score)
    cut = len(ordered) / 3
    for rank, d in enumerate(ordered):
        band = DIFFICULTY_BANDS[min(int(rank // cut), 2)]
        profile[d.ingredient.lower()] = Difficulty(
            d.ingredient, d.phoneme_length, d.nearest_distance, d.confidence,
            d.score, band,
        )
    return profile


def _stable_order(item_id: str, seed: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(f"{seed}\x1f{item_id}".encode(), digest_size=8).digest(), "big"
    )


def sample_anchor_set(
    spans: list[Span],
    profile: dict[str, Difficulty],
    n: int = ANCHOR_TARGET,
    seed: str = "dose-r-anchor-v1",
    min_combinations: int = 2,
) -> list[Span]:
    """Pick the 40-60 spans a human should verify by hand.

    Proportional allocation across name_type x difficulty band, largest
    remainder for the leftovers, deterministic given the seed. Combination-
    product spans are force-included to a floor because they are 3% of the set
    but carry the aggregation decision that the row scores depend on.
    """
    low, high = ANCHOR_BOUNDS
    if not low <= n <= high:
        raise ValueError(f"anchor set must be {low}-{high} items, got {n}")

    def stratum_of(span: Span) -> str:
        band = profile[span.ingredient.lower()].band
        return f"{span.name_type}/{band}"

    buckets: dict[str, list[Span]] = defaultdict(list)
    for span in spans:
        buckets[stratum_of(span)].append(span)
    for group in buckets.values():
        group.sort(key=lambda s: _stable_order(s.item_id, seed))

    total = len(spans)
    exact = {k: len(v) * n / total for k, v in buckets.items()}
    alloc = {k: min(int(v), len(buckets[k])) for k, v in exact.items()}
    remainder = sorted(
        buckets, key=lambda k: (-(exact[k] - int(exact[k])), k)
    )
    i = 0
    while sum(alloc.values()) < n and i < len(remainder) * 4:
        k = remainder[i % len(remainder)]
        if alloc[k] < len(buckets[k]):
            alloc[k] += 1
        i += 1

    chosen = [s for k, count in alloc.items() for s in buckets[k][:count]]

    combos = [s for s in chosen if s.is_combination]
    if len(combos) < min_combinations:
        pool = sorted(
            (s for s in spans if s.is_combination and s not in chosen),
            key=lambda s: _stable_order(s.item_id, seed),
        )
        needed = min_combinations - len(combos)
        droppable = [s for s in reversed(chosen) if not s.is_combination][:needed]
        chosen = [s for s in chosen if s not in droppable] + pool[:needed]

    return sorted(chosen, key=lambda s: s.item_id)


def anchor_set_summary(spans: list[Span], profile: dict[str, Difficulty]) -> dict:
    counts: dict[str, int] = defaultdict(int)
    for s in spans:
        counts[f"{s.name_type}/{profile[s.ingredient.lower()].band}"] += 1
    return {
        "n": len(spans),
        "by_stratum": dict(sorted(counts.items())),
        "combination_spans": sum(s.is_combination for s in spans),
    }


@dataclass(frozen=True)
class TripwireResult:
    improved: int
    improved_without_margin_gain: int
    share: float
    offenders: tuple[str, ...]
    mean_score_delta: float
    mean_margin_delta: float

    @property
    def flagged(self) -> bool:
        return (
            self.share > TRIPWIRE_SHARE
            or (self.mean_score_delta > 0 and self.mean_margin_delta <= 0)
        )

    def to_dict(self) -> dict:
        return {
            "flagged": self.flagged,
            "verdict": "FLAGGED, not certified" if self.flagged else "consistent",
            "improved_items": self.improved,
            "improved_without_margin_gain": self.improved_without_margin_gain,
            "share": round(self.share, 4),
            "mean_score_delta": round(self.mean_score_delta, 4),
            "mean_margin_delta": round(self.mean_margin_delta, 4),
            "offenders": list(self.offenders),
        }


def tripwire(
    baseline: dict[str, tuple[int, float]],
    candidate: dict[str, tuple[int, float]],
) -> TripwireResult:
    """Do the primary score and the confusability margin move together?

    Each mapping is item_id -> (primary 0-5 score, confusability margin). A
    candidate that scores better on the primary metric while its renderings sit
    no further from the nearest other drug name has learned to satisfy this
    judge, not to be more reliable. That is flagged, never certified.
    """
    shared = sorted(set(baseline) & set(candidate))
    if not shared:
        raise ValueError("baseline and candidate share no items")

    improved, offenders = [], []
    score_deltas, margin_deltas = [], []
    for item in shared:
        (s0, m0), (s1, m1) = baseline[item], candidate[item]
        score_deltas.append(s1 - s0)
        margin_deltas.append(m1 - m0)
        if s1 > s0:
            improved.append(item)
            if m1 <= m0:
                offenders.append(item)

    return TripwireResult(
        improved=len(improved),
        improved_without_margin_gain=len(offenders),
        share=len(offenders) / len(improved) if improved else 0.0,
        offenders=tuple(offenders),
        mean_score_delta=sum(score_deltas) / len(shared),
        mean_margin_delta=sum(margin_deltas) / len(shared),
    )


def tune_weights(*_args, **_kwargs):
    """ENTRY POINT FOR CALIBRATION STEP 2 -- DELIBERATELY NOT IMPLEMENTED.

    When the human-verified anchor set exists, tuning fits the free parameters
    of the phonetic scorer -- the feature weights in `phonemes.py`, the indel and
    stress costs in `distance.py`, and the thresholds in `phonetic_scorer.py` --
    to maximise agreement with the anchor labels.

    Binding constraints on whoever implements this:

      * Fit against the anchor set ONLY. DOSE's nine published system scores are
        nine aggregate percentages; fitting a dozen parameters to them is
        curve-fitting to noise and would invalidate the leaderboard as an
        out-of-sample check.
      * Hold out at least a third of the anchor set and report agreement on the
        held-out part, not the fitted part.
      * Re-run `stress_test.py` and `tripwire` after any change. A weighting
        that improves anchor agreement while degrading known-good/known-bad
        separation is overfit to 50 items.
      * Record the parameter values and the anchor-set version in the results;
        a tuned judge with no recorded provenance is not reproducible.
    """
    raise NotImplementedError(
        "Calibration step 2 requires the human-verified anchor set, which does "
        "not exist yet. Run sample_anchor_set() to produce the worklist."
    )
