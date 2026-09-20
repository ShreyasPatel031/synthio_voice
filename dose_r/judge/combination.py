"""Aggregating ingredient-span scores up to a DOSE row score.

Nine of the 274 DOSE rows are combination products naming two or three
ingredients, which is why there are 286 spans. The scoring unit is the span; the
row score is a deliberate choice about how to combine them.

DOSE-R's default is WORST-INGREDIENT (the minimum). The framing of the benchmark
is patient safety: a pharmacist who hears "sacubitril and valsartan" with
valsartan mangled has not received three quarters of a correct prescription,
they have received an ambiguous one. Averaging lets a system hide a failure on
one ingredient behind success on another, and the only ingredient that matters
clinically is the one that was wrong.

`mean` is offered because it is what a reader will assume if we do not say, and
publishing both makes the size of the choice visible -- on 9 of 274 rows it can
only move the headline pass rate by about three percentage points, which is
worth stating rather than arguing about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .dataset import Row

AGGREGATORS: dict[str, Callable[[list[int]], int]] = {
    "worst": min,
    "mean": lambda s: round(sum(s) / len(s)),
    "best": max,
}
DEFAULT_AGGREGATOR = "worst"
PASS_SCORE = 4


@dataclass(frozen=True)
class RowScore:
    row_id: str
    name: str
    name_type: str
    is_combination: bool
    score: int
    aggregator: str
    ingredient_scores: tuple[tuple[str, int], ...]

    @property
    def passed(self) -> bool:
        return self.score >= PASS_SCORE

    @property
    def limiting_ingredient(self) -> str:
        return min(self.ingredient_scores, key=lambda p: p[1])[0]

    @property
    def internal_spread(self) -> int:
        scores = [s for _, s in self.ingredient_scores]
        return max(scores) - min(scores)

    def to_dict(self) -> dict:
        return {
            "row_id": self.row_id,
            "name": self.name,
            "name_type": self.name_type,
            "is_combination": self.is_combination,
            "score": self.score,
            "passed": self.passed,
            "aggregator": self.aggregator,
            "limiting_ingredient": self.limiting_ingredient,
            "internal_spread": self.internal_spread,
            "ingredient_scores": [
                {"ingredient": i, "score": s} for i, s in self.ingredient_scores
            ],
        }


def aggregate_row(
    row: Row, span_scores: dict[str, int], aggregator: str = DEFAULT_AGGREGATOR
) -> RowScore:
    """Combine this row's span scores, keyed by `Span.item_id`."""
    pairs = tuple((span.ingredient, span_scores[span.item_id]) for span in row.spans)
    return RowScore(
        row_id=row.row_id,
        name=row.name,
        name_type=row.name_type,
        is_combination=row.is_combination,
        score=AGGREGATORS[aggregator]([s for _, s in pairs]),
        aggregator=aggregator,
        ingredient_scores=pairs,
    )


def aggregate_all(
    rows: list[Row], span_scores: dict[str, int], aggregator: str = DEFAULT_AGGREGATOR
) -> list[RowScore]:
    return [aggregate_row(r, span_scores, aggregator) for r in rows]


def aggregator_sensitivity(rows: list[Row], span_scores: dict[str, int]) -> dict:
    """How much the headline pass rate depends on the aggregation choice.

    Reported alongside results so the combination decision is visible rather
    than load-bearing and hidden.
    """
    out = {}
    for name in AGGREGATORS:
        scored = aggregate_all(rows, span_scores, name)
        out[name] = {
            "pass_rate": sum(r.passed for r in scored) / len(scored),
            "mean_score": sum(r.score for r in scored) / len(scored),
        }
    combos = [r for r in rows if r.is_combination]
    out["rows_affected"] = len(combos)
    out["max_pass_rate_delta"] = (
        max(v["pass_rate"] for k, v in out.items() if isinstance(v, dict))
        - min(v["pass_rate"] for k, v in out.items() if isinstance(v, dict))
    )
    return out
