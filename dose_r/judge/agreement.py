"""Agreement between the two scorers, and the divergence flag file.

DOSE-R publishes two scorers rather than picking one, so the interesting number
is not either score but where they disagree. Items where the phonetic scorer and
the audio panel diverge sharply are exactly the items where one of them is
wrong, and averaging them would destroy that information. Those items are
written out for a human to listen to.

Divergence is defined two ways, and an item is flagged if either fires:

  * a score gap of >= 2 on the 0-5 scale, or
  * disagreement about pass/fail, which is the only distinction DOSE's headline
    number actually uses.

A pass/fail flip at a gap of 1 (3 vs 4) matters far more than a gap of 1 in the
middle of the scale, which is why it gets its own condition.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

PASS_SCORE = 4
DIVERGENCE_GAP = 2


@dataclass(frozen=True)
class ItemAgreement:
    item_id: str
    ingredient: str
    stratum: str
    phonetic_score: int
    panel_score: int
    panel_spread: int

    @property
    def gap(self) -> int:
        return self.phonetic_score - self.panel_score

    @property
    def exact(self) -> bool:
        return self.gap == 0

    @property
    def within_one(self) -> bool:
        return abs(self.gap) <= 1

    @property
    def pass_agreement(self) -> bool:
        return (self.phonetic_score >= PASS_SCORE) == (self.panel_score >= PASS_SCORE)

    @property
    def flagged(self) -> bool:
        return abs(self.gap) >= DIVERGENCE_GAP or not self.pass_agreement

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "ingredient": self.ingredient,
            "stratum": self.stratum,
            "phonetic_score": self.phonetic_score,
            "panel_score": self.panel_score,
            "gap": self.gap,
            "panel_spread": self.panel_spread,
            "pass_agreement": self.pass_agreement,
            "reason": (
                "pass/fail disagreement"
                if not self.pass_agreement
                else f"score gap {abs(self.gap)}"
            ),
        }


def _cohens_kappa(pairs: list[tuple[bool, bool]]) -> float:
    """Chance-corrected agreement on the binary pass decision.

    Raw agreement is misleading when most items pass: two scorers that both say
    "pass" 90% of the time agree 82% of the time by luck alone.
    """
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if expected == 1.0 else (observed - expected) / (1 - expected)


def summarize(items: list[ItemAgreement]) -> dict:
    n = len(items)
    pairs = [
        (i.phonetic_score >= PASS_SCORE, i.panel_score >= PASS_SCORE) for i in items
    ]
    return {
        "n": n,
        "exact_agreement": sum(i.exact for i in items) / n,
        "within_one": sum(i.within_one for i in items) / n,
        "pass_agreement": sum(i.pass_agreement for i in items) / n,
        "pass_kappa": _cohens_kappa(pairs),
        "mean_signed_gap": sum(i.gap for i in items) / n,
        "phonetic_pass_rate": sum(a for a, _ in pairs) / n,
        "panel_pass_rate": sum(b for _, b in pairs) / n,
        "mean_panel_spread": sum(i.panel_spread for i in items) / n,
        "flagged": sum(i.flagged for i in items),
    }


def by_stratum(items: list[ItemAgreement]) -> dict[str, dict]:
    buckets: dict[str, list[ItemAgreement]] = defaultdict(list)
    for item in items:
        buckets[item.stratum].append(item)
    return {name: summarize(group) for name, group in sorted(buckets.items())}


def report(items: list[ItemAgreement]) -> dict:
    return {
        "overall": summarize(items),
        "by_stratum": by_stratum(items),
        "flagged_items": [i.to_dict() for i in items if i.flagged],
    }


def write_flagged(items: list[ItemAgreement], path: str | Path) -> Path:
    """Emit the manual-listen-through worklist.

    Sorted worst-divergence first, because whoever works through it will stop
    partway and should have spent that time on the worst cases.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flagged = sorted(
        (i for i in items if i.flagged),
        key=lambda i: (i.pass_agreement, -abs(i.gap)),
    )
    with path.open("w") as f:
        for item in flagged:
            f.write(json.dumps(item.to_dict()) + "\n")
    return path
