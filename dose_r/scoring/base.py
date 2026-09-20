"""The scoring contract.

Workstream 2 owns the runner; Workstream 1 owns the judge. This module is the seam
between them. A judge is anything implementing `Scorer`, so the hybrid judge
(phonetic-distance + 3-judge audio-LLM panel) drops in without touching the runner.

Scale is fixed to DOSE's: 0-5, pass at >= 4, so Workstream 1d's comparison against
the published leaderboard stays apples-to-apples.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..adapters.base import SynthesisResult
from ..dataset import DoseItem

PASS_THRESHOLD = 4.0
SCALE_MIN, SCALE_MAX = 0.0, 5.0


@dataclass
class ScoreResult:
    """One judgement of one synthesized item."""

    scorer_id: str
    item_id: str
    system_id: str

    score: float | None                 # 0-5, None when unscoreable
    scoreable: bool = True
    error: str | None = None

    # Per-component sub-scores (e.g. {"phonetic": 3.8, "llm_panel_median": 4.0}).
    # Kept alongside the combined score because the plan explicitly retains the raw
    # distribution rather than collapsing everything to pass/fail.
    components: dict[str, float] = field(default_factory=dict)

    # Confidence inherited from the gold-reference layer (high|medium|low).
    # Low-confidence references are reported separately so a soft spot in the gold
    # layer is not mistaken for a model failure.
    reference_confidence: str = "unknown"

    # Confusability guardrail: does this land closer to a *different* drug name?
    # Populated by Workstream 1's reference layer; None until then.
    confusable_with: str | None = None
    confusability_margin: float | None = None

    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool | None:
        if not self.scoreable or self.score is None:
            return None
        return self.score >= PASS_THRESHOLD

    def to_record(self) -> dict[str, Any]:
        return {
            "scorer_id": self.scorer_id,
            "item_id": self.item_id,
            "system_id": self.system_id,
            "score": self.score,
            "passed": self.passed,
            "scoreable": self.scoreable,
            "error": self.error,
            "components": self.components,
            "reference_confidence": self.reference_confidence,
            "confusable_with": self.confusable_with,
            "confusability_margin": self.confusability_margin,
            "notes": self.notes,
            "metadata": self.metadata,
        }


class Scorer(ABC):
    """Base class for a judge."""

    #: Set False on any scorer that does not actually measure pronunciation.
    #: The report generator refuses to present such numbers as DOSE-comparable.
    measures_pronunciation: bool = True

    @property
    @abstractmethod
    def scorer_id(self) -> str: ...

    @abstractmethod
    def score(self, item: DoseItem, result: SynthesisResult) -> ScoreResult: ...

    def score_batch(
        self, pairs: list[tuple[DoseItem, SynthesisResult]]
    ) -> list[ScoreResult]:
        """Override for judges that batch efficiently (e.g. an LLM panel)."""
        return [self.score(item, res) for item, res in pairs]
