"""The phonetic-distance half of the DOSE-R judge.

Deterministic, auditable, and decomposable to the phoneme and the syllable that
failed. Blind to naturalness, timbre and prosody beyond lexical stress -- that
blindness is the point of running it alongside the audio-LLM panel rather than
instead of it.


The 0-5 mapping
---------------

The scorer produces a cost in phoneme error units (PEU, see `distance.py`),
length-normalised, and cuts it into DOSE's 0-5 ordinal scale. The mapping is a
table of thresholds in this module, not a constant buried in a function, because
it is the single most contestable decision in the judge and it should be
possible to argue with it without reading the implementation.

Normalisation is by the SQUARE ROOT of reference length, pivoted at an 8-phoneme
name (about the median of the DOSE ingredient set). Full length normalisation --
dividing by phoneme count -- is wrong for this benchmark: it makes a single
wrong phoneme almost free in a long name, when in fact one wrong phoneme is
exactly what turns one drug name into another regardless of length. No
normalisation is also wrong: a 22-phoneme name accumulates small transcription
noise that an 6-phoneme name cannot. Square-root scaling gives a 20-phoneme name
1.6x the error budget of an 8-phoneme one rather than 2.5x.

The thresholds are anchored on named error types rather than picked to produce a
target distribution:

    5   <= 0.20   at most a sub-phonemic slip: a merged vowel pair (AA/AO,
                  0.16 PEU), an unstressed-vowel difference, a flap for /t/.
                  No listener distinguishes these from correct.
    4   <= 0.45   at most one single-feature substitution (voicing, 0.23 PEU;
                  one step of place, 0.17-0.38 PEU) or one relocation of primary
                  stress (0.40 PEU). Audible, still unambiguously the right
                  drug. This is DOSE's pass boundary.
    3   <= 0.95   about one whole phoneme wrong, or a dropped consonant
                  (0.90 PEU). The name is recognisable but mispronounced.
    2   <= 1.70   a dropped or spurious syllable (1.15 PEU), or two whole
                  phonemes wrong.
    1   <= 2.80   heavily degraded; several segments wrong.
    0   above     unrecognisable as the target name.

Because DOSE passes at >= 4, the load-bearing threshold is 0.45 PEU: the claim
that one single-feature phoneme slip or one stress relocation still passes, and
two do not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .distance import Alignment, align
from .phonemes import parse
from .references import Reference, ReferenceSet

PIVOT_LENGTH = 8.0
MIN_EFFECTIVE_LENGTH = 4.0

SCORE_THRESHOLDS: tuple[tuple[float, int], ...] = (
    (0.20, 5),
    (0.45, 4),
    (0.95, 3),
    (1.70, 2),
    (2.80, 1),
)
PASS_SCORE = 4


@dataclass(frozen=True)
class VariantScore:
    variant_index: int
    ipa: str
    normalized_error: float
    alignment: Alignment


@dataclass(frozen=True)
class PhoneticScore:
    item_id: str
    ingredient: str
    name_type: str
    score: int
    normalized_error: float
    segmental_cost: float
    stress_cost: float
    reference_length: int
    best_variant_index: int
    best_variant_ipa: str
    variant_count: int
    reference_confidence: str
    alignment: Alignment
    per_variant: tuple[VariantScore, ...]

    @property
    def passed(self) -> bool:
        return self.score >= PASS_SCORE

    @property
    def used_alternate_variant(self) -> bool:
        """True when a non-preferred variant scored the system best.

        This is the evidence for the paper's claim that a system should not be
        failed for a clinically accepted alternate pronunciation.
        """
        return self.best_variant_index != 0

    def worst_syllable(self):
        return self.alignment.worst_syllable

    def to_dict(self) -> dict:
        worst = self.worst_syllable()
        return {
            "item_id": self.item_id,
            "ingredient": self.ingredient,
            "name_type": self.name_type,
            "score": self.score,
            "passed": self.passed,
            "normalized_error": round(self.normalized_error, 4),
            "segmental_cost": round(self.segmental_cost, 4),
            "stress_cost": round(self.stress_cost, 4),
            "reference_length": self.reference_length,
            "reference_confidence": self.reference_confidence,
            "best_variant_index": self.best_variant_index,
            "best_variant_ipa": self.best_variant_ipa,
            "variant_count": self.variant_count,
            "used_alternate_variant": self.used_alternate_variant,
            "reference_arpabet": " ".join(self.alignment.reference),
            "hypothesis_arpabet": " ".join(self.alignment.hypothesis),
            "errors": [
                {
                    "kind": op.kind,
                    "ref": op.ref,
                    "hyp": op.hyp,
                    "cost": round(op.cost, 4),
                    "syllable": op.syllable,
                    "detail": op.describe(),
                }
                for op in self.alignment.errors()
            ],
            "syllables": [
                {
                    "index": s.index,
                    "reference": s.reference,
                    "cost": round(s.cost, 4),
                    "segmental_cost": round(s.segmental_cost, 4),
                    "stress_cost": round(s.stress_cost, 4),
                    "reference_stress": s.reference_stress,
                    "hypothesis_stress": s.hypothesis_stress,
                }
                for s in self.alignment.syllables
            ],
            "worst_syllable": (
                {"index": worst.index, "reference": worst.reference,
                 "cost": round(worst.cost, 4)}
                if worst and worst.cost > 0
                else None
            ),
        }


def length_scale(reference_length: int) -> float:
    return math.sqrt(max(reference_length, MIN_EFFECTIVE_LENGTH) / PIVOT_LENGTH)


def normalize(alignment: Alignment) -> float:
    return alignment.total_cost / length_scale(len(alignment.reference))


def error_to_score(normalized_error: float) -> int:
    for threshold, score in SCORE_THRESHOLDS:
        if normalized_error <= threshold:
            return score
    return 0


def score_against_reference(
    hypothesis: list[str] | str,
    reference: Reference,
    item_id: str = "",
) -> PhoneticScore:
    """Score one recognised phoneme sequence against a reference variant SET.

    Every variant is scored and the best-matching one wins. Taking only the
    preferred variant would penalise systems for clinically accepted alternate
    pronunciations, which is exactly the failure mode DOSE-R argues against.
    """
    hyp = parse(hypothesis)
    per_variant = []
    for variant in reference.variants:
        alignment = align(list(variant.arpabet), hyp)
        per_variant.append(
            VariantScore(variant.index, variant.ipa, normalize(alignment), alignment)
        )

    best = min(per_variant, key=lambda v: (v.normalized_error, v.variant_index))
    return PhoneticScore(
        item_id=item_id,
        ingredient=reference.ingredient,
        name_type=reference.name_type,
        score=error_to_score(best.normalized_error),
        normalized_error=best.normalized_error,
        segmental_cost=best.alignment.segmental_cost,
        stress_cost=best.alignment.stress_cost,
        reference_length=len(best.alignment.reference),
        best_variant_index=best.variant_index,
        best_variant_ipa=best.ipa,
        variant_count=len(reference.variants),
        reference_confidence=reference.confidence,
        alignment=best.alignment,
        per_variant=tuple(per_variant),
    )


class PhoneticScorer:
    """Scores hypotheses for named ingredients against a loaded reference set."""

    def __init__(self, references: ReferenceSet):
        self.references = references

    def score(
        self, ingredient: str, hypothesis: list[str] | str, item_id: str = ""
    ) -> PhoneticScore:
        return score_against_reference(
            hypothesis, self.references.require(ingredient), item_id
        )

    def best_distance(self, ingredient: str, hypothesis: list[str]) -> float:
        """Normalised error to the nearest variant. Used by `confusability.py`."""
        reference = self.references.require(ingredient)
        return min(
            normalize(align(list(v.arpabet), hypothesis)) for v in reference.variants
        )
