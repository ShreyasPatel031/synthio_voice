"""LASA guardrail: did the synthesis land closer to a different drug?

Look-Alike/Sound-Alike name confusion is a recognised medication-error category
(WHO Patient Safety Solution #1, the ISMP confused-drug-name list, FDA
post-market name-safety review). A scalar "how close to correct" score cannot
express it: a rendering can be 80% correct and still be a different marketed
drug, and that is far more dangerous than a rendering that is 60% correct and
unambiguously not anything else.

So the guardrail asks a different question. Instead of "how far is this from the
intended name", it asks "of all 286 ingredient names in the DOSE set, which one
is this rendering closest to, and by how much does it beat the runner-up". The
answer is a nearest neighbour and a margin, never a bare boolean.

It reuses the reference IPA/ARPABET layer and the same weighted distance as the
primary scorer, so it needs no additional model and no additional annotation.

One deliberate difference from the primary score: the neighbour search uses the
SEGMENTAL cost only, with no stress term. Stress is not what turns one drug name
into another -- valsartan and sacubitril are not separated by where the emphasis
falls -- and including it would let a stress difference decide which drug a
rendering is nearest to. Distances reported here are therefore slightly smaller
than the corresponding `PhoneticScore.normalized_error`, and are not
interchangeable with it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .distance import DELETE_CONSONANT, INSERT_CONSONANT, bounded_cost, encode
from .phonemes import parse
from .phonetic_scorer import length_scale
from .references import Reference, ReferenceSet

MIN_INDEL_COST = min(DELETE_CONSONANT, INSERT_CONSONANT)

# Two names separated by less than this in normalised error are not reliably
# distinguishable: 0.25 is about one single-feature phoneme substitution's worth
# of separation at the pivot name length. Inside that band the rendering is
# reported as a narrow call even when the intended name still wins.
NARROW_MARGIN = 0.25


@dataclass(frozen=True)
class Neighbor:
    ingredient: str
    distance: float
    variant_index: int


@dataclass(frozen=True)
class ConfusabilityReport:
    item_id: str
    ingredient: str
    intended_distance: float
    neighbors: tuple[Neighbor, ...]

    @property
    def nearest(self) -> Neighbor | None:
        return self.neighbors[0] if self.neighbors else None

    @property
    def margin(self) -> float:
        """How much closer the rendering is to the intended name than to the
        nearest other drug. Negative means it landed on the wrong drug."""
        return self.nearest.distance - self.intended_distance if self.nearest else float("inf")

    @property
    def confusable(self) -> bool:
        return self.margin <= 0.0

    @property
    def narrow(self) -> bool:
        return 0.0 < self.margin < NARROW_MARGIN

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "ingredient": self.ingredient,
            "intended_distance": round(self.intended_distance, 4),
            "nearest_other": self.nearest.ingredient if self.nearest else None,
            "nearest_distance": round(self.nearest.distance, 4) if self.nearest else None,
            "margin": round(self.margin, 4),
            "confusable": self.confusable,
            "narrow": self.narrow,
            "neighbors": [
                {"ingredient": n.ingredient, "distance": round(n.distance, 4)}
                for n in self.neighbors
            ],
        }


def _lower_bound(reference_length: int, hypothesis_length: int) -> float:
    """Cheapest possible normalised error given only the length difference.

    No alignment can cost less than one indel per length unit of difference, so
    a candidate whose bound already exceeds the running best can be skipped
    without running the DP.
    """
    gap = abs(reference_length - hypothesis_length)
    return (gap * MIN_INDEL_COST) / length_scale(reference_length)


class ConfusabilityIndex:
    """Nearest-drug search over the reference set."""

    def __init__(self, references: ReferenceSet, ingredients: list[str] | None = None):
        pool = (
            [references.require(i) for i in ingredients]
            if ingredients is not None
            else list(references)
        )
        self.references = references
        self._pool: dict[str, Reference] = {r.key: r for r in pool}
        self._encoded: dict[str, tuple[tuple[tuple[int, ...], int, float], ...]] = {
            r.key: tuple(
                (encode(v.arpabet), v.index, length_scale(len(v.arpabet)))
                for v in r.variants
            )
            for r in pool
        }

    @property
    def size(self) -> int:
        return len(self._pool)

    def _encode_reference(self, reference: Reference):
        cached = self._encoded.get(reference.key)
        if cached is None:
            cached = tuple(
                (encode(v.arpabet), v.index, length_scale(len(v.arpabet)))
                for v in reference.variants
            )
            self._encoded[reference.key] = cached
        return cached

    def _distance_to(
        self, reference: Reference, hyp: tuple[int, ...], hyp_len: int, ceiling: float
    ) -> tuple[float, int]:
        best, best_index = float("inf"), 0
        for codes, index, scale in self._encode_reference(reference):
            limit = min(best, ceiling)
            if _lower_bound(len(codes), hyp_len) >= limit:
                continue
            cost = bounded_cost(codes, hyp, limit * scale)
            d = cost / scale
            if d < best:
                best, best_index = d, index
        return best, best_index

    def distance(self, ingredient: str, hypothesis: list[str] | str) -> float:
        hyp = encode(parse(hypothesis))
        return self._distance_to(
            self.references.require(ingredient), hyp, len(hyp), float("inf")
        )[0]

    def report(
        self,
        ingredient: str,
        hypothesis: list[str] | str,
        item_id: str = "",
        k: int = 3,
    ) -> ConfusabilityReport:
        hyp = encode(parse(hypothesis))
        hyp_len = len(hyp)
        intended_key = ingredient.lower()
        intended = self._distance_to(
            self.references.require(ingredient), hyp, hyp_len, float("inf")
        )[0]

        found: list[Neighbor] = []
        ceiling = float("inf")
        for key, reference in self._pool.items():
            if key == intended_key:
                continue
            d, variant_index = self._distance_to(reference, hyp, hyp_len, ceiling)
            if d < ceiling:
                found.append(Neighbor(reference.ingredient, d, variant_index))
                found.sort(key=lambda n: n.distance)
                del found[k:]
                if len(found) == k:
                    ceiling = found[-1].distance

        return ConfusabilityReport(item_id, ingredient, intended, tuple(found))

    def neighbor_graph(self, k: int = 1) -> dict[str, list[Neighbor]]:
        """Nearest confusable names for every reference's preferred variant.

        This is a property of the DOSE name set itself, independent of any
        system under test: it says which names are intrinsically hard to keep
        apart, and it is the right denominator for reporting how often a system
        crosses a boundary that was narrow to begin with.
        """
        return {
            reference.ingredient: self.report(
                reference.ingredient, list(reference.preferred.arpabet), k=k
            ).neighbors
            for reference in self._pool.values()
        }
