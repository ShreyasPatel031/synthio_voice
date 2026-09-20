"""Loading the gold reference layer produced by Workstream 1a.

The contract (`dose_r/references/references.jsonl`, one object per unique
ingredient):

    {"ingredient", "name_type", "ipa_variants", "arpabet_variants",
     "sources", "confidence", "notes"}

`ipa_variants` and `arpabet_variants` are parallel and ordered most-preferred
first. Both are empty only for a `low`-confidence record with no ground truth
at all (no external source, and no rule-based fallback stands in for one);
`ReferenceSet.load` treats such a record as absent, not as a zero-length
reference to score against, so it surfaces through `missing()` like any other
ingredient nothing was ever found for.

The reference is a variant SET, not a string. A system that says atorvastatin
with the stress pattern of a different but clinically accepted variant has not
made an error, and DOSE-R intends to publish that argument, so every consumer in
this package scores against the best-matching variant and reports which one won.
Nothing may quietly take `arpabet_variants[0]`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .phonemes import parse

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "references" / "references.jsonl"
CONFIDENCE_LEVELS = ("high", "medium", "low")


@dataclass(frozen=True)
class Variant:
    index: int
    arpabet: tuple[str, ...]
    ipa: str

    @property
    def preferred(self) -> bool:
        return self.index == 0


@dataclass(frozen=True)
class Reference:
    ingredient: str
    name_type: str
    variants: tuple[Variant, ...]
    confidence: str
    sources: tuple[dict, ...] = ()
    notes: str = ""

    @property
    def key(self) -> str:
        return self.ingredient.lower()

    @property
    def preferred(self) -> Variant:
        return self.variants[0]


class ReferenceError(ValueError):
    pass


def _reference_from_obj(obj: dict) -> Reference | None:
    """`None` when the ingredient has no ground truth at all (empty
    `arpabet_variants`) -- the caller skips such a row rather than loading a
    `Reference` with nothing in it, since there is no variant to score
    against."""
    arpabet = obj["arpabet_variants"]
    if not arpabet:
        return None
    ipa = obj.get("ipa_variants") or [""] * len(arpabet)
    if len(ipa) != len(arpabet):
        raise ReferenceError(
            f"{obj['ingredient']!r}: {len(ipa)} ipa variants vs "
            f"{len(arpabet)} arpabet variants -- the lists must be parallel"
        )
    confidence = obj.get("confidence", "low")
    if confidence not in CONFIDENCE_LEVELS:
        raise ReferenceError(f"{obj['ingredient']!r}: bad confidence {confidence!r}")

    return Reference(
        ingredient=obj["ingredient"],
        name_type=obj.get("name_type", "generic"),
        variants=tuple(
            Variant(i, tuple(parse(a)), p) for i, (a, p) in enumerate(zip(arpabet, ipa))
        ),
        confidence=confidence,
        sources=tuple(obj.get("sources", ())),
        notes=obj.get("notes", ""),
    )


class ReferenceSet:
    """All reference pronunciations, keyed by case-folded ingredient name."""

    def __init__(self, references: list[Reference]):
        self._by_key: dict[str, Reference] = {}
        for ref in references:
            if ref.key in self._by_key:
                raise ReferenceError(f"duplicate ingredient {ref.ingredient!r}")
            self._by_key[ref.key] = ref

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PATH) -> "ReferenceSet":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"no reference layer at {path}. Workstream 1a owns this file; "
                f"use dose_r/judge/fixtures/synthetic_references.jsonl to develop."
            )
        with path.open() as f:
            objs = [json.loads(line) for line in f if line.strip()]
        refs = [r for r in (_reference_from_obj(o) for o in objs) if r is not None]
        return cls(refs)

    def __contains__(self, ingredient: str) -> bool:
        return ingredient.lower() in self._by_key

    def __len__(self) -> int:
        return len(self._by_key)

    def __iter__(self):
        return iter(self._by_key.values())

    def get(self, ingredient: str) -> Reference | None:
        return self._by_key.get(ingredient.lower())

    def require(self, ingredient: str) -> Reference:
        ref = self.get(ingredient)
        if ref is None:
            raise ReferenceError(f"no reference pronunciation for {ingredient!r}")
        return ref

    def keys(self) -> list[str]:
        return list(self._by_key)

    def missing(self, ingredients) -> list[str]:
        return sorted({i for i in ingredients if i.lower() not in self._by_key})
