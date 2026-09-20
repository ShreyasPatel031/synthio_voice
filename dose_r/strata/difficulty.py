"""Difficulty tiering: easy / medium / hard, approximated from three signals.

DOSE reports 63 easy / 102 medium / 109 hard and does not say how the tiers
were built. None of the public columns carry a difficulty label, so this
module reconstructs one from properties that are actually measurable here:

    phoneme_count   total ARPABET phonemes across a row's ingredient(s), taken
                    from the gold reference layer. A better proxy for spoken
                    complexity than raw character count, since it already
                    accounts for silent letters and digraphs.
    name_length     total characters in the row's name, a coarse tiebreaker
                    within a phoneme-count band.
    usan_stem       the name ends in a recognised USAN/INN stem (-umab,
                    -tinib/-inib, -tide, -zole) -- a marker of coined,
                    non-English pharmaceutical morphology.
    biologic_suffix the FDA's four-letter biologic qualifier ("-vikg"), the
                    clearest possible non-English-morphology signal there is.

The composite score and its two cutoffs were chosen by grid search to land
close to DOSE's marginal counts. That is curve-fitting to a known target, not
validation: matching 63/102/109 (or close to it) shows the tier *sizes* agree,
not that any individual row carries the tier DOSE assigned it. See STRATA.md
for the fit quality and sensitivity to the cutoffs and feature weights.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..judge.phonemes import parse

# "-tinib" subsumes "-inib" in this dataset -- every name ending "-inib" here
# also ends "-tinib" -- so the brief's combined "-tinib/-inib" bucket is one
# stem, not two, to avoid double-counting the same names.
USAN_STEMS = ("umab", "tinib", "tide", "zole")
BIOLOGIC_SUFFIX = re.compile(r"-[a-z]{4}$")

# Chosen by exhaustive grid search over cutoff pairs on this dataset's score
# distribution: the pair minimizing sum(|count - target|) against
# (63, 102, 109). These are exact midpoints between adjacent distinct scores,
# not round numbers -- moving either one even slightly re-crosses a tie band
# of a dozen-plus rows (see STRATA.md, "Tie sensitivity").
LENGTH_WEIGHT = 0.02
STEM_BONUS = 2.0
EASY_MAX = 6.16
MEDIUM_MAX = 10.18


def has_usan_stem(name: str) -> bool:
    low = name.lower()
    return any(low.endswith(stem) for stem in USAN_STEMS)


def has_biologic_suffix(name: str) -> bool:
    return bool(BIOLOGIC_SUFFIX.search(name.lower()))


def phoneme_count(arpabet: str) -> int:
    return len(parse(arpabet))


@dataclass(frozen=True)
class DifficultyFeatures:
    phoneme_count: int
    name_length: int
    usan_stem: bool
    biologic_suffix: bool

    @property
    def score(self) -> float:
        bonus = STEM_BONUS if (self.usan_stem or self.biologic_suffix) else 0.0
        return self.phoneme_count + LENGTH_WEIGHT * self.name_length + bonus

    @property
    def tier(self) -> str:
        score = self.score
        if score <= EASY_MAX:
            return "easy"
        if score <= MEDIUM_MAX:
            return "medium"
        return "hard"


def features_for_row(name: str, ingredients: list[str], arpabet_by_ingredient: dict[str, str]) -> DifficultyFeatures:
    total_phonemes = sum(phoneme_count(arpabet_by_ingredient[ing]) for ing in ingredients)
    stem = any(has_usan_stem(ing) for ing in ingredients)
    suffix = any(has_biologic_suffix(ing) for ing in ingredients)
    return DifficultyFeatures(
        phoneme_count=total_phonemes,
        name_length=len(name),
        usan_stem=stem,
        biologic_suffix=suffix,
    )
