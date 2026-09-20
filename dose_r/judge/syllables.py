"""Syllabification of ARPABET sequences by the Maximal Onset Principle.

The scorer has to be able to say *which syllable* failed, not just that the name
was wrong -- "the system drops the third syllable of tenofovir" is an actionable
finding, "score 3.2" is not. Syllable boundaries are derived rather than stored
so that the reference layer only has to supply phonemes.

The algorithm is the standard one: every vowel is a nucleus; each intervocalic
consonant cluster is split so that the longest legal English onset attaches to
the following syllable and the remainder closes the preceding one.
"""

from __future__ import annotations

from dataclasses import dataclass

from .phonemes import is_vowel, strip_stress

# Legal word-initial English onsets of length 2 and 3. Anything longer than the
# longest match here has to close the previous syllable.
ONSETS_3 = {
    ("S", "P", "R"), ("S", "P", "L"), ("S", "T", "R"), ("S", "K", "R"),
    ("S", "K", "W"), ("S", "K", "L"), ("S", "P", "Y"), ("S", "T", "Y"),
    ("S", "K", "Y"),
}

ONSETS_2 = {
    ("P", "R"), ("B", "R"), ("T", "R"), ("D", "R"), ("K", "R"), ("G", "R"),
    ("F", "R"), ("TH", "R"), ("SH", "R"),
    ("P", "L"), ("B", "L"), ("K", "L"), ("G", "L"), ("F", "L"), ("S", "L"),
    ("T", "W"), ("D", "W"), ("K", "W"), ("G", "W"), ("S", "W"), ("TH", "W"),
    ("P", "Y"), ("B", "Y"), ("K", "Y"), ("G", "Y"), ("F", "Y"), ("V", "Y"),
    ("M", "Y"), ("N", "Y"), ("HH", "Y"), ("L", "Y"),
    ("S", "P"), ("S", "T"), ("S", "K"), ("S", "M"), ("S", "N"), ("S", "F"),
    ("Z", "L"), ("V", "R"), ("DH", "R"),
}


@dataclass(frozen=True)
class Syllable:
    index: int
    phonemes: tuple[str, ...]
    start: int
    end: int
    nucleus_offset: int

    @property
    def nucleus(self) -> str:
        return self.phonemes[self.nucleus_offset]

    @property
    def stress(self) -> int | None:
        last = self.nucleus[-1]
        return int(last) if last.isdigit() else None

    def label(self) -> str:
        return " ".join(self.phonemes)


def _longest_legal_onset(cluster: list[str]) -> int:
    """How many phonemes from the END of `cluster` may open the next syllable."""
    bare = tuple(strip_stress(p) for p in cluster)
    if len(bare) >= 3 and bare[-3:] in ONSETS_3:
        return 3
    if len(bare) >= 2 and bare[-2:] in ONSETS_2:
        return 2
    return 1 if bare else 0


def syllabify(phonemes: list[str]) -> list[Syllable]:
    """Split an ARPABET sequence into syllables.

    A sequence with no vowel at all (a recogniser can emit one) is returned as a
    single degenerate syllable so that downstream attribution never loses
    phonemes.
    """
    nuclei = [i for i, p in enumerate(phonemes) if is_vowel(p)]
    if not nuclei:
        return (
            [Syllable(0, tuple(phonemes), 0, len(phonemes), 0)] if phonemes else []
        )

    boundaries = [0]
    for left, right in zip(nuclei, nuclei[1:]):
        cluster = phonemes[left + 1 : right]
        onset_len = _longest_legal_onset(cluster) if cluster else 0
        boundaries.append(right - onset_len)
    boundaries.append(len(phonemes))

    syllables = []
    for idx, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        nucleus_offset = next(i for i in range(end - start) if is_vowel(phonemes[start + i]))
        syllables.append(
            Syllable(idx, tuple(phonemes[start:end]), start, end, nucleus_offset)
        )
    return syllables


def syllable_index_of(syllables: list[Syllable], position: int) -> int:
    """Which syllable a reference phoneme position belongs to."""
    for syl in syllables:
        if syl.start <= position < syl.end:
            return syl.index
    return syllables[-1].index if syllables else 0
