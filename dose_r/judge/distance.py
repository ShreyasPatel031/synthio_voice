"""Weighted phoneme edit distance with alignment, decomposition, and stress.

Everything here is denominated in phoneme error units (PEU). 1.0 PEU is one
maximally distant substitution. A voicing slip is ~0.23 PEU, a deletion of a
stressed vowel is 1.15 PEU, misplacing primary stress is ~0.40 PEU. Scores in
`phonetic_scorer.py` are cut on PEU totals, so every constant in this file is a
statement about how bad a particular kind of error is relative to "completely
wrong phoneme", and can be argued with directly.

The alignment is a standard Needleman-Wunsch traceback rather than a bare DP
cost, because the per-phoneme and per-syllable decomposition is the reason this
half of the judge exists at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .phonemes import is_vowel, stress_of, substitution_cost
from .syllables import Syllable, syllabify, syllable_index_of

# Indel costs. Deletions are dearer than insertions: a dropped segment removes
# information the listener needs to identify the drug, whereas an epenthetic
# segment leaves the name recoverable. Vowels are dearer than consonants on both
# sides because a vowel is a syllable nucleus, and losing or gaining a syllable
# changes the shape of the word.
DELETE_VOWEL = 1.15
DELETE_CONSONANT = 0.90
INSERT_VOWEL = 1.00
INSERT_CONSONANT = 0.80

# English reduces unstressed vowels toward schwa, so a substitution between two
# vowels that are both unstressed is largely a transcription artefact rather
# than an audible error.
UNSTRESSED_VOWEL_DISCOUNT = 0.35

# PEU charged for a full primary-to-absent stress flip on one vowel. Relocating
# primary stress costs two flips (~0.40 PEU) -- audible and worth a point, but
# never enough on its own to turn a correct segmental rendering into a failure.
STRESS_FLIP_COST = 0.20
_STRESS_LEVEL = {None: 0.0, 0: 0.0, 2: 0.5, 1: 1.0}

INSERTED = "<ins>"
DELETED = "<del>"


@dataclass(frozen=True)
class Op:
    kind: str  # "match" | "sub" | "del" | "ins"
    ref: str
    hyp: str
    cost: float
    ref_index: int | None
    hyp_index: int | None
    syllable: int

    def describe(self) -> str:
        if self.kind == "match":
            return f"{self.ref} ok"
        if self.kind == "sub":
            return f"{self.ref} -> {self.hyp}"
        if self.kind == "del":
            return f"{self.ref} dropped"
        return f"{self.hyp} inserted"


@dataclass(frozen=True)
class SyllableReport:
    index: int
    reference: str
    segmental_cost: float
    stress_cost: float
    reference_stress: int | None
    hypothesis_stress: int | None
    ops: tuple[Op, ...]

    @property
    def cost(self) -> float:
        return self.segmental_cost + self.stress_cost


@dataclass(frozen=True)
class Alignment:
    reference: tuple[str, ...]
    hypothesis: tuple[str, ...]
    ops: tuple[Op, ...]
    segmental_cost: float
    stress_cost: float
    syllables: tuple[SyllableReport, ...] = field(default=())

    @property
    def total_cost(self) -> float:
        return self.segmental_cost + self.stress_cost

    @property
    def worst_syllable(self) -> SyllableReport | None:
        return max(self.syllables, key=lambda s: s.cost, default=None)

    def errors(self) -> tuple[Op, ...]:
        return tuple(op for op in self.ops if op.kind != "match")


def _delete_cost(phoneme: str) -> float:
    return DELETE_VOWEL if is_vowel(phoneme) else DELETE_CONSONANT


def _insert_cost(phoneme: str) -> float:
    return INSERT_VOWEL if is_vowel(phoneme) else INSERT_CONSONANT


def pair_cost(ref: str, hyp: str) -> float:
    """Segmental substitution cost, with the unstressed-vowel discount applied."""
    cost = substitution_cost(ref, hyp)
    if cost and is_vowel(ref) and is_vowel(hyp):
        if not stress_of(ref) and not stress_of(hyp):
            cost *= UNSTRESSED_VOWEL_DISCOUNT
    return cost


def stress_mismatch(ref: str, hyp: str) -> float:
    """Stress cost for one aligned vowel pair, in PEU."""
    if not (is_vowel(ref) and is_vowel(hyp)):
        return 0.0
    delta = abs(_STRESS_LEVEL[stress_of(ref)] - _STRESS_LEVEL[stress_of(hyp)])
    return STRESS_FLIP_COST * delta


def align(reference: list[str], hypothesis: list[str]) -> Alignment:
    """Minimum-cost alignment of `hypothesis` against `reference`.

    Stress is excluded from the DP objective so that the alignment chosen is the
    one that best explains the *segments*; stress is then charged on top of
    whatever vowel pairs that alignment produced. Letting stress steer the
    alignment would let a stress difference silently rewrite which phonemes are
    considered to correspond.
    """
    n, m = len(reference), len(hypothesis)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    back: list[list[str | None]] = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + _delete_cost(reference[i - 1])
        back[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + _insert_cost(hypothesis[j - 1])
        back[0][j] = "ins"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = dp[i - 1][j - 1] + pair_cost(reference[i - 1], hypothesis[j - 1])
            up = dp[i - 1][j] + _delete_cost(reference[i - 1])
            left = dp[i][j - 1] + _insert_cost(hypothesis[j - 1])
            best = min(diag, up, left)
            dp[i][j] = best
            back[i][j] = "diag" if best == diag else ("del" if best == up else "ins")

    syllables = syllabify(reference)
    ops: list[Op] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = back[i][j]
        if move == "diag":
            ref, hyp = reference[i - 1], hypothesis[j - 1]
            cost = pair_cost(ref, hyp)
            ops.append(
                Op(
                    "match" if cost == 0 else "sub",
                    ref,
                    hyp,
                    cost,
                    i - 1,
                    j - 1,
                    syllable_index_of(syllables, i - 1),
                )
            )
            i, j = i - 1, j - 1
        elif move == "del":
            ref = reference[i - 1]
            ops.append(
                Op("del", ref, DELETED, _delete_cost(ref), i - 1, None,
                   syllable_index_of(syllables, i - 1))
            )
            i -= 1
        else:
            hyp = hypothesis[j - 1]
            ops.append(
                Op("ins", INSERTED, hyp, _insert_cost(hyp), None, j - 1,
                   syllable_index_of(syllables, max(i - 1, 0)))
            )
            j -= 1
    ops.reverse()

    segmental = sum(op.cost for op in ops)
    stress = sum(
        stress_mismatch(op.ref, op.hyp) for op in ops if op.kind in ("match", "sub")
    )

    return Alignment(
        reference=tuple(reference),
        hypothesis=tuple(hypothesis),
        ops=tuple(ops),
        segmental_cost=segmental,
        stress_cost=stress,
        syllables=_syllable_reports(syllables, ops),
    )


def _syllable_reports(
    syllables: list[Syllable], ops: tuple[Op, ...] | list[Op]
) -> tuple[SyllableReport, ...]:
    reports = []
    for syl in syllables:
        own = tuple(op for op in ops if op.syllable == syl.index)
        nucleus_op = next(
            (op for op in own if op.ref_index == syl.start + syl.nucleus_offset), None
        )
        hyp_stress = (
            stress_of(nucleus_op.hyp)
            if nucleus_op and nucleus_op.kind in ("match", "sub") and is_vowel(nucleus_op.hyp)
            else None
        )
        reports.append(
            SyllableReport(
                index=syl.index,
                reference=syl.label(),
                segmental_cost=sum(op.cost for op in own),
                stress_cost=(
                    stress_mismatch(nucleus_op.ref, nucleus_op.hyp)
                    if nucleus_op and nucleus_op.kind in ("match", "sub")
                    else 0.0
                ),
                reference_stress=syl.stress,
                hypothesis_stress=hyp_stress,
                ops=own,
            )
        )
    return tuple(reports)
