"""Phoneme-level pronunciation similarity -- a pure, offline-testable module.

Why this exists
----------------
`dose_r/scoring/asr_roundtrip.py` scores pronunciation by round-tripping
synthesized audio through Google Cloud Speech-to-Text, a word-level
recognizer constrained to a fixed English dictionary. `docs/
REFERENCE_AUDIO_GROUNDING.md` measured that this recognizer fails to
transcribe a *human* correctly saying a drug name 44.3% of the time
(98/176 reference clips) -- it "autocorrects" an unfamiliar name to the
nearest real word or phrase regardless of how well it was pronounced. That
is a structural ceiling, not noise: no coined drug name that isn't in the
recognizer's dictionary can ever be transcribed back correctly, so those
98 names are unscoreable by that method for anyone, TTS or human.

This module compares raw phoneme sequences instead of dictionary words
(produced by `phoneme_model.transcribe_phonemes()`, a wav2vec2 CTC model
fine-tuned to emit espeak/IPA phoneme symbols with no notion of "valid
English word" to fall back on). Because there is no dictionary to snap to,
a coined name that is pronounced clearly should come back as roughly the
right *sounds* even though it would never come back as the right *word*
under word-level ASR.

This is a proxy, not ground truth -- see `phoneme_model.py`'s module
docstring for the phoneme recognizer's own error modes and vowel-merging
biases. Nothing in this module claims to have solved pronunciation
scoring; it trades one bias (word-dictionary autocorrection) for a
different, much smaller one (phoneme-recognizer noise and a coarser,
language-general symbol inventory).

Cost
----
This is local computation over two already-decoded phoneme strings -- zero
network calls, zero API spend, whether it's called once or a million
times. (The upstream audio->phoneme step in `phoneme_model.py` is also a
local model run, with no per-clip cost -- see that module's docstring.)

Tokenization
------------
espeak/wav2vec2-lv-60-espeak-cv-ft's decoded output is a single string with
one phoneme symbol per whitespace-separated token, confirmed empirically,
e.g. `transcribe_phonemes()` on the reference clip for "Abilify" returns
`"ɐ b ɪ l ʌ f aɪ"` -- seven tokens, including multi-character IPA symbols
such as the diphthong "aɪ" kept as one token, not one-character-per-token.
So `phoneme_distance()` tokenizes on whitespace (`str.split()`), never on
individual characters -- splitting on characters would incorrectly cut
"aɪ" into "a" and "ɪ", two different vowel-quality tokens, double-counting
a single phoneme as an edit.

`phoneme_distance()`: Phoneme Error Rate
------------------------------------------
Levenshtein (insert/delete/substitute) edit distance between the two
whitespace-tokenized phoneme sequences, normalized by the length of the
longer sequence:

    PER = edit_distance(expected_tokens, recognized_tokens) / max(1, max(len(expected_tokens), len(recognized_tokens)))

This is literally Phoneme Error Rate, the standard ASR evaluation metric,
repurposed here as a pronunciation-similarity measure: comparing a
recognizer's phoneme hypothesis for a synthesized clip against its own
phoneme hypothesis for a human reference clip of the same name, instead of
against a ground-truth phoneme transcript (which this project does not
have). Normalizing by the longer sequence (rather than the expected
sequence's length, the usual ASR-PER convention) keeps the metric bounded
in [0, 1] even when the recognized sequence is much longer than expected
(e.g. a spurious extra syllable), rather than letting PER exceed 1.0 in
that case -- important here because `score_phoneme_match()` needs a fixed
input range to map onto the project's fixed 0-5 output scale.

`score_phoneme_match()`: PER -> 0-5 scale
--------------------------------------------
Mirrors how `asr_roundtrip.score_pronunciation()` documents its own
mapping rationale, so the two scorers' outputs stay legible side by side
even though they measure different things.

  - PER == 0 (token sequences identical) -> 5.0. The strongest possible
    signal this method can give: the recognizer heard exactly the same
    phoneme sequence for the synthesized clip as it did for a human saying
    the name.
  - PER >= 1.0 (edit distance at or past total replacement of the longer
    sequence) -> 0.0.
  - In between, `score = 5.0 * (1 - PER) ** GAMMA` with `GAMMA = 1.6`.
    A linear `5.0 * (1 - PER)` was considered and rejected: PER counts
    every substituted phoneme with equal weight, but a single wrong vowel
    in an otherwise-correct word (PER ~0.15 for a 7-phoneme name) is a much
    smaller pronunciation problem than dropping or swapping a third of the
    name's sounds (PER ~0.33) -- and given this model's own documented bias
    toward confusing *adjacent* reduced-vowel symbols (`phoneme_model.py`),
    treating a small PER as proportionally forgivable is intentional, not
    just a curve-fitting choice. Concretely: PER=0.1 -> 4.224 (still a firm
    pass), PER=0.2 -> 3.499, PER=0.4 -> 2.208, PER=0.6 -> 1.154. The convex
    curve (`GAMMA > 1`) is still concave-down relative to `1-PER`, so it
    sits above a straight line `5*(1-PER)` for every PER in (0, 1): a
    one-vowel slip is scored somewhat *more* forgivingly than linear
    (PER=0.1 -> 4.224 vs. linear's 4.5 -- both comfortably pass, so the
    difference is not the point there), while the gap widens as PER grows,
    so a name with a third to a half of its phonemes wrong is pushed
    further below the pass line than a linear map would put it (PER=0.4 ->
    2.208 vs. linear's 3.0; PER=0.5 -> 1.649 vs. linear's 2.5). That
    compounding-errors-are-worse-than-proportional shape, not the exact
    exponent, is the deliberate choice; GAMMA=1.6 is a reasonable pick
    within it, not a fitted constant.
  - `PASS_THRESHOLD = 4.0` (from `dose_r.scoring.base`) is crossed at
    PER ~= 0.130 under this curve -- i.e. roughly one wrong phoneme out of
    every 7-8, consistent with "the name is recognizably, but not
    perfectly, pronounced" rather than requiring bit-for-bit identity.

Neither constant (GAMMA, the PER pass boundary) has been validated against
human pronunciation-quality judgements; both are a documented, reasoned
choice pending exactly the kind of ground truth Workstream 1's hybrid
judge is meant to eventually provide, in the same spirit as
`asr_roundtrip.py`'s own score-mapping constants.
"""

from __future__ import annotations

from .base import SCALE_MAX, SCALE_MIN

GAMMA = 1.6


def _levenshtein(a: list[str], b: list[str]) -> int:
    """Classic O(len(a)*len(b)) dynamic-programming edit distance over token
    lists (not characters -- see module docstring on tokenization). No
    external dependency: this project's only other edit-distance need
    (`asr_roundtrip.py`) uses `jellyfish`'s string-similarity functions, not
    raw Levenshtein, so there is nothing to reuse here.
    """
    if a == b:
        return 0
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost,  # substitution / match
            )
        prev, curr = curr, prev
    return prev[m]


def phoneme_distance(expected_phonemes: str, recognized_phonemes: str) -> float:
    """Normalized edit distance (Phoneme Error Rate) between two
    whitespace-tokenized phoneme sequences. Returns a float in [0, 1]: 0.0
    means the tokenized sequences are identical, 1.0 means the edit distance
    is at least as large as the longer sequence (effectively unrelated).

    Both empty -> 0.0 (two silences/no-signal cases are trivially "equal",
    not maximally different). Exactly one empty -> 1.0 (nothing recognized
    at all where something was expected, or vice versa, is a total miss).
    """
    exp_tokens = expected_phonemes.split()
    rec_tokens = recognized_phonemes.split()

    if not exp_tokens and not rec_tokens:
        return 0.0
    if not exp_tokens or not rec_tokens:
        return 1.0

    dist = _levenshtein(exp_tokens, rec_tokens)
    denom = max(len(exp_tokens), len(rec_tokens))
    return min(1.0, dist / denom)


def score_phoneme_match(expected_phonemes: str, recognized_phonemes: str) -> tuple[float, dict[str, float]]:
    """PER -> the project's 0-5 scale (`PASS_THRESHOLD = 4.0`, see module
    docstring for the curve and rationale). Pure and offline: no model
    loading, no I/O -- takes already-decoded phoneme strings.
    """
    per = phoneme_distance(expected_phonemes, recognized_phonemes)
    score = SCALE_MAX * (1.0 - per) ** GAMMA
    score = round(min(max(score, SCALE_MIN), SCALE_MAX), 3)
    return score, {"phoneme_error_rate": round(per, 4)}
