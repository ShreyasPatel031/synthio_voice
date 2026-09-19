"""Path 3: phoneme-level pronunciation distance, scored directly against
dictionary IPA -- no synthesized reference audio, no human recording
required. Built for the 99/274 items (36% of the benchmark) with zero
pronunciation reference of any kind (no human clip, no dictionary
respelling) once mapped through this project's IPA-cleanup effort
(`references.jsonl`, currently on another branch: 280/284 ingredients now
have at least one sourced IPA variant).

Why this exists, after an extensive Path 4 investigation failed
--------------------------------------------------------------------
This session spent a long stretch trying to synthesize a stand-in
REFERENCE AUDIO clip from dictionary IPA (Path 4), across 4 TTS engines
(Cloud TTS, Gemini, Piper, Kokoro) and 2 genuinely-verified phoneme-
injection mechanisms (Cloud TTS's SSML tag confirmed to sometimes be
silently ignored via byte-identity checks; Piper's direct phoneme-ID
injection confirmed to cover the full IPA vocabulary used). Every
phoneme-injection attempt made output sound WORSE or foreign compared to
the same engine reading the bare spelling with zero pronunciation help --
traced to a concrete, verified cause (a notation mismatch, 'r' vs 'ɹ', that
only partially closed the gap after being fixed) and a structural one
(these models' prosody/duration components were only ever trained on
their OWN grapheme-to-phoneme's output shape, so correct-but-differently-
shaped phoneme input is out-of-distribution for them, not simply "more
correct"). No engine or technique tested reached human-level correctness
on the confirmed-bad items (Vyloy, Adquey, Voranigo). That is treated here
as a real, evidenced dead end, not a temporary gap -- see
`docs/PATH4_SYNTHETIC_REFERENCE_INVESTIGATION.md` for the full trail.

Path 3 sidesteps the entire problem: it never synthesizes anything. It
takes the CANDIDATE's own audio, decodes it to phonemes with the same CTC
model already used for forced alignment (`scoring.phoneme_model`,
`facebook/wav2vec2-lv-60-espeak-cv-ft`), and compares that decoded string
directly against the dictionary's IPA using a phonetically-weighted edit
distance -- no synthesis-quality confound, no prosody-mismatch confound,
because nothing is being synthesized on the reference side at all.

Known, fixed notation mismatch: r vs ɹ
----------------------------------------
The CTC model's own decoded output uses 'ɹ' (the actual English
approximant -- confirmed empirically: decoding a human saying "advair"
gives "ɛ d v ɛ ɹ"). The dictionary IPA in `references.jsonl` uses plain
'r' throughout (245 of 249 r-containing entries, USAN/DailyMed/etc.'s
common broad-transcription convention) -- which in strict IPA denotes an
alveolar TRILL, a different sound. Comparing the two without normalizing
would charge every r-containing word an r-vs-ɹ edit cost that reflects a
NOTATION difference, not a pronunciation difference -- the exact same bug
already found and fixed on the synthesis side this session. Normalized
here before any distance is computed.

Mechanism
---------
1. Decode the candidate's drug-name span (already isolated via
   `forced_align.extract_drug_span_forced_align`) with
   `phoneme_model.transcribe_phonemes()`.
2. Normalize both the decoded string and each dictionary IPA variant
   (`normalize_phonemes`): unify r/ɹ, strip stress marks (the CTC model
   does not reliably mark stress, so comparing it would penalize the
   model's own limitation, not the candidate's pronunciation -- the same
   reasoning `phoneme_model.py`'s docstring already applies to vowel
   confusions).
3. Segment both into IPA phoneme units with `panphon` (correctly handles
   multi-character sequences; diphthongs are segmented as two vowel
   targets, consistently on both sides, so comparisons stay fair even
   though it isn't a single-unit treatment).
4. Compute `panphon.distance.Distance().weighted_feature_edit_distance`
   between the candidate and EACH dictionary variant, take the MINIMUM
   (the multi-reference principle already used in
   `scoring.candidate_eval` -- credit the candidate for matching ANY
   accepted variant, not one arbitrarily chosen reference).
5. Normalize by the winning reference's own segment count, since raw
   distance scales with word length and a 15-phoneme name will show a
   bigger raw distance than a 4-phoneme name for an equivalent relative
   error -- this rate is what gets compared across items of different
   lengths.

Not yet validated at corpus scale -- see
`scripts/validate_phoneme_distance.py` for the human-ceiling and
discrimination checks this needs before being trusted, mirroring the
validation discipline `speech_similarity.py` was held to.
"""

from __future__ import annotations

import re
from functools import lru_cache

_STRESS_RE = re.compile(r"[ˈˌ]")


@lru_cache(maxsize=1)
def _get_panphon():
    import panphon
    import panphon.distance

    return panphon.FeatureTable(), panphon.distance.Distance()


def normalize_phonemes(s: str) -> str:
    """Strip stress marks and unify the r/ɹ notation mismatch (see module
    docstring) before any comparison -- both sides of every comparison
    made by this module go through this first.
    """
    s = _STRESS_RE.sub("", s)
    s = s.replace("r", "ɹ")
    return s.replace(" ", "")


def phoneme_distance_to_variant(candidate: str, reference_ipa: str) -> tuple[float, int]:
    """Normalized candidate vs. ONE reference IPA variant -> (raw weighted
    feature edit distance, reference's own segment count after
    normalization). Segment count is returned alongside the distance so a
    caller can normalize by the WINNING variant's length after picking the
    minimum across variants, not before (an early per-variant normalization
    would bias the minimum toward whichever variant happens to be longest).
    """
    ft, dist = _get_panphon()
    cand_norm = normalize_phonemes(candidate)
    ref_norm = normalize_phonemes(reference_ipa)
    d = dist.weighted_feature_edit_distance(cand_norm, ref_norm)
    ref_len = len(ft.ipa_segs(ref_norm)) or 1
    return d, ref_len


def best_phoneme_distance(candidate: str, ipa_variants: list[str]) -> dict:
    """Candidate phoneme string vs. every accepted dictionary IPA variant,
    crediting the candidate for matching ANY of them (mirrors
    `scoring.candidate_eval.score_against_best_reference`'s multi-reference
    principle, applied here to multiple accepted PHONEME spellings instead
    of multiple human recordings).

    Returns the winning variant's raw distance, its length-normalized rate
    (distance / segment count -- comparable across words of different
    length), and which variant won, so a caller can inspect whether a
    candidate matched one variant clearly or was mediocre against all of
    them.
    """
    if not ipa_variants:
        raise ValueError("no IPA variants given -- caller should check "
                         "coverage before calling this")

    results = [(v, *phoneme_distance_to_variant(candidate, v)) for v in ipa_variants]
    best_variant, best_dist, best_len = min(results, key=lambda r: r[1] / r[2])
    return {
        "best_variant": best_variant,
        "distance": round(best_dist, 4),
        "rate": round(best_dist / best_len, 4),
        "all_variants": {v: round(d / n, 4) for v, d, n in results},
    }
