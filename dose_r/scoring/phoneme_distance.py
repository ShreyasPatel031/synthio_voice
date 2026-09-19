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

from ..adapters.base import SynthesisResult
from ..dataset import DoseItem
from ..forced_align import extract_drug_span_forced_align
from ..references.ipa_references import ipa_variants_for
from .base import ScoreResult, Scorer
from .phoneme_model import transcribe_phonemes

_STRESS_RE = re.compile(r"[ˈˌ]")

# Anchor for the rate -> 0-5 mapping, taken directly from this module's own
# validation (scripts/validate_phoneme_distance.py): deliberately-mismatched
# audio/IPA pairs averaged rate=3.853 across 8 items. A linear map that puts
# 0 there and 5 at rate=0 happens to place the correct-pair validation mean
# (rate=0.876) at ~3.86/5 -- close to Path 2's independently-measured human
# ceiling of 3.79/5, which is a reassuring cross-check between two unrelated
# metrics, not something this mapping was tuned to hit. Simplest possible
# mapping, deliberately not fit to a larger calibration set -- see
# speech_similarity.py's own score_speech_similarity for the same philosophy.
_MISMATCH_RATE_ANCHOR = 3.853

# Multi-word ingredient names (e.g. "tenofovir alafenamide") were the
# dominant failure mode in this module's own human-ceiling validation: the
# human reference CLIP often only covers the first word, while the
# dictionary IPA covers the full multi-word name, producing a spurious
# length-mismatch distance that has nothing to do with pronunciation
# accuracy. Confirmed on 4 of the worst 5 human-ceiling outliers. Flagged
# in metadata rather than silently scored -- a known, unresolved data-scope
# issue, not a defect in the distance metric itself.
def _is_multiword_ingredient(drug: str) -> bool:
    return " " in drug.strip()


# A second, distinct data-coverage gap found running this scorer at corpus
# scale: FDA-mandated "biosimilar distinguishing suffixes" (e.g.
# "risankizumab-rzaa", "-rzaa" being a deliberately arbitrary 4-letter code
# with no established pronunciation of any kind by design). The dictionary
# IPA for these covers only the base name, not the suffix -- confirmed
# directly: "risankizumab-rzaa"'s reference is "rɪsænˈkɪzjuːmæb" (no suffix
# at all), while Gemini's actual audio attempts the full name including an
# attempt at sounding out "R-Z-A-A" as letters, decoded tail "ɑːɹ ɹ ɛs eɪ".
# Comparing the longer candidate string against the shorter, suffix-less
# reference inflates the distance for a reason that has nothing to do with
# pronunciation accuracy. Confirmed on 12 items in the real corpus run
# (all scoring nearly 0, dragging hyphenated-name mean to 2.06 vs 3.76 for
# the rest) -- but NOT every hyphenated name is affected: genuine two-word
# compound names with real phonetic coverage on both halves (e.g.
# "dimethyl-fumarate", "insulin-glargine") score normally. The length-ratio
# check below, restricted to hyphenated names, separates the two: a
# candidate decoding to notably more phoneme segments than the winning
# reference variant, but ONLY when the name is hyphenated (a plain
# mispronunciation like Adquey's can also decode longer than its reference,
# so the ratio alone is not a safe signal without the hyphen restriction --
# checked directly on the real corpus run before adding this condition).
_SUFFIX_GAP_LENGTH_RATIO = 1.4


def _is_likely_suffix_coverage_gap(drug: str, decoded_len: int, ref_len: int) -> bool:
    if "-" not in drug or ref_len == 0:
        return False
    return decoded_len > _SUFFIX_GAP_LENGTH_RATIO * ref_len


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


def rate_to_score(rate: float) -> float:
    """Length-normalized phoneme distance -> this project's 0-5 scale. See
    `_MISMATCH_RATE_ANCHOR`'s docstring for where the anchor comes from.
    """
    return round(5.0 * max(0.0, min(1.0, 1.0 - rate / _MISMATCH_RATE_ANCHOR)), 3)


class PhonemeDistanceScorer(Scorer):
    """Path 3: phoneme-level pronunciation distance against dictionary IPA.
    No synthesized reference audio, no human recording required -- built
    specifically for the 99/274 items with neither, once mapped through
    Workstream 1's IPA cleanup (`references.ipa_references`). See this
    module's docstring for why Path 4 (synthesizing a reference clip) was
    abandoned first, and `scripts/validate_phoneme_distance.py` for the
    human-ceiling and discrimination checks this scorer's mapping is
    grounded in.
    """

    measures_pronunciation = True

    @property
    def scorer_id(self) -> str:
        return "phoneme-distance-v1"

    def score(self, item: DoseItem, result: SynthesisResult) -> ScoreResult:
        base = dict(scorer_id=self.scorer_id, item_id=item.item_id,
                    system_id=result.system_id)

        if not result.ok or not result.audio:
            return ScoreResult(**base, score=0.0, scoreable=True,
                               error=result.error or "no audio returned",
                               notes="synthesis failed upstream")

        ipa_variants = ipa_variants_for(item.drug)
        if not ipa_variants:
            return ScoreResult(
                **base, score=None, scoreable=False,
                error=f"no dictionary IPA available for {item.drug!r}",
                notes="Path 3 covers ingredients with a sourced IPA variant "
                      "in Workstream 1's references.jsonl snapshot "
                      "(280/284 as of the commit this repo has).",
            )

        try:
            span = extract_drug_span_forced_align(result.audio, item.sentence, item.drug)
        except Exception as exc:
            return ScoreResult(**base, score=None, scoreable=False,
                               error=f"drug-span extraction failed: {exc}")
        if span is None:
            return ScoreResult(
                **base, score=None, scoreable=False,
                error="could not locate the drug name's audio span via forced "
                      "alignment against the sentence text",
            )

        try:
            decoded = transcribe_phonemes(span)
            result_dist = best_phoneme_distance(decoded, ipa_variants)
        except Exception as exc:
            return ScoreResult(**base, score=None, scoreable=False,
                               error=f"phoneme decoding/distance failed: {exc}")

        score = rate_to_score(result_dist["rate"])
        multiword = _is_multiword_ingredient(item.drug)

        ft, _ = _get_panphon()
        decoded_len = len(ft.ipa_segs(normalize_phonemes(decoded)))
        ref_len = len(ft.ipa_segs(normalize_phonemes(result_dist["best_variant"])))
        suffix_gap = _is_likely_suffix_coverage_gap(item.drug, decoded_len, ref_len)

        return ScoreResult(
            **base, score=score,
            components={"rate": result_dist["rate"], "distance": result_dist["distance"]},
            metadata={
                "decoded_phonemes": decoded,
                "best_matching_ipa_variant": result_dist["best_variant"],
                "all_variant_rates": result_dist["all_variants"],
                "multiword_ingredient_caveat": multiword,
                "suffix_coverage_gap_caveat": suffix_gap,
            },
            notes=(
                "Phoneme-level edit distance (panphon-weighted) between the "
                "candidate's decoded pronunciation and dictionary IPA -- no "
                "synthesized reference audio or human recording involved. "
                + ("CAVEAT: multi-word ingredient name -- this module's own "
                   "validation found the dominant human-ceiling failure mode "
                   "is a clip/IPA scope mismatch on multi-word names (the "
                   "reference clip often covers only the first word); read "
                   "this score with that in mind. " if multiword else "")
                + ("CAVEAT: likely FDA biosimilar-suffix coverage gap -- the "
                   "dictionary IPA for this hyphenated name appears to cover "
                   "only the base name, not the (deliberately arbitrary, "
                   "unpronounceable-by-design) suffix, and the candidate's "
                   "decoded length is notably longer than the reference; "
                   "this score likely understates the candidate's real "
                   "accuracy on the part that IS meant to be pronounced. " if suffix_gap else "")
                + "Not yet run at corpus scale; validated on discrimination "
                "(mismatched pairs averaged 3.85 vs 0.88 for correct pairs) "
                "and human-ceiling checks only."
            ),
        )
