"""ASR round-trip scorer -- the first scorer in this repo that actually attempts
to measure pronunciation.

Why this exists instead of waiting for Workstream 1
-----------------------------------------------------
The original project plan considered, and dropped, an ASR round-trip guardrail
(synthesize -> transcribe -> compare) on the reasoning that standing up and
validating an ASR model would be its own project -- not worth it as a stopgap
for a benchmark that already has a hybrid judge on its roadmap. That premise no
longer holds in this sandbox: Google Cloud Speech-to-Text is already enabled
and reachable with the same service-account credential the TTS adapters use
(`dose_r.auth`). There is no model to stand up, tune or host -- it is one
authenticated REST call per clip. Given that, an ASR round-trip is a real,
available-today pronunciation proxy: if a TTS system says "retatrutide" in a
way that is phonetically wrong, a general-purpose ASR system has no reason to
transcribe it back as "retatrutide" by coincidence. This module exists to get
a first real number on the board while Workstream 1's gold-reference + hybrid
judge is still outstanding, not to replace it.

Mechanism
---------
1. POST the synthesized WAV to `speech:recognize` (LINEAR16, 24 kHz, en-US,
   model=latest_long, enableWordTimeOffsets=true).
2. Locate the token span in the recognized transcript that corresponds to the
   drug name's position in the known carrier sentence (`item.sentence`,
   `item.drug` guaranteed to appear in it -- see `dataset.validate()`).
   Alignment is done with `difflib.SequenceMatcher` over whitespace-tokenized
   word lists so it degrades gracefully when ASR drops, adds or splits words
   elsewhere in the sentence, not just at the drug span. See
   `locate_recognized_span()`.
3. Score the recognized span against the expected name with `score_pronunciation()`
   -- a pure, offline-testable function (no network, no I/O) so score-mapping
   behavior can be pinned by unit tests independent of the live API.

Score mapping (0-5, `PASS_THRESHOLD = 4.0`)
--------------------------------------------
Both strings are first "collapsed" (lowercased, non-alphanumeric characters --
spaces, hyphens, commas -- stripped) before comparison. Collapsing means a
multi-word drug recognized with different word breaks (e.g. reference
"insulin icodec-abae" vs. recognized "insulin I codec abae") is judged on
whether the *sounds* line up, not on ASR's arbitrary word segmentation.

  - `exact_match` (collapsed strings identical, case-insensitive) -> 5.0.
  - Otherwise, `jaro_winkler` similarity (0-1, `jellyfish.jaro_winkler_similarity`)
    and `metaphone_match` (double-metaphone-style coarse phonetic code equality,
    `jellyfish.metaphone`) combine as:
      - metaphone codes match  -> score = 3.5 + 1.5 * jaro_winkler   (band: 3.5-5.0)
      - metaphone codes differ -> score = 3.0 * jaro_winkler          (band: 0.0-3.0)
    The gate is deliberate: two strings can share a lot of surface characters
    (and so score high on Jaro-Winkler, which rewards common prefixes) while
    sounding nothing alike, e.g. "retatrutide" vs. ASR's "retro tide" scores
    jaro_winkler=0.88 but the metaphone codes (RTTRTT vs RTRTT) disagree, so it
    is capped at 2.64 -- correctly below `PASS_THRESHOLD` -- rather than the
    ~4.4 an ungated jaro_winkler-only formula would give it. A metaphone
    mismatch is treated as evidence the rendered sound is a different word,
    which no amount of surface-character overlap should be able to buy back
    above the pass line.
  - Recognized span empty (no hypothesis words aligned to the drug's position
    at all) -> 0.0, `exact_match=jaro_winkler=metaphone_match=0.0`. This is
    still `scoreable=True` -- a real, scoreable "the drug name is not audibly
    present/recognizable at that position" result -- distinct from the STT
    call itself failing (see below).

CRITICAL ASYMMETRY -- an ASR miss is not proof of TTS mispronunciation
------------------------------------------------------------------------
Cloud STT's language model has its own drug-name vocabulary gaps. It may fail
to transcribe a *correctly pronounced* rare generic (e.g. "nogapendekin alfa
inbakicept-pmln") simply because that string of sounds is nowhere in its
training distribution or lexicon, not because the TTS system said it wrong.
This scorer therefore measures "TTS-mispronunciation OR ASR-does-not-know-this-
word", and the two are conflated in a single low score. It cannot by itself
attribute a low score to one cause or the other. Every `ScoreResult` this
scorer returns carries this caveat in `notes`, and `asr_confidence` (the
top-alternative confidence field from the whole-utterance transcript, not
specific to the drug span) is recorded in `components` and `metadata` so
low-confidence recognitions -- more likely to be ASR's fault -- can be
filtered out downstream rather than trusted at face value. Any report built
from this scorer's output must repeat this caveat; do not present its pass
rates as equivalent to a human or hybrid pronunciation judgement.

If Cloud STT returns literally no results for a clip (silence, corrupt audio,
API error), the item is `scoreable=False` with an `error`, not scored 0 --
scoring it 0 would silently conflate "we have no signal" with "confirmed
mispronunciation", exactly the failure mode `docs/WORKSTREAM1_HANDOFF.md`
warns a hybrid judge must not fall into either.
"""

from __future__ import annotations

import base64
import difflib
import re
from typing import Any

import jellyfish
import requests

from .. import auth
from ..adapters.base import SynthesisResult
from ..dataset import DoseItem
from .base import SCALE_MAX, SCALE_MIN, ScoreResult, Scorer

_ENDPOINT = "https://speech.googleapis.com/v1/speech:recognize"
_DEFAULT_SAMPLE_RATE = 24_000

_ASYMMETRY_NOTE = (
    "ASR round-trip proxy: a low score means either the TTS mispronounced the "
    "drug name OR Cloud STT simply does not recognize that word -- this scorer "
    "cannot distinguish the two. asr_confidence is recorded for downstream "
    "filtering. Not a substitute for the Workstream 1 hybrid judge."
)

# Strip everything but letters/digits so word-segmentation differences between
# the reference sentence and the ASR transcript (spaces, hyphens, commas) don't
# masquerade as pronunciation differences.
_NON_ALNUM = re.compile(r"[^a-z0-9]")

# Strip only leading/trailing punctuation for the *alignment* pass (difflib
# matches on these tokens) -- internal hyphens are kept here because they are
# how "insulin icodec-abae" appears as a single reference token.
_EDGE_PUNCT = re.compile(r"^[^\w]+|[^\w]+$")

_TOKEN_RE = re.compile(r"\S+")


def _collapse(s: str) -> str:
    return _NON_ALNUM.sub("", s.lower())


def _norm_token(tok: str) -> str:
    return _EDGE_PUNCT.sub("", tok).lower()


def _tokenize_with_offsets(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group()) for m in _TOKEN_RE.finditer(text)]


def score_pronunciation(expected: str, recognized: str) -> tuple[float, dict[str, float]]:
    """Pure, offline-testable phonetic scoring of a recognized span against the
    expected drug name. See the module docstring for the mapping rationale.
    """
    exp_c = _collapse(expected)
    rec_c = _collapse(recognized)

    if not rec_c:
        return 0.0, {"exact_match": 0.0, "jaro_winkler": 0.0, "metaphone_match": 0.0}

    exact = bool(exp_c) and exp_c == rec_c
    jw = jellyfish.jaro_winkler_similarity(exp_c, rec_c)
    meta_match = jellyfish.metaphone(exp_c) == jellyfish.metaphone(rec_c)

    if exact:
        score = SCALE_MAX
    elif meta_match:
        score = 3.5 + 1.5 * jw
    else:
        score = 3.0 * jw

    score = round(min(max(score, SCALE_MIN), SCALE_MAX), 3)
    return score, {
        "exact_match": 1.0 if exact else 0.0,
        "jaro_winkler": round(jw, 4),
        "metaphone_match": 1.0 if meta_match else 0.0,
    }


def locate_recognized_span(sentence: str, drug: str, hyp_words: list[str]) -> str:
    """Find the substring of `hyp_words` (the ASR transcript, tokenized) that
    corresponds to `drug`'s position inside the known `sentence`.

    Uses `difflib.SequenceMatcher` over normalized whitespace-token lists so
    the mapping stays robust when ASR drops/adds/splits words anywhere else in
    the sentence -- a fixed word-index lookup would silently misalign as soon
    as the recognized word count differs from the reference, which is the
    common case for anything but a perfect transcription.

    Returns "" if no hypothesis words could be attributed to the drug's
    position at all (a real, scoreable "not recognized here" result).
    """
    lower_sentence = sentence.lower()
    d_start = lower_sentence.find(drug.lower())
    if d_start == -1:
        # dataset.validate() guarantees this doesn't happen for the shipped
        # dataset; guard anyway rather than raising on an unexpected input.
        return ""
    d_end = d_start + len(drug)

    ref_tok_offsets = _tokenize_with_offsets(sentence)
    drug_ref_idx = {i for i, (s, e, _) in enumerate(ref_tok_offsets) if s < d_end and e > d_start}
    if not drug_ref_idx:
        return ""

    norm_ref = [_norm_token(t) for _, _, t in ref_tok_offsets]
    norm_hyp = [_norm_token(w) for w in hyp_words]

    matcher = difflib.SequenceMatcher(None, norm_ref, norm_hyp, autojunk=False)

    hyp_idx: set[int] = set()
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        overlap = [i for i in drug_ref_idx if i1 <= i < i2]
        if not overlap:
            continue
        if tag == "delete":
            # These reference words (including, potentially, the drug) have no
            # counterpart in the hypothesis at all -- nothing to attribute.
            continue
        if tag == "equal":
            # 1:1 aligned block -- take the exact corresponding hyp index per
            # overlapping ref index rather than the whole block.
            hyp_idx.update(j1 + (i - i1) for i in overlap)
        else:  # "replace" (many-to-many); "insert" never overlaps (i1 == i2)
            hyp_idx.update(range(j1, j2))

    if not hyp_idx:
        return ""
    lo, hi = min(hyp_idx), max(hyp_idx)
    return " ".join(hyp_words[lo : hi + 1])


def _extract_words(response_json: dict) -> tuple[list[str], float | None]:
    """Flatten every result's top alternative into one word list, and average
    their confidences. Cloud STT can return multiple `results` entries for one
    clip (segment boundaries); scoring only the first would silently drop the
    tail of longer sentences.
    """
    results = response_json.get("results") or []
    words: list[str] = []
    confidences: list[float] = []
    for r in results:
        alts = r.get("alternatives") or []
        if not alts:
            continue
        alt = alts[0]
        w = alt.get("words")
        if w:
            words.extend(wi["word"] for wi in w)
        elif alt.get("transcript"):
            words.extend(alt["transcript"].split())
        if "confidence" in alt:
            confidences.append(alt["confidence"])
    avg_conf = sum(confidences) / len(confidences) if confidences else None
    return words, avg_conf


class AsrRoundTripScorer(Scorer):
    """Synthesize -> Cloud STT -> compare recognized drug span to the expected
    name. See the module docstring for the full mechanism, score mapping and
    -- importantly -- the ASR-bias caveat this scorer cannot resolve on its
    own.
    """

    measures_pronunciation = True

    def __init__(self, *, language_code: str = "en-US", timeout_s: float = 60.0,
                 session: requests.Session | None = None):
        self.language_code = language_code
        self.timeout_s = timeout_s
        self._session = session or requests.Session()

    @property
    def scorer_id(self) -> str:
        return "asr-roundtrip-v1"

    def _recognize(self, audio: bytes, sample_rate_hz: int) -> dict:
        body = {
            "config": {
                "encoding": "LINEAR16",
                "sampleRateHertz": sample_rate_hz,
                "languageCode": self.language_code,
                "model": "latest_long",
                "enableWordTimeOffsets": True,
            },
            "audio": {"content": base64.b64encode(audio).decode("ascii")},
        }
        resp = self._session.post(
            _ENDPOINT, headers=auth.auth_headers(), json=body, timeout=self.timeout_s
        )
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def score(self, item: DoseItem, result: SynthesisResult) -> ScoreResult:
        base = dict(scorer_id=self.scorer_id, item_id=item.item_id,
                    system_id=result.system_id)

        if not result.ok or not result.audio:
            return ScoreResult(**base, score=0.0, scoreable=True,
                                error=result.error or "no audio returned",
                                notes="synthesis failed upstream")

        try:
            response_json = self._recognize(
                result.audio, result.sample_rate_hz or _DEFAULT_SAMPLE_RATE
            )
        except Exception as exc:
            # A transport/API failure is "we have no signal", never a score of
            # 0 -- that would conflate an ASR outage with a confirmed
            # mispronunciation.
            return ScoreResult(**base, score=None, scoreable=False,
                                error=f"Cloud STT request failed: {exc}")

        hyp_words, asr_confidence = _extract_words(response_json)
        if not hyp_words:
            return ScoreResult(**base, score=None, scoreable=False,
                                error="Cloud STT returned no results for this clip",
                                notes=_ASYMMETRY_NOTE)

        recognized_span = locate_recognized_span(item.sentence, item.drug, hyp_words)
        score, components = score_pronunciation(item.drug, recognized_span)
        components["asr_confidence"] = round(asr_confidence, 4) if asr_confidence is not None else 0.0

        return ScoreResult(
            **base, score=score, components=components,
            notes=_ASYMMETRY_NOTE,
            metadata={
                "recognized_span": recognized_span,
                "full_transcript": " ".join(hyp_words),
                "asr_confidence": asr_confidence,
            },
        )
