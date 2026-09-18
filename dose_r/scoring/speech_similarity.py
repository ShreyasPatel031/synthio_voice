"""Audio-to-audio pronunciation similarity -- SpeechBERTScore-style comparison.

Why this exists
----------------
The ASR round-trip and audio-LLM scorers both go through some form of "what
word/phoneme was said," collapsing the audio to a symbolic guess before
comparing anything. This module compares two audio clips directly, without
ever deciding what either one "says."

The obvious naive way to do that -- raw waveform or MFCC distance, or DTW over
raw spectral features -- is wrong for this project's purpose: two different
people (or a human and a TTS voice) saying the same drug name correctly have
different pitch, timbre, speaking rate and recording conditions, and a raw
acoustic distance would penalize all of that as if it were mispronunciation.
That is exactly the failure mode this module exists to avoid (see the
voice-invariance validation below, which measures it directly rather than
assuming it away).

Mechanism: SpeechBERTScore (Saeki et al., Interspeech 2024)
-------------------------------------------------------------
1. Extract frame-level hidden states from a self-supervised speech model
   (`facebook/wav2vec2-base` -- the plain SSL-pretrained encoder, NOT a
   CTC/ASR-fine-tuned checkpoint like the phoneme model in phoneme_model.py).
   A model trained to predict masked audio from context is pushed to encode
   *what sound was made* rather than *who made it or how loud*, which is
   exactly the invariance property needed here. This is not a phoneme
   recognizer and never assigns a phoneme label to anything -- it is a
   representation, not a decoder.
2. Compare the two frame sequences the way text BERTScore compares two token
   sequences: build the full cosine-similarity matrix between every frame of
   clip A and every frame of clip B, take each row's max (best match for each
   A-frame) and each column's max (best match for each B-frame), average
   those into precision/recall, and combine as their harmonic mean (F1). No
   alignment path is imposed -- a frame can match any frame in the other
   clip, so a difference in speaking rate does not by itself hurt the score.
3. Map F1 (empirically close to [0, 1] for two real speech clips; clamped
   defensively) onto this project's 0-5 scale.

This is a proxy, not ground truth
------------------------------------
- Requires a REFERENCE recording. It only covers the ~62% of ingredients
  with a committed human reference clip (`dose_r.references.reference_clips`)
  -- structurally the same coverage ceiling as `references/audio_grounded.py`.
- The SSL encoder was pretrained on read/conversational English speech, not
  specifically tuned for pronunciation assessment; using it this way is
  standard in the SpeechBERTScore literature but not something this project
  has independently proven optimal.
- Which layer's hidden states to use matters a great deal and was NOT a safe
  default: the first draft of this module used the final hidden layer and
  it produced zero discrimination between correct and deliberately
  mismatched audio/reference pairs (mismatched pairs scored as high as, or
  higher than, correct ones). A layer sweep (see `_LAYER`'s comment) found
  layers 10-12 uniformly broken and 6-9 all working; layer 9 is used. This
  was found on a 5-item slice, not swept exhaustively -- treat it as a
  validated starting point, not a proven optimum.
- Also required a real fix, not just layer selection: comparing a full
  synthesized SENTENCE's embeddings against an isolated ~1s reference WORD's
  embeddings failed discrimination outright, because a 7-second sentence and
  a 1-second word resemble each other in generic "this is speech" ways
  regardless of content. `dose_r.audio_span.extract_drug_span` slices out
  just the drug's audio from the sentence before any comparison happens --
  skipping this step (as the first draft did) silently breaks the metric.
- Validated on a small slice (`scripts/validate_speech_similarity.py`), with
  an honest mixed result, not a clean pass on every check:
  * Discrimination: PASS. 5 correct TTS/reference pairs (min score 2.65) all
    scored above 3 deliberately mismatched pairs (max score 2.18) -- a clean
    gap.
  * Beats the naive baseline: on the same items, plain MFCC+DTW averaged 1.28
    while this method averaged 2.99 -- real evidence the SSL approach is less
    confounded by voice identity than raw acoustic distance, not just an
    assumption from the literature.
  * Voice-invariance (two different humans, same correct word): 4/5 pairs
    scored reasonably (3.07-3.56); one, "Aspirin", scored 1.77, well below
    the others. Duration mismatch was ruled out as the cause (Advil's two
    clips differ in length by a similar ratio and still scored 3.18). The
    leading hypothesis is a genuine American-English pronunciation variant
    for "aspirin" (full middle syllable /ˈæspərɪn/ vs. elided /ˈæsprɪn/) --
    i.e. the metric may be correctly detecting that the two "canonical"
    human sources do not say the word identically, not failing. This is a
    hypothesis, not confirmed, and the sample is one item out of five.
  Net: usable with this caveat attached to every result, not a fully clean
  validation. Not yet validated at corpus scale, and the voice-invariance
  check should be re-run on a larger sample before this caveat is either
  resolved or promoted to a known limitation.

Reference source: Merriam-Webster preferred, with a real trade-off
----------------------------------------------------------------------
`dose_r.references.reference_clips` prefers Merriam-Webster over Drugs.com
when both exist -- it is an actual pronouncing dictionary (carries a written
respelling, e.g. aspirin's "as-p(schwa-)rin" documents the schwa as an
explicitly optional variant, which is what resolved the Aspirin
voice-invariance outlier as a real-but-accepted variant rather than an error).

That preference has a measured cost: Merriam-Webster's own clips run
noticeably longer than Drugs.com's for the same word (WS1's own
`AUDIO_COVERAGE.md`: 1.3-2x, described as a slower teaching-recording pace).
Confirmed here on Abilify: scoring against Drugs.com's reference (1.05s) gave
3.22; the identical synthesized clip against Merriam-Webster's reference
(2.14s) gives 2.60. That is a large swing driven by which "correct" reference
was picked, not by anything about the candidate audio -- a real limitation of
comparing against a single reference recording's pace, not fixed by this
module. Treat scores as comparative across systems scored against the SAME
reference, not as an absolute, pace-independent pronunciation measure.

Cost
----
Local model, CPU inference, no network calls, no per-clip spend -- same as
`phoneme_model.py`.
"""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from ..adapters.base import SynthesisResult
from ..audio_span import extract_drug_span
from ..dataset import DoseItem
from ..references.reference_clips import ReferenceClip, available_clips
from .base import ScoreResult, Scorer

MODEL_ID = "facebook/wav2vec2-base"
_TARGET_SR = 16_000

# Layer 9 of 12, chosen by a validation sweep (docs/EVALUATION_PATHS_PLAN.md /
# the Path 2 validation script), not the paper's default or a guess. Layers
# 10-12 (including the final layer, this module's first draft) showed ZERO
# discrimination between correct and deliberately mismatched audio/reference
# pairs on a 5-item slice -- late wav2vec2 layers drift toward something less
# tied to phonetic identity, consistent with the SSL layer-probing literature
# (e.g. SUPERB). Layers 6-9 all separated correctly; 9 had the widest margin
# (correct-pair minimum 0.531 vs. mismatched-pair maximum 0.436 F1). This is
# an empirical choice from one small slice, not a swept optimum -- revisit if
# broader validation suggests a different layer generalizes better.
_LAYER = 9


@lru_cache(maxsize=1)
def _get_model():
    """Lazy, memoized load -- importing this module must not pull in torch or
    download anything until a caller actually needs embeddings."""
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    model = Wav2Vec2Model.from_pretrained(MODEL_ID)
    model.eval()
    return torch, extractor, model


def extract_frame_embeddings(audio_source, sample_rate: int | None = None) -> np.ndarray:
    """audio -> (T, 768) frame-level SSL hidden states (final layer).

    Accepts a path (any librosa-readable format), raw in-memory WAV bytes
    (e.g. from `dose_r.audio_span.extract_drug_span`, which never touches
    disk), or a raw float array (in which case `sample_rate` is required for
    resampling to the model's 16kHz).
    """
    torch, extractor, model = _get_model()

    if isinstance(audio_source, np.ndarray):
        if sample_rate is None:
            raise ValueError("sample_rate is required when passing a raw array")
        audio = audio_source
        if sample_rate != _TARGET_SR:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sample_rate,
                                     target_sr=_TARGET_SR)
    elif isinstance(audio_source, (bytes, bytearray)):
        import soundfile as sf
        raw, sr = sf.read(BytesIO(bytes(audio_source)), dtype="float32")
        audio = raw if raw.ndim == 1 else raw.mean(axis=1)  # downmix if stereo
        if sr != _TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=_TARGET_SR)
    else:
        audio, _ = librosa.load(str(audio_source), sr=_TARGET_SR, mono=True)

    inputs = extractor(audio, sampling_rate=_TARGET_SR, return_tensors="pt")
    with torch.no_grad():
        out = model(inputs.input_values, output_hidden_states=True)
    return out.hidden_states[_LAYER][0].numpy()  # (T, 768) -- see _LAYER's comment


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(T_a, D) x (T_b, D) -> (T_a, T_b) cosine similarity. Pure numpy, no
    model -- kept separate from `extract_frame_embeddings` so the comparison
    logic is testable on synthetic arrays without loading anything.
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T


def speech_bertscore(feats_a: np.ndarray, feats_b: np.ndarray) -> dict[str, float]:
    """SpeechBERTScore's precision/recall/F1 between two frame-embedding
    sequences. Pure, offline-testable: takes already-extracted embeddings, no
    audio or model involved. Returns raw values in [-1, 1]; the caller maps
    to this project's 0-5 scale.
    """
    if feats_a.shape[0] == 0 or feats_b.shape[0] == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    sim = _cosine_similarity_matrix(feats_a, feats_b)
    precision = float(sim.max(axis=1).mean())  # best match for each A-frame
    recall = float(sim.max(axis=0).mean())     # best match for each B-frame
    f1 = 0.0 if (precision + recall) <= 0 else 2 * precision * recall / (precision + recall)
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def score_speech_similarity(feats_a: np.ndarray, feats_b: np.ndarray) -> tuple[float, dict[str, float]]:
    """SpeechBERTScore F1 -> this project's 0-5 scale (PASS_THRESHOLD=4.0).

    Mapping is linear and UNCALIBRATED: `score = 5 * clamp(f1, 0, 1)`. Unlike
    asr_roundtrip.py's and phoneme_scorer.py's score curves, no reasoning
    about a nonlinear shape is offered here, because there is not yet a
    validated F1 distribution to reason from -- this is deliberately the
    simplest possible mapping, to be revisited once the validation script's
    results give real numbers to calibrate against instead of a guess.
    """
    result = speech_bertscore(feats_a, feats_b)
    score = round(5.0 * min(max(result["f1"], 0.0), 1.0), 3)
    return score, result


class SpeechSimilarityScorer(Scorer):
    """Path 2: audio-to-audio pronunciation similarity against a human
    reference clip. See module docstring for the method, its validation
    results (mixed -- one open caveat on voice-invariance calibration), and
    what it does and doesn't establish.
    """

    measures_pronunciation = True

    def __init__(self, *, reference_clips: dict[str, ReferenceClip] | None = None,
                 stt_session: Any = None):
        # Looked up once per scorer instance, not per item -- available_clips()
        # reads the whole manifest and stats every candidate file on disk;
        # reusing it across a run avoids doing that once per of 1000+ items.
        self._clips = reference_clips if reference_clips is not None else available_clips()
        self._stt_session = stt_session

    @property
    def scorer_id(self) -> str:
        return "speech-similarity-v1"

    def score(self, item: DoseItem, result: SynthesisResult) -> ScoreResult:
        base = dict(scorer_id=self.scorer_id, item_id=item.item_id,
                    system_id=result.system_id)

        if not result.ok or not result.audio:
            return ScoreResult(**base, score=0.0, scoreable=True,
                               error=result.error or "no audio returned",
                               notes="synthesis failed upstream")

        clip = self._clips.get(item.drug)
        if clip is None:
            return ScoreResult(
                **base, score=None, scoreable=False,
                error=f"no reference clip available for {item.drug!r}",
                notes="Path 2 covers only ingredients with a committed/fetched "
                      "human reference clip (~65% currently: 179/284).",
            )

        try:
            span = extract_drug_span(result.audio, item.sentence, item.drug,
                                     session=self._stt_session)
        except Exception as exc:
            return ScoreResult(**base, score=None, scoreable=False,
                               error=f"drug-span extraction failed: {exc}")
        if span is None:
            return ScoreResult(
                **base, score=None, scoreable=False,
                error="could not locate the drug name's audio span in the "
                      "synthesized clip's transcript",
            )

        try:
            feats_synth = extract_frame_embeddings(span)
            feats_ref = extract_frame_embeddings(clip.path)
        except Exception as exc:
            return ScoreResult(**base, score=None, scoreable=False,
                               error=f"embedding extraction failed: {exc}")

        score, components = score_speech_similarity(feats_synth, feats_ref)

        return ScoreResult(
            **base, score=score, components=components,
            metadata={
                "reference_source": clip.source,
                "reference_respelling": clip.respelling,
                "reference_duration_s": clip.duration_s,
                "layer": _LAYER,
            },
            notes=(
                "Audio-to-audio comparison (SpeechBERTScore-style) against a "
                f"{clip.source} human reference clip -- no ASR, no LLM, no "
                "phoneme decoding involved. Validated on a small slice: clean "
                "discrimination and a clear win over a naive MFCC/DTW "
                "baseline, but voice-invariance (two humans, same word) had "
                "one unexplained low outlier out of five in that slice -- "
                "treat any single score as noisier than the discrimination "
                "result alone would suggest. Covers only ~65% of ingredients "
                "(those with a reference clip). Not Workstream 1's hybrid "
                "judge."
            ),
        )
