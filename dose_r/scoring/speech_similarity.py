"""Audio-to-audio pronunciation similarity -- SpeechBERTScore (Saeki et al.,
Interspeech 2024), matching the published method after actually reading the
paper rather than assuming its details.

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
That is exactly the failure mode this module exists to avoid.

Correction history -- read before trusting any number this module produces
-------------------------------------------------------------------------------
The first draft of this module got three things wrong relative to the actual
paper, found by fetching and reading it (not by search-snippet paraphrase):

1. **Formula.** First draft combined precision and recall into F1. The paper's
   own words: "While the original BERTScore defines precision, recall and
   F1-score, we use the precision as we found that it performed the best in
   our preliminary experiment." Their Eq. 2 is precision only -- for each
   CANDIDATE frame, how good is its best match anywhere in the reference,
   averaged. This module now does the same. Practically, precision-only is
   also less punishing when the reference recording is simply longer/slower
   than the candidate (common here -- see the Merriam-Webster note below):
   recall would penalize the candidate for not covering extra reference
   frames it was never going to produce; precision does not.
2. **Model.** First draft used `facebook/wav2vec2-base`. The paper tested 7
   SSL models (Table 5); wav2vec2-base was the WEAKEST of the six real ones
   (LCC 0.560, only `encodec` did worse at 0.087). Their best performer,
   `wavlm-large` (LCC 0.581), is used here instead.
3. **Layer.** First draft found wav2vec2-base's late layers (10-12 of 12)
   completely broken for this task -- mismatched pairs scored as high as
   correct ones. Re-swept for wavlm-large (24 transformer layers) on the same
   validation slice: nearly every layer from 2 onward discriminates correctly,
   and the LAST layer (24, i.e. plain `last_hidden_state`, no special layer
   indexing needed) gives by far the widest margin (correct-pair minimum
   precision 0.467 vs. mismatched-pair maximum 0.240 -- a 0.227 gap, several
   times wider than wav2vec2-base ever achieved even at its best layer). This
   matches the paper's own claim that "the SSL models except for hubert-base
   had the beneficial property of being highly robust to layer selection" --
   wavlm is one of the models they say is robust; wav2vec2-base's brittleness
   in the first draft is consistent with it NOT being on that robust list.

One thing the paper does NOT validate, that this project uses it for anyway:
their whole benchmark assumes reference and candidate always contain the SAME
words (correlating with human naturalness/quality ratings of a matched pair).
They never test whether the method can detect that the WRONG word was said --
that discrimination check is this project's own addition, not something their
published results establish. It happens to work well here (see validation
below), but that is this project's finding, not theirs.

Layer aggregation vs. single layer: the paper picks one "best-performing
layer" per model (their own words), not a learned weighted combination across
layers (the alternative used elsewhere, e.g. SUPERB-style probing). This
module does the same -- a weighted-aggregate approach would need labeled data
to fit the weights against, which this project does not have without
overfitting to a handful of validation examples.

Mechanism
---------
1. Extract frame-level hidden states from `microsoft/wavlm-large`'s final
   layer (`last_hidden_state`) for both clips.
2. For every candidate frame, find its single best cosine-similarity match
   anywhere in the reference's frames; average those best-match scores. No
   alignment path is imposed -- a frame can match any frame in the other
   clip, so a difference in speaking rate does not by itself hurt the score.
3. That average (in roughly [0, 1] for real speech pairs) maps linearly onto
   this project's 0-5 scale.

Validation (`scripts/validate_speech_similarity.py`), current state
------------------------------------------------------------------------
With wavlm-large + precision-only + final layer, re-run against the same
5-item slice as the first draft:
- Discrimination: PASS, and by a wide margin. 5 correct pairs (min precision
  0.467) all comfortably above 3 deliberately mismatched pairs (max precision
  0.240).
- Voice-invariance (two different humans, same correct word): all 5 pairs now
  score consistently (0.747-0.827) -- notably, "Aspirin" (the first draft's
  unexplained low outlier at 1.77/5 under wav2vec2-base+F1) is NO LONGER an
  outlier at all here (0.768, squarely in the middle of the other four). That
  is worth being honest about: the earlier hypothesis (a genuine dictionary-
  documented pronunciation variant for aspirin, see the Merriam-Webster note
  below) was independently confirmed against real data and is still a
  legitimate fact -- but the 1.77 score itself turns out to have been mostly a
  METHODOLOGY artifact (wrong model/formula/layer), not primarily the
  reference disagreement it was first attributed to. Preferring Merriam-
  Webster as the reference source remains the right call on its own merits;
  it just was not what fixed this particular number.
Both checks now pass with considerably more margin than the first draft ever
achieved. Not yet validated at corpus scale, and still a proxy, not
Workstream 1's hybrid judge.

Reference source: Merriam-Webster preferred, with a real trade-off
----------------------------------------------------------------------
`dose_r.references.reference_clips` prefers Merriam-Webster over Drugs.com
when both exist -- it is an actual pronouncing dictionary (carries a written
respelling, e.g. aspirin's "as-p(schwa-)rin" documents the schwa as an
explicitly optional variant) rather than just an audio file with no
transcription behind it.

That preference still has a real, separate cost: Merriam-Webster's own clips
run noticeably longer than Drugs.com's for the same word (WS1's own
`AUDIO_COVERAGE.md`: 1.3-2x, described as a slower teaching-recording pace).
Using precision rather than F1 (see "Correction history" above) reduces, but
does not eliminate, this pace-sensitivity, since precision only iterates over
candidate frames and does not require covering every reference frame.

Cost
----
Local model, CPU inference, no network calls, no per-clip spend.
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

MODEL_ID = "microsoft/wavlm-large"
_TARGET_SR = 16_000


@lru_cache(maxsize=1)
def _get_model():
    """Lazy, memoized load -- importing this module must not pull in torch or
    download anything until a caller actually needs embeddings."""
    import torch
    from transformers import Wav2Vec2FeatureExtractor, WavLMModel

    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    model = WavLMModel.from_pretrained(MODEL_ID)
    model.eval()
    return torch, extractor, model


def extract_frame_embeddings(audio_source, sample_rate: int | None = None) -> np.ndarray:
    """audio -> (T, 1024) frame-level wavlm-large hidden states, final layer.

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
        out = model(inputs.input_values)
    return out.last_hidden_state[0].numpy()  # (T, 1024)


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(T_a, D) x (T_b, D) -> (T_a, T_b) cosine similarity. Pure numpy, no
    model -- kept separate from `extract_frame_embeddings` so the comparison
    logic is testable on synthetic arrays without loading anything.
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T


def speech_bertscore(feats_a: np.ndarray, feats_b: np.ndarray) -> dict[str, float]:
    """SpeechBERTScore precision/recall between two frame-embedding sequences,
    `a` treated as the candidate and `b` as the reference (order matters for
    precision, per the paper's Eq. 2). Pure, offline-testable: takes
    already-extracted embeddings, no audio or model involved.

    `precision` is the paper's actual metric (Eq. 2: for each candidate frame,
    its best match anywhere in the reference, averaged) and is what
    `score_speech_similarity` uses. `recall` and `f1` are also returned for
    diagnostic/logging purposes only -- the paper explicitly found precision
    alone performs best and this module follows that, not F1.
    """
    if feats_a.shape[0] == 0 or feats_b.shape[0] == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    sim = _cosine_similarity_matrix(feats_a, feats_b)
    precision = float(sim.max(axis=1).mean())  # best match for each candidate frame
    recall = float(sim.max(axis=0).mean())     # best match for each reference frame
    f1 = 0.0 if (precision + recall) <= 0 else 2 * precision * recall / (precision + recall)
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def score_speech_similarity(feats_candidate: np.ndarray,
                            feats_reference: np.ndarray) -> tuple[float, dict[str, float]]:
    """SpeechBERTScore precision (Saeki et al. Eq. 2) -> this project's 0-5
    scale (PASS_THRESHOLD=4.0).

    Mapping is linear and UNCALIBRATED: `score = 5 * clamp(precision, 0, 1)`.
    Precision, not F1 -- see the module docstring's "Correction history" for
    why. No reasoning about a nonlinear shape is offered, because there is not
    yet a validated precision distribution large enough to calibrate a curve
    against; this is deliberately the simplest possible mapping.
    """
    result = speech_bertscore(feats_candidate, feats_reference)
    score = round(5.0 * min(max(result["precision"], 0.0), 1.0), 3)
    return score, result


class SpeechSimilarityScorer(Scorer):
    """Path 2: audio-to-audio pronunciation similarity against a human
    reference clip, using SpeechBERTScore precision (wavlm-large, final
    layer). See module docstring for the method, its correction history, and
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
        return "speech-similarity-v2"  # v2: precision+wavlm-large, not v1's F1+wav2vec2-base

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
                "model": MODEL_ID,
                "layer": "final",
            },
            notes=(
                "Audio-to-audio comparison (SpeechBERTScore precision, "
                f"{MODEL_ID}) against a {clip.source} human reference clip -- "
                "no ASR, no LLM, no phoneme decoding involved. Validated on a "
                "small slice with a wide discrimination margin and consistent "
                "voice-invariance across all 5 pairs tested (see module "
                "docstring). Covers only ~65% of ingredients (those with a "
                "reference clip). Not Workstream 1's hybrid judge."
            ),
        )
