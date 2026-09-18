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
Three rounds of fixes, in order, each found by testing rather than assuming:

1. **Model and layer (first draft -> corrected).** First draft used
   `facebook/wav2vec2-base`, final layer. The paper tested 7 SSL models
   (Table 5); wav2vec2-base was the WEAKEST of the six real ones (LCC 0.560;
   only `encodec` did worse at 0.087), and its final layers turned out to be
   completely broken for this task -- mismatched audio/reference pairs scored
   as high as correct ones. Switched to `microsoft/wavlm-large` (their best
   performer, LCC 0.581) and re-swept layers on the same slice: nearly every
   layer from 2 onward discriminates correctly, with the LAST layer (plain
   `last_hidden_state`, no special indexing) giving by far the widest margin.
   This matches the paper's own claim that most of their SSL models (unlike
   hubert-base) are "highly robust to layer selection" -- wav2vec2-base's
   first-draft brittleness is consistent with it not being on that list.

2. **Formula (F1 -> precision, matching the paper).** First draft used F1.
   The paper's own words: "While the original BERTScore defines precision,
   recall and F1-score, we use the precision as we found that it performed
   the best in our preliminary experiment" -- their Eq. 2 is precision only.
   Switched to match, initially without independently checking why.

3. **Formula (precision -> back to F1, deliberately diverging from the
   paper).** The paper's precision justification above is the ENTIRE
   explanation given -- one sentence, no ablation table, unlike their
   layer-choice (Figure 2) and model-choice (Table 5) claims which do show
   data. Tested directly on this project's own data whether that holds here:
   took a correct candidate clip and progressively truncated it (100% down to
   25% of its length), re-scoring against the same human reference each time.
   Precision fell ~40% (0.54->0.33); recall fell ~60% (0.62->0.24) over the
   same range -- precision is markedly less sensitive to a candidate that
   drops or truncates part of the drug name. That failure mode (a TTS cutting
   off part of a name) is exactly what this project needs to catch, unlike
   the paper's own task (correlating with naturalness ratings against ONE
   specific human recording, where over-penalizing a good synthesis for not
   replicating that recording's incidental quirks is the bigger risk they were
   guarding against). Switched back to F1 and re-ran the discrimination and
   voice-invariance checks precision had passed: F1 matched or slightly beat
   precision on both (see Validation below) -- so this gains truncation
   sensitivity at no measured cost on this project's own checks.

One thing the paper does NOT validate at all, that this project uses it for
anyway: their whole benchmark assumes reference and candidate always contain
the SAME words (correlating with human naturalness/quality ratings of a
matched pair). They never test whether the method can detect that the WRONG
word was said -- that discrimination check is this project's own addition,
not something their published results establish. It happens to work well
here (see validation below), but that is this project's finding, not theirs.

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
With wavlm-large + final layer, both F1 and precision were checked on the
same 5-item slice as the first draft:
- Discrimination: PASS for both, by a wide margin. F1: 5 correct pairs (min
  0.490) all comfortably above 3 mismatched pairs (max 0.255). Precision:
  0.467 vs. 0.240. F1's margin (0.235) is marginally wider than precision's
  (0.227).
- Voice-invariance (two different humans, same correct word): both score all
  5 pairs consistently -- F1: 0.747-0.802; precision: 0.747-0.827 (F1's range
  is slightly tighter). Notably, "Aspirin" (the first draft's unexplained low
  outlier at 1.77/5 under wav2vec2-base+F1) is NOT an outlier under EITHER
  metric with the corrected model (F1: 0.786; precision: 0.768 -- both
  squarely with the other four). Worth being honest about: the earlier
  hypothesis (a genuine dictionary-documented pronunciation variant for
  aspirin, see the Merriam-Webster note below) was independently confirmed
  against real data and is still a legitimate fact -- but the 1.77 score
  itself turns out to have been mostly a METHODOLOGY artifact (wrong
  model/layer, not the formula), not primarily the reference disagreement it
  was first attributed to.
- Truncation sensitivity (the deciding factor for F1 over precision -- see
  "Correction history" step 3): F1 tracks a truncated candidate's declining
  quality more steeply than precision does.
F1 is used because it does at least as well as precision on every check run
so far, and is more sensitive to the failure mode (truncation) this project
cares most about. Not yet validated at corpus scale, and still a proxy, not
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
Measured directly on Abilify (same candidate clip, both references): scoring
against Drugs.com's 1.05s reference gives precision 0.639 / F1 0.662; against
Merriam-Webster's 2.14s reference, precision 0.606 (~5% relative drop) / F1
0.506 (~24% relative drop). F1's recall term expects the candidate to account
for the reference's full length, so it is markedly more exposed to reference
pace than precision alone -- a real, accepted trade-off for gaining
truncation sensitivity (see "Correction history" step 3), not one this module
resolves. Scores should be read as comparative across systems scored against
the SAME reference, not as an absolute, pace-independent measure.

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
    """SpeechBERTScore precision/recall/F1 between two frame-embedding
    sequences, `a` treated as the candidate and `b` as the reference (order
    matters for precision, per the paper's Eq. 2). Pure, offline-testable:
    takes already-extracted embeddings, no audio or model involved.

    `precision` is the paper's own metric (Eq. 2: for each candidate frame,
    its best match anywhere in the reference, averaged); `recall` is the
    same computed the other way. `score_speech_similarity` uses `f1`, NOT
    `precision` -- a deliberate divergence from the paper, made after testing
    (see that function's docstring and the module docstring's "Correction
    history" step 3): precision was found weakly sensitive to a candidate
    truncating part of the drug name, which this project needs to catch.
    All three values are returned so callers can inspect the components a
    result was computed from, not just the final score.
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
    """SpeechBERTScore F1 -> this project's 0-5 scale (PASS_THRESHOLD=4.0).

    DELIBERATE, TESTED DEVIATION from the paper. Saeki et al. use precision
    alone, justified by one unexplained sentence with no ablation shown (see
    the module docstring's "Correction history"). A direct test on this
    project's own data (progressively truncating a correct candidate clip
    and re-scoring against its reference) found precision only weakly
    sensitive to truncation -- it dropped ~40% (0.54->0.33) from full-length
    to 25%-kept, while recall dropped ~60% (0.62->0.24) over the same range.
    Since a TTS system truncating or dropping part of a drug name is exactly
    the kind of failure this project needs to catch (unlike the paper's own
    task, general naturalness correlation against one specific recording,
    where over-penalizing a good synthesis for not replicating incidental
    recording quirks is the more relevant risk), F1 is used here instead.
    Re-validated with F1 on the same discrimination and voice-invariance
    checks precision passed (with the corrected wavlm-large/final-layer
    setup): equally clean discrimination margin (0.235 vs. precision's
    0.227) and an even tighter voice-invariance range (0.747-0.802 vs.
    0.747-0.827) -- switching to F1 cost nothing on the checks already run
    and gained truncation sensitivity.

    Mapping is linear and UNCALIBRATED: `score = 5 * clamp(f1, 0, 1)`. No
    reasoning about a nonlinear shape is offered, because there is not yet a
    validated F1 distribution large enough to calibrate a curve against;
    this is deliberately the simplest possible mapping.
    """
    result = speech_bertscore(feats_candidate, feats_reference)
    score = round(5.0 * min(max(result["f1"], 0.0), 1.0), 3)
    return score, result


class SpeechSimilarityScorer(Scorer):
    """Path 2: audio-to-audio pronunciation similarity against a human
    reference clip, using SpeechBERTScore F1 (wavlm-large, final layer) --
    a deliberate, tested divergence from the paper's own precision-only
    choice, made for truncation sensitivity. See module docstring for the
    full correction history and what this does and doesn't establish.
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
        return "speech-similarity-v3"  # v1: F1+wav2vec2-base. v2: precision+wavlm-large.
                                       # v3: F1+wavlm-large (current) -- precision was tried
                                       # and dropped for weak truncation sensitivity, see
                                       # module docstring's "Correction history" step 3.

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
                "Audio-to-audio comparison (SpeechBERTScore F1, "
                f"{MODEL_ID}) against a {clip.source} human reference clip -- "
                "no ASR, no LLM, no phoneme decoding involved. F1 rather than "
                "the paper's precision-only choice: tested more sensitive to "
                "a candidate truncating/dropping part of the name (see module "
                "docstring), at the cost of more sensitivity to reference "
                "recording pace. Validated on a small slice with a wide "
                "discrimination margin and consistent voice-invariance across "
                "all 5 pairs tested. Covers only ~65% of ingredients (those "
                "with a reference clip). Not Workstream 1's hybrid judge."
            ),
        )
