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
- Which layer's hidden states to use is a real, undocumented choice in the
  wider literature (different layers encode different amounts of phonetic
  vs. speaker information). This module uses the final hidden layer, chosen
  for simplicity, not because it was swept and found best. Flagged as an
  open tuning question, not resolved here.
- This has NOT yet been validated -- see the module's companion validation
  script before trusting any number it produces. In particular, the
  voice-invariance property is a *claim from the literature*, not something
  this codebase has confirmed on drug-name audio specifically, until that
  validation is run and reported.

Cost
----
Local model, CPU inference, no network calls, no per-clip spend -- same as
`phoneme_model.py`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np

MODEL_ID = "facebook/wav2vec2-base"
_TARGET_SR = 16_000


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


def extract_frame_embeddings(audio_path_or_array, sample_rate: int | None = None) -> np.ndarray:
    """audio -> (T, 768) frame-level SSL hidden states (final layer).

    Accepts a path (any librosa-readable format) or a raw float array (in
    which case `sample_rate` is required for resampling to the model's 16kHz).
    """
    torch, extractor, model = _get_model()

    if isinstance(audio_path_or_array, np.ndarray):
        if sample_rate is None:
            raise ValueError("sample_rate is required when passing a raw array")
        audio = audio_path_or_array
        if sample_rate != _TARGET_SR:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sample_rate,
                                     target_sr=_TARGET_SR)
    else:
        audio, _ = librosa.load(str(audio_path_or_array), sr=_TARGET_SR, mono=True)

    inputs = extractor(audio, sampling_rate=_TARGET_SR, return_tensors="pt")
    with torch.no_grad():
        out = model(inputs.input_values)
    return out.last_hidden_state[0].numpy()  # (T, 768)


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
