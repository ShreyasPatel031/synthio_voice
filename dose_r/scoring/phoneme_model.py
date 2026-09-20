"""Audio -> phoneme transcription, and locating a drug name's phoneme span
inside a full-sentence synthesized clip.

Why a phoneme recognizer instead of word-level ASR
---------------------------------------------------
`dose_r/scoring/asr_roundtrip.py` uses Google Cloud Speech-to-Text, a
word-level recognizer whose output is constrained to a fixed English
dictionary. `docs/REFERENCE_AUDIO_GROUNDING.md` measured directly that this
recognizer fails to transcribe even a *human* correctly saying a drug name
44.3% of the time (98/176 reference clips) -- not because the name was
mispronounced, but because the recognizer's language model has no slot for
that string of sounds and "autocorrects" it to the nearest real word or
phrase ("tofacitinib" -> "tofu Sydney"). That is a structural ceiling on
scoreability, not a fixable noise floor: no amount of tuning make a
dictionary-constrained decoder recognize a word that isn't in its
dictionary.

`facebook/wav2vec2-lv-60-espeak-cv-ft` (Xu et al., fine-tuned from
wav2vec2-large-lv60) is a CTC model whose output vocabulary is ~392 espeak/
IPA phoneme symbols, not English words. It has no dictionary to fall back
on and no notion of "valid English word" at all -- it can only emit the
sounds its acoustic model thinks it heard. That means it is structurally
unable to "autocorrect" an unfamiliar coined drug name into an unrelated
real word, which is exactly the failure mode this module exists to avoid.
It is not a fix for ASR bias in general, just a much-less-vocabulary-biased
recognizer for this specific purpose.

This is a proxy, not ground truth
----------------------------------
A phoneme recognizer has its own error rate and its own biases:
- It was trained on read/conversational speech across many languages via
  CommonVoice; synthesized TTS audio, especially the more robotic Standard
  tier, is out of its training distribution to some degree.
- espeak's ~392-symbol multilingual phoneme inventory does not cleanly
  1:1 cover every English allophone. Some acoustically-close sounds are
  routinely merged in the model's own output (e.g. its confusions are
  concentrated in vowel quality: schwa-like vowels such as /ə/, /ɐ/, /ᵻ/,
  /ʌ/ are often used near-interchangeably for the same reduced-vowel
  sound). A phoneme-level score that differs only in one of these vowels
  is a much weaker signal of mispronunciation than a full consonant or
  syllable-count mismatch.
- The model's own CTC decoding can drop or insert a phoneme the way any
  ASR system can drop or insert a word, independent of what was actually
  said.
This is why `phoneme_scorer.py` documents PER as a *similarity proxy*, not
a verified-correct pronunciation judgement, and why this module's docstring
repeats the caveat rather than presenting the result as solved.

Cost
----
This is a local model run entirely on CPU in-process. No network calls, no
per-clip API spend -- unlike `asr_roundtrip.py`'s Google Cloud Speech-to-Text
calls, running this scorer over the whole corpus costs $0.

Mechanism
---------
1. `transcribe_phonemes()` -- resample audio to 16 kHz (the model's expected
   input rate; our WAVs are a mix of 16k/22050/24000/44100/48000) and run it
   through the CTC model, returning a whitespace-separated string of espeak
   phoneme symbols (its native decoding join, confirmed empirically -- see
   `phoneme_scorer.py`'s module docstring for why whitespace tokenization is
   correct for this model's output rather than assumed).
2. The human reference clips are isolated recordings of just the drug name
   (~1s), but the cached synthesized clips in `runs/standin-v1/audio/` are
   full carrier sentences (~7s: "I recommend taking Advil to help relieve
   ..."). Comparing the reference's phonemes directly against the full
   sentence's phonemes would mostly measure "how long is the sentence",
   swamping any pronunciation signal -- so `locate_drug_phonemes()` first
   extracts just the drug name's phoneme span out of the full-sentence
   recognition, using the same alignment strategy
   `asr_roundtrip.locate_recognized_span()` uses for words (difflib
   sequence alignment between an expected token sequence and the
   recognized one, so it degrades gracefully when the recognizer drops or
   adds phonemes elsewhere in the sentence), reimplemented independently
   here at phoneme granularity: the "expected" sequence is built by
   phonemizing the sentence word-by-word with `phonemizer`'s espeak
   backend (the same phonemizer/backend the model itself was fine-tuned
   against, so the symbol inventories line up -- confirmed empirically),
   which also gives us the word boundary of each phoneme chunk so the
   drug's word span is known exactly rather than guessed.
"""

from __future__ import annotations

import difflib
import re
from functools import lru_cache

import librosa
import numpy as np

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
_TARGET_SR = 16_000

_EDGE_PUNCT = re.compile(r"^[^\w-]+|[^\w-]+$")
_TOKEN_RE = re.compile(r"\S+")


@lru_cache(maxsize=1)
def _get_model():
    """Lazy, memoized model+processor load (torch/transformers import cost and
    the ~1GB checkpoint download/load only happen once per process, and only
    when a caller actually needs audio transcription -- importing this module
    must not require a network call or a GPU/CPU-heavy load just to reuse the
    pure phoneme-distance functions elsewhere).
    """
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    # Decoding CTC ids does not need espeak-ng. phonemizer/espeak is only
    # required to phonemize *text*. This environment has no espeak binary,
    # so skip the backend. `phonemize_word()` still needs espeak and will
    # fail if called.
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID, do_phonemize=False)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    model.eval()
    return torch, processor, model


@lru_cache(maxsize=1)
def _get_phonemizer_backend():
    from phonemizer.backend import EspeakBackend

    return EspeakBackend("en-us", preserve_punctuation=False, with_stress=False,
                          language_switch="remove-flags")


@lru_cache(maxsize=1)
def _get_phoneme_separator():
    from phonemizer.separator import Separator

    # phone=' ' is what actually splits a word's phonemization into individual
    # symbol tokens -- the backend's default separator leaves them concatenated
    # (e.g. "ɹɛkəmɛnd" as one string), which would make `.split()` treat a
    # whole word as a single "phoneme" token and silently break alignment.
    return Separator(phone=" ", word="", syllable="")


def transcribe_phonemes(audio_source, sample_rate: int | None = None) -> str:
    """audio_source -> whitespace-separated espeak phoneme string.

    Accepts a path (str/Path) to any librosa-readable audio file, raw
    in-memory WAV bytes (e.g. from `forced_align.extract_drug_span_forced_align`,
    which never touches disk -- mirrors `speech_similarity.extract_frame_embeddings`'s
    same three-way input handling), or an already-loaded numpy float array
    (in which case `sample_rate` must be given so it can be resampled to the
    model's required 16 kHz). Handles resampling for our WAVs at
    16k/22050/24000/44100/48000 -- whatever the source rate is.
    """
    torch, processor, model = _get_model()

    if isinstance(audio_source, np.ndarray):
        if sample_rate is None:
            raise ValueError("sample_rate is required when passing a raw array")
        audio = audio_source
        if sample_rate != _TARGET_SR:
            audio = librosa.resample(
                audio.astype(np.float32), orig_sr=sample_rate, target_sr=_TARGET_SR
            )
    elif isinstance(audio_source, (bytes, bytearray)):
        import io

        import soundfile as sf

        raw, sr = sf.read(io.BytesIO(bytes(audio_source)), dtype="float32")
        audio = raw if raw.ndim == 1 else raw.mean(axis=1)
        if sr != _TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=_TARGET_SR)
    else:
        audio, _ = librosa.load(str(audio_source), sr=_TARGET_SR, mono=True)

    inputs = processor(audio, sampling_rate=_TARGET_SR, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]


def _norm_token(tok: str) -> str:
    return _EDGE_PUNCT.sub("", tok)


def phonemize_word(word: str) -> list[str]:
    """A single word/token -> its list of espeak phoneme symbols (text-based,
    via `phonemizer`'s espeak backend -- the same backend this model was
    fine-tuned against, per the model card). Used only to figure out *where*
    the drug name's phonemes should land inside a full-sentence recognition,
    not as a pronunciation target itself (the human reference clip's own
    audio transcription is the target -- see `locate_drug_phonemes`).
    """
    backend = _get_phonemizer_backend()
    cleaned = _norm_token(word)
    if not cleaned:
        return []
    phonemized = backend.phonemize([cleaned], separator=_get_phoneme_separator(), strip=True)[0]
    return phonemized.split()


def locate_drug_phonemes(sentence: str, drug: str, recognized_phonemes: str) -> str:
    """Extract the substring of `recognized_phonemes` (space-separated, from
    `transcribe_phonemes()` run on the full-sentence synthesized clip) that
    corresponds to `drug`'s position inside `sentence`.

    Builds an "expected" phoneme sequence by phonemizing the sentence
    word-by-word (so each expected phoneme token can be traced back to the
    word it came from), then aligns that expected sequence against the
    recognized one with `difflib.SequenceMatcher` -- mirroring
    `asr_roundtrip.locate_recognized_span()`'s approach at word level, applied
    here to phonemes so it stays robust when the model drops, adds, or
    mis-recognizes phonemes anywhere else in the sentence, not just at the
    drug's position.

    Returns "" if no recognized phonemes could be attributed to the drug's
    position at all.
    """
    lower_sentence = sentence.lower()
    d_start = lower_sentence.find(drug.lower())
    if d_start == -1:
        return ""
    d_end = d_start + len(drug)

    word_spans = [(m.start(), m.end(), m.group()) for m in _TOKEN_RE.finditer(sentence)]
    drug_word_idx = {i for i, (s, e, _) in enumerate(word_spans) if s < d_end and e > d_start}
    if not drug_word_idx:
        return ""

    expected_tokens: list[str] = []
    expected_word_of: list[int] = []
    for i, (_, _, w) in enumerate(word_spans):
        for ph in phonemize_word(w):
            expected_tokens.append(ph)
            expected_word_of.append(i)

    expected_drug_idx = {i for i, w in enumerate(expected_word_of) if w in drug_word_idx}
    if not expected_drug_idx:
        return ""

    hyp_tokens = recognized_phonemes.split()
    matcher = difflib.SequenceMatcher(None, expected_tokens, hyp_tokens, autojunk=False)

    hyp_idx: set[int] = set()
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        overlap = [i for i in expected_drug_idx if i1 <= i < i2]
        if not overlap:
            continue
        if tag == "delete":
            continue
        if tag == "equal":
            hyp_idx.update(j1 + (i - i1) for i in overlap)
        else:  # "replace" (many-to-many); "insert" never overlaps (i1 == i2)
            hyp_idx.update(range(j1, j2))

    if not hyp_idx:
        return ""
    lo, hi = min(hyp_idx), max(hyp_idx)
    return " ".join(hyp_tokens[lo : hi + 1])
