"""Extract the drug-name audio span via forced alignment against the KNOWN
sentence text -- a replacement for `audio_span.extract_drug_span`'s
Cloud-STT-timestamp approach.

Why this exists
----------------
`audio_span.py` locates the drug's audio span by asking Cloud STT to
*recognize* the sentence and reading off its word timestamps. Two real,
distinct bugs were found that way (see that module's docstring), and even
after three layered guardrails (padding clamp, duration-plausibility check,
retranscription verification), the underlying problem remains: Cloud STT's
own per-word timestamp is only approximately correct, off by a few hundred
milliseconds in a way none of those guardrails detect, because a *plausible*
timestamp is not necessarily an *exact* one. Confirmed by ear on
"esomeprazole" even after both bugs were fixed: the extracted clip still
audibly cut into the tail of the preceding word ("start").

Forced alignment does not have this problem because it never has to guess
*what* was said -- the target text (the full carrier sentence) is already
known exactly, from the dataset itself. `facebook/wav2vec2-lv-60-espeak-cv-ft`
(the phoneme CTC model already used by `scoring/phoneme_model.py` for Path 3)
is run once over the whole clip to get per-frame phoneme emission
probabilities, and `torchaudio.functional.forced_align` finds the
highest-probability frame-to-phoneme alignment CONSTRAINED to emit exactly
that known phoneme sequence, in order. There is no "did it recognize the
right word" step at all -- alignment, not recognition, and no ASR call
(no network cost either). Validated on "esomeprazole": produced a clean
1.021s-1.902s span, confirmed correct by the user listening to the result
(audio_span.py's Cloud-STT-based extraction still bled into "start" at that
point).

Mechanism
---------
1. Resample the full-sentence clip to 16kHz mono (the model's rate).
2. Run it through the CTC model once to get per-frame log-probabilities,
   shape (T_frames, vocab_size).
3. Phonemize the full sentence WORD BY WORD (`phoneme_model.phonemize_word`),
   keeping track of which word each phoneme symbol came from, and drop any
   symbol not in the model's own vocabulary (a real, observed gap: "tp" from
   "ATPase" elsewhere in one sentence, unrelated to any drug name so far) --
   `forced_align` requires every target token to be in-vocabulary.
4. Map the filtered phoneme symbols to token ids and call
   `torchaudio.functional.forced_align(log_probs, targets, blank=pad_token_id)`.
   Its return value is a per-FRAME vocab label, NOT a target-sequence
   position index -- getting this backwards was the first, wrong attempt at
   this (produced a nonsensical span spanning almost the whole clip, because
   phonemes repeat across a sentence and matching on raw vocab id matched
   every occurrence of any of the drug's constituent phonemes anywhere in
   the sentence, not just its own).
5. Walk the frame-label sequence in order, collapsing consecutive-identical
   runs (dropping the blank label) into segments. These segments correspond,
   1:1 AND IN ORDER, to the filtered target phoneme sequence built in step
   3 -- segment i's frame range is exactly target-phoneme i's alignment.
6. Locate the drug's word(s) in the sentence by plain text matching (no ASR
   involved -- the sentence and the drug name are both already-known ground
   truth strings, unlike audio_span.py which had to search a Cloud STT
   *hypothesis* transcript that could itself be wrong), find the
   filtered-phoneme index range belonging to those words, and take the
   min/max frame boundaries of that range's segments.
7. Convert frame indices to seconds using this clip's own measured frame
   stride (duration / num_frames -- not a hardcoded constant, since the
   exact value depends on input length and the model's conv stack padding),
   apply a small pad (much smaller than audio_span.py's 0.12s, since these
   boundaries are exact rather than approximate), clamped to the
   neighboring phoneme segment's own boundary so padding can never bleed
   into an adjacent word, and slice the original PCM.

This module does not need audio_span.py's duration-plausibility or
retranscription-verification guardrails -- those existed specifically to
catch Cloud STT's own timestamp corruption, which forced alignment (being
constrained to the correct, known text throughout) cannot exhibit in the
same way. A different failure mode remains possible in principle: forced
alignment is FORCED to fit the known text somewhere even for badly
mispronounced, truncated, or garbled audio, so a bad alignment could still
occur without any error being raised. Not yet stress-tested beyond the
items already checked by ear (esomeprazole, plus a validation slice before
any full-corpus rerun -- see scripts/validate_forced_align.py).
"""

from __future__ import annotations

import re
import wave
from io import BytesIO

_TOKEN_RE = re.compile(r"\S+")
_TARGET_SR = 16_000
_PAD_S = 0.03  # small: forced-align boundaries are exact, not approximate like audio_span.py's


def _locate_drug_word_indices(sentence: str, drug: str) -> set[int] | None:
    """Ground-truth word index set for `drug`'s position inside `sentence`
    -- no ASR/recognition involved, both strings are already known exactly.
    """
    lower_sentence = sentence.lower()
    d_start = lower_sentence.find(drug.lower())
    if d_start == -1:
        return None
    d_end = d_start + len(drug)

    word_spans = [(m.start(), m.end()) for m in _TOKEN_RE.finditer(sentence)]
    idx = {i for i, (s, e) in enumerate(word_spans) if s < d_end and e > d_start}
    return idx or None


def _build_target_sequence(sentence: str, vocab: dict[str, int]) -> tuple[list[int], list[int]]:
    """Phonemize `sentence` word-by-word, drop any symbol outside the model's
    vocab (see module docstring), and return (token_ids, word_of_token) --
    parallel lists; `word_of_token[i]` is the index (into
    `_TOKEN_RE.finditer(sentence)`) of the word token `i` came from.
    """
    from .scoring import phoneme_model

    token_ids: list[int] = []
    word_of_token: list[int] = []
    for i, m in enumerate(_TOKEN_RE.finditer(sentence)):
        for ph in phoneme_model.phonemize_word(m.group()):
            if ph not in vocab:
                continue
            token_ids.append(vocab[ph])
            word_of_token.append(i)
    return token_ids, word_of_token


def _decode_segments(frame_labels: list[int], blank_id: int) -> list[tuple[int, int, int]]:
    """Collapse consecutive-identical frame labels (dropping blank) into
    ordered (label, start_frame, end_frame_exclusive) segments -- these map
    1:1, IN ORDER, to the target token sequence `forced_align` was given.
    """
    segments: list[tuple[int, int, int]] = []
    prev = None
    seg_start = 0
    for f, lbl in enumerate(frame_labels + [None]):
        if lbl != prev:
            if prev is not None and prev != blank_id:
                segments.append((prev, seg_start, f))
            seg_start = f
            prev = lbl
    return segments


def extract_drug_span_forced_align(audio_bytes: bytes, sentence: str, drug: str,
                                   *, pad_s: float = _PAD_S) -> bytes | None:
    """Full-sentence WAV -> just the drug-name's audio, via forced alignment
    against the KNOWN sentence text. Returns None if the drug can't be
    located in the sentence text, if none of its phonemes survive vocab
    filtering, or if the alignment comes back degenerate (segment count
    mismatch) -- never guesses at a fallback span.
    """
    import librosa
    import torchaudio

    from .scoring import phoneme_model

    torch, processor, model = phoneme_model._get_model()
    vocab = processor.tokenizer.get_vocab()
    blank_id = processor.tokenizer.pad_token_id

    word_idx = _locate_drug_word_indices(sentence, drug)
    if word_idx is None:
        return None

    target_ids, word_of_token = _build_target_sequence(sentence, vocab)
    drug_token_positions = [i for i, w in enumerate(word_of_token) if w in word_idx]
    if not drug_token_positions:
        return None

    audio, _ = librosa.load(BytesIO(audio_bytes), sr=_TARGET_SR, mono=True)
    duration_s = len(audio) / _TARGET_SR

    inputs = processor(audio, sampling_rate=_TARGET_SR, return_tensors="pt")
    with torch.no_grad():
        log_probs = torch.log_softmax(model(inputs.input_values).logits, dim=-1)

    targets = torch.tensor([target_ids], dtype=torch.int32)
    input_lengths = torch.tensor([log_probs.shape[1]])
    target_lengths = torch.tensor([len(target_ids)])
    aligned_tokens, _scores = torchaudio.functional.forced_align(
        log_probs, targets, input_lengths, target_lengths, blank=blank_id,
    )
    frame_labels = aligned_tokens[0].tolist()
    segments = _decode_segments(frame_labels, blank_id)
    if len(segments) != len(target_ids):
        return None  # alignment degenerate; don't guess

    frame_stride = duration_s / log_probs.shape[1]
    drug_segments = [segments[i] for i in drug_token_positions]
    start_s = min(s for _, s, _ in drug_segments) * frame_stride
    end_s = max(e for _, _, e in drug_segments) * frame_stride

    # Pad, clamped to the neighboring phoneme segment's own boundary so
    # padding can never bleed into an adjacent word (mirrors
    # audio_span._clamp_padding's reasoning, against exact forced-align
    # boundaries instead of approximate Cloud STT ones).
    first_pos, last_pos = min(drug_token_positions), max(drug_token_positions)
    if first_pos > 0:
        prev_end_s = segments[first_pos - 1][2] * frame_stride
        start_s = max(prev_end_s, start_s - pad_s)
    else:
        start_s = max(0.0, start_s - pad_s)
    if last_pos < len(segments) - 1:
        next_start_s = segments[last_pos + 1][1] * frame_stride
        end_s = min(next_start_s, end_s + pad_s)
    else:
        end_s = min(duration_s, end_s + pad_s)

    with wave.open(BytesIO(audio_bytes), "rb") as w:
        rate, width, channels = w.getframerate(), w.getsampwidth(), w.getnchannels()
        w.setpos(max(0, int(start_s * rate)))
        n = int((end_s - start_s) * rate)
        frames = w.readframes(min(n, w.getnframes() - w.tell()))
    if not frames:
        return None

    out = BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(frames)
    return out.getvalue()
