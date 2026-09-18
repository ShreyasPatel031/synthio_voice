"""Extract the drug-name audio span from a full-sentence synthesized clip.

Why this exists
----------------
Path 2's audio-to-audio comparison (`scoring/speech_similarity.py`) needs to
compare like with like: the human reference clips are isolated ~1s
recordings of just the drug name, but every cached synthesized clip is a
full ~7s carrier sentence ("We plan to start Abilify at ten milligrams
daily..."). The first validation run compared whole-sentence embeddings
against single-word embeddings directly -- and failed its own discrimination
check as a result (mismatched audio/reference pairs scored as high as
correct ones), because a 7-second sentence and a 1-second word are similar
in generic "this is speech" ways regardless of which word either contains.
This module fixes the actual bug: slice out just the drug's audio before
any comparison happens.

`asr_roundtrip.py` already requests `enableWordTimeOffsets: True` from Cloud
STT, but only keeps the recognized word *text*
(`_extract_words` -> `wi["word"]`), discarding `wi["startTime"]`/
`wi["endTime"]`. That data was never persisted, so getting it back means one
new STT call per clip -- this module is a separate, one-time cost from the
scoring itself, not a per-scorer-run cost: cache the extracted span once per
(system, item) and every downstream metric that needs "just the drug's
audio" reuses the same cached clip.

Mechanism
---------
1. Call Cloud STT with `enableWordTimeOffsets: True` on the full-sentence
   clip (same endpoint `asr_roundtrip.py` uses).
2. Locate the drug's word(s) in the recognized transcript using
   `asr_roundtrip.locate_recognized_span`'s alignment logic, applied here to
   words-with-timing instead of bare word strings.
3. Slice the raw PCM samples between the matched span's start and end time,
   with a fixed padding buffer on each side (attack/decay of the word can
   extend slightly past ASR's word boundary), and re-wrap as a standalone
   WAV file.

Failure mode: if the drug's span can't be located in the transcript (the
same case `asr_roundtrip.py` reports as an empty `recognized_span`), this
returns `None` rather than guessing at a arbitrary audio range -- comparing
against a wrong slice would be worse than not comparing at all.

Two real, confirmed contamination bugs (found by listening to extracted
spans, not by inspection alone) -- read before trusting extraction on a
coined/novel drug name
------------------------------------------------------------------------------
1. **Padding bleeding into a neighboring word.** A fixed padding buffer was
   applied without checking whether it crosses into an adjacent word's own
   timespan. Confirmed on "esomeprazole": Cloud STT garbled the word into
   two unrelated real words ("a", "summer", 0.8-1.6s) -- the alignment
   correctly located that time range, but the immediately preceding word
   ("start") ends at exactly 0.8s, so subtracting the padding pulled the
   clip's start back to 0.68s, INSIDE "start"'s own span. The extracted clip
   audibly contained the tail of "start" plus "esomeprazole" plus the start
   of the next word. Fixed by `_clamp_padding`: padding is capped at the
   midpoint of the gap to the neighboring word, never past it.
2. **Cloud STT returning a single corrupted, absurdly-wide timestamp.**
   Confirmed on "talquetamab": STT collapsed six actual spoken words
   ("talquetamab targets GPRC5D on multiple myeloma") into ONE garbled
   token ("tultul") with a reported span of 0.6s to 5.7s -- a single "word"
   5.1 seconds long. The alignment correctly identified that token as
   corresponding to the drug's position; the bug is that Cloud STT's own
   timestamp for it cannot be trusted at all. No amount of padding logic
   fixes this -- it requires an independent sanity check, which
   `_duration_is_plausible` and `_verify_by_retranscription` provide. An
   item failing either check returns `None` (unscoreable) rather than
   silently including a contaminated span; this project's Scorer classes
   already treat that as scoreable=False, never as a false 0.
"""

from __future__ import annotations

import base64
import statistics
import struct
import wave
from io import BytesIO
from typing import Any

import requests

from . import auth

_STT_ENDPOINT = "https://speech.googleapis.com/v1/speech:recognize"
_PAD_S = 0.12  # seconds of padding on each side of the matched word span, before clamping

# A span longer than this multiple of the sentence's own average per-word
# duration is treated as almost certainly contaminated (either a
# too-wide alignment match or, per bug #2 above, a corrupted STT timestamp).
# 4x is deliberately generous: drug names are typically longer than an
# average English word (more syllables), so a plain word-count-based ratio
# would flag legitimate long names as false positives. Chosen to comfortably
# clear "esomeprazole" (a real 5-syllable name) while catching "talquetamab"'s
# 5.1s single-token blob, which was ~13x the sentence's own average.
_MAX_DURATION_RATIO = 4.0

# After extraction, the span is re-transcribed in isolation (a second,
# independent STT call) and rejected if it contains implausibly many
# recognizable words for what should be one drug name. +2 tolerates a
# multi-word ingredient (e.g. "insulin icodec-abae") plus one garbled extra
# token without over-tolerating genuine contamination.
_MAX_VERIFICATION_WORDS_SLACK = 2


def _extract_words_with_timing(response_json: dict) -> list[dict[str, Any]]:
    """[{word, start_s, end_s}, ...] from a Cloud STT response, or [] if none."""
    words: list[dict[str, Any]] = []
    for result in response_json.get("results", []):
        alt = result.get("alternatives", [{}])[0]
        for wi in alt.get("words", []):
            start = wi.get("startTime", "0s")
            end = wi.get("endTime", "0s")
            words.append({
                "word": wi["word"],
                "start_s": float(start.rstrip("s")) if isinstance(start, str) else start,
                "end_s": float(end.rstrip("s")) if isinstance(end, str) else end,
            })
    return words


def _locate_span_indices(sentence: str, drug: str,
                          words: list[dict[str, Any]]) -> tuple[int, int] | None:
    """Like the old `_locate_span_with_timing`, but returns the matched word
    INDICES (i, j) into `words` rather than just their timestamps -- needed so
    the caller can look at the neighboring words for padding clamping.
    """
    from .scoring.asr_roundtrip import locate_recognized_span

    hyp_words = [w["word"] for w in words]
    matched = locate_recognized_span(sentence, drug, hyp_words)
    if not matched:
        return None

    matched_words = matched.split()
    for i in range(len(hyp_words) - len(matched_words) + 1):
        if hyp_words[i : i + len(matched_words)] == matched_words:
            return i, i + len(matched_words) - 1
    return None


def _clamp_padding(words: list[dict[str, Any]], i: int, j: int,
                   pad_s: float = _PAD_S) -> tuple[float, float]:
    """Pad the matched span [i, j] by `pad_s` on each side, but never past the
    midpoint of the gap to a neighboring word -- fixes the confirmed
    "esomeprazole" bug where a fixed pad bled into the tail of "start".
    """
    start_s, end_s = words[i]["start_s"], words[j]["end_s"]

    if i > 0:
        gap_before = start_s - words[i - 1]["end_s"]
        start_s -= min(pad_s, max(0.0, gap_before) / 2)
    else:
        start_s = max(0.0, start_s - pad_s)

    if j < len(words) - 1:
        gap_after = words[j + 1]["start_s"] - end_s
        end_s += min(pad_s, max(0.0, gap_after) / 2)
    else:
        end_s += pad_s

    return start_s, end_s


def _duration_is_plausible(words: list[dict[str, Any]], span_duration_s: float) -> bool:
    """Sanity check independent of the alignment itself: is this span's
    duration consistent with the sentence's own observed speaking pace?
    Catches the confirmed "talquetamab" bug (Cloud STT returning one
    corrupted 5.1s timestamp for what should be a single word) that no
    amount of padding-clamping can fix, since the corruption is in STT's
    own output, not in this module's alignment logic.
    """
    word_durations = [w["end_s"] - w["start_s"] for w in words if w["end_s"] > w["start_s"]]
    if not word_durations:
        return True  # nothing to compare against; do not block on this alone
    avg = statistics.mean(word_durations)
    return span_duration_s <= avg * _MAX_DURATION_RATIO


def _verify_by_retranscription(clip_bytes: bytes, drug: str, sample_rate: int,
                               session: Any, timeout_s: float) -> bool:
    """Second, independent check (per the observation that a duration check
    alone would not catch every contamination case): re-transcribe the
    extracted span in isolation and reject it if it contains implausibly many
    recognizable words for what should be one drug name. A garbled or empty
    transcription is EXPECTED and fine -- that is the whole reason ASR-based
    scoring is unreliable for coined names in the first place -- so this only
    rejects on too many recognized words, never on zero or an unrecognized
    word.
    """
    body = {
        "config": {"encoding": "LINEAR16", "sampleRateHertz": sample_rate,
                  "languageCode": "en-US", "model": "latest_long"},
        "audio": {"content": base64.b64encode(clip_bytes).decode("ascii")},
    }
    try:
        resp = session.post(_STT_ENDPOINT, headers=auth.auth_headers(), json=body,
                            timeout=timeout_s)
        if resp.status_code != 200:
            return True  # a transport hiccup on the verification call should not
                         # itself invalidate an otherwise-good span
        results = resp.json().get("results", [])
    except Exception:
        return True

    n_words = sum(len(r.get("alternatives", [{}])[0].get("transcript", "").split())
                 for r in results)
    max_allowed = len(drug.split()) + _MAX_VERIFICATION_WORDS_SLACK
    return n_words <= max_allowed


def extract_drug_span(audio_bytes: bytes, sentence: str, drug: str,
                      *, session: Any = None, timeout_s: float = 60.0,
                      verify: bool = True) -> bytes | None:
    """Full-sentence WAV -> just the drug-name's audio, or None if it
    couldn't be located OR failed a contamination check (see module
    docstring's two confirmed bugs). Costs one Cloud STT call, plus a second
    verification call unless `verify=False`.
    """
    session = session if session is not None else requests
    body = {
        "config": {"encoding": "LINEAR16", "sampleRateHertz": 24000,
                  "languageCode": "en-US", "model": "latest_long",
                  "enableWordTimeOffsets": True},
        "audio": {"content": base64.b64encode(audio_bytes).decode("ascii")},
    }
    resp = session.post(_STT_ENDPOINT, headers=auth.auth_headers(), json=body, timeout=timeout_s)
    if resp.status_code != 200:
        raise RuntimeError(f"STT HTTP {resp.status_code}: {resp.text[:300]}")

    words = _extract_words_with_timing(resp.json())
    if not words:
        return None
    indices = _locate_span_indices(sentence, drug, words)
    if indices is None:
        return None
    i, j = indices

    start_s, end_s = _clamp_padding(words, i, j)
    if not _duration_is_plausible(words, end_s - start_s):
        return None  # e.g. the confirmed "talquetamab" 5.1s corrupted-timestamp case

    with wave.open(BytesIO(audio_bytes), "rb") as w:
        rate, width, channels = w.getframerate(), w.getsampwidth(), w.getnchannels()
        w.setpos(max(0, int(start_s * rate)))
        n_frames = int((end_s - start_s) * rate)
        frames = w.readframes(min(n_frames, w.getnframes() - w.tell()))

    if not frames:
        return None

    out = BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(frames)
    clip = out.getvalue()

    if verify and not _verify_by_retranscription(clip, drug, rate, session, timeout_s):
        return None  # e.g. the confirmed "esomeprazole" padding-bleed case, if
                     # clamping somehow still left extra recognizable content

    return clip
