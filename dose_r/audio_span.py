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
"""

from __future__ import annotations

import base64
import struct
import wave
from io import BytesIO
from typing import Any

import requests

from . import auth

_STT_ENDPOINT = "https://speech.googleapis.com/v1/speech:recognize"
_PAD_S = 0.12  # seconds of padding on each side of the matched word span


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


def _locate_span_with_timing(sentence: str, drug: str,
                              words: list[dict[str, Any]]) -> tuple[float, float] | None:
    """Reuses asr_roundtrip's word-alignment logic, but on timed words."""
    from .scoring.asr_roundtrip import locate_recognized_span

    hyp_words = [w["word"] for w in words]
    matched = locate_recognized_span(sentence, drug, hyp_words)
    if not matched:
        return None

    matched_words = matched.split()
    # Find the contiguous run in hyp_words that produced `matched` -- the
    # alignment already guarantees `matched` is a contiguous slice, so a
    # single index search is sufficient and avoids re-implementing alignment.
    for i in range(len(hyp_words) - len(matched_words) + 1):
        if hyp_words[i : i + len(matched_words)] == matched_words:
            return words[i]["start_s"], words[i + len(matched_words) - 1]["end_s"]
    return None


def extract_drug_span(audio_bytes: bytes, sentence: str, drug: str,
                      *, session: Any = None, timeout_s: float = 60.0) -> bytes | None:
    """Full-sentence WAV -> just the drug-name's audio, or None if it
    couldn't be located. Costs one Cloud STT call.
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
    span = _locate_span_with_timing(sentence, drug, words)
    if span is None:
        return None

    start_s, end_s = span
    start_s = max(0.0, start_s - _PAD_S)
    end_s = end_s + _PAD_S

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
    return out.getvalue()
