"""Score synthesized audio against a human reference recording instead of the
drug's spelling.

Every scorer so far compared a TTS system's recognized speech to the *spelled*
drug name (`item.drug`). That conflates two different failure modes that a
prior run could not separate (see `runs/asr-roundtrip-v1-full`, where
"tofacitinib" -> "tofu Sydney" was flagged as ambiguous between the two):

  1. The TTS system mispronounced the name.
  2. Cloud STT does not recognize the name at all, regardless of who says it.

With a real human pronouncing the name (Workstream 1's Drugs.com reference
clips), (2) becomes directly measurable: run the SAME recognizer on the
reference clip. If it also fails to transcribe a human saying "tofacitinib"
correctly, that failure is attributable to the recognizer, not the TTS system,
for every synthesized clip of that item. This module computes exactly that
per-ingredient baseline and re-scores synthesized audio against it.

This is still not Workstream 1's hybrid judge -- it is a second, independent,
audio-grounded proxy, limited to the ~62% of items with a committed reference
clip (see `dose_r.references.reference_clips`) and to whatever Cloud STT
itself can recognize.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import requests

from ..scoring.asr_roundtrip import score_pronunciation
from ..scoring.base import PASS_THRESHOLD
from .reference_clips import ReferenceClip, stt_config_for_clip

_STT_ENDPOINT = "https://speech.googleapis.com/v1/speech:recognize"


@dataclass(frozen=True)
class ReferenceTranscript:
    ingredient: str
    source: str
    recognized: str | None      # None if Cloud STT returned no result at all
    confidence: float | None
    asr_recognizable: bool      # did the recognizer, on a HUMAN saying it, get it right?
    recognizable_score: float | None  # score_pronunciation(ingredient, recognized)


def transcribe_reference_clip(
    clip: ReferenceClip, auth_headers: dict[str, str], *, timeout_s: float = 60.0,
    session: Any = None,
) -> ReferenceTranscript:
    """One Cloud STT call against a human reference recording.

    `session` defaults to the `requests` module itself (its `.post` has the
    same signature as `requests.Session.post`), and accepts a fake in tests --
    the same injection pattern the TTS adapters use for offline testing.
    """
    session = session if session is not None else requests
    audio_b64 = base64.b64encode(clip.path.read_bytes()).decode("ascii")
    body = {"config": stt_config_for_clip(clip), "audio": {"content": audio_b64}}

    resp = session.post(_STT_ENDPOINT, headers=auth_headers, json=body, timeout=timeout_s)
    if resp.status_code != 200:
        raise RuntimeError(f"STT HTTP {resp.status_code}: {resp.text[:300]}")

    results = resp.json().get("results", [])
    if not results:
        # No speech recognized at all -- the clip itself is unrecognizable to
        # this ASR, which is itself the finding, not an error to hide.
        return ReferenceTranscript(clip.ingredient, clip.source, None, None,
                                   asr_recognizable=False, recognizable_score=None)

    alt = results[0]["alternatives"][0]
    recognized = alt.get("transcript", "").strip()
    confidence = alt.get("confidence")

    score, _ = score_pronunciation(clip.ingredient, recognized)
    return ReferenceTranscript(
        clip.ingredient, clip.source, recognized, confidence,
        asr_recognizable=score >= PASS_THRESHOLD, recognizable_score=score,
    )


def score_against_reference(
    reference: ReferenceTranscript, synth_recognized_span: str,
) -> dict[str, Any]:
    """Compare a synthesized clip's recognized span to the reference's, not to
    the spelled name. Returns a dict rather than a full ScoreResult so callers
    can merge it into an existing scored record without redefining the schema.
    """
    if reference.recognized is None:
        # The reference clip itself produced no transcript. Scoring the synth
        # clip against "nothing" would silently manufacture a failure that has
        # nothing to do with the TTS system.
        return {"scoreable": False, "reason": "reference clip unrecognized by ASR",
                "reference_asr_recognizable": False}

    score, components = score_pronunciation(reference.recognized, synth_recognized_span)
    return {
        "scoreable": True,
        "score_vs_reference": round(score, 3),
        "passed_vs_reference": score >= PASS_THRESHOLD,
        "components_vs_reference": components,
        "reference_transcript": reference.recognized,
        "reference_asr_recognizable": reference.asr_recognizable,
        "reference_source": reference.source,
    }
