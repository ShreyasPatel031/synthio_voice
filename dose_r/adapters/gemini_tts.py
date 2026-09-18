"""Gemini TTS adapter via Vertex AI generateContent.

This is the tier the project plan actually asked for as the cheap iteration
target -- the closest available proxy to the Gemini TTS system DOSE itself
scores at 74.5%. It was believed unreachable from this project: a GET on the
publisher-model resource path 404s. That GET is not how you invoke the model,
though -- it is a probe of the resource descriptor, not a call to it. A real
POST .../publishers/google/models/<model>:generateContent request returns 400
INVALID_ARGUMENT (a malformed body, but a live one) rather than 404, and with a
correctly-shaped body returns 200 with real audio. Confirmed manually before
writing this adapter: gemini-2.5-flash-preview-tts, 24 kHz PCM, HTTP 200.

Audio comes back as inline base64 PCM (audio/L16;codec=pcm;rate=<hz>), not the
WAV container Cloud TTS returns, so this adapter wraps the raw PCM in a WAV
header itself rather than reusing GoogleTTSAdapter's decode path.
"""

from __future__ import annotations

import base64
import re
import struct
from typing import Any

import requests

from .. import auth
from ..config import GCP_PROJECT, VoiceSpec
from .base import NonRetryableError, RetryableError, TTSAdapter

_LOCATION = "us-central1"
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# e.g. "audio/L16;codec=pcm;rate=24000" -- the sample rate is the only field
# we need out of it; codec/bit-depth are assumed to be what the docs promise
# (16-bit signed PCM) since Gemini TTS does not offer an alternative today.
_MIME_RATE_RE = re.compile(r"rate=(\d+)")


def _wav_header(n_bytes: int, sample_rate: int, *, channels: int = 1,
                bits: int = 16) -> bytes:
    block_align = channels * bits // 8
    return (
        b"RIFF" + struct.pack("<I", 36 + n_bytes) + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate,
                      sample_rate * block_align, block_align, bits)
        + b"data" + struct.pack("<I", n_bytes)
    )


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


class GeminiTTSAdapter(TTSAdapter):
    """Non-streaming Vertex AI `generateContent` call with audio output."""

    def __init__(self, spec: VoiceSpec, *, timeout_s: float = 90.0,
                 session: requests.Session | None = None,
                 project: str = GCP_PROJECT, location: str = _LOCATION,
                 speaker: str = "Kore"):
        super().__init__(spec)
        self.timeout_s = timeout_s
        self._session = session or requests.Session()
        self._url = (
            f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}"
            f"/locations/{location}/publishers/google/models/{spec.voice_name}"
            f":generateContent"
        )
        # spec.voice_name is the Gemini *model* id for this tier (e.g.
        # gemini-2.5-flash-preview-tts) -- there is no separate voice selector
        # the way Cloud TTS has one. The *speaker* is a distinct concept: one of
        # Gemini's prebuilt voice names (Kore, Puck, ...), passed explicitly
        # rather than smuggled through VoiceSpec.tags, which is a set of
        # unordered labels and not a reliable place to encode an ordered field.
        self._speaker = speaker

    def _synthesize(self, text: str) -> tuple[bytes, dict[str, Any]]:
        body = {
            "contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self._speaker}}
                },
            },
        }
        try:
            resp = self._session.post(
                self._url, headers=auth.auth_headers(), json=body, timeout=self.timeout_s
            )
        except requests.Timeout as exc:
            raise RetryableError(f"timeout after {self.timeout_s}s: {exc}") from exc
        except requests.ConnectionError as exc:
            raise RetryableError(f"connection error: {exc}") from exc

        if resp.status_code != 200:
            detail = f"HTTP {resp.status_code}: {resp.text[:300]}"
            if resp.status_code in _RETRYABLE_STATUS:
                retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                raise RetryableError(detail, retry_after_s=retry_after)
            raise NonRetryableError(detail)

        data = resp.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"unexpected response shape: {str(data)[:300]}") from exc

        inline = next((p["inlineData"] for p in parts if "inlineData" in p), None)
        if inline is None:
            raise RuntimeError(f"no audio part in response: {str(parts)[:300]}")

        mime = inline.get("mimeType", "")
        rate_match = _MIME_RATE_RE.search(mime)
        sample_rate = int(rate_match.group(1)) if rate_match else 24_000

        pcm = base64.b64decode(inline["data"])
        wav = _wav_header(len(pcm), sample_rate) + pcm

        return wav, {
            "audio_format": "wav",
            "sample_rate_hz": sample_rate,
            "streaming": False,
            "ttfa_ms": None,
            "extra": {"model": self.spec.voice_name, "speaker": self._speaker,
                      "tier": self.spec.tier, "raw_mime_type": mime},
        }
