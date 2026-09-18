"""Google Cloud Text-to-Speech adapter (the cheap iteration tier).

Covers Standard, WaveNet, Neural2 and Chirp3-HD from one class -- they differ only
by voice name and price, which is exactly the "config entry, not new code" property
the adapter layer is supposed to have.
"""

from __future__ import annotations

import base64
from typing import Any

import requests

from .. import auth
from ..config import VoiceSpec
from .base import NonRetryableError, RetryableError, TTSAdapter

_ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"
_SAMPLE_RATE = 24_000

# 429 (quota) and 5xx (backend trouble) are transient by nature -- the same
# request tends to succeed moments later. Everything else in the 4xx range
# (400 bad input, 401/403 auth, 404 not-found, ...) reflects something wrong
# with the request itself, which retrying cannot fix.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _parse_retry_after(value: str | None) -> float | None:
    """Cloud TTS sends Retry-After as a plain integer number of seconds (the
    HTTP-date form exists in the spec but isn't what this API returns). Treat
    anything we can't parse as absent rather than raising -- worst case we
    fall back to our own backoff schedule.
    """
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


class GoogleTTSAdapter(TTSAdapter):
    """Non-streaming REST synthesis against Cloud TTS."""

    def __init__(self, spec: VoiceSpec, *, timeout_s: float = 90.0,
                 session: requests.Session | None = None):
        super().__init__(spec)
        self.timeout_s = timeout_s
        # One session per adapter: connection reuse keeps latency measurements from
        # being dominated by repeated TLS handshakes.
        self._session = session or requests.Session()

    def _synthesize(self, text: str) -> tuple[bytes, dict[str, Any]]:
        body = {
            "input": {"text": text},
            "voice": {
                "languageCode": self.spec.language_code,
                "name": self.spec.voice_name,
            },
            "audioConfig": {
                "audioEncoding": "LINEAR16",
                "sampleRateHertz": _SAMPLE_RATE,
            },
        }
        try:
            resp = self._session.post(
                _ENDPOINT, headers=auth.auth_headers(), json=body, timeout=self.timeout_s
            )
        except requests.Timeout as exc:
            # A slow response is not evidence the request was bad -- retry it.
            raise RetryableError(f"timeout after {self.timeout_s}s: {exc}") from exc
        except requests.ConnectionError as exc:
            raise RetryableError(f"connection error: {exc}") from exc

        if resp.status_code != 200:
            detail = f"HTTP {resp.status_code}: {resp.text[:300]}"
            if resp.status_code in _RETRYABLE_STATUS:
                retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                raise RetryableError(detail, retry_after_s=retry_after)
            raise NonRetryableError(detail)

        payload = resp.json().get("audioContent")
        if not payload:
            raise RuntimeError("response contained no audioContent")

        audio = base64.b64decode(payload)
        if not audio.startswith(b"RIFF"):
            raise RuntimeError("decoded audio is not RIFF/WAV")

        return audio, {
            "audio_format": "wav",
            "sample_rate_hz": _SAMPLE_RATE,
            "streaming": False,   # REST is request/response, so TTFA is not measurable
            "ttfa_ms": None,
            "extra": {"voice": self.spec.voice_name, "tier": self.spec.tier},
        }
