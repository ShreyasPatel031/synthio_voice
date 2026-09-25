"""OpenAI-compatible HTTP TTS (`POST /v1/audio/speech`).

Covers self-hosted servers that speak the OpenAI speech schema — Kokoro Misaki
on Cloud Run, and any later OSS stack that reuses the same path. Endpoint URL,
API key, and payload extras live in systems.yaml `options`.
"""

from __future__ import annotations

import os
import time

import requests

from .audio import wav_duration_s
from .base import (
    AudioFormat,
    PermanentError,
    RawAudio,
    SynthesisRequest,
    TTSAdapter,
    TransientError,
    Usage,
)

TRANSIENT_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}


class OpenAICompatibleHTTPAdapter(TTSAdapter):
    backend = "openai_compatible_http"
    supports_streaming = False

    def __init__(self, config):
        super().__init__(config)
        o = config.options
        base = (o.get("base_url") or os.environ.get("OSS_TTS_BASE_URL") or "").rstrip("/")
        self.base_url = base
        self.api_key = o.get("api_key") or os.environ.get("OSS_TTS_API_KEY") or ""
        self.speed = float(o.get("speed", 1.0))
        self.response_format = o.get("response_format", "wav")
        self.extra_body = dict(o.get("extra_body") or {})
        self._session = requests.Session()

    def close(self) -> None:
        self._session.close()

    def preflight(self) -> None:
        if not self.base_url:
            raise PermanentError(
                "openai_compatible_http needs options.base_url or OSS_TTS_BASE_URL"
            )
        url = f"{self.base_url}/healthz"
        try:
            r = self._session.get(url, timeout=15)
        except requests.RequestException as exc:
            raise PermanentError(f"preflight failed: {exc}") from exc
        if not r.ok:
            # Some servers lack /healthz; try models as a soft check.
            r2 = self._session.get(f"{self.base_url}/v1/models", timeout=15)
            if not r2.ok:
                raise PermanentError(f"preflight {r.status_code}: {r.text[:200]}")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "audio/wav, */*"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _body(self, request: SynthesisRequest) -> dict:
        body = {
            "model": self.config.model or "kokoro-misaki",
            "input": request.text,
            "voice": self.config.voice or "af_heart",
            "speed": self.speed,
            "response_format": self.response_format,
        }
        body.update(self.extra_body)
        return body

    def _raise_for_status(self, response: requests.Response) -> None:
        if response.ok:
            return
        detail = response.text[:500]
        message = f"openai_compatible_http {response.status_code}: {detail}"
        if response.status_code in TRANSIENT_STATUSES:
            raise TransientError(message)
        raise PermanentError(message)

    def _synthesize(self, request: SynthesisRequest) -> RawAudio:
        if not self.base_url:
            raise PermanentError(
                "openai_compatible_http needs options.base_url or OSS_TTS_BASE_URL"
            )
        url = f"{self.base_url}/v1/audio/speech"
        start = time.perf_counter()
        try:
            response = self._session.post(
                url,
                headers=self._headers(),
                json=self._body(request),
                timeout=(10.0, self.config.timeout_s),
            )
        except requests.Timeout as exc:
            raise TransientError(f"timeout: {exc}") from exc
        except requests.RequestException as exc:
            raise TransientError(f"request failed: {exc}") from exc
        self._raise_for_status(response)
        # Non-streaming: TTFA ≃ full latency once body lands.
        ttfa_ms = (time.perf_counter() - start) * 1000
        data = response.content
        if not data or len(data) < 44:
            raise PermanentError("empty or truncated audio response")
        fmt = AudioFormat(
            container="wav",
            encoding="pcm_s16le",
            sample_rate_hz=self.config.audio.sample_rate_hz,
            channels=1,
        )
        try:
            duration = wav_duration_s(data)
        except Exception:
            duration = None
        return RawAudio(
            data=data,
            audio_format=fmt,
            usage=Usage(characters=len(request.text), audio_seconds=duration or 0.0),
            ttfa_ms=ttfa_ms,
            audio_duration_s=duration,
            provider_meta={"endpoint": url},
        )
