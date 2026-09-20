"""Gemini TTS on Vertex AI.

Covers both planned Gemini tiers -- the cheap iteration tier (Flash TTS) and the
SOTA tier (Pro TTS) -- because they differ only by model id and price list, which
are config. Streaming is the default transport purely so time-to-first-audio is
measurable; the audio is identical either way.

WARNING: every successful call here is billable. Nothing in this module runs at
import time, and `check_connectivity` deliberately reads model metadata rather
than synthesising anything.
"""

from __future__ import annotations

import base64
import json
import re
import time

import requests

from . import gcp_auth
from .audio import pcm_duration_s, wrap_pcm_as_wav
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
RATE_RE = re.compile(r"rate=(\d+)")


def _host(location: str) -> str:
    return (
        "https://aiplatform.googleapis.com"
        if location == "global"
        else f"https://{location}-aiplatform.googleapis.com"
    )


def _model_url(project: str, location: str, model: str, method: str) -> str:
    return (
        f"{_host(location)}/v1/projects/{project}/locations/{location}"
        f"/publishers/google/models/{model}:{method}"
    )


class GeminiTTSAdapter(TTSAdapter):
    backend = "gemini_vertex"
    supports_streaming = True

    def __init__(self, config):
        super().__init__(config)
        o = config.options
        self.location = o.get("location", "global")
        self.stream = bool(o.get("stream", True))
        self.temperature = o.get("temperature")
        self.style_prompt = o.get("style_prompt")
        self.project = o.get("project") or None
        self._session = requests.Session()

    def close(self) -> None:
        self._session.close()

    def preflight(self) -> None:
        self.project = self.project or gcp_auth.project_id()
        gcp_auth.access_token()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {gcp_auth.access_token()}",
            "Content-Type": "application/json",
        }

    def _body(self, request: SynthesisRequest) -> dict:
        text = request.text
        if self.style_prompt:
            text = f"{self.style_prompt}: {text}"
        generation_config: dict = {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self.config.voice}}
            },
        }
        if self.temperature is not None:
            generation_config["temperature"] = float(self.temperature)
        return {
            "contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": generation_config,
        }

    def _raise_for_status(self, response: requests.Response) -> None:
        if response.ok:
            return
        detail = response.text[:500]
        message = f"vertex {response.status_code}: {detail}"
        if response.status_code in TRANSIENT_STATUSES:
            raise TransientError(message)
        raise PermanentError(message)

    def _synthesize(self, request: SynthesisRequest) -> RawAudio:
        project = self.project or gcp_auth.project_id()
        self.project = project
        method = "streamGenerateContent" if self.stream else "generateContent"
        url = _model_url(project, self.location, self.config.model, method)
        params = {"alt": "sse"} if self.stream else None
        timeout = (10.0, self.config.timeout_s)

        start = time.perf_counter()
        try:
            response = self._session.post(
                url,
                params=params,
                headers=self._headers(),
                json=self._body(request),
                timeout=timeout,
                stream=self.stream,
            )
        except requests.Timeout as exc:
            raise TransientError(f"vertex timeout: {exc}") from exc
        except requests.RequestException as exc:
            raise TransientError(f"vertex transport error: {exc}") from exc

        with response:
            self._raise_for_status(response)
            if self.stream:
                chunks, ttfa_ms, usage_meta, mime = self._consume_sse(response, start)
            else:
                payload = response.json()
                chunks, usage_meta, mime = self._collect([payload])
                ttfa_ms = None

        if not chunks:
            raise TransientError("vertex returned no audio parts")

        pcm = b"".join(chunks)
        fmt = self._format_for(mime)
        duration = pcm_duration_s(pcm, fmt)
        data = wrap_pcm_as_wav(pcm, fmt) if self.config.audio.container == "wav" else pcm

        return RawAudio(
            data=data,
            audio_format=fmt,
            usage=Usage(
                characters=len(request.text),
                audio_seconds=duration,
                input_tokens=usage_meta.get("promptTokenCount"),
                output_tokens=usage_meta.get("candidatesTokenCount"),
            ),
            ttfa_ms=ttfa_ms,
            audio_duration_s=duration,
            provider_request_id=response.headers.get("x-request-id"),
            provider_meta={"mime_type": mime, "usage_metadata": usage_meta},
        )

    def _consume_sse(self, response: requests.Response, start: float):
        chunks: list[bytes] = []
        usage_meta: dict = {}
        mime = ""
        ttfa_ms: float | None = None
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            new, meta, event_mime = self._collect([event])
            usage_meta.update(meta)
            mime = event_mime or mime
            if new and ttfa_ms is None:
                ttfa_ms = (time.perf_counter() - start) * 1000
            chunks.extend(new)
        return chunks, ttfa_ms, usage_meta, mime

    @staticmethod
    def _collect(events: list[dict]):
        chunks: list[bytes] = []
        usage_meta: dict = {}
        mime = ""
        for event in events:
            if "error" in event:
                raise PermanentError(f"vertex error payload: {event['error']}")
            usage_meta.update(event.get("usageMetadata") or {})
            for candidate in event.get("candidates") or []:
                for part in (candidate.get("content") or {}).get("parts") or []:
                    inline = part.get("inlineData") or part.get("inline_data")
                    if inline and inline.get("data"):
                        chunks.append(base64.b64decode(inline["data"]))
                        mime = inline.get("mimeType") or inline.get("mime_type") or mime
        return chunks, usage_meta, mime

    def _format_for(self, mime: str) -> AudioFormat:
        configured = self.config.audio
        match = RATE_RE.search(mime or "")
        sample_rate = int(match.group(1)) if match else configured.sample_rate_hz
        return AudioFormat(
            container=configured.container,
            encoding="pcm_s16le",
            sample_rate_hz=sample_rate,
            channels=1,
            sample_width_bytes=2,
        )


def check_connectivity(system_id: str = "gemini-flash-tts", timeout: float = 20.0) -> dict:
    """One non-billable GET against Vertex AI: mint a token, read the publisher
    model's metadata. No synthesis, no spend."""
    from .config import load_system

    config = load_system(system_id)
    location = config.options.get("location", "global")
    url = f"{_host(location)}/v1/publishers/google/models/{config.model}"
    result = {"system_id": system_id, "url": url, "location": location}
    try:
        token = gcp_auth.access_token()
        result["token_minted"] = True
    except Exception as exc:
        return {**result, "token_minted": False, "reachable": False, "error": f"{type(exc).__name__}: {exc}"}

    try:
        response = requests.get(
            url, headers={"Authorization": f"Bearer {token}"}, timeout=timeout
        )
    except requests.RequestException as exc:
        return {**result, "reachable": False, "error": f"{type(exc).__name__}: {exc}"}

    return {
        **result,
        "reachable": response.ok,
        "status_code": response.status_code,
        "body": response.text[:400],
    }


if __name__ == "__main__":
    print(json.dumps(check_connectivity(), indent=2))
