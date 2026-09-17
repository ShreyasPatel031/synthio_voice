"""Gemini adapter internals, exercised offline. No live call is made here."""

from __future__ import annotations

import base64
import json

import pytest

from dose_r.adapters import PermanentError, TransientError, build_adapter, load_system
from dose_r.adapters.gemini import _model_url


class StubResponse:
    def __init__(self, status_code=200, text="", lines=(), headers=None):
        self.status_code = status_code
        self.text = text
        self.ok = 200 <= status_code < 300
        self._lines = lines
        self.headers = headers or {}

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def adapter():
    a = build_adapter(load_system("gemini-flash-tts"))
    yield a
    a.close()


def sse(payload: dict) -> str:
    return "data: " + json.dumps(payload)


def audio_event(pcm: bytes, usage=None) -> dict:
    event = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "audio/L16;codec=pcm;rate=24000",
                                "data": base64.b64encode(pcm).decode(),
                            }
                        }
                    ]
                }
            }
        ]
    }
    if usage:
        event["usageMetadata"] = usage
    return event


def test_url_uses_global_host_for_global_location():
    assert _model_url("p", "global", "m", "streamGenerateContent") == (
        "https://aiplatform.googleapis.com/v1/projects/p/locations/global"
        "/publishers/google/models/m:streamGenerateContent"
    )
    assert _model_url("p", "us-central1", "m", "generateContent").startswith(
        "https://us-central1-aiplatform.googleapis.com/"
    )


def test_request_body_carries_audio_modality_and_voice(adapter):
    from dose_r.adapters import SynthesisRequest

    body = adapter._body(SynthesisRequest("dose-000", "Abilify ten milligrams."))
    config = body["generationConfig"]
    assert config["responseModalities"] == ["AUDIO"]
    assert config["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Kore"
    assert body["contents"][0]["parts"][0]["text"] == "Abilify ten milligrams."


def test_sse_stream_concatenates_chunks_and_times_first_audio(adapter):
    pcm = b"\x00\x01" * 12000  # 1s at 24kHz mono s16
    lines = [
        sse(audio_event(pcm[: len(pcm) // 2])),
        "",
        sse(audio_event(pcm[len(pcm) // 2 :], usage={"promptTokenCount": 31, "candidatesTokenCount": 420})),
        "data: [DONE]",
    ]
    import time

    chunks, ttfa_ms, usage, mime = adapter._consume_sse(StubResponse(lines=lines), time.perf_counter())
    assert b"".join(chunks) == pcm
    assert ttfa_ms is not None and ttfa_ms >= 0
    assert usage == {"promptTokenCount": 31, "candidatesTokenCount": 420}
    assert adapter._format_for(mime).sample_rate_hz == 24000


def test_sample_rate_follows_the_mime_type_not_the_config(adapter):
    assert adapter._format_for("audio/L16;codec=pcm;rate=16000").sample_rate_hz == 16000
    assert adapter._format_for("").sample_rate_hz == adapter.config.audio.sample_rate_hz


@pytest.mark.parametrize("status", [429, 500, 503, 504])
def test_retryable_statuses_are_transient(adapter, status):
    with pytest.raises(TransientError):
        adapter._raise_for_status(StubResponse(status, "slow down"))


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_client_errors_are_permanent(adapter, status):
    with pytest.raises(PermanentError):
        adapter._raise_for_status(StubResponse(status, "bad request"))


def test_error_payload_inside_a_stream_is_permanent(adapter):
    with pytest.raises(PermanentError, match="vertex error payload"):
        adapter._collect([{"error": {"code": 400, "message": "nope"}}])


def test_pricing_is_flagged_unverified_until_confirmed():
    for system_id in ("gemini-flash-tts", "gemini-pro-tts"):
        pricing = load_system(system_id).pricing
        assert not pricing.verified
        assert pricing.per_million_output_tokens_usd > 0
