"""Offline tests for openai_compatible_http — no live network."""

from __future__ import annotations

import io
import wave
from unittest.mock import MagicMock, patch

import pytest

from dose_r.adapters import PermanentError, TransientError, build_adapter
from dose_r.adapters.base import SynthesisRequest
from dose_r.adapters.config import system_from_dict


def _wav_bytes(frames: int = 2400, rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * frames)
    return buf.getvalue()


def _cfg(**opts):
    return system_from_dict(
        "oss-selfhosted",
        {
            "backend": "openai_compatible_http",
            "tier": "open_source",
            "model": "kokoro-misaki",
            "voice": "af_heart",
            "options": {"base_url": "https://example.test", "speed": 1.0, **opts},
            "pricing": {"id": "t", "as_of": "2026-09-21", "verified": False},
        },
    )


def test_registry_loads_openai_compatible():
    a = build_adapter(_cfg())
    assert a.backend == "openai_compatible_http"


def test_body_shape():
    a = build_adapter(_cfg())
    body = a._body(SynthesisRequest(item_id="x", text="Xolair today."))
    assert body["input"] == "Xolair today."
    assert body["voice"] == "af_heart"
    assert body["speed"] == 1.0
    assert body["response_format"] == "wav"


def test_synthesize_ok():
    a = build_adapter(_cfg())
    wav = _wav_bytes()
    resp = MagicMock()
    resp.ok = True
    resp.status_code = 200
    resp.content = wav
    resp.text = ""
    with patch.object(a._session, "post", return_value=resp) as post:
        raw = a._synthesize(SynthesisRequest(item_id="x", text="hi"))
    assert raw.data == wav
    assert raw.audio_duration_s == pytest.approx(0.1, abs=1e-3)
    post.assert_called_once()
    assert post.call_args.args[0].endswith("/v1/audio/speech")


def test_transient_on_503():
    a = build_adapter(_cfg())
    resp = MagicMock()
    resp.ok = False
    resp.status_code = 503
    resp.text = "busy"
    with patch.object(a._session, "post", return_value=resp):
        with pytest.raises(TransientError):
            a._synthesize(SynthesisRequest(item_id="x", text="hi"))


def test_permanent_on_400():
    a = build_adapter(_cfg())
    resp = MagicMock()
    resp.ok = False
    resp.status_code = 400
    resp.text = "bad"
    with patch.object(a._session, "post", return_value=resp):
        with pytest.raises(PermanentError):
            a._synthesize(SynthesisRequest(item_id="x", text="hi"))


def test_missing_base_url():
    a = build_adapter(_cfg())
    a.base_url = ""
    with pytest.raises(PermanentError, match="base_url"):
        a._synthesize(SynthesisRequest(item_id="x", text="hi"))
