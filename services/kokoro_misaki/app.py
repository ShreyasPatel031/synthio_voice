#!/usr/bin/env python3
"""Kokoro-82M + Misaki pins — OpenAI-compatible speech API for Synthio.

POST /v1/audio/speech
  {"input": "text", "voice": "af_heart", "speed": 1.0, "response_format": "wav"}

Supports Misaki injects: [Name](/phonemes/).
When input has no inject, known drug names from pins.json are auto-injected.
"""

from __future__ import annotations

import io
import json
import os
import re
import threading
from pathlib import Path

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / "pins.json").read_text())
ALIASES = json.loads((HERE / "aliases.json").read_text())
VOICE_DEFAULT = os.environ.get("KOKORO_VOICE", "af_heart")
SPEED_DEFAULT = float(os.environ.get("KOKORO_SPEED", "1.0"))
# Speed lock: same rule as local search. Override only for deliberate experiments.
SPEED_LOCK = os.environ.get("KOKORO_SPEED_LOCK", "1") != "0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_INJECT_RE = re.compile(r"\[[^\]]+\]\(/[^/]+/\)")
_pipeline = None
_pipe_lock = threading.Lock()

app = FastAPI(title="Kokoro Misaki TTS", version="0.1.0")


class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1)
    model: str | None = "kokoro-misaki"
    voice: str | None = None
    speed: float | None = None
    response_format: str | None = "wav"


def get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _pipe_lock:
        if _pipeline is not None:
            return _pipeline
        from kokoro import KPipeline

        _pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=DEVICE)
        return _pipeline


def apply_pins(text: str) -> str:
    """Insert Misaki overrides for known drugs when the caller sent plain text."""
    if _INJECT_RE.search(text):
        return text
    # longest alias first so "lebrikizumab-lbkz" wins over shorter fragments
    for alias in sorted(ALIASES.keys(), key=len, reverse=True):
        slug = ALIASES[alias]
        pin = PINS.get(slug)
        if not pin:
            continue
        misaki = pin["misaki"]
        word = pin.get("word") or alias

        def repl(m: re.Match[str], w=word, phones=misaki) -> str:
            return f"[{m.group(0)}](/{phones}/)"

        # word-boundary-ish: don't break mid-token
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])")
        text, n = pattern.subn(repl, text, count=1)
        if n:
            # one drug inject per pass is enough for DoSE sentences; continue for multi-drug
            pass
    return text


def synthesize(text: str, voice: str, speed: float) -> bytes:
    pipeline = get_pipeline()
    text = apply_pins(text)
    result = next(pipeline(text, voice=voice, speed=speed))
    if result.audio is None:
        raise RuntimeError("Kokoro returned no audio")
    buf = io.BytesIO()
    sf.write(buf, result.audio.detach().cpu().numpy(), 24000, format="WAV")
    return buf.getvalue()


@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "device": DEVICE,
        "voice_default": VOICE_DEFAULT,
        "speed_default": SPEED_DEFAULT,
        "pins": len(PINS),
        "cuda": torch.cuda.is_available(),
    }


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [{"id": "kokoro-misaki", "object": "model", "owned_by": "synthio_voice"}],
    }


@app.post("/v1/audio/speech")
def speech(body: SpeechRequest, request: Request):
    voice = body.voice or VOICE_DEFAULT
    speed = SPEED_DEFAULT if body.speed is None else float(body.speed)
    if SPEED_LOCK and abs(speed - 1.0) > 1e-6:
        raise HTTPException(status_code=400, detail="speed is locked at 1.0")
    fmt = (body.response_format or "wav").lower()
    if fmt not in {"wav", "pcm"}:
        raise HTTPException(status_code=400, detail="response_format must be wav or pcm")
    try:
        wav = synthesize(body.input, voice=voice, speed=speed)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if fmt == "pcm":
        # skip 44-byte WAV header
        return Response(content=wav[44:], media_type="application/octet-stream")
    return Response(content=wav, media_type="audio/wav")


@app.get("/")
def root():
    return {
        "service": "kokoro-misaki",
        "docs": "/docs",
        "speech": "POST /v1/audio/speech",
        "health": "/healthz",
    }
