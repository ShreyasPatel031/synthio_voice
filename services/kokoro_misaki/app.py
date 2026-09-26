#!/usr/bin/env python3
"""Stock Kokoro-82M + scored Misaki lexicon — OpenAI-compatible speech API.

POST /v1/audio/speech
  {"input": "text", "voice": "af_heart", "speed": 1.0, "response_format": "wav"}

Weights stay hexgrad/Kokoro-82M. The only model change is loading
data/kokoro_misaki_lexicon.json into pipeline.g2p.lexicon.golds before synth.
"""

from __future__ import annotations

import io
import json
import os
import threading
from pathlib import Path

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
LEXICON_PATH = HERE / "kokoro_misaki_lexicon.json"
LEXICON = json.loads(LEXICON_PATH.read_text())
VOICE_DEFAULT = os.environ.get("KOKORO_VOICE", LEXICON.get("voice", "af_heart"))
SPEED_DEFAULT = float(os.environ.get("KOKORO_SPEED", str(LEXICON.get("speed", 1.0))))
SPEED_LOCK = os.environ.get("KOKORO_SPEED_LOCK", "1") != "0"
LANG_CODE = LEXICON.get("lang_code", "a")
REPO_ID = LEXICON.get("model", "hexgrad/Kokoro-82M")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_pipeline = None
_pipe_lock = threading.Lock()

app = FastAPI(title="Kokoro Misaki lexicon TTS", version="0.2.0")


class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1)
    model: str | None = "kokoro-misaki"
    voice: str | None = None
    speed: float | None = None
    response_format: str | None = "wav"


def apply_lexicon(pipeline) -> int:
    """Load scored golds. Same injection as the 0.7533 Cloud Standard-C run."""
    golds = pipeline.g2p.lexicon.golds
    n = 0
    for entry in LEXICON["entries"]:
        word = entry["word"]
        phones = entry["misaki"]
        golds[word] = phones
        n += 1
        # Misaki looks up the token as it appears in the sentence.
        lowered = word.lower()
        if lowered != word:
            golds[lowered] = phones
    return n


def get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _pipe_lock:
        if _pipeline is not None:
            return _pipeline
        from kokoro import KPipeline

        pipeline = KPipeline(lang_code=LANG_CODE, repo_id=REPO_ID, device=DEVICE)
        apply_lexicon(pipeline)
        _pipeline = pipeline
        return _pipeline


def synthesize(text: str, voice: str, speed: float) -> bytes:
    pipeline = get_pipeline()
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
        "cuda": torch.cuda.is_available(),
        "model": REPO_ID,
        "voice_default": VOICE_DEFAULT,
        "speed_default": SPEED_DEFAULT,
        "lexicon_entries": len(LEXICON["entries"]),
        "lexicon_mean_f1": LEXICON.get("mean_f1"),
        "finetune": False,
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
        return Response(content=wav[44:], media_type="application/octet-stream")
    return Response(content=wav, media_type="audio/wav")


@app.get("/")
def root():
    return {
        "service": "kokoro-misaki",
        "model": REPO_ID,
        "lexicon": len(LEXICON["entries"]),
        "docs": "/docs",
        "speech": "POST /v1/audio/speech",
        "health": "/healthz",
    }
