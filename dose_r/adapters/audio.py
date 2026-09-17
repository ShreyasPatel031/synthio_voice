"""PCM helpers. Most TTS APIs hand back headerless linear PCM; the judge and
every audio tool want a container."""

from __future__ import annotations

import io
import wave

from .base import AudioFormat


def pcm_duration_s(pcm: bytes, fmt: AudioFormat) -> float:
    frame_bytes = fmt.sample_width_bytes * fmt.channels
    return len(pcm) / (frame_bytes * fmt.sample_rate_hz)


def wrap_pcm_as_wav(pcm: bytes, fmt: AudioFormat) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(fmt.channels)
        w.setsampwidth(fmt.sample_width_bytes)
        w.setframerate(fmt.sample_rate_hz)
        w.writeframes(pcm)
    return buf.getvalue()


def wav_duration_s(data: bytes) -> float:
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.getnframes() / w.getframerate()
