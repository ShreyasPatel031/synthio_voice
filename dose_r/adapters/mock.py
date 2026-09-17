"""Deterministic mock backend.

Two uses: offline harness tests with no network or spend, and control baselines.

The defect modes below are deliberately *acoustic* faults (truncation, silence,
clipping), not mispronunciations. A mock cannot produce a genuine mispronunciation,
and pretending otherwise would give Workstream 1 a false confidence signal. These
controls prove the runner and report correctly propagate low scores and failures;
separating a *mispronounced* real utterance from a correct one is the real judge's
job, tested against the human-verified anchor set.
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Any

from ..config import VoiceSpec
from .base import TTSAdapter

_SAMPLE_RATE = 24_000
_MS_PER_CHAR = 0.07

#: Acoustic defect modes the mock can emit.
DEFECT_MODES = ("perfect", "truncated", "silent", "clipped", "overlong")


def _wav(samples: bytes, sample_rate: int = _SAMPLE_RATE) -> bytes:
    """Minimal 16-bit mono RIFF container."""
    return (
        b"RIFF" + struct.pack("<I", 36 + len(samples)) + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data" + struct.pack("<I", len(samples)) + samples
    )


class MockTTSAdapter(TTSAdapter):
    """Emits a deterministic waveform exhibiting the configured defect mode."""

    def __init__(self, spec: VoiceSpec, *, mode: str = "perfect",
                 fail_rate: float = 0.0):
        super().__init__(spec)
        if mode not in DEFECT_MODES:
            raise ValueError(f"unknown mode {mode!r}; known: {DEFECT_MODES}")
        self.mode = mode
        self.fail_rate = fail_rate

    def _synthesize(self, text: str) -> tuple[bytes, dict[str, Any]]:
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        if self.fail_rate and (seed % 1000) / 1000.0 < self.fail_rate:
            raise RuntimeError("simulated transient backend failure")

        nominal = max(int(_SAMPLE_RATE * len(text) * _MS_PER_CHAR), _SAMPLE_RATE // 10)
        n = {
            "truncated": max(nominal // 6, 512),
            "overlong": nominal * 4,
        }.get(self.mode, nominal)

        amplitude = {"silent": 0.0005, "clipped": 1.0}.get(self.mode, 0.35)
        freq = 220.0 + (seed % 60)

        frames = bytearray()
        for i in range(n):
            v = math.sin(2 * math.pi * freq * i / _SAMPLE_RATE) * amplitude
            frames += struct.pack("<h", max(-32768, min(32767, int(v * 32767))))

        return _wav(bytes(frames)), {
            "audio_format": "wav", "sample_rate_hz": _SAMPLE_RATE,
            "streaming": False, "ttfa_ms": None,
            "extra": {"mock_mode": self.mode},
        }
