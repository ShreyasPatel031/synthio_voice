"""Deterministic offline backend. The default system, and the one the harness
is validated against: same text plus same system_id always yields byte-identical
audio, so a manifest diff across two runs isolates harness changes from model
changes.

It also fakes the failure modes worth exercising -- transient errors that must
be retried, permanent errors that must not be -- because those paths otherwise
only run against a live API.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np

from .audio import pcm_duration_s, wrap_pcm_as_wav
from .base import (
    PermanentError,
    RawAudio,
    SynthesisRequest,
    TTSAdapter,
    TransientError,
    Usage,
    sha256_hex,
)


class MockTTSAdapter(TTSAdapter):
    backend = "mock"
    supports_streaming = True

    def __init__(self, config):
        super().__init__(config)
        o = config.options
        self.chars_per_second = float(o.get("chars_per_second", 14.5))
        self.base_latency_s = float(o.get("base_latency_s", 0.0))
        self.per_char_latency_s = float(o.get("per_char_latency_s", 0.0))
        self.ttfa_fraction = float(o.get("ttfa_fraction", 0.35))
        self.seed = int(o.get("seed", 20260917))
        self.transient_failures_per_item = int(o.get("transient_failures_per_item", 0))
        self.permanent_failure_ids = set(o.get("permanent_failure_ids", []))
        self._attempts: dict[str, int] = {}
        self._lock = threading.Lock()

    def _item_seed(self, request: SynthesisRequest) -> int:
        digest = sha256_hex(f"{self.seed}|{self.config.system_id}|{request.item_id}|{request.text}")
        return int(digest[:16], 16)

    def _synthesize(self, request: SynthesisRequest) -> RawAudio:
        with self._lock:
            n = self._attempts.get(request.item_id, 0) + 1
            self._attempts[request.item_id] = n

        if request.item_id in self.permanent_failure_ids:
            raise PermanentError(f"mock: item {request.item_id} configured to fail permanently")

        latency_s = self.base_latency_s + self.per_char_latency_s * len(request.text)
        time.sleep(latency_s * self.ttfa_fraction)
        ttfa_ms = latency_s * self.ttfa_fraction * 1000

        if n <= self.transient_failures_per_item:
            raise TransientError(f"mock: simulated transient failure (attempt {n})")

        time.sleep(latency_s * (1.0 - self.ttfa_fraction))

        fmt = self.config.audio
        pcm = self._render(request, fmt.sample_rate_hz, fmt.channels)
        duration = pcm_duration_s(pcm, fmt)
        data = wrap_pcm_as_wav(pcm, fmt) if fmt.container == "wav" else pcm

        return RawAudio(
            data=data,
            audio_format=fmt,
            usage=Usage(
                characters=len(request.text),
                audio_seconds=duration,
                input_tokens=math.ceil(len(request.text) / 4),
                output_tokens=round(duration * 25),
            ),
            ttfa_ms=ttfa_ms,
            audio_duration_s=duration,
            provider_request_id=f"mock-{self._item_seed(request):016x}",
            provider_meta={"attempt_in_backend": n},
        )

    def _render(self, request: SynthesisRequest, sample_rate: int, channels: int) -> bytes:
        rng = np.random.default_rng(self._item_seed(request))
        duration = max(0.25, len(request.text) / self.chars_per_second)
        t = np.arange(int(duration * sample_rate)) / sample_rate

        fundamental = 95.0 + rng.random() * 90.0
        wave = sum(
            (0.6 ** k) * np.sin(2 * np.pi * fundamental * (k + 1) * t + rng.random() * 6.28)
            for k in range(4)
        )
        syllable = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)
        signal = wave * syllable
        signal /= np.abs(signal).max()

        pcm = (signal * 0.6 * 32767).astype("<i2")
        if channels > 1:
            pcm = np.repeat(pcm[:, None], channels, axis=1).ravel()
        return pcm.tobytes()
