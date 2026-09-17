"""The adapter contract every TTS system is reached through.

Adding a system must be a config entry, not new code in the runner. Everything the
runner needs -- audio, timing, cost, failure -- comes back in SynthesisResult.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..config import VoiceSpec


@dataclass
class SynthesisResult:
    """One synthesis call: the audio plus everything the tradeoff blueprint needs."""

    system_id: str
    item_id: str
    ok: bool
    audio: bytes | None = None
    audio_format: str = "wav"
    sample_rate_hz: int | None = None

    # Timing. `ttfa_ms` is time-to-first-audio and is only meaningful for streaming
    # backends; it stays None for request/response REST calls rather than being
    # silently aliased to total_ms, which would make a non-streaming system look
    # like it had great first-byte latency.
    ttfa_ms: float | None = None
    total_ms: float | None = None
    streaming: bool = False

    # Cost. `cost_estimated` is True whenever the figure comes from a list price
    # rather than a billing export.
    billable_chars: int = 0
    cost_usd: float | None = None
    cost_estimated: bool = True
    price_verified: bool = False

    attempts: int = 1
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """JSON-safe row for the run log. Audio is referenced by path, not inlined."""
        return {
            "system_id": self.system_id,
            "item_id": self.item_id,
            "ok": self.ok,
            "audio_format": self.audio_format,
            "sample_rate_hz": self.sample_rate_hz,
            "audio_bytes": len(self.audio) if self.audio else 0,
            "ttfa_ms": self.ttfa_ms,
            "total_ms": self.total_ms,
            "streaming": self.streaming,
            "billable_chars": self.billable_chars,
            "cost_usd": self.cost_usd,
            "cost_estimated": self.cost_estimated,
            "price_verified": self.price_verified,
            "attempts": self.attempts,
            "error": self.error,
            "metadata": self.metadata,
        }


class TTSAdapter(ABC):
    """Base class for a TTS system under test."""

    def __init__(self, spec: VoiceSpec):
        self.spec = spec

    @property
    def system_id(self) -> str:
        return self.spec.system_id

    def estimate_cost(self, n_chars: int) -> tuple[float, bool]:
        p = self.spec.pricing
        return n_chars * p.usd_per_million_chars / 1_000_000, p.verified

    @abstractmethod
    def _synthesize(self, text: str) -> tuple[bytes, dict[str, Any]]:
        """Perform one call. Raise on failure; the retry wrapper handles it."""

    def synthesize(self, text: str, item_id: str, *, max_attempts: int = 3,
                   backoff_s: float = 2.0) -> SynthesisResult:
        """Call `_synthesize` with retries, capturing timing and cost."""
        cost, verified = self.estimate_cost(len(text))
        last_error: str | None = None

        for attempt in range(1, max_attempts + 1):
            t0 = time.perf_counter()
            try:
                audio, meta = self._synthesize(text)
            except Exception as exc:  # adapter-specific failures are all reported alike
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < max_attempts:
                    time.sleep(backoff_s * (2 ** (attempt - 1)))
                continue

            return SynthesisResult(
                system_id=self.system_id, item_id=item_id, ok=True, audio=audio,
                audio_format=meta.get("audio_format", "wav"),
                sample_rate_hz=meta.get("sample_rate_hz"),
                ttfa_ms=meta.get("ttfa_ms"),
                total_ms=(time.perf_counter() - t0) * 1000,
                streaming=meta.get("streaming", False),
                billable_chars=len(text), cost_usd=cost,
                cost_estimated=True, price_verified=verified,
                attempts=attempt, metadata=meta.get("extra", {}),
            )

        return SynthesisResult(
            system_id=self.system_id, item_id=item_id, ok=False,
            billable_chars=len(text), cost_usd=0.0, price_verified=verified,
            attempts=max_attempts, error=last_error,
        )
