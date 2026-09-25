"""The adapter contract every TTS system in DOSE-R implements.

A backend supplies one thing: bytes of audio for a string of text. Everything
the benchmark actually reasons about -- wall-clock latency, time-to-first-audio,
per-call cost, retries -- is measured and recorded here, identically for every
system, so that numbers from two vendors are comparable by construction rather
than by convention.
"""

from __future__ import annotations

import hashlib
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, ClassVar

SCHEMA_VERSION = "dose-r/synthesis-record/1"


class TransientError(RuntimeError):
    """Rate limit, timeout, 5xx -- worth retrying."""


class PermanentError(RuntimeError):
    """Bad request, auth failure, unsupported voice -- retrying cannot help."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class AudioFormat:
    container: str
    encoding: str
    sample_rate_hz: int
    channels: int = 1
    sample_width_bytes: int = 2

    @property
    def extension(self) -> str:
        return self.container

    @property
    def is_linear_pcm(self) -> bool:
        return self.encoding.startswith("pcm_")

    def to_dict(self) -> dict[str, Any]:
        return {
            "container": self.container,
            "encoding": self.encoding,
            "sample_rate_hz": self.sample_rate_hz,
            "channels": self.channels,
        }


@dataclass(frozen=True)
class Usage:
    """Billable quantities, kept separate from the money they cost.

    Storing usage alongside cost means a price change (or a correction to our
    price table) is a recomputation over the manifest, not a re-run of the set.
    """

    characters: int = 0
    audio_seconds: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "characters": self.characters,
            "audio_seconds": round(self.audio_seconds, 4),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass(frozen=True)
class Pricing:
    """A dated price list. `as_of` and `source` exist so a stale number is
    visible in the manifest rather than silently propagated into a report."""

    id: str
    as_of: str
    source: str = ""
    per_request_usd: float = 0.0
    per_million_characters_usd: float = 0.0
    per_second_audio_usd: float = 0.0
    per_million_input_tokens_usd: float = 0.0
    per_million_output_tokens_usd: float = 0.0
    verified: bool = False

    def breakdown(self, usage: Usage) -> dict[str, float]:
        parts = {
            "request": self.per_request_usd,
            "characters": usage.characters * self.per_million_characters_usd / 1e6,
            "audio_seconds": usage.audio_seconds * self.per_second_audio_usd,
            "input_tokens": (usage.input_tokens or 0)
            * self.per_million_input_tokens_usd
            / 1e6,
            "output_tokens": (usage.output_tokens or 0)
            * self.per_million_output_tokens_usd
            / 1e6,
        }
        return {k: v for k, v in parts.items() if v}

    def cost_usd(self, usage: Usage) -> float:
        return sum(self.breakdown(usage).values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "as_of": self.as_of,
            "source": self.source,
            "verified": self.verified,
            "per_request_usd": self.per_request_usd,
            "per_million_characters_usd": self.per_million_characters_usd,
            "per_second_audio_usd": self.per_second_audio_usd,
            "per_million_input_tokens_usd": self.per_million_input_tokens_usd,
            "per_million_output_tokens_usd": self.per_million_output_tokens_usd,
        }


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    initial_backoff_s: float = 0.5
    max_backoff_s: float = 30.0
    multiplier: float = 2.0
    jitter: float = 0.25

    def backoff_s(self, attempt: int, rng: random.Random) -> float:
        """Delay before attempt number `attempt + 1` (attempts are 1-based)."""
        raw = self.initial_backoff_s * self.multiplier ** (attempt - 1)
        capped = min(raw, self.max_backoff_s)
        return capped * (1.0 + rng.uniform(-self.jitter, self.jitter))

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_attempts": self.max_attempts,
            "initial_backoff_s": self.initial_backoff_s,
            "max_backoff_s": self.max_backoff_s,
            "multiplier": self.multiplier,
            "jitter": self.jitter,
        }


@dataclass
class SynthesisResult:
    """Audio plus call metadata. Present so the WS2 scorers can import."""

    system_id: str
    item_id: str
    ok: bool
    audio: bytes | None = None
    audio_format: str = "wav"
    sample_rate_hz: int | None = None
    ttfa_ms: float | None = None
    total_ms: float | None = None
    streaming: bool = False
    billable_chars: int = 0
    cost_usd: float | None = None
    cost_estimated: bool = True
    price_verified: bool = False
    attempts: int = 1
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesisRequest:
    """One DOSE row. `item` carries the judge-relevant dataset fields so a
    manifest record is self-contained for scoring."""

    item_id: str
    text: str
    item: dict[str, Any] = field(default_factory=dict)


@dataclass
class RawAudio:
    """What a backend returns. Timing of the whole call is the base class's job;
    only TTFA is the backend's, because only it knows when bytes started."""

    data: bytes
    audio_format: AudioFormat
    usage: Usage
    ttfa_ms: float | None = None
    audio_duration_s: float | None = None
    provider_request_id: str | None = None
    provider_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Attempt:
    n: int
    latency_ms: float
    error: str | None = None
    error_class: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
            "error_class": self.error_class,
        }


@dataclass
class SynthesisRecord:
    """One line of the manifest: the unit the judge, the fidelity report and the
    cost/latency blueprint all read.

    Latency is recorded three ways on purpose:
      latency_ms    -- the successful call alone; the product number
      ttfa_ms       -- first audio byte; a different product decision entirely,
                       and null (not zero) when the API cannot express it
      wall_clock_ms -- including failed attempts and backoff; what a batch costs
    """

    run_id: str
    item_id: str
    system_id: str
    backend: str
    model: str | None
    voice: str | None
    text: str
    text_sha256: str
    text_chars: int
    status: str
    started_at: str
    finished_at: str
    latency_ms: float | None
    ttfa_ms: float | None
    ttfa_source: str
    wall_clock_ms: float
    attempts: list[Attempt]
    usage: Usage
    pricing_id: str
    pricing_as_of: str
    cost_usd: float
    cost_breakdown: dict[str, float]
    item: dict[str, Any] = field(default_factory=dict)
    audio_format: AudioFormat | None = None
    audio_bytes: int | None = None
    audio_sha256: str | None = None
    audio_duration_s: float | None = None
    audio_path: str | None = None
    provider_request_id: str | None = None
    provider_meta: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    error_class: str | None = None
    audio: bytes | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "item_id": self.item_id,
            "system_id": self.system_id,
            "backend": self.backend,
            "model": self.model,
            "voice": self.voice,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "text": self.text,
            "text_sha256": self.text_sha256,
            "text_chars": self.text_chars,
            "item": self.item,
            "audio_path": self.audio_path,
            "audio_sha256": self.audio_sha256,
            "audio_bytes": self.audio_bytes,
            "audio_duration_s": self.audio_duration_s,
            "audio_format": self.audio_format.to_dict() if self.audio_format else None,
            "latency_ms": round(self.latency_ms, 2) if self.latency_ms is not None else None,
            "ttfa_ms": round(self.ttfa_ms, 2) if self.ttfa_ms is not None else None,
            "ttfa_source": self.ttfa_source,
            "wall_clock_ms": round(self.wall_clock_ms, 2),
            "attempt_count": len(self.attempts),
            "attempts": [a.to_dict() for a in self.attempts],
            "usage": self.usage.to_dict(),
            "pricing_id": self.pricing_id,
            "pricing_as_of": self.pricing_as_of,
            "cost_usd": self.cost_usd,
            "cost_breakdown": self.cost_breakdown,
            "provider_request_id": self.provider_request_id,
            "provider_meta": self.provider_meta,
            "error": self.error,
            "error_class": self.error_class,
            "score": None,
            "judge": None,
        }


class TTSAdapter(ABC):
    """Subclass, set `backend`, implement `_synthesize`. Nothing else.

    Timing, retry, cost and record construction are final here so no backend can
    measure them differently.
    """

    backend: ClassVar[str]
    supports_streaming: ClassVar[bool] = False

    def __init__(self, config: "SystemConfig"):  # noqa: F821 (see config.py)
        self.config = config
        self._rng = random.Random(sha256_hex(config.system_id)[:16])

    @abstractmethod
    def _synthesize(self, request: SynthesisRequest) -> RawAudio:
        """Perform one attempt. Raise TransientError / PermanentError."""

    def preflight(self) -> None:
        """Fail fast on missing credentials or config before a 274-item run."""

    def close(self) -> None:
        pass

    def __enter__(self) -> "TTSAdapter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def synthesize(self, request: SynthesisRequest, run_id: str = "adhoc") -> SynthesisRecord:
        cfg = self.config
        policy = cfg.retry
        attempts: list[Attempt] = []
        started_at = _utc_now()
        wall_start = time.perf_counter()
        raw: RawAudio | None = None
        last_error: Exception | None = None
        success_latency_ms: float | None = None

        for n in range(1, policy.max_attempts + 1):
            attempt_start = time.perf_counter()
            try:
                raw = self._synthesize(request)
            except PermanentError as exc:
                latency = (time.perf_counter() - attempt_start) * 1000
                attempts.append(Attempt(n, latency, str(exc), "PermanentError"))
                last_error = exc
                break
            except Exception as exc:  # TransientError and anything unclassified
                latency = (time.perf_counter() - attempt_start) * 1000
                attempts.append(Attempt(n, latency, str(exc), type(exc).__name__))
                last_error = exc
                if n < policy.max_attempts:
                    time.sleep(policy.backoff_s(n, self._rng))
                continue
            success_latency_ms = (time.perf_counter() - attempt_start) * 1000
            attempts.append(Attempt(n, success_latency_ms))
            last_error = None
            break

        wall_clock_ms = (time.perf_counter() - wall_start) * 1000
        finished_at = _utc_now()

        if raw is None or last_error is not None:
            usage = Usage(characters=len(request.text))
            return SynthesisRecord(
                run_id=run_id,
                item_id=request.item_id,
                system_id=cfg.system_id,
                backend=self.backend,
                model=cfg.model,
                voice=cfg.voice,
                text=request.text,
                text_sha256=sha256_hex(request.text),
                text_chars=len(request.text),
                status="error",
                started_at=started_at,
                finished_at=finished_at,
                latency_ms=None,
                ttfa_ms=None,
                ttfa_source=self._ttfa_source(None),
                wall_clock_ms=wall_clock_ms,
                attempts=attempts,
                usage=usage,
                pricing_id=cfg.pricing.id,
                pricing_as_of=cfg.pricing.as_of,
                cost_usd=0.0,
                cost_breakdown={},
                item=request.item,
                error=str(last_error),
                error_class=type(last_error).__name__ if last_error else "UnknownError",
            )

        usage = raw.usage
        if not usage.characters:
            usage = Usage(
                characters=len(request.text),
                audio_seconds=usage.audio_seconds,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            )
        breakdown = {k: round(v, 10) for k, v in cfg.pricing.breakdown(usage).items()}

        return SynthesisRecord(
            run_id=run_id,
            item_id=request.item_id,
            system_id=cfg.system_id,
            backend=self.backend,
            model=cfg.model,
            voice=cfg.voice,
            text=request.text,
            text_sha256=sha256_hex(request.text),
            text_chars=len(request.text),
            status="ok",
            started_at=started_at,
            finished_at=finished_at,
            latency_ms=success_latency_ms,
            ttfa_ms=raw.ttfa_ms,
            ttfa_source=self._ttfa_source(raw.ttfa_ms),
            wall_clock_ms=wall_clock_ms,
            attempts=attempts,
            usage=usage,
            pricing_id=cfg.pricing.id,
            pricing_as_of=cfg.pricing.as_of,
            cost_usd=round(sum(breakdown.values()), 10),
            cost_breakdown=breakdown,
            item=request.item,
            audio_format=raw.audio_format,
            audio_bytes=len(raw.data),
            audio_sha256=sha256_hex(raw.data),
            audio_duration_s=raw.audio_duration_s,
            provider_request_id=raw.provider_request_id,
            provider_meta=raw.provider_meta,
            audio=raw.data,
        )

    def _ttfa_source(self, ttfa_ms: float | None) -> str:
        if not self.supports_streaming:
            return "unsupported"
        return "stream_first_chunk" if ttfa_ms is not None else "stream_unavailable"
