"""System configs. Adding a TTS system is an entry in systems.yaml; adding a
new *protocol* is a backend class. Those are deliberately different acts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .base import AudioFormat, Pricing, RetryPolicy

DEFAULT_SYSTEMS_PATH = Path(__file__).with_name("systems.yaml")


@dataclass(frozen=True)
class SystemConfig:
    system_id: str
    backend: str
    tier: str = "unclassified"
    status: str = "ready"
    model: str | None = None
    voice: str | None = None
    audio: AudioFormat = field(
        default_factory=lambda: AudioFormat("wav", "pcm_s16le", 24000, 1)
    )
    pricing: Pricing = field(
        default_factory=lambda: Pricing(id="unpriced", as_of="unknown")
    )
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    concurrency: int = 4
    timeout_s: float = 120.0
    notes: str = ""
    options: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_id": self.system_id,
            "backend": self.backend,
            "tier": self.tier,
            "status": self.status,
            "model": self.model,
            "voice": self.voice,
            "audio": self.audio.to_dict(),
            "pricing": self.pricing.to_dict(),
            "retry": self.retry.to_dict(),
            "concurrency": self.concurrency,
            "timeout_s": self.timeout_s,
            "notes": self.notes,
            "options": self.options,
        }


def _audio_from(raw: dict[str, Any] | None) -> AudioFormat:
    raw = raw or {}
    return AudioFormat(
        container=raw.get("container", "wav"),
        encoding=raw.get("encoding", "pcm_s16le"),
        sample_rate_hz=int(raw.get("sample_rate_hz", 24000)),
        channels=int(raw.get("channels", 1)),
        sample_width_bytes=int(raw.get("sample_width_bytes", 2)),
    )


def _pricing_from(raw: dict[str, Any] | None, system_id: str) -> Pricing:
    raw = dict(raw or {})
    return Pricing(
        id=raw.get("id", f"{system_id}/unpriced"),
        as_of=str(raw.get("as_of", "unknown")),
        source=raw.get("source", ""),
        per_request_usd=float(raw.get("per_request_usd", 0.0)),
        per_million_characters_usd=float(raw.get("per_million_characters_usd", 0.0)),
        per_second_audio_usd=float(raw.get("per_second_audio_usd", 0.0)),
        per_million_input_tokens_usd=float(raw.get("per_million_input_tokens_usd", 0.0)),
        per_million_output_tokens_usd=float(
            raw.get("per_million_output_tokens_usd", 0.0)
        ),
        verified=bool(raw.get("verified", False)),
    )


def _retry_from(raw: dict[str, Any] | None) -> RetryPolicy:
    raw = dict(raw or {})
    return RetryPolicy(
        max_attempts=int(raw.get("max_attempts", 3)),
        initial_backoff_s=float(raw.get("initial_backoff_s", 0.5)),
        max_backoff_s=float(raw.get("max_backoff_s", 30.0)),
        multiplier=float(raw.get("multiplier", 2.0)),
        jitter=float(raw.get("jitter", 0.25)),
    )


def system_from_dict(system_id: str, raw: dict[str, Any]) -> SystemConfig:
    return SystemConfig(
        system_id=system_id,
        backend=raw["backend"],
        tier=raw.get("tier", "unclassified"),
        status=raw.get("status", "ready"),
        model=raw.get("model"),
        voice=raw.get("voice"),
        audio=_audio_from(raw.get("audio")),
        pricing=_pricing_from(raw.get("pricing"), system_id),
        retry=_retry_from(raw.get("retry")),
        concurrency=int(raw.get("concurrency", 4)),
        timeout_s=float(raw.get("timeout_s", 120.0)),
        notes=raw.get("notes", ""),
        options=dict(raw.get("options") or {}),
    )


def load_systems(path: str | Path = DEFAULT_SYSTEMS_PATH) -> dict[str, SystemConfig]:
    doc = yaml.safe_load(Path(path).read_text()) or {}
    defaults = doc.get("defaults") or {}
    systems = {}
    for system_id, raw in (doc.get("systems") or {}).items():
        merged = {**defaults, **(raw or {})}
        for key in ("audio", "pricing", "retry"):
            if key in defaults or key in (raw or {}):
                merged[key] = {**(defaults.get(key) or {}), **((raw or {}).get(key) or {})}
        systems[system_id] = system_from_dict(system_id, merged)
    return systems


def load_system(
    system_id: str, path: str | Path = DEFAULT_SYSTEMS_PATH
) -> SystemConfig:
    systems = load_systems(path)
    if system_id not in systems:
        raise KeyError(f"unknown system {system_id!r}; have {sorted(systems)}")
    return systems[system_id]
