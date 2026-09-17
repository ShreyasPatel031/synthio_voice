"""Adapter registry: system_id -> live adapter instance."""

from __future__ import annotations

from ..config import ALL_SYSTEMS, VoiceSpec
from .base import SynthesisResult, TTSAdapter
from .google_tts import GoogleTTSAdapter
from .mock import MockTTSAdapter

__all__ = ["SynthesisResult", "TTSAdapter", "GoogleTTSAdapter", "MockTTSAdapter",
           "build_adapter", "available_systems"]


def build_adapter(system_id: str, **kwargs) -> TTSAdapter:
    spec: VoiceSpec | None = ALL_SYSTEMS.get(system_id)
    if spec is None:
        raise KeyError(
            f"unknown system_id {system_id!r}; known: {sorted(ALL_SYSTEMS)}"
        )
    if spec.tier == "mock":
        return MockTTSAdapter(spec, mode=system_id.removeprefix("mock-"), **kwargs)
    return GoogleTTSAdapter(spec, **kwargs)


def available_systems() -> list[str]:
    return sorted(ALL_SYSTEMS)
