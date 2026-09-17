"""DOSE-R TTS adapters: one contract, one record schema, one runner."""

from .base import (
    AudioFormat,
    PermanentError,
    Pricing,
    RawAudio,
    RetryPolicy,
    SCHEMA_VERSION,
    SynthesisRecord,
    SynthesisRequest,
    TTSAdapter,
    TransientError,
    Usage,
)
from .config import SystemConfig, load_system, load_systems
from .registry import backends, build_adapter, register

__all__ = [
    "AudioFormat",
    "PermanentError",
    "Pricing",
    "RawAudio",
    "RetryPolicy",
    "SCHEMA_VERSION",
    "SynthesisRecord",
    "SynthesisRequest",
    "SystemConfig",
    "TTSAdapter",
    "TransientError",
    "Usage",
    "backends",
    "build_adapter",
    "load_system",
    "load_systems",
    "register",
]
