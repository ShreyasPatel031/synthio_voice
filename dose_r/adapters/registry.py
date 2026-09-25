"""Backend name -> class. A backend is only imported when it is built, so a
missing optional dependency for one vendor cannot break the others."""

from __future__ import annotations

from typing import Callable

from .base import TTSAdapter
from .config import SystemConfig

_BACKENDS: dict[str, Callable[[], type[TTSAdapter]]] = {}


def register(name: str, loader: Callable[[], type[TTSAdapter]]) -> None:
    _BACKENDS[name] = loader


def _load_mock() -> type[TTSAdapter]:
    from .mock import MockTTSAdapter

    return MockTTSAdapter


def _load_gemini() -> type[TTSAdapter]:
    from .gemini import GeminiTTSAdapter

    return GeminiTTSAdapter


def _load_openai_compatible() -> type[TTSAdapter]:
    from .openai_compatible import OpenAICompatibleHTTPAdapter

    return OpenAICompatibleHTTPAdapter


register("mock", _load_mock)
register("gemini_vertex", _load_gemini)
register("openai_compatible_http", _load_openai_compatible)


def backends() -> list[str]:
    return sorted(_BACKENDS)


def build_adapter(config: SystemConfig) -> TTSAdapter:
    if config.backend not in _BACKENDS:
        raise KeyError(
            f"system {config.system_id!r} needs backend {config.backend!r}, "
            f"which is not implemented; have {backends()}"
        )
    return _BACKENDS[config.backend]()(config)
