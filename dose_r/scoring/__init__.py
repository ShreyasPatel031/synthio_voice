"""Scoring layer. Workstream 1's hybrid judge registers here.

Scorer classes are imported lazily. `forced_align` loads this package only to
reach `phoneme_model`, and several scorers pull GCP / parquet dependencies
that a pronunciation eval does not need at import time.
"""

from __future__ import annotations

from .base import PASS_THRESHOLD, ScoreResult, Scorer

_REGISTRY: dict[str, type[Scorer]] | None = None

_SCORER_MODULES = {
    "standin": (".standin", "StandInScorer"),
    "asr-roundtrip": (".asr_roundtrip", "AsrRoundTripScorer"),
    "llm-panel": (".llm_panel", "AudioLLMPanelScorer"),
    "speech-similarity": (".speech_similarity", "SpeechSimilarityScorer"),
    "phoneme-distance": (".phoneme_distance", "PhonemeDistanceScorer"),
}


def _registry() -> dict[str, type[Scorer]]:
    global _REGISTRY
    if _REGISTRY is None:
        import importlib
        loaded: dict[str, type[Scorer]] = {}
        for name, (module, cls_name) in _SCORER_MODULES.items():
            mod = importlib.import_module(module, __package__)
            loaded[name] = getattr(mod, cls_name)
        _REGISTRY = loaded
    return _REGISTRY


def register_scorer(name: str, cls: type[Scorer]) -> None:
    """Workstream 1 calls this to plug the real judge in."""
    _registry()[name] = cls


def build_scorer(name: str, **kwargs) -> Scorer:
    registry = _registry()
    if name not in registry:
        raise KeyError(f"unknown scorer {name!r}; known: {sorted(registry)}")
    return registry[name](**kwargs)


def available_scorers() -> list[str]:
    return sorted(_registry())


__all__ = [
    "PASS_THRESHOLD", "ScoreResult", "Scorer",
    "register_scorer", "build_scorer", "available_scorers",
]
