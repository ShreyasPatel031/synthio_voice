"""Scoring layer. Workstream 1's hybrid judge registers here."""

from __future__ import annotations

from .base import PASS_THRESHOLD, ScoreResult, Scorer
from .standin import StandInScorer
from .asr_roundtrip import AsrRoundTripScorer

_REGISTRY: dict[str, type[Scorer]] = {
    "standin": StandInScorer,
    "asr-roundtrip": AsrRoundTripScorer,
}


def register_scorer(name: str, cls: type[Scorer]) -> None:
    """Workstream 1 calls this to plug the real judge in."""
    _REGISTRY[name] = cls


def build_scorer(name: str, **kwargs) -> Scorer:
    if name not in _REGISTRY:
        raise KeyError(f"unknown scorer {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available_scorers() -> list[str]:
    return sorted(_REGISTRY)


__all__ = ["PASS_THRESHOLD", "ScoreResult", "Scorer", "StandInScorer",
           "AsrRoundTripScorer", "register_scorer", "build_scorer",
           "available_scorers"]
