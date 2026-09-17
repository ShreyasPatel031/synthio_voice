"""Central configuration: paths, pricing, and voice registry."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
RUNS_DIR = REPO_ROOT / "runs"

DOSE_PARQUET = DATA_RAW / "dose_v1.parquet"
DOSE_CANONICAL = DATA_PROCESSED / "dose_v1_canonical.csv"

GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "project-amer-scs-sandbox")

# DOSE's published leaderboard, used only as an out-of-sample check (Workstream 1d).
# Never as a tuning target -- see the calibration plan.
DOSE_LEADERBOARD = {
    "synthio-rxpronounce": 91.2,
    "cartesia-sonic-3.6": 80.3,
    "elevenlabs-eleven_v3": 79.2,
    "openai-gpt-4o-mini-tts": 77.4,
    "xai-grok-tts": 77.0,
    "google-gemini-3.1-flash-tts": 74.5,
    "deepgram-aura-2-thalia-en": 72.6,
    "cartesia-sonic-3.5": 69.0,
    "azure-dragonhd-neural": 63.1,
}


@dataclass(frozen=True)
class Pricing:
    """Per-million-character USD price for a TTS tier.

    `verified` is False for every entry until someone checks it against the live
    published price sheet. Cost numbers in any report must carry this flag through
    so an unverified price is never mistaken for a measured one.
    """

    usd_per_million_chars: float
    source: str
    verified: bool = False


# Google Cloud Text-to-Speech list prices. Transcribed from public pricing docs,
# NOT yet reconciled against a billing export -- treat as indicative.
GOOGLE_TTS_PRICING = {
    "standard": Pricing(4.00, "cloud.google.com/text-to-speech/pricing", verified=False),
    "wavenet": Pricing(16.00, "cloud.google.com/text-to-speech/pricing", verified=False),
    "neural2": Pricing(16.00, "cloud.google.com/text-to-speech/pricing", verified=False),
    "studio": Pricing(160.00, "cloud.google.com/text-to-speech/pricing", verified=False),
    "chirp-hd": Pricing(30.00, "cloud.google.com/text-to-speech/pricing", verified=False),
    "chirp3-hd": Pricing(30.00, "cloud.google.com/text-to-speech/pricing", verified=False),
    "news": Pricing(16.00, "cloud.google.com/text-to-speech/pricing", verified=False),
    "polyglot": Pricing(16.00, "cloud.google.com/text-to-speech/pricing", verified=False),
    "casual": Pricing(16.00, "cloud.google.com/text-to-speech/pricing", verified=False),
    "mock": Pricing(0.00, "n/a -- mock backend", verified=True),
}


@dataclass(frozen=True)
class VoiceSpec:
    """A concrete, runnable TTS configuration."""

    system_id: str
    voice_name: str
    tier: str
    language_code: str = "en-US"
    notes: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def pricing(self) -> Pricing:
        return GOOGLE_TTS_PRICING[self.tier]


# --- Cheap iteration tier -------------------------------------------------
# Deliberately spans ~8x in list price and a real latency spread, so the harness
# is exercised against genuine cross-system variation rather than one flat backend.
CHEAP_TIER: dict[str, VoiceSpec] = {
    v.system_id: v
    for v in [
        VoiceSpec("gtts-standard-c", "en-US-Standard-C", "standard",
                  notes="Concatenative baseline. Cheapest; expected worst pronunciation.",
                  tags=("cheap", "baseline")),
        VoiceSpec("gtts-wavenet-c", "en-US-Wavenet-C", "wavenet",
                  notes="Classic neural vocoder tier.", tags=("cheap",)),
        VoiceSpec("gtts-neural2-c", "en-US-Neural2-C", "neural2",
                  notes="Mid tier.", tags=("cheap",)),
        VoiceSpec("gtts-chirp3hd-achernar", "en-US-Chirp3-HD-Achernar", "chirp3-hd",
                  notes="Gemini-family LLM-based voice; closest cheap proxy to the "
                        "Gemini TTS system DOSE scores at 74.5%.",
                  tags=("cheap", "gemini-family", "dose-overlap-proxy")),
    ]
}

# Control baselines. These exercise the runner/report failure paths offline; they
# are acoustic faults, not mispronunciations (see adapters/mock.py).
MOCK_TIER: dict[str, VoiceSpec] = {
    f"mock-{mode}": VoiceSpec(f"mock-{mode}", "mock", "mock",
                              notes=f"Control baseline: {mode} audio.",
                              tags=("control",))
    for mode in ("perfect", "truncated", "silent", "clipped", "overlong")
}

ALL_SYSTEMS: dict[str, VoiceSpec] = {**CHEAP_TIER, **MOCK_TIER}
