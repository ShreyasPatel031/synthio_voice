"""Verification for downloaded reference-audio bytes: decodability, duration
sanity, format. Pure functions over bytes already on disk -- no network here.

A single drug name read aloud is short. `MIN_DURATION_S` / `MAX_DURATION_S`
bound what "short" means so a truncated download or a misattributed longer
clip (a whole sentence, a phrase entry) is flagged instead of silently kept.
"""

from __future__ import annotations

import io
import re
import statistics
import wave
from dataclasses import dataclass

from mutagen.mp3 import MP3
from mutagen.mp3 import error as MutagenMP3Error

MIN_DURATION_S = 0.3
MAX_DURATION_S = 4.0

_VOWEL_GROUPS = re.compile(r"[aeiouy]+")


@dataclass
class AudioProbe:
    ok: bool
    format: str | None = None
    duration_s: float | None = None
    sample_rate_hz: int | None = None
    channels: int | None = None
    error: str | None = None


def probe_mp3(data: bytes) -> AudioProbe:
    try:
        info = MP3(io.BytesIO(data)).info
    except MutagenMP3Error as exc:
        return AudioProbe(ok=False, format="mp3", error=str(exc))
    except Exception as exc:
        return AudioProbe(ok=False, format="mp3", error=str(exc))
    return AudioProbe(
        ok=True,
        format="mp3",
        duration_s=round(info.length, 4),
        sample_rate_hz=info.sample_rate,
        channels=info.channels,
    )


def probe_wav(data: bytes) -> AudioProbe:
    try:
        with wave.open(io.BytesIO(data)) as w:
            frames = w.getnframes()
            rate = w.getframerate()
            channels = w.getnchannels()
            sampwidth = w.getsampwidth()
    except Exception as exc:
        return AudioProbe(ok=False, format="wav", error=str(exc))

    if sampwidth != 2:
        return AudioProbe(
            ok=False, format="wav", error=f"expected 16-bit PCM, got {sampwidth * 8}-bit"
        )
    duration = frames / rate if rate else 0.0
    return AudioProbe(
        ok=True,
        format="wav",
        duration_s=round(duration, 4),
        sample_rate_hz=rate,
        channels=channels,
    )


def duration_flag(duration_s: float | None) -> str | None:
    if duration_s is None:
        return None
    if duration_s < MIN_DURATION_S:
        return f"duration {duration_s:.3f}s is below the {MIN_DURATION_S}s floor for a single name"
    if duration_s > MAX_DURATION_S:
        return f"duration {duration_s:.3f}s is above the {MAX_DURATION_S}s ceiling for a single name"
    return None


def estimate_syllables(name: str) -> int:
    """A crude vowel-group count, good enough to gauge relative clip length."""
    words = re.findall(r"[a-zA-Z]+", name)
    return sum(max(1, len(_VOWEL_GROUPS.findall(w.lower()))) for w in words) or 1


def syllable_outliers(
    durations: dict[str, float], low: float = 0.5, high: float = 2.0
) -> dict[str, str]:
    """Names whose seconds-per-syllable falls outside [low, high] x this batch's median.

    A clip disproportionately long for its name's syllable count is the
    signature of a page's audio actually pronouncing a related name (e.g. a
    brand page's clip saying its generic) rather than the name it was
    collected for. This flags candidates for a human ear check; it never
    decides a clip is wrong on its own.
    """
    if not durations:
        return {}
    per_syllable = {name: d / estimate_syllables(name) for name, d in durations.items()}
    median = statistics.median(per_syllable.values())
    lo, hi = median * low, median * high
    return {
        name: (
            f"{rate:.3f}s/syllable is outside [{lo:.3f}, {hi:.3f}]s/syllable "
            f"for this batch (median {median:.3f})"
        )
        for name, rate in per_syllable.items()
        if rate < lo or rate > hi
    }
