"""Selecting and transcribing the human reference-pronunciation clips.

Workstream 1's manifest (`data/reference_audio/manifest.jsonl`) lists clips from
three sources -- Drugs.com, Merriam-Webster, UMich -- but only the Drugs.com WAVs
are actually committed to git. The other two are gitignored deliberately (their
own .gitignore: Merriam-Webster's MP3s are re-fetchable from the `source_url` in
each manifest record, so keeping a copy in git is redundant; the same is true in
spirit for UMich). That means `local_path` in a Merriam-Webster or UMich record
points to a file that exists in Workstream 1's container, not in this one.

This module only ever reads a clip whose `local_path` resolves on disk, so those
un-committed records are silently skipped rather than treated as an error --
they are a known gap, not a bug (see `docs/REFERENCE_AUDIO_GROUNDING.md`).
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import audio_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Preference order when more than one source has a usable clip for the same
# ingredient. Drugs.com is the only source verified for every clip in this
# environment (collected by hand because the site blocks automated fetches);
# preferring it first means the "which clip did we grade against" choice does
# not silently change if Merriam-Webster/UMich clips are pulled in later.
_SOURCE_PRIORITY = ("drugs.com", "merriam-webster", "umich")


@dataclass(frozen=True)
class ReferenceClip:
    ingredient: str
    name_type: str
    source: str
    path: Path
    audio_format: str
    sample_rate_hz: int
    duration_s: float


def _resolve(record: dict[str, Any]) -> Path:
    p = Path(record["local_path"])
    return p if p.is_absolute() else REPO_ROOT / p


def available_clips(
    manifest_path: Path | None = None,
) -> dict[str, ReferenceClip]:
    """One best clip per ingredient, restricted to files that exist on disk."""
    records = audio_manifest.load(manifest_path or audio_manifest.MANIFEST_PATH)
    by_ingredient: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        if r.get("coverage") == "full" and _resolve(r).exists():
            by_ingredient.setdefault(r["ingredient"], []).append(r)

    out: dict[str, ReferenceClip] = {}
    for ingredient, recs in by_ingredient.items():
        recs.sort(key=lambda r: _SOURCE_PRIORITY.index(r["source"])
                   if r["source"] in _SOURCE_PRIORITY else len(_SOURCE_PRIORITY))
        best = recs[0]
        out[ingredient] = ReferenceClip(
            ingredient=ingredient, name_type=best["name_type"], source=best["source"],
            path=_resolve(best), audio_format=best["format"],
            sample_rate_hz=best["sample_rate_hz"], duration_s=best["duration_s"],
        )
    return out


def stt_config_for_clip(clip: ReferenceClip) -> dict[str, Any]:
    """Build the Cloud STT `config` block for this clip's actual encoding.

    Read directly from the WAV header rather than trusting the manifest's
    recorded sample rate, in case a clip was re-encoded after the manifest was
    written. MP3s can't be introspected as cheaply, so those fall back to the
    manifest's numbers -- Cloud STT requires `encoding` and `sampleRateHertz`
    to be declared for MP3 rather than auto-detecting them.
    """
    if clip.audio_format == "wav":
        with wave.open(str(clip.path), "rb") as w:
            channels = w.getnchannels()
            config: dict[str, Any] = {
                "encoding": "LINEAR16", "sampleRateHertz": w.getframerate(),
                "languageCode": "en-US", "model": "latest_long",
            }
            # A handful of the Drugs.com clips are stereo. Cloud STT defaults to
            # assuming mono and rejects anything else outright (HTTP 400) unless
            # told the real channel count -- found by running this against the
            # full corpus, not by inspecting the files upfront.
            if channels > 1:
                config["audioChannelCount"] = channels
            return config
    if clip.audio_format == "mp3":
        return {"encoding": "MP3", "sampleRateHertz": clip.sample_rate_hz,
                "languageCode": "en-US", "model": "latest_long"}
    raise ValueError(f"unsupported reference audio format: {clip.audio_format!r}")
