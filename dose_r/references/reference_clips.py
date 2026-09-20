"""Selecting and transcribing the human reference-pronunciation clips.

Workstream 1's manifest (`data/reference_audio/manifest.jsonl`) lists clips from
three sources -- Drugs.com, Merriam-Webster, UMich. Only the Drugs.com WAVs are
committed to git on their branch (MW/UMich are gitignored there as re-fetchable
from each record's public `source_url`). This repo fetches Merriam-Webster's 82
clips directly from that URL (small, public, per-word audio files from their
dictionary API) into `data/reference_audio/mw/`, matching the manifest's
`local_path` convention so `available_clips()` needs no special-casing. UMich's
clips are not fetched (no direct per-clip URL recorded, only a Wayback Machine
page) and remain a known coverage gap.

This module only ever reads a clip whose `local_path` resolves on disk, so any
still-missing record is silently skipped rather than treated as an error --
a known gap, not a bug (see `docs/REFERENCE_AUDIO_GROUNDING.md`).
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import audio_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Preference order when more than one source has a usable clip for the same
# ingredient. Merriam-Webster is preferred first: it is an actual pronouncing
# dictionary, not just an audio file -- its manifest record carries a written
# respelling (e.g. aspirin: "as-p(schwa-)rin", the parenthetical marking the
# schwa as an explicitly optional, dictionary-documented variant) that lets a
# disagreement between sources be checked against a real authority instead of
# guessed at. Concretely: Drugs.com's and Merriam-Webster's Aspirin clips
# sound different (elided vs. unelided middle syllable) -- checking MW's own
# transcription confirmed both are the SAME dictionary entry's accepted
# variants, not a real conflict. Drugs.com remains the fallback for the
# ~33% of ingredients (95/284 per AUDIO_COVERAGE.md) that only it covers.
_SOURCE_PRIORITY = ("merriam-webster", "drugs.com", "umich")


@dataclass(frozen=True)
class ReferenceClip:
    ingredient: str
    name_type: str
    source: str
    path: Path
    audio_format: str
    sample_rate_hz: int
    duration_s: float
    # Written dictionary respelling, e.g. "as-p(schwa-)rin" for aspirin -- only
    # Merriam-Webster records carry this. Real provenance for why a given clip
    # was treated as the reference, not just an audio file with no transcript
    # behind it. None for sources (Drugs.com, UMich) that don't provide one.
    respelling: str | None = None


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
            respelling=best.get("respelling"),
        )
    return out


def available_clips_all(
    manifest_path: Path | None = None,
) -> dict[str, list[ReferenceClip]]:
    """ALL usable clips per ingredient, not just the single best one.

    `available_clips()` picks one clip per ingredient (Merriam-Webster
    preferred) for Path 2's own scoring, which is the right contract for a
    single, stable reference per item -- but it silently discards the OTHER
    clip for the 79 ingredients that have both a Drugs.com AND a
    Merriam-Webster recording. That is fine for scoring a fixed candidate
    consistently over time, but wrong for evaluating a NEW candidate model
    against "how a human says this" in general: two real humans can say the
    same drug correctly in genuinely different ways (see aspirin's
    dictionary-documented schwa variant), so scoring against only one of them
    risks the exact false-positive this project already found this session
    (aripiprazole/acoramidis scoring as "wrong" against one reference despite
    being valid alternate pronunciations). Candidate-model evaluation should
    use ALL available clips and credit the candidate for matching ANY of
    them -- see `scoring.candidate_eval.score_against_best_reference`.
    """
    records = audio_manifest.load(manifest_path or audio_manifest.MANIFEST_PATH)
    by_ingredient: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        if r.get("coverage") == "full" and _resolve(r).exists():
            by_ingredient.setdefault(r["ingredient"], []).append(r)

    out: dict[str, list[ReferenceClip]] = {}
    for ingredient, recs in by_ingredient.items():
        out[ingredient] = [
            ReferenceClip(
                ingredient=ingredient, name_type=r["name_type"], source=r["source"],
                path=_resolve(r), audio_format=r["format"],
                sample_rate_hz=r["sample_rate_hz"], duration_s=r["duration_s"],
                respelling=r.get("respelling"),
            )
            for r in recs
        ]
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
