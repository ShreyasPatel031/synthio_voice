"""Selecting and transcribing the human reference-pronunciation clips.

Workstream 1's manifest (`data/reference_audio/manifest.jsonl`) lists clips from
Drugs.com, Merriam-Webster, UMich, ClinCalc, and NCI. Only the Drugs.com WAVs
are committed to git (other sources are gitignored as re-fetchable from each
record's public `source_url`). NCI MP3s come from `nci-media.cancer.gov` via
`scripts/fetch_nci_reference_audio.py`. Merriam-Webster's 82 clips are fetched
into `data/reference_audio/mw/`. UMich's clips are not fetched (no direct
per-clip URL recorded, only a Wayback Machine page) and remain a known
coverage gap.

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
# ingredient. NCI Dictionary of Cancer Terms is gold when present: it is an
# official NIH recording of the name (often the full multi-word generic),
# not a consumer-site clip that sometimes says only the stem. Merriam-Webster
# is next -- an actual pronouncing dictionary with a written respelling --
# then Drugs.com, then UMich. Secondary human recordings (Wiktionary,
# Commons, Forvo, …) are last: they fill names with no gold clip, and must
# not override NCI/MW/Drugs.com when those exist.
_SOURCE_PRIORITY = (
    "nci",
    "merriam-webster",
    "drugs.com",
    "umich",
    "clinicalinfo",
    "wiktionary",
    "wikipedia",
    "commons",
    "medlineplus",
    "forvo",
    "youtube",
    "web",
)


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

    `available_clips()` picks one clip per ingredient (NCI gold when present,
    then Merriam-Webster, then Drugs.com) for Path 2's own scoring, which is
    the right contract for a single, stable reference per item -- but it
    silently discards the OTHER clip for ingredients that have more than one
    recording. That is fine for scoring a fixed candidate
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
