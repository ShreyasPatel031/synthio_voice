"""Merriam-Webster pronunciation *audio*, on top of `sources.py`'s API plumbing.

`sources.merriam_webster()` only ever kept the extracted respelling text; it
throws away the rest of each API response. This module re-fetches (cached,
same as `sources.py`) the full entry list for a query and adds the two things
the respelling path never needed:

- the playable audio URL, built from `hwi.prs[i].sound.audio` by MW's
  documented convention;
- the headword (`meta.id`) each entry actually belongs to, so a query that
  returns several homographs or related phrases (e.g. "insulin" also returns
  "insulin aspart", "insulin glargine", "insulin-like growth factor") can be
  checked against the ingredient it is meant to answer, rather than assumed.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request

from . import sources

AUDIO_BASE = "https://media.merriam-webster.com/audio/prons/en/us/mp3"
_HOMOGRAPH_SUFFIX = re.compile(r":\d+$")

_MIN_INTERVAL_S = 0.2
_next_ok = [0.0]


def audio_url(filename: str) -> str:
    """The playable MP3 URL for an MW `sound.audio` filename."""
    if filename.startswith("bix"):
        subdir = "bix"
    elif filename.startswith("gg"):
        subdir = "gg"
    elif not filename[:1].isalpha():
        subdir = "number"
    else:
        subdir = filename[0].lower()
    return f"{AUDIO_BASE}/{subdir}/{filename}.mp3"


def normalize_headword(value: str) -> str:
    """MW headword text -> comparable form: no homograph suffix, no case."""
    return _HOMOGRAPH_SUFFIX.sub("", value or "").strip().lower()


def mw_entries(query: str) -> list[dict] | None:
    """Full raw API entries for `query`, cached on disk like `sources.py`.

    None means the lookup could not be made at all (no key, transient
    failure); an empty result from a real response is cached as `[]`, which
    this returns as `None` too since callers only care about "answered" vs
    "did not answer".
    """
    import os

    key = os.environ.get("MW_MEDICAL_KEY")
    if not key:
        return None

    cache_key = f"mw-entries::{query}"
    hit = sources._cached(cache_key)
    if hit is not None:
        return hit or None

    wait = _next_ok[0] - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _next_ok[0] = time.monotonic() + _MIN_INTERVAL_S

    url = f"{sources.MW_API_BASE}/{urllib.parse.quote(query)}?key={urllib.parse.quote(key)}"
    try:
        req = urllib.request.Request(url, headers=sources.UA)
        raw = urllib.request.urlopen(req, timeout=sources.TIMEOUT).read().decode(
            "utf-8", "ignore"
        )
        entries = json.loads(raw)
    except Exception:
        return None  # not cached: a transient failure shouldn't poison the cache

    entries = [e for e in entries if isinstance(e, dict)]
    sources._store(cache_key, entries)
    return entries or None


def find_headword_entry(entries: list[dict], target: str) -> dict | None:
    """The entry among `entries` whose headword is exactly `target`, or None.

    Exact match only, deliberately: `entries` routinely includes related
    phrases and homographs (see module docstring), and guessing among them is
    exactly the failure mode this whole task exists to catch.
    """
    target_norm = normalize_headword(target)
    for entry in entries:
        if normalize_headword(entry.get("meta", {}).get("id", "")) == target_norm:
            return entry
    return None


def entry_headword(entry: dict) -> str:
    """Display headword, syllable-boundary `*` marks stripped."""
    hw = entry.get("hwi", {}).get("hw") or entry.get("meta", {}).get("id", "")
    return hw.replace("*", "")


def entry_audio(entry: dict) -> dict | None:
    """The first pronunciation on `entry` that carries playable audio."""
    for pr in entry.get("hwi", {}).get("prs", []):
        sound = pr.get("sound") or {}
        filename = sound.get("audio")
        if filename:
            return {
                "filename": filename,
                "respelling": pr.get("mw"),
                "url": audio_url(filename),
            }
    return None
