"""Drugs.com pronunciation audio, imported from browser-collected JSON.

Drugs.com returns 403 to this environment on every request, page and audio
alike (see `data/collected/HANDOFF_AUDIO_COLLECTION.md`), so the clips arrive
pre-collected: a person browsed the site and dumped `{name, status,
audio_url, respell, audio_b64}` records to JSON. This module turns those
records into a name-keyed lookup ready for verification and import.

`respell` is dropped entirely -- it captured arbitrary page text, not a
pronunciation. `status: 404` does not mean no audio exists; it means the
collector's URL guess for that name missed (many drugs live outside the
`/{slug}.html` pattern it tried first). Only a null `audio_b64` is a miss.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def decode_audio(record: dict) -> bytes:
    return base64.b64decode(record["audio_b64"])


def merge_records(paths: list[Path]) -> dict[str, dict]:
    """name -> its collected record, later files winning whole-record."""
    merged: dict[str, dict] = {}
    for path in paths:
        for record in load_records(path):
            merged[record["name"]] = record
    return merged


def audio_conflicts(paths: list[Path]) -> list[str]:
    """Names where two input files carry different audio for the same name."""
    seen: dict[str, str] = {}
    conflicts: set[str] = set()
    for path in paths:
        for record in load_records(path):
            if not record.get("audio_b64"):
                continue
            name = record["name"]
            if name in seen and seen[name] != record["audio_b64"]:
                conflicts.add(name)
            seen[name] = record["audio_b64"]
    return sorted(conflicts)
