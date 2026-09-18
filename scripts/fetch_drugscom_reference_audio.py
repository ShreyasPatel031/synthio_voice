"""Task 2: import Drugs.com pronunciation audio, collected by hand, into the
shared reference-audio manifest.

Drugs.com 403s this environment on every request, so unlike Merriam-Webster
this has no live fetch path -- it imports JSON already collected through a
browser (see `data/collected/HANDOFF_AUDIO_COLLECTION.md`). For every
ingredient with a collected clip this:

1. decodes the base64 WAV and writes it under `data/reference_audio/drugs_com/`,
   named after drugs.com's own audio URL so a re-run neither re-decodes nor
   re-writes a file already on disk;
2. verifies it decodes, that its duration is a plausible single-name length,
   and that its seconds-per-syllable is not a batch outlier (see
   `audio_verify.syllable_outliers`) -- the signature of a brand page's clip
   actually pronouncing its generic instead of the brand asked for;
3. appends to the shared manifest with source "drugs.com", leaving every
   other source's rows untouched (`audio_manifest.replace_source`, which
   itself makes re-running idempotent: it swaps this source's rows, it never
   accumulates duplicates).

A null `audio_b64` is a miss, not an error, and a 404 status is not treated
as "no audio exists" -- both are reported, not silently dropped.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dose_r.adapters.base import sha256_hex
from dose_r.references import audio_manifest, audio_sources_drugscom, audio_verify

REFERENCES = ROOT / "dose_r" / "references" / "references.jsonl"
AUDIO_DIR = ROOT / "data" / "reference_audio" / "drugs_com"
SOURCE = "drugs.com"
SOURCE_NAME = "drugs.com"

DEFAULT_INPUTS = [
    ROOT / "data" / "collected" / "drugscom_246.json",
    ROOT / "data" / "collected" / "62a75a68-drugscom_284.json",
]


def name_types() -> dict[str, str]:
    with REFERENCES.open() as f:
        records = [json.loads(line) for line in f if line.strip()]
    return {r["ingredient"]: r["name_type"] for r in records}


def build_clip(name: str, name_type: str, record: dict) -> dict:
    filename = record["audio_url"].rsplit("/", 1)[-1]
    local_path = AUDIO_DIR / filename
    if local_path.exists():
        data = local_path.read_bytes()
    else:
        data = audio_sources_drugscom.decode_audio(record)
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)

    probe = audio_verify.probe_wav(data)
    flags = []
    if not probe.ok:
        flags.append(f"undecodable: {probe.error}")
    else:
        dflag = audio_verify.duration_flag(probe.duration_s)
        if dflag:
            flags.append(dflag)

    return {
        "ingredient": name,
        "name_type": name_type,
        "source": SOURCE,
        "source_name": SOURCE_NAME,
        "query": name,
        "coverage": "full",
        "headword": name,
        "respelling": None,
        "source_url": record["audio_url"],
        "local_path": str(local_path.relative_to(ROOT)),
        "sha256": sha256_hex(data),
        "bytes": len(data),
        "format": probe.format,
        "duration_s": probe.duration_s,
        "sample_rate_hz": probe.sample_rate_hz,
        "channels": probe.channels,
        "status": "ok" if probe.ok and not flags else "flagged",
        "flags": flags,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def build_miss(name: str, name_type: str, record: dict) -> dict:
    reason = "no_audio_on_page" if record["status"] == 200 else "url_not_found_not_absent"
    return {
        "ingredient": name,
        "name_type": name_type,
        "status": record["status"],
        "reason": reason,
    }


def run(paths: list[Path] | None = None) -> tuple[list[dict], list[dict], list[str]]:
    """(clips, misses, audio_conflicts) across the union of `paths`."""
    paths = paths or DEFAULT_INPUTS
    merged = audio_sources_drugscom.merge_records(paths)
    conflicts = audio_sources_drugscom.audio_conflicts(paths)
    types = name_types()

    clips, misses = [], []
    for name, record in merged.items():
        name_type = types[name]
        if record.get("audio_b64"):
            clips.append(build_clip(name, name_type, record))
        else:
            misses.append(build_miss(name, name_type, record))

    durations = {c["ingredient"]: c["duration_s"] for c in clips if c["duration_s"] is not None}
    for name, note in audio_verify.syllable_outliers(durations).items():
        clip = next(c for c in clips if c["ingredient"] == name)
        clip["flags"].append(f"possible name mismatch: {note}")
        clip["status"] = "flagged"

    return clips, misses, conflicts


def main() -> None:
    clips, misses, conflicts = run()
    existing = audio_manifest.load()
    merged = audio_manifest.replace_source(existing, clips, SOURCE)
    audio_manifest.save(merged)

    ok = sum(1 for c in clips if c["status"] == "ok")
    flagged = sum(1 for c in clips if c["status"] == "flagged")
    print(
        f"drugs.com: {ok} ok, {flagged} flagged, {len(misses)} missing, "
        f"{len(clips) + len(misses)} ingredients total"
    )
    if flagged:
        print("flagged clips:")
        for c in clips:
            if c["status"] == "flagged":
                print(f"  {c['ingredient']}: {c['flags']}")

    if conflicts:
        print(f"conflicting audio across input files for: {conflicts}")

    by_reason: dict[str, int] = {}
    for m in misses:
        by_reason[m["reason"]] = by_reason.get(m["reason"], 0) + 1
    if by_reason:
        print("misses by reason:")
        for reason, n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
            print(f"  {reason}: {n}")

    dupes = audio_manifest.duplicate_groups(merged)
    if dupes:
        print("byte-identical audio shared across ingredients:")
        for h, ings in dupes.items():
            print(f"  {h[:12]}: {ings}")


if __name__ == "__main__":
    main()
