"""Task 3: pull University of Michigan College of Pharmacy student
pronunciation audio for DOSE ingredients that appear on its drug-name list.

The live site 403s this environment (Cloudflare JS challenge on both the page
and every audio file), so this goes through the Wayback Machine instead (see
`audio_sources_umich.py`): fetch one archived snapshot of the page, parse out
its 450 `<source>` audio URLs, and keep only the names that match a DOSE
ingredient -- this list predates DOSE and is a teaching set of established
drugs, so the overlap is small by nature, not a bug to chase.

For each match:

1. follow the `im_` URL's redirect to the snapshot that actually holds the
   file and download it, caching on disk by the page's own filename so a
   re-run costs no network calls;
2. verify it decodes, that its duration is a plausible single-name length,
   and that its seconds-per-syllable is not a batch outlier;
3. append to the shared manifest with source "umich", replacing this
   source's prior rows the same way the other two drivers do.
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
from dose_r.references import audio_manifest, audio_sources_umich, audio_verify

REFERENCES = ROOT / "dose_r" / "references" / "references.jsonl"
AUDIO_DIR = ROOT / "data" / "reference_audio" / "umich"
SOURCE = "umich"
SOURCE_NAME = "umich/wayback-machine"
FETCH_DELAY_S = 1.0


def name_types() -> dict[str, str]:
    with REFERENCES.open() as f:
        records = [json.loads(line) for line in f if line.strip()]
    return {r["ingredient"]: r["name_type"] for r in records}


def match_ingredients(
    page_names: dict[str, str], ingredients: list[str]
) -> dict[str, tuple[str, str]]:
    """ingredient -> (page filename, its Wayback audio URL), case-insensitive."""
    by_lower = {name.lower(): (name, url) for name, url in page_names.items()}
    return {
        ingredient: by_lower[ingredient.lower()]
        for ingredient in ingredients
        if ingredient.lower() in by_lower
    }


def build_clip(ingredient: str, name_type: str, page_name: str, url: str) -> dict:
    local_path = AUDIO_DIR / f"{page_name}.wav"
    if local_path.exists():
        data = local_path.read_bytes()
    else:
        data = audio_sources_umich.polite_fetch_audio(url, FETCH_DELAY_S)
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
        "ingredient": ingredient,
        "name_type": name_type,
        "source": SOURCE,
        "source_name": SOURCE_NAME,
        "query": ingredient,
        "coverage": "full",
        "headword": page_name,
        "respelling": None,
        "source_url": url,
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


def run(html: str | None = None) -> tuple[list[dict], list[dict]]:
    """(clips, misses) across every DOSE ingredient with a matching UMich filename."""
    html = html if html is not None else audio_sources_umich.fetch_page()
    page_names = audio_sources_umich.parse_audio_urls(html)
    types = name_types()
    matches = match_ingredients(page_names, list(types))

    clips, misses = [], []
    for ingredient, (page_name, url) in matches.items():
        name_type = types[ingredient]
        try:
            clips.append(build_clip(ingredient, name_type, page_name, url))
        except Exception as exc:
            misses.append(
                {
                    "ingredient": ingredient,
                    "name_type": name_type,
                    "reason": "download_failed",
                    "detail": str(exc),
                }
            )

    durations = {c["ingredient"]: c["duration_s"] for c in clips if c["duration_s"] is not None}
    for name, note in audio_verify.syllable_outliers(durations).items():
        clip = next(c for c in clips if c["ingredient"] == name)
        clip["flags"].append(f"possible name mismatch: {note}")
        clip["status"] = "flagged"

    return clips, misses


def main() -> None:
    clips, misses = run()
    existing = audio_manifest.load()
    merged = audio_manifest.replace_source(existing, clips, SOURCE)
    audio_manifest.save(merged)

    ok = sum(1 for c in clips if c["status"] == "ok")
    flagged = sum(1 for c in clips if c["status"] == "flagged")
    print(
        f"umich: {ok} ok, {flagged} flagged, {len(misses)} missing, "
        f"{len(clips) + len(misses)} matched ingredients total"
    )
    if flagged:
        print("flagged clips:")
        for c in clips:
            if c["status"] == "flagged":
                print(f"  {c['ingredient']}: {c['flags']}")
    if misses:
        print("misses:")
        for m in misses:
            print(f"  {m['ingredient']}: {m['reason']} -- {m['detail']}")


if __name__ == "__main__":
    main()
