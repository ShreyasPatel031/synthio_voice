"""Pull NCI Dictionary of Cancer Terms pronunciation audio for every DOSE
ingredient the dictionary lists.

`sources.nci_pronunciation` already does the lookup (its own cached JSON
fetch against `webapis.cancer.gov/glossary/v1/`, reverse-engineered from the
site's own app the same way the AMA USAN search API was found) and returns a
`pronunciation.audio` URL alongside the text respelling whenever the term is
in the dictionary at all -- there is no separate parsing step the way
ClinCalc/UMich need, and no per-ingredient miss reason beyond "not in this
dictionary" (checked directly in an earlier session: names with no
oncology/supportive-care connection, e.g. valsartan, aren't in it at all).

Every match gets a clip, whether or not that ingredient already has audio
from another source -- an ingredient with, say, Drugs.com audio already
still benefits from NCI's audio as a second, independent government
recording for cross-source agreement checking (see `AUDIO_COVERAGE.md`'s
"Cross-source agreement" section), the same reasoning that already applies
to every other source in this manifest.

For each ingredient:

1. look up its NCI record (cached; see `sources.nci_pronunciation`);
2. skip it if the dictionary doesn't have it, or has it with no audio field;
3. download the audio, caching the file on disk by NCI's own numeric media
   ID so a re-run costs no network calls;
4. verify it decodes, that its duration is a plausible single-name length,
   and that its seconds-per-syllable is not a batch outlier;
5. append to the shared manifest with source "nci", replacing this source's
   prior rows the same way every other driver does.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dose_r.adapters.base import sha256_hex
from dose_r.references import audio_manifest, audio_verify, sources

REFERENCES = ROOT / "dose_r" / "references" / "references.jsonl"
DATASET = ROOT / "data" / "dose_v1.jsonl"
AUDIO_DIR = ROOT / "data" / "reference_audio" / "nci"
UA = {"User-Agent": "Mozilla/5.0 (compatible; DOSE-R research benchmark)"}
TIMEOUT = 25
SOURCE = "nci"
SOURCE_NAME = "nci/glossary-api"


def ingredient_types() -> dict[str, str]:
    with REFERENCES.open() as f:
        return {r["ingredient"]: r["name_type"] for r in (json.loads(l) for l in f if l.strip())}


def all_ingredients() -> list[str]:
    with DATASET.open() as f:
        return sorted({i for line in f for i in json.loads(line)["ingredients"]})


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=TIMEOUT).read()


def build_clip(ingredient: str, name_type: str, record: dict) -> dict:
    audio_url = record["audio"]
    media_id = audio_url.rsplit("/", 1)[-1]
    local_path = AUDIO_DIR / media_id
    if local_path.exists():
        data = local_path.read_bytes()
    else:
        data = _download(audio_url)
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)

    probe = audio_verify.probe_mp3(data)
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
        "headword": ingredient,
        "respelling": record["raw"],
        "source_url": audio_url,
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


def run() -> tuple[list[dict], list[dict]]:
    """(clips, misses) across every DOSE ingredient the NCI dictionary lists."""
    types = ingredient_types()
    clips, misses = [], []
    for ingredient in all_ingredients():
        name_type = types.get(ingredient, "unknown")
        record = sources.nci_pronunciation(ingredient)
        if not record:
            continue
        if not record.get("audio"):
            misses.append(
                {
                    "ingredient": ingredient,
                    "name_type": name_type,
                    "reason": "no_audio_field",
                    "detail": "NCI has this term but its pronunciation has no audio recording",
                }
            )
            continue
        try:
            clips.append(build_clip(ingredient, name_type, record))
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
        f"nci: {ok} ok, {flagged} flagged, {len(misses)} missing, "
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
