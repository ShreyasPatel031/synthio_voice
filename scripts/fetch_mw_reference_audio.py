"""Task 1: pull Merriam-Webster pronunciation audio for every DOSE ingredient
whose gold reference already carries an MW Medical API source.

For each such (ingredient, query) pair -- a query is usually the ingredient
itself, but a multi-word ingredient that only resolved word-by-word (see
`references/build.py`) contributes one query per word that answered -- this:

1. fetches the full set of API entries for that query (cached; see
   `audio_sources.mw_entries`),
2. finds the entry whose headword is *exactly* that query (never guesses
   among the homographs and related phrases MW routinely returns alongside
   it),
3. downloads its audio if it has any, caching the file on disk by MW's own
   filename so a re-run costs no network calls, and
4. verifies the download (decodable, plausible single-word duration) before
   it is accepted into the manifest.

Every query that does not clear all four steps is recorded as a miss with a
reason, not silently dropped -- `build_audio_coverage.py` reports them.
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dose_r.adapters.base import sha256_hex
from dose_r.references import audio_manifest, audio_sources, audio_verify

REFERENCES = ROOT / "dose_r" / "references" / "references.jsonl"
AUDIO_DIR = ROOT / "data" / "reference_audio" / "mw"
UA = {"User-Agent": "Mozilla/5.0 (compatible; DOSE-R research benchmark)"}
TIMEOUT = 25
SOURCE = "merriam-webster"
SOURCE_NAME = "merriam-webster/medical-api"


def load_references() -> list[dict]:
    with REFERENCES.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def mw_queries(record: dict) -> list[tuple[str, str]]:
    """[(query, coverage)] for every MW medical-api source on `record`.

    `coverage` is "full" when the query is the whole ingredient and
    "component" when it is only one word of a multi-word ingredient -- audio
    for the latter is a real clip, but it pronounces less than the full name.
    """
    ingredient = record["ingredient"]
    out = []
    for s in record.get("sources") or []:
        if s["name"] != SOURCE_NAME:
            continue
        query = urllib.parse.unquote(s["url"].rsplit("/", 1)[-1])
        coverage = "full" if query.lower() == ingredient.lower() else "component"
        out.append((query, coverage))
    return out


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=TIMEOUT).read()


def _miss(ingredient, name_type, query, coverage, reason, detail) -> dict:
    return {
        "ingredient": ingredient,
        "name_type": name_type,
        "query": query,
        "coverage": coverage,
        "reason": reason,
        "detail": detail,
    }


def fetch_one(
    ingredient: str,
    name_type: str,
    query: str,
    coverage: str,
    entries_cache: dict[str, list[dict] | None],
) -> dict:
    """A clip record on success, or a miss record (see `_miss`) on failure."""
    if query not in entries_cache:
        entries_cache[query] = audio_sources.mw_entries(query)
    entries = entries_cache[query]

    if not entries:
        return _miss(ingredient, name_type, query, coverage, "no_mw_entries",
                      "the API returned no entries for this query")

    entry = audio_sources.find_headword_entry(entries, query)
    if entry is None:
        headwords = sorted({audio_sources.entry_headword(e) for e in entries})
        return _miss(ingredient, name_type, query, coverage, "no_exact_headword",
                      f"no returned entry's headword equals the query; got {headwords}")

    headword = audio_sources.entry_headword(entry)
    audio = audio_sources.entry_audio(entry)
    if audio is None:
        return _miss(ingredient, name_type, query, coverage, "no_audio_on_entry",
                      f"the matching entry ({headword!r}) carries no sound.audio")

    local_path = AUDIO_DIR / f"{audio['filename']}.mp3"
    if local_path.exists():
        data = local_path.read_bytes()
    else:
        try:
            data = _download(audio["url"])
        except Exception as exc:
            return _miss(ingredient, name_type, query, coverage, "download_failed", str(exc))
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

    if coverage != "full":
        flags.append(f"partial coverage: audio pronounces only {query!r}, not the full ingredient")

    return {
        "ingredient": ingredient,
        "name_type": name_type,
        "source": SOURCE,
        "source_name": SOURCE_NAME,
        "query": query,
        "coverage": coverage,
        "headword": headword,
        "respelling": audio["respelling"],
        "source_url": audio["url"],
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
    """(clips, misses) across every MW-sourced ingredient in references.jsonl."""
    clips, misses = [], []
    entries_cache: dict[str, list[dict] | None] = {}
    for r in load_references():
        for query, coverage in mw_queries(r):
            result = fetch_one(r["ingredient"], r["name_type"], query, coverage, entries_cache)
            (misses if "reason" in result else clips).append(result)
    return clips, misses


def main() -> None:
    clips, misses = run()
    existing = audio_manifest.load()
    merged = audio_manifest.replace_source(existing, clips, SOURCE)
    audio_manifest.save(merged)

    ok = sum(1 for c in clips if c["status"] == "ok")
    flagged = sum(1 for c in clips if c["status"] == "flagged")
    print(f"merriam-webster: {ok} ok, {flagged} flagged, {len(misses)} missing, "
          f"{len(clips) + len(misses)} queries total")
    if misses:
        print("misses by reason:")
        by_reason: dict[str, int] = {}
        for m in misses:
            by_reason[m["reason"]] = by_reason.get(m["reason"], 0) + 1
        for reason, n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
            print(f"  {reason}: {n}")


if __name__ == "__main__":
    main()
