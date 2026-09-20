"""Task 4: pull ClinCalc "Top 250 Drugs" pronunciation audio for DOSE
ingredients that appear on its list, generic and/or brand.

ClinCalc is reachable directly (no Cloudflare block), so unlike Drugs.com and
UMich this is a live fetch, same shape as Merriam-Webster's: one index fetch,
then only the drug pages a DOSE ingredient might actually match (see
`audio_sources_clincalc.candidate_slugs`), each with up to two audio clips --
a generic-name recording and a brand-name recording, kept as two separate
manifest rows because they are two distinct recordings of two distinct names.

For each drug page fetched:

1. every name its "generic"/"brand" headed block actually plays (there can be
   several, joined by "; ", for a combination product or a several-brands-
   share-one-clip page) is checked against every DOSE ingredient
   (`audio_sources_clincalc.name_matches`, case-insensitive, salt/ester-
   suffix tolerant);
2. each match downloads (once; cached on disk by the audio's own filename)
   and verifies its clip exactly as the other three sources do;
3. a clip whose block names more than one thing gets `coverage: "component"`
   and a flag, the same convention Merriam-Webster's word-level partial
   clips use -- it is a real recording, but not of the ingredient alone;
4. when the *other* block on the same page also has audio, that clip's
   presence is recorded in this row's `respelling` (unused by every other
   source) so a human can find and listen to the paired name even when that
   name is not itself a DOSE ingredient and so never becomes its own row.

A DOSE ingredient the index suggested for a page but the page's own headed
names never confirm (the fluticasone/Flonase case -- see module docstring) is
reported as a miss, not silently dropped.
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
from dose_r.references import audio_manifest, audio_sources_clincalc, audio_verify

REFERENCES = ROOT / "dose_r" / "references" / "references.jsonl"
AUDIO_DIR = ROOT / "data" / "reference_audio" / "clincalc"
SOURCE = "clincalc"
SOURCE_NAME = "clincalc/top250-pronunciation"
FETCH_DELAY_S = 0.3


def name_types() -> dict[str, str]:
    with REFERENCES.open() as f:
        records = [json.loads(line) for line in f if line.strip()]
    return {r["ingredient"]: r["name_type"] for r in records}


def _download(url: str) -> bytes:
    filename = url.rsplit("/", 1)[-1]
    local_path = AUDIO_DIR / filename
    if local_path.exists():
        return local_path.read_bytes()
    data = audio_sources_clincalc.polite_fetch_audio(url, FETCH_DELAY_S)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(data)
    return data


def build_clip(
    ingredient: str,
    name_type: str,
    headword: str,
    components: list[str],
    url: str,
    other_side: str | None,
    other_headword: str | None,
    other_url: str | None,
) -> dict:
    filename = url.rsplit("/", 1)[-1]
    local_path = AUDIO_DIR / filename
    data = _download(url)

    probe = audio_verify.probe_mp3(data)
    flags = []
    if not probe.ok:
        flags.append(f"undecodable: {probe.error}")
    else:
        dflag = audio_verify.duration_flag(probe.duration_s)
        if dflag:
            flags.append(dflag)

    coverage = "full" if len(components) == 1 else "component"
    if coverage != "full":
        flags.append(
            f"partial coverage: this clip pronounces {headword!r} together, "
            f"not {ingredient!r} alone"
        )

    respelling = None
    if other_url:
        respelling = (
            f"clincalc also has a separate {other_side}-name clip on this "
            f"page: {other_headword!r} ({other_url})"
        )

    return {
        "ingredient": ingredient,
        "name_type": name_type,
        "source": SOURCE,
        "source_name": SOURCE_NAME,
        "query": headword,
        "coverage": coverage,
        "headword": headword,
        "respelling": respelling,
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


def matches_for_side(name: str, ingredients: list[str]) -> list[tuple[str, str]]:
    """[(ingredient, component)] for every DOSE ingredient a headed name's
    components name -- several when the block's clip covers more than one
    name at once."""
    out = []
    for component in audio_sources_clincalc.split_names(name):
        for ingredient in ingredients:
            if audio_sources_clincalc.name_matches(component, ingredient):
                out.append((ingredient, component))
    return out


def process_page(
    slug: str, entry: "audio_sources_clincalc.IndexEntry", ingredients: list[str], types: dict[str, str]
) -> tuple[list[dict], list[dict]]:
    url = audio_sources_clincalc.page_url(slug)
    try:
        html = audio_sources_clincalc.polite_fetch(url, FETCH_DELAY_S)
    except Exception as exc:
        expected = {
            ing
            for n in entry.generics + entry.brands
            for ing in ingredients
            if audio_sources_clincalc.name_matches(n, ing)
        }
        return [], [
            {"ingredient": ing, "reason": "page_fetch_failed", "detail": str(exc)}
            for ing in sorted(expected)
        ]

    page = audio_sources_clincalc.parse_page(html, url)
    sides = [
        ("generic", page.generic_name, page.generic_url, "brand", page.brand_name, page.brand_url),
        ("brand", page.brand_name, page.brand_url, "generic", page.generic_name, page.generic_url),
    ]

    clips = []
    found: set[str] = set()
    seen_clip: set[tuple[str, str]] = set()
    for side, name, side_url, other_side, other_name, other_url in sides:
        if not name or not side_url:
            continue
        for ingredient, _component in matches_for_side(name, ingredients):
            found.add(ingredient)
            if (ingredient, side_url) in seen_clip:
                # ClinCalc's own generic and brand blocks can name and link
                # the exact same clip (aspirin's page heads both "Aspirin"),
                # which is one recording, not two -- keep it once.
                continue
            seen_clip.add((ingredient, side_url))
            components = audio_sources_clincalc.split_names(name)
            clips.append(
                build_clip(
                    ingredient,
                    types[ingredient],
                    name,
                    components,
                    side_url,
                    other_side,
                    other_name,
                    other_url,
                )
            )

    expected = {
        ing
        for n in entry.generics + entry.brands
        for ing in ingredients
        if audio_sources_clincalc.name_matches(n, ing)
    }
    misses = [
        {
            "ingredient": ing,
            "reason": "index_page_mismatch",
            "detail": (
                f"the index lists {ing!r} for HowToPronounce/{slug}, but that "
                "page's own generic/brand headings do not name it"
            ),
        }
        for ing in sorted(expected - found)
    ]
    return clips, misses


def run(html: str | None = None) -> tuple[list[dict], list[dict]]:
    html = html if html is not None else audio_sources_clincalc.fetch(audio_sources_clincalc.INDEX_URL)
    entries = audio_sources_clincalc.parse_index(html)
    types = name_types()
    ingredients = list(types)
    slugs = audio_sources_clincalc.candidate_slugs(entries, ingredients)

    clips, misses = [], []
    for slug in sorted(slugs):
        page_clips, page_misses = process_page(slug, slugs[slug], ingredients, types)
        clips += page_clips
        misses += page_misses

    durations = {c["ingredient"]: c["duration_s"] for c in clips if c["duration_s"] is not None}
    for name, note in audio_verify.syllable_outliers(durations).items():
        for clip in clips:
            if clip["ingredient"] == name:
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
        f"clincalc: {ok} ok, {flagged} flagged, {len(misses)} missing, "
        f"{len(clips)} clips across {len({c['ingredient'] for c in clips})} ingredients"
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

    dupes = audio_manifest.duplicate_groups(merged)
    if dupes:
        print("byte-identical audio shared across ingredients:")
        for h, ings in dupes.items():
            print(f"  {h[:12]}: {ings}")


if __name__ == "__main__":
    main()
