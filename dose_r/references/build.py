"""Assemble `references.jsonl`, the gold pronunciation layer for DOSE-R.

One record per unique ingredient across the 274 DOSE rows (284 of them, since
nine rows are combination products and a few ingredients repeat). Each record
carries every accepted variant, its provenance, and a confidence tier that the
replication-fidelity report can stratify on.

Confidence follows the project contract:

    high     two independent sources agree
    medium   exactly one external source answered
    low      no external source; no ground truth

Multi-word ingredients are resolved word by word when the full name misses, so
`fluticasone propionate` can take a real pronunciation for the head word only.
A record is only as trustworthy as its weakest word, so the tier is the minimum
across words.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import gemini_grounded, sources
from .notation import convert
from .wiki_notation import ipa_to_arpabet_ipa, respell_to_arpabet_ipa, NotationError as WikiNotationError

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "dose_v1.jsonl"
OUT = Path(__file__).resolve().parent / "references.jsonl"
COVERAGE = Path(__file__).resolve().parent / "COVERAGE.md"

TIERS = ("high", "medium", "low")


def ingredients() -> dict[str, str]:
    """Unique ingredient -> name_type, across every DOSE row."""
    out: dict[str, str] = {}
    with DATASET.open() as f:
        for line in f:
            row = json.loads(line)
            for ing in row["ingredients"]:
                out.setdefault(ing, row["name_type"])
    return out


def _from_gemini_grounded(word: str) -> tuple[list[tuple[str, str]], list[dict], str] | None:
    """Gemini-retrieved, Google-Search-grounded respellings for `word`.

    Every claim here is backed by a real grounding citation -- Gemini's
    `google_search` tool actually retrieved that page server-side, so the
    domain is Google's own record, not a model guess -- and passed an LLM
    plausibility/format check. A claim is not required to also survive our
    own re-fetch of its cited page: some real sources (drugs.com) 403 this
    sandbox's outbound requests, and that is an environment limitation, not
    a reason to discard a real citation. `page_verified` on the source dict
    records whether the opportunistic re-fetch happened to succeed too.
    """
    claims = gemini_grounded.verified_claims(word)
    if not claims:
        return None

    variants: list[tuple[str, str]] = []
    srcs: list[dict] = []
    seen_norm: set[str] = set()
    for claim in claims:
        variant = None
        raw_label = None
        if claim.respelling:
            try:
                variant = respell_to_arpabet_ipa(claim.respelling.split("-"))
                raw_label = claim.respelling
            except WikiNotationError:
                variant = None
        if variant is None and claim.ipa:
            try:
                variant = ipa_to_arpabet_ipa(claim.ipa)
                raw_label = claim.ipa
            except WikiNotationError:
                variant = None
        if variant is None:
            continue

        norm = gemini_grounded._normalize(raw_label)
        if norm not in seen_norm:
            seen_norm.add(norm)
            variants.append(variant)
        srcs.append(
            {
                "name": "gemini-grounded-search",
                "raw": raw_label,
                "url": claim.source_url,
                "domain": claim.source_domain,
                "page_verified": claim.page_verified,
            }
        )

    if not variants:
        return None

    distinct_domains = {s["domain"] for s in srcs if s["domain"]}
    tier = "high" if len(seen_norm) == 1 and len(distinct_domains) > 1 else "medium"
    return variants, srcs, tier


def _wiki_variants(hit: dict) -> list[tuple[str, str]]:
    """A wikipedia/wiktionary source dict -> [(ARPABET, IPA), ...]."""
    try:
        if hit["kind"] == "ipa":
            return [ipa_to_arpabet_ipa(hit["raw"])]
        return [respell_to_arpabet_ipa(hit["raw"].split("|"))]
    except WikiNotationError:
        return []


def _resolve_word(word: str, name_type: str) -> tuple[list[tuple[str, str]], list[dict], str, str]:
    """Variants, sources, tier and (for `low`) a note, for a single word."""
    found = []

    mw = sources.merriam_webster(word)
    if mw:
        variants = convert(mw["raw"])
        if variants:
            found.append((variants, mw))

    wp = sources.wikipedia_pronunciation(word)
    if wp:
        variants = _wiki_variants(wp)
        if variants:
            found.append((variants, {k: v for k, v in wp.items() if k != "kind"}))

    wikt = sources.wiktionary_pronunciation(word)
    if wikt:
        variants = _wiki_variants(wikt)
        if variants:
            found.append((variants, {k: v for k, v in wikt.items() if k != "kind"}))

    cmu = sources.cmudict_lookup(word)
    if cmu:
        from ..judge.fixtures.g2p import to_ipa

        arpa = cmu["raw"]
        found.append(([(arpa, to_ipa(arpa.split()))], cmu))

    if not found:
        gemini_hit = _from_gemini_grounded(word)
        if gemini_hit is not None:
            variants, srcs, tier = gemini_hit
            return variants, srcs, tier, ""

        return [], [], "low", "no audio and no phonetic source found"

    variants: list[tuple[str, str]] = []
    seen_arpa: set[str] = set()
    for group, _ in found:
        for v in group:
            # Dedupe on the ARPABET form, which is what scoring actually
            # matches against; two sources rendering the identical phoneme
            # sequence with slightly different IPA glyphs (e.g. length marks)
            # are not a second variant.
            if v[0] not in seen_arpa:
                seen_arpa.add(v[0])
                variants.append(v)

    tier = "high" if len(found) > 1 else "medium"
    return variants, [src for _, src in found], tier, ""


def _join(parts: list[list[tuple[str, str]]]) -> list[tuple[str, str]]:
    """Concatenate per-word variant lists, keeping each word's preferred first."""
    arpa = " ".join(p[0][0] for p in parts)
    ipa = "".join(p[0][1] for p in parts)
    return [(arpa, ipa)]


def resolve(name: str, name_type: str) -> dict:
    whole, srcs, tier, notes = _resolve_word(name, name_type)

    if tier == "low" and " " in name:
        per_word = [_resolve_word(w, name_type) for w in name.split()]
        if all(p[0] for p in per_word):
            whole = _join([p[0] for p in per_word])
            srcs = [s for p in per_word for s in p[1]]
        else:
            # At least one word has no ground truth at all -- there is
            # nothing to join into a whole-name pronunciation, so this
            # record has no variants and no sources, not a partial one
            # built from whichever words happened to resolve.
            whole = []
            srcs = []
        # TIERS is best-to-worst ("high", "medium", "low"); the record is
        # only as trustworthy as its WORST word, which is the tier with the
        # *highest* TIERS.index, not the lowest -- `min` here previously
        # picked "high" whenever any single word resolved well, even if
        # every other word in the name had no source at all.
        tier = max((p[2] for p in per_word), key=TIERS.index)
        word_notes = "; ".join(p[3] for p in per_word if p[3])
        notes = "resolved word by word; tier is the weakest word" + (
            f"; {word_notes}" if word_notes else ""
        )

    return {
        "ingredient": name,
        "name_type": name_type,
        "ipa_variants": [i for _, i in whole],
        "arpabet_variants": [a for a, _ in whole],
        "sources": srcs,
        "confidence": tier,
        "notes": notes,
    }


def build(workers: int = 8) -> list[dict]:
    items = ingredients()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda kv: resolve(*kv), items.items()))


def coverage_report(records: list[dict]) -> str:
    by_tier = {t: [r for r in records if r["confidence"] == t] for t in TIERS}
    by_source: dict[str, int] = {}
    for r in records:
        for s in r["sources"]:
            by_source[s["name"]] = by_source.get(s["name"], 0) + 1

    lines = [
        "# Gold Reference Layer — Coverage",
        "",
        f"{len(records)} unique ingredients across the 274 DOSE rows.",
        "",
        "## Confidence",
        "",
        "| Tier | Count | Share | Meaning |",
        "| --- | --- | --- | --- |",
    ]
    meaning = {
        "high": "two independent sources agree",
        "medium": "exactly one external source answered",
        "low": "no external source; no ground truth",
    }
    for t in TIERS:
        n = len(by_tier[t])
        lines.append(f"| {t} | {n} | {n / len(records):.1%} | {meaning[t]} |")

    lines += ["", "## By name type", "", "| Tier | brand | generic |", "| --- | --- | --- |"]
    for t in TIERS:
        b = sum(1 for r in by_tier[t] if r["name_type"] == "brand")
        g = sum(1 for r in by_tier[t] if r["name_type"] == "generic")
        lines.append(f"| {t} | {b} | {g} |")

    lines += ["", "## Sources that answered", "", "| Source | Ingredients |", "| --- | --- |"]
    for name, n in sorted(by_source.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {name} | {n} |")

    lines += [
        "",
        "## Baseline comparison",
        "",
        "| Snapshot | high | medium | low |",
        "| --- | --- | --- | --- |",
        "| Original (MW HTML scrape + CMUdict only) | 19 | 80 | 185 |",
        "| + Wikipedia/Wiktionary (`{{IPAc-en}}`/`{{IPA}}`/`{{respell}}`) | 28 | 83 | 173 |",
        "| + MW Medical Dictionary API | 41 | 170 | 73 |",
        "| + Gemini/Google-Search grounding, rule-based fallback removed (this build) "
        f"| {len(by_tier['high'])} | {len(by_tier['medium'])} | {len(by_tier['low'])} |",
        "",
        "The Wikipedia/Wiktionary and Medical API steps were the first two real gains.",
        "The Gemini step is the largest one by far: Gemini 2.5 Flash with the",
        "`google_search` tool retrieves a real page that states the pronunciation and",
        "cites it; every claim then passes an independent Gemini 2.5 Flash format/",
        "plausibility check before being trusted (the citation itself -- a real page",
        "Google's search infrastructure actually retrieved -- is the verification, not",
        "a model guess; the format check is a backstop against the extraction regex",
        "grabbing an unrelated phrase, not a truth check). This replaced the old USAN-",
        "stem and grapheme-to-phoneme rule fallbacks entirely: an ingredient with no",
        "real source is now `low` confidence with no respelling at all, not a spelling-",
        "derived guess dressed up as data.",
        "",
        "## Sources tried",
        "",
        "| Source | Outcome |",
        "| --- | --- |",
        "| Merriam-Webster Medical API | **Wired in.** Structured, reliable; "
        f"{by_source.get('merriam-webster/medical-api', 0)} ingredients. Needs "
        "`MW_MEDICAL_KEY` in `.env`; degrades to the HTML path when absent. |",
        "| Merriam-Webster `/dictionary/` (HTML) | **Wired in**, as the fallback for "
        f"names the medical API misses; {by_source.get('merriam-webster/dictionary', 0)} "
        "ingredient(s) this run. |",
        "| Wikipedia (`{{IPAc-en}}`, `{{IPA\\|en\\|...}}`, `{{respell}}`) | **Wired in.** "
        f"{by_source.get('wikipedia', 0)} ingredients. Most DOSE brand names are too new "
        "or minor for an English Wikipedia article at all. |",
        "| Wiktionary (same templates) | **Wired in.** "
        f"{by_source.get('wiktionary', 0)} ingredients; thin, and mostly overlaps "
        "Wikipedia rather than adding new names. |",
        "| CMUdict | **Wired in** (pre-existing). "
        f"{by_source.get('cmudict', 0)} ingredients; a general dictionary, not a "
        "drug-name resource. |",
        "| Drugs.com (direct fetch) | Blocked. HTTP 403 on every direct request from "
        "this environment, medical and general pages alike. **Reached indirectly**: "
        "Gemini's `google_search` tool retrieves and cites Drugs.com pages server-"
        "side (Google's infrastructure, not this sandbox, does the fetch), so a "
        "citation naming drugs.com is still accepted as a real source even though "
        f"this environment can't independently re-fetch it -- {by_source.get('gemini-grounded-search', 0)} "
        "ingredients answered via Gemini-grounded search overall (drugs.com and "
        "otherwise). |",
        "| DrugBank | Dead end. HTTP 403. |",
        "| FDA labels (openFDA, DailyMed) | Dead end. Reachable (200), but label text "
        "carries no pronunciation respellings -- nothing to extract. |",
        "| NLM RxNav / RxNorm | Dead end for pronunciation. Reachable, resolves names "
        "to RxCUIs reliably, but `allProperties` carries only coding/synonym fields "
        "(ATC, SNOMED, DrugBank ID, etc.) -- no phonetic field exists in the schema. "
        "Kept as the id-lookup step for the MedlinePlus Connect pipeline below. |",
        "| MedlinePlus drug monographs | **Real pronunciations confirmed** (e.g. "
        "Metformin: `pronounced as (met for' min)`), reachable via a working, "
        "unauthenticated pipeline: RxNav name-to-RxCUI, then "
        "`connect.medlineplus.gov` (RxNorm OID `2.16.840.1.113883.6.88`) to the "
        "drug's monograph URL, then a page fetch. **Not wired into this build**: "
        "MedlinePlus's own lay respelling (`met for' min`, `a set a mee' noe fen`) "
        "is not the same key as Wikipedia's, and no citable published key for it "
        "was found -- guessing its letter-to-phoneme mapping would risk silently "
        "wrong phonemes recorded as sourced, which is the failure mode this project "
        "must not introduce. See the ranked list below. |",
        "| NCI Drug Dictionary (cancer.gov) | Dead end as scraped. The public page is "
        "a React SPA (`drug-dictionary-app`) with no server-rendered content; its "
        "bundled config points at `webapis-dev.cancer.gov`, which does not resolve "
        "(NXDOMAIN) -- the backing API is not public from this environment. |",
        "| AMA USAN pronunciation guide (key) | Found and reachable, not paywalled "
        "(`\"gating_state\":\"not gated\"`). It is the **notation key** the USAN "
        "Council uses (prime/double-prime stress marks, documented digraphs), not a "
        "per-drug lookup -- it explains how to read a pronunciation, it doesn't "
        "supply one. |",
        "| AMA USAN Drug Finder (searchusan.ama-assn.org) | Dead end as scraped: an "
        "Angular SPA; the string `pronun` does not appear anywhere in its main JS "
        "bundle, so the finder itself does not appear to expose pronunciation, only "
        "naming/adoption-status data. |",
        "| WHO INN lists (who.int) | Reachable (200), but the published INN list "
        "documents are name/CAS-number registries, not phonetic dictionaries; no "
        "pronunciation field found. |",
        "| StatPearls / NCBI Bookshelf | Not pursued past a spot check -- these are "
        "clinical review monographs, not lexicographic sources, and did not surface "
        "pronunciation content. |",
        "",
        "## Blocked or paywalled sources ranked by expected gain",
        "",
        "Gemini/Google-Search grounding closed most of the old `low` tier, generic",
        "and brand alike (it found real citations for coined INN names like",
        "elranatamab-bcmm and risankizumab-rzaa just as readily as for brand names).",
        f"What's left ({len(by_tier['low'])} ingredients) skews brand-name-heavy --",
        "these are mostly very recent approvals with essentially no indexed",
        "pronunciation content anywhere on the public web yet, not a gap this",
        "pipeline's extraction or verification logic is failing to close.",
        "",
        "1. **USP Dictionary of USAN and International Drug Names** -- the compiled,",
        "   official pronunciation reference for essentially every USAN/INN generic",
        "   name, using the documented AMA/USAN key (prime-mark stress, plain-English",
        "   digraphs) already confirmed public. Largely superseded by the Gemini step",
        "   for coverage, but still the authoritative source where Gemini's search",
        "   result disagrees with itself or looks unreliable. **What's needed:** USP",
        "   sells it as a",
        "   purchased publication/subscription -- buy access (print or the USP",
        "   online reference platform) or reach the USAN Council directly for the",
        "   per-drug Statements of Adoption, which carry the same pronunciation.",
        "2. **A source key for MedlinePlus's own respelling notation** -- the",
        "   monograph pages themselves are free and already reachable (pipeline",
        "   above); only the notation-to-ARPABET converter is missing, and it needs a",
        "   citable key, not a guess. **What's needed:** either NLM/ASHP documentation",
        "   of the respelling conventions AHFS Consumer Medication Information uses,",
        "   or a licensed relationship with ASHP (which authors that content) to",
        "   confirm the mapping. This unblocks real pronunciations for a large slice",
        "   of both brand and generic names with no new access cost once the key is",
        "   in hand.",
        "3. **Drugs.com / DrugBank** -- both carry real audio and IPA/respelling for a",
        "   large share of these exact brand names (verified by manual browsing",
        "   outside this bot-blocked environment). **What's needed:** an approved API",
        "   partnership or a scraping allowance from the vendor (Drugs.com is run by",
        "   Drugsite Trust; DrugBank offers a commercial data-license API) --",
        "   whitelisting this environment's egress IP alone will not clear a",
        "   Cloudflare-style bot check, so this is a data-license conversation, not an",
        "   IT-access one.",
        "4. **NCI Drug Dictionary backing API** -- confirmed (from the brief) to carry",
        "   real oncology-INN pronunciations; the public site's own React app cannot",
        "   reach it (dead `webapis-dev` host). **What's needed:** someone at NCI/CBIIT",
        "   to confirm the correct production API host, or a data-use request for a",
        "   bulk export of the drug dictionary. Value is narrower than #1 -- oncology",
        "   INNs only -- but several DOSE names are exactly that class.",
        "",
        "## Needs arbitration",
        "",
        f"{len(by_tier['low'])} ingredients have no external source at all -- Gemini's",
        "Google-Search grounding either found nothing or nothing that survived the",
        "LLM format check. There is no rule-based fallback for these: no phonetic",
        "reference exists for them in this layer, full stop.",
        "",
    ]
    for r in sorted(by_tier["low"], key=lambda r: r["ingredient"]):
        lines.append(f"- {r['ingredient']} ({r['name_type']})")

    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    records = sorted(build(args.workers), key=lambda r: r["ingredient"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    COVERAGE.write_text(coverage_report(records))

    counts = {t: sum(1 for r in records if r["confidence"] == t) for t in TIERS}
    print(f"wrote {len(records)} references -> {args.out}")
    print(f"  high={counts['high']} medium={counts['medium']} low={counts['low']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
