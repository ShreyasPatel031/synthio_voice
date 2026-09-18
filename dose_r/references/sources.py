"""Pronunciation sources for the gold reference layer, best authority first.

Coverage reality, measured against the 284 unique DOSE ingredients rather than
assumed:

- Merriam-Webster Medical Dictionary API (structured, needs `MW_MEDICAL_KEY`
  in the environment/`.env`) first, then the HTML `/dictionary/` (general)
  page as a fallback for brand names that only the general dictionary lists.
  The API alone answers fewer names than the old HTML scraper (it is
  medical-only, so it misses brand names that entered common usage and only
  show up in the general dictionary), but the two combined beat either alone.
  Drugs.com and DrugBank return 403, and FDA labels via openFDA and DailyMed
  turn out to carry no pronunciation respellings at all.
- Wikipedia and Wiktionary carry real pronunciations too, but not in the
  plaintext extract -- they live in the wikitext as `{{IPAc-en|...}}` /
  `{{IPA|en|...}}` templates (raw IPA) or `{{respell|...}}` templates
  (Wikipedia's own respelling key, see `wiki_notation.py`). Coverage is much
  thinner than Merriam-Webster's -- most DOSE brand names are too new or too
  minor to have an English Wikipedia article at all -- but where it answers
  it is an independent, real, citable source, which is exactly what lets an
  MW entry be corroborated into "high" confidence instead of just "medium".
- CMUdict covers 13 ingredients. Almost every DOSE name is a coined trade or INN
  name that no general dictionary lists, so a pronunciation dictionary is a
  rounding error here, not a backbone.
- Everything left over falls to the rule-based G2P in `judge/fixtures/g2p.py`,
  which is explicitly not an authority and is recorded as low confidence.

Network responses are cached on disk so a rebuild costs nothing and so the
coverage numbers in COVERAGE.md are reproducible without re-fetching.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Populate os.environ from a simple KEY=VALUE `.env`, without adding a
    python-dotenv dependency. Never overwrites a variable already set."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        os.environ.setdefault(key, value)


_load_dotenv()

CACHE = Path(__file__).resolve().parent / ".cache"
MW_BASE = "https://www.merriam-webster.com"
UA = {"User-Agent": "Mozilla/5.0 (compatible; DOSE-R research benchmark)"}
WIKI_UA = {
    "User-Agent": "DOSE-R-research-bot/1.0 "
    "(https://github.com/ -- gold pronunciation reference layer; "
    "contact: shreyas.patel@searce.com)"
}
TIMEOUT = 25

_PRON = re.compile(r"prons?[^>]*>([^<]{2,80})<")
_USABLE = re.compile(r"[ˈˌə\-]")


def _cache_path(key: str) -> Path:
    safe = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    return CACHE / f"{safe}.json"


def _cached(key: str):
    path = _cache_path(key)
    if path.exists():
        return json.loads(path.read_text())
    return None


def _store(key: str, value) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    _cache_path(key).write_text(json.dumps(value))


def _scrape(name: str, section: str) -> str | None:
    url = f"{MW_BASE}/{section}/{urllib.parse.quote(name)}"
    try:
        req = urllib.request.Request(url, headers=UA)
        html = urllib.request.urlopen(req, timeout=TIMEOUT).read().decode("utf-8", "ignore")
    except Exception:
        return None

    for raw in _PRON.findall(html):
        candidate = raw.replace("&nbsp;", " ").strip()
        if len(candidate) > 3 and _USABLE.search(candidate):
            return candidate
    return None


MW_API_BASE = "https://www.dictionaryapi.com/api/v3/references/medical/json"


def _mw_medical_api(name: str) -> dict | None:
    """MW's Medical Dictionary API, if `MW_MEDICAL_KEY` is set.

    Response is a JSON list of entries (a miss is `[]`, or a list of plain
    strings that are spelling suggestions -- both mean "no answer"). Real
    entries carry `hwi.prs[i].mw`, in MW's own respelling notation, and can
    have more than one `prs` item; every one found is kept, comma-joined, so
    `notation.convert()` expands them exactly as it already does for the
    scraped page.
    """
    key = os.environ.get("MW_MEDICAL_KEY")
    if not key:
        return None

    cache_key = f"mw-api::{name}"
    hit = _cached(cache_key)
    if hit is not None:
        return hit or None

    url = f"{MW_API_BASE}/{urllib.parse.quote(name)}?key={urllib.parse.quote(key)}"
    try:
        req = urllib.request.Request(url, headers=UA)
        raw = urllib.request.urlopen(req, timeout=TIMEOUT).read().decode("utf-8", "ignore")
        entries = json.loads(raw)
    except Exception:
        return None  # not cached: a transient failure shouldn't poison the cache

    respellings = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue  # a bare string is a spelling suggestion, not an entry
        for pr in entry.get("hwi", {}).get("prs", []):
            mw = pr.get("mw")
            if mw:
                respellings.append(mw)

    result = None
    if respellings:
        result = {
            "name": "merriam-webster/medical-api",
            "raw": ", ".join(respellings),
            "url": f"{MW_BASE}/medical/{urllib.parse.quote(name)}",
        }
    _store(cache_key, result or {})
    return result


def merriam_webster(name: str) -> dict | None:
    """MW's respelling for `name`: the Medical API first, then the general
    dictionary's HTML page for names the medical reference does not list."""
    api = _mw_medical_api(name)
    if api:
        return api

    hit = _cached(f"mw-html::{name}")
    if hit is not None:
        return hit or None

    raw = _scrape(name, "dictionary")
    result = None
    if raw:
        result = {
            "name": "merriam-webster/dictionary",
            "raw": raw,
            "url": f"{MW_BASE}/dictionary/{urllib.parse.quote(name)}",
        }
    _store(f"mw-html::{name}", result or {})
    return result


_WIKI_LOCK = threading.Lock()
_WIKI_NEXT_OK = [0.0]
_WIKI_MIN_INTERVAL = 0.4  # be polite: this build runs many words concurrently


def _wiki_throttle() -> None:
    with _WIKI_LOCK:
        wait = _WIKI_NEXT_OK[0] - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _WIKI_NEXT_OK[0] = time.monotonic() + _WIKI_MIN_INTERVAL


def _wiki_wikitext(domain: str, title: str) -> str | None:
    """Raw wikitext of `title` on `domain` (en.wikipedia.org / en.wiktionary.org)."""
    cache_key = f"wikitext::{domain}::{title}"
    hit = _cached(cache_key)
    if hit is not None:
        return hit["text"] or None

    url = (
        f"https://{domain}/w/api.php?action=parse&page="
        f"{urllib.parse.quote(title)}&prop=wikitext&format=json"
    )
    text: str | None = None
    for attempt in range(4):
        _wiki_throttle()
        try:
            req = urllib.request.Request(url, headers=WIKI_UA)
            raw = urllib.request.urlopen(req, timeout=TIMEOUT).read().decode(
                "utf-8", "ignore"
            )
        except Exception:
            text = None
            break

        if "too many requests" in raw.lower() or "rate limit" in raw.lower():
            time.sleep(2**attempt)  # transient 429 from a shared IP; back off
            continue

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            text = None
            break

        if "error" in data:
            text = None  # missingtitle, invalidtitle, etc. -- no such page
        else:
            text = data.get("parse", {}).get("wikitext", {}).get("*")
        break

    _store(cache_key, {"text": text})
    return text


# Longest-content-first: {{IPAc-en|...}} and {{IPA|en|...}} carry real IPA and
# are preferred over {{respell|...}}, which is a lossy lay respelling of the
# same pronunciation and is usually present *alongside* one of the other two.
_IPAC_EN = re.compile(r"\{\{\s*IPAc-en\s*\|([^{}]+)\}\}", re.IGNORECASE)
_IPA_EN = re.compile(r"\{\{\s*IPA\s*\|\s*en\s*\|([^{}]+)\}\}", re.IGNORECASE)
_RESPELL = re.compile(r"\{\{\s*respell\s*\|([^{}]+)\}\}", re.IGNORECASE)


def _args(inner: str) -> list[str]:
    return [a.strip() for a in inner.split("|") if a.strip() and "=" not in a]


def _extract_pronunciation(wikitext: str) -> dict | None:
    """The first usable pronunciation template in a page's wikitext."""
    m = _IPAC_EN.search(wikitext)
    if m:
        args = _args(m.group(1))
        if args:
            return {"kind": "ipa", "raw": "".join(args)}

    m = _IPA_EN.search(wikitext)
    if m:
        args = _args(m.group(1))
        if args:
            return {"kind": "ipa", "raw": args[0]}

    m = _RESPELL.search(wikitext)
    if m:
        args = _args(m.group(1))
        if args:
            return {"kind": "respell", "raw": args}

    return None


def _wiki_source(domain: str, source_name: str, name: str) -> dict | None:
    cache_key = f"wiki-pron::{domain}::{name}"
    hit = _cached(cache_key)
    if hit is not None:
        return hit or None

    title = name[:1].upper() + name[1:] if name else name
    wikitext = _wiki_wikitext(domain, title)
    result = None
    if wikitext:
        pron = _extract_pronunciation(wikitext)
        if pron:
            raw = pron["raw"]
            result = {
                "name": source_name,
                "raw": raw if isinstance(raw, str) else "|".join(raw),
                "kind": pron["kind"],
                "url": f"https://{domain}/wiki/{urllib.parse.quote(title)}",
            }

    _store(cache_key, result or {})
    return result


def wikipedia_pronunciation(name: str) -> dict | None:
    """A real pronunciation from the English Wikipedia article for `name`."""
    return _wiki_source("en.wikipedia.org", "wikipedia", name)


def wiktionary_pronunciation(name: str) -> dict | None:
    """A real pronunciation from the English Wiktionary entry for `name`."""
    return _wiki_source("en.wiktionary.org", "wiktionary", name)


def cmudict_lookup(name: str) -> dict | None:
    """CMUdict ARPABET for every word of `name`, or None if any word is absent."""
    import cmudict

    table = _cmudict_table()
    words = name.lower().split()
    sequences = []
    for word in words:
        entries = table.get(word.strip("-,"))
        if not entries:
            return None
        sequences.append(entries[0])

    return {
        "name": "cmudict",
        "raw": " ".join(" ".join(s) for s in sequences),
        "url": "https://github.com/cmusphinx/cmudict",
    }


_TABLE = None


def _cmudict_table():
    global _TABLE
    if _TABLE is None:
        import cmudict

        _TABLE = cmudict.dict()
    return _TABLE
