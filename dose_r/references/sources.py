"""Pronunciation sources for the gold reference layer, best authority first.

Coverage reality, measured against the 284 unique DOSE ingredients rather than
assumed:

- Merriam-Webster (medical, then general) carries real lexicographer-assigned
  pronunciations and often several accepted variants. It is the only external
  source that answered: Drugs.com returns 403, and FDA labels via openFDA and
  DailyMed turn out to carry no pronunciation respellings at all.
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
import re
import urllib.parse
import urllib.request
from pathlib import Path

CACHE = Path(__file__).resolve().parent / ".cache"
MW_BASE = "https://www.merriam-webster.com"
UA = {"User-Agent": "Mozilla/5.0 (compatible; DOSE-R research benchmark)"}
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


def merriam_webster(name: str) -> dict | None:
    """MW's respelling for `name`, with the section that answered."""
    hit = _cached(f"mw::{name}")
    if hit is not None:
        return hit or None

    for section in ("medical", "dictionary"):
        raw = _scrape(name, section)
        if raw:
            result = {
                "name": f"merriam-webster/{section}",
                "raw": raw,
                "url": f"{MW_BASE}/{section}/{urllib.parse.quote(name)}",
            }
            _store(f"mw::{name}", result)
            return result

    _store(f"mw::{name}", {})
    return None


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
