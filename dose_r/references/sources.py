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
- DailyMed (NLM's public mirror of FDA-approved Structured Product Labeling)
  turns out NOT to be a dead end after all, correcting an earlier claim in
  this docstring: many Medication Guides state the brand's own phonetic
  respelling right in their title line, e.g. `AMBELVIST (am bel' vist)`, in
  the same USAN prime-stress notation `gemini_grounded.py` already parses.
  openFDA's structured label JSON and the raw SPL XML do not carry this line
  (it is only in the rendered guide text), which is presumably why the
  earlier openFDA-only check missed it; DailyMed's own rendered HTML does,
  and unlike drugs.com/DrugBank it is reachable from this environment.
- Everything left over has no rule-based fallback standing in for it: an
  ingredient no source above (or Gemini-grounded search, see
  `gemini_grounded.py`) answers for is `low` confidence with no
  respelling at all, not a guess dressed up as data.

Network responses are cached on disk so a rebuild costs nothing and so the
coverage numbers in COVERAGE.md are reproducible without re-fetching.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

# pypdf logs a "fontTools is required..." warning per unusual font per page
# for USAN's PDFs -- harmless (extract_text() still works without it), but
# floods build output otherwise.
logging.getLogger("pypdf").setLevel(logging.ERROR)

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
_BOLD_TERM = re.compile(r"'''([^']+)'''")
# `[ \t]*`, not `\s*`, after "=" -- a `\s*` here crosses the newline when
# the field is empty ("| pronounce          =\n| tradename ...") and
# silently captures the NEXT infobox line as if it were this field's own
# value, confirmed as a real bug on empagliflozin's own article (whose
# `pronounce` field is genuinely empty; a `\s*` version captured
# "| tradename          = Jardiance, others" as the "pronunciation").
_INFOBOX_PRONOUNCE = re.compile(r"\|\s*pronounce\s*=[ \t]*([^\n]*)")
# {{Infobox drug}} repurposes its `source` field (normally the antibody's
# source-organism code -- "u"/"o"/"xi"/"zu" for human/murine/chimeric/
# humanized) to carry the PRONUNCIATION instead when `mab_type` is set,
# i.e. for monoclonal antibodies. Confirmed directly on secukinumab's own
# article: `| source = {{IPAc-en|...}}<br />{{respell|...}}` sits above
# the "Clinical data" section, while its own `pronounce` field (inside
# "Clinical data") is empty -- the same empty-`pronounce` shape as
# empagliflozin, but here the real pronunciation was never missing, just
# filed under a different key. Gated on `mab_type` appearing anywhere on
# the page so a coincidental `source = ...` elsewhere (e.g. a citation
# template) on a non-antibody article is never mistaken for this field.
_INFOBOX_MAB_SOURCE = re.compile(r"\|\s*source\s*=[ \t]*([^\n]*)")


def _args(inner: str) -> list[str]:
    return [a.strip() for a in inner.split("|") if a.strip() and "=" not in a]


def _pronunciation_region(wikitext: str, target: str, domain: str) -> str:
    """The slice of `wikitext` that may legitimately carry `target`'s own
    pronunciation template, not some other bolded name's.

    A drug's lead sentence commonly bolds and parenthesizes a pronunciation
    for BOTH its names, one right after the other ("'''Empagliflozin''',
    sold under the brand name '''Jardiance''' ({{IPAc-en|...}})") -- a
    prose-wide search finds Jardiance's, not empagliflozin's own, because
    the generic name in that lead sentence has no pronunciation guide of
    its own at all, only the brand does (confirmed directly: its own
    infobox `pronounce` field, see `_INFOBOX_PRONOUNCE`, is empty).
    Restricting the search to the text between `target`'s own bolded
    mention and the NEXT bolded term (a different name) keeps a template
    from being attributed to the wrong word just because it's the first
    one on the page. This is the fallback for a page with no (or no
    populated) infobox `pronounce` field, not the primary path -- that
    field, when present and non-empty, is Wikipedia's own explicit,
    unambiguous label for the article subject's own pronunciation and
    needs no positional inference at all.

    This restriction is Wikipedia-specific and actively harmful on
    Wiktionary: confirmed directly on Benadryl's Wiktionary entry, whose
    `===Pronunciation===` section sits right after the etymology near the
    top of the page with no bolded "Benadryl" anywhere nearby (Wiktionary
    has no {{Infobox drug}}-style lead-sentence bolding convention at
    all) -- the restriction instead found an unrelated LATER bolded
    mention of "Benadryl" inside a "Benadryl challenge" trivia section and
    started the search window after it, skipping the real pronunciation
    entirely. Wiktionary is also one page per exact spelling, so it has
    no brand/generic dual-naming ambiguity for this restriction to guard
    against in the first place -- a non-Wikipedia domain just searches
    the whole page.
    """
    if domain != "en.wikipedia.org":
        return wikitext

    target_norm = _norm_for_match(target)
    bolds = list(_BOLD_TERM.finditer(wikitext))
    target_bold = next((m for m in bolds if _norm_for_match(m.group(1)) == target_norm), None)
    if target_bold is None:
        # `target` isn't bolded in its own article's lead at all (unusual,
        # but seen for some redirects/stubs) -- fall back to the whole page
        # rather than finding nothing, since there's no other name's
        # bolding to accidentally prefer over.
        return wikitext

    start = target_bold.end()
    next_other_bold = next(
        (m for m in bolds if m.start() > start and _norm_for_match(m.group(1)) != target_norm),
        None,
    )
    end = next_other_bold.start() if next_other_bold else len(wikitext)
    return wikitext[start:end]


def _templates_in(region: str) -> dict | None:
    m = _IPAC_EN.search(region)
    if m:
        args = _args(m.group(1))
        if args:
            return {"kind": "ipa", "raw": "".join(args)}

    m = _IPA_EN.search(region)
    if m:
        args = _args(m.group(1))
        if args:
            return {"kind": "ipa", "raw": args[0]}

    m = _RESPELL.search(region)
    if m:
        args = _args(m.group(1))
        if args:
            return {"kind": "respell", "raw": args}

    return None


def _extract_pronunciation(wikitext: str, target: str, domain: str) -> dict | None:
    """`target`'s own pronunciation template: the infobox `pronounce` field
    when it's present and non-empty (Wikipedia's own explicit label for
    the article subject's pronunciation, so no attribution guesswork
    needed at all), else the mAb-infobox `source` field (see
    `_INFOBOX_MAB_SOURCE`), else the prose fallback -- see
    `_pronunciation_region`.
    """
    infobox = _INFOBOX_PRONOUNCE.search(wikitext)
    if infobox and infobox.group(1).strip():
        found = _templates_in(infobox.group(1))
        if found:
            return found

    if "mab_type" in wikitext:
        mab_source = _INFOBOX_MAB_SOURCE.search(wikitext)
        if mab_source and mab_source.group(1).strip():
            found = _templates_in(mab_source.group(1))
            if found:
                return found

    return _templates_in(_pronunciation_region(wikitext, target, domain))


def _wiki_source(domain: str, source_name: str, name: str) -> dict | None:
    cache_key = f"wiki-pron::{domain}::{name}"
    hit = _cached(cache_key)
    if hit is not None:
        return hit or None

    title = name[:1].upper() + name[1:] if name else name
    wikitext = _wiki_wikitext(domain, title)
    result = None
    if wikitext:
        pron = _extract_pronunciation(wikitext, name, domain)
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


DAILYMED_SEARCH = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
DAILYMED_LOOKUP = "https://dailymed.nlm.nih.gov/dailymed/lookup.cfm"

# The respelling always sits in parentheses right after the drug's own name
# in a Medication Guide's title line, e.g. "AMBELVIST (am bel' vist)" --
# match the name, then require the parenthetical to have *some* internal
# structure (a space, hyphen, or stress prime) so a bare repeat of the
# generic name in parens right after ("AMBELVIST (gadoquatrane)") isn't
# mistaken for one. A registered/trademark mark often sits between the name
# and the parenthetical too (`YUVIWEL ® (YOU-vih-well)`, once its own
# `<span>` tag is stripped down to the bare glyph) -- `[®™\s]*` absorbs it.
#
# The inner character class originally left out `-`, which is the single
# most common respelling separator (`YOU-vih-well`, `ky-ZAH-treks`,
# `rev-tor-pik`) -- every hyphenated respelling silently failed to match at
# all, while a single-word or space-only parenthetical (a repeated generic
# name, a dosing note) matched fine and masked the miss.
#
# Not every Medication Guide uses parentheses either -- `LYNAVOY [LIN-ah-
# voy]` brackets it -- so both delimiters are tried, each requiring its own
# matching close so `(foo]` can't match.
def _dailymed_respell_pattern(name: str) -> re.Pattern:
    inner = r"[a-zA-Z][a-zA-Z\"'’ \-]{1,60}?"
    return re.compile(
        rf"\b{re.escape(name)}[®™\s]*(?:\(\s*({inner})\s*\)|\[\s*({inner})\s*\])",
        re.IGNORECASE,
    )


def _dailymed_setids(name: str) -> list[str]:
    cache_key = f"dailymed-search::{name}"
    hit = _cached(cache_key)
    if hit is not None:
        return hit.get("setids", [])

    setids: list[str] = []
    try:
        url = f"{DAILYMED_SEARCH}?drug_name={urllib.parse.quote(name)}"
        req = urllib.request.Request(url, headers=UA)
        raw = urllib.request.urlopen(req, timeout=TIMEOUT).read().decode("utf-8", "ignore")
        setids = [d["setid"] for d in json.loads(raw).get("data", []) if d.get("setid")]
    except Exception:
        pass
    _store(cache_key, {"setids": setids})
    return setids


def dailymed_pronunciation(name: str) -> dict | None:
    """The phonetic respelling many FDA Medication Guides state right after
    the drug's own name (`AMBELVIST (am bel' vist)`), pulled from DailyMed.

    Only tried for brand names: a Medication Guide's title line is
    `BRAND (respelling)` followed by `(generic name)` on its own line --
    matching a generic name here would just find that second, unrelated
    parenthetical, not a pronunciation of the generic itself.

    The "has a space or a prime" structural filter alone isn't enough: a
    table cell like `YUVIWEL (gross content per vial)` also has a space and
    would otherwise pass. Every structurally-plausible candidate is checked
    against `gemini_grounded._judge_format`, the same LLM plausibility
    backstop the Gemini-grounded path uses, before being accepted.
    """
    cache_key = f"dailymed-pron::{name}"
    hit = _cached(cache_key)
    if hit is not None:
        return hit or None

    import html as html_module  # stdlib entity decoder; shadowed by no local var here

    from . import gemini_grounded  # local import: keeps this a soft, in-package dependency

    pattern = _dailymed_respell_pattern(name)
    result = None
    for setid in _dailymed_setids(name):
        url = f"{DAILYMED_LOOKUP}?setid={setid}"
        try:
            req = urllib.request.Request(url, headers=UA)
            raw_html = urllib.request.urlopen(req, timeout=TIMEOUT).read().decode("utf-8", "ignore")
        except Exception:
            continue

        # A registered-trademark mark sits between the name and its
        # respelling as its own tag (`YUVIWEL<span class="Sup">®</span>
        # (YOU-vih-well)`), which silently defeated a regex applied to the
        # raw markup -- stripping tags first (and decoding entities so a
        # curly apostrophe inside the respelling itself, e.g. an escaped
        # `&#8217;`, matches the plain apostrophe this pattern expects)
        # finds it.
        text = html_module.unescape(re.sub(r"<[^>]+>", " ", raw_html))

        # The brand name alone (no respelling) recurs throughout the body
        # text ("AMBELVIST (gadoquatrane) injection is..."), so the FIRST
        # match in the document is usually not the one with a respelling --
        # that one lives in the Medication Guide's title line, further
        # down. Scan every match with real structure and judge each one.
        for m in pattern.finditer(text):
            respelling = (m.group(1) or m.group(2)).strip()
            if not any(c in respelling for c in (" ", "-", "'", "’")):
                continue
            if gemini_grounded._judge_format(name, respelling, None):
                result = {"name": "dailymed", "raw": respelling, "url": url}
                break
        if result:
            break

    _store(cache_key, result or {})
    return result


NCI_GLOSSARY_BASE = "https://webapis.cancer.gov/glossary/v1/Terms/Cancer.gov/Patient/en"


def nci_pronunciation(name: str) -> dict | None:
    """A drug's own respelling and audio recording from the NCI Dictionary
    of Cancer Terms (nih.gov's National Cancer Institute) -- a federal
    government reference on the same standing as DailyMed or MedlinePlus,
    not a secondhand citation of one.

    Its own site (`cancer.gov/publications/dictionaries/cancer-terms/def/
    {name}`) is a React SPA with no server-rendered pronunciation text at
    all -- a dead end as scraped, same as the NCI Drug Dictionary already
    documented as such in this module's docstring. Its real backing API
    isn't a dead end, though: reverse-engineered from the app's own JS
    bundle (`termDefinition: "/Terms/${dictionary}/${audience}/${language}"`,
    `dictionaryEndpoint: "https://webapis.cancer.gov/glossary/v1/"`), it
    returns a structured JSON record per term with its own
    `pronunciation.key` (a capitalized-syllable, hyphenated respelling
    already in the same notation `respell_to_arpabet_ipa` parses, e.g.
    "(uh-see-tuh-MIH-nuh-fen)" for acetaminophen) and a real hosted
    `pronunciation.audio` recording (`nci-media.cancer.gov`) -- both
    stored here, even though only `key` is converted into a phonetic
    variant; the audio URL is kept in the citation as a second,
    independently checkable form of the same official record.

    Coverage here is narrow and oncology-skewed on purpose: this is a
    *cancer* terms dictionary, so it only carries the subset of DOSE's
    ingredients that come up in oncology/supportive-care contexts
    (confirmed directly: a name like "valsartan" or "clopidogrel" isn't
    merely spelled differently in this index, `Autosuggest` finds nothing
    resembling it at all -- it's a real, expected gap, not a bug in how
    this source is queried). Biosimilar suffixes are stripped the same
    way `usan_pronunciation` strips them ("bevacizumab", not
    "bevacizumab-vikg") -- this dictionary indexes the parent generic
    name only.
    """
    from . import usan_stems  # local import: avoids a hard, one-way dependency

    slug, _ = usan_stems._split_fda_suffix(name.lower().replace(" ", "-"))
    cache_key = f"nci-pron::{slug}"
    hit = _cached(cache_key)
    if hit is not None:
        return hit or None

    url = f"{NCI_GLOSSARY_BASE}/{urllib.parse.quote(slug)}"
    result = None
    try:
        req = urllib.request.Request(url, headers={**UA, "Accept": "application/json"})
        raw = urllib.request.urlopen(req, timeout=TIMEOUT).read().decode("utf-8", "ignore")
        data = json.loads(raw)
        key = (data.get("pronunciation") or {}).get("key") or ""
        respelling = key.strip().strip("()").strip()
        if respelling:
            result = {
                "name": "nci-dictionary-of-cancer-terms",
                "raw": respelling,
                "audio": data.get("pronunciation", {}).get("audio"),
                "url": f"https://www.cancer.gov/publications/dictionaries/cancer-terms/def/{slug}",
            }
    except Exception:
        pass

    _store(cache_key, result or {})
    return result


USAN_SEARCH_BASE = "https://searchusan.ama-assn.org/usan/search"
USAN_DOC_BASE = "https://searchusan.ama-assn.org/usan/documentDownload"

# "PRONUNCIATION" is followed by the respelling and then the next section
# header, reliably "THERAPEUTIC CLAIM" in every USAN Statement on file --
# confirmed directly against real documents for cipepofol and copper
# histidinate (see the PDFs this source is built from). USAN's own PDF for
# oveporexton has a typo in the header itself ("PRONOUNCIATION", confirmed
# directly against that document), which an exact-string match on
# "PRONUNCIATION" silently missed entirely -- `PRONO?UNCIATION` tolerates
# either spelling.
# pypdf's text extraction sometimes breaks a word across a line boundary
# mid-word, not just between words -- confirmed directly in empagliflozin's
# own Statement, which extracts as "...THERAPEUTIC CLAI\nM Treatment...".
# An exact "THERAPEUTIC CLAIM" match silently misses this, and silently is
# the operative word: it looks exactly like "this document has no
# PRONUNCIATION field" or "this name has no USAN document", not a parse
# failure, so it would never have surfaced without deliberately auditing
# every generic this source claims to have found nothing for. `_loose`
# tolerates whitespace (including a newline) appearing before any letter
# of the phrase, so a mid-word break like this can't defeat the match.
def _loose(phrase: str) -> str:
    return "".join(rf"\s*{re.escape(c)}" if c != " " else r"\s+" for c in phrase)


# oveporexton's own Statement has a typo in the header itself
# ("PRONOUNCIATION", confirmed directly against that document) -- an
# extra "O" before the "U" ("PRON-OU-NCIATION" vs the correctly-spelled
# "PRON-U-NCIATION"). The optional "O" tolerates either spelling on top of
# `_loose`'s tolerance for a mid-word line break.
_USAN_PRONUNCIATION = re.compile(
    rf"{_loose('PRON')}(?:{_loose('O')})?{_loose('U')}{_loose('NCIATION')}"
    rf"\s*(.+?)\s*{_loose('THERAPEUTIC CLAIM')}",
    re.DOTALL | re.IGNORECASE,
)


def _norm_for_match(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _is_salt_form_of(target: str, candidate: str) -> bool:
    """True when `candidate` (a search result's title or filename slug) is
    exactly `target` followed by a salt/ester qualifier word -- e.g.
    "VARENICLINE  TARTRATE" or the slug "ibuprofen-sodium" for a query of
    "varenicline"/"ibuprofen". USAN files a single Statement per salt
    form even for a plain generic that has no separate USAN entry of its
    own (confirmed directly: searching "varenicline" alone finds only
    "varenicline_tartrate.pdf", never a bare "varenicline.pdf"), and that
    Statement's own PRONUNCIATION field only ever respells the coined
    stem itself, never the salt qualifier, which is already an ordinary,
    independently pronounceable English/chemistry word ("tartrate",
    "sodium", "hydrochloride") -- see `_first_word_group`, which isolates
    just the stem's own portion of that combined field. An exact prefix
    match this specific is a categorically stronger signal than the
    general similarity score in `usan_pronunciation`, so it's accepted
    even when the appended qualifier drags the overall ratio under that
    function's 0.85 acceptance threshold.
    """
    cleaned = re.sub(r"[\s_-]+", " ", candidate.strip().lower())
    target_norm = re.sub(r"[\s_-]+", " ", target.strip().lower())
    if not cleaned.startswith(target_norm):
        return False
    rest = cleaned[len(target_norm) :]
    return rest == "" or rest[0] == " "


def _first_word_group(raw: str) -> str | None:
    """The leading word's own syllable-token run, isolated from a raw
    PRONUNCIATION field that actually spans more than one word (a
    salt-form Statement's own combined respelling for both its stem and
    its salt qualifier, e.g. "ar” i pip’ ra zole  lawr ox’ il" for
    "aripiprazole lauroxil").

    There is no single consistent separator between the two words'
    syllable runs -- confirmed against real documents with two different
    conventions: a single space between syllables of the same word and a
    double space at the true word boundary (aripiprazole lauroxil), and a
    uniform double space between EVERY syllable with only the one true
    boundary bumped up to a triple space (esomeprazole strontium). Either
    way the boundary is reliably the FIRST whitespace run strictly wider
    than the modal (most common) gap elsewhere in the string -- also
    confirmed against a three-word case (ibuprofen trelamine
    hydrochloride: modal gap 1, both real word boundaries showing gap 2,
    the first one landing exactly after "fen", i.e. after "ibuprofen").

    Returns None when the string has no second word at all, or when
    every gap is the same width (no distinguishable boundary to trust) --
    callers must treat that as "can't isolate the stem safely" and not
    guess, not as "there is no boundary".
    """
    parts = re.split(r"( +)", raw.strip())
    words = parts[0::2]
    gaps = [len(g) for g in parts[1::2]]
    if len(words) < 2 or not gaps:
        return None

    mode_gap, mode_count = Counter(gaps).most_common(1)[0]
    if mode_count == len(gaps):
        return None

    boundary = next((i for i, g in enumerate(gaps) if g > mode_gap), None)
    if boundary is None:
        return None
    return " ".join(words[: boundary + 1])


def _split_at_word_boundaries(raw: str, n_words: int) -> list[str] | None:
    """`raw`'s own syllable-token run, split into `n_words` groups -- the
    generalization of `_first_word_group`'s "the boundary is the widest gap"
    signal to a multi-word ingredient name USAN's own Statement respells in
    full (not just a salt/ester qualifier tacked onto one word).

    This is what makes it possible to store a stored `raw` from a *plain*
    multi-word generic's own combined USAN respelling (e.g. "copper
    histidinate" -> "kop' er  his' ti di nate", `dimethyl fumarate` ->
    "dye  meth' il     fue' ma  rate") with its two words' phonemes joined
    by a real space in the final IPA -- confirmed the same gap-width signal
    survives across every real multi-word generic in this dataset that
    USAN answers in full: each has exactly `n_words - 1` gaps clearly wider
    than the rest, in left-to-right order, at exactly the true word
    boundaries. Returns None rather than guessing when that isn't true
    (fewer than `n_words - 1` gaps stand out, or a name has only one word),
    the same conservative rule `_first_word_group` already uses.
    """
    if n_words < 2:
        return None
    parts = re.split(r"( +)", raw.strip())
    words = parts[0::2]
    gaps = [len(g) for g in parts[1::2]]
    if len(words) < n_words or not gaps:
        return None

    mode_gap, mode_count = Counter(gaps).most_common(1)[0]
    if mode_count == len(gaps):
        return None

    boundaries = sorted(i for i, g in enumerate(gaps) if g > mode_gap)
    if len(boundaries) != n_words - 1:
        return None

    groups, start = [], 0
    for b in boundaries:
        groups.append(" ".join(words[start : b + 1]))
        start = b + 1
    groups.append(" ".join(words[start:]))
    return groups


def _collapse_pronunciation_whitespace(raw_full: str, name: str) -> str:
    """`raw_full`'s own whitespace, collapsed for storage -- a real word
    boundary is kept as a double space when `name` is multi-word and
    `_split_at_word_boundaries` can find one for every word, so
    `build._respelling_text_to_variant` can join each word's own IPA with
    a real space instead of running a multi-word ingredient's phonemes
    together with no boundary at all (confirmed a real, silent gap on
    every multi-word generic USAN answers in full, e.g. "copper
    histidinate" -> "kɒpɛrhɪstɪdɪneɪt", one unreadable run-on word,
    before this fix). Every other run of whitespace, within a word or
    when no boundary could be found, collapses to a single space same as
    before -- this never invents a boundary it isn't confident about.
    """
    words = name.split()
    if len(words) >= 2:
        groups = _split_at_word_boundaries(raw_full.strip(), len(words))
        if groups:
            return "  ".join(re.sub(r"\s+", " ", g).strip() for g in groups)
    return re.sub(r"\s+", " ", raw_full).strip()


def _doc_slug(document_uri: str) -> str:
    """A search result's `document_uri` -> its bare filename, decoded and
    without the `.pdf` extension -- more reliable to match a query against
    than the free-text `title` field (see `usan_pronunciation`'s
    docstring), since the filename consistently follows the
    "{name}.pdf"/"{name}-{suffix}.pdf" convention while a title can carry
    extra descriptive words or, rarely, a typo.
    """
    name = urllib.parse.unquote(document_uri.rsplit("/", 1)[-1])
    return re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE)


def _usan_search(term: str) -> list[dict]:
    """Raw hits from the real full-text search API behind the (otherwise
    dead-end, Angular-rendered) searchusan.ama-assn.org UI -- reverse-
    engineered from its own JS bundle (`this.searchUrl = "/" +
    this.collection + "/search/" + term + "/" + sort + "/" + pageNum`,
    `collection` defaulting to `"usan"`), not guessed at from the app's
    rendered HTML. It's a MarkLogic index (`cts:search` shows up in the
    response's own `report` field) that returns each match's real
    `document-uri` and `title` directly -- no filename to guess, and no
    HTTP-200-with-an-error-body ambiguity the document-download endpoint
    alone has.
    """
    cache_key = f"usan-search::{term}"
    hit = _cached(cache_key)
    if hit is not None:
        return hit.get("results", [])

    results: list[dict] = []
    try:
        url = f"{USAN_SEARCH_BASE}/{urllib.parse.quote(term)}/relevant/1"
        req = urllib.request.Request(url, headers=UA)
        raw = urllib.request.urlopen(req, timeout=TIMEOUT).read().decode("utf-8", "ignore")
        data = json.loads(raw)
        for r in data.get("results", []):
            content = r.get("extracted", {}).get("content", [])
            title = next((c["title"] for c in content if "title" in c), None)
            doc_uri = next((c["document-uri"] for c in content if "document-uri" in c), None)
            if title and doc_uri:
                results.append({"title": title, "document_uri": doc_uri})
    except Exception:
        pass

    _store(cache_key, {"results": results})
    return results


def usan_pronunciation(name: str) -> dict | None:
    """The official USAN Statement's own PRONUNCIATION field, found via
    AMA's real search index rather than guessed at (an earlier version of
    this function tried constructing the document filename directly --
    e.g. "elranatamab.pdf" -- which is fragile: the real file for that name
    turns out to be "elranatamab-.pdf", a trailing-hyphen quirk that isn't
    guessable and that blind pattern-trying missed for several names this
    function now finds correctly by searching instead).

    Generic names only: this is the USAN Council's own per-drug adopted-
    name record (`si pep' oh fol` for cipepofol, `kop' er his' ti di nate`
    for copper histidinate). Searching still needs the FDA biosimilar
    suffix stripped first ("risankizumab", not "risankizumab-rzaa") -- the
    index has no entry for the suffixed form at all, confirmed directly
    (0 results either way this function tries it).

    A search can return multiple documents (a plain name's own statement
    *and* a salt/hydrate variant's -- "troriluzole" and "troriluzole
    hydrochloride" are both real, separate USAN entries), and USAN's own
    title field can contain a typo relative to the query ("PRADEMEGENE
    ZAMIKERACEL" for a query of "prademagene zamikeracel" -- confirmed
    directly against the real API response). An exact string match would
    wrongly reject that typo'd hit and wrongly accept nothing for a query
    like "wakix" (a brand, which returns its generic ingredient
    pitolisant's document as the closest full-text match, not "wakix"'s
    own -- USAN doesn't register brand names at all). Similarity scoring
    handles both: the typo'd title still scores ~0.95 similar, "pitolisant"
    to "wakix" scores far below the acceptance threshold.

    A plain generic can also have NO Statement of its own in the index at
    all, only a salt/ester-qualified one -- confirmed directly:
    "varenicline" alone finds nothing above the acceptance threshold, but
    "varenicline_tartrate.pdf" is the only real varenicline USAN document
    that exists, filed under the marketed salt. Its own PRONUNCIATION
    field only ever respells the stem itself, never the salt qualifier
    (an ordinary, independently pronounceable word), so `_is_salt_form_of`
    + `_first_word_group` recover just that stem's portion when the
    fuzzy-similarity match alone would otherwise reject the whole
    document as "not a match for this generic".
    """
    from . import usan_stems  # local import: avoids a hard, one-way dependency

    base_name, _ = usan_stems._split_fda_suffix(name.lower().replace(" ", "-"))
    search_term = base_name.replace("-", " ")

    cache_key = f"usan-pron::{base_name}"
    hit = _cached(cache_key)
    if hit is not None:
        return hit or None

    target_norm = _norm_for_match(search_term)
    candidates = _usan_search(search_term)
    best = None
    best_ratio = 0.0
    for c in candidates:
        title_ratio = difflib.SequenceMatcher(None, target_norm, _norm_for_match(c["title"])).ratio()
        slug_ratio = difflib.SequenceMatcher(
            None, target_norm, _norm_for_match(_doc_slug(c["document_uri"]))
        ).ratio()
        # The filename slug is the more reliable signal -- a real title can
        # carry extra descriptive words a plain drug-name title normally
        # wouldn't ("ENSARTINIB nonproprietary drug name" for the correct,
        # plain "ensartinib.pdf", scoring lower against the title alone
        # than the WRONG "ensartinib-hydrochloride.pdf" salt variant did,
        # confirmed as a real mismatch this max() fixes), while the
        # filename itself consistently follows the "{name}.pdf" convention.
        ratio = max(title_ratio, slug_ratio)
        if ratio > best_ratio:
            best_ratio, best = ratio, c

    def _fetch_pronunciation(document_uri: str) -> str:
        url = f"{USAN_DOC_BASE}?uri={urllib.parse.quote(document_uri)}"
        try:
            req = urllib.request.Request(url, headers=UA)
            content = urllib.request.urlopen(req, timeout=TIMEOUT).read()
        except Exception:
            return ""
        if not content.startswith(b"%PDF"):
            return ""
        try:
            from pypdf import PdfReader
            import io

            reader = PdfReader(io.BytesIO(content))
            text = "\n".join(p.extract_text() or "" for p in reader.pages[:2])
            m = _USAN_PRONUNCIATION.search(text)
            return m.group(1).strip() if m else ""
        except Exception:
            return ""

    result = None
    if best and best_ratio >= 0.85:
        raw_full = _fetch_pronunciation(best["document_uri"])
        raw = _collapse_pronunciation_whitespace(raw_full, name) if raw_full else ""
        if raw:
            url = f"{USAN_DOC_BASE}?uri={urllib.parse.quote(best['document_uri'])}"
            result = {"name": "usan-official", "raw": raw, "url": url}
    else:
        # No candidate scored high enough to be the plain generic's own
        # Statement -- before giving up, check whether the index instead
        # only has this stem filed under a salt/ester qualifier (USAN
        # sometimes never digitized a bare-stem Statement at all even for
        # a name with no separate salt-form USAN entry, e.g.
        # "varenicline" only exists in the index as
        # "varenicline_tartrate.pdf"). Every such candidate is tried, not
        # just the first: some salt-form Statements respell only the
        # stem with nothing appended at all (varenicline tartrate,
        # ibuprofen sodium -- uniform inter-syllable spacing throughout,
        # no boundary for `_first_word_group` to find, and no way to
        # prove algorithmically that nothing beyond the stem is present),
        # while others (ibuprofen trelamine, aripiprazole lauroxil)
        # genuinely respell both words and DO show a detectable boundary
        # -- so a stem with several salt-form entries in the index is
        # still recoverable via whichever one happens to include the
        # salt name's own respelling too, even though the others alone
        # would have to be silently discarded as unverifiable.
        for c in candidates:
            if not (
                _is_salt_form_of(search_term, c["title"])
                or _is_salt_form_of(search_term, _doc_slug(c["document_uri"]))
            ):
                continue
            raw_full = _fetch_pronunciation(c["document_uri"])
            if not raw_full:
                continue
            stem = _first_word_group(raw_full)
            if not stem:
                continue
            raw = re.sub(r"\s+", " ", stem).strip()
            if raw:
                url = f"{USAN_DOC_BASE}?uri={urllib.parse.quote(c['document_uri'])}"
                result = {"name": "usan-official", "raw": raw, "url": url}
                break

    _store(cache_key, result or {})
    return result
