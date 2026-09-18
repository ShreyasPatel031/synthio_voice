"""ClinCalc's "Top 250 Drugs" pronunciation pages.

Reachable directly (no Cloudflare block, unlike Drugs.com/DrugBank/UMich), and
the first source in this corpus that records the generic name and a brand
name as two *separate* audio clips rather than one page's one ambiguous
recording -- each drug page embeds up to two `<audio>` blocks, headed "The
generic name '...' is pronounced:" and "The brand name '...' is pronounced:".

A block's headed name can itself list several names joined by "; " -- either
because the drug is a combination product (`"Acetaminophen; hydrocodone"`) or
because several brands share one page and its one recording
(`"Vicodin; Norco; Lortab (many more)"`, a single clip that says all three,
not three separate ones). `split_names` peels the "(many more)" / "(and many
more)" tail ClinCalc appends to a truncated brand list and splits the rest,
so a caller can tell a single-name clip from a several-names-at-once one.

The index page's own labels (`generic (Brand1; Brand2)`) are used only to
decide which of the 250 drug pages are worth fetching -- the page's own
headed names are the source of truth for what its audio actually says,
because the index occasionally lists a drug under two labels that resolve to
the *same* page (both "fluticasone (inhaled) (Flovent)" and "fluticasone
(nasal) (Flonase)" link to `HowToPronounce/fluticasone`, whose one brand
block only ever plays Flovent) -- a site data-quality quirk, not something to
route around by guessing an unlisted URL.
"""

from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

INDEX_URL = "https://clincalc.com/pronouncetop200drugs/"
PAGE_URL = "https://clincalc.com/pronouncetop200drugs/HowToPronounce/{slug}"
UA = {"User-Agent": "Mozilla/5.0 (compatible; DOSE-R research benchmark)"}
TIMEOUT = 25

_INDEX_ENTRY = re.compile(r'<a href="HowToPronounce/([^"]+)">(.*?)</a>')
_MANY_MORE = re.compile(r"\s*\(\s*(?:and\s+)?many more\s*\)\s*$", re.I)
_ROUTE_ANNOTATION = re.compile(r"\s*\([^)]*\)\s*$")
_NAMED_BLOCK = re.compile(
    r"The (generic|brand) name '([^']+)' is pronounced:.*?"
    r'<source src="([^"]+\.mp3)" type="audio/mpeg">',
    re.S,
)


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=TIMEOUT).read().decode("utf-8", "ignore")


def fetch_audio(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=TIMEOUT).read()


def polite_fetch(url: str, delay_s: float = 0.3) -> str:
    time.sleep(delay_s)
    return fetch(url)


def polite_fetch_audio(url: str, delay_s: float = 0.3) -> bytes:
    time.sleep(delay_s)
    return fetch_audio(url)


def strip_many_more(text: str) -> str:
    return _MANY_MORE.sub("", text).strip()


def split_names(text: str) -> list[str]:
    """A headed or index-label name string -> its individual names.

    Splits on ";" after dropping a trailing "(many more)" / "(and many
    more)" -- that tail describes the list, it is never itself a name.
    """
    return [p.strip() for p in strip_many_more(text).split(";") if p.strip()]


def strip_route_annotation(name: str) -> str:
    """Drop a single trailing parenthetical, e.g. "Fluticasone (inhaled)" ->
    "Fluticasone" -- ClinCalc's way of distinguishing two entries for the
    same generic name by route of administration."""
    return _ROUTE_ANNOTATION.sub("", name).strip()


def name_matches(clincalc_name: str, ingredient: str) -> bool:
    """True if `clincalc_name` names `ingredient`, exactly or as its bare
    INN/brand stem -- the common case for a DOSE ingredient recorded with a
    salt or ester suffix ClinCalc's own list omits (`"fluticasone"` /
    `"fluticasone propionate"`, `"tiotropium"` / `"tiotropium bromide"`).
    A strict word-prefix, not a substring match, so it cannot cross a word
    boundary into an unrelated name.
    """
    c = strip_route_annotation(clincalc_name).lower()
    i = ingredient.lower()
    if c == i:
        return True
    c_words, i_words = c.split(), i.split()
    return len(c_words) < len(i_words) and i_words[: len(c_words)] == c_words


@dataclass
class IndexEntry:
    slug: str
    generics: list[str] = field(default_factory=list)
    brands: list[str] = field(default_factory=list)


def _split_index_label(label: str) -> tuple[list[str], list[str]]:
    """"generic; generic (Brand1; Brand2 (many more))" -> (generics, brands).

    The brand group is the *last* top-level parenthesized group, found by
    depth-counting from the end -- not the first "(" ")" pair, because a
    generic name can carry its own parenthetical route annotation before the
    brand group (`"fluticasone (inhaled) (Flovent)"`), and the brand group
    can itself contain balanced parens (the "(many more)" tail).
    """
    label = label.strip()
    if not label.endswith(")"):
        return split_names(label), []

    depth, start = 0, None
    for i in range(len(label) - 1, -1, -1):
        if label[i] == ")":
            depth += 1
        elif label[i] == "(":
            depth -= 1
            if depth == 0:
                start = i
                break
    if start is None:
        return split_names(label), []
    return split_names(label[:start]), split_names(label[start + 1 : -1])


def parse_index(html: str) -> list[IndexEntry]:
    entries = []
    for slug, label in _INDEX_ENTRY.findall(html):
        generics, brands = _split_index_label(label)
        entries.append(IndexEntry(slug, generics, brands))
    return entries


def merge_by_slug(entries: list[IndexEntry]) -> dict[str, IndexEntry]:
    """slug -> one IndexEntry with every generics/brands name seen under it.

    More than one index label can point at the same page (see module
    docstring); merging keeps every name the index ever suggested for that
    page as a candidate, so a name the real page turns out not to confirm is
    reported as a miss rather than silently dropped.
    """
    merged: dict[str, IndexEntry] = {}
    for e in entries:
        m = merged.setdefault(e.slug, IndexEntry(e.slug))
        for g in e.generics:
            if g not in m.generics:
                m.generics.append(g)
        for b in e.brands:
            if b not in m.brands:
                m.brands.append(b)
    return merged


def candidate_slugs(entries: list[IndexEntry], ingredients: list[str]) -> dict[str, IndexEntry]:
    """slug -> its merged IndexEntry, for every page worth fetching.

    A coarse filter only: it decides which pages to fetch, not which
    ingredients ultimately get a clip -- `parse_page`'s own headed names
    decide that.
    """
    return {
        slug: entry
        for slug, entry in merge_by_slug(entries).items()
        if any(name_matches(n, ing) for n in entry.generics + entry.brands for ing in ingredients)
    }


@dataclass
class DrugPage:
    generic_name: str | None = None
    generic_url: str | None = None
    brand_name: str | None = None
    brand_url: str | None = None


def page_url(slug: str) -> str:
    return PAGE_URL.format(slug=slug)


def parse_page(html: str, base_url: str) -> DrugPage:
    page = DrugPage()
    for kind, name, src in _NAMED_BLOCK.findall(html):
        url = urllib.parse.urljoin(base_url, src)
        if kind == "generic":
            page.generic_name, page.generic_url = name, url
        else:
            page.brand_name, page.brand_url = name, url
    return page
