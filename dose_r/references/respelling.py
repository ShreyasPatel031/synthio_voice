"""One dictionary-respelling per ingredient, one source, one ASCII format.

The gold layer in `references.jsonl` kept every citation and converted each
to IPA. That is the opposite of what a TTS-facing pronunciation string needs:
91% of sources already wrote a lay respelling, but in three incompatible
notations (USAN primes, hyphenated CAPS, Merriam-Webster phonetic), and 19
names only had MW phonetic so they were counted as "not a dictionary
respelling" even though MW is a dictionary.

This module does a return over that snapshot (no new fetches):

1. Pick exactly one source, USAN/AMA first, then FDA/NCI, then MW medical.
2. Rewrite its raw string into one hyphenated form: unstressed syllables
   lowercase, exactly one primary-stress syllable ALL CAPS per word.

Wikipedia IPA, Wiktionary IPA, and CMUdict are never selected. They are not
dictionary respellings.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from . import gemini_grounded
from .notation import split_variants

# USAN/AMA first, then official medical (FDA Medication Guide, NCI), then
# MW medical, then MW general, then Gemini-retrieved respellings. Wikipedia /
# Wiktionary / CMUdict are deliberately absent.
SOURCE_PRIORITY: tuple[str, ...] = (
    "usan-official",
    "dailymed",
    "nci-dictionary-of-cancer-terms",
    "merriam-webster/medical-api",
    "merriam-webster/dictionary",
    "gemini-grounded-search",
)

_MW_MARK = re.compile(r"[āēīōüäȯəᵊˈˌ]")
_IPA_PHONE = re.compile(r"[ɪʊɛæɑɒɔʌʃʒθðɡɹːᵻ]")
_PRIME = re.compile(r"[\'’‘′\"”″\uE000-\uF8FF]")
_CANON_SYL = re.compile(r"^[a-z]+$|^[A-Z]+$")
_LETTERS = re.compile(r"[^a-zA-Z\- ]")

# MW phonetic graphemes -> Wikipedia/NCI respelling key (longest first).
# ī -> "ye" matches USAN's own "sye"/"lye"/"zye" for /aɪ/.
_MW_GRAPHEMES: list[tuple[str, str]] = [
    ("au̇", "ow"),
    ("ȯi", "oy"),
    ("ər", "ur"),
    ("th̸", "dh"),
    ("ch", "ch"),
    ("sh", "sh"),
    ("zh", "zh"),
    ("th", "th"),
    ("ng", "ng"),
    ("ā", "ay"),
    ("ē", "ee"),
    ("ī", "ye"),
    ("ō", "oh"),
    ("ü", "oo"),
    ("u̇", "uu"),
    ("ȯ", "aw"),
    ("ä", "ah"),
    ("ə", "uh"),
    ("ᵊ", "uh"),
    ("ŋ", "ng"),
]


def is_canonical(text: str) -> bool:
    """True iff `text` is hyphenated ASCII with exactly one CAPS syllable
    per stressed word. A single lowercase syllable (`chloride`) is allowed
    for USAN salt/qualifier words the source did not respell."""
    if not text or _LETTERS.search(text):
        return False
    words = text.split()
    if not words:
        return False
    if not any(any(c.isupper() for c in word) for word in words):
        return False
    for word in words:
        parts = word.split("-")
        if not parts or any(not p or not _CANON_SYL.match(p) for p in parts):
            return False
        if len(parts) == 1 and parts[0].islower():
            continue
        caps = sum(1 for p in parts if p.isupper() and p.lower() != p)
        if caps != 1:
            return False
    return True


def _one_primary(parts: list[str], inferred: bool = False) -> tuple[str, bool]:
    """Lowercase every syllable, then restore exactly one ALL-CAPS primary.

    Preference: the first syllable that already contains an uppercase
    letter (NCI `MIH`, DailyMed `ZOL`, title-case `Kres`); else the first
    syllable and `inferred=True`.
    """
    clean = [re.sub(r"[^a-zA-Z]", "", p) for p in parts]
    clean = [p for p in clean if p]
    if not clean:
        return "", inferred
    if len(clean) == 1 and not any(c.isupper() for c in clean[0]):
        # Unmarked single token: said as spelled, not a guessed primary.
        return clean[0].lower(), False
    idx = next(
        (i for i, p in enumerate(clean) if any(c.isupper() for c in p)),
        None,
    )
    if idx is None:
        idx = 0
        inferred = True
    out = [p.lower() for p in clean]
    out[idx] = out[idx].upper()
    return "-".join(out), inferred


def mw_to_respelling(raw: str) -> str | None:
    """Merriam-Webster phonetic (`ə-ˈbi-lə-ˌfī`) -> `uh-BI-luh-fye`."""
    variants = split_variants(unicodedata.normalize("NFC", raw))
    if not variants:
        return None
    text = variants[0]
    text = re.sub(r"\([ˈˌ]\)", "", text)
    # Keep optional-sound letters, drop the parentheses: van(t)s -> vants,
    # p(ə-)rən -> pə-rən.
    text = re.sub(r"\(([^)]*)\)", lambda m: m.group(1), text)
    text = text.replace("·", "-")
    parts: list[str] = []
    inferred = False
    for syllable in text.split("-"):
        syllable = syllable.strip()
        if not syllable:
            continue
        stress = ""
        if syllable.startswith("ˈ"):
            stress = "primary"
            syllable = syllable[1:]
        elif syllable.startswith("ˌ"):
            syllable = syllable[1:]
        body = []
        i = 0
        s = syllable
        while i < len(s):
            hit = False
            for graph, repl in _MW_GRAPHEMES:
                if s.startswith(graph, i):
                    body.append(repl)
                    i += len(graph)
                    hit = True
                    break
            if hit:
                continue
            ch = s[i]
            if ch.isalpha():
                body.append(ch.lower())
            i += 1
        letters = "".join(body)
        letters = re.sub(r"[^a-zA-Z]", "", letters)
        if not letters:
            continue
        if stress == "primary":
            letters = letters.upper()
        parts.append(letters)
    if not parts:
        return None
    canon, _ = _one_primary(parts, inferred=False)
    return canon or None


def _looks_mw(raw: str) -> bool:
    """MW phonetic. Do not put `u̇` in a character class — it matches bare `u`."""
    nfc = unicodedata.normalize("NFC", raw)
    return bool(_MW_MARK.search(nfc) or "u̇" in nfc or "u\u0307" in raw)


def _usan_respelling(tokens: list[str]) -> str | None:
    """USAN prime-stress tokens -> hyphenated CAPS. If the source only
    marked secondary stress, promote the first secondary to primary."""
    got = gemini_grounded.stress_tokens_to_respelling(tokens)
    if got:
        return got
    promoted = []
    did = False
    for t in tokens:
        kind = gemini_grounded._stress_kind(t) if t else None
        if not did and kind == "secondary":
            letters = re.sub(r"[^a-zA-Z]", "", t)
            promoted.append(letters + "'")
            did = True
        else:
            promoted.append(t)
    if not did:
        return None
    return gemini_grounded.stress_tokens_to_respelling(promoted)


def _split_mid_token_primes(raw: str) -> str:
    """`trip'tir` -> `trip' tir`. Leave trailing USAN primes (`em’’`) alone."""
    text = raw.replace("-", " ")
    return re.sub(
        r"[\'’‘′\"”″\uE000-\uF8FF](?=[A-Za-z])",
        lambda m: m.group(0) + " ",
        text,
    )


def _hyphenated_or_spaced(raw: str) -> tuple[str | None, bool]:
    """USAN primes, NCI/DailyMed hyphenation, or space-separated CAPS."""
    text = unicodedata.normalize("NFC", raw).strip()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"^(pronounced|pronunciation)\s+", "", text, flags=re.I)

    # Already hyphenated, stress by CAPS, no primes: NCI / most DailyMed.
    if "-" in text and not _PRIME.search(text):
        canon, inferred = _one_primary(text.split("-"))
        return (canon or None), inferred

    # DailyMed `leh-kem’-bee`, `trip'tir`: primes as stress, hyphens as breaks.
    primed = _split_mid_token_primes(text)
    tokens = [t for t in primed.split() if t]
    usan = _usan_respelling(tokens)
    if usan:
        canon, inferred = _one_primary(usan.split("-"))
        return (canon or None), inferred

    # Space-separated mixed case, no primes: `jar DEE ans`, `AD vair`.
    if tokens and any(t.isalpha() for t in tokens):
        if any(any(c.isupper() for c in t) for t in tokens) or "-" not in text:
            canon, inferred = _one_primary(tokens)
            return (canon or None), inferred

    return None, False


def to_canonical(raw: str, source_name: str = "") -> tuple[str | None, bool]:
    """Raw citation -> (canonical respelling, stress_inferred).

    `stress_inferred` is True when the source marked no primary and we
    defaulted to the first syllable. Callers can surface those.
    """
    if not (raw or "").strip():
        return None, False
    raw = unicodedata.normalize("NFC", raw).strip().strip("()[]")

    if source_name in {"wikipedia", "wiktionary", "cmudict"}:
        return None, False
    # Native IPA is not a dictionary respelling. MW uses ə/ˈ/ˌ too, but with
    # hyphenated syllables and macrons, never ɪ/æ/ɔ.
    if _IPA_PHONE.search(raw):
        return None, False
    if raw.startswith("/") or (raw.startswith("[") and raw.endswith("]")):
        return None, False

    if source_name.startswith("merriam-webster") or _looks_mw(raw):
        got = mw_to_respelling(raw)
        if got:
            return got, False

    words = [w for w in re.split(r"\s{2,}", raw) if w.strip()]
    if len(words) < 2:
        words = [raw]

    out_words = []
    inferred_any = False
    for word in words:
        canon, inferred = _hyphenated_or_spaced(word)
        if not canon:
            return None, False
        out_words.append(canon)
        inferred_any = inferred_any or inferred
    joined = " ".join(out_words)
    if not is_canonical(joined):
        return None, False
    return joined, inferred_any


def pick_source(sources: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Highest-priority source that converts to a canonical respelling."""
    by_name: dict[str, list[dict[str, Any]]] = {}
    for src in sources or []:
        by_name.setdefault(src.get("name", ""), []).append(src)
    for name in SOURCE_PRIORITY:
        for src in by_name.get(name, []):
            canon, inferred = to_canonical(src.get("raw") or "", name)
            if canon:
                return {
                    "source": name,
                    "source_raw": src.get("raw") or "",
                    "url": src.get("url") or "",
                    "trust_tier": src.get("trust_tier") or "",
                    "respelling": canon,
                    "stress_inferred": inferred,
                    "domain": src.get("domain") or "",
                }
    return None


def homogenize_record(rec: dict[str, Any]) -> dict[str, Any]:
    """One references.jsonl row -> one homogenized respelling row."""
    picked = pick_source(rec.get("sources") or [])
    out: dict[str, Any] = {
        "ingredient": rec["ingredient"],
        "name_type": rec.get("name_type", ""),
        "respelling": None,
        "source": None,
        "source_raw": None,
        "url": None,
        "trust_tier": None,
        "stress_inferred": False,
        "dropped_sources": [
            s.get("name")
            for s in rec.get("sources") or []
            if picked is None or s.get("name") != picked["source"]
        ],
    }
    if picked is None:
        out["confidence"] = "low"
        out["notes"] = "no dictionary respelling after USAN/DailyMed/NCI/MW/Gemini"
        return out
    out.update(picked)
    out["confidence"] = "sourced"
    out["notes"] = ""
    return out
