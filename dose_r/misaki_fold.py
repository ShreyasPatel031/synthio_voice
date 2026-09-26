"""Fold locked gold IPA into American Misaki. Do not invent phones.

These maps are the Misaki EN_PHONES.md diphthong letters plus restress.
They are not G2P. They do not read DailyMed/USAN respelling.
"""

from __future__ import annotations

import re

STRESSES = "ˌˈ"
VOWELS = frozenset("AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ")
IPA_MARKERS = ("oʊ", "aɪ", "eɪ", "aʊ", "ɔɪ", "əʊ", "ː", "ɒ")


def ipa_pieces(raw: str) -> list[str]:
    return [p for p in re.split(r"[\s/]+", raw.strip()) if p]


def ipa_to_misaki(raw: str, symbols: set[str]) -> str:
    """Fold published IPA into Misaki's American symbols. Do not invent phones."""
    s = raw.strip().strip("/[]")
    # Slashes in the gold string are separators, not phonemes
    # (e.g. a two-word name stored as "ipa/ /ipa").
    s = s.replace("/", "")
    s = s.replace("'", "ˈ").replace(".", "")
    for old, new in (
        ("tʃ", "ʧ"),
        ("dʒ", "ʤ"),
        ("eɪ", "A"),
        ("aɪ", "I"),
        ("aʊ", "W"),
        ("ɔɪ", "Y"),
        ("oʊ", "O"),
        ("əʊ", "O"),
        ("ɝ", "ɜɹ"),
        ("ɚ", "əɹ"),
    ):
        s = s.replace(old, new)
    s = s.replace("ː", "").replace("r", "ɹ").replace("g", "ɡ").replace(" ", "")
    missing = sorted({ch for ch in s if ch not in symbols})
    if missing:
        raise ValueError(f"not in Misaki vocab: {missing} from {raw!r} -> {s!r}")
    return s


def fold_documented(raw: str, symbols: set[str]) -> str:
    """American fold from misaki EN_PHONES.md (from_espeak) and en.G2P.

    https://github.com/hexgrad/misaki/blob/main/EN_PHONES.md
    Longest tie/diphthong first, then the American-only rewrites:
    bare e→A, leftover o→ɔ, ɜː→ɜɹ, ɪə→iə, ː dropped.
    British-only symbols become their American pair (a→æ, ɒ→ɑ).
    Live G2P then rewrites ɾ→T and ʔ→t unless version is 2.0.
    """
    s = raw.strip().strip("/[]").replace("'", "ˈ").replace(".", "")
    for old, new in (
        ("tʃ", "ʧ"),
        ("dʒ", "ʤ"),
        ("eɪ", "A"),
        ("aɪ", "I"),
        ("aʊ", "W"),
        ("ɔɪ", "Y"),
        ("oʊ", "O"),
        ("əʊ", "O"),
        ("ɝ", "ɜɹ"),
        ("ɚ", "əɹ"),
        ("ɜːɹ", "ɜɹ"),
        ("ɜː", "ɜɹ"),
        ("ɐ", "ə"),
        ("x", "k"),
        ("ç", "k"),
    ):
        s = s.replace(old, new)
    s = s.replace("ɪə", "iə")
    s = s.replace("ː", "")
    s = s.replace("r", "ɹ").replace("g", "ɡ")
    s = s.replace("e", "A").replace("o", "ɔ")
    s = s.replace("a", "æ").replace("ɒ", "ɑ")
    s = re.sub(r"(\S)\u0329", r"ᵊ\1", s)
    s = s.replace("ɾ", "T").replace("ʔ", "t")
    s = s.replace(" ", "")
    missing = sorted({ch for ch in s if ch not in symbols})
    if missing:
        raise ValueError(f"not in Misaki vocab: {missing} from {raw!r} -> {s!r}")
    return s


def restress(ps: str) -> str:
    """Misaki's restress: ˈ and ˌ move to immediately before the next vowel.

    IPA writes the tick at the start of the syllable (ˈɹɪn). Kokoro was
    trained on the tick sitting in front of the vowel (ɹˈɪn).
    """
    ips = list(enumerate(ps))
    moved = {}
    for i, p in ips:
        if p not in STRESSES:
            continue
        nxt = next((j for j, v in ips[i + 1 :] if v in VOWELS), None)
        if nxt is None:
            raise ValueError(f"stress with no following vowel: {ps!r}")
        moved[i] = nxt
    for i, j in moved.items():
        ips[i] = (j - 0.5, ips[i][1])
    return "".join(p for _, p in sorted(ips))


def docs_phones(ipa: str, spoken: str, symbols: set[str]) -> str:
    words = spoken.split()
    pieces = [restress(fold_documented(p, symbols)) for p in ipa_pieces(ipa)]
    if len(pieces) != len(words):
        pieces = [restress(fold_documented(ipa, symbols))]
    return " ".join(pieces)


def gold_misaki_candidates(ipa: str, spoken: str, symbols: set[str]) -> list[tuple[str, str]]:
    """Misaki strings that are the locked gold IPA, folded. No vowel roulette."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(src: str, phones: str) -> None:
        phones = (phones or "").strip()
        if not phones or phones in seen:
            return
        if any(m in phones for m in IPA_MARKERS):
            return
        if any(ch not in symbols and not ch.isspace() for ch in phones):
            return
        seen.add(phones)
        out.append((src, phones))

    try:
        raw = ipa_to_misaki(ipa, symbols)
        add("gold-ipa", raw)
        add("gold-stress", restress(raw))
    except ValueError:
        pass
    try:
        add("gold-docs", docs_phones(ipa, spoken, symbols))
    except ValueError:
        pass
    return out
