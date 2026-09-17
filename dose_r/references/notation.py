"""Merriam-Webster pronunciation notation to IPA and ARPABET.

MW writes drug pronunciations in its own respelling system, e.g.

    dapagliflozin   ˌda-pə-glə-ˈflō-zən
    fluticasone     flü-ˈtik-ə-ˌsōn, -ˌzōn

Syllables are hyphen-separated, `ˈ` marks primary stress on the syllable that
follows it and `ˌ` marks secondary stress, and a comma introduces an accepted
variant. A variant beginning with `-` replaces only the trailing syllables of
the form before it, which is how MW writes the fluticasone case above.

ARPABET carries stress on the vowel, so the syllable-level MW markers are
resolved onto each syllable's nucleus during conversion. Syllables MW leaves
unmarked are unstressed, except that a form with no `ˈ` anywhere is treated as
having primary stress on its first vowel rather than emitting a word with no
stressed syllable at all.
"""

from __future__ import annotations

import re
import unicodedata

PRIMARY = "ˈ"
SECONDARY = "ˌ"

# Longest match first. The value is (arpabet_without_stress, ipa).
VOWELS: list[tuple[str, tuple[str, str]]] = [
    ("au̇", ("AW", "aʊ")),
    ("ȯi", ("OY", "ɔɪ")),
    ("ər", ("ER", "ər")),
    ("ā", ("EY", "eɪ")),
    ("ē", ("IY", "iː")),
    ("ī", ("AY", "aɪ")),
    ("ō", ("OW", "oʊ")),
    ("ü", ("UW", "uː")),
    ("u̇", ("UH", "ʊ")),
    ("ȯ", ("AO", "ɔ")),
    ("ä", ("AA", "ɑ")),
    ("ə", ("AH", "ə")),
    ("ᵊ", ("AH", "ə")),
    ("a", ("AE", "æ")),
    ("e", ("EH", "ɛ")),
    ("i", ("IH", "ɪ")),
    ("o", ("AA", "ɑ")),
    ("u", ("AH", "ʌ")),
]

CONSONANTS: list[tuple[str, tuple[str, str]]] = [
    ("th̸", ("DH", "ð")),
    ("ch", ("CH", "tʃ")),
    ("sh", ("SH", "ʃ")),
    ("zh", ("ZH", "ʒ")),
    ("th", ("TH", "θ")),
    ("ng", ("NG", "ŋ")),
    ("ŋ", ("NG", "ŋ")),
    ("j", ("JH", "dʒ")),
    ("y", ("Y", "j")),
    ("k", ("K", "k")),
    ("g", ("G", "ɡ")),
    ("p", ("P", "p")),
    ("b", ("B", "b")),
    ("t", ("T", "t")),
    ("d", ("D", "d")),
    ("m", ("M", "m")),
    ("n", ("N", "n")),
    ("l", ("L", "l")),
    ("r", ("R", "r")),
    ("s", ("S", "s")),
    ("z", ("Z", "z")),
    ("f", ("F", "f")),
    ("v", ("V", "v")),
    ("w", ("W", "w")),
    ("h", ("HH", "h")),
]

RULES = VOWELS + CONSONANTS
VOWEL_ARPA = {a for a, _ in (v for _, v in VOWELS)}

_DROP = dict.fromkeys(map(ord, "()’'`· "), None)


class NotationError(ValueError):
    pass


def split_variants(raw: str) -> list[str]:
    """MW's comma-separated alternates, with trailing-only variants expanded."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return []

    out = [parts[0]]
    for part in parts[1:]:
        if part.startswith("-"):
            tail = part.lstrip("-")
            head = out[0].rsplit("-", tail.count("-") + 1)[0]
            out.append(f"{head}-{tail}")
        else:
            out.append(part)
    return out


def _segments(syllable: str) -> list[tuple[str, str]]:
    text = syllable
    out = []
    while text:
        for graph, mapped in RULES:
            if text.startswith(graph):
                out.append(mapped)
                text = text[len(graph) :]
                break
        else:
            text = text[1:]
    return out


def to_arpabet_ipa(raw: str) -> tuple[str, str]:
    """One MW respelling to (ARPABET string, IPA string)."""
    cleaned = unicodedata.normalize("NFC", raw).translate(_DROP)
    cleaned = re.sub(r"[^\wˈˌəᵊāēīōüȯäŋ̸̇\-]", "", cleaned)
    if not cleaned:
        raise NotationError(f"nothing usable in {raw!r}")

    arpa: list[str] = []
    ipa: list[str] = []
    saw_primary = False

    for syllable in cleaned.split("-"):
        if not syllable:
            continue
        stress = 0
        if syllable.startswith(PRIMARY):
            stress, saw_primary = 1, True
            syllable = syllable[1:]
        elif syllable.startswith(SECONDARY):
            stress = 2
            syllable = syllable[1:]

        nucleus_done = False
        for arpa_sym, ipa_sym in _segments(syllable):
            if arpa_sym in VOWEL_ARPA:
                arpa.append(f"{arpa_sym}{stress if not nucleus_done else 0}")
                nucleus_done = True
            else:
                arpa.append(arpa_sym)
            ipa.append(ipa_sym)

    if not arpa:
        raise NotationError(f"no phonemes recovered from {raw!r}")

    if not saw_primary:
        for i, sym in enumerate(arpa):
            if sym[:-1] in VOWEL_ARPA and sym[-1].isdigit():
                arpa[i] = f"{sym[:-1]}1"
                break

    return " ".join(arpa), "".join(ipa)


def convert(raw: str) -> list[tuple[str, str]]:
    """Every accepted variant in an MW entry, as (ARPABET, IPA) pairs."""
    out = []
    for variant in split_variants(raw):
        try:
            out.append(to_arpabet_ipa(variant))
        except NotationError:
            continue
    return out
