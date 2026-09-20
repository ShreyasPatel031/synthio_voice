"""A crude rule-based English grapheme-to-phoneme converter.

THIS IS NOT A PRONUNCIATION AUTHORITY. It exists so that the judge can be
developed and stress-tested at full scale (286 spans) before Workstream 1a
delivers the real reference layer. Its output is plausible, deterministic and
distinct per name -- which is all the confusability search and the stress test
need -- and it is wrong often enough that it must never be scored against.

Everything it produces is written to `synthetic_references.jsonl`, which carries
`"confidence": "low"` and a loud note on every record.
"""

from __future__ import annotations

import re

VOWEL_LETTERS = "aeiouy"

# Longest-match-first consonant rules. `_` marks a position filled by a vowel
# that the vowel pass will handle; consonant rules only emit consonants.
CONSONANT_RULES: list[tuple[str, list[str]]] = [
    ("tch", ["CH"]),
    ("sch", ["S", "K"]),
    ("sci", ["S", "AY"]),
    ("psy", ["S", "AY"]),
    ("ph", ["F"]),
    ("th", ["TH"]),
    ("sh", ["SH"]),
    ("ch", ["K"]),
    ("ck", ["K"]),
    ("gh", ["G"]),
    ("qu", ["K", "W"]),
    ("wr", ["R"]),
    ("kn", ["N"]),
    ("x", ["K", "S"]),
    ("j", ["JH"]),
    ("z", ["Z"]),
    ("v", ["V"]),
    ("f", ["F"]),
    ("b", ["B"]),
    ("p", ["P"]),
    ("d", ["D"]),
    ("t", ["T"]),
    ("k", ["K"]),
    ("m", ["M"]),
    ("n", ["N"]),
    ("l", ["L"]),
    ("r", ["R"]),
    ("w", ["W"]),
    ("h", ["HH"]),
]

VOWEL_DIGRAPHS = {
    "ai": "EY", "ay": "EY", "ei": "EY", "ey": "EY",
    "ee": "IY", "ea": "IY", "ie": "IY",
    "oa": "OW", "oe": "OW",
    "oo": "UW", "ue": "UW", "ui": "UW", "eu": "UW", "ew": "UW",
    "ou": "AW", "ow": "AW",
    "oi": "OY", "oy": "OY",
    "au": "AO", "aw": "AO",
}

OPEN_VOWEL = {"a": "EY", "e": "IY", "i": "AY", "o": "OW", "u": "UW", "y": "AY"}
CLOSED_VOWEL = {"a": "AE", "e": "EH", "i": "IH", "o": "AA", "u": "AH", "y": "IH"}
PRE_R_VOWEL = {"a": "AA", "e": "ER", "i": "ER", "o": "AO", "u": "ER", "y": "ER"}

ARPABET_TO_IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ə", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ",
    "EH": "ɛ", "ER": "ɚ", "EY": "eɪ", "IH": "ɪ", "IY": "i", "OW": "oʊ",
    "OY": "ɔɪ", "UH": "ʊ", "UW": "u",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f", "G": "ɡ", "HH": "h",
    "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ", "P": "p",
    "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ",
}
IPA_STRESS = {1: "ˈ", 2: "ˌ"}


def _soft(letter: str, nxt: str) -> bool:
    return nxt in "eiy"


def _word_to_phonemes(word: str) -> list[str]:
    """Convert one orthographic word to unstressed ARPABET."""
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return []
    # A final silent 'e' lengthens the preceding vowel rather than sounding.
    silent_e = len(word) > 3 and word.endswith("e") and word[-2] not in VOWEL_LETTERS
    if silent_e:
        word = word[:-1] + "~"

    out: list[str] = []
    i = 0
    while i < len(word):
        ch = word[i]
        if ch == "~":
            i += 1
            continue

        if ch == "c" and not word.startswith(("ch", "ck"), i):
            out.append("S" if i + 1 < len(word) and _soft(ch, word[i + 1]) else "K")
            i += 1
            continue
        if ch == "g":
            out.append("JH" if i + 1 < len(word) and _soft(ch, word[i + 1]) else "G")
            i += 1
            continue
        if word.startswith("ti", i) and i + 2 < len(word) and word[i + 2] in "ao":
            out.append("SH")
            i += 2
            continue
        if ch == "s" and 0 < i < len(word) - 1 and word[i - 1] in VOWEL_LETTERS and word[i + 1] in VOWEL_LETTERS:
            out.append("Z")
            i += 1
            continue
        if ch == "s":
            out.append("S")
            i += 1
            continue
        if word.startswith("ng", i) and i + 2 >= len(word):
            out.append("NG")
            i += 2
            continue

        if ch in VOWEL_LETTERS:
            digraph = word[i : i + 2]
            if digraph in VOWEL_DIGRAPHS:
                out.append(VOWEL_DIGRAPHS[digraph])
                i += 2
                continue
            rest = word[i + 1 :].replace("~", "")
            if rest.startswith("r"):
                vowel = PRE_R_VOWEL[ch]
                out.append(vowel)
                if vowel == "ER":
                    i += 2  # ER already carries the /r/
                    continue
            else:
                consonants = len(rest) - len(rest.lstrip("bcdfghjklmnpqrstvwxyz"))
                following_vowel = len(rest) > consonants
                open_syllable = (
                    consonants <= 1 and following_vowel
                ) or (consonants == 0 and word[i + 1 : i + 2] == "~")
                out.append((OPEN_VOWEL if open_syllable else CLOSED_VOWEL)[ch])
            i += 1
            continue

        for pattern, phonemes in CONSONANT_RULES:
            if word.startswith(pattern, i):
                if not (out and phonemes == out[-1:]):
                    out.extend(phonemes)
                i += len(pattern)
                break
        else:
            i += 1

    return out


def _nuclei(phonemes: list[str]) -> list[int]:
    return [i for i, p in enumerate(phonemes) if p in ARPABET_TO_IPA and p[0] in "AEIOU"]


def _apply_stress(phonemes: list[str], primary_from_end: int) -> list[str]:
    """Return a copy with stress digits, primary N syllables from the end."""
    nuclei = _nuclei(phonemes)
    if not nuclei:
        return list(phonemes)
    primary = nuclei[max(len(nuclei) - primary_from_end, 0)]
    secondary = nuclei[0] if len(nuclei) >= 4 and nuclei[0] != primary else None

    out = list(phonemes)
    for idx in nuclei:
        level = 1 if idx == primary else (2 if idx == secondary else 0)
        out[idx] = f"{out[idx]}{level}"
    return out


def to_arpabet_variants(name: str) -> list[list[str]]:
    """Base pronunciation plus plausible alternates, most-preferred first."""
    words = [w for w in re.split(r"[\s\-]+", name.strip()) if w]
    bare: list[str] = []
    for w in words:
        bare.extend(_word_to_phonemes(w))
    if not bare:
        return [["AH0"]]

    n_syllables = len(_nuclei(bare))
    variants = [_apply_stress(bare, 3 if n_syllables >= 3 else n_syllables)]
    if n_syllables >= 2:
        alternate = _apply_stress(bare, 2)
        if alternate != variants[0]:
            variants.append(alternate)
    if n_syllables >= 4:
        alternate = _apply_stress(bare, 4)
        if alternate not in variants:
            variants.append(alternate)
    return variants


def to_ipa(arpabet: list[str]) -> str:
    out = []
    for token in arpabet:
        stress = int(token[-1]) if token[-1].isdigit() else 0
        base = token.rstrip("012")
        out.append(IPA_STRESS.get(stress, "") + ARPABET_TO_IPA[base])
    return "".join(out)
