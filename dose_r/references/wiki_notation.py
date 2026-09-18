"""Wikipedia/Wiktionary pronunciation markup to ARPABET + IPA.

Wikipedia and Wiktionary record English pronunciations in wikitext through a
small number of templates, and this module converts the two that carry actual
phonemic content:

    {{IPA|en|/ˈsɛf.ə.piːm/}}              -- raw IPA, slash- or bracket-delimited
    {{IPAc-en|m|ɛ|t|ˈ|f|ɔːr|m|ᵻ|n}}       -- the same alphabet, pre-split into
                                              one phoneme (or stress mark) per
                                              template argument
    {{respell|met|FOR|min}}               -- Wikipedia's own respelling key
                                              (see Help:Pronunciation
                                              respelling key), one syllable per
                                              argument, ALL CAPS = stressed

`{{IPAc-en|...}}` uses exactly the IPA inventory documented at
Help:IPA/English, so both IPA forms are handled by one converter: join the
pipe-separated arguments of an IPAc-en template back into a single string and
it parses identically to a slash-delimited `{{IPA|en|...}}` string.

As in `notation.py`, IPA output carries no stress marks -- only the ARPABET
digits do -- to keep the two converters' output shape identical.
"""

from __future__ import annotations

import re
import unicodedata

from .notation import VOWEL_ARPA

PRIMARY = "ˈ"
SECONDARY = "ˌ"

# --- IPA (Help:IPA/English inventory) --------------------------------------
#
# Longest match first. R-coloured vowels that are a single English phoneme
# (the NURSE and comma-er vowels) are listed whole so they are not mistakenly
# decomposed into a plain vowel plus a consonant /r/; the other r-colored
# vowels (CURE, SQUARE, NEAR, START, NORTH) *are* vowel-plus-/r/ in General
# American and are left to decompose naturally once "r" -> R is in the table.
IPA_VOWELS: list[tuple[str, tuple[list[str], str]]] = [
    ("ɜːr", (["ER"], "ɜːr")),
    ("ər", (["ER"], "ər")),
    ("ɪər", (["IH", "R"], "ɪər")),
    ("ʊər", (["UH", "R"], "ʊər")),
    ("ɛər", (["EH", "R"], "ɛər")),
    ("ɑːr", (["AA", "R"], "ɑːr")),
    ("ɔːr", (["AO", "R"], "ɔːr")),
    ("eɪ", (["EY"], "eɪ")),
    ("aɪ", (["AY"], "aɪ")),
    ("ɔɪ", (["OY"], "ɔɪ")),
    ("aʊ", (["AW"], "aʊ")),
    ("oʊ", (["OW"], "oʊ")),
    ("iː", (["IY"], "iː")),
    ("ɑː", (["AA"], "ɑː")),
    ("ɔː", (["AO"], "ɔː")),
    ("uː", (["UW"], "uː")),
    ("æ", (["AE"], "æ")),
    ("ɛ", (["EH"], "ɛ")),
    ("ɪ", (["IH"], "ɪ")),
    ("ɒ", (["AA"], "ɒ")),
    ("ʊ", (["UH"], "ʊ")),
    ("ʌ", (["AH"], "ʌ")),
    ("ə", (["AH"], "ə")),
    ("ᵻ", (["AH"], "ᵻ")),  # KIT/schwa alternation ("roses", "-in" suffixes)
    ("ᵿ", (["UH"], "ᵿ")),  # FOOT/schwa alternation, the same idea for /ʊ~ə/
    ("i", (["IY"], "i")),  # weak "happY" vowel, written without length mark
    ("u", (["UW"], "u")),  # weak "influence"-type vowel, ditto
    ("a", (["AA"], "a")),
]

IPA_CONSONANTS: list[tuple[str, tuple[list[str], str]]] = [
    ("tʃ", (["CH"], "tʃ")),
    ("dʒ", (["JH"], "dʒ")),
    ("hw", (["HH", "W"], "hw")),
    ("p", (["P"], "p")),
    ("b", (["B"], "b")),
    ("t", (["T"], "t")),
    ("d", (["D"], "d")),
    ("k", (["K"], "k")),
    ("ɡ", (["G"], "ɡ")),
    ("g", (["G"], "ɡ")),
    ("f", (["F"], "f")),
    ("v", (["V"], "v")),
    ("θ", (["TH"], "θ")),
    ("ð", (["DH"], "ð")),
    ("s", (["S"], "s")),
    ("z", (["Z"], "z")),
    ("ʃ", (["SH"], "ʃ")),
    ("ʒ", (["ZH"], "ʒ")),
    ("h", (["HH"], "h")),
    ("m", (["M"], "m")),
    ("n", (["N"], "n")),
    ("ŋ", (["NG"], "ŋ")),
    ("l", (["L"], "l")),
    ("r", (["R"], "r")),
    ("j", (["Y"], "j")),
    ("w", (["W"], "w")),
    ("x", (["HH"], "x")),  # no velar fricative in ARPABET; closest is /h/
]

IPA_TABLE = IPA_VOWELS + IPA_CONSONANTS

_IPA_STRIP = dict.fromkeys(map(ord, "/[]().‿ "), None)


class NotationError(ValueError):
    pass


def _clean_ipa(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(_IPA_STRIP)
    # Drop combining marks (tie bars, diacritics) that the table doesn't need,
    # but keep the (non-combining) stress and length marks.
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace(".", "")
    return text


def _ipa_segments(text: str) -> list[tuple[list[str], str]]:
    """Greedy longest-match tokenisation of a cleaned IPA string."""
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in (PRIMARY, SECONDARY):
            out.append(([ch], ""))
            i += 1
            continue
        for graph, mapped in IPA_TABLE:
            if text.startswith(graph, i):
                out.append(mapped)
                i += len(graph)
                break
        else:
            i += 1  # unrecognised symbol (tone marks, etc.) -- skip it
    return out


def ipa_to_arpabet_ipa(text: str) -> tuple[str, str]:
    """One IPA transcription (slash/bracket form, or IPAc-en args re-joined)
    to (ARPABET string, IPA string)."""
    cleaned = _clean_ipa(text)
    if not cleaned:
        raise NotationError(f"nothing usable in {text!r}")

    segments = _ipa_segments(cleaned)
    arpa: list[str] = []
    ipa: list[str] = []
    pending_stress = 0
    saw_primary = False
    first_vowel_index: int | None = None

    for arpa_syms, ipa_sym in segments:
        if arpa_syms == [PRIMARY]:
            pending_stress = 1
            saw_primary = True
            continue
        if arpa_syms == [SECONDARY]:
            pending_stress = 2
            continue

        used_stress = False
        for sym in arpa_syms:
            if sym in VOWEL_ARPA:
                stress = pending_stress if not used_stress else 0
                arpa.append(f"{sym}{stress}")
                if first_vowel_index is None:
                    first_vowel_index = len(arpa) - 1
                used_stress = True
            else:
                arpa.append(sym)
        if used_stress:
            pending_stress = 0
        ipa.append(ipa_sym)

    if not arpa:
        raise NotationError(f"no phonemes recovered from {text!r}")

    if not saw_primary and first_vowel_index is not None:
        sym = arpa[first_vowel_index]
        arpa[first_vowel_index] = f"{sym[:-1]}1"

    return " ".join(arpa), "".join(ipa)


def ipac_en_args_to_arpabet_ipa(args: list[str]) -> tuple[str, str]:
    """`{{IPAc-en|...}}` template arguments (already split, params dropped)."""
    return ipa_to_arpabet_ipa("".join(args))


# --- Wikipedia respelling key -----------------------------------------------
#
# Wikipedia:Help:Pronunciation respelling key, transcribed directly from that
# page (fetched 2026-09-18). Longest grapheme match first; syllables are
# ALL CAPS for a stressed syllable, lower-case for unstressed, and the key
# does not distinguish primary from secondary stress ("the difference is
# automatic"), so the first stressed syllable found is treated as primary and
# any further stressed syllables as secondary.
RESPELL_VOWELS: list[tuple[str, tuple[list[str], str]]] = [
    ("uurr", (["UH", "R"], "ʊr")),
    ("air", (["EH", "R"], "ɛər")),
    ("arr", (["AE", "R"], "ær")),
    ("eer", (["IH", "R"], "ɪər")),
    ("err", (["EH", "R"], "ɛr")),
    ("eye", (["AY"], "aɪ")),
    ("ire", (["AY", "ER"], "aɪər")),
    ("irr", (["IH", "R"], "ɪr")),
    ("oir", (["OY", "R"], "ɔɪər")),
    ("oor", (["UH", "R"], "ʊər")),
    ("orr", (["AO", "R"], "ɒr")),
    ("our", (["AW", "ER"], "aʊər")),
    ("ure", (["Y", "UH", "R"], "jʊər")),
    ("urr", (["AH", "R"], "ʌr")),
    ("ah", (["AA"], "ɑː")),
    ("ar", (["AA", "R"], "ɑːr")),
    ("aw", (["AO"], "ɔː")),
    ("ay", (["EY"], "eɪ")),
    ("ee", (["IY"], "iː")),
    ("eh", (["EH"], "ɛ")),
    ("ew", (["Y", "UW"], "juː")),
    ("ih", (["IH"], "ɪ")),
    ("oh", (["OW"], "oʊ")),
    ("oo", (["UW"], "uː")),
    ("or", (["AO", "R"], "ɔːr")),
    ("ow", (["AW"], "aʊ")),
    ("oy", (["OY"], "ɔɪ")),
    ("uh", (["AH"], "ʌ")),
    ("ur", (["ER"], "ɜːr")),
    ("uu", (["UH"], "ʊ")),
    ("ər", (["ER"], "ər")),
    ("a", (["AE"], "æ")),
    ("e", (["EH"], "ɛ")),
    ("i", (["IH"], "ɪ")),
    ("o", (["AA"], "ɒ")),
    ("u", (["AH"], "ʌ")),
    ("ə", (["AH"], "ə")),
]

RESPELL_CONSONANTS: list[tuple[str, tuple[list[str], str]]] = [
    ("tch", (["CH"], "tʃ")),
    ("ch", (["CH"], "tʃ")),
    ("dh", (["DH"], "ð")),
    ("gh", (["G"], "ɡ")),
    ("kh", (["HH"], "x")),
    ("ng", (["NG"], "ŋ")),
    ("nk", (["NG", "K"], "ŋk")),
    ("sh", (["SH"], "ʃ")),
    ("ss", (["S"], "s")),
    ("th", (["TH"], "θ")),
    ("wh", (["HH", "W"], "hw")),
    ("zh", (["ZH"], "ʒ")),
    ("b", (["B"], "b")),
    ("d", (["D"], "d")),
    ("f", (["F"], "f")),
    ("g", (["G"], "ɡ")),
    ("h", (["HH"], "h")),
    ("j", (["JH"], "dʒ")),
    ("k", (["K"], "k")),
    ("l", (["L"], "l")),
    ("m", (["M"], "m")),
    ("n", (["N"], "n")),
    ("p", (["P"], "p")),
    ("r", (["R"], "r")),
    ("s", (["S"], "s")),
    ("t", (["T"], "t")),
    ("v", (["V"], "v")),
    ("w", (["W"], "w")),
    ("z", (["Z"], "z")),
]

# "y" is genuinely ambiguous in the key: word-initially/finally it is the
# /aɪ/ nucleus ("eye", "sky"); elsewhere it is the /j/ onset ("you"). Resolved
# contextually in `_respell_segments` rather than placed in the table above.
RESPELL_TABLE = RESPELL_VOWELS + RESPELL_CONSONANTS


def _respell_segments(syllable: str) -> list[tuple[list[str], str]]:
    out = []
    i = 0
    n = len(syllable)
    while i < n:
        if syllable[i] == "y":
            if i + 1 < n:
                out.append((["Y"], "j"))
            else:
                out.append((["AY"], "aɪ"))
            i += 1
            continue
        for graph, mapped in RESPELL_TABLE:
            if syllable.startswith(graph, i):
                out.append(mapped)
                i += len(graph)
                break
        else:
            i += 1  # stray punctuation
    return out


def respell_to_arpabet_ipa(syllables: list[str]) -> tuple[str, str]:
    """`{{respell|...}}` template arguments (already split, params dropped)."""
    cleaned = [re.sub(r"[^a-zA-Z]", "", s) for s in syllables]
    cleaned = [s for s in cleaned if s]
    if not cleaned:
        raise NotationError(f"nothing usable in {syllables!r}")

    arpa: list[str] = []
    ipa: list[str] = []
    saw_primary = False
    first_vowel_index: int | None = None
    stress_seen = 0

    for syllable in cleaned:
        stressed = syllable.isupper() and syllable.lower() != syllable
        stress = 0
        if stressed:
            stress_seen += 1
            stress = 1 if stress_seen == 1 else 2
            if stress == 1:
                saw_primary = True

        used_stress = False
        for arpa_syms, ipa_sym in _respell_segments(syllable.lower()):
            for sym in arpa_syms:
                if sym in VOWEL_ARPA:
                    arpa.append(f"{sym}{stress if not used_stress else 0}")
                    if first_vowel_index is None:
                        first_vowel_index = len(arpa) - 1
                    used_stress = True
                else:
                    arpa.append(sym)
            ipa.append(ipa_sym)

    if not arpa:
        raise NotationError(f"no phonemes recovered from {syllables!r}")

    if not saw_primary and first_vowel_index is not None:
        sym = arpa[first_vowel_index]
        arpa[first_vowel_index] = f"{sym[:-1]}1"

    return " ".join(arpa), "".join(ipa)
