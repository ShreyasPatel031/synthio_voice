"""TTS pronunciation helpers.

BANNED: `ipa_from_canonical` — converting canonical respelling to IPA.
That G2P produced Dupixent `ˈdʌpɪksɛnt` ("duh") from `DU-pix-ent`.
Use the original source string. IPA sidecar is allowed only when the
source actually published IPA. See `dose_r/references/README.md`.

`compact_ascii` still exists (hyphens stripped). Do not feed hyphenated
respelling to Cloud TTS; that inserts pauses.
"""

from __future__ import annotations

import re

from .wiki_notation import RespellToIpaBanned, _BAN

# IPA nuclei / stress. ASCII aeiou alone is not enough — that let
# `co-BEN-fee` through as if it were IPA.
_IPA_MARK = re.compile(r"[ɪʊɛæɑɒɔʌəɚɝʃʒθðŋɡːˈˌɹɐɨᵻ]")
_IPA_VOWEL = "iɪeɛæaɑɒɔoʊuʌəɚɝɨᵻyɐ"
_IPA_CONS = "bdfghjklmnpqrstvwxyzʃʒθðŋɡɹɾ"


def compact_ascii(canonical: str) -> str:
    """Drop hyphens inside each word; keep real word spaces.

    `ak-oh-RAM-id-is` -> `akohramidis`
    `TROSE-pee-um chloride` -> `trosepeeum chloride`
    """
    return " ".join(word.replace("-", "").lower() for word in canonical.split() if word)


def ipa_from_canonical(canonical: str) -> str:
    """Dead. Never G2P a dictionary respelling into IPA."""
    raise RespellToIpaBanned(_BAN)


def to_cloud_en_us_ipa(ipa: str) -> str:
    """Pass source IPA through. Do not rewrite phones."""
    return ipa


def is_source_ipa(s: str) -> bool:
    """True for published IPA. False for respelling or howtopronounce junk.

    Does not rewrite the string. Does not G2P `DU-pix-ent` / `co-BEN-fee`.
    """
    text = (s or "").strip().strip("/[]() ")
    if not text:
        return False
    if re.search(r"[āēīōüăěŏ\"“”]", text):
        return False
    if ".." in text:
        return False
    # DailyMed / Drugs.com / MW hyphen respelling, not IPA (IPA uses .).
    if "-" in text or "–" in text or re.search(r"[A-Z]{2,}", text):
        return False
    if not _IPA_MARK.search(text):
        return False
    # Spanish/phonology-page scrapes (`ʝ̞`, `aˈʝ̞eɾ`) are not English drug IPA.
    if re.search(r"[ʝχβɲʎɣʔɓɗ]|[̀-ͯ]|̞|̩|̯", text):
        return False
    # A full word, not a Help:IPA diphthong snippet (`aɪ`) scraped off Wikipedia.
    if len(re.findall(rf"[{_IPA_VOWEL}]+", text, re.I)) < 2:
        return False
    # howtopronounce puts stress inside the onset: spˈɪ, sˈæ, nˈɜ, skˈaɪ.
    # Real IPA puts the mark at the start of the syllable (`spɪˈriːvə`).
    if re.search(
        rf"(?:^|[^{_IPA_VOWEL}])[{_IPA_CONS}][ˈˌ][{_IPA_VOWEL}]",
        text,
        re.I,
    ):
        return False
    for syl in re.split(r"[. ]+", re.sub(r"[ˈˌ]", "", text)):
        if not syl:
            continue
        if not re.search(rf"[{_IPA_VOWEL}]", syl, re.I):
            return False
        onset = re.match(rf"[^{_IPA_VOWEL}]*", syl, re.I)
        cons = onset.group(0) if onset else ""
        if len(cons) >= 3 and cons[0] not in "sSʃ":
            return False
    return True


def page_is_for_word(word: str, url: str) -> bool:
    """False when the page is a different lemma (Utebzi → Wiktionary `utzi`).

    Wikipedia/Wiktionary title must contain the queried word. Opaque paths
    (`leskoff.com/s01710-0`) are left alone. Does not G2P anything.
    """
    if not url:
        return True
    token = re.sub(r"[^a-z0-9]+", "", word.lower())
    if not token:
        return False
    from urllib.parse import unquote, urlparse

    parsed = urlparse(url)
    path = unquote(parsed.path or "").lower()
    host = (parsed.netloc or "").lower()
    if "wikipedia.org" in host or "wiktionary.org" in host:
        if "/wiki/" not in path:
            return False
        lemma = re.sub(r"[^a-z0-9]+", "", path.split("/wiki/", 1)[-1])
        return bool(lemma) and (token == lemma or token in lemma)
    return True


# IPA → X-SAMPA for Cloud `PHONETIC_ENCODING_X_SAMPA`. Longest match first.
# This is a rewrite of *already-published IPA*, not G2P of a respelling.
_IPA_TO_XSAMPA: tuple[tuple[str, str], ...] = (
    ("tʃ", "tS"),
    ("dʒ", "dZ"),
    ("aɪ", "aI"),
    ("aʊ", "aU"),
    ("eɪ", "eI"),
    ("oʊ", "oU"),
    ("ɔɪ", "OI"),
    ("ɪə", "I@"),
    ("ʊə", "U@"),
    ("ɛə", "E@"),
    ("ɑː", "A:"),
    ("ɔː", "O:"),
    ("uː", "u:"),
    ("iː", "i:"),
    ("eː", "e:"),
    ("ɜː", "3:"),
    ("oː", "o:"),
    ("ɝ", "3`"),
    ("ɚ", "@`"),
    ("ɹ", "r"),
    ("æ", "{"),
    ("ə", "@"),
    ("ɪ", "I"),
    ("ʊ", "U"),
    ("ɛ", "E"),
    ("ɑ", "A"),
    ("ɔ", "O"),
    ("ʌ", "V"),
    ("ɒ", "Q"),
    ("ʃ", "S"),
    ("ʒ", "Z"),
    ("θ", "T"),
    ("ð", "D"),
    ("ŋ", "N"),
    ("ɡ", "g"),
    ("ᵻ", "I"),
    ("ᵿ", "U"),
    ("ɐ", "6"),
    ("ˈ", '"'),
    ("ˌ", "%"),
    ("ː", ":"),
)


def ipa_to_xsampa(ipa: str) -> str:
    """Map source IPA to X-SAMPA for Cloud customPronunciations."""
    out: list[str] = []
    i = 0
    while i < len(ipa):
        hit = None
        for src, dst in _IPA_TO_XSAMPA:
            if ipa.startswith(src, i):
                hit = (src, dst)
                break
        if hit:
            out.append(hit[1])
            i += len(hit[0])
        else:
            out.append(ipa[i])
            i += 1
    return "".join(out)


# FDA biologic/biosimilar suffix is four ASCII letters after a hyphen
# (`teplizumab-mzwv`). Devoid of meaning; not spoken. Real extra words
# (`chloride`, `alfa`) stay.
_FDA_WORD = re.compile(r"^(.+)-([A-Za-z]{4})$")

# English letter *names* for that suffix only. Not drug-name G2P, not the
# banned Wikipedia respelling key. `mzwv` is four letters of the alphabet.
_LETTER_NAME_IPA = {
    "a": "eɪ",
    "b": "biː",
    "c": "siː",
    "d": "diː",
    "e": "iː",
    "f": "ɛf",
    "g": "dʒiː",
    "h": "eɪtʃ",
    "i": "aɪ",
    "j": "dʒeɪ",
    "k": "keɪ",
    "l": "ɛl",
    "m": "ɛm",
    "n": "ɛn",
    "o": "oʊ",
    "p": "piː",
    "q": "kjuː",
    "r": "ɑɹ",
    "s": "ɛs",
    "t": "tiː",
    "u": "juː",
    "v": "viː",
    "w": "dʌbəlju",
    "x": "ɛks",
    "y": "waɪ",
    "z": "ziː",
}


def name_parts(name: str) -> list[str]:
    """Whitespace words, with a trailing FDA 4-letter suffix split off.

    The suffix is not spoken (`spoken_parts`). Split so callers can drop it.
    Does not convert respelling to IPA.
    """
    parts: list[str] = []
    for word in name.split():
        matched = _FDA_WORD.match(word)
        if matched:
            parts.append(matched.group(1))
            parts.append(matched.group(2))
        else:
            parts.append(word)
    return parts


def is_fda_letter_suffix(part: str, name: str) -> bool:
    """True when `part` is the 4-letter FDA suffix of this name."""
    bits = name_parts(name)
    return bool(bits) and part == bits[-1] and _FDA_WORD.match(name.split()[-1]) is not None


def spoken_parts(name: str) -> list[str]:
    """Words Cloud should say: drop FDA 4-letter codes, keep real words."""
    return [p for p in name_parts(name) if not is_fda_letter_suffix(p, name)]


def spoken_text(name: str) -> str:
    """Written name without FDA suffixes so Cloud does not letter-spell them."""
    bits: list[str] = []
    for word in name.split():
        matched = _FDA_WORD.match(word)
        bits.append(matched.group(1) if matched else word)
    return " ".join(bits)


def respelling_alias(canonical: str, name: str) -> str:
    """English syllables Cloud should read from the source respelling.

    `a-TA-ki-sept` → `a ta ki sept`; `zoe-li-floe-DAY-sin` → `zoe li floe day sin`.
    Hyphens become spaces so Standard-C does not spell letters. Never IPA.
    Never Wikipedia-key G2P. FDA 4-letter suffixes are not spoken.
    Extra name words the respelling omitted (`alfa`, `propionate`) stay English.
    """
    words: list[list[str]] = []
    for word in (canonical or "").split():
        pieces = [
            piece.strip().lower()
            for piece in re.split(r"[-·•]+", word)
            if piece.strip()
        ]
        if pieces:
            words.append(pieces)
    pieces = [p for group in words for p in group]
    n_words = name.split()
    c_words = (canonical or "").split()
    if len(n_words) > len(c_words):
        for word in n_words[len(c_words) :]:
            matched = _FDA_WORD.match(word)
            leftover = matched.group(1) if matched else word
            if leftover.lower() not in {p.lower() for p in pieces}:
                pieces.append(leftover.lower())
    return " ".join(pieces)


def letter_code_ipa(code: str) -> str:
    """IPA of English letter names for an FDA 4-letter suffix.

    Independent of DailyMed/USAN respelling. Do not use this for drug stems.
    """
    out: list[str] = []
    for ch in code.lower():
        ipa = _LETTER_NAME_IPA.get(ch)
        if not ipa:
            raise ValueError(f"not an English letter: {code!r}")
        out.append(ipa)
    return "".join(out)


def custom_pronunciations_for_parts(part_ipas: list[tuple[str, str]]) -> dict:
    """One Cloud phrase per name part. Phrase = exact substring of spoken text."""
    return {
        "pronunciations": [
            {
                "phrase": phrase,
                "phoneticEncoding": "PHONETIC_ENCODING_IPA",
                "pronunciation": ipa,
            }
            for phrase, ipa in part_ipas
            if phrase and ipa
        ]
    }


def custom_pronunciation(
    phrase: str,
    pronunciation: str,
    encoding: str = "PHONETIC_ENCODING_IPA",
) -> dict:
    """Cloud TTS `SynthesisInput.customPronunciations` block.

    `phrase` must be an exact substring of the text being synthesized
    (the ingredient name). `pronunciation` must be source-published IPA
    (or its X-SAMPA rewrite), never Wikipedia-key G2P of DailyMed/USAN.

    `encoding` is a Cloud `PhoneticEncoding` value:
    https://docs.cloud.google.com/text-to-speech/docs/reference/rest/Shared.Types/PhoneticEncoding
    """
    return {
        "pronunciations": [
            {
                "phrase": phrase,
                "phoneticEncoding": encoding,
                "pronunciation": pronunciation,
            }
        ]
    }
