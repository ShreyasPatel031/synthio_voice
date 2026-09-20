"""Merriam-Webster pronunciation notation to IPA and ARPABET.

This is MW's **own published** phonetic alphabet (macrons, schwa, stress
marks), not Wikipedia-key G2P of DailyMed/USAN ASCII. Do not reuse this
table on `DU-pix-ent`. See `dose_r/references/README.md`.

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
    """MW's comma-separated alternates, with partial variants expanded
    against the full form they abbreviate.

    MW abbreviates an alternate from either end. A LEADING "-"
    ("flü-ˈtik-ə-ˌsōn, -ˌzōn") replaces only the trailing syllables; a
    TRAILING "-" ("ə-ˌsēt-ə-ˈmin-ə-fən, ˌas-ət-") replaces only the
    leading ones, and is just as common. Only the first was handled
    before, so a trailing-"-" alternate was emitted as the bare fragment
    it looks like in isolation -- "ˌas-ət-" became the two-syllable
    "ˈˌæsət", stored as if it were a real, complete alternate
    pronunciation of "acetaminophen". That is worse than merely useless
    as ground truth: `phonetic_scorer.score_against_reference` keeps the
    BEST-matching variant, so a fragment can only ever make scoring more
    lenient -- a system that said just "asset" would have matched it
    almost exactly and scored as correct. Expanding it against the base
    form instead recovers what MW actually meant
    ("ˌas-ət-ə-ˈmin-ə-fən", the real second pronunciation).
    """
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return []

    out = [parts[0]]
    for part in parts[1:]:
        if part.startswith("-") and part.endswith("-"):
            # Truncated at BOTH ends ("-ˈmē-prə-" in esomeprazole's
            # "ˌes-ō-ˈmep-rə-ˌzōl, -ˈmē-prə-, -ˌzȯl"): a middle
            # replacement, and how many base syllables sit on each side of
            # it is genuinely ambiguous from the notation alone -- three
            # remaining base syllables could split 1+2 or 2+1. Treating it
            # as a leading-"-" alternate (what happened before) silently
            # dropped the tail instead, emitting "ˌɛsoʊˈmiːprə" for a word
            # that ends in "-zole". Skipped rather than guessed at: losing
            # one real alternate is recoverable, a wrong one recorded as
            # ground truth is not.
            continue
        if part.startswith("-"):
            tail = part.lstrip("-")
            head = out[0].rsplit("-", tail.count("-") + 1)[0]
            out.append(f"{head}-{tail}")
        elif part.endswith("-"):
            head = part.rstrip("-")
            base = out[0].split("-")
            tail = base[head.count("-") + 1 :]
            out.append("-".join([head, *tail]) if tail else head)
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
    """One MW respelling to (ARPABET string, IPA string).

    The output IPA now carries a stress mark per stressed syllable, the
    same as the output ARPABET's stress digit -- MW's own `ˈ`/`ˌ` already
    marks syllable-initial stress in its input, in IPA's own convention (on
    the syllable, not the vowel), so this only needs to carry that mark
    through to the output instead of consuming it for ARPABET alone and
    discarding it. Confirmed a real, near-total gap before this fix: none
    of the 88 Merriam-Webster-sourced records in this project's reference
    set had a stress mark in their stored IPA.
    """
    # A parenthesised stress mark is MW's notation for an OPTIONAL stress
    # ("(ˈ)dī-ˈhī-ˌdrāt" for dihydrate, where the first syllable may or may
    # not take one). `_DROP` strips the parentheses but not the mark inside
    # them, which left two literal primaries in one word -- malformed, and
    # exactly what the per-word "more primaries than words" check flags.
    # The word's other, unparenthesised mark is the real one, so the
    # optional one is dropped whole.
    cleaned = re.sub(r"\([ˈˌ]\)", "", unicodedata.normalize("NFC", raw))
    cleaned = cleaned.translate(_DROP)
    cleaned = re.sub(r"[^\wˈˌəᵊāēīōüȯäŋ̸̇\-]", "", cleaned)
    if not cleaned:
        raise NotationError(f"nothing usable in {raw!r}")

    arpa: list[str] = []
    ipa: list[str] = []
    saw_primary = False
    first_vowel_syllable_start: int | None = None
    first_vowel_seen = False

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

        syllable_start = len(ipa)
        nucleus_done = False
        for arpa_sym, ipa_sym in _segments(syllable):
            if arpa_sym in VOWEL_ARPA:
                arpa.append(f"{arpa_sym}{stress if not nucleus_done else 0}")
                if not first_vowel_seen:
                    first_vowel_seen = True
                    first_vowel_syllable_start = syllable_start
                nucleus_done = True
            else:
                arpa.append(arpa_sym)
            ipa.append(ipa_sym)

        if stress == 1:
            ipa.insert(syllable_start, PRIMARY)
        elif stress == 2:
            ipa.insert(syllable_start, SECONDARY)

    if not arpa:
        raise NotationError(f"no phonemes recovered from {raw!r}")

    if not saw_primary:
        for i, sym in enumerate(arpa):
            if sym[:-1] in VOWEL_ARPA and sym[-1].isdigit():
                arpa[i] = f"{sym[:-1]}1"
                break
        if first_vowel_syllable_start is not None:
            # That syllable may already carry a SECONDARY mark, which is
            # exactly the case this fallback exists for -- MW writes some
            # entries with only secondaries and no primary at all
            # ("ˌsər-trə-ˌlēn" for sertraline). Promote that mark to
            # primary rather than stacking a second one in front of it,
            # which would emit the malformed "ˈˌ".
            if ipa[first_vowel_syllable_start : first_vowel_syllable_start + 1] == [SECONDARY]:
                ipa[first_vowel_syllable_start] = PRIMARY
            else:
                ipa.insert(first_vowel_syllable_start, PRIMARY)

    return " ".join(arpa), "".join(ipa)


def convert(raw: str) -> list[tuple[str, str]]:
    """Every accepted variant in an MW entry, as (ARPABET, IPA) pairs.

    Note that an MW entry for a multi-word term writes both words as one
    hyphen run ("bə-ˈläk-sə-ˌvir-mär-ˈbäk-səl" for baloxavir marboxil),
    and MW's notation marks no word boundary within it: the second word's
    own primary stress shows that one EXISTS, but not where it starts, and
    its unstressed onset syllable ("mär-") is indistinguishable from a
    continuation of the first word. Splitting at the second primary was
    tried and is wrong -- a single word's primary need not fall on its
    first syllable ("ə-ˌtȯr-və-ˈsta-tᵊn", atorvastatin), so that rule tears
    ordinary single words in half. Such a variant is left as the converter
    produces it here and dropped downstream instead; see
    `build._is_malformed_multiword`.
    """
    out = []
    for variant in split_variants(raw):
        try:
            out.append(to_arpabet_ipa(variant))
        except NotationError:
            continue
    return out
