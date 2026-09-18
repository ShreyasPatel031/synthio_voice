"""USAN/INN stem rules: a hand-encoded, documented alternative to `g2p.py` for
the tail of a generic drug name, used only when nothing else has grounded it.

The US Adopted Names (USAN) Council and WHO's International Nonproprietary
Names (INN) programme assign generic drug names out of a controlled
vocabulary of morphological stems, each carrying an established pronunciation
and stress convention -- see the USAN Stem List
(https://www.ama-assn.org/about/united-states-adopted-names/naming-guidelines
-usan-stems) and the WHO INN stem list
(https://www.who.int/teams/health-product-and-policy-standards/inn/stem-book).
This is domain knowledge, not a guess: every stem below is a published,
citable convention, not a plausible-sounding pattern invented for this file.

Each `Stem` fixes the ARPABET pronunciation of the suffix itself and where
stress falls relative to it; the word's ROOT -- whatever precedes the stem --
is still run through the existing `g2p.py` engine, because the stem rule's
whole value is fixing the ending, which is exactly where that generic engine
is weakest (it has no notion of Latinate/Greek stress-shifting suffixes and
treats every vowel as if it were ordinary English spelling).

Three stress modes cover every stem here, chosen per stem from its real,
independently-sourced examples in `references.jsonl` (see
`backtest_stems.py` and `STEM_BACKTEST.md` for the measurements that justify
each choice):

    "before"  primary stress falls on the ROOT's last syllable; the stem's
              own final syllable takes secondary stress, earlier stem
              syllables are unstressed. This is the classic USAN convention
              ("stress the syllable before the stem") and covers `-mab`,
              `-tinib`/`-inib`, `-zole`, `-vir`/`-ciclovir`, `-pril`, `-olol`,
              `-dronate`.
    "stem"    the stem carries its OWN fixed primary stress on a named
              syllable inside itself, independent of the root; the root's own
              last syllable is demoted to secondary. Covers `-statin` and
              `-sartan`, whose stress is on "sta"/"sar" regardless of what
              precedes them (compare "atorvastatin", "valsartan").
    "keep"    the stem does not move stress at all; only the segmental
              (phoneme) identity of the ending is corrected, and the word's
              natural, length-driven stress placement (the same algorithm
              `g2p.py` already uses) is left alone. Covers `-tide`, whose
              only well-documented property is that it is pronounced to
              rhyme with the English word "tide" -- there is no documented
              stem-relative stress rule for peptide names, and the sourced
              examples in this dataset do not agree on one either.

A stem that turned out, on back-test, not to help (or to hurt) is dropped
from this table entirely rather than kept as an untested or disproven rule --
see STEM_BACKTEST.md for the ones that did not make the cut.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..judge.fixtures import g2p

FDA_SUFFIX = re.compile(r"-[a-z]{4}$")  # FDA biosimilar/biologic distinguishing suffix


@dataclass(frozen=True)
class Stem:
    suffix: str
    tail: tuple[str, ...]
    mode: str  # "before" | "stem" | "keep"
    citation: str
    primary_index: int | None = None  # nucleus index within `tail`, for mode "stem"
    tested: bool = False  # has at least one sourced example in this dataset


# Ordered by documentation confidence, not alphabetically. Every entry names
# the stem class and a real-world prototype the pronunciation is attested on.
STEMS: dict[str, Stem] = {
    # The monoclonal-antibody stem is really the SOURCE substem
    # ("-umab" human, "-zumab" humanized, "-ximab" chimeric, "-omab" murine),
    # not bare "-mab": the linker vowel immediately before "mab" is part of
    # the stem's own syllable, and a back-test against the bare 3-letter
    # "-mab" cut that syllable in half and put stress on the linker vowel
    # instead of the true root syllable before it (see STEM_BACKTEST.md,
    # where bare "-mab" measurably made things worse and was dropped).
    "umab": Stem(
        suffix="umab",
        tail=("Y", "UH", "M", "AE", "B"),
        mode="before",
        citation=(
            "USAN substem '-umab', fully human monoclonal antibodies (e.g. "
            "adalimumab): pronounced /juːmæb/, stress on the syllable "
            "before it."
        ),
        tested=True,
    ),
    "zumab": Stem(
        suffix="zumab",
        tail=("Z", "AH", "M", "AE", "B"),
        mode="before",
        citation=(
            "USAN substem '-zumab', humanized monoclonal antibodies (e.g. "
            "omalizumab): pronounced /zəmæb/, stress on the syllable before "
            "it."
        ),
        tested=True,
    ),
    "ximab": Stem(
        suffix="ximab",
        tail=("Z", "IH", "M", "AE", "B"),
        mode="before",
        citation=(
            "USAN substem '-ximab', chimeric monoclonal antibodies (e.g. "
            "rituximab, infliximab): the AMA/USAN pronunciation key notates "
            "the 'x' as /z/; stress on the syllable before it. Not present "
            "in this dataset with a source to test against."
        ),
        tested=False,
    ),
    "omab": Stem(
        suffix="omab",
        tail=("OW", "M", "AE", "B"),
        mode="before",
        citation=(
            "USAN substem '-omab', murine monoclonal antibodies (e.g. "
            "muromonab): stress on the syllable before it. Not present in "
            "this dataset."
        ),
        tested=False,
    ),
    "tinib": Stem(
        suffix="tinib",
        tail=("T", "IH", "N", "IH", "B"),
        mode="before",
        citation=(
            "USAN stem '-tinib', kinase inhibitors (e.g. imatinib, "
            "osimertinib): stress on the syllable before the stem, "
            "secondary stress on the stem's own final syllable."
        ),
        tested=True,
    ),
    "inib": Stem(
        suffix="inib",
        tail=("IH", "N", "IH", "B"),
        mode="before",
        citation="USAN stem '-inib' (kinase inhibitors not using the '-tinib' variant); same convention as '-tinib'.",
        tested=False,
    ),
    "prazole": Stem(
        suffix="prazole",
        tail=("P", "R", "AE", "Z", "OW", "L"),
        mode="before",
        citation=(
            "USAN stem '-prazole', proton-pump inhibitors (e.g. omeprazole, "
            "esomeprazole): the 'pra' syllable is always unstressed and "
            "'zole' rhymes with 'hole', stress on the syllable before "
            "'-prazole' as a whole. A back-test against the shorter, bare "
            "'-zole' suffix (which cuts into the 'a' linker the same way "
            "bare '-mab' cuts into its linker) measurably hurt these three "
            "words, which is why '-prazole' is its own entry rather than "
            "folded into '-zole' (see STEM_BACKTEST.md)."
        ),
        tested=True,
    ),
    "zole": Stem(
        suffix="zole",
        tail=("Z", "OW", "L"),
        mode="before",
        citation=(
            "USAN stem '-azole', antifungal/other azole-ring drugs not "
            "using the '-prazole' PPI variant (e.g. riluzole): 'zole' "
            "rhymes with 'hole', stress on the syllable before it."
        ),
        tested=False,
    ),
    "gliflozin": Stem(
        suffix="gliflozin",
        tail=("G", "L", "IH", "F", "L", "OW", "Z", "IH", "N"),
        mode="stem",
        primary_index=1,  # the "flo" syllable
        citation=(
            "USAN stem '-gliflozin', SGLT2 inhibitors (e.g. dapagliflozin, "
            "empagliflozin): primary stress on 'flo' regardless of the root."
        ),
        tested=True,
    ),
    "statin": Stem(
        suffix="statin",
        tail=("S", "T", "AE", "T", "IH", "N"),
        mode="stem",
        primary_index=0,  # the "sta" syllable
        citation=(
            "USAN stem '-statin', HMG-CoA reductase inhibitors (e.g. "
            "atorvastatin, rosuvastatin): primary stress on 'sta'."
        ),
        tested=True,
    ),
    "sartan": Stem(
        suffix="sartan",
        tail=("S", "AA", "R", "T", "AE", "N"),
        mode="stem",
        primary_index=0,  # the "sar" syllable
        citation=(
            "USAN stem '-sartan', angiotensin II receptor antagonists (e.g. "
            "losartan, valsartan): primary stress on 'sar'."
        ),
        tested=True,
    ),
    "tide": Stem(
        suffix="tide",
        tail=("T", "AY", "D"),
        mode="keep",
        citation=(
            "USAN stem '-tide', peptides (e.g. semaglutide, exenatide): "
            "always pronounced to rhyme with 'tide' ('-tid' is wrong); no "
            "documented stem-relative stress rule, so word stress is left "
            "to the normal length-driven placement."
        ),
        tested=True,
    ),
    "ciclovir": Stem(
        suffix="ciclovir",
        tail=("S", "IH", "K", "L", "OW", "V", "IH", "R"),
        mode="before",
        citation=(
            "USAN stem '-ciclovir', nucleoside antiviral analogues (e.g. "
            "acyclovir, ganciclovir): rhymes with 'fir', stress before the "
            "stem. Not present in this dataset; included for completeness."
        ),
        tested=False,
    ),
    "vir": Stem(
        suffix="vir",
        tail=("V", "IH", "R"),
        mode="before",
        citation=(
            "USAN stem '-vir', antivirals (e.g. acyclovir, oseltamivir): "
            "rhymes with 'fir', stress on the syllable before it."
        ),
        tested=False,
    ),
    "pril": Stem(
        suffix="pril",
        tail=("P", "R", "IH", "L"),
        mode="before",
        citation=(
            "USAN stem '-pril', ACE inhibitors (e.g. captopril, "
            "lisinopril): stress before the stem. Not present in this "
            "dataset; included for completeness."
        ),
        tested=False,
    ),
    "olol": Stem(
        suffix="olol",
        tail=("OW", "L", "AA", "L"),
        mode="before",
        citation=(
            "USAN stem '-olol', beta blockers (e.g. propranolol, "
            "atenolol): stress before the stem. Not present in this "
            "dataset; included for completeness."
        ),
        tested=False,
    ),
    "dronate": Stem(
        suffix="dronate",
        tail=("D", "R", "OW", "N", "EY", "T"),
        mode="before",
        citation=(
            "USAN stem '-dronate', bisphosphonates (e.g. alendronate, "
            "risedronate): stress before the stem. Not present in this "
            "dataset; included for completeness."
        ),
        tested=False,
    ),
}


def _is_vowel(base: str) -> bool:
    return base in g2p.ARPABET_TO_IPA and base[0] in "AEIOU"


def _split_fda_suffix(word: str) -> tuple[str, str]:
    """Peel off a trailing FDA biosimilar suffix (e.g. '-vikg'), if present.

    These four-letter codes are deliberately meaningless distinguishing
    labels with no documented pronunciation convention of their own, so they
    are never stem-matched -- only the real INN name in front of them is.
    """
    m = FDA_SUFFIX.search(word)
    return (word[: m.start()], word[m.start() :]) if m else (word, "")


def match_stem(word: str) -> Stem | None:
    """The longest recognized stem at the end of `word`, or None."""
    base, _ = _split_fda_suffix(word.lower())
    for suffix in sorted(STEMS, key=len, reverse=True):
        if base.endswith(suffix) and len(base) > len(suffix):
            return STEMS[suffix]
    return None


def _mark(phonemes: list[str], nuclei: list[int], stress: dict[int, int]) -> list[str]:
    out = list(phonemes)
    for idx in nuclei:
        out[idx] = f"{phonemes[idx]}{stress.get(idx, 0)}"
    return out


def _apply_before(root: list[str], tail: list[str]) -> list[str]:
    combined = root + tail
    root_nuclei = [i for i, p in enumerate(root) if _is_vowel(p)]
    tail_nuclei = [len(root) + i for i, p in enumerate(tail) if _is_vowel(p)]
    nuclei = root_nuclei + tail_nuclei
    stress = {i: 0 for i in nuclei}
    if root_nuclei:
        stress[root_nuclei[-1]] = 1
        if tail_nuclei:
            stress[tail_nuclei[-1]] = 2
    elif tail_nuclei:
        stress[tail_nuclei[0]] = 1
    return _mark(combined, nuclei, stress)


def _apply_stem_primary(root: list[str], tail: list[str], primary_index: int) -> list[str]:
    combined = root + tail
    root_nuclei = [i for i, p in enumerate(root) if _is_vowel(p)]
    tail_nuclei = [len(root) + i for i, p in enumerate(tail) if _is_vowel(p)]
    nuclei = root_nuclei + tail_nuclei
    stress = {i: 0 for i in nuclei}
    stress[tail_nuclei[primary_index]] = 1
    if root_nuclei:
        stress[root_nuclei[-1]] = 2
    return _mark(combined, nuclei, stress)


def _apply_keep(root: list[str], tail: list[str]) -> list[str]:
    combined = root + tail
    n_syllables = sum(1 for p in combined if _is_vowel(p))
    primary_from_end = 3 if n_syllables >= 3 else n_syllables
    return g2p._apply_stress(combined, primary_from_end)


def to_arpabet_variants(word: str) -> list[list[str]] | None:
    """`g2p.to_arpabet_variants`-compatible output, or None if no stem matches.

    The root -- everything before the recognized stem -- keeps whatever
    `g2p` would have produced for the FULL word, sliced at the stem's first
    nucleus, rather than being re-derived from the root spelling in
    isolation. Re-deriving it in isolation was tried and measurably hurt
    (see STEM_BACKTEST.md): `g2p`'s open-syllable and silent-`e` rules both
    depend on what follows a letter, so cutting the word before the stem
    changes what the ROOT's own letters look like context-free, e.g. making
    a mid-word vowel look word-final. Only the stem's own tail is
    hand-specified; a trailing FDA biosimilar suffix, if present, is
    appended verbatim via the plain `g2p` engine, since no stem rule applies
    to it.
    """
    base, code = _split_fda_suffix(word.lower())
    stem = match_stem(base)
    if stem is None:
        return None

    full_phonemes = g2p._word_to_phonemes(base)
    nuclei = [i for i, p in enumerate(full_phonemes) if _is_vowel(p)]
    tail_phonemes = list(stem.tail)
    tail_nucleus_count = sum(1 for p in tail_phonemes if _is_vowel(p))
    if len(nuclei) < tail_nucleus_count:
        return None

    leading_consonants = 0
    for p in tail_phonemes:
        if _is_vowel(p):
            break
        leading_consonants += 1

    root_nucleus_count = len(nuclei) - tail_nucleus_count
    tail_start = nuclei[root_nucleus_count] if root_nucleus_count < len(nuclei) else len(full_phonemes)
    splice_at = tail_start - leading_consonants
    root_phonemes = full_phonemes[:splice_at]

    if stem.mode == "before":
        combined = _apply_before(root_phonemes, tail_phonemes)
    elif stem.mode == "stem":
        combined = _apply_stem_primary(root_phonemes, tail_phonemes, stem.primary_index)
    else:
        combined = _apply_keep(root_phonemes, tail_phonemes)

    if code:
        combined += g2p.to_arpabet_variants(code.lstrip("-"))[0]

    return [combined]


# Measured on `references.jsonl`'s sourced (high/medium confidence) generics
# by `backtest_stems.py` -- see STEM_BACKTEST.md for the full method and the
# per-word breakdown. `improvement` is the average reduction in normalized
# PEU error versus plain `g2p.py`, positive meaning the stem engine is
# closer to the real, sourced pronunciation.
BACKTEST_RESULTS: dict[str, dict] = {
    "gliflozin": {"n": 2, "improvement": 0.0957},
    "prazole": {"n": 3, "improvement": 0.2114},
    "sartan": {"n": 1, "improvement": 0.3000},
    "statin": {"n": 2, "improvement": 0.4846},
    "tide": {"n": 6, "improvement": 0.0019},
    "tinib": {"n": 7, "improvement": 0.1755},
    "umab": {"n": 5, "improvement": -0.1086},
    "vir": {"n": 3, "improvement": 0.6685},
    "ximab": {"n": 1, "improvement": 0.1120},
    "zole": {"n": 1, "improvement": -0.8583},
    "zumab": {"n": 7, "improvement": 0.1681},
}

# Adding the AMA USAN Statement PDF as a directly-fetched, primary source
# (see sources.usan_pronunciation) rather than hoping Gemini's web search
# surfaces it moved this measurement again, mostly upward: `-ximab` and
# `-zumab` flip from measured-harmful to positive, `-tinib` and `-vir`
# strengthen substantially. This makes sense rather than looking like noise
# -- USAN's own Statement is where these stem rules were reverse-engineered
# from in the first place, so measuring the stem engine against USAN's own
# adopted-name pronunciation (now the primary source for these words,
# displacing a secondhand citation of it) is measuring the rule against the
# convention it was built to follow. `-zole` remains a sample of one
# (troriluzole) and still measures harmful; `-umab` remains harmful on a
# larger sample (n=5) despite the general upward trend elsewhere.
# Per the project rule that a stem which does not measurably help must be
# dropped rather than kept on faith, this snapshot is exactly what
# `backtest_stems.summarize(backtest_stems.backtest())` measures against the
# current `references.jsonl` -- see test_backtest_stems.py, which fails loudly
# the moment this dict drifts from that live measurement again.
DROPPED_STEMS = frozenset(
    suffix for suffix, result in BACKTEST_RESULTS.items() if result["improvement"] <= 0
)


def resolve(word: str) -> tuple[list[list[str]], str] | None:
    """Stem-rule variants and a confidence note for `word`, or None.

    None means either no stem was recognized, or the recognized stem is one
    the back-test measured as not helping -- in both cases the caller
    (`build.py`) should fall back to plain `g2p.py`.
    """
    stem = match_stem(word)
    if stem is None or stem.suffix in DROPPED_STEMS:
        return None

    variants = to_arpabet_variants(word)
    if variants is None:
        return None

    result = BACKTEST_RESULTS.get(stem.suffix)
    basis = (
        f"back-tested PEU improvement: {result['improvement']:+.2f} over "
        f"{result['n']} sourced example(s)"
        if result
        else "not back-tested: no sourced example of this stem in this dataset"
    )
    note = (
        f"stem-rule applied: -{stem.suffix} ({basis}; see STEM_BACKTEST.md). "
        "This is a documented USAN/INN naming convention, not a per-drug "
        "source -- TODO: still needs human review before it is trusted."
    )
    return variants, note
