import json
from pathlib import Path

import pytest

from dose_r.judge.phonemes import parse
from dose_r.judge.references import ReferenceSet
from dose_r.references.notation import convert, to_arpabet_ipa
from dose_r.references.wiki_notation import (
    ipa_to_arpabet_ipa,
    ipac_en_args_to_arpabet_ipa,
    respell_to_arpabet_ipa,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ROOT / "dose_r" / "references" / "references.jsonl"
DATASET = ROOT / "data" / "dose_v1.jsonl"

TIERS = {"high", "medium", "low"}


@pytest.fixture(scope="module")
def records():
    with REFERENCES.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def test_every_dose_ingredient_has_a_row(records):
    """Every DOSE ingredient gets a references.jsonl row, sourced or not --
    nothing is silently dropped from the build. Whether that row is
    *scoreable* (non-empty arpabet_variants) is a separate question, covered
    by test_reference_set_missing_is_exactly_the_unsourced_ingredients."""
    with DATASET.open() as f:
        ingredients = {i for line in f for i in json.loads(line)["ingredients"]}
    known = {r["ingredient"].lower() for r in records}
    dropped = sorted(i for i in ingredients if i.lower() not in known)
    assert dropped == [], f"missing from references.jsonl entirely: {dropped}"


def test_reference_set_missing_is_exactly_the_unsourced_ingredients(records):
    """A `low`-confidence ingredient with no ground truth (empty
    arpabet_variants) has no rule-based fallback standing in for it anymore
    -- ReferenceSet.load skips it, so it surfaces via missing() exactly like
    an ingredient that was never looked up at all. This pins down that the
    loader isn't silently dropping (or silently keeping) anything beyond
    that known set."""
    with DATASET.open() as f:
        ingredients = {i for line in f for i in json.loads(line)["ingredients"]}
    expected_missing = sorted(r["ingredient"] for r in records if not r["arpabet_variants"])
    assert ReferenceSet.load(REFERENCES).missing(ingredients) == expected_missing


def test_variant_lists_are_parallel_and_empty_only_when_unsourced(records):
    for r in records:
        assert len(r["ipa_variants"]) == len(r["arpabet_variants"]), r["ingredient"]
        if not r["arpabet_variants"]:
            assert r["confidence"] == "low", r["ingredient"]
            assert not r["sources"], r["ingredient"]


def test_every_variant_is_valid_arpabet(records):
    for r in records:
        for variant in r["arpabet_variants"]:
            parse(variant)


def test_confidence_tiers_are_declared(records):
    for r in records:
        assert r["confidence"] in TIERS, r["ingredient"]


def test_sourced_records_carry_provenance(records):
    for r in records:
        if r["confidence"] in {"high", "medium"}:
            assert r["sources"], r["ingredient"]
            for s in r["sources"]:
                assert s["name"] and s["url"]


def test_every_source_is_trust_tagged(records):
    valid = {"official_medical", "verified_secondary", "third_party_unverified"}
    for r in records:
        for s in r["sources"]:
            assert s.get("trust_tier") in valid, (r["ingredient"], s)


def test_no_source_is_third_party_unverified(records):
    """A third_party_unverified citation (a crowdsourced pronunciation
    site, a YouTube upload, a blog) must never make it into the final
    data at all -- build._variants_from_claims drops one before a
    respelling is even extracted from it, the same treatment as a claim
    that fails the format-plausibility check. This test is the guarantee,
    not a report: it should never need updating to tolerate an exception.
    """
    for r in records:
        for s in r["sources"]:
            assert s.get("trust_tier") != "third_party_unverified", (
                r["ingredient"],
                s,
            )


def test_unsourced_records_are_flagged_low_and_noted(records):
    for r in records:
        if not r["sources"]:
            assert r["confidence"] == "low", r["ingredient"]
            assert r["notes"], r["ingredient"]


def test_no_duplicate_ingredients(records):
    keys = [r["ingredient"].lower() for r in records]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize(
    "respelling,expected",
    [
        ("ə-ˌtȯr-və-ˈsta-tᵊn", "AH0 T AO2 R V AH0 S T AE1 T AH0 N"),
        ("ˈnek-sē-əm", "N EH1 K S IY0 AH0 M"),
    ],
)
def test_mw_notation_conversion(respelling, expected):
    assert to_arpabet_ipa(respelling)[0] == expected


def test_mw_trailing_variant_expands_to_a_full_form():
    variants = convert("flü-ˈtik-ə-ˌsōn, -ˌzōn")
    assert len(variants) == 2
    assert variants[0][0].replace(" S ", " Z ") == variants[1][0]


# --- Wikipedia / Wiktionary notation ---------------------------------------
#
# The three known-good cases from the project brief: a Wikipedia {{respell}}
# for a word with a mid-word stressed syllable (Metformin), a {{respell}} with
# a four-syllable coined INN (suzetrigine), and a Wiktionary {{IPA|en|...}}
# already in IPA (cefepime), which needs the IPA-to-ARPABET path rather than
# the respelling-key path.


def test_respell_metformin():
    arpa, ipa = respell_to_arpabet_ipa(["met", "FOR", "min"])
    assert arpa == "M EH0 T F AO1 R M IH0 N"
    assert ipa == "mɛtfɔːrmɪn"


def test_respell_suzetrigine():
    arpa, ipa = respell_to_arpabet_ipa(["soo", "ZE", "tri", "jeen"])
    assert arpa == "S UW0 Z EH1 T R IH0 JH IY0 N"
    assert ipa == "suːzɛtrɪdʒiːn"


def test_ipa_cefepime():
    arpa, ipa = ipa_to_arpabet_ipa("/ˈsɛf.ə.piːm/")
    assert arpa == "S EH1 F AH0 P IY0 M"
    assert ipa == "sɛfəpiːm"


def test_ipac_en_metformin_agrees_with_respell_and_mw():
    # Wikipedia's IPAc-en transcription of the same word, pre-split into one
    # template argument per phoneme/stress-mark, as it actually appears in
    # the "Metformin" article's wikitext.
    arpa, _ = ipac_en_args_to_arpabet_ipa(
        ["m", "ɛ", "t", "ˈ", "f", "ɔːr", "m", "ᵻ", "n"]
    )
    # Same segmental content and stress as the respelling and as Merriam-
    # Webster's "met-'fOr-m at n" (MEH0-T-F-AO1-R-M-*-N); only the reduced
    # final vowel differs, which is exactly what the /ɪ~ə/ marker ᵻ encodes.
    assert arpa == "M EH0 T F AO1 R M AH0 N"


def test_ipa_to_arpabet_defaults_to_first_vowel_when_unmarked():
    arpa, _ = ipa_to_arpabet_ipa("/sɛfəpiːm/")
    assert arpa.split()[1] == "EH1"


# --- Respelling-converter letter bugs ---------------------------------------
#
# `respell_to_arpabet_ipa` is the shared syllable-to-phoneme converter behind
# Wikipedia's {{respell}} template AND every USAN/NCI/DailyMed respelling
# this project parses (via `build._respelling_text_to_variant`) -- a bug here
# is silent in the worst way: the record still carries a real, verified
# citation, so nothing flags a wrong phoneme string as suspect. Each case
# below is a real citation from `references.jsonl` that was confirmed wrong
# before the fix, not a synthetic example.


def test_respell_x_is_ks():
    # Xanax's own NCI Dictionary of Cancer Terms respelling, "ZAN-ax" --
    # confirmed silently dropping the "x" entirely before this test existed
    # ("x" had no entry at all in RESPELL_CONSONANTS, so it fell through to
    # "stray punctuation" and vanished without a trace): AE1 N AE0 for
    # "an-ax", not AE1 N AE0 K S.
    arpa, ipa = respell_to_arpabet_ipa(["ZAN", "ax"])
    assert arpa == "Z AE1 N AE0 K S"
    assert ipa == "zænæks"


def test_respell_hard_and_soft_c():
    # "c" had no entry at all either, for the same reason "x" didn't --
    # Casgevy's own DailyMed respelling "cass-JEH-vee" lost its initial hard
    # /k/ entirely. Soft c (etanercept's NCI respelling "ee-TA-ner-cept",
    # /s/ before "e") and hard c at a syllable boundary before a consonant
    # both need the same fix, not just the hard-c case.
    arpa, _ = respell_to_arpabet_ipa(["cass", "JEH", "vee"])
    assert arpa == "K AE0 S JH EH1 V IY0"

    arpa, _ = respell_to_arpabet_ipa(["ee", "TA", "ner", "cept"])
    assert arpa == "IY0 T AE1 N EH0 R S EH0 P T"


def test_respell_c_at_end_of_syllable_is_hard_not_soft():
    # A syllable-final "c" ("zac") must stay hard: `nxt` (the letter after
    # "c") defaults to "" at the end of a syllable, and "" is a substring of
    # every string in Python -- a naive `nxt in "eiy"` check would treat
    # that empty string as if it matched "e"/"i"/"y" and wrongly call it
    # soft. Prozac's own DailyMed respelling "PRO-zac" pinned this down as a
    # real regression caught while writing this fix, not a hypothetical.
    arpa, _ = respell_to_arpabet_ipa(["PRO", "zac"])
    assert arpa == "P R AA1 Z AE0 K"


def test_respell_silent_final_e():
    # The "magic e" spelling convention (a syllable-final "e" after a single
    # consonant makes the preceding vowel long and is itself silent) had no
    # handling at all -- every bare trailing "e" fell through to the plain
    # "e" -> EH table entry. omeprazole's own USAN respelling ends in "zole"
    # (rhymes with "hole"): confirmed silently producing an extra EH0
    # syllable that isn't there (Z AA1 L EH0) before this fix.
    arpa, _ = respell_to_arpabet_ipa(["oh", "MEH", "pruh", "zole"])
    assert arpa.endswith("Z OW0 L")
    assert "EH0 L" not in arpa

    # The zero-consonant case (vowel directly against the silent "e") is the
    # same convention, not a separate one -- ibuprofen's own USAN
    # respelling "eye bue proe' fen" needs both "bue" -> /bjuː/ and "proe"
    # -> /proʊ/, neither of which has a consonant between the vowel and the
    # "e".
    arpa, _ = respell_to_arpabet_ipa(["eye", "bue", "proe", "fen"])
    assert arpa == "AY1 B Y UW0 P R OW0 F EH0 N"


def test_respell_magic_e_requires_exactly_one_consonant():
    # Two consonants between the vowel and the "e" is NOT the magic-e
    # pattern (a made-up "holpe" is not "hole") -- deliberately left
    # unconverted (falls through to the ordinary per-letter handling, which
    # may still not be perfect, but must not guess at a vowel-lengthening
    # this conservative either way).
    from dose_r.references.wiki_notation import _apply_magic_e

    assert _apply_magic_e("zole") == "zohl"
    assert _apply_magic_e("olde") == "olde"
