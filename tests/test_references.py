import json
from pathlib import Path

import pytest

from dose_r.judge.phonemes import parse
from dose_r.judge.references import ReferenceSet
from dose_r.references.notation import convert, split_variants, to_arpabet_ipa
from dose_r.references.wiki_notation import (
    RespellToIpaBanned,
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
# Source-published IPA (`{{IPA}}`, `{{IPAc-en}}`) is still parsed.
# G2P of respelling (`respell_to_arpabet_ipa`) is banned — see
# dose_r/references/README.md.


def test_respelling_to_ipa_g2p_is_banned():
    with pytest.raises(RespellToIpaBanned):
        respell_to_arpabet_ipa(["met", "FOR", "min"])
    with pytest.raises(RespellToIpaBanned):
        respell_to_arpabet_ipa(["DU", "pix", "ent"])


def test_ipa_cefepime():
    arpa, ipa = ipa_to_arpabet_ipa("/ˈsɛf.ə.piːm/")
    assert arpa == "S EH1 F AH0 P IY0 M"
    assert ipa == "ˈsɛfəpiːm"


def test_ipac_en_metformin_agrees_with_mw_segments():
    # Wikipedia's IPAc-en transcription — this is published IPA, not G2P.
    arpa, _ = ipac_en_args_to_arpabet_ipa(
        ["m", "ɛ", "t", "ˈ", "f", "ɔːr", "m", "ᵻ", "n"]
    )
    assert arpa == "M EH0 T F AO1 R M AH0 N"


def test_ipa_to_arpabet_defaults_to_first_vowel_when_unmarked():
    arpa, _ = ipa_to_arpabet_ipa("/sɛfəpiːm/")
    assert arpa.split()[1] == "EH1"


def test_mw_ipa_carries_stress():
    arpa, ipa = to_arpabet_ipa("flü-ˈtik-ə-ˌsōn")
    assert ipa == "fluːˈtɪkəˌsoʊn"


def test_ipac_en_preserves_its_own_stress_mark():
    # Wikipedia's IPAc-en template already states stress explicitly in its
    # own input (the "ˈ" argument here) -- this must carry it through to
    # the output IPA, not just consume it for ARPABET and drop it.
    # Wikipedia's non-standard ᵻ ("either /ɪ/ or /ə/") is resolved to ə on
    # the way out, matching what ARPABET already does with it -- a strict
    # IPA consumer can't be handed a symbol that isn't in the alphabet.
    _, ipa = ipac_en_args_to_arpabet_ipa(["m", "ɛ", "t", "ˈ", "f", "ɔːr", "m", "ᵻ", "n"])
    assert ipa == "mɛtˈfɔːrmən"


# --- Multi-word IPA word boundaries -----------------------------------------
#
# A multi-word ingredient's per-word IPA strings were concatenated with no
# separator at all ("copper histidinate" -> "kɒpɛrhɪstɪdɪneɪt", one
# unreadable run-on word) while the ARPABET join already used a space --
# confirmed as a real gap affecting every one of this dataset's 21
# multi-word ingredients, not just a hypothetical.


def test_join_separates_words_in_ipa_not_just_arpabet():
    from dose_r.references.build import _join

    (arpa, ipa), = _join([[("K AA1 P", "kɑːp")], [("HH IH1 S", "hɪs")]])
    assert arpa == "K AA1 P HH IH1 S"
    assert ipa == "kɑːp hɪs"


def test_respelling_span_g2p_is_banned():
    from dose_r.references.build import _respelling_span_to_variant
    from dose_r.references.wiki_notation import RespellToIpaBanned

    with pytest.raises(RespellToIpaBanned):
        _respelling_span_to_variant("bik-TEG-ra-vir SO-di-um")


# --- Partial whole-name-source detection ------------------------------------
#
# A source queried with a full multi-word ingredient name can still only
# answer for part of it: real, confirmed citations, not hypotheticals --
# Merriam-Webster's own "fluticasone propionate" entry never respells
# "propionate" at all, and the AMA USAN Statement filed under
# "efgartigimod-alfa.pdf" never respells "alfa" in its own PRONUNCIATION
# field. Both used to be stored as if they were the whole name's own
# pronunciation, with nothing distinguishing them from a real one.


def test_covers_full_name_rejects_a_missing_trailing_word():
    from dose_r.references.build import _covers_full_name

    # efgartigimod alfa's real, confirmed-broken USAN citation: covers
    # "efgartigimod" (5 syllables) but never "alfa" (2 more). "alfa" is
    # also exactly four letters, the same length `_split_fda_suffix` looks
    # for on a real biosimilar code -- this must still be rejected, not
    # have "alfa" silently excused from the expected count as if it were
    # one.
    assert not _covers_full_name(
        "efgartigimod alfa", [("EH0 F G AA0 R T IH1 G IH0 M AA0 D", "")]
    )


def test_covers_full_name_accepts_natural_undercounting():
    from dose_r.references.build import _covers_full_name

    # exagamglogene autotemcel's real USAN citation ("ex gam gloe jeen aw
    # toe tem sel") covers both words in full, just with fewer actual
    # spoken syllables than the crude vowel-letter estimate over the
    # written name expects -- must NOT be flagged as partial.
    assert _covers_full_name(
        "exagamglogene autotemcel",
        [("EH0 K S G AE0 M G L OW0 JH IY0 N AO0 T OW0 T EH1 M S EH0 L", "")],
    )


def test_covers_full_name_ignores_fda_biosimilar_suffix():
    from dose_r.references.build import _covers_full_name

    # "-abae" is a meaningless FDA-assigned distinguishing code with no
    # intended pronunciation of its own (confirmed elsewhere in this
    # project: USAN's own index has no entry for the suffixed form at
    # all) -- it must not count toward the expected syllable total, or a
    # genuinely complete citation for "insulin icodec" would be wrongly
    # flagged as short one syllable it was never supposed to have.
    assert _covers_full_name(
        "insulin icodec-abae",
        [("IH0 N S AH1 L IH0 N AY2 K OW0 D EH0 K", "")],
    )


# --- USAN's own multi-word word-boundary signal -----------------------------
#
# A real USAN Statement respelling BOTH words of a multi-word generic name
# in one combined PRONUNCIATION field, back to back, with no consistent
# single/double-space convention marking which gap is the true word
# boundary versus an ordinary between-syllable gap -- but the true boundary
# is reliably the widest gap in the string, the same signal
# `sources._first_word_group` already uses for the (different) salt-form
# case. Confirmed against real citations, not synthetic ones: "copper
# histidinate" -> "kop' er  his' ti di nate" (double space at the boundary,
# single elsewhere), "dimethyl fumarate" -> "dye  meth' il     fue' ma
# rate" (a 5-space boundary against 2-space syllable gaps elsewhere).


def test_split_at_word_boundaries_finds_the_widest_gap():
    from dose_r.references.sources import _split_at_word_boundaries

    assert _split_at_word_boundaries("kop' er  his' ti di nate", 2) == [
        "kop' er",
        "his' ti di nate",
    ]
    # A wider outlier gap (3, against a modal gap of 1 elsewhere) is still
    # found correctly regardless of its exact width, and each returned
    # group's own internal syllables are single-space-joined.
    assert _split_at_word_boundaries("dye meth' il   fue' ma rate", 2) == [
        "dye meth' il",
        "fue' ma rate",
    ]


def test_split_at_word_boundaries_declines_when_no_gap_stands_out():
    from dose_r.references.sources import _split_at_word_boundaries

    # Uniform spacing throughout -- no distinguishable boundary to trust,
    # must return None rather than guess where one word ends.
    assert _split_at_word_boundaries("ef gar tig i mod", 2) is None


def test_collapse_pronunciation_whitespace_marks_the_real_boundary():
    from dose_r.references.sources import _collapse_pronunciation_whitespace

    collapsed = _collapse_pronunciation_whitespace(
        "kop' er  his' ti di nate", "copper histidinate"
    )
    assert collapsed == "kop' er  his' ti di nate"

    # Single-word names, or a boundary that can't be found confidently,
    # fall back to a plain single-space collapse -- never invents a
    # boundary it isn't sure of.
    assert _collapse_pronunciation_whitespace("ef gar tig i mod", "efgartigimod alfa") == (
        "ef gar tig i mod"
    )


def test_respelling_text_to_variant_g2p_is_banned():
    from dose_r.references.build import _respelling_text_to_variant
    from dose_r.references.wiki_notation import RespellToIpaBanned

    with pytest.raises(RespellToIpaBanned):
        _respelling_text_to_variant("kop' er  his' ti di nate")
    with pytest.raises(RespellToIpaBanned):
        _respelling_text_to_variant("trose' pee um  chloride")


def test_respelling_word_g2p_is_banned():
    from dose_r.references.build import _respelling_word_to_variant
    from dose_r.references.wiki_notation import RespellToIpaBanned

    with pytest.raises(RespellToIpaBanned):
        _respelling_word_to_variant("jar DEE ans")




# --- Ground-truth conformance defects found in the pre-evaluation audit -----
#
# Every case below was a real defect in `references.jsonl` found while
# validating the IPA before it gets used as evaluation ground truth, not a
# synthetic example. They share a failure mode worth stating plainly:
# `phonetic_scorer.score_against_reference` keeps the BEST-matching variant
# of a reference set, deliberately, so that a system isn't penalised for a
# legitimate alternate pronunciation. That makes a short or garbled variant
# strictly dangerous -- it can only ever pull a score UP, never down, so a
# system that mumbled a fragment would be recorded as correct.


def test_mw_trailing_dash_alternate_is_expanded_not_left_a_fragment():
    # acetaminophen's real MW entry. "ˌas-ət-" abbreviates the alternate by
    # its LEADING syllables and used to be emitted as the bare 2-syllable
    # "ˈˌæsət" -- both a fragment and malformed (doubled stress mark).
    variants = split_variants("ə-ˌsēt-ə-ˈmin-ə-fən, ˌas-ət-")
    assert variants == ["ə-ˌsēt-ə-ˈmin-ə-fən", "ˌas-ət-ə-ˈmin-ə-fən"]
    assert [i for _, i in convert("ə-ˌsēt-ə-ˈmin-ə-fən, ˌas-ət-")] == [
        "əˌsiːtəˈmɪnəfən",
        "ˌæsətəˈmɪnəfən",
    ]


def test_mw_both_ends_truncated_alternate_is_skipped_not_guessed():
    # esomeprazole's real MW entry. "-ˈmē-prə-" replaces a MIDDLE span and
    # its position is ambiguous (three remaining base syllables could split
    # 1+2 or 2+1), so it is dropped rather than silently losing the "-zole"
    # tail as it did before.
    assert split_variants("ˌes-ō-ˈmep-rə-ˌzōl, -ˈmē-prə-, -ˌzȯl") == [
        "ˌes-ō-ˈmep-rə-ˌzōl",
        "ˌes-ō-ˈmep-rə-ˌzȯl",
    ]


def test_mw_parenthesised_optional_stress_does_not_double_the_mark():
    # dihydrate's real MW entry: "(ˈ)" is an OPTIONAL stress, and stripping
    # only the parentheses left two literal primaries in one word.
    arpa, ipa = to_arpabet_ipa("(ˈ)dī-ˈhī-ˌdrāt")
    assert ipa == "daɪˈhaɪˌdreɪt"
    assert ipa.count("ˈ") == 1


def test_mw_entry_with_no_primary_promotes_its_secondary():
    # sertraline's real MW entry marks only secondaries. ARPABET already
    # defaulted the first vowel to primary; IPA used to stack its own mark
    # in front of the existing secondary, emitting the malformed "ˈˌ".
    _, ipa = to_arpabet_ipa("ˌsər-trə-ˌlēn")
    assert "ˈˌ" not in ipa
    assert ipa == "ˈsərtrəˌliːn"


def test_is_fragment_rejects_a_variant_too_short_for_the_name():
    from dose_r.references.build import _is_fragment

    # Retatrutide's sole citation respelled only "-trutide".
    assert _is_fragment("Retatrutide", "T R UW1 T AY0 D")
    assert not _is_fragment("Retatrutide", "R EH2 T AH0 T R UW1 T AY0 D")


def test_is_malformed_multiword_rejects_two_primaries_in_one_run():
    from dose_r.references.build import _is_malformed_multiword

    # MW writes a two-word term as one hyphen run with two primaries and
    # no placeable word boundary (baloxavir marboxil).
    assert _is_malformed_multiword("baloxavir marboxil", "bəˈlɑksəˌvɪrmɑrˈbɑksəl")
    assert not _is_malformed_multiword("baloxavir marboxil", "bælˈɒksævɪr mɑːrˈbɒksɪl")
