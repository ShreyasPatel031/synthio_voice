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


def test_every_dose_ingredient_has_a_reference():
    with DATASET.open() as f:
        ingredients = {i for line in f for i in json.loads(line)["ingredients"]}
    assert ReferenceSet.load(REFERENCES).missing(ingredients) == []


def test_variant_lists_are_parallel_and_non_empty(records):
    for r in records:
        assert r["arpabet_variants"], r["ingredient"]
        assert len(r["ipa_variants"]) == len(r["arpabet_variants"]), r["ingredient"]


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


def test_unsourced_records_are_flagged_low_and_noted(records):
    for r in records:
        if not r["sources"]:
            assert r["confidence"] == "low", r["ingredient"]
            assert "TODO" in r["notes"], r["ingredient"]


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
