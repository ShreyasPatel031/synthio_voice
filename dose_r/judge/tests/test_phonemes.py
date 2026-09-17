"""The weighting is the whole argument for the phonetic scorer, so it is pinned
by ordering assertions rather than by exact values: the claim is that acoustically
close confusions cost less than distant ones, not that /t/-/d/ costs 0.231."""

import pytest

from dose_r.judge import phonemes as ph


def test_identity_is_free_and_stress_is_invisible():
    assert ph.substitution_cost("AE1", "AE1") == 0.0
    assert ph.substitution_cost("AE1", "AE0") == 0.0
    assert ph.substitution_cost("T", "T") == 0.0


def test_costs_are_symmetric_and_bounded():
    for pair, cost in ph.cost_matrix().items():
        a, b = tuple(pair)
        assert 0.0 < cost <= 1.0, (a, b, cost)
        assert ph.substitution_cost(a, b) == ph.substitution_cost(b, a)


def test_voicing_is_the_cheapest_consonant_confusion():
    voicing = ph.substitution_cost("T", "D")
    place = ph.substitution_cost("T", "K")
    manner = ph.substitution_cost("T", "S")
    distant = ph.substitution_cost("T", "M")
    assert voicing < place < distant
    assert voicing < manner < distant


def test_adjacent_place_costs_less_than_distant_place():
    assert ph.substitution_cost("P", "T") < ph.substitution_cost("P", "K")
    assert ph.substitution_cost("S", "SH") < ph.substitution_cost("S", "HH")


def test_vowel_neighbours_cost_less_than_vowel_opposites():
    assert ph.substitution_cost("IH", "IY") < ph.substitution_cost("IH", "AA")
    assert ph.substitution_cost("IY", "EY") < ph.substitution_cost("IY", "UW")


def test_diphthong_shares_cost_with_its_nucleus():
    assert ph.substitution_cost("AY", "AA") < ph.substitution_cost("AY", "UW")


def test_cross_class_is_maximal_except_for_sonorants():
    assert ph.substitution_cost("AA", "T") == ph.CROSS_CLASS_COST
    assert ph.substitution_cost("AA", "M") == ph.CROSS_CLASS_SONORANT_COST
    assert ph.substitution_cost("AA", "M") < ph.substitution_cost("AA", "T")


def test_documented_overrides_beat_the_feature_model():
    assert ph.substitution_cost("ER", "R") == 0.12
    assert ph.substitution_cost("ER1", "R") == 0.12
    assert ph.substitution_cost("AA", "AO") < ph.substitution_cost("AA", "AE")


def test_every_phoneme_is_closer_to_its_own_class():
    for vowel in ph.VOWELS:
        worst_vowel = max(
            ph.substitution_cost(vowel, v) for v in ph.VOWELS if v != vowel
        )
        assert worst_vowel < ph.CROSS_CLASS_COST


def test_parse_normalises_and_rejects_non_arpabet():
    assert ph.parse("  ah0 t  ao1 ") == ["AH0", "T", "AO1"]
    assert ph.parse(["AH0", "T"]) == ["AH0", "T"]
    with pytest.raises(ph.UnknownPhoneme):
        ph.parse("AH0 QQ")


def test_stress_helpers():
    assert ph.strip_stress("AH0") == "AH"
    assert ph.strip_stress("NG") == "NG"
    assert ph.stress_of("AE2") == 2
    assert ph.stress_of("S") is None
    assert ph.is_vowel("ER1") and not ph.is_vowel("R")
