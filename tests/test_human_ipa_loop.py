from dose_r.references.human_ipa_loop import (
    align_replace,
    from_human_ctc,
    name_span,
    propose_round,
    tokenize_ipa,
    transplant_vowels,
)


def test_advair_diskus_tail_is_dropped():
    # CTC heard the product name Advair Diskus. Do not copy the tail.
    sidecar = "ˈædvɛər"
    human = "æ d v eɪ t ɪ s k ʌ s"
    phones = name_span(human.split(), tokenize_ipa(sidecar))
    joined = "".join(phones)
    assert "skʌs" not in joined
    assert "kʌs" not in joined
    ipa = from_human_ctc(sidecar, human)
    assert "skʌs" not in ipa
    assert "ˈ" in ipa


def test_nurtec_extra_tail_is_dropped():
    ipa = from_human_ctc("ˈnɜːrtɛk", "n ɜː t ɛ k oʊ d i t i")
    assert "oʊditi" not in ipa
    assert "ˈ" in ipa
    assert "ɜː" in ipa


def test_nexium_vowel_transplant_uses_human_i():
    # Sidecar isolated-e /ɛ/; human CTC /i/.
    ipa = transplant_vowels("ˈnɛksɛʌm", "n ɛ k s i ʌ m")
    assert "i" in ipa
    assert "ˈ" in ipa


def test_ctc_r_becomes_cloud_r():
    ipa = from_human_ctc("ˈvreɪlɑːr", "ɹ eɪ l ɑː")
    assert "ɹ" not in ipa
    assert "r" in ipa or "lɑː" in ipa


def test_propose_rounds_are_distinct_and_cloud_safe():
    sidecar = "ˈædkiː"
    human = "a d k w aɪ"
    seen = []
    for i in range(1, 6):
        got = propose_round(sidecar, human, None, i)
        assert got
        assert "ɪə" not in got
        seen.append(got)
    assert from_human_ctc(sidecar, human) == seen[0]


def test_sanitize_rejects_vowel_less_junk():
    from dose_r.references.human_ipa_loop import sanitize_cloud_ipa

    assert sanitize_cloud_ipa("prstnb", "juːpædæˈsaɪtɪnɪb") == "juːpædæˈsaɪtɪnɪb"


def test_sanitize_collapses_rr_and_keeps_word_space():
    from dose_r.references.human_ipa_loop import sanitize_cloud_ipa

    got = sanitize_cloud_ipa("tɛstrroʊnʌndɛk", "tɛsˈtɒstɛraʊn ʌnˈdɛkænoʊeɪt")
    assert "rr" not in got
    assert " " in got


def test_residual_changes_one_phone():
    from dose_r.references.human_ipa_loop import residual_one_edit

    got = residual_one_edit("ˈnɛksɛʌm", "n ɛ k s i ʌ m")
    assert "i" in got
    assert got != "ˈnɛksɛʌm"
