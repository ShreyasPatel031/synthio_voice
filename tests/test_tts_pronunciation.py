from dose_r.references.tts_pronunciation import (
    compact_ascii,
    custom_pronunciation,
    ipa_from_canonical,
)


def test_compact_is_one_token_per_word():
    assert compact_ascii("wee-GOH-vee") == "weegohvee"
    assert compact_ascii("ZEL-jans") == "zeljans"
    assert compact_ascii("ak-oh-RAM-id-is") == "akohramidis"
    assert compact_ascii("TROSE-pee-um chloride") == "trosepeeum chloride"
    assert compact_ascii("pra-DEM-a-jeen ZAM-i-ker-a-sel") == (
        "prademajeen zamikerasel"
    )


def test_ipa_uses_ye_as_aɪ_and_keeps_stress():
    assert ipa_from_canonical("wee-GOH-vee") == "wiːˈɡoʊviː"
    assert ipa_from_canonical("VRAY-lar") == "ˈvreɪlɑːr"
    assert ipa_from_canonical("ZEL-jans") == "ˈzɛldʒæns"
    # USAN sye = /aɪ/, not Wikipedia y=/j/.
    ipa = ipa_from_canonical("toe-fa-SYE-ti-nib")
    assert "saɪ" in ipa
    assert "jɛ" not in ipa
    assert "ˈ" in ipa


def test_ipa_heals_broken_vi_r_split():
    # Canonical still has the bad USAN split; IPA concatenation is tenofovir.
    ipa = ipa_from_canonical("ten-OF-oh-vi-r al-a-FEN-a-mide")
    assert "vɪr" in ipa
    assert " " in ipa


def test_custom_pronunciation_keeps_the_real_spelling_as_the_phrase():
    block = custom_pronunciation("Xeljanz", "ˈzɛldʒæns")
    rec = block["pronunciations"][0]
    assert rec["phrase"] == "Xeljanz"
    assert rec["phoneticEncoding"] == "PHONETIC_ENCODING_IPA"
    assert rec["pronunciation"] == "ˈzɛldʒæns"
