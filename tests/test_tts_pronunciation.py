from dose_r.references.tts_pronunciation import (
    compact_ascii,
    custom_pronunciation,
    ipa_from_canonical,
    to_cloud_en_us_ipa,
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


def test_cloud_en_us_folds_near_diphthong():
    assert to_cloud_en_us_ipa("ˈlɪərɪkɑː") == "ˈlɪrɪkɑː"
    assert "ɪə" not in to_cloud_en_us_ipa(ipa_from_canonical("LEER-i-kah"))


def test_custom_pronunciation_keeps_the_real_spelling_as_the_phrase():
    block = custom_pronunciation("Xeljanz", "ˈzɛldʒæns")
    rec = block["pronunciations"][0]
    assert rec["phrase"] == "Xeljanz"
    assert rec["phoneticEncoding"] == "PHONETIC_ENCODING_IPA"
    assert rec["pronunciation"] == "ˈzɛldʒæns"


def test_pronunciations_jsonl_covers_every_canonical_respelling():
    from pathlib import Path
    import json

    root = Path(__file__).resolve().parents[1]
    resp = {
        json.loads(l)["ingredient"]: json.loads(l)
        for l in (root / "dose_r/references/respellings.jsonl").read_text().splitlines()
    }
    pron = {
        json.loads(l)["ingredient"]: json.loads(l)
        for l in (root / "dose_r/references/pronunciations.jsonl").read_text().splitlines()
    }
    assert set(resp) == set(pron)
    for name, rec in resp.items():
        p = pron[name]
        if rec.get("respelling"):
            assert p["ipa"]
            assert p["ipa_cloud"]
            assert p["compact"]
            assert "ɪə" not in p["ipa_cloud"]
        else:
            assert p["ipa"] == ""
