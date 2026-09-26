import pytest

from dose_r.references.tts_pronunciation import (
    compact_ascii,
    custom_pronunciation,
    custom_pronunciations_for_parts,
    ipa_from_canonical,
    is_fda_letter_suffix,
    is_source_ipa,
    letter_code_ipa,
    name_parts,
    page_is_for_word,
    spoken_parts,
    spoken_text,
    respelling_alias,
    to_cloud_en_us_ipa,
)
from dose_r.references.wiki_notation import RespellToIpaBanned, respell_to_arpabet_ipa


def test_compact_is_one_token_per_word():
    assert compact_ascii("wee-GOH-vee") == "weegohvee"
    assert compact_ascii("ZEL-jans") == "zeljans"
    assert compact_ascii("ak-oh-RAM-id-is") == "akohramidis"
    assert compact_ascii("TROSE-pee-um chloride") == "trosepeeum chloride"
    assert compact_ascii("pra-DEM-a-jeen ZAM-i-ker-a-sel") == (
        "prademajeen zamikerasel"
    )


def test_respelling_to_ipa_conversion_is_banned():
    with pytest.raises(RespellToIpaBanned):
        ipa_from_canonical("DU-pix-ent")
    with pytest.raises(RespellToIpaBanned):
        ipa_from_canonical("AH-troo-be")
    with pytest.raises(RespellToIpaBanned):
        respell_to_arpabet_ipa(["DU", "pix", "ent"])


def test_cloud_en_us_does_not_rewrite_ipa():
    assert to_cloud_en_us_ipa("ˈlɪərɪkɑː") == "ˈlɪərɪkɑː"
    assert to_cloud_en_us_ipa("kwɪˈtaɪ.əˌpin") == "kwɪˈtaɪ.əˌpin"
    assert to_cloud_en_us_ipa("kwɪˈtaɪəˌpin") == "kwɪˈtaɪəˌpin"


def test_is_source_ipa_accepts_published_strings():
    assert is_source_ipa("kwɪˈtaɪ.əˌpin")
    assert is_source_ipa("/ˈwɪn.rɛ.vɛər/")
    assert is_source_ipa("spɪˈriːvə")
    assert is_source_ipa("oʊˈzɛmpɪk")
    assert is_source_ipa("koʊˈbɛnfi")
    assert is_source_ipa("ˈkleɹ.ə.tɪn")
    assert is_source_ipa("ˌæɹ.ɪˈpɪp.ɹəˌzoʊl")


def test_is_source_ipa_rejects_respelling_and_howtopronounce_junk():
    assert not is_source_ipa("co-BEN-fee")
    assert not is_source_ipa("DU-pix-ent")
    assert not is_source_ipa("ˈklar-ə-ˌtin")
    assert not is_source_ipa("spˈɪ.ɹ.ɪvə")
    assert not is_source_ipa("sˈæ.lm.ɪ.ɾɚɹɑːl")
    assert not is_source_ipa("skˈaɪɹɪzi")
    assert not is_source_ipa("kwˈɛʃɪ..æp.aɪn")
    assert not is_source_ipa("lˈʊ.ɹɹɐs.ɪd.oʊn")
    assert not is_source_ipa("ˈoʊ.zmpɪk")
    assert not is_source_ipa("aɪ")
    assert not is_source_ipa("aˈʝ̞eɾ")
    assert not is_source_ipa("ˈkaʝ̞e̞")


def test_page_is_for_word_rejects_wiktionary_mixups():
    assert not page_is_for_word("Utebzi", "https://en.wiktionary.org/wiki/utzi")
    assert not page_is_for_word("Casgevy", "https://en.wiktionary.org/wiki/cashew")
    assert not page_is_for_word("Tzield", "https://en.wiktionary.org/wiki/-tizide")
    assert page_is_for_word("bevacizumab", "https://en.wiktionary.org/wiki/bevacizumab")
    assert page_is_for_word(
        "datopotamab",
        "https://www.cancer.gov/publications/dictionaries/cancer-terms/def/datopotamab-deruxtecan",
    )
    assert page_is_for_word("Utebzi", "https://www.webmd.com")
    assert page_is_for_word("teplizumab", "https://www.drugs.com/teplizumab.html")
    assert not page_is_for_word(
        "oveporexton", "https://en.wiktionary.org/wiki/propafenone"
    )


def test_name_parts_splits_spaces_and_fda_suffix():
    assert name_parts("teplizumab-mzwv") == ["teplizumab", "mzwv"]
    assert name_parts("atacicept-vymj") == ["atacicept", "vymj"]
    assert name_parts("insulin icodec-abae") == ["insulin", "icodec", "abae"]
    assert name_parts("pivekimab sunirine-pvzy") == ["pivekimab", "sunirine", "pvzy"]
    assert name_parts("baloxavir marboxil") == ["baloxavir", "marboxil"]
    assert name_parts("zoliflodacin") == ["zoliflodacin"]
    assert name_parts("formoterol fumarate dihydrate") == [
        "formoterol",
        "fumarate",
        "dihydrate",
    ]
    assert is_fda_letter_suffix("mzwv", "teplizumab-mzwv")
    assert not is_fda_letter_suffix("teplizumab", "teplizumab-mzwv")
    assert not is_fda_letter_suffix("marboxil", "baloxavir marboxil")
    assert spoken_parts("teplizumab-mzwv") == ["teplizumab"]
    assert spoken_parts("insulin icodec-abae") == ["insulin", "icodec"]
    assert spoken_parts("nogapendekin alfa inbakicept-pmln") == [
        "nogapendekin",
        "alfa",
        "inbakicept",
    ]
    assert spoken_parts("trospium chloride") == ["trospium", "chloride"]
    assert spoken_text("bevacizumab-vikg") == "bevacizumab"
    assert spoken_text("nogapendekin alfa inbakicept-pmln") == (
        "nogapendekin alfa inbakicept"
    )
    assert spoken_text("trospium chloride") == "trospium chloride"


def test_respelling_alias_is_english_syllables_not_ipa():
    assert respelling_alias("a-TA-ki-sept", "atacicept-vymj") == "a ta ki sept"
    assert respelling_alias("tep-LIZ-oo-mab", "teplizumab-mzwv") == "tep liz oo mab"
    assert respelling_alias("zoe-li-floe-DAY-sin", "zoliflodacin") == (
        "zoe li floe day sin"
    )
    assert respelling_alias("pi-VEK-i-mab SOO-ni-reen", "pivekimab sunirine-pvzy") == (
        "pi vek i mab soo ni reen"
    )
    assert respelling_alias("ef-gar-TIG-i-mod", "efgartigimod alfa") == (
        "ef gar tig i mod alfa"
    )
    assert respelling_alias("floo-TIK-uh-sohn", "fluticasone propionate") == (
        "floo tik uh sohn propionate"
    )
    # Must not invent IPA from the respelling.
    with pytest.raises(RespellToIpaBanned):
        ipa_from_canonical("zoe-li-floe-DAY-sin")


def test_letter_code_ipa_is_english_alphabet_not_drug_g2p():
    assert letter_code_ipa("mzwv") == "ɛmziːdʌbəljuviː"
    assert letter_code_ipa("vikg") == "viːaɪkeɪdʒiː"
    block = custom_pronunciations_for_parts(
        [("teplizumab", "tɛpˈlɪzʊmæb"), ("mzwv", letter_code_ipa("mzwv"))]
    )
    phrases = [p["phrase"] for p in block["pronunciations"]]
    assert phrases == ["teplizumab", "mzwv"]


def test_custom_pronunciation_keeps_the_real_spelling_as_the_phrase():
    block = custom_pronunciation("Xeljanz", "ˈzɛldʒæns")
    rec = block["pronunciations"][0]
    assert rec["phrase"] == "Xeljanz"
    assert rec["phoneticEncoding"] == "PHONETIC_ENCODING_IPA"
    assert rec["pronunciation"] == "ˈzɛldʒæns"


def test_ipa_to_xsampa_matches_cloud_apple_example():
    from dose_r.references.tts_pronunciation import ipa_to_xsampa

    # Cloud docs: apple → "{ p@l"  (space optional in our compact form).
    assert ipa_to_xsampa("ˈæpəl") == '"{p@l'
    assert ipa_to_xsampa("əˈtruːbi") == '@"tru:bi'
    xs = custom_pronunciation("Attruby", '@"tru:bi', "PHONETIC_ENCODING_X_SAMPA")
    assert xs["pronunciations"][0]["phoneticEncoding"] == "PHONETIC_ENCODING_X_SAMPA"


def test_pronunciations_jsonl_keeps_respelling_and_no_converted_ipa():
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
        assert p.get("ipa", "") == ""
        assert p.get("ipa_cloud", "") == ""
        if rec.get("respelling"):
            assert p["respelling"] == rec["respelling"]
            assert p["compact"]
        else:
            assert p["compact"] == ""
