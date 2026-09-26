from dose_r.misaki_fold import gold_misaki_candidates, ipa_to_misaki, restress

SYMBOLS = set("AIWYbdfhijklmnpstuvwzðŋɑɔəɛɜɡɪɹʃʊʌʒʤʧˈˌθᵊOæɾᵻT ")


def test_wegovy_fold_is_oh_not_plain_o():
    ipa = "wɪˈɡoʊvi"
    raw = ipa_to_misaki(ipa, SYMBOLS)
    assert "O" in raw
    assert "oʊ" not in raw
    cands = dict(gold_misaki_candidates(ipa, "Wegovy", SYMBOLS))
    assert "gold-stress" in cands
    assert cands["gold-stress"] == restress(raw)


def test_gedatolisib_keeps_affricate_not_hard_g():
    ipa = "ˌdʒɛdætəˈlɪsɪb"
    cands = dict(gold_misaki_candidates(ipa, "gedatolisib", SYMBOLS))
    joined = " ".join(cands.values())
    assert "ʤ" in joined
    assert not any(p.startswith("ɡɛ") for p in cands.values())


def test_cypsedo_drops_length_and_oh_diphthong():
    ipa = "sɪpˈsiːdoʊ"
    cands = dict(gold_misaki_candidates(ipa, "Cypsedo", SYMBOLS))
    for phones in cands.values():
        assert "ː" not in phones
        assert "oʊ" not in phones
        assert "O" in phones


def test_candidates_do_not_invent_last_vowels():
    ipa = "ləˈtuːdə"
    cands = dict(gold_misaki_candidates(ipa, "Latuda", SYMBOLS))
    # Old loop offered ləˈtudɑ / ləˈtʊdə / ləˈtudʌ. Those are not gold.
    assert "ləˈtudɑ" not in cands.values()
    assert "ləˈtʊdə" not in cands.values()
    assert "ləˈtudʌ" not in cands.values()


def test_etanercept_stress_rests_on_ash():
    ipa = "ɪˈtænərsɛpt"
    cands = dict(gold_misaki_candidates(ipa, "etanercept", SYMBOLS))
    stress = cands["gold-stress"]
    assert "ˈæ" in stress or "tˈæ" in stress
    assert not stress.endswith("ˈɛpt")
