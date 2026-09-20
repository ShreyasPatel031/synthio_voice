import json
from pathlib import Path

from dose_r.references.respelling import (
    SOURCE_PRIORITY,
    homogenize_record,
    is_canonical,
    mw_to_respelling,
    pick_source,
    to_canonical,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ROOT / "dose_r" / "references" / "references.jsonl"
RESPELLINGS = ROOT / "dose_r" / "references" / "respellings.jsonl"


def _records():
    with REFERENCES.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def test_canonical_examples():
    assert to_canonical('toe" fa sye\' ti nib', "usan-official")[0] == "toe-fa-SYE-ti-nib"
    assert to_canonical("nu-ZOL-vence", "dailymed")[0] == "nu-ZOL-vence"
    assert to_canonical("uh-see-tuh-MIH-nuh-fen", "nci-dictionary-of-cancer-terms")[0] == (
        "uh-see-tuh-MIH-nuh-fen"
    )
    assert to_canonical("ə-ˈbi-lə-ˌfī", "merriam-webster/medical-api")[0] == "uh-BI-luh-fye"
    assert to_canonical("jar DEE ans", "dailymed")[0] == "jar-DEE-ans"
    assert to_canonical("pronounced aw-kat-zil", "dailymed")[0] == "AW-kat-zil"
    assert to_canonical('ex gam" gloe jeen  aw" toe tem’ sel', "usan-official")[0] == (
        "ex-GAM-gloe-jeen aw-toe-TEM-sel"
    )


def test_mw_optional_and_first_variant_only():
    assert mw_to_respelling("ˈvī-ˌvan(t)s") == "VYE-vants"
    # Second MW alternate is dropped; one string per name.
    assert mw_to_respelling("lə-ˈrat-ə-ˌdēn, -ˌdīn") == "luh-RAT-uh-deen"
    # Same pair listed the other way: keep official `deen` (/diːn/), not `dyen`.
    assert mw_to_respelling("fə-ˈmōt-ə-ˌdīn, -ˌdēn") == "fuh-MOHT-uh-deen"


def test_is_canonical_rejects_mixed_formats():
    assert is_canonical("toe-fa-SYE-ti-nib")
    assert is_canonical("pra-DEM-a-jeen ZAM-i-ker-a-sel")
    assert is_canonical("TROSE-pee-um chloride")
    assert not is_canonical("chloride")
    assert not is_canonical("toe\" fa sye' ti nib")
    assert not is_canonical("ə-ˈbi-lə-ˌfī")
    assert not is_canonical("toe-fa-SYE-TI-nib")  # two primaries


def test_wikipedia_ipa_and_cmudict_are_never_canonical():
    assert to_canonical("/ˈbɛnədɹɪl/", "wiktionary")[0] is None
    assert to_canonical("mɛtˈfɔːrmᵻn", "wikipedia")[0] is None
    assert to_canonical("T AY1 L AH0 N AO2 L", "cmudict")[0] is None
    assert to_canonical("priˈɡæbəlɪn", "gemini-grounded-search")[0] is None


def test_usan_beats_mw_on_tofacitinib():
    rec = next(r for r in _records() if r["ingredient"] == "tofacitinib")
    picked = pick_source(rec["sources"])
    assert picked["source"] == "usan-official"
    assert picked["respelling"] == "toe-fa-SYE-ti-nib"


def test_one_source_per_homogenized_row():
    rec = next(r for r in _records() if r["ingredient"] == "omalizumab")
    out = homogenize_record(rec)
    assert out["source"] == "usan-official"
    assert "merriam-webster/medical-api" in out["dropped_sources"]
    assert is_canonical(out["respelling"])


def test_priority_order():
    assert SOURCE_PRIORITY[0] == "usan-official"
    assert SOURCE_PRIORITY[1] == "dailymed"
    assert "wikipedia" not in SOURCE_PRIORITY
    assert "cmudict" not in SOURCE_PRIORITY


def test_built_respellings_cover_every_ingredient_and_are_canonical():
    assert RESPELLINGS.exists(), "run scripts/build_respellings.py"
    refs = {r["ingredient"] for r in _records()}
    rows = [json.loads(l) for l in RESPELLINGS.read_text().splitlines() if l.strip()]
    assert {r["ingredient"] for r in rows} == refs
    sourced = [r for r in rows if r.get("respelling")]
    for r in sourced:
        assert is_canonical(r["respelling"]), (r["ingredient"], r["respelling"])
        assert r["source"] in {
            "usan-official",
            "dailymed",
            "nci-dictionary-of-cancer-terms",
            "merriam-webster/medical-api",
            "merriam-webster/dictionary",
            "gemini-grounded-search",
        }
        assert r["confidence"] == "sourced"
    # Every previously-sourced IPA record must still have a respelling,
    # including the 19 that were MW-phonetic only.
    previously_sourced = {r["ingredient"] for r in _records() if r.get("sources")}
    have = {r["ingredient"] for r in sourced}
    assert previously_sourced <= have
    unsourced = [r for r in rows if not r.get("respelling")]
    assert {r["ingredient"] for r in unsourced} == {
        r["ingredient"] for r in _records() if not r.get("sources")
    }
