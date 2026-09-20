from dose_r.judge.phonemes import parse
from dose_r.references import usan_stems

# `usan_stems.py` still exists and is still tested here as a standalone
# module -- what changed is that `build.py` no longer calls it (or plain
# g2p) as a fallback for an ingredient with no real source. That removal is
# covered by tests/test_references.py; the build-integration tests that used
# to live in this file (`build._from_usan(...)`, low-confidence notes
# distinguishing "stem-rule applied" from plain fallback) tested code paths
# that no longer exist and were deleted along with them.


def test_match_stem_prefers_the_longest_suffix():
    # "tofacitinib" ends in both "-inib" and the more specific "-tinib".
    assert usan_stems.match_stem("tofacitinib").suffix == "tinib"
    # "esomeprazole" ends in both "-zole" and the more specific "-prazole".
    assert usan_stems.match_stem("esomeprazole").suffix == "prazole"


def test_match_stem_requires_a_non_empty_root():
    assert usan_stems.match_stem("mab") is None
    assert usan_stems.match_stem("tide") is None


def test_match_stem_strips_the_fda_biosimilar_suffix_first():
    stem = usan_stems.match_stem("bevacizumab-vikg")
    assert stem is not None and stem.suffix == "zumab"


def test_match_stem_none_for_an_unrecognized_ending():
    assert usan_stems.match_stem("acetaminophen") is None
    assert usan_stems.match_stem("Abilify") is None


def test_to_arpabet_variants_is_none_without_a_stem():
    assert usan_stems.to_arpabet_variants("acetaminophen") is None


def test_to_arpabet_variants_output_is_valid_arpabet():
    for word in ["tofacitinib", "omeprazole", "valsartan", "semaglutide", "bevacizumab-vikg"]:
        variants = usan_stems.to_arpabet_variants(word)
        assert variants and all(variants)
        for v in variants:
            parse(v)  # raises UnknownPhoneme on anything not real ARPABET


def test_before_mode_stresses_the_syllable_before_the_stem():
    # "before" mode: primary stress goes on the root's last syllable (the
    # one right before "-tinib" starts), and "-tinib"'s own final syllable
    # takes secondary stress rather than the root engine's usual guess.
    arpa = usan_stems.to_arpabet_variants("tofacitinib")[0]
    primary = [i for i, p in enumerate(arpa) if p.endswith("1")]
    assert len(primary) == 1
    assert arpa[primary[0] + 1 :] == ["T", "IH0", "N", "IH2", "B"]  # the stem tail, untouched
    assert arpa[-2:] == ["IH2", "B"]  # "-nib" carries secondary stress


def test_stem_mode_stresses_its_own_fixed_syllable():
    # -statin always stresses "sta" regardless of the root (atorvastatin,
    # rosuvastatin); -sartan always stresses "sar" (valsartan).
    assert "AE1" in usan_stems.to_arpabet_variants("atorvastatin")[0]
    assert "AA1" in usan_stems.to_arpabet_variants("valsartan")[0]


def test_keep_mode_only_fixes_the_segmental_ending():
    # -tide is always pronounced to rhyme with "tide" (T AY D), never "tid".
    arpa = usan_stems.to_arpabet_variants("semaglutide")[0]
    assert arpa[-3] == "T"
    assert arpa[-2].startswith("AY")
    assert arpa[-1] == "D"


def test_fda_suffix_is_pronounced_separately_via_plain_g2p():
    # "-vikg" has no documented pronunciation of its own (see
    # `_split_fda_suffix`), so it is appended via plain `g2p`, distinct from
    # the stem-ruled "-zumab" ending that precedes it.
    with_suffix = usan_stems.to_arpabet_variants("bevacizumab-vikg")[0]
    without_suffix = usan_stems.to_arpabet_variants("bevacizumab")[0]
    assert with_suffix[: len(without_suffix)] == without_suffix
    assert with_suffix[len(without_suffix) :]  # the code contributed something


def test_dropped_stem_is_not_applied():
    assert "umab" in usan_stems.DROPPED_STEMS
    assert usan_stems.resolve("adalimumab") is None


def test_resolve_none_for_unmatched_word():
    assert usan_stems.resolve("acetaminophen") is None


def test_resolve_note_names_the_stem_and_is_back_tested_when_measured():
    hit = usan_stems.resolve("tofacitinib")
    assert hit is not None
    _, note = hit
    assert "-tinib" in note
    assert "back-tested PEU improvement" in note


def test_resolve_note_says_untested_when_no_backtest_data():
    # "-zole" (troriluzole's suffix) used to be this test's example, but
    # Gemini-grounded retrieval (see gemini_grounded.py) has since sourced a
    # real troriluzole pronunciation, so `-zole` now has backtest data (and
    # measures negative -- see BACKTEST_RESULTS). "-pril" has no sourced
    # example in this dataset yet, so it is still genuinely untested.
    hit = usan_stems.resolve("lisinopril")
    assert hit is not None
    _, note = hit
    assert "-pril" in note
    assert "not back-tested" in note


