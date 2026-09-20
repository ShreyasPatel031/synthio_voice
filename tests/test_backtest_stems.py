import pytest

from dose_r.references import backtest_stems, usan_stems


def test_load_sourced_generics_excludes_low_confidence_and_brands():
    rows = backtest_stems.load_sourced_generics()
    assert rows
    for r in rows:
        assert r["name_type"] == "generic"
        assert r["confidence"] in {"high", "medium"}


def test_load_sourced_generics_excludes_the_known_bad_source():
    rows = backtest_stems.load_sourced_generics()
    assert all(r["ingredient"].lower() != "empagliflozin" for r in rows)


def test_backtest_only_scores_words_with_a_recognized_stem():
    per_stem = backtest_stems.backtest()
    for suffix, examples in per_stem.items():
        assert suffix in usan_stems.STEMS
        for e in examples:
            assert usan_stems.match_stem(e["ingredient"]).suffix == suffix


def test_backtest_error_is_non_negative():
    per_stem = backtest_stems.backtest()
    for examples in per_stem.values():
        for e in examples:
            assert e["g2p_error"] >= 0
            assert e["stem_error"] >= 0


def test_summarize_improvement_matches_the_hardcoded_backtest_results():
    # usan_stems.BACKTEST_RESULTS is a hand-mirrored snapshot of this
    # script's output (see the docstring in backtest_stems.py); this test
    # catches the two drifting silently apart.
    summary = {r["suffix"]: r for r in backtest_stems.summarize(backtest_stems.backtest())}
    for suffix, recorded in usan_stems.BACKTEST_RESULTS.items():
        measured = summary[suffix]
        assert measured["n"] == recorded["n"], suffix
        assert measured["avg_improvement"] == pytest.approx(recorded["improvement"], abs=1e-3), suffix
