"""Fidelity report harness: exercised entirely against synthetic inputs, since
no real TTS run exists in this repo yet (see dose_r/report/FIDELITY.md)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from dose_r.report import fidelity as fid
from dose_r.report import power
from dose_r.report import stats
from dose_r.report.synthetic import synthetic_rows, synthetic_strata

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "dose_r" / "report" / "published_leaderboard.json"


# --- stats primitives --------------------------------------------------------


def test_norm_ppf_matches_known_values():
    assert stats.norm_ppf(0.5) == pytest.approx(0.0, abs=1e-6)
    assert stats.norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-4)
    assert stats.norm_ppf(0.9) == pytest.approx(1.281552, abs=1e-4)
    assert stats.norm_ppf(0.8) == pytest.approx(0.841621, abs=1e-4)


def test_wilson_ci_contains_point_estimate_and_widens_at_small_n():
    lo, hi = stats.wilson_ci(70, 100)
    assert lo < 0.7 < hi
    lo_small, hi_small = stats.wilson_ci(7, 10)
    assert (hi_small - lo_small) > (hi - lo)


def test_wilson_ci_degenerate_cases_stay_in_bounds():
    lo, hi = stats.wilson_ci(0, 10)
    assert 0.0 <= lo <= hi <= 1.0
    lo, hi = stats.wilson_ci(10, 10)
    assert 0.0 <= lo <= hi <= 1.0


def test_two_proportion_z_no_difference_gives_zero_z():
    z, p = stats.two_proportion_z(70, 100, 70, 100)
    assert z == pytest.approx(0.0, abs=1e-9)
    assert p == pytest.approx(1.0, abs=1e-9)


def test_two_proportion_z_large_difference_is_significant():
    z, p = stats.two_proportion_z(90, 100, 40, 100)
    assert abs(z) > 5
    assert p < 0.001


def test_spearman_perfect_agreement():
    x = {f"s{i}": float(i) for i in range(6)}
    y = {f"s{i}": float(i) for i in range(6)}
    rho, n = stats.spearman(x, y)
    assert rho == pytest.approx(1.0)
    assert n == 6


def test_spearman_perfect_disagreement():
    x = {f"s{i}": float(i) for i in range(6)}
    y = {f"s{i}": float(5 - i) for i in range(6)}
    rho, n = stats.spearman(x, y)
    assert rho == pytest.approx(-1.0)


def test_spearman_below_minimum_n_is_none():
    x = {"a": 1.0, "b": 2.0, "c": 3.0}
    y = {"a": 1.0, "b": 2.0, "c": 3.0}
    rho, n = stats.spearman(x, y)
    assert rho is None
    assert n == 3


def test_spearman_handles_ties():
    x = {"a": 1.0, "b": 1.0, "c": 2.0, "d": 3.0}
    y = {"a": 1.0, "b": 1.0, "c": 2.0, "d": 3.0}
    rho, n = stats.spearman(x, y)
    assert rho == pytest.approx(1.0)


# --- fidelity: stratified pass rates -----------------------------------------


def test_stratified_pass_rate_recovers_a_planted_era_gap():
    strata = synthetic_strata(n=2000, seed=1)
    rows = synthetic_rows(strata, established_p=0.90, new_p=0.60, seed=2)
    by_era = fid.stratified_pass_rate(rows, strata, "era")
    assert by_era["established"].rate == pytest.approx(0.90, abs=0.03)
    assert by_era["new"].rate == pytest.approx(0.60, abs=0.03)


def test_pass_rate_matches_manual_count():
    strata = synthetic_strata(n=300, seed=3)
    rows = synthetic_rows(strata, established_p=0.8, new_p=0.5, seed=4)
    stat = fid.pass_rate(rows)
    manual = sum(r.passed for r in rows.values()) / len(rows)
    assert stat.rate == pytest.approx(manual)
    assert stat.n == len(rows)


def test_pass_rate_on_empty_subset_is_none():
    strata = synthetic_strata(n=10, seed=5)
    rows = synthetic_rows(strata, established_p=0.8, new_p=0.5, seed=6)
    stat = fid.pass_rate(rows, row_ids=set())
    assert stat.n == 0
    assert stat.rate is None


def test_gap_table_only_covers_common_systems():
    observed = {"a": 0.7, "b": 0.5, "c": 0.9}
    published = {"a": 0.75, "b": 0.4}
    table = fid.gap_table(observed, published)
    assert {row["system"] for row in table} == {"a", "b"}
    a = next(r for r in table if r["system"] == "a")
    assert a["gap_pp"] == pytest.approx(-5.0)


def test_rank_correlation_flags_too_few_systems():
    result = fid.rank_correlation({"a": 0.9, "b": 0.6}, {"a": 0.91, "b": 0.63})
    assert result["meaningful"] is False
    assert result["spearman_rho"] is None


def test_rank_correlation_meaningful_with_enough_systems():
    observed = {f"s{i}": 1.0 - 0.1 * i for i in range(5)}
    published = {f"s{i}": 1.0 - 0.1 * i for i in range(5)}
    result = fid.rank_correlation(observed, published)
    assert result["meaningful"] is True
    assert result["spearman_rho"] == pytest.approx(1.0)


# --- attribution --------------------------------------------------------------


def test_attribution_by_confidence_flags_a_planted_low_tier_penalty():
    strata = synthetic_strata(n=2000, seed=7)
    # Build a run whose pass rate depends on era_confidence, not era, to
    # isolate the signal attribution_by_reference_confidence is meant to catch.
    import random

    rng = random.Random(8)
    rows = {}
    for row_id, s in strata.items():
        p = 0.9 if s["era_confidence"] == "high" else (0.75 if s["era_confidence"] == "medium" else 0.5)
        passed = rng.random() < p
        rows[row_id] = fid.RowResult(row_id, 4 if passed else 2, "brand", False)

    report = fid.attribution_by_reference_confidence(rows, strata)
    assert report["by_tier"]["high"]["rate"] > report["by_tier"]["low"]["rate"]
    assert report["high_vs_low"]["significant_at_0.05"] is True
    assert report["high_vs_low"]["gap_pp"] > 20


def test_attribution_by_confidence_flat_when_no_real_effect():
    strata = synthetic_strata(n=3000, seed=9)
    rows = synthetic_rows(strata, established_p=0.7, new_p=0.7, seed=10)  # era-blind, tier-blind
    report = fid.attribution_by_reference_confidence(rows, strata)
    assert not report["high_vs_low"]["significant_at_0.05"]


def test_attribution_against_hand_verified_counts_disagreements():
    rows = {
        "a": fid.RowResult("a", 4, "brand", False),  # passed
        "b": fid.RowResult("b", 2, "brand", False),  # failed
        "c": fid.RowResult("c", 5, "brand", False),  # passed
    }
    hand_verified = {"a": True, "b": True, "c": False}  # b and c disagree
    result = fid.attribution_against_hand_verified(rows, hand_verified)
    assert result["n"] == 3
    assert result["agreement"] == pytest.approx(1 / 3)
    assert {d["row_id"] for d in result["disagreements"]} == {"b", "c"}


def test_load_rows_jsonl_roundtrip(tmp_path):
    path = tmp_path / "rows.jsonl"
    records = [
        {"row_id": "dose-000", "score": 5, "name_type": "brand", "is_combination": False},
        {"row_id": "dose-001", "score": 2, "name_type": "generic", "is_combination": True},
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    rows = fid.load_rows_jsonl(path)
    assert rows["dose-000"].passed is True
    assert rows["dose-001"].passed is False
    assert len(rows) == 2


# --- power --------------------------------------------------------------------


def test_min_detectable_gap_shrinks_with_more_data():
    small = stats.min_detectable_gap(n=50, p=0.7)
    large = stats.min_detectable_gap(n=500, p=0.7)
    assert large < small


def test_min_detectable_gap_widest_near_50_percent():
    at_half = stats.min_detectable_gap(n=200, p=0.5)
    at_extreme = stats.min_detectable_gap(n=200, p=0.05)
    assert at_half > at_extreme


def test_sampling_power_matches_wilson_ci_order_of_magnitude():
    result = power.sampling_power(n=274, baseline_p=0.7)
    # Both numbers describe "how much noise is in a ~274-item pass rate";
    # they should be within a factor of 2 of each other, not orders apart.
    assert 0.5 < result.min_gap_pp / result.ci_halfwidth_pp < 3.0


def test_effective_sample_size_matches_arithmetic():
    result = power.effective_sample_size(total=274, low_confidence=185)
    assert result["effective_n_excluding_low_confidence"] == 89
    assert result["low_confidence_share"] == pytest.approx(185 / 274, abs=1e-3)


def test_propagate_reference_uncertainty_widens_ci_and_orders_by_error_rate():
    rows = power.propagate_reference_uncertainty(
        n_total=274, n_low_confidence=185, p_hat=0.7, assumed_label_error_rates=(0.1, 0.3)
    )
    lo_e, hi_e = rows[0], rows[1]
    assert hi_e["ci_halfwidth_pp_with_reference_uncertainty"] > lo_e["ci_halfwidth_pp_with_reference_uncertainty"]
    for row in rows:
        assert (
            row["ci_halfwidth_pp_with_reference_uncertainty"]
            >= row["ci_halfwidth_pp_sampling_only"]
        )


def test_propagate_reference_uncertainty_with_no_low_confidence_rows_equals_sampling_only():
    rows = power.propagate_reference_uncertainty(
        n_total=274, n_low_confidence=0, p_hat=0.7, assumed_label_error_rates=(0.2,)
    )
    row = rows[0]
    assert row["ci_halfwidth_pp_with_reference_uncertainty"] == pytest.approx(
        row["ci_halfwidth_pp_sampling_only"], abs=0.01
    )


# --- published leaderboard file ----------------------------------------------


def test_published_leaderboard_is_honest_about_coverage():
    data = json.loads(PUBLISHED.read_text())
    assert data["known_system_count"] == len(data["systems"])
    assert data["known_system_count"] < data["total_system_count"]
    gemini = data["systems"]["gemini_tts"]
    assert gemini["overall_pass_rate"] == pytest.approx(0.745)
    assert gemini["established_pass_rate"] == pytest.approx(0.891)
    assert gemini["new_pass_rate"] == pytest.approx(0.616)
    rx = data["systems"]["rxpronounce"]
    dragon = data["systems"]["azure_dragonhd"]
    assert rx["overall_pass_rate"] > gemini["overall_pass_rate"] > dragon["overall_pass_rate"]
