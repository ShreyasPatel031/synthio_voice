"""Strata reconstruction: era (openFDA-derived) and difficulty (phonetic)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dose_r.strata import difficulty as diff
from dose_r.strata.build_strata import build_records, load_dataset, load_references
from dose_r.strata.era import CUTOFF_DATE, classify_row
from dose_r.strata.fda_lookup import FdaClient, Lookup, candidate_queries

ROOT = Path(__file__).resolve().parents[1]
STRATA = ROOT / "dose_r" / "strata" / "strata.jsonl"
DATASET = ROOT / "data" / "dose_v1.jsonl"


@pytest.fixture(scope="module")
def records():
    with STRATA.open() as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(scope="module")
def dataset_ids():
    return {json.loads(line)["id"] for line in DATASET.read_text().splitlines() if line.strip()}


def test_covers_every_dataset_row_exactly_once(records, dataset_ids):
    ids = [r["id"] for r in records]
    assert set(ids) == dataset_ids
    assert len(ids) == len(set(ids))


def test_era_and_difficulty_are_declared(records):
    for r in records:
        assert r["era"] in {"established", "new"}
        assert r["era_confidence"] in {"high", "medium", "low"}
        assert r["difficulty"] in {"easy", "medium", "hard"}


def test_heuristic_rows_never_carry_a_fabricated_date(records):
    for r in records:
        if r["era_source"] == "heuristic_no_fda_match":
            assert r["max_approval_date"] is None
            assert r["era"] == "new"
            assert r["era_confidence"] == "low"


def test_non_heuristic_rows_carry_a_real_date(records):
    for r in records:
        if r["era_source"] != "heuristic_no_fda_match":
            assert r["max_approval_date"] is not None


def test_era_matches_cutoff(records):
    for r in records:
        if r["max_approval_date"] is None:
            continue
        expected = "established" if r["max_approval_date"] < CUTOFF_DATE else "new"
        assert r["era"] == expected, r["id"]


def test_era_split_is_close_to_dose_targets(records):
    established = sum(1 for r in records if r["era"] == "established")
    new = sum(1 for r in records if r["era"] == "new")
    assert established + new == 274
    assert abs(established - 128) <= 5
    assert abs(new - 146) <= 5


def test_difficulty_tiers_cover_every_row_and_are_non_trivial(records):
    counts = {"easy": 0, "medium": 0, "hard": 0}
    for r in records:
        counts[r["difficulty"]] += 1
    assert sum(counts.values()) == 274
    assert all(n > 0 for n in counts.values())


def test_difficulty_score_is_monotonic_in_phoneme_count():
    low = diff.DifficultyFeatures(phoneme_count=2, name_length=6, usan_stem=False, biologic_suffix=False)
    high = diff.DifficultyFeatures(phoneme_count=20, name_length=6, usan_stem=False, biologic_suffix=False)
    assert low.score < high.score
    assert low.tier == "easy"
    assert high.tier == "hard"


def test_usan_stem_and_biologic_suffix_push_toward_hard():
    plain = diff.DifficultyFeatures(phoneme_count=6, name_length=8, usan_stem=False, biologic_suffix=False)
    stemmed = diff.DifficultyFeatures(phoneme_count=6, name_length=8, usan_stem=True, biologic_suffix=False)
    assert stemmed.score > plain.score
    assert stemmed.tier != "easy" or plain.tier == "easy"


def test_has_usan_stem_matches_known_examples():
    assert diff.has_usan_stem("adalimumab")
    assert diff.has_usan_stem("osimertinib")
    assert diff.has_usan_stem("semaglutide")
    assert diff.has_usan_stem("omeprazole")
    assert not diff.has_usan_stem("aspirin")


def test_has_biologic_suffix_matches_fda_qualifier():
    assert diff.has_biologic_suffix("bevacizumab-vikg")
    assert not diff.has_biologic_suffix("bevacizumab")


def test_candidate_queries_include_raw_products_fallback():
    """Regression test for the killed run's bug: openfda.* alone misses real,
    long-approved drugs (Eliquis, Benadryl, Biktarvy, Ubrelvy, Wegovy)."""
    queries = candidate_queries("eliquis", "brand")
    assert any(q.startswith("products.brand_name:") for q in queries)
    assert any(q.startswith("products.active_ingredients.name:") for q in queries)


def test_fixed_query_fields_find_known_openfda_misses():
    client = FdaClient()
    for name in ["Eliquis", "Benadryl", "Biktarvy", "Ubrelvy", "Wegovy"]:
        result = client.lookup(name, "brand")
        assert result.status == "hit", name
        assert result.matched_query.startswith("products."), name


def test_classify_row_combination_takes_latest_ingredient_date():
    lookups = {
        "old": Lookup("old", "20100101", "A1", "openfda.generic_name", "hit"),
        "new": Lookup("new", "20220101", "A2", "openfda.generic_name", "hit"),
    }
    result = classify_row(["old", "new"], lookups)
    assert result.max_approval_date == "2022-01-01"
    assert result.era == "new"


def test_classify_row_all_unresolved_is_heuristic_new():
    lookups = {"mystery": Lookup("mystery", None, None, None, "not_found")}
    result = classify_row(["mystery"], lookups)
    assert result.era == "new"
    assert result.era_confidence == "low"
    assert result.era_source == "heuristic_no_fda_match"
    assert result.max_approval_date is None


def test_build_records_is_deterministic():
    rows = load_dataset()
    arpabet, name_type = load_references()
    a = build_records(rows, arpabet, name_type)
    b = build_records(rows, arpabet, name_type)
    assert a == b
