"""Harness tests. Everything here runs offline against the mock backend."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dose_r import dataset, report
from dose_r.adapters import build_adapter
from dose_r.adapters.mock import DEFECT_MODES
from dose_r.runner import BenchmarkRunner, RunConfig
from dose_r.scoring import build_scorer
from dose_r.scoring.base import PASS_THRESHOLD


# --- dataset ---------------------------------------------------------------
def test_dataset_counts_match_published_split():
    df = dataset.load_raw()
    rep = dataset.validate(df)
    assert rep["ok"], rep["problems"]
    assert rep["rows"] == 274
    assert rep["strata"] == {"brand": 143, "generic": 131}


def test_generic_label_suffix_is_stripped():
    """All 131 generic rows ship as 'name (generic)'; the span target must not."""
    items = dataset.load_items()
    generics = [i for i in items if i.name_type == "generic"]
    assert len(generics) == 131
    assert all(i.drug_raw.endswith(" (generic)") for i in generics)
    assert not any(i.drug.endswith(" (generic)") for i in generics)


def test_every_sentence_contains_its_normalized_drug_name():
    """Without this, the judge has no span to score and items fail silently."""
    for item in dataset.load_items():
        assert item.drug.lower() in item.sentence.lower(), item.drug


def test_normalize_is_idempotent_and_leaves_brands_alone():
    n = dataset.normalize_drug_name
    assert n("acetaminophen (generic)") == "acetaminophen"
    assert n(n("acetaminophen (generic)")) == "acetaminophen"
    assert n("Abilify") == "Abilify"
    # A real parenthetical mid-name must survive; only the trailing label goes.
    assert n("insulin icodec-abae") == "insulin icodec-abae"


# --- adapters --------------------------------------------------------------
@pytest.mark.parametrize("mode", DEFECT_MODES)
def test_mock_adapter_emits_valid_wav(mode):
    res = build_adapter(f"mock-{mode}").synthesize("Take Abilify daily.", "abilify")
    assert res.ok and res.audio.startswith(b"RIFF")
    assert res.total_ms > 0


def test_mock_is_deterministic():
    a = build_adapter("mock-perfect").synthesize("Take Abilify daily.", "x")
    b = build_adapter("mock-perfect").synthesize("Take Abilify daily.", "x")
    assert a.audio == b.audio


def test_retry_then_failure_is_reported_not_raised():
    from dose_r.adapters.mock import MockTTSAdapter
    from dose_r.config import ALL_SYSTEMS

    adapter = MockTTSAdapter(ALL_SYSTEMS["mock-perfect"], fail_rate=1.0)
    res = adapter.synthesize("anything", "x", max_attempts=2, backoff_s=0.0)
    assert not res.ok and res.attempts == 2 and res.error


def test_non_streaming_adapter_reports_no_ttfa():
    """TTFA must stay None rather than being aliased to total_ms."""
    res = build_adapter("mock-perfect").synthesize("Take Abilify daily.", "x")
    assert res.streaming is False
    assert res.ttfa_ms is None


def test_cost_scales_with_characters():
    from dose_r.adapters import GoogleTTSAdapter
    from dose_r.config import CHEAP_TIER

    a = GoogleTTSAdapter(CHEAP_TIER["gtts-standard-c"])
    c1, _ = a.estimate_cost(1000)
    c2, _ = a.estimate_cost(2000)
    assert c2 == pytest.approx(c1 * 2)
    assert c1 > 0


# --- scoring ---------------------------------------------------------------
def test_standin_scorer_is_flagged_as_not_pronunciation():
    assert build_scorer("standin").measures_pronunciation is False


def test_controls_separate_good_from_defective():
    """Calibration step 4: the harness must cleanly split known-good from known-bad."""
    item = dataset.load_items()[0]
    scorer = build_scorer("standin")

    def s(mode):
        res = build_adapter(f"mock-{mode}").synthesize(item.sentence, item.item_id)
        return scorer.score(item, res).score

    good = s("perfect")
    assert good >= PASS_THRESHOLD
    for bad in ("truncated", "silent", "overlong"):
        assert s(bad) < good, f"{bad} scored >= perfect"


def test_failed_synthesis_scores_zero_not_none():
    from dose_r.adapters.base import SynthesisResult

    item = dataset.load_items()[0]
    failed = SynthesisResult(system_id="x", item_id=item.item_id, ok=False,
                             error="boom")
    sr = build_scorer("standin").score(item, failed)
    assert sr.score == 0.0 and sr.passed is False


# --- runner / report -------------------------------------------------------
def test_run_is_resumable_and_does_not_duplicate(tmp_path):
    items = dataset.load_items()[:6]
    cfg = RunConfig(systems=["mock-perfect"], run_id="t", concurrency=2, limit=6)

    BenchmarkRunner(cfg, scorer=build_scorer("standin"), runs_dir=tmp_path).run(items)
    first = report.load_records(tmp_path / "t" / "results.jsonl")

    # Second pass should find everything cached and add nothing.
    BenchmarkRunner(cfg, scorer=build_scorer("standin"), runs_dir=tmp_path).run(items)
    second = report.load_records(tmp_path / "t" / "results.jsonl")

    assert len(first) == 6
    assert len(second) == 6


def test_runner_persists_audio_and_manifest(tmp_path):
    items = dataset.load_items()[:3]
    cfg = RunConfig(systems=["mock-perfect"], run_id="t", limit=3)
    runner = BenchmarkRunner(cfg, scorer=build_scorer("standin"), runs_dir=tmp_path)
    runner.run(items)

    manifest = json.loads((runner.run_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["scorer_measures_pronunciation"] is False

    for rec in report.load_records(runner.results_path):
        assert (runner.run_dir / rec["audio_path"]).exists()


def test_report_refuses_dose_comparison_for_standin_scorer(tmp_path):
    items = dataset.load_items()[:3]
    cfg = RunConfig(systems=["mock-perfect"], run_id="t", limit=3)
    runner = BenchmarkRunner(cfg, scorer=build_scorer("standin"), runs_dir=tmp_path)
    runner.run(items)

    manifest = json.loads((runner.run_dir / "manifest.json").read_text())
    summaries = report.summarize(report.load_records(runner.results_path))

    with pytest.raises(report.NotLeaderboardComparable):
        report.compare_to_dose(summaries, manifest,
                               {"mock-perfect": "google-gemini-3.1-flash-tts"})


def test_report_warns_loudly_on_standin_scorer(tmp_path):
    items = dataset.load_items()[:3]
    cfg = RunConfig(systems=["mock-perfect"], run_id="t", limit=3)
    runner = BenchmarkRunner(cfg, scorer=build_scorer("standin"), runs_dir=tmp_path)
    runner.run(items)

    manifest = json.loads((runner.run_dir / "manifest.json").read_text())
    text = report.render_text(
        report.summarize(report.load_records(runner.results_path)), manifest)
    assert "NOT A PRONUNCIATION JUDGEMENT" in text
