"""Runner: resumability, concurrency, manifest integrity, run accounting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dose_r.adapters import build_adapter, load_system
from dose_r.adapters.config import system_from_dict
from dose_r.adapters.runner import RunPaths, load_dataset, load_manifest, run

DATASET = Path("data/dose_v1.jsonl")


@pytest.fixture(scope="module")
def rows():
    return load_dataset(DATASET)


def system(**options):
    return system_from_dict(
        "test-mock",
        {
            "backend": "mock",
            "model": "mock-v1",
            "voice": "mock-neutral",
            "concurrency": 8,
            "pricing": {
                "id": "test/priced",
                "as_of": "2026-09-17",
                "per_million_characters_usd": 15.0,
            },
            "retry": {"max_attempts": 3, "initial_backoff_s": 0.001, "jitter": 0.0},
            "options": options,
        },
    )


def test_dataset_shape(rows):
    assert len(rows) == 274
    assert sum(len(r["ingredients"]) for r in rows) == 286
    assert sum(r["is_combination"] for r in rows) == 9


def test_full_set_runs_end_to_end(tmp_path, rows):
    result = run(system(), rows, tmp_path / "full", dataset_path=DATASET)

    assert result.summary["synthesized"] == 274
    assert result.summary["failed"] == 0
    assert len(list(result.paths.audio_dir.glob("*.wav"))) == 274

    manifest = load_manifest(result.paths.manifest)
    assert len(manifest) == 274
    assert {r["id"] for r in rows} == set(manifest)
    assert all(Path(rec["audio_path"]).exists() for rec in manifest.values())

    meta = json.loads(result.paths.meta.read_text())
    assert meta["dataset"]["rows"] == 274
    assert meta["dataset"]["ingredient_spans"] == 286
    assert meta["system"]["pricing"]["id"] == "test/priced"
    assert meta["summary"]["cost_usd_this_run"] > 0


def test_manifest_carries_judge_inputs(tmp_path, rows):
    result = run(system(), rows[:3], tmp_path / "judge", dataset_path=DATASET)
    record = load_manifest(result.paths.manifest)["dose-000"]
    assert record["item"]["name"] == "Abilify"
    assert record["item"]["spans"] == [[17, 24]]
    assert record["text"].startswith("We plan to start Abilify")
    assert record["score"] is None  # the judge fills this in later


def test_resume_skips_completed_and_does_not_recharge(tmp_path, rows):
    subset = rows[:20]
    run_dir = tmp_path / "resume"

    first = run(system(), subset, run_dir, dataset_path=DATASET)
    assert first.summary["synthesized"] == 20
    first_cost = first.summary["cost_usd_this_run"]

    second = run(system(), subset, run_dir, dataset_path=DATASET)
    assert second.summary["synthesized"] == 0
    assert second.summary["resumed_skipped"] == 20
    assert second.summary["cost_usd_this_run"] == 0.0
    assert second.summary["cost_usd_including_resumed"] == pytest.approx(first_cost)

    # an interrupted run resumes into the same manifest and only pays the rest
    third = run(system(), rows[:30], run_dir, dataset_path=DATASET)
    assert third.summary["synthesized"] == 10
    assert third.summary["resumed_skipped"] == 20
    assert len(load_manifest(third.paths.manifest)) == 30


def test_resume_re_synthesizes_when_source_text_changes(tmp_path, rows):
    subset = [dict(r) for r in rows[:5]]
    run_dir = tmp_path / "textchange"
    run(system(), subset, run_dir, dataset_path=DATASET)

    subset[2]["sentence"] = subset[2]["sentence"].replace("the", "a", 1)
    again = run(system(), subset, run_dir, dataset_path=DATASET)

    assert again.summary["synthesized"] == 1
    assert again.records[0].item_id == subset[2]["id"]


def test_resume_re_synthesizes_when_audio_file_is_missing(tmp_path, rows):
    run_dir = tmp_path / "missingaudio"
    result = run(system(), rows[:5], run_dir, dataset_path=DATASET)
    (result.paths.audio_dir / "dose-002.wav").unlink()

    again = run(system(), rows[:5], run_dir, dataset_path=DATASET)
    assert [r.item_id for r in again.records] == ["dose-002"]
    assert (result.paths.audio_dir / "dose-002.wav").exists()


def test_resume_retries_prior_failures(tmp_path, rows):
    run_dir = tmp_path / "failures"
    first = run(system(permanent_failure_ids=["dose-001"]), rows[:4], run_dir, dataset_path=DATASET)
    assert first.summary["failed"] == 1

    second = run(system(), rows[:4], run_dir, dataset_path=DATASET)
    assert second.summary["synthesized"] == 1
    assert second.summary["resumed_skipped"] == 3
    assert load_manifest(second.paths.manifest)["dose-001"]["status"] == "ok"


def test_no_resume_flag_reruns_everything(tmp_path, rows):
    run_dir = tmp_path / "noresume"
    run(system(), rows[:5], run_dir, dataset_path=DATASET)
    again = run(system(), rows[:5], run_dir, resume=False, dataset_path=DATASET)
    assert again.summary["synthesized"] == 5
    assert again.summary["resumed_skipped"] == 0


def test_manifest_last_record_per_item_wins(tmp_path, rows):
    run_dir = tmp_path / "lastwins"
    run(system(permanent_failure_ids=["dose-000"]), rows[:2], run_dir, dataset_path=DATASET)
    run(system(), rows[:2], run_dir, dataset_path=DATASET)

    lines = [json.loads(l) for l in RunPaths(run_dir).manifest.read_text().splitlines()]
    assert sum(1 for l in lines if l["item_id"] == "dose-000") == 2
    assert load_manifest(RunPaths(run_dir).manifest)["dose-000"]["status"] == "ok"


def test_truncated_manifest_line_is_survivable(tmp_path, rows):
    run_dir = tmp_path / "truncated"
    result = run(system(), rows[:3], run_dir, dataset_path=DATASET)
    with result.paths.manifest.open("a") as f:
        f.write('{"item_id": "dose-003", "stat')

    again = run(system(), rows[:3], run_dir, dataset_path=DATASET)
    assert again.summary["resumed_skipped"] == 3


def test_concurrency_is_honoured_and_output_is_order_independent(tmp_path, rows):
    subset = rows[:12]
    serial = run(system(), subset, tmp_path / "c1", concurrency=1, dataset_path=DATASET)
    parallel = run(system(), subset, tmp_path / "c8", concurrency=8, dataset_path=DATASET)

    assert json.loads(parallel.paths.meta.read_text())["concurrency"] == 8
    assert [r.item_id for r in serial.records] == [r.item_id for r in parallel.records]
    assert [r.audio_sha256 for r in serial.records] == [r.audio_sha256 for r in parallel.records]


def test_latency_and_ttfa_percentiles_reported(tmp_path, rows):
    result = run(
        system(base_latency_s=0.02, ttfa_fraction=0.25), rows[:8], tmp_path / "lat", dataset_path=DATASET
    )
    latency = result.summary["latency_ms"]
    ttfa = result.summary["ttfa_ms"]
    assert latency["n"] == ttfa["n"] == 8
    assert result.summary["ttfa_coverage"] == "8/8"
    assert ttfa["p50"] < latency["p50"]
    assert latency["p50"] <= latency["p90"] <= latency["max"]


def test_retry_counts_surface_in_summary(tmp_path, rows):
    result = run(system(transient_failures_per_item=1), rows[:4], tmp_path / "retry", dataset_path=DATASET)
    assert result.summary["synthesized"] == 4
    assert result.summary["items_with_retries"] == 4


def test_failed_items_recorded_with_error_class(tmp_path, rows):
    result = run(
        system(permanent_failure_ids=["dose-000", "dose-002"]), rows[:4], tmp_path / "err", dataset_path=DATASET
    )
    assert result.summary["failed"] == 2
    assert result.summary["errors_by_class"] == {"PermanentError": 2}
    assert not (result.paths.audio_dir / "dose-000.wav").exists()


def test_cli_runs_a_subset(tmp_path, capsys):
    from dose_r.adapters.runner import main

    code = main(["--system", "mock", "--limit", "3", "--run-dir", str(tmp_path / "cli")])
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["synthesized"] == 3


def test_default_system_is_the_offline_mock():
    assert load_system("mock").backend == "mock"
    assert build_adapter(load_system("mock")).backend == "mock"
