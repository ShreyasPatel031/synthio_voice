import base64
import io
import json
import struct
import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dose_r.references import audio_manifest, audio_sources_drugscom, audio_verify
from scripts import fetch_drugscom_reference_audio as fdr

MANIFEST = ROOT / "data" / "reference_audio" / "manifest.jsonl"
DATASET = ROOT / "data" / "dose_v1.jsonl"


def make_wav(duration_s: float, rate: int = 16000, channels: int = 1) -> bytes:
    n_frames = int(duration_s * rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack(f"<{n_frames * channels}h", *([0] * (n_frames * channels))))
    return buf.getvalue()


def wav_b64(duration_s: float, **kw) -> str:
    return base64.b64encode(make_wav(duration_s, **kw)).decode()


# --- audio_verify -----------------------------------------------------------


def test_probe_wav_reads_duration_rate_channels():
    probe = audio_verify.probe_wav(make_wav(1.0, rate=22050, channels=2))
    assert probe.ok
    assert probe.format == "wav"
    assert probe.duration_s == pytest.approx(1.0, abs=0.01)
    assert probe.sample_rate_hz == 22050
    assert probe.channels == 2


def test_probe_wav_rejects_garbage():
    probe = audio_verify.probe_wav(b"not a wav file")
    assert not probe.ok
    assert probe.error


def test_duration_flag_bounds():
    assert audio_verify.duration_flag(0.29) is not None
    assert audio_verify.duration_flag(0.3) is None
    assert audio_verify.duration_flag(4.0) is None
    assert audio_verify.duration_flag(4.01) is not None
    assert audio_verify.duration_flag(None) is None


def test_estimate_syllables():
    assert audio_verify.estimate_syllables("Advil") == 2
    assert audio_verify.estimate_syllables("insulin glargine") == 6
    assert audio_verify.estimate_syllables("") == 1


def test_syllable_outliers_flags_disproportionate_duration():
    durations = {"Short": 0.4, "Also": 0.42, "Names": 0.38, "Suspect": 3.5}
    outliers = audio_verify.syllable_outliers(durations)
    assert "Suspect" in outliers
    assert "Short" not in outliers


def test_syllable_outliers_empty_input():
    assert audio_verify.syllable_outliers({}) == {}


# --- audio_manifest -----------------------------------------------------------


def test_record_key_defaults_query_to_ingredient():
    key = audio_manifest.record_key({"ingredient": "Advil", "source": "drugs.com"})
    assert key == ("Advil", "drugs.com", "Advil")


def test_save_load_round_trip(tmp_path):
    path = tmp_path / "manifest.jsonl"
    records = [
        {"ingredient": "B", "source": "s", "query": "B"},
        {"ingredient": "A", "source": "s", "query": "A"},
    ]
    audio_manifest.save(records, path)
    loaded = audio_manifest.load(path)
    assert [r["ingredient"] for r in loaded] == ["A", "B"]


def test_replace_source_only_touches_named_source():
    existing = [
        {"ingredient": "A", "source": "merriam-webster", "query": "A"},
        {"ingredient": "B", "source": "drugs.com", "query": "B", "old": True},
    ]
    fresh = [{"ingredient": "B", "source": "drugs.com", "query": "B", "old": False}]
    merged = audio_manifest.replace_source(existing, fresh, "drugs.com")
    assert len(merged) == 2
    by_ing = {r["ingredient"]: r for r in merged}
    assert by_ing["A"]["source"] == "merriam-webster"
    assert by_ing["B"]["old"] is False


def test_duplicate_groups_detects_shared_hash():
    records = [
        {"ingredient": "A", "sha256": "same"},
        {"ingredient": "B", "sha256": "same"},
        {"ingredient": "C", "sha256": "other"},
    ]
    dupes = audio_manifest.duplicate_groups(records)
    assert dupes == {"same": ["A", "B"]}


# --- audio_sources_drugscom ---------------------------------------------------


def test_merge_records_last_file_wins(tmp_path):
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "b.json"
    f1.write_text(json.dumps([{"name": "X", "status": 404, "audio_url": None, "audio_b64": None}]))
    f2.write_text(json.dumps([{"name": "X", "status": 200, "audio_url": "u", "audio_b64": "Zg=="}]))
    merged = audio_sources_drugscom.merge_records([f1, f2])
    assert merged["X"]["status"] == 200
    assert merged["X"]["audio_b64"] == "Zg=="


def test_audio_conflicts_detects_disagreement(tmp_path):
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "b.json"
    f1.write_text(json.dumps([{"name": "X", "status": 200, "audio_url": "u1", "audio_b64": "AAAA"}]))
    f2.write_text(json.dumps([{"name": "X", "status": 200, "audio_url": "u2", "audio_b64": "BBBB"}]))
    assert audio_sources_drugscom.audio_conflicts([f1, f2]) == ["X"]


def test_audio_conflicts_none_when_only_one_file_has_audio(tmp_path):
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "b.json"
    f1.write_text(json.dumps([{"name": "X", "status": 404, "audio_url": None, "audio_b64": None}]))
    f2.write_text(json.dumps([{"name": "X", "status": 200, "audio_url": "u", "audio_b64": "AAAA"}]))
    assert audio_sources_drugscom.audio_conflicts([f1, f2]) == []


def test_decode_audio_round_trips():
    raw = make_wav(0.5)
    record = {"audio_b64": base64.b64encode(raw).decode()}
    assert audio_sources_drugscom.decode_audio(record) == raw


# --- fetch_drugscom_reference_audio (the driver) ------------------------------


@pytest.fixture
def audio_dir(tmp_path, monkeypatch):
    scratch = ROOT / "data" / "reference_audio" / f"_test_scratch_{tmp_path.name}"
    monkeypatch.setattr(fdr, "AUDIO_DIR", scratch)
    yield scratch
    if scratch.exists():
        import shutil

        shutil.rmtree(scratch)


@pytest.fixture
def collected_file(tmp_path):
    path = tmp_path / "collected.json"
    path.write_text(
        json.dumps(
            [
                {"name": "Abilify", "status": 200, "audio_url": "https://x/1.wav",
                 "respell": "junk", "audio_b64": wav_b64(1.0)},
                {"name": "Advair", "status": 404, "audio_url": None,
                 "respell": None, "audio_b64": None},
                {"name": "Aspirin", "status": 200, "audio_url": None,
                 "respell": None, "audio_b64": None},
            ]
        )
    )
    return path


def test_run_builds_one_clip_and_two_misses(collected_file, audio_dir):
    clips, misses, conflicts = fdr.run([collected_file])
    assert conflicts == []
    assert [c["ingredient"] for c in clips] == ["Abilify"]
    assert clips[0]["source"] == "drugs.com"
    assert clips[0]["coverage"] == "full"
    assert clips[0]["respelling"] is None
    assert clips[0]["status"] == "ok"

    by_reason = {m["ingredient"]: m["reason"] for m in misses}
    assert by_reason["Advair"] == "url_not_found_not_absent"
    assert by_reason["Aspirin"] == "no_audio_on_page"


def test_run_ignores_respell_field(collected_file, audio_dir):
    clips, _, _ = fdr.run([collected_file])
    assert "junk" not in json.dumps(clips)


def test_run_is_idempotent_on_disk(collected_file, audio_dir):
    fdr.run([collected_file])
    written = audio_dir / "1.wav"
    assert written.exists()
    mtime_before = written.stat().st_mtime_ns

    clips_again, _, _ = fdr.run([collected_file])
    assert written.stat().st_mtime_ns == mtime_before
    assert len(clips_again) == 1


def test_run_flags_a_syllable_outlier(tmp_path, audio_dir):
    path = tmp_path / "collected.json"
    path.write_text(
        json.dumps(
            [
                {"name": "Advil", "status": 200, "audio_url": "https://x/a.wav",
                 "respell": None, "audio_b64": wav_b64(0.4)},
                {"name": "Aspirin", "status": 200, "audio_url": "https://x/b.wav",
                 "respell": None, "audio_b64": wav_b64(0.45)},
                {"name": "Zepbound", "status": 200, "audio_url": "https://x/c.wav",
                 "respell": None, "audio_b64": wav_b64(3.9)},
            ]
        )
    )
    clips, _, _ = fdr.run([path])
    by_ing = {c["ingredient"]: c for c in clips}
    assert by_ing["Zepbound"]["status"] == "flagged"
    assert any("possible name mismatch" in f for f in by_ing["Zepbound"]["flags"])
    assert by_ing["Advil"]["status"] == "ok"


# --- the real, checked-in manifest ---------------------------------------------


@pytest.fixture(scope="module")
def manifest_records():
    return audio_manifest.load(MANIFEST)


@pytest.fixture(scope="module")
def dataset_ingredients():
    with DATASET.open() as f:
        return {i for line in f for i in json.loads(line)["ingredients"]}


def test_manifest_has_both_sources(manifest_records):
    sources = {r["source"] for r in manifest_records}
    assert {"merriam-webster", "drugs.com"} <= sources


def test_manifest_drugs_com_row_count(manifest_records):
    drugscom = [r for r in manifest_records if r["source"] == "drugs.com"]
    assert len(drugscom) == 143
    assert len({r["ingredient"] for r in drugscom}) == 143


def test_manifest_no_duplicate_keys(manifest_records):
    keys = [audio_manifest.record_key(r) for r in manifest_records]
    assert len(keys) == len(set(keys))


def test_manifest_ingredients_are_in_dataset(manifest_records, dataset_ingredients):
    for r in manifest_records:
        assert r["ingredient"] in dataset_ingredients


def test_manifest_drugscom_rows_have_no_respelling(manifest_records):
    for r in manifest_records:
        if r["source"] == "drugs.com":
            assert r["respelling"] is None


def test_manifest_local_files_exist_and_hash_matches(manifest_records):
    for r in manifest_records:
        if r["source"] != "drugs.com":
            continue
        local_path = ROOT / r["local_path"]
        assert local_path.exists(), r["ingredient"]
        data = local_path.read_bytes()
        assert len(data) == r["bytes"]
        import hashlib

        assert hashlib.sha256(data).hexdigest() == r["sha256"]


def test_manifest_drugscom_durations_within_bounds(manifest_records):
    for r in manifest_records:
        if r["source"] == "drugs.com":
            assert audio_verify.MIN_DURATION_S <= r["duration_s"] <= audio_verify.MAX_DURATION_S


def test_manifest_flagged_rows_carry_a_reason(manifest_records):
    for r in manifest_records:
        if r["status"] == "flagged":
            assert r["flags"]
