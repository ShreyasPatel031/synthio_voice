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


# --- retry/backoff classification -------------------------------------------
# Regression coverage for the 8/274 gtts-chirp3hd-achernar failures in
# runs/standin-v1: all 8 were HTTP 429 RESOURCE_EXHAUSTED, clustered together
# in item order (i.e. concurrency-correlated), and all 3 attempts were burned
# with a deterministic 2s/4s backoff that lets 6 concurrent workers collide
# with the rate limit in lockstep. Re-running the same 8 items sequentially
# against the live API succeeded first-try, confirming the input text was
# never the problem. Everything below runs offline against a scripted fake
# `requests.Session` -- no network, no spend.

class _FakeResponse:
    def __init__(self, status_code: int, text: str = "", json_data=None,
                 headers=None):
        self.status_code = status_code
        self.text = text
        self._json = json_data or {}
        self.headers = headers or {}

    def json(self):
        return self._json


class _ScriptedSession:
    """Stands in for requests.Session: returns/raises each scripted item in order."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _ok_response():
    import base64
    audio = b"RIFF" + b"\x00" * 40  # just needs to start with RIFF
    return _FakeResponse(200, json_data={"audioContent": base64.b64encode(audio).decode()})


@pytest.fixture(autouse=True)
def _no_real_auth(monkeypatch):
    """Adapter tests below never touch the network, but _synthesize still calls
    auth.auth_headers() before the (fake) request -- stub it so these tests
    don't depend on GOOGLE_APPLICATION_CREDENTIALS_JSON being set, and never
    trigger a real OAuth token refresh."""
    monkeypatch.setattr("dose_r.adapters.google_tts.auth.auth_headers", lambda: {})


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Backoff sleeps are real seconds otherwise; tests only care that sleep
    *was requested* with the right duration, not that it actually elapsed."""
    monkeypatch.setattr("dose_r.adapters.base.time.sleep", lambda s: None)


def test_429_is_retried_and_can_still_succeed():
    """This is exactly runs/standin-v1's failure mode: a 429 that clears up
    a moment later must not be treated as permanent."""
    from dose_r.adapters import GoogleTTSAdapter
    from dose_r.config import CHEAP_TIER

    session = _ScriptedSession([
        _FakeResponse(429, text='{"error": {"status": "RESOURCE_EXHAUSTED"}}'),
        _ok_response(),
    ])
    adapter = GoogleTTSAdapter(CHEAP_TIER["gtts-chirp3hd-achernar"], session=session)
    res = adapter.synthesize("Take Abilify daily.", "x", max_attempts=3, backoff_s=0.01)

    assert res.ok
    assert res.attempts == 2
    assert session.calls == 2


def test_429_exhausting_all_attempts_reports_the_429():
    from dose_r.adapters import GoogleTTSAdapter
    from dose_r.adapters.base import RetryableError
    from dose_r.config import CHEAP_TIER

    session = _ScriptedSession([_FakeResponse(429, text="quota")] * 3)
    adapter = GoogleTTSAdapter(CHEAP_TIER["gtts-chirp3hd-achernar"], session=session)
    res = adapter.synthesize("Take Abilify daily.", "x", max_attempts=3, backoff_s=0.01)

    assert not res.ok
    assert res.attempts == 3
    assert "429" in res.error
    assert RetryableError.__name__ in res.error


def test_400_is_not_retried():
    """A bad-request 400 will fail identically on every attempt -- retrying it
    just burns two backoff sleeps and two attempts for nothing."""
    from dose_r.adapters import GoogleTTSAdapter
    from dose_r.config import CHEAP_TIER

    session = _ScriptedSession([_FakeResponse(400, text="invalid voice name")])
    adapter = GoogleTTSAdapter(CHEAP_TIER["gtts-chirp3hd-achernar"], session=session)
    res = adapter.synthesize("Take Abilify daily.", "x", max_attempts=3, backoff_s=0.01)

    assert not res.ok
    assert res.attempts == 1          # failed fast, did not spend all 3
    assert session.calls == 1         # and never called the backend again
    assert "400" in res.error


def test_503_is_retryable_like_429():
    from dose_r.adapters import GoogleTTSAdapter
    from dose_r.config import CHEAP_TIER

    session = _ScriptedSession([_FakeResponse(503, text="backend unavailable"), _ok_response()])
    adapter = GoogleTTSAdapter(CHEAP_TIER["gtts-chirp3hd-achernar"], session=session)
    res = adapter.synthesize("Take Abilify daily.", "x", max_attempts=3, backoff_s=0.01)

    assert res.ok and res.attempts == 2


def test_retry_after_header_is_honored():
    """When the backend names its own cooldown, the retry loop should wait at
    least that long rather than substituting a shorter guess."""
    from dose_r.adapters.base import _backoff_delay

    delay = _backoff_delay(base_s=2.0, attempt=1, retry_after_s=5.0)
    assert delay >= 5.0


def test_backoff_has_jitter_not_lockstep():
    """The bug: a deterministic 2s/4s backoff lets N concurrent workers that
    hit a shared rate limit at the same instant all retry at the same instant
    again, colliding with it repeatedly. Jitter must make the delays vary."""
    from dose_r.adapters.base import _backoff_delay

    delays = {_backoff_delay(base_s=2.0, attempt=2, retry_after_s=None) for _ in range(20)}
    assert len(delays) > 1, "20 draws all identical -- backoff is not jittered"
    assert all(0 <= d <= 4.0 for d in delays), "delay exceeded the attempt-2 ceiling"


def test_timeout_is_retryable_not_fatal():
    """A slow response is not evidence the request was malformed."""
    import requests

    from dose_r.adapters import GoogleTTSAdapter
    from dose_r.config import CHEAP_TIER

    session = _ScriptedSession([requests.Timeout("read timed out"), _ok_response()])
    adapter = GoogleTTSAdapter(CHEAP_TIER["gtts-chirp3hd-achernar"], session=session)
    res = adapter.synthesize("Take Abilify daily.", "x", max_attempts=3, backoff_s=0.01)

    assert res.ok and res.attempts == 2


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


# --- ASR round-trip scorer ---------------------------------------------------
# All offline: score_pronunciation() and locate_recognized_span() are pure
# functions (no network), and AsrRoundTripScorer.score() below is exercised
# against a scripted fake requests.Session, exactly like the GoogleTTSAdapter
# retry tests above -- no real Cloud STT call is ever made in this suite.
from dose_r.scoring.asr_roundtrip import (  # noqa: E402
    AsrRoundTripScorer, locate_recognized_span, score_pronunciation,
)


def test_score_pronunciation_exact_match_is_five():
    score, components = score_pronunciation("Abilify", "Abilify")
    assert score == 5.0
    assert components["exact_match"] == 1.0


def test_score_pronunciation_is_case_and_spacing_insensitive():
    """Collapsing (lowercase, strip non-alphanumerics) means a multi-word drug
    recognized with different word breaks still counts as an exact match --
    ASR's word segmentation is arbitrary, not part of pronunciation."""
    score, components = score_pronunciation("insulin icodec-abae", "insulin I codec abae")
    assert score == 5.0
    assert components["exact_match"] == 1.0


def test_score_pronunciation_metaphone_mismatch_caps_below_pass_threshold():
    """'retatrutide' -> 'retro tide': shares enough characters for a high
    Jaro-Winkler score (common prefix), but the metaphone codes disagree, so
    it must not be able to buy its way to a pass on surface similarity alone."""
    score, components = score_pronunciation("retatrutide", "retro tide")
    assert components["exact_match"] == 0.0
    assert components["metaphone_match"] == 0.0
    assert components["jaro_winkler"] > 0.5      # surface similarity is real...
    assert score < PASS_THRESHOLD                # ...but must not pass on it alone


def test_score_pronunciation_metaphone_match_scores_above_floor():
    """Same phonetic code but not an exact string -- should land in the upper
    band (3.5-5.0), not be punished the way a metaphone mismatch is."""
    score, components = score_pronunciation("Prozac", "Pro-zack")
    assert components["metaphone_match"] == 1.0
    assert score >= 3.5


def test_score_pronunciation_empty_recognized_span_scores_zero():
    """No hypothesis words aligned to the drug at all -- a real, scoreable
    'not recognized here' result, not a crash and not a false phonetic match."""
    score, components = score_pronunciation("retatrutide", "")
    assert score == 0.0
    assert components == {"exact_match": 0.0, "jaro_winkler": 0.0, "metaphone_match": 0.0}


def test_score_pronunciation_never_exceeds_scale():
    score, _ = score_pronunciation("x", "x")
    assert score <= 5.0


def test_locate_recognized_span_simple_case():
    sentence = "I recommend taking Advil to help relieve the pain."
    hyp = "I recommend taking Advil to help relieve the pain".split()
    assert locate_recognized_span(sentence, "Advil", hyp) == "Advil"


def test_locate_recognized_span_handles_multiword_drug():
    sentence = ("By activating the insulin receptor, insulin icodec-abae "
                "effectively stimulates peripheral glucose uptake.")
    # ASR frequently splits "icodec-abae" into separate tokens and drops the comma.
    hyp = ("by activating the insulin receptor insulin I codec abae "
           "effectively stimulates peripheral glucose uptake").split()
    span = locate_recognized_span(sentence, "insulin icodec-abae", hyp)
    assert span == "insulin I codec abae"


def test_locate_recognized_span_robust_to_mismatch_elsewhere_in_sentence():
    """An ASR error far from the drug name must not disturb the drug span
    alignment -- this is exactly why word-index-based lookup would be wrong
    and difflib-based alignment is used instead."""
    sentence = "I recommend taking Advil to help relieve the minor pain today."
    # "recommend" -> "suggest" (substitution) well before the drug; "today" is
    # dropped at the very end, well after it.
    hyp = "I suggest taking Advil to help relieve the minor pain".split()
    assert locate_recognized_span(sentence, "Advil", hyp) == "Advil"


def test_locate_recognized_span_empty_when_drug_dropped_entirely():
    """ASR recognized the rest of the sentence but produced nothing at all
    for the drug's position -- must come back empty, not a neighboring word."""
    sentence = "I recommend taking Advil to help relieve the pain."
    hyp = "I recommend taking to help relieve the pain".split()
    assert locate_recognized_span(sentence, "Advil", hyp) == ""


def test_locate_recognized_span_unknown_drug_name_does_not_crash():
    """dataset.validate() guarantees the drug appears in its sentence for the
    shipped data, but this must degrade gracefully rather than raise if that
    ever stops being true."""
    assert locate_recognized_span("nothing to see here", "not-present-drug", ["a", "b"]) == ""


class _FakeSTTResponse:
    def __init__(self, status_code: int, json_data: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._json = json_data or {}
        self.text = text

    def json(self):
        return self._json


class _ScriptedSTTSession:
    def __init__(self, response):
        self._response = response
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _stt_result(transcript_words: list[str], confidence: float = 0.9) -> dict:
    return {
        "results": [{
            "alternatives": [{
                "transcript": " ".join(transcript_words),
                "confidence": confidence,
                "words": [{"word": w} for w in transcript_words],
            }],
        }],
    }


def test_asr_roundtrip_scorer_is_flagged_as_measuring_pronunciation():
    assert AsrRoundTripScorer().measures_pronunciation is True


def test_asr_roundtrip_happy_path_scores_and_carries_asymmetry_note():
    item = next(i for i in dataset.load_items() if i.drug == "Advil")
    words = item.sentence.replace(",", "").replace(".", "").split()
    session = _ScriptedSTTSession(_FakeSTTResponse(200, _stt_result(words, confidence=0.95)))
    scorer = AsrRoundTripScorer(session=session)

    from dose_r.adapters.base import SynthesisResult
    res = SynthesisResult(system_id="x", item_id=item.item_id, ok=True,
                          audio=b"RIFF" + b"\x00" * 40, sample_rate_hz=24000)
    sr = scorer.score(item, res)

    assert sr.scoreable is True
    assert sr.score == 5.0
    assert sr.components["exact_match"] == 1.0
    assert sr.components["asr_confidence"] == pytest.approx(0.95)
    assert "ASR round-trip proxy" in sr.notes  # the asymmetry caveat, always present
    assert session.calls == 1


def test_asr_roundtrip_empty_results_is_scoreable_false_not_zero():
    """This is the critical honesty requirement: no signal must never be
    silently reported as a confirmed mispronunciation (score 0)."""
    item = dataset.load_items()[0]
    session = _ScriptedSTTSession(_FakeSTTResponse(200, {"results": []}))
    scorer = AsrRoundTripScorer(session=session)

    from dose_r.adapters.base import SynthesisResult
    res = SynthesisResult(system_id="x", item_id=item.item_id, ok=True,
                          audio=b"RIFF" + b"\x00" * 40, sample_rate_hz=24000)
    sr = scorer.score(item, res)

    assert sr.scoreable is False
    assert sr.score is None
    assert sr.error


def test_asr_roundtrip_request_failure_is_scoreable_false_not_zero():
    item = dataset.load_items()[0]
    session = _ScriptedSTTSession(_FakeSTTResponse(500, text="backend error"))
    scorer = AsrRoundTripScorer(session=session)

    from dose_r.adapters.base import SynthesisResult
    res = SynthesisResult(system_id="x", item_id=item.item_id, ok=True,
                          audio=b"RIFF" + b"\x00" * 40, sample_rate_hz=24000)
    sr = scorer.score(item, res)

    assert sr.scoreable is False
    assert sr.score is None


def test_asr_roundtrip_synthesis_failure_scores_zero_not_none():
    """Mirrors test_failed_synthesis_scores_zero_not_none for the standin
    scorer: an upstream synthesis failure is a known 0, not an unscoreable."""
    item = dataset.load_items()[0]
    from dose_r.adapters.base import SynthesisResult
    failed = SynthesisResult(system_id="x", item_id=item.item_id, ok=False, error="boom")
    sr = AsrRoundTripScorer(session=_ScriptedSTTSession(Exception("must not be called"))).score(
        item, failed
    )
    assert sr.score == 0.0 and sr.scoreable is True


# --- reference-audio grounding ----------------------------------------------
def _write_manifest(tmp_path, records):
    p = tmp_path / "manifest.jsonl"
    with p.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


def _make_wav(path, *, channels=1, rate=16000, n_frames=800):
    import struct
    import wave
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack(f"<{n_frames * channels}h", *([0] * (n_frames * channels))))


def test_available_clips_prefers_merriam_webster_over_drugs_com(tmp_path):
    """Merriam-Webster is preferred: it is an actual pronouncing dictionary
    (carries a written respelling) rather than just an audio file. Confirmed
    on real data -- Drugs.com's and MW's Aspirin clips sound different, but
    MW's own respelling "as-p(schwa-)rin" documents the schwa as optional, so
    the two clips are the same entry's accepted variants, not a conflict."""
    from dose_r.references import reference_clips

    wav = tmp_path / "clip.wav"
    _make_wav(wav)
    manifest = _write_manifest(tmp_path, [
        {"ingredient": "Abilify", "name_type": "brand", "source": "drugs.com",
         "coverage": "full", "local_path": str(wav), "format": "wav",
         "sample_rate_hz": 16000, "duration_s": 1.0},
        {"ingredient": "Abilify", "name_type": "brand", "source": "merriam-webster",
         "coverage": "full", "local_path": str(wav), "format": "wav",
         "sample_rate_hz": 16000, "duration_s": 1.0, "respelling": "uh-BIL-uh-fy"},
    ])
    clips = reference_clips.available_clips(manifest)
    assert clips["Abilify"].source == "merriam-webster"
    assert clips["Abilify"].respelling == "uh-BIL-uh-fy"


def test_available_clips_skips_records_whose_file_is_missing(tmp_path):
    from dose_r.references import reference_clips

    manifest = _write_manifest(tmp_path, [
        {"ingredient": "Ghostidine", "name_type": "generic", "source": "merriam-webster",
         "coverage": "full", "local_path": str(tmp_path / "does-not-exist.mp3"),
         "format": "mp3", "sample_rate_hz": 22050, "duration_s": 1.0},
    ])
    # The gitignored-source gap (Merriam-Webster/UMich audio not committed to
    # this repo) must be silently skipped, not raised as an error.
    assert reference_clips.available_clips(manifest) == {}


def test_stt_config_declares_channel_count_only_when_not_mono(tmp_path):
    from dose_r.references.reference_clips import ReferenceClip, stt_config_for_clip

    mono = tmp_path / "mono.wav"
    stereo = tmp_path / "stereo.wav"
    _make_wav(mono, channels=1, rate=22050)
    _make_wav(stereo, channels=2, rate=44100)

    mono_clip = ReferenceClip("x", "brand", "drugs.com", mono, "wav", 22050, 1.0)
    stereo_clip = ReferenceClip("y", "brand", "drugs.com", stereo, "wav", 44100, 1.0)

    mono_cfg = stt_config_for_clip(mono_clip)
    stereo_cfg = stt_config_for_clip(stereo_clip)

    assert "audioChannelCount" not in mono_cfg
    assert mono_cfg["sampleRateHertz"] == 22050
    assert stereo_cfg["audioChannelCount"] == 2
    assert stereo_cfg["sampleRateHertz"] == 44100


def test_stt_config_for_mp3_uses_manifest_rate():
    from dose_r.references.reference_clips import ReferenceClip, stt_config_for_clip

    clip = ReferenceClip("x", "brand", "merriam-webster", Path("/nonexistent.mp3"),
                         "mp3", 11025, 1.0)
    cfg = stt_config_for_clip(clip)
    assert cfg == {"encoding": "MP3", "sampleRateHertz": 11025,
                   "languageCode": "en-US", "model": "latest_long"}


def test_transcribe_reference_clip_empty_result_is_unrecognizable_not_error(tmp_path):
    from dose_r.references.audio_grounded import transcribe_reference_clip
    from dose_r.references.reference_clips import ReferenceClip

    wav = tmp_path / "silent.wav"
    _make_wav(wav)
    clip = ReferenceClip("Mystery", "generic", "drugs.com", wav, "wav", 16000, 0.5)

    session = _ScriptedSTTSession(_FakeSTTResponse(200, {"results": []}))
    t = transcribe_reference_clip(clip, {}, session=session)

    assert session.calls == 1
    assert t.recognized is None
    assert t.asr_recognizable is False


def test_transcribe_reference_clip_happy_path(tmp_path):
    from dose_r.references.audio_grounded import transcribe_reference_clip
    from dose_r.references.reference_clips import ReferenceClip

    wav = tmp_path / "abilify.wav"
    _make_wav(wav)
    clip = ReferenceClip("Abilify", "brand", "drugs.com", wav, "wav", 16000, 0.5)

    session = _ScriptedSTTSession(_FakeSTTResponse(200, _stt_result(["Abilify"], 0.95)))
    t = transcribe_reference_clip(clip, {}, session=session)

    assert t.recognized == "Abilify"
    assert t.asr_recognizable is True
    assert t.confidence == 0.95


def test_transcribe_reference_clip_raises_on_http_error(tmp_path):
    from dose_r.references.audio_grounded import transcribe_reference_clip
    from dose_r.references.reference_clips import ReferenceClip

    wav = tmp_path / "abilify.wav"
    _make_wav(wav)
    clip = ReferenceClip("Abilify", "brand", "drugs.com", wav, "wav", 16000, 0.5)

    session = _ScriptedSTTSession(_FakeSTTResponse(400, text="bad request"))
    with pytest.raises(RuntimeError):
        transcribe_reference_clip(clip, {}, session=session)


def test_score_against_reference_when_reference_itself_unrecognized():
    from dose_r.references.audio_grounded import ReferenceTranscript, score_against_reference

    unrecognized_ref = ReferenceTranscript("Retatrutide", "drugs.com", None, None,
                                           asr_recognizable=False, recognizable_score=None)
    result = score_against_reference(unrecognized_ref, "some synth transcript")
    assert result["scoreable"] is False
    assert result["reference_asr_recognizable"] is False


def test_score_against_reference_compares_to_reference_not_spelling():
    from dose_r.references.audio_grounded import ReferenceTranscript, score_against_reference

    # The reference itself was misheard as "a lift trick" -- a synth clip
    # recognized the SAME way should score as a match against the reference,
    # even though neither string matches the drug's spelling.
    ref = ReferenceTranscript("Alyftrek", "drugs.com", "a lift trick", 0.91,
                              asr_recognizable=True, recognizable_score=4.74)
    result = score_against_reference(ref, "a lift trick")
    assert result["scoreable"] is True
    assert result["passed_vs_reference"] is True
    assert result["reference_transcript"] == "a lift trick"


def test_score_against_reference_flags_genuine_mismatch():
    from dose_r.references.audio_grounded import ReferenceTranscript, score_against_reference

    ref = ReferenceTranscript("Abilify", "drugs.com", "Abilify", 0.95,
                              asr_recognizable=True, recognizable_score=5.0)
    result = score_against_reference(ref, "unrelated garbage")
    assert result["passed_vs_reference"] is False


# --- phoneme_scorer: pure PER + score-mapping, no model loading -----------

from dose_r.scoring.phoneme_scorer import (  # noqa: E402
    phoneme_distance, score_phoneme_match,
)


def test_phoneme_distance_identical_sequences_is_zero():
    assert phoneme_distance("ɐ b ɪ l ʌ f aɪ", "ɐ b ɪ l ʌ f aɪ") == 0.0


def test_phoneme_distance_completely_different_is_one():
    # No tokens shared between the two sequences at all, same length -> every
    # position must be substituted, so distance equals the full length.
    dist = phoneme_distance("t oʊ f ə s aɪ t ɪ n ɪ b", "k æ d ɡ h aʊ w ɔː v m z")
    assert dist == 1.0


def test_phoneme_distance_partial_overlap_is_between_zero_and_one():
    # One vowel swapped ("ʌ" -> "ɐ") out of seven tokens.
    dist = phoneme_distance("ɐ b ɪ l ʌ f aɪ", "ɐ b ɪ l ɐ f aɪ")
    assert 0.0 < dist < 1.0
    assert dist == pytest.approx(1 / 7)


def test_phoneme_distance_normalizes_by_the_longer_sequence():
    # Recognized has one extra inserted phoneme relative to expected (5 vs 6
    # tokens) -- one edit, normalized by the longer (6-token) sequence.
    dist = phoneme_distance("æ d v ɪ l", "æ d v ɪ l z")
    assert dist == pytest.approx(1 / 6)


def test_phoneme_distance_both_empty_is_zero_not_undefined():
    assert phoneme_distance("", "") == 0.0


def test_phoneme_distance_one_empty_is_total_miss():
    assert phoneme_distance("ɐ b ɪ l ʌ f aɪ", "") == 1.0
    assert phoneme_distance("", "ɐ b ɪ l ʌ f aɪ") == 1.0


def test_phoneme_distance_tokenizes_on_whitespace_not_characters():
    # "aɪ" is a single diphthong token; splitting on characters would score
    # this as a mismatch even though the phoneme sequences are identical.
    assert phoneme_distance("f aɪ", "f aɪ") == 0.0


def test_score_phoneme_match_identical_scores_five():
    score, components = score_phoneme_match("ɐ b ɪ l ʌ f aɪ", "ɐ b ɪ l ʌ f aɪ")
    assert score == 5.0
    assert components["phoneme_error_rate"] == 0.0


def test_score_phoneme_match_completely_different_scores_low():
    score, _ = score_phoneme_match("t oʊ f ə s aɪ t ɪ n ɪ b", "k æ d ɡ h aʊ w ɔː v m z")
    assert score == 0.0


def test_score_phoneme_match_partial_overlap_lands_in_between():
    # One vowel of seven wrong: neither a perfect 5.0 nor a floor 0.0.
    score, _ = score_phoneme_match("ɐ b ɪ l ʌ f aɪ", "ɐ b ɪ l ɐ f aɪ")
    assert 0.0 < score < 5.0


def test_score_phoneme_match_never_exceeds_scale():
    score, _ = score_phoneme_match("x", "x")
    assert score <= 5.0


def test_score_phoneme_match_is_monotonic_in_distance():
    # More edits -> strictly lower score, never a reversal.
    good, _ = score_phoneme_match("ɐ b ɪ l ʌ f aɪ", "ɐ b ɪ l ɐ f aɪ")       # 1/7 wrong
    bad, _ = score_phoneme_match("ɐ b ɪ l ʌ f aɪ", "z z z z ʌ f aɪ")        # 4/7 wrong
    assert good > bad


def test_score_phoneme_match_high_per_fails_low_per_passes():
    """The whole point of this scorer: a badly-mangled rendering must fail and
    a near-perfect one must pass, using the project's shared PASS_THRESHOLD."""
    close_score, _ = score_phoneme_match("t oʊ f ə s aɪ t ɪ n ɪ b", "t oʊ f ə s aɪ t ɪ n ɪ b")
    mangled_score, _ = score_phoneme_match("t oʊ f ə s aɪ t ɪ n ɪ b", "t oʊ f uː s ɪ d n iː")
    assert close_score >= PASS_THRESHOLD
    assert mangled_score < PASS_THRESHOLD
    assert close_score > mangled_score


# --- audio-LLM panel scorer --------------------------------------------------
class _FakeVertexResponse:
    def __init__(self, status_code: int, text: str = "", json_data=None):
        self.status_code = status_code
        self.text = text
        self._json = json_data or {}

    def json(self):
        return self._json


def _vertex_ok(score, heard="Abilify", reason="clear", fence=False,
              prompt_tokens=221, candidates_tokens=8):
    body = f'{{"heard":"{heard}","score":{score},"reason":"{reason}"}}'
    text = f"```json\n{body}\n```" if fence else body
    return _FakeVertexResponse(200, json_data={
        "candidates": [{"content": {"parts": [{"text": text}]}}],
        "usageMetadata": {"promptTokenCount": prompt_tokens,
                          "candidatesTokenCount": candidates_tokens},
    })


class _ScriptedVertexSession:
    """One scripted response per model, keyed by call order across models
    (the scorer calls judge_models in order for a single item)."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _mock_synth_result(item_id="abilify"):
    from dose_r.adapters.base import SynthesisResult
    return SynthesisResult(system_id="gtts-standard-c", item_id=item_id, ok=True,
                           audio=b"RIFF" + b"\x00" * 40, sample_rate_hz=24000)


@pytest.fixture(autouse=True)
def _no_real_auth_llm_panel(monkeypatch):
    monkeypatch.setattr("dose_r.scoring.llm_panel.auth.auth_headers", lambda: {})


@pytest.fixture(autouse=True)
def _no_real_sleep_llm_panel(monkeypatch):
    monkeypatch.setattr("dose_r.scoring.llm_panel.time.sleep", lambda s: None)


def test_llm_panel_single_judge_mode_scores_on_one_answer():
    """The bug this guards: a hardcoded '>= 2 judges' threshold would make a
    1-judge panel always unscoreable, even when its only judge succeeds."""
    from dose_r.scoring.llm_panel import AudioLLMPanelScorer

    session = _ScriptedVertexSession([_vertex_ok(5)])
    scorer = AudioLLMPanelScorer(judge_models=("gemini-2.5-flash-lite",), session=session)
    item = dataset.load_items()[0]

    result = scorer.score(item, _mock_synth_result(item.item_id))

    assert result.scoreable is True
    assert result.score == 5.0
    assert session.calls == 1
    assert "one judge is configured" in result.notes


def test_llm_panel_full_panel_takes_median_of_three():
    from dose_r.scoring.llm_panel import AudioLLMPanelScorer, JUDGE_MODELS

    session = _ScriptedVertexSession([_vertex_ok(5), _vertex_ok(3), _vertex_ok(4)])
    scorer = AudioLLMPanelScorer(judge_models=JUDGE_MODELS, session=session)
    item = dataset.load_items()[0]

    result = scorer.score(item, _mock_synth_result(item.item_id))

    assert result.score == 4.0  # median of 5, 3, 4
    assert result.components["spread"] == 2.0
    assert session.calls == 3


def test_llm_panel_full_panel_tolerates_one_failure():
    from dose_r.scoring.llm_panel import AudioLLMPanelScorer, JUDGE_MODELS

    session = _ScriptedVertexSession([
        _vertex_ok(5), _FakeVertexResponse(500, text="server error"), _vertex_ok(4),
    ])
    scorer = AudioLLMPanelScorer(judge_models=JUDGE_MODELS, session=session,
                                 max_attempts=1)
    item = dataset.load_items()[0]

    result = scorer.score(item, _mock_synth_result(item.item_id))

    assert result.scoreable is True
    assert result.score == 4.5  # median of the 2 that answered: 5, 4
    assert "1 judge(s) failed" in result.notes


def test_llm_panel_full_panel_two_failures_is_unscoreable():
    from dose_r.scoring.llm_panel import AudioLLMPanelScorer, JUDGE_MODELS

    session = _ScriptedVertexSession([
        _vertex_ok(5),
        _FakeVertexResponse(500, text="err1"),
        _FakeVertexResponse(500, text="err2"),
    ])
    scorer = AudioLLMPanelScorer(judge_models=JUDGE_MODELS, session=session,
                                 max_attempts=1)
    item = dataset.load_items()[0]

    result = scorer.score(item, _mock_synth_result(item.item_id))

    assert result.scoreable is False
    assert result.score is None


def test_llm_panel_single_judge_failure_is_unscoreable_not_zero():
    from dose_r.scoring.llm_panel import AudioLLMPanelScorer

    session = _ScriptedVertexSession([_FakeVertexResponse(500, text="boom")])
    scorer = AudioLLMPanelScorer(judge_models=("gemini-2.5-flash-lite",),
                                 session=session, max_attempts=1)
    item = dataset.load_items()[0]

    result = scorer.score(item, _mock_synth_result(item.item_id))

    assert result.scoreable is False
    assert result.score is None  # never silently 0


def test_llm_panel_parses_fenced_json():
    from dose_r.scoring.llm_panel import AudioLLMPanelScorer

    session = _ScriptedVertexSession([_vertex_ok(5, fence=True)])
    scorer = AudioLLMPanelScorer(judge_models=("gemini-2.5-flash-lite",), session=session)
    item = dataset.load_items()[0]

    result = scorer.score(item, _mock_synth_result(item.item_id))

    assert result.score == 5.0


def test_llm_panel_discrimination_mismatched_audio_scores_low():
    """Mirrors the manual check run before this scorer was built: mismatched
    audio must not score as a pass."""
    from dose_r.scoring.llm_panel import AudioLLMPanelScorer

    session = _ScriptedVertexSession([
        _vertex_ok(0, heard="abilify", reason="clearly says abilify, not the expected drug"),
    ])
    scorer = AudioLLMPanelScorer(judge_models=("gemini-2.5-flash-lite",), session=session)
    item = dataset.load_items()[0]

    result = scorer.score(item, _mock_synth_result(item.item_id))

    assert result.score == 0.0
    assert result.passed is False


def test_llm_panel_synthesis_failure_scores_zero_not_none():
    from dose_r.scoring.llm_panel import AudioLLMPanelScorer
    from dose_r.adapters.base import SynthesisResult

    scorer = AudioLLMPanelScorer(judge_models=("gemini-2.5-flash-lite",),
                                 session=_ScriptedVertexSession([Exception("must not be called")]))
    item = dataset.load_items()[0]
    failed = SynthesisResult(system_id="x", item_id=item.item_id, ok=False, error="boom")

    result = scorer.score(item, failed)

    assert result.score == 0.0 and result.scoreable is True


def test_llm_panel_tracks_token_usage_per_model():
    from dose_r.scoring.llm_panel import AudioLLMPanelScorer

    session = _ScriptedVertexSession([_vertex_ok(5, prompt_tokens=221, candidates_tokens=8)])
    scorer = AudioLLMPanelScorer(judge_models=("gemini-2.5-flash-lite",), session=session)
    item = dataset.load_items()[0]
    scorer.score(item, _mock_synth_result(item.item_id))

    usage = scorer.usage_summary()["gemini-2.5-flash-lite"]
    assert usage == {"prompt_tokens": 221, "candidates_tokens": 8, "calls": 1}


# --- speech similarity (Path 2: audio-to-audio) -----------------------------
def test_speech_bertscore_identical_sequences_scores_one():
    import numpy as np
    from dose_r.scoring.speech_similarity import speech_bertscore

    feats = np.random.RandomState(0).randn(10, 8)
    result = speech_bertscore(feats, feats)
    assert result["f1"] == pytest.approx(1.0, abs=1e-6)


def test_speech_bertscore_empty_sequence_scores_zero():
    import numpy as np
    from dose_r.scoring.speech_similarity import speech_bertscore

    empty = np.zeros((0, 8))
    feats = np.random.RandomState(0).randn(5, 8)
    result = speech_bertscore(empty, feats)
    assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_speech_bertscore_orthogonal_sequences_score_low():
    import numpy as np
    from dose_r.scoring.speech_similarity import speech_bertscore

    a = np.eye(4, 8)       # 4 orthogonal unit vectors
    b = -np.eye(4, 8)      # their exact opposites
    result = speech_bertscore(a, b)
    assert result["f1"] < 0.1


def test_score_speech_similarity_maps_f1_not_precision_to_0_5_scale():
    """The paper (Saeki et al.) uses precision alone; this project
    deliberately diverges (module docstring, "Correction history" step 3) --
    precision was tested and found weakly sensitive to a candidate that
    truncates part of the drug name, which matters more here than the
    paper's own use case. Construct a case where precision and F1 genuinely
    differ (a short candidate fully covered by a much longer reference: high
    precision, low recall -- e.g. mimicking a truncated candidate compared
    against a full-length reference) and confirm the score tracks F1, not
    precision alone."""
    import numpy as np
    from dose_r.scoring.speech_similarity import score_speech_similarity

    rng = np.random.RandomState(1)
    candidate = rng.randn(3, 8)          # short candidate (e.g. truncated)
    reference = np.vstack([candidate, rng.randn(20, 8)])  # candidate + lots extra

    score, components = score_speech_similarity(candidate, reference)

    assert components["precision"] == pytest.approx(1.0, abs=1e-6)  # every candidate frame matches perfectly
    assert components["recall"] < 0.5    # most reference frames have no good match
    assert components["f1"] < 0.7         # F1 is pulled down by the low recall
    assert score == pytest.approx(components["f1"] * 5.0, abs=1e-3)  # score follows F1, not precision
    assert score < 4.0  # a precision-only score would have been a false 5.0 "pass" here


def test_score_speech_similarity_identical_sequences_scores_five():
    import numpy as np
    from dose_r.scoring.speech_similarity import score_speech_similarity

    feats = np.random.RandomState(1).randn(6, 8)
    score, components = score_speech_similarity(feats, feats)
    assert score == pytest.approx(5.0, abs=1e-3)
    assert components["precision"] == pytest.approx(1.0, abs=1e-6)


def test_cosine_similarity_matrix_shape_and_range():
    import numpy as np
    from dose_r.scoring.speech_similarity import _cosine_similarity_matrix

    a = np.random.RandomState(2).randn(5, 8)
    b = np.random.RandomState(3).randn(7, 8)
    sim = _cosine_similarity_matrix(a, b)
    assert sim.shape == (5, 7)
    assert sim.max() <= 1.0 + 1e-6 and sim.min() >= -1.0 - 1e-6


# --- audio_span (drug-name span extraction) ---------------------------------
def test_extract_words_with_timing_parses_seconds_strings():
    from dose_r.audio_span import _extract_words_with_timing

    response = {"results": [{"alternatives": [{"words": [
        {"word": "Take", "startTime": "0s", "endTime": "0.3s"},
        {"word": "Advil", "startTime": "0.3s", "endTime": "0.9s"},
    ]}]}]}
    words = _extract_words_with_timing(response)
    assert words == [
        {"word": "Take", "start_s": 0.0, "end_s": 0.3},
        {"word": "Advil", "start_s": 0.3, "end_s": 0.9},
    ]


def test_extract_words_with_timing_empty_on_no_results():
    from dose_r.audio_span import _extract_words_with_timing

    assert _extract_words_with_timing({"results": []}) == []


def test_locate_span_indices_returns_none_when_drug_absent():
    from dose_r.audio_span import _locate_span_indices

    words = [{"word": "Take", "start_s": 0.0, "end_s": 0.3}]
    result = _locate_span_indices("Take Advil now.", "Advil", words)
    assert result is None


def test_locate_span_indices_finds_matching_word():
    from dose_r.audio_span import _locate_span_indices

    words = [
        {"word": "Take", "start_s": 0.0, "end_s": 0.3},
        {"word": "Advil", "start_s": 0.3, "end_s": 0.9},
        {"word": "now", "start_s": 0.9, "end_s": 1.1},
    ]
    result = _locate_span_indices("Take Advil now.", "Advil", words)
    assert result == (1, 1)


# --- regression tests for the two confirmed span-contamination bugs --------
# Both found by listening to extracted spans on real Gemini Flash TTS audio,
# not by inspection -- see the module docstring's "Two real, confirmed
# contamination bugs" section for the full story and the real STT responses
# these fixtures are drawn from.

def test_clamp_padding_does_not_bleed_into_previous_word():
    """The confirmed "esomeprazole" bug: Cloud STT garbled the drug name into
    "a"/"summer" (0.8-1.6s); the immediately preceding word "start" ends at
    exactly 0.8s, so a naive fixed 0.12s pad pulled the clip start back to
    0.68s -- INSIDE "start"'s own span. Padding must stop at the midpoint of
    the gap to the neighbor, and here the gap is zero (0.8s to 0.8s)."""
    from dose_r.audio_span import _clamp_padding

    words = [
        {"word": "start", "start_s": 0.5, "end_s": 0.8},
        {"word": "a", "start_s": 0.8, "end_s": 1.4},
        {"word": "summer", "start_s": 1.4, "end_s": 1.6},
        {"word": "once", "start_s": 1.6, "end_s": 2.6},
    ]
    start_s, end_s = _clamp_padding(words, i=1, j=2)
    assert start_s >= 0.8   # never reaches back into "start"'s span (ends at 0.8)
    assert end_s <= 1.6 + 0.12  # end-side gap is generous, full pad is fine there


def test_clamp_padding_respects_a_narrow_gap_on_both_sides():
    """A tighter, symmetric case: words with only 0.04s of silence on each
    side should clamp padding to half that gap (0.02s), not the full 0.12s."""
    from dose_r.audio_span import _clamp_padding

    words = [
        {"word": "the", "start_s": 0.0, "end_s": 0.46},
        {"word": "Advil", "start_s": 0.50, "end_s": 0.90},
        {"word": "now", "start_s": 0.94, "end_s": 1.20},
    ]
    start_s, end_s = _clamp_padding(words, i=1, j=1)
    assert start_s == pytest.approx(0.48, abs=1e-6)  # 0.50 - min(0.12, 0.04/2)
    assert end_s == pytest.approx(0.92, abs=1e-6)     # 0.90 + min(0.12, 0.04/2)


def test_duration_plausibility_rejects_corrupted_stt_timestamp():
    """The confirmed "talquetamab" bug: Cloud STT collapsed six real spoken
    words into one garbled token with a reported span of 0.6s to 5.7s (5.1s)
    while the sentence's other words average well under 1s each. No padding
    fix can catch this -- it is STT's own timestamp that is corrupted."""
    from dose_r.audio_span import _duration_is_plausible

    normal_paced_words = [
        {"word": "since", "start_s": 0.1, "end_s": 0.6},
        {"word": "it", "start_s": 5.7, "end_s": 6.2},
        {"word": "offers", "start_s": 6.2, "end_s": 6.6},
        {"word": "a", "start_s": 6.6, "end_s": 7.1},
    ]
    assert _duration_is_plausible(normal_paced_words, span_duration_s=5.1) is False
    assert _duration_is_plausible(normal_paced_words, span_duration_s=0.6) is True


def test_verify_by_retranscription_accepts_garbled_or_empty_transcript():
    """A coined drug name being unrecognizable to ASR is EXPECTED and must
    not itself fail verification -- only too MANY recognized words should."""
    from dose_r.audio_span import _verify_by_retranscription

    empty = _FakeResponse(200, json_data={"results": []})
    session = _ScriptedSession([empty])
    assert _verify_by_retranscription(b"fake", "esomeprazole", 24000, session, 30) is True


def test_verify_by_retranscription_rejects_too_many_words():
    """Mirrors the "start esomeprazole once" contamination shape: a span that
    re-transcribes to several unrelated real words should fail verification
    for a single-word drug name."""
    from dose_r.audio_span import _verify_by_retranscription

    contaminated = _FakeResponse(200, json_data={"results": [
        {"alternatives": [{"transcript": "start a summer once"}]},
    ]})
    session = _ScriptedSession([contaminated])
    assert _verify_by_retranscription(b"fake", "esomeprazole", 24000, session, 30) is False


def test_verify_by_retranscription_transport_failure_does_not_reject():
    """A hiccup on the verification call itself must not invalidate an
    otherwise-good span -- fail open, not closed, on transport errors."""
    from dose_r.audio_span import _verify_by_retranscription

    session = _ScriptedSession([_FakeResponse(500, text="server error")])
    assert _verify_by_retranscription(b"fake", "esomeprazole", 24000, session, 30) is True


def test_speech_similarity_scorer_no_reference_clip_is_unscoreable():
    from dose_r.scoring.speech_similarity import SpeechSimilarityScorer

    item = dataset.DoseItem(drug="Zzznotreal", name_type="generic", sentence="Take Zzznotreal now.")
    scorer = SpeechSimilarityScorer(reference_clips={})  # empty -- nothing available
    result = scorer.score(item, _mock_synth_result(item.item_id))

    assert result.scoreable is False
    assert "no reference clip" in result.error


def test_speech_similarity_scorer_synthesis_failure_scores_zero_not_none():
    from dose_r.adapters.base import SynthesisResult
    from dose_r.scoring.speech_similarity import SpeechSimilarityScorer

    item = dataset.load_items()[0]
    scorer = SpeechSimilarityScorer(reference_clips={})
    failed = SynthesisResult(system_id="x", item_id=item.item_id, ok=False, error="boom")

    result = scorer.score(item, failed)

    assert result.score == 0.0 and result.scoreable is True


def test_speech_similarity_scorer_missing_span_is_unscoreable(monkeypatch):
    from dose_r.references.reference_clips import ReferenceClip
    from dose_r.scoring.speech_similarity import SpeechSimilarityScorer

    item = dataset.load_items()[0]
    clip = ReferenceClip(item.drug, item.name_type, "drugs.com", Path("/x.wav"), "wav", 16000, 1.0)
    scorer = SpeechSimilarityScorer(reference_clips={item.drug: clip})

    monkeypatch.setattr("dose_r.scoring.speech_similarity.extract_drug_span",
                        lambda *a, **k: None)  # span not locatable

    result = scorer.score(item, _mock_synth_result(item.item_id))

    assert result.scoreable is False
    assert "could not locate" in result.error


def test_speech_similarity_scorer_end_to_end_with_mocked_embeddings(monkeypatch):
    import numpy as np
    from dose_r.references.reference_clips import ReferenceClip
    from dose_r.scoring.speech_similarity import SpeechSimilarityScorer

    item = dataset.load_items()[0]
    clip = ReferenceClip(item.drug, item.name_type, "merriam-webster", Path("/x.mp3"),
                         "mp3", 16000, 1.5, respelling="uh-BIL-uh-fy")
    scorer = SpeechSimilarityScorer(reference_clips={item.drug: clip})

    monkeypatch.setattr("dose_r.scoring.speech_similarity.extract_drug_span",
                        lambda *a, **k: b"fake-wav-bytes")
    identical = np.random.RandomState(0).randn(10, 8)
    monkeypatch.setattr("dose_r.scoring.speech_similarity.extract_frame_embeddings",
                        lambda source: identical)

    result = scorer.score(item, _mock_synth_result(item.item_id))

    assert result.scoreable is True
    assert result.score == pytest.approx(5.0, abs=1e-2)
    assert result.metadata["reference_source"] == "merriam-webster"
    assert result.metadata["reference_respelling"] == "uh-BIL-uh-fy"
