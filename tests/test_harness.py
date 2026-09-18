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
