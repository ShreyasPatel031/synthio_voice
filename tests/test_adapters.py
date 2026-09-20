"""Adapter contract, cost accounting, latency accounting, retry policy."""

from __future__ import annotations

import wave
import io

import pytest

from dose_r.adapters import (
    PermanentError,
    Pricing,
    RetryPolicy,
    SynthesisRequest,
    TransientError,
    Usage,
    build_adapter,
    load_system,
    load_systems,
)
from dose_r.adapters.base import SCHEMA_VERSION
from dose_r.adapters.config import system_from_dict

SENTENCE = (
    "We plan to start Abilify at ten milligrams daily to help manage the "
    "patient's schizophrenia symptoms effectively."
)


def mock_system(**options):
    return system_from_dict(
        "test-mock",
        {
            "backend": "mock",
            "model": "mock-v1",
            "voice": "mock-neutral",
            "pricing": {
                "id": "test/priced",
                "as_of": "2026-09-17",
                "per_request_usd": 0.001,
                "per_million_characters_usd": 10.0,
                "per_million_input_tokens_usd": 0.5,
                "per_million_output_tokens_usd": 10.0,
            },
            "retry": {"max_attempts": 3, "initial_backoff_s": 0.001, "jitter": 0.0},
            "options": options,
        },
    )


def request(item_id="dose-000", text=SENTENCE):
    return SynthesisRequest(item_id=item_id, text=text, item={"name": "Abilify", "spans": [[17, 24]]})


def test_mock_produces_playable_wav():
    record = build_adapter(mock_system()).synthesize(request())
    assert record.ok
    with wave.open(io.BytesIO(record.audio), "rb") as w:
        assert w.getframerate() == 24000
        assert w.getnchannels() == 1
        assert w.getnframes() > 0
    assert record.audio_duration_s == pytest.approx(len(SENTENCE) / 14.5, rel=0.01)


def test_mock_is_deterministic_and_item_specific():
    a = build_adapter(mock_system()).synthesize(request())
    b = build_adapter(mock_system()).synthesize(request())
    c = build_adapter(mock_system()).synthesize(request(item_id="dose-001"))
    assert a.audio_sha256 == b.audio_sha256
    assert a.audio_sha256 != c.audio_sha256


def test_record_schema_is_complete():
    d = build_adapter(mock_system()).synthesize(request(), run_id="r1").to_dict()
    assert d["schema_version"] == SCHEMA_VERSION
    for key in (
        "run_id", "item_id", "system_id", "backend", "model", "voice", "status",
        "text_sha256", "text_chars", "item", "audio_path", "audio_sha256",
        "audio_duration_s", "audio_format", "latency_ms", "ttfa_ms", "ttfa_source",
        "wall_clock_ms", "attempt_count", "usage", "pricing_id", "cost_usd",
        "cost_breakdown", "error", "score",
    ):
        assert key in d, key
    assert d["item"]["spans"] == [[17, 24]]
    assert "audio" not in d  # bytes never land in the manifest


def test_cost_is_usage_times_price():
    record = build_adapter(mock_system()).synthesize(request())
    usage = record.usage
    expected = (
        0.001
        + usage.characters * 10.0 / 1e6
        + usage.input_tokens * 0.5 / 1e6
        + usage.output_tokens * 10.0 / 1e6
    )
    assert record.cost_usd == pytest.approx(expected, rel=1e-9)
    assert sum(record.cost_breakdown.values()) == pytest.approx(record.cost_usd, rel=1e-9)
    assert set(record.cost_breakdown) == {"request", "characters", "input_tokens", "output_tokens"}


def test_cost_recomputable_from_stored_usage():
    """A price correction must be a recomputation over the manifest."""
    record = build_adapter(mock_system()).synthesize(request())
    corrected = Pricing(id="x", as_of="2026-10-01", per_million_characters_usd=20.0)
    stored = Usage(**{k: v for k, v in record.usage.to_dict().items()})
    assert corrected.cost_usd(stored) == pytest.approx(stored.characters * 20.0 / 1e6)


def test_free_pricing_yields_zero_cost_and_empty_breakdown():
    record = build_adapter(load_system("mock")).synthesize(request())
    assert record.cost_usd == 0.0
    assert record.cost_breakdown == {}


def test_ttfa_separated_from_total_latency():
    system = mock_system(base_latency_s=0.2, ttfa_fraction=0.25)
    record = build_adapter(system).synthesize(request())
    assert record.ttfa_source == "stream_first_chunk"
    assert record.ttfa_ms == pytest.approx(50, abs=30)
    assert record.latency_ms >= record.ttfa_ms
    assert record.latency_ms == pytest.approx(200, abs=80)


def test_transient_failures_are_retried_and_counted():
    system = mock_system(transient_failures_per_item=2)
    record = build_adapter(system).synthesize(request())
    assert record.ok
    assert len(record.attempts) == 3
    assert [a.error_class for a in record.attempts] == ["TransientError", "TransientError", None]
    assert record.wall_clock_ms > record.latency_ms  # backoff is in wall clock, not latency


def test_retries_exhausted_produces_error_record_with_no_cost():
    system = mock_system(transient_failures_per_item=5)
    record = build_adapter(system).synthesize(request())
    assert not record.ok
    assert record.status == "error"
    assert record.error_class == "TransientError"
    assert len(record.attempts) == 3
    assert record.cost_usd == 0.0
    assert record.audio is None
    assert record.to_dict()["latency_ms"] is None


def test_permanent_failure_is_not_retried():
    system = mock_system(permanent_failure_ids=["dose-000"])
    record = build_adapter(system).synthesize(request())
    assert not record.ok
    assert record.error_class == "PermanentError"
    assert len(record.attempts) == 1


def test_backoff_grows_and_is_capped():
    import random

    policy = RetryPolicy(initial_backoff_s=1.0, multiplier=2.0, max_backoff_s=5.0, jitter=0.0)
    rng = random.Random(0)
    assert [policy.backoff_s(n, rng) for n in (1, 2, 3, 4)] == [1.0, 2.0, 4.0, 5.0]


def test_every_configured_system_parses():
    systems = load_systems()
    assert systems["mock"].backend == "mock"
    for system in systems.values():
        assert system.pricing.as_of
        assert system.retry.max_attempts >= 1
        if system.status == "ready":
            assert system.pricing.verified or system.backend == "gemini_vertex"


def test_planned_system_fails_loudly_rather_than_silently():
    with pytest.raises(KeyError, match="not implemented"):
        build_adapter(load_system("cartesia-sonic"))


def test_gemini_adapter_builds_without_network():
    adapter = build_adapter(load_system("gemini-flash-tts"))
    assert adapter.backend == "gemini_vertex"
    assert adapter.supports_streaming
    assert adapter.config.model == "gemini-2.5-flash-preview-tts"
    adapter.close()
