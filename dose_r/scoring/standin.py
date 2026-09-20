"""Stand-in scorer -- harness validation only.

IMPORTANT: this does NOT measure pronunciation. It has no gold reference layer, so
it cannot possibly know whether "retatrutide" was said correctly. It scores
*audio deliverability*: did a well-formed, plausibly-long, non-silent, non-clipped
waveform come back for this text.

Its only jobs are (a) to exercise the runner -> scorer -> report path end to end
before Workstream 1 ships the real judge, and (b) to satisfy calibration step 4 by
cleanly separating known-good from known-bad control audio.

`measures_pronunciation = False` marks every number it produces as NOT
leaderboard-comparable, and the report generator enforces that.
"""

from __future__ import annotations

import io
import wave

from ..adapters.base import SynthesisResult
from ..dataset import DoseItem
from .base import SCALE_MAX, SCALE_MIN, ScoreResult, Scorer

# A natural clinical sentence read aloud runs roughly 55-95ms per character.
# Well outside that band means truncated or padded audio, not a real reading.
_MS_PER_CHAR_LOW, _MS_PER_CHAR_HIGH = 40.0, 130.0


def _probe_wav(audio: bytes) -> dict[str, float]:
    """Duration, mean absolute amplitude, and clipping fraction."""
    with wave.open(io.BytesIO(audio), "rb") as w:
        n_frames, rate, width, channels = (
            w.getnframes(), w.getframerate(), w.getsampwidth(), w.getnchannels()
        )
        raw = w.readframes(n_frames)

    if width != 2:
        raise ValueError(f"expected 16-bit PCM, got {width * 8}-bit")

    import array
    samples = array.array("h")
    samples.frombytes(raw[: len(raw) - (len(raw) % 2)])
    if channels > 1:
        samples = samples[::channels]
    if not samples:
        raise ValueError("no audio samples")

    peak = 32767
    total = len(samples)
    abs_sum = 0
    clipped = 0
    for s in samples:
        a = -s if s < 0 else s
        abs_sum += a
        if a >= peak - 8:
            clipped += 1

    return {
        "duration_ms": (n_frames / rate) * 1000 if rate else 0.0,
        "mean_amplitude": abs_sum / total / peak,
        "clipped_fraction": clipped / total,
        "sample_rate_hz": float(rate),
    }


class StandInScorer(Scorer):
    """Deterministic audio-plausibility proxy. Not a pronunciation judge."""

    measures_pronunciation = False

    @property
    def scorer_id(self) -> str:
        return "standin-audio-plausibility-v1"

    def score(self, item: DoseItem, result: SynthesisResult) -> ScoreResult:
        base = dict(scorer_id=self.scorer_id, item_id=item.item_id,
                    system_id=result.system_id)

        if not result.ok or not result.audio:
            return ScoreResult(**base, score=0.0, scoreable=True,
                               error=result.error or "no audio returned",
                               notes="synthesis failed")

        try:
            probe = _probe_wav(result.audio)
        except Exception as exc:
            return ScoreResult(**base, score=None, scoreable=False,
                               error=f"unreadable audio: {exc}")

        components: dict[str, float] = {}

        # 1. Duration plausibility for the text length.
        ms_per_char = probe["duration_ms"] / max(len(item.sentence), 1)
        if _MS_PER_CHAR_LOW <= ms_per_char <= _MS_PER_CHAR_HIGH:
            components["duration_plausibility"] = 1.0
        else:
            edge = _MS_PER_CHAR_LOW if ms_per_char < _MS_PER_CHAR_LOW else _MS_PER_CHAR_HIGH
            components["duration_plausibility"] = max(
                0.0, 1.0 - abs(ms_per_char - edge) / edge
            )

        # 2. Signal present but not crushed against the rails.
        amp = probe["mean_amplitude"]
        components["level_sanity"] = 1.0 if 0.02 <= amp <= 0.5 else (
            max(0.0, amp / 0.02) if amp < 0.02 else max(0.0, 1.0 - (amp - 0.5) * 2)
        )

        # 3. Not clipped.
        components["clipping"] = max(0.0, 1.0 - probe["clipped_fraction"] * 20)

        weights = {"duration_plausibility": 0.5, "level_sanity": 0.3, "clipping": 0.2}
        combined = sum(components[k] * w for k, w in weights.items())
        score = round(min(max(combined * SCALE_MAX, SCALE_MIN), SCALE_MAX), 3)

        return ScoreResult(
            **base, score=score, components=components,
            notes="audio plausibility proxy; NOT a pronunciation judgement",
            metadata={k: round(v, 4) for k, v in probe.items()} | {
                "ms_per_char": round(ms_per_char, 2)
            },
        )
