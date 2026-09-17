"""Audio-LLM judge panel: interface and deterministic mocks.

DOSE describes its judge as "automated audio comparison, several independent
judgments combined into one result, no humans in the loop". This is DOSE-R's
reading of that: three independent audio-capable judges each return an integer
0-5, and the panel reports the MEDIAN.

Median, not mean, for two reasons. It is robust to a single judge misfiring,
which is the dominant failure mode of LLM panels; and it keeps the panel output
on the same integer 0-5 ordinal scale as DOSE's own, whereas a mean invents
intermediate values that the rubric does not define.

No live model is called from this module. The audio-capable endpoint is
Workstream 1c's deliverable; `AudioJudge` is the seam it drops into. The mocks
here are deterministic so that agreement reporting, the combination layer and
the stress test can all be exercised end to end today.
"""

from __future__ import annotations

import hashlib
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

PANEL_SIZE = 3
PASS_SCORE = 4

RUBRIC = """\
You are judging whether a text-to-speech system pronounced a drug name correctly.

You will hear one sentence. Attend ONLY to the drug name "{ingredient}" as it
occurs in that sentence; ignore the rest of the sentence, the voice, the
recording quality and the speaking rate.

Reference pronunciation (IPA): {ipa}

Score 0-5:
  5  Indistinguishable from the reference, or an accepted regional variant.
  4  Clearly the right drug name; at most one minor segmental or stress
     difference that would not cause a listener to hear a different word.
  3  Recognisable but mispronounced; a listener would notice the error.
  2  Substantially wrong; a listener would have to infer the name from context.
  1  Barely related to the reference.
  0  Not the drug name, or unintelligible.

Reply with the integer only."""


@dataclass(frozen=True)
class JudgeRequest:
    item_id: str
    ingredient: str
    sentence: str
    span: tuple[int, int]
    audio_uri: str
    reference_ipa: str

    def prompt(self) -> str:
        return RUBRIC.format(ingredient=self.ingredient, ipa=self.reference_ipa)


@dataclass(frozen=True)
class JudgeVerdict:
    judge_id: str
    score: int
    rationale: str = ""
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PanelVerdict:
    item_id: str
    ingredient: str
    score: int
    verdicts: tuple[JudgeVerdict, ...]

    @property
    def passed(self) -> bool:
        return self.score >= PASS_SCORE

    @property
    def spread(self) -> int:
        """Max minus min across judges. A large spread means the panel itself is
        unsure, which is information the aggregate score throws away."""
        scores = [v.score for v in self.verdicts]
        return max(scores) - min(scores)

    @property
    def unanimous(self) -> bool:
        return self.spread == 0

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "ingredient": self.ingredient,
            "score": self.score,
            "passed": self.passed,
            "spread": self.spread,
            "unanimous": self.unanimous,
            "judges": [
                {"judge_id": v.judge_id, "score": v.score, "rationale": v.rationale}
                for v in self.verdicts
            ],
        }


class AudioJudge(ABC):
    """One audio-capable judge. Workstream 1c implements this against a real
    endpoint; nothing else in the judge package may know how it is served."""

    judge_id: str

    @abstractmethod
    def score(self, request: JudgeRequest) -> JudgeVerdict:
        ...


class NotWiredJudge(AudioJudge):
    """Placeholder for the real client. Fails loudly rather than returning a
    number nobody should trust."""

    def __init__(self, judge_id: str = "unwired"):
        self.judge_id = judge_id

    def score(self, request: JudgeRequest) -> JudgeVerdict:
        raise NotImplementedError(
            "No audio-capable model endpoint is wired up. Workstream 1c owns the "
            "client; until it lands, use a mock judge explicitly."
        )


def _digest(*parts: str) -> int:
    return int.from_bytes(
        hashlib.blake2b("\x1f".join(parts).encode(), digest_size=8).digest(), "big"
    )


class HashMockJudge(AudioJudge):
    """Deterministic pseudo-random scores. Exercises the interface only; its
    scores carry no signal and must never appear in a result table."""

    def __init__(self, judge_id: str, seed: str = ""):
        self.judge_id = judge_id
        self.seed = seed

    def score(self, request: JudgeRequest) -> JudgeVerdict:
        value = _digest(self.seed, self.judge_id, request.item_id) % 6
        return JudgeVerdict(self.judge_id, value, "hash mock")


class PhoneticProxyMockJudge(AudioJudge):
    """A mock that agrees with the phonetic scorer, plus per-judge disagreement.

    Stands in for a real audio judge well enough to develop and test agreement
    reporting, divergence flagging and the stress test: it tracks ground truth
    the way a competent judge would, while disagreeing item-by-item and
    judge-by-judge the way a real panel does. `jitter_rate` is the share of
    items on which this judge departs from the phonetic score by one point.
    """

    def __init__(
        self,
        judge_id: str,
        phonetic_scores: dict[str, int],
        jitter_rate: float = 0.25,
        seed: str = "dose-r",
    ):
        self.judge_id = judge_id
        self.phonetic_scores = phonetic_scores
        self.jitter_rate = jitter_rate
        self.seed = seed

    def score(self, request: JudgeRequest) -> JudgeVerdict:
        base = self.phonetic_scores[request.item_id]
        draw = _digest(self.seed, self.judge_id, request.item_id)
        if (draw % 1000) / 1000.0 < self.jitter_rate:
            base += 1 if (draw >> 10) % 2 else -1
        return JudgeVerdict(self.judge_id, max(0, min(5, base)), "phonetic-proxy mock")


class JudgePanel:
    """Runs N independent judges and takes the median of their 0-5 scores."""

    def __init__(self, judges: list[AudioJudge]):
        if len(judges) < 2 or len(judges) % 2 == 0:
            raise ValueError(
                f"panel needs an odd number of judges >= 3, got {len(judges)}; "
                "an even panel has no unambiguous median"
            )
        if len({j.judge_id for j in judges}) != len(judges):
            raise ValueError("judge_id must be unique within a panel")
        self.judges = judges

    def judge(self, request: JudgeRequest) -> PanelVerdict:
        verdicts = tuple(j.score(request) for j in self.judges)
        for v in verdicts:
            if not 0 <= v.score <= 5:
                raise ValueError(f"{v.judge_id} returned {v.score}, outside 0-5")
        median = int(statistics.median(sorted(v.score for v in verdicts)))
        return PanelVerdict(request.item_id, request.ingredient, median, verdicts)

    def judge_all(self, requests: list[JudgeRequest]) -> list[PanelVerdict]:
        return [self.judge(r) for r in requests]


def mock_panel(phonetic_scores: dict[str, int], jitter_rate: float = 0.25) -> JudgePanel:
    return JudgePanel(
        [
            PhoneticProxyMockJudge(f"mock-{i}", phonetic_scores, jitter_rate)
            for i in range(PANEL_SIZE)
        ]
    )
