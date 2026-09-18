"""Audio-LLM panel scorer.

Designed as a 3-judge panel (median score, see `JUDGE_MODELS` below), but the
judge list is a constructor argument -- `AudioLLMPanelScorer(judge_models=(...))`
-- so a single-judge configuration is a supported, deliberate mode, not a
workaround. The first full run uses gemini-2.5-flash-lite alone: run 3x fewer
judge calls (1096 vs. 3288 for the full 274x4-item corpus) at the cost of
losing the cross-judge disagreement signal `spread` normally provides. Every
score's `notes` field states plainly which of these two modes produced it
(`_panel_note`) rather than always describing the 3-judge design.

Why this exists
----------------
Every pronunciation proxy in this repo before this one (`asr_roundtrip.py`,
`references/audio_grounded.py`) is built on Google Cloud Speech-to-Text, a
word-level ASR system. `docs/REFERENCE_AUDIO_GROUNDING.md` measured directly
that Cloud STT fails to transcribe **44.3% of human reference clips** --
coined drug names said correctly by a real person still come back as
garbage ("tofacitinib" -> "tofu Sydney") because the recognizer's language
model has never seen the word. That is not a TTS-quality signal; it is a
ceiling imposed by the judge, not the system under test.

An audio-LLM does not have that specific failure mode: it is not decoding
through a fixed word lexicon, it is listening to whether the audio's sounds
match the name it was told to expect. Manually verified before building this:
"tofacitinib" scored 5/5 against both human and TTS audio of the same word
said correctly, and 0/5 with a correct stated reason when given mismatched
audio. See the validation-slice results this module was built against for
the fuller picture (step 2 in the task this shipped under).

Mechanism
---------
Three genuinely distinct models (not three samples of one model -- the
whole point of a panel is judges that can fail independently):
gemini-2.5-flash, gemini-2.5-pro, gemini-2.5-flash-lite, all called via
Vertex AI `generateContent` with the synthesized WAV as inline audio and
`temperature=0.0` (determinism from model diversity, not from suppressing
sampling noise on one model). Each judge is asked for a strict 0-5 score in
JSON; `_extract_json` copes with judges that wrap their answer in a
` ```json ` fence (observed from gemini-2.5-pro in testing) or add stray
prose around the object.

The item's score is the **median** of the judges that answered. All three
raw scores are kept in `components` (`judge_<model>`), plus `median` and
`spread` (max-min, a disagreement signal -- a wide spread on an item is
itself worth surfacing, not just averaged away). Each judge's `heard` and
`reason` strings, and every judge's raw token usage, land in `metadata` for
audit and cost reconstruction.

Failure handling
-----------------
A judge that errors (transport failure, non-200, or a response that does
not parse as the expected JSON shape) is recorded and excluded, not
defaulted to a score. The item is `scoreable=False` with an `error` --
never scored 0 -- unless enough of the configured judges still answered
(at most one failure tolerated, scaled to panel size: >=2 of 3 for the
full panel, the single judge itself for a 1-judge run). Scoring an
unanswerable item as 0 would silently conflate "we could not get a
judgement" with "the panel confirmed a mispronunciation", exactly the
failure mode the ASR scorer's docstring and this project's house rules
warn against.

CRITICAL CAVEAT -- self-preference bias (read before trusting any chirp3hd
number this scorer produces)
--------------------------------------------------------------------------
All three judges are Gemini models. One of the four systems under test in
this project, `gtts-chirp3hd-achernar`, is itself a Gemini-family voice
(Cloud TTS's LLM-based Chirp3-HD tier). "Language models rate their own
family's output more favorably" is a well-documented evaluation bias
(self-preference / self-enhancement bias in LLM-as-judge setups). This
scorer has no mechanism to detect or correct for that, and nothing in the
validation slice this module shipped with attempted to measure it (that
would require a non-Gemini judge or non-Gemini-family audio held out as a
control, neither of which was in scope here). Any report built from this
scorer's output MUST flag that a chirp3hd-vs-legacy-tier comparison may be
inflated by judge/system family overlap, not present it as a clean result.

Also unresolved by this module: LLM judges are known to be lenient, and
`docs/FINDING_voice_identity_instability.md`'s finding that Standard/
WaveNet/Neural2 return byte-identical audio on 19-40% of items still
applies here -- this scorer judges whatever audio it is handed, and cannot
tell a genuinely distinct Standard-tier clip from one that is secretly
Neural2's bytes. Cross-tier comparisons among those three remain
contaminated regardless of judge quality.

These numbers are the most DOSE-comparable pronunciation numbers this
project has produced so far. They are NOT validated against human-verified
ground truth -- Workstream 1's anchor set does not exist yet -- and must
not be reported as such.
"""

from __future__ import annotations

import base64
import json
import re
import statistics
import threading
import time
from typing import Any

import requests

from .. import auth
from ..adapters.base import SynthesisResult
from ..config import GCP_PROJECT
from ..dataset import DoseItem
from .base import SCALE_MAX, SCALE_MIN, ScoreResult, Scorer

_LOCATION = "us-central1"

JUDGE_MODELS: tuple[str, ...] = (
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
)

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

_PROMPT_TEMPLATE = (
    'This audio should contain the drug name "{drug}". Listen to how "{drug}" '
    'is pronounced. If the name is absent or clearly a different word, score '
    '0. Reply ONLY JSON: {{"heard":"<what you heard>","score":<0-5>,'
    '"reason":"<brief>"}}'
)

def _panel_note(judge_models: tuple[str, ...]) -> str:
    """Describes the panel actually configured, not the 3-judge design this
    module was written for -- a single-judge instance (see
    scripts/score_with_llm_panel.py, run for cost reasons rather than the
    full panel) must not carry a note claiming disagreement was measured
    across judges that were never called.
    """
    plural = "judge" if len(judge_models) == 1 else "judges"
    agreement = ("no cross-judge agreement signal: only one judge is configured"
                if len(judge_models) == 1
                else "score = median of judges that answered")
    return (
        f"audio-LLM {plural} ({'/'.join(judge_models)}), {agreement}. "
        "All configured judges are Gemini models -- gtts-chirp3hd-achernar "
        "is itself a Gemini-family voice, so any favorable result for it "
        "may reflect self-preference bias, not measured pronunciation "
        "quality. Not validated against human ground truth."
    )

# ```json { ... } ``` or bare ``` { ... } ``` fences, non-greedy so a fence
# containing extra trailing prose after the object doesn't get swallowed.
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _judge_key(model: str) -> str:
    """gemini-2.5-flash -> gemini_2_5_flash, for component/metadata keys."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", model).strip("_")


def _extract_json(text: str) -> dict[str, Any]:
    """Parse a judge's reply into the expected {"heard","score","reason"} dict,
    tolerant of a ```json fence around the object or stray prose around it.
    Raises ValueError on anything that still doesn't parse.
    """
    text = text.strip()
    m = _FENCE_RE.search(text)
    candidate = m.group(1) if m else text
    if m is None:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = candidate[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"could not parse JSON from judge reply: {exc}") from exc
    if not isinstance(parsed, dict) or "score" not in parsed:
        raise ValueError(f"parsed JSON missing 'score' field: {candidate[:200]}")
    return parsed


def _jittered_backoff(attempt: int, base_s: float = 2.0) -> float:
    import random

    return random.uniform(0, base_s * (2 ** (attempt - 1)))


class AudioLLMPanelScorer(Scorer):
    """3-judge audio-LLM pronunciation panel. See module docstring for the
    self-preference-bias caveat that applies to every chirp3hd result.
    """

    measures_pronunciation = True

    def __init__(self, *, judge_models: tuple[str, ...] = JUDGE_MODELS,
                 project: str = GCP_PROJECT, location: str = _LOCATION,
                 timeout_s: float = 60.0, max_attempts: int = 3,
                 session: requests.Session | None = None):
        self.judge_models = judge_models
        self.project = project
        self.location = location
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self._session = session or requests.Session()
        self._usage_lock = threading.Lock()
        # model -> {"prompt_tokens": int, "candidates_tokens": int, "calls": int}
        self._usage: dict[str, dict[str, int]] = {
            m: {"prompt_tokens": 0, "candidates_tokens": 0, "calls": 0}
            for m in judge_models
        }

    @property
    def scorer_id(self) -> str:
        return "llm-panel-v1"

    def _url_for(self, model: str) -> str:
        return (
            f"https://{self.location}-aiplatform.googleapis.com/v1/projects/"
            f"{self.project}/locations/{self.location}/publishers/google/"
            f"models/{model}:generateContent"
        )

    def usage_summary(self) -> dict[str, dict[str, int]]:
        """In-process token accumulator (best-effort; the authoritative record
        for a resumable run is the per-judge usage stored in each
        ScoreResult.metadata, since this dict does not survive a process
        restart)."""
        with self._usage_lock:
            return {m: dict(v) for m, v in self._usage.items()}

    def _accumulate_usage(self, model: str, prompt_tokens: int | None,
                           candidates_tokens: int | None) -> None:
        with self._usage_lock:
            bucket = self._usage[model]
            bucket["calls"] += 1
            bucket["prompt_tokens"] += prompt_tokens or 0
            bucket["candidates_tokens"] += candidates_tokens or 0

    def _call_judge(self, model: str, audio_b64: str, drug: str) -> dict[str, Any]:
        """One judge call. Returns a dict always carrying 'ok'; on success also
        'score', 'heard', 'reason'; on failure 'error'. Token counts are
        included whenever the response returned usageMetadata, even on a
        parse failure, so cost isn't lost to a judge that answered but badly.
        """
        body = {
            "contents": [{"role": "user", "parts": [
                {"text": _PROMPT_TEMPLATE.format(drug=drug)},
                {"inlineData": {"mimeType": "audio/wav", "data": audio_b64}},
            ]}],
            "generationConfig": {"temperature": 0.0},
        }
        url = self._url_for(model)

        last_error = "unknown transport failure"
        resp = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                resp = self._session.post(
                    url, headers=auth.auth_headers(), json=body, timeout=self.timeout_s
                )
            except requests.Timeout as exc:
                last_error = f"timeout after {self.timeout_s}s: {exc}"
                resp = None
            except requests.ConnectionError as exc:
                last_error = f"connection error: {exc}"
                resp = None
            else:
                if resp.status_code == 200:
                    break
                last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                if resp.status_code not in _RETRYABLE_STATUS:
                    return {"ok": False, "error": last_error}
                resp = None
            if attempt < self.max_attempts:
                time.sleep(_jittered_backoff(attempt))

        if resp is None or resp.status_code != 200:
            return {"ok": False, "error": last_error}

        data = resp.json()
        usage = data.get("usageMetadata") or {}
        prompt_tokens = usage.get("promptTokenCount")
        candidates_tokens = usage.get("candidatesTokenCount")

        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if "text" in p)
            if not text:
                raise ValueError("no text part in judge response")
            parsed = _extract_json(text)
            score = float(parsed["score"])
            score = min(max(score, SCALE_MIN), SCALE_MAX)
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            return {
                "ok": False,
                "error": f"unparseable judge response: {exc}: {str(data)[:300]}",
                "prompt_tokens": prompt_tokens,
                "candidates_tokens": candidates_tokens,
            }

        return {
            "ok": True,
            "score": score,
            "heard": str(parsed.get("heard", "")),
            "reason": str(parsed.get("reason", "")),
            "prompt_tokens": prompt_tokens,
            "candidates_tokens": candidates_tokens,
        }

    def score(self, item: DoseItem, result: SynthesisResult) -> ScoreResult:
        base = dict(scorer_id=self.scorer_id, item_id=item.item_id,
                    system_id=result.system_id)

        if not result.ok or not result.audio:
            return ScoreResult(**base, score=0.0, scoreable=True,
                                error=result.error or "no audio returned",
                                notes="synthesis failed upstream")

        audio_b64 = base64.b64encode(result.audio).decode("ascii")

        judge_results: dict[str, dict[str, Any]] = {}
        for model in self.judge_models:
            jr = self._call_judge(model, audio_b64, item.drug)
            judge_results[model] = jr
            self._accumulate_usage(model, jr.get("prompt_tokens"),
                                    jr.get("candidates_tokens"))

        succeeded = {m: r for m, r in judge_results.items() if r.get("ok")}
        failed = {m: r for m, r in judge_results.items() if not r.get("ok")}

        components: dict[str, float] = {}
        metadata: dict[str, Any] = {"judges": {}}
        for model, r in judge_results.items():
            key = _judge_key(model)
            metadata["judges"][model] = {k: v for k, v in r.items() if k != "ok"}
            if r.get("ok"):
                components[f"judge_{key}"] = round(r["score"], 3)

        # Tolerate at most one judge failing before declaring the item
        # unscoreable, scaled to however many judges this instance actually
        # runs -- a hardcoded ">= 2" would make a single-judge configuration
        # (see JUDGE_MODELS usage in build_scorer callers) always unscoreable
        # even when its one judge succeeds, since 1 < 2.
        min_needed = max(1, len(self.judge_models) - 1)
        if len(succeeded) < min_needed:
            errs = "; ".join(f"{m}: {r.get('error')}" for m, r in failed.items())
            return ScoreResult(
                **base, score=None, scoreable=False,
                error=(f"only {len(succeeded)}/{len(judge_results)} judges "
                      f"answered (need >= {min_needed}): {errs}"),
                components=components, metadata=metadata, notes=_panel_note(self.judge_models),
            )

        scores = [r["score"] for r in succeeded.values()]
        median_score = statistics.median(scores)
        spread = max(scores) - min(scores)
        components["median"] = round(median_score, 3)
        components["spread"] = round(spread, 3)

        notes = _panel_note(self.judge_models)
        if failed:
            notes += (f" ({len(failed)} judge(s) failed and were excluded from "
                      f"the median: {', '.join(failed)})")

        return ScoreResult(
            **base, score=round(median_score, 3), components=components,
            notes=notes, metadata=metadata,
        )
