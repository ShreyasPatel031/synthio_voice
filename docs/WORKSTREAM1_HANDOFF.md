# Workstream 1 integration contract

This is the seam between Workstream 2 (runner, adapters, latency/cost
capture) and Workstream 1 (gold pronunciation reference layer, hybrid judge).
Workstream 1 needs to implement one interface and register it; everything
else in the harness — the CLI, the runner, the report, resumability — already
works against it via the stand-in scorer (`dose_r/scoring/standin.py`).

Do not edit `dose_r/runner.py`, `dose_r/report.py`, or the adapter layer to
integrate the judge. If the interface below doesn't fit what the hybrid judge
needs, that's a conversation to have about the interface, not a reason to
reach into the runner.

## 1. The `Scorer` interface

Defined in `dose_r/scoring/base.py`. Full source is short; read it directly,
but here is the exact contract:

```python
PASS_THRESHOLD = 4.0
SCALE_MIN, SCALE_MAX = 0.0, 5.0

class Scorer(ABC):
    """Base class for a judge."""

    #: Set False on any scorer that does not actually measure pronunciation.
    #: The report generator refuses to present such numbers as DOSE-comparable.
    measures_pronunciation: bool = True

    @property
    @abstractmethod
    def scorer_id(self) -> str: ...

    @abstractmethod
    def score(self, item: DoseItem, result: SynthesisResult) -> ScoreResult: ...

    def score_batch(
        self, pairs: list[tuple[DoseItem, SynthesisResult]]
    ) -> list[ScoreResult]:
        """Override for judges that batch efficiently (e.g. an LLM panel)."""
        return [self.score(item, res) for item, res in pairs]
```

- `scorer_id` — a stable string identifying this judge version, e.g.
  `"hybrid-judge-v1"`. It is written into every `ScoreResult` and into the
  run manifest, so bump it whenever judge logic changes in a way that would
  make old and new scores non-comparable.
- `score(item, result)` — called once per (item, system) synthesis by
  `BenchmarkRunner._persist` (`dose_r/runner.py`), synchronously, in the
  worker thread that just produced `result`. Must not raise for expected
  failure modes (see `ScoreResult.scoreable`/`error` below) — only raise for
  genuine bugs, since an uncaught exception here would propagate out of
  `_persist` and abort the run for that item's writer thread ordering. Look
  at `StandInScorer.score` for the pattern of returning a `ScoreResult` with
  `score=0.0` on synthesis failure and `scoreable=False` on unrecoverable
  input.
- `score_batch(pairs)` — optional override. The runner does not currently
  call this (see "What's not wired up yet" below); it exists on the
  interface for a batching judge (e.g. one LLM call scoring several items)
  to implement against once the runner is extended to call it instead of
  `score()` one at a time. If you need batching now, flag it — that's a
  small runner change, not something to work around unilaterally.
- `measures_pronunciation` — **this exact flag is what unlocks DOSE
  leaderboard comparison.** `report.compare_to_dose()` checks
  `manifest["scorer_measures_pronunciation"] is True` (written from this
  attribute at run time) and raises `NotLeaderboardComparable` otherwise:

  ```python
  def compare_to_dose(summaries, manifest, mapping) -> dict:
      """Workstream 1d comparison. Refuses to run on a non-pronunciation scorer."""
      if manifest.get("scorer_measures_pronunciation") is not True:
          raise NotLeaderboardComparable(...)
      ...
  ```

  Set `measures_pronunciation = True` (the default) on the real judge class.
  Leave it `False` only on scorers that are explicitly not attempting
  pronunciation judgement (like `StandInScorer`).

### `ScoreResult` — every field

```python
@dataclass
class ScoreResult:
    scorer_id: str
    item_id: str
    system_id: str

    score: float | None          # 0-5, None when unscoreable
    scoreable: bool = True
    error: str | None = None

    components: dict[str, float] = field(default_factory=dict)
    reference_confidence: str = "unknown"
    confusable_with: str | None = None
    confusability_margin: float | None = None

    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool | None:
        if not self.scoreable or self.score is None:
            return None
        return self.score >= PASS_THRESHOLD
```

- `scorer_id`, `item_id`, `system_id` — echo back the ids passed in; used to
  join into `results.jsonl`.
- `score` — 0-5 on DOSE's scale, matching `PASS_THRESHOLD = 4.0`. `None`
  when the item genuinely could not be scored (distinct from a low score).
- `scoreable` — `False` for e.g. corrupt/unreadable audio where no
  judgement can be made at all. `passed` returns `None` in that case rather
  than `False`, so a bad recording is never silently counted as a fail in
  the pass-rate denominator logic downstream — check `report.summarize`,
  which only includes rows where `r["score"].get("score") is not None` in
  `scored`.
- `error` — human-readable reason when `scoreable=False` or `score=0.0` due
  to upstream synthesis failure.
- `components` — named sub-scores that combine into `score`, e.g.
  `{"phonetic_distance": 3.8, "llm_panel_median": 4.0}` for the intended
  phonetic-distance + 3-judge audio-LLM panel design. Kept because
  `report.render_text` deliberately retains the raw score distribution
  instead of collapsing to pass/fail only (`SystemSummary.score_mean`,
  `score_median`, `score_p10`) — component names are yours to choose, the
  harness does not currently interpret them, just stores and forwards them.
- `reference_confidence` — `"high"|"medium"|"low"` (or `"unknown"`, the
  default), inherited from the gold-reference layer's confidence in the
  reference pronunciation used for this item. Report generation does not
  yet branch on this field — it exists so low-confidence references can be
  surfaced/filtered separately later rather than mixed silently into the
  headline pass rate.
- `confusable_with` — the name of a *different* drug this synthesized audio
  might be judged as saying instead (a LASA — Look-Alike/Sound-Alike —
  guardrail). `None` until the reference layer populates it.
- `confusability_margin` — a numeric margin/confidence for that confusion
  call, paired with `confusable_with`.
- `notes`, `metadata` — free text and free-form JSON-safe dict for anything
  else (e.g. raw judge-panel transcripts, phoneme alignments).

`to_record()` serializes all of this into the `"score"` field of each
`results.jsonl` row; nothing needs to be added by hand on the runner side.

## 2. Registering the judge

`dose_r/scoring/__init__.py`:

```python
_REGISTRY: dict[str, type[Scorer]] = {
    "standin": StandInScorer,
}

def register_scorer(name: str, cls: type[Scorer]) -> None:
    """Workstream 1 calls this to plug the real judge in."""
    _REGISTRY[name] = cls
```

Call `dose_r.scoring.register_scorer("hybrid", HybridJudgeScorer)` (module
import time is fine, e.g. in your package's `__init__.py`, as long as it
runs before `build_scorer` is called). Then run the harness with:

```bash
python scripts/run_benchmark.py --tier cheap --scorer hybrid
```

`scripts/run_benchmark.py` calls `build_scorer(args.scorer)`
(`dose_r/scoring/__init__.py`) which does `_REGISTRY[name](**kwargs)` — your
`Scorer` subclass's `__init__` should therefore work with zero required
positional args if it needs to be constructible from just the CLI name
(commonly: read config/credentials from environment or module-level
defaults inside `__init__`).

## 3. Worked example — skeleton hybrid judge

```python
# dose_r_hybrid/scorer.py  (or wherever Workstream 1's package lives)
from __future__ import annotations

from dose_r.adapters.base import SynthesisResult
from dose_r.dataset import DoseItem
from dose_r.scoring.base import SCALE_MAX, SCALE_MIN, ScoreResult, Scorer
from dose_r.scoring import register_scorer


class HybridJudgeScorer(Scorer):
    """Phonetic-distance + 3-judge audio-LLM panel."""

    measures_pronunciation = True  # unlocks report.compare_to_dose()

    def __init__(self, reference_layer=None, panel_client=None):
        # Load the gold reference layer and any judge-panel client here.
        # Keep this constructible with no required args so
        # `build_scorer("hybrid")` (no kwargs) works from the CLI.
        self.reference_layer = reference_layer or _default_reference_layer()
        self.panel_client = panel_client or _default_panel_client()

    @property
    def scorer_id(self) -> str:
        return "hybrid-judge-v1"

    def score(self, item: DoseItem, result: SynthesisResult) -> ScoreResult:
        base = dict(scorer_id=self.scorer_id, item_id=item.item_id,
                    system_id=result.system_id)

        if not result.ok or not result.audio:
            return ScoreResult(**base, score=0.0, scoreable=True,
                                error=result.error or "no audio returned",
                                notes="synthesis failed upstream")

        ref = self.reference_layer.lookup(item.drug)  # item.drug is normalized!
        if ref is None:
            return ScoreResult(**base, score=None, scoreable=False,
                                error=f"no gold reference for {item.drug!r}")

        phonetic = self._phonetic_distance_component(item, result, ref)
        panel = self._llm_panel_component(item, result, ref)

        components = {"phonetic_distance": phonetic, "llm_panel_median": panel}
        combined = 0.4 * phonetic + 0.6 * panel  # placeholder weighting
        score = round(min(max(combined, SCALE_MIN), SCALE_MAX), 3)

        confusable_with, margin = self._confusability_check(item, result, ref)

        return ScoreResult(
            **base, score=score, components=components,
            reference_confidence=ref.confidence,       # "high"|"medium"|"low"
            confusable_with=confusable_with,
            confusability_margin=margin,
            metadata={"reference_source": ref.source},
        )

    # def score_batch(self, pairs): ...  # implement if the LLM panel batches

    def _phonetic_distance_component(self, item, result, ref) -> float: ...
    def _llm_panel_component(self, item, result, ref) -> float: ...
    def _confusability_check(self, item, result, ref) -> tuple[str | None, float | None]: ...


register_scorer("hybrid", HybridJudgeScorer)
```

## 4. What the runner already hands you, per item

From `BenchmarkRunner._persist` (`dose_r/runner.py`), `score(item, result)`
receives:

**`item: DoseItem`** (`dose_r/dataset.py`):

```python
@dataclass(frozen=True)
class DoseItem:
    drug: str              # normalized name, as spoken in the sentence
    name_type: str         # "brand" | "generic"
    sentence: str
    drug_raw: str = ""     # field exactly as shipped, kept for provenance

    @property
    def item_id(self) -> str: ...   # drug.lower() with spaces/slashes -> "-"

    @property
    def char_count(self) -> int: ...  # len(sentence); billable chars
```

**`result: SynthesisResult`** (`dose_r/adapters/base.py`):

```python
@dataclass
class SynthesisResult:
    system_id: str
    item_id: str
    ok: bool
    audio: bytes | None = None
    audio_format: str = "wav"
    sample_rate_hz: int | None = None

    ttfa_ms: float | None = None      # time-to-first-audio; None for non-streaming
    total_ms: float | None = None
    streaming: bool = False

    billable_chars: int = 0
    cost_usd: float | None = None
    cost_estimated: bool = True
    price_verified: bool = False

    attempts: int = 1
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

`result.audio` is the raw WAV bytes (24 kHz, 16-bit PCM mono from the current
Google TTS and mock adapters — check `audio_format`/`sample_rate_hz` rather
than assuming). On disk, the same bytes live at
`runs/<run_id>/audio/<system_id>/<item_id>.wav` (relative path stored as
`audio_path` in each `results.jsonl` record) whenever the run was not started
with `--no-audio`. If your judge pipeline runs as a separate offline batch
job rather than inline during `score()`, that path — not re-synthesizing —
is the way to get the audio for a completed run.

## 5. Data notes for the gold reference layer

### 5.1 The `(generic)` suffix issue (see also root `README.md`)

All 131 generic rows ship `drug_raw` as e.g. `"acetaminophen (generic)"` while
the sentence contains only `"acetaminophen"`. **Use `item.drug`
(post-`normalize_drug_name`), never `item.drug_raw`, as the reference-layer
lookup key and as the span target inside the sentence/audio.** The regex is
`r"\s*\((?:generic|brand|inn)\)\s*$"` in `dose_r/dataset.py`;
`dataset.validate()` asserts the normalized name appears in every sentence,
so this is enforced at load time, not just documented.

### 5.2 Biosimilar 4-letter suffixes — open reference-design question

A subset of the generic set are biosimilars with an FDA-assigned four-lowercase-letter
nonproprietary suffix, e.g. (confirmed present in `data/processed/dose_v1_canonical.csv`):

```
atacicept-vymj
bevacizumab-vikg
bulevirtide-gmod
elranatamab-bcmm
faricimab-svoa
lebrikizumab-lbkz
nipocalimab-aahu
nogapendekin alfa inbakicept-pmln
pegvaliase-pqpz
pegzilarginase-nbln
pivekimab sunirine-pvzy
risankizumab-rzaa
sotatercept-csrk
teplizumab-mzwv
tividenofusp alfa-eknm
veligrotug-vvze
```

**Nobody has decided how the gold reference should pronounce the suffix**,
and it materially changes the reference: spoken as individual letters
("vymj" -> "vee-why-em-jay"), as a pronounceable syllable ("vymj" -> "vim"),
or elided entirely and scored on the base name only. This is an open
question for Workstream 1's reference-design decision, not something the
harness or dataset resolves — flag it explicitly in the reference-layer
design doc rather than picking silently, since it will change gold audio for
17 items and any judge trained/tuned against one convention will
mis-penalize the other.

### 5.3 Multi-word generics

Several generic names are multi-word combination names, e.g.
`insulin icodec-abae`, `pivekimab sunirine-pvzy`,
`nogapendekin alfa inbakicept-pmln`, `bictegravir, emtricitabine, and
tenofovir alafenamide`, `elexacaftor, tezacaftor, and ivacaftor`,
`vanzacaftor, tezacaftor, and deutivacaftor`. These combine the multi-word
issue with the suffix issue in some cases (e.g. `insulin icodec-abae`,
`pivekimab sunirine-pvzy`, `nogapendekin alfa inbakicept-pmln`). The
normalized `item.drug` is the full string as it appears in the sentence
(commas included where present) — the reference layer needs a phrase-level
match, not a single-token lookup.

### 5.4 No era or difficulty columns — both must be re-derived

The public parquet (`data/raw/dose_v1.parquet`, loaded via
`dose_r/dataset.load_raw`) has exactly four usable columns: `drug`,
`drug_raw`, `name_type`, `sentence`. **There is no era column
(established vs. newly-approved) and no difficulty tier.** DOSE's published
leaderboard reports splits of 128/146 (era) and 63/102/109
(easy/medium/hard), and reproducing those breakdowns requires re-deriving
both:

- **Era (established vs. new)**: needs an FDA approval-date lookup per drug
  (e.g. Orange Book / Drugs@FDA) and a cutoff decision — DOSE's own cutoff
  is not published in this dataset and must be inferred or independently
  chosen.
- **Difficulty tier (easy/medium/hard)**: needs a re-derived proxy — likely
  syllable count and/or morphological complexity (consonant clusters,
  non-English phoneme sequences, suffix count) — since DOSE does not publish
  its own tiering methodology in the released set either.

Neither derivation is implemented anywhere in this codebase today. Until
Workstream 1 (or a joint decision) produces both, per-stratum reporting is
limited to `name_type` (brand/generic), which is what `report.summarize`
currently breaks out (`pass_rate_by_stratum`). Treat any era/difficulty
number quoted before this work is done as invented, not derived.

## 6. Control baselines — what they do and don't validate

`dose_r/config.py` `MOCK_TIER` defines five control systems, all backed by
`dose_r/adapters/mock.py`:

- `mock-perfect`
- `mock-truncated`
- `mock-silent`
- `mock-clipped`
- `mock-overlong`

Run them with `python scripts/run_benchmark.py --tier control`.

Each emits a deterministic synthetic sine-wave WAV (seeded from a hash of the
input text, so runs are reproducible) exhibiting one acoustic defect mode:
truncated duration, near-silent amplitude, clipped waveform, or excessive
duration. `mock-perfect` has none of these defects.

**Honest caveat, directly from the mock module's docstring:** these are
*acoustic* faults, not mispronunciations. A mock backend cannot produce a
genuine mispronunciation — there's no linguistic content in a sine wave. The
control baselines exist to validate that the runner and report pipeline
correctly propagate failures and low scores end-to-end (audio persists
correctly, latency/cost still get recorded, a bad recording produces a low
`StandInScorer` score, `report.compare_to_dose()` correctly refuses
comparison, etc.) — i.e. **they validate plumbing, not judge quality.**

Validating that the real hybrid judge can actually tell a correct
pronunciation from an incorrect one requires a **human-verified anchor set**
of roughly 40-60 items with known-correct and known-incorrect gold audio —
that anchor set does not exist yet in this repo and is Workstream 1's to
build. Do not treat a clean pass on the mock controls as evidence the judge
works; it only shows the harness wiring works.
