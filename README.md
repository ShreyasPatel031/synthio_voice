# DOSE-R

DOSE-R is a replication and extension harness for [DOSE v1.0](https://synthiolabs.com/synthio-dose-benchmark)
("Drug-name Oral Synthesis Evaluation"), Synthio's benchmark for how accurately
TTS systems pronounce pharmaceutical drug names inside real clinical sentences.
DOSE scores each of 274 items 0-5 with an audio judge; an item **passes** at
score >= 4. The public dataset ships the held-out test set only (drug name,
name type, sentence) — no reference pronunciations, evaluation audio, or
per-system scores, so it cannot be used as an answer key on its own.

## Two workstreams

- **Workstream 1** owns the gold pronunciation reference layer and the hybrid
  judge (phonetic-distance + audio-LLM panel) that actually scores
  pronunciation. See `dose_r/scoring/base.py` for the interface they implement
  and `docs/WORKSTREAM1_HANDOFF.md` for the full integration contract.
- **Workstream 2** (this codebase, as it stands) owns the runner harness,
  the TTS adapters, latency/cost capture, and the accuracy/latency/cost
  tradeoff blueprint (`dose_r/report.py`).

The two meet at the `Scorer` interface: the runner calls whatever scorer is
registered, and nothing else in the harness needs to change when Workstream 1's
judge replaces the stand-in.

## Setup

Use the repo's virtualenv directly:

```bash
/home/user/synthio_voice/.venv/bin/python scripts/run_benchmark.py --list
```

Credentials for the Google Cloud TTS adapter come from the
`GOOGLE_APPLICATION_CREDENTIALS_JSON` environment variable: a service-account
key, either raw JSON or base64-encoded JSON. `dose_r/auth.py` decodes it in
memory and never writes it to disk — the harness persists audio, timings and
costs, never credentials. Without this variable set, only the `mock-*` systems
(no network, no spend) are usable.

```bash
export GOOGLE_APPLICATION_CREDENTIALS_JSON="$(base64 -w0 /path/to/service-account.json)"
```

## Quickstart

List every configured system (cheap Cloud TTS tier + mock controls):

```bash
python scripts/run_benchmark.py --list
```

Offline harness self-test — no network, no spend, runs against the mock
backend:

```bash
python scripts/run_benchmark.py --systems mock-perfect mock-truncated --limit 20
```

Full cheap-tier run over all 274 items against real Google Cloud TTS voices
(requires `GOOGLE_APPLICATION_CREDENTIALS_JSON`):

```bash
python scripts/run_benchmark.py --tier cheap
```

Other flags on `scripts/run_benchmark.py` (see its `--help` / docstring for
the authoritative list): `--systems <ids...>` (explicit system ids, mutually
exclusive with `--tier`), `--tier {cheap,control,all}`, `--scorer <name>`
(default `standin`), `--limit N` (first N items, for smoke tests),
`--concurrency N` (default 4), `--run-id <id>`, `--no-audio` (skip persisting
audio), `--notes "<text>"`.

Run the test suite (all offline, against the mock backend):

```bash
python -m pytest tests/
```

## Dataset

`dose_r/dataset.py` loads `data/raw/dose_v1.parquet` (274 rows: 143 brand,
131 generic) and writes a canonical CSV plus an integrity manifest to
`data/processed/` via `dataset.stage()`. `dataset.validate()` asserts the
published row count and brand/generic split, checks for duplicate drug names,
and — most importantly — asserts that every sentence actually contains its
(normalized) drug name.

### The normalization gotcha

**All 131 generic rows ship their `drug` field with a trailing label suffix**,
e.g. `"acetaminophen (generic)"`, while the carrier sentence contains only
`"acetaminophen"`. The `(generic)` / `(brand)` / `(inn)` suffix duplicates
`name_type` and is not part of the spoken name. If a scorer is handed the raw
`drug` field as the text span to search for or align against, **all 131
generic items become unmatchable and silently unscoreable.**

`dose_r.dataset.normalize_drug_name()` strips this suffix with the regex
`r"\s*\((?:generic|brand|inn)\)\s*$"`. `DoseItem.drug` is always the
normalized name; `DoseItem.drug_raw` keeps the exact shipped value for
provenance. `dataset.validate()` asserts the normalized name appears in every
sentence, so a re-pull that changes this convention fails loudly instead of
silently. `tests/test_harness.py::test_generic_label_suffix_is_stripped` and
`test_every_sentence_contains_its_normalized_drug_name` pin this behavior.

**Anyone building a scorer against `DoseItem` must use `.drug`, never
`.drug_raw`, as the span target.** See
`docs/WORKSTREAM1_HANDOFF.md` for the full detail relevant to the reference
layer.

## Run artifact layout

Each run writes to `runs/<run_id>/`:

- `manifest.json` — run config, scorer id, `scorer_measures_pronunciation`,
  per-system ok/failed/cost totals, wall-clock time, GCP project, platform.
- `results.jsonl` — one JSON record per (system, item) call: the `DoseItem`
  fields, the `audio_path` (relative to the run dir), the full
  `SynthesisResult` record (latency, cost, retries, error), and the `score`
  record if a scorer was attached.
- `report.txt` — the human-readable tradeoff table (`report.render_text`).
- `summary.json` — the same aggregates as machine-readable `SystemSummary`
  dicts (`report.summarize`).
- `audio/<system_id>/<item_id>.wav` — the synthesized audio, when
  `--no-audio` is not passed.

Runs are resumable: `results.jsonl` is scanned for already-completed
`(system_id, item_id)` pairs and only the remainder is synthesized.

See `runs/standin-v1/` for a completed example (4 cheap-tier systems, full
274 items, stand-in scorer).

## IMPORTANT: the current scorer is a stand-in, not a pronunciation judge

`dose_r/scoring/standin.py` (`StandInScorer`, `scorer_id =
"standin-audio-plausibility-v1"`) is the only scorer implemented today. It
checks **audio deliverability** — plausible duration for the text length,
sane signal level, absence of clipping — not whether the drug name was
pronounced correctly. It has no gold pronunciation reference and cannot know
whether "retatrutide" was said right.

Its `measures_pronunciation` class attribute is `False`. `report.render_text`
prints a loud warning banner whenever a manifest carries a non-pronunciation
scorer, and `report.compare_to_dose()` **raises
`NotLeaderboardComparable`** if anyone attempts to compare its pass rates
against `config.DOSE_LEADERBOARD`. Every pass rate and score in
`runs/standin-v1/report.txt` is a stand-in figure and is **not
DOSE-comparable**.

**Latency and cost figures in every run ARE real measurements** — they come
from actual wall-clock timings and `estimate_cost()` against the synthesized
character counts, independent of which scorer is attached.

## Cost figures are unverified list prices

`dose_r/config.py`'s `GOOGLE_TTS_PRICING` table is transcribed from public
Cloud TTS pricing docs and has **not** been reconciled against a billing
export. Every `Pricing` entry (standard, wavenet, neural2, studio, chirp-hd,
chirp3-hd, news, polyglot, casual) has `verified=False` except the `mock`
tier (`verified=True`, $0). `SynthesisResult.cost_estimated` is `True` for
every live call, and `report.render_text` appends a "Cost note: figures are
list-price estimates, not billing-export verified" line whenever any system
in the run has `cost_estimated=True`. Treat every dollar figure in a report
as indicative, not billed.
