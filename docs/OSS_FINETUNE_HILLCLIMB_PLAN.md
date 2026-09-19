# Fine-tuning open-source TTS to beat RxPronounce on DOSE

Planning doc only — no execution yet. Written so any agent (human or Claude)
can pick this up cold and start the first iteration without re-deriving any
of the context below. Grounded directly in this repo's existing work, not
guessed at.

## The target

[synthiolabs.com/synthio-dose-benchmark](https://synthiolabs.com/synthio-dose-benchmark#leaderboard),
274 drug names, one carrier sentence each, scored against a reference
sourced from "labels, manufacturer materials, established medical
references" (textual/dictionary grounded, not human audio), thresholded at
pass@4.0 on a 0–5 scale:

| Model | Pass rate |
| --- | ---: |
| **Synthio RxPronounce** | **91.2%** |
| Cartesia sonic-3.6 (beta) | 80.3% |
| ElevenLabs eleven_v3 | 79.2% |
| OpenAI gpt-4o-mini-tts | 77.4% |
| xAI Grok TTS | 77.0% |
| Google gemini-3.1-flash-tts-preview | 74.5% |
| Deepgram aura-2-thalia-en | 72.6% |
| Cartesia sonic-3.5 | 69.0% |
| Azure DragonHD Neural | 63.1% |

Every listed model is a commercial cloud API. **Zero open-source baselines
exist on that board today.** Whatever open-weight model this workstream
ships first will be the first entry of its kind — the bar to clear for a
genuine win is RxPronounce's 91.2%, not just "better than nothing."

This is a hill-climbing exercise: cheap, fast, honestly-measured iterations,
not one big fine-tuning run. Nothing below should be read as "do all of
this before checking anything" — the whole point of the hard subset (see
below) is to get a signal back in minutes, not days.

## Two things this workstream built that the next step depends on

**This branch (WS1)**: the gold pronunciation reference layer,
`dose_r/references/references.jsonl` — 284 ingredients, IPA + ARPABET,
sourced from USAN/DailyMed/NCI/Merriam-Webster/Wikipedia/CMUdict/Gemini,
just made TTS-ready (real stress marks, real word boundaries, no fragments
or malformed variants — see the last few commits on this branch). Also the
reference audio manifest, `data/reference_audio/manifest.jsonl` — 358
full-coverage human recordings across 182 of the 284 ingredients, 5 sources
(Drugs.com, Merriam-Webster, UMich, ClinCalc, NCI).

**A sibling branch (WS2), `origin/claude/eloquent-mayer-0i2a52`**: the
candidate-model evaluation harness. **Not on this branch — the two have
diverged and need reconciling before any of this plan can run (see
Prerequisite 0 below).** What it contains, read directly off that branch
before writing this plan:

- `docs/CANDIDATE_MODEL_EVAL_RUNBOOK.md` — the eval protocol. Synthesize the
  carrier sentence, extract the drug-name span with
  `dose_r.forced_align.extract_drug_span_forced_align` (validated 274/274,
  zero misses — NOT the older Cloud-STT timestamp method, which has
  confirmed uncorrectable timing bugs), then score with
  `dose_r.scoring.candidate_eval.score_against_best_reference()` against
  **every** available human reference clip for that ingredient, keeping the
  best match. That last part matters concretely: the same real Gemini clip
  for Abilify scores 0.77 against its Drugs.com reference but only 0.59
  against Merriam-Webster — a 0.9-point swing on a 0–5 scale depending
  purely on which single recording you'd picked. Best-of-N is why this
  project can trust its own regression numbers at all.
- `runs/hard-subset-v1.json` — 54 items (~20% of 274), built from data
  already collected, not sampled at random. Seven tiers: 3 confirmed-bad-
  by-ear, 2 reference-flagged (the human clip itself may be wrong), 20
  Gemini's-worst-quartile, 6 ASR-can't-transcribe-even-a-human, 7
  dual-reference-coverage, 8 ceiling-control (should always pass — a
  regression here means the harness broke, not the model), 8
  no-reference-coverage (tracked, not scoreable today). **This subset is
  explicitly not population-representative** (72% brand vs. 52% in the
  full corpus) and must never be reported as a pass rate on its own — it
  exists purely so an iteration loop doesn't need the full 274 every time.
- `dose_r/scoring/candidate_eval.py`, `dose_r/scoring/speech_similarity.py`
  — the actual scorer: wavlm frame embeddings + BERTScore-style F1 between
  the candidate's extracted span and each human reference clip.
- `dose_r/adapters/systems.yaml` already has an `oss-selfhosted` entry
  (`backend: openai_compatible_http`, `status: planned`) — the slot exists,
  the backend class does not. `dose_r/adapters/registry.py` only registers
  `mock` and `gemini_vertex` today.

Two open, unresolved gaps that WS2's own runbook flags and this plan
inherits rather than re-litigates:

1. **The human reference audio isn't guaranteed correct either.** A
   13-item spot check found 2 items (talquetamab, acoramidis) where the
   reference recording plausibly diverges from its own dictionary IPA. A
   full sweep across all 178 dual-coverage items hasn't been run.
2. **Whether the audio-embedding F1 metric even detects a single-phoneme
   error is unresolved.** A 13-item probe found phonetic-hint injection
   (SSML IPA, Gemini-with-IPA-in-prompt) never clearly beat naive
   plain-spelling synthesis (0.719–0.725 vs. 0.520 baseline) — consistent
   with either the injection not working or the metric being too coarse to
   reward it either way. A deliberate minimal-pair test (same word, one
   phoneme deliberately wrong, does the score actually drop) was proposed
   and never run.

**Gap 2 is the one that should block trusting hill-climb signal, not just
be noted in passing**: if the scoring metric can't reliably detect a
one-phoneme fix, then fine-tuning iterations will look like noise
regardless of whether the model is actually improving. Prerequisite 2
below is that minimal-pair test, promoted from "open question" to "must
run before iteration 2."

## Prerequisites (before the first real fine-tuning run)

**0. Reconcile WS1 and WS2.** They currently share no common recent
history — WS2's branch deletes and rewrites files WS1 has since changed
further (`requirements.txt`, `scripts/fetch_*_reference_audio.py`,
`tests/test_references.py`, `tests/test_reference_audio.py`, among
others). Before anyone can run the eval harness against WS1's newly
TTS-ready references, someone has to merge deliberately: WS1's
`references.jsonl`/`notation.py`/`wiki_notation.py`/audio manifest work is
newer and more correct than whatever WS2 branched from, and WS2's
`forced_align.py`/`scoring/`/hard-subset/runbook don't exist on this
branch at all. This is not a fast-forward; it needs a person or agent to
read both diffs and merge with intent, not `git merge -X theirs`.

**1. Land the `oss-selfhosted` adapter backend.** The YAML slot exists;
`registry.py` needs an `openai_compatible_http`-speaking backend class
(most self-hosted OSS TTS servers — see candidates below — expose an
OpenAI-compatible `/v1/audio/speech` endpoint, or close enough that one
adapter class covers several candidates with a config field for the
payload shape). This is a small, mechanical addition following the exact
pattern `dose_r/adapters/gemini.py` already establishes; the harness,
runner, and record schema need no changes (systems.yaml's own comment says
so).

**2. Run the minimal-pair metric-sensitivity test.** Take 10–15 items,
synthesize each with a deliberately correct phoneme sequence and a
deliberately wrong one (one phoneme swapped — e.g. via SSML `<phoneme>`
injection against a model that honors it), score both against the same
human reference(s), and confirm the score actually separates them by a
meaningful margin. If it doesn't, fix the scoring approach (candidate:
score in phoneme/ARPABET space using WS1's `dose_r/judge/phonetic_scorer.py`
directly on the extracted span's recognized phonemes, alongside — not
instead of — the audio-embedding score, the same two-signal idea DOSE-R's
own hybrid judge already uses elsewhere in this project) before trusting
any hard-subset number as a real training signal.

**3. Confirm what "beat RxPronounce on their own benchmark" requires.**
The leaderboard's own methodology scores against a **textual/dictionary**
reference at pass@4.0 on a 0–5 scale — not the audio-embedding best-of-N
approach WS2 built for internal iteration. An internal number produced by
our own harness is a good hill-climbing signal but is not automatically
comparable to the public board. Two options, not mutually exclusive: (a)
find out if Synthio accepts external submissions to their own eval and run
the winning candidate through it directly, or (b) reverse-engineer their
scoring as closely as documented (0–5 scale, pass@4.0, textual reference)
and report both numbers side by side, clearly labeled, so nobody mistakes
one for the other later.

## Candidate open-source models to fine-tune

Ranked by how directly they fit this problem, not by general TTS quality.
The single biggest lever this project has that a generic TTS fine-tune
doesn't is the gold IPA reference layer — so a model with a **native
phoneme/IPA input path** lets us try the cheap fix (override the
pronunciation lexicon, no training at all) before spending any compute on
weight updates. Validate every model's current license, weights
availability, and fine-tuning recipe at execution time — this list is a
starting point, not a promise any of these haven't changed.

**Tier 1 — phoneme-native, try the free fix first:**

- **StyleTTS2** — phonemizes input via `phonemizer`/espeak-ng into IPA
  before synthesis; the most direct match to this project's own reference
  format. A custom pronunciation dictionary (map each of the 284 drug names
  to its gold IPA string, short-circuiting espeak's own grapheme-to-phoneme
  guess) is close to a zero-training-cost first experiment. LoRA/full
  fine-tuning recipes are well-documented if the lexicon override alone
  doesn't close the gap (e.g. on names needing prosody/duration fixes a
  static phoneme swap can't reach).
- **Kokoro** (82M params) — also espeak-ng-based phonemization, small and
  fast enough for a full 274-item run to cost almost nothing per iteration.
  Same lexicon-override-first approach applies. Worth trying purely for
  iteration speed even if peak quality trails larger models.
- **Zonos** (Zyphra) — real-time, open-weight, phoneme conditioning via
  espeak; newer and less battle-tested for fine-tuning than StyleTTS2 but
  worth a spot check given stated real-time performance.

**Tier 2 — strong voice quality, fine-tunable, no native IPA path (needs
either weight fine-tuning or a text-normalization trick to spell
pronunciation phonetically in the input string):**

- **F5-TTS** — flow-matching, strong zero-shot quality, actively used for
  fine-tuning in the open community; well-documented LoRA recipes.
- **CosyVoice2** (Alibaba) — open weights, supports fine-tuning and has
  some phoneme/pinyin-style control precedent worth checking for an
  English IPA equivalent.
- **Fish-Speech / OpenAudio** (Fish Audio) — open weights, good quality,
  documented fine-tuning path.
- **XTTS-v2** (Coqui architecture; community-maintained after Coqui's
  shutdown) — the most heavily documented open-source TTS fine-tuning
  recipe that exists, by volume of community tutorials, even though the
  original company is gone. Worth including for that reason alone: more
  prior art to debug against when something goes wrong.
- **Orpheus TTS** (Canopy Labs, Llama-backbone) — newer, LLM-backbone
  models are usually easier to steer via in-context examples in the
  prompt (few-shot "here's how to say X" pairs) in addition to weight
  fine-tuning, which could be a cheap middle rung between "lexicon
  override" and "full fine-tune."
- **Parler-TTS** (Hugging Face) — natural-language style-prompt control;
  less relevant to pronunciation specifically but cheap to try given how
  well-documented its fine-tuning path is.

**Recommended order for the first pass**: StyleTTS2 lexicon-override (no
training) → Kokoro lexicon-override (no training, cross-check) → if
neither closes enough of the gap, StyleTTS2 or F5-TTS LoRA fine-tune on
the 358 real reference clips → XTTS-v2 as a fallback if the LoRA fine-tune
underperforms and more community prior art is needed to debug why.

## Fine-tuning data

`data/reference_audio/manifest.jsonl` — 358 full-name-coverage human clips
across 182 of 284 ingredients, 5 sources. Real constraints to design
around, not to discover mid-run:

- **Thin per-word coverage.** Most ingredients have 1–4 clips total, one
  take each, no multiple speakers. This is a pronunciation-correction
  signal, not enough data for a general acoustic fine-tune — LoRA/adapter
  fine-tuning (small parameter delta, low forgetting risk) is the right
  scale of intervention, not a full retrain.
- **Isolated word clips, not carrier-sentence clips.** The human
  references say just the drug name; DOSE scores the name inside a full
  carrier sentence. Fine-tuning on isolated words risks a model that
  pronounces the name right in isolation but reverts under sentence-level
  coarticulation — validate this specifically (synthesize the SAME name
  inside its real carrier sentence, not standalone, before trusting a
  fine-tune improved anything).
- **99/284 ingredients (36% of the full corpus, though the ~20% hard
  subset skews this differently) have no human audio at all.** For these,
  there is no audio fine-tuning target — the gold IPA/ARPABET reference
  (this branch's `references.jsonl`) is the only ground truth, so
  correction for these names has to route through the phoneme-input path
  (lexicon override or IPA-conditioned synthesis), not audio-target
  fine-tuning.
- **Augmentation options worth testing, not assuming will work:**
  synthesizing the same gold IPA sequence in the base model's own
  zero-shot voice-cloning across several reference voices (multiplies
  apparent speaker diversity without new human recordings, but the
  "ground truth" then depends entirely on the phoneme-to-audio step being
  correct, so validate a sample by ear); mixing real clips 1:1 or higher
  with the fine-tune's own pre-training data to reduce catastrophic
  forgetting of general speech quality on the other 90 carrier-sentence
  words a candidate also has to say correctly.

## The hill-climb loop

1. Pick a candidate + technique (lexicon override, LoRA, or full
   fine-tune) from the tiers above.
2. Run it against `runs/hard-subset-v1.json` (54 items) only. Score with
   both the audio-embedding best-of-N path (WS2) and the phoneme-distance
   path (WS1's `dose_r/judge/phonetic_scorer.py`) once Prerequisite 2
   confirms the audio metric is trustworthy on its own; until then, treat
   the phoneme score as primary.
3. A regression on the 8 ceiling-control items means the harness or
   adapter broke, not that the model got worse — fix that before reading
   anything else in the run.
4. Iterate technique/data/candidate on the hard subset until a candidate
   clearly clears its own baseline by a margin bigger than the
   minimal-pair test's noise floor (Prerequisite 2 sets what "margin"
   means quantitatively).
5. Only then run the full 274-item corpus, scored the same way, before
   comparing anything to the real leaderboard or claiming a number beats
   RxPronounce. Record it the same way existing runs are recorded
   (`runs/<name>/manifest.jsonl` + `run.json` + `summary.json`, the
   convention already established in this repo).
6. Repeat from step 1 with the next candidate/technique. Nothing here is
   sequential-only — once the adapter backend (Prerequisite 1) exists,
   multiple candidates can run the hard-subset loop in parallel.

## What "done" looks like for this workstream

Not a single fine-tuned model — a **leaderboard-comparable score for at
least one open-source candidate**, honestly measured (full 274 corpus,
methodology gap from Prerequisite 3 resolved or explicitly caveated), plus
a clear next-candidate queue for whoever picks this up next. Beating
91.2% is the stretch goal every iteration is aimed at; the floor goal is
"the first real open-source number on that board at all," which by itself
is new information nobody else has published yet.
