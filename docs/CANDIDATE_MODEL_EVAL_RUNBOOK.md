# Evaluating a new candidate TTS model against human recordings

> Pronunciation hints: original source only. Never convert DailyMed/USAN
> respelling to IPA. See `dose_r/references/README.md`.

Workstream 2, 2026-09-19. For Workstream 1's upcoming open-source candidate
model evaluations. Answers: how to score a new model against human
pronunciation (using every available recording, not just one), and which
subset of the 274-item benchmark to run first so a full pass isn't needed
every iteration.

## External grounding: the real DOSE leaderboard

Checked https://synthiolabs.com/synthio-dose-benchmark#leaderboard directly
before writing this. 9 models listed, **all commercial cloud APIs, zero
open-source baselines**:

| Model | Pass rate |
| --- | ---: |
| Synthio RxPronounce | 91.2% |
| Cartesia sonic-3.6 (beta) | 80.3% |
| ElevenLabs eleven_v3 | 79.2% |
| OpenAI gpt-4o-mini-tts | 77.4% |
| xAI Grok TTS | 77.0% |
| Google gemini-3.1-flash-tts-preview | 74.5% |
| Deepgram aura-2-thalia-en | 72.6% |
| Cartesia sonic-3.5 | 69.0% |
| Azure DragonHD Neural | 63.1% |

Methodology per that page: 274 names, one carrier sentence each, scored 0-5
against a reference and thresholded at pass@4.0, reference sourced from
"labels, manufacturer materials, established medical references" (i.e.
textual/dictionary, not human-recorded audio), brand names score ~12 points
higher than generic on average (94.4% vs 87.8%), and newly-approved drugs
are the hardest tier for every model. There is no existing baseline for an
open-source model on that leaderboard -- Workstream 1's numbers will be the
first.

## How to score one candidate model

For each item in the eval set:

1. **Synthesize** the item's full carrier sentence with the candidate model
   (same sentences `dose_r.dataset` already has -- do not write new ones).
2. **Extract the drug-name span** with `dose_r.forced_align.extract_drug_span_forced_align`
   (aligns against the KNOWN sentence text via the phoneme CTC model + forced
   alignment -- validated this session on 274/274 items with zero
   extraction misses after two rounds of fixes; do not use the older
   Cloud-STT-timestamp approach in `dose_r.audio_span`, which has confirmed,
   uncorrectable-at-the-margin timing bugs).
3. **Score against EVERY available human reference, take the best match** --
   `dose_r.references.reference_clips.available_clips_all()` +
   `dose_r.scoring.candidate_eval.score_against_best_reference()`. This is
   NOT the same call the shipped Path 2 report uses
   (`SpeechSimilarityScorer`/`available_clips()`), which deliberately picks
   ONE reference (Merriam-Webster preferred) for score stability across
   repeated runs of the SAME candidate over time. Evaluating a NEW candidate
   is a different question -- "is this an accepted pronunciation at all,"
   not "does it match this one specific recording" -- and scoring against
   only one reference already produced a real false positive this session:
   aripiprazole and acoramidis scored as top-5-worst of 171 items against a
   single Merriam-Webster reference, when the user confirmed by ear they
   were valid alternate pronunciations, just not that specific recording's.
   Concrete demonstration, checked directly while building this: Abilify's
   real Gemini clip scores 0.77 F1 against its Drugs.com reference but only
   0.59 against its Merriam-Webster reference -- a 0.9-point swing on this
   project's 0-5 scale depending on which single human recording you picked.
   Best-of-N avoids that.

## What this does NOT fix, and what's still open

- **The human reference itself is not guaranteed correct.** Checked
  directly this session: decoding the human reference audio's own phonemes
  and comparing to its dictionary IPA (USAN/DailyMed/MW/NCI, from another
  workstream's `references.jsonl`) found 2 of 13 spot-checked items
  (talquetamab, acoramidis) where the human recording plausibly diverges
  from the official pronunciation (talquetamab's reference audio appears to
  be missing the "l" sound entirely relative to USAN's `tælˈkwɛtæmæb`).
  Best-of-N reference scoring does not catch this -- it only helps when
  MULTIPLE references exist and at least one is trustworthy. A systematic
  audio-vs-dictionary-IPA consistency sweep across all 178 dual-coverage
  items has not yet been run; the runbook above should be treated as
  provisional until that gate is in place.
- **Whether F1-over-wavlm-embeddings is even sensitive to single-phoneme
  errors is unresolved.** A 13-item probe this session found phonetic-hint
  injection (SSML IPA, Gemini-with-IPA-in-text) never clearly beat naive
  plain-spelling synthesis on average (0.719-0.725 vs the baseline's 0.520)
  -- consistent with either the injection mechanism not working, or the F1
  metric itself not being fine-grained enough to reward it. Undecided; a
  deliberate minimal-pair test (same word, one phoneme deliberately wrong)
  would settle it and has not yet been run.
- **99/274 items (36%) have no human reference at all.** Best-of-N only
  helps the ~29% (79/274) with two references; it does nothing for the
  36% with zero. That gap is unresolved -- see the Path 3/4 discussion in
  this session (`runs/synthetic-reference-probe-v2.json`) for why naive
  synthetic-reference generation isn't a safe stand-in yet either.

## The hard subset: `runs/hard-subset-v1.json`

54 items (~20% of 274), built by `scripts/select_hard_subset.py` from data
already collected this session -- no tier is a guess:

| Tier | n | Source |
| --- | ---: | --- |
| A. confirmed-bad-by-ear | 3 | vyloy, adquey, voranigo -- a human listened and confirmed genuine mispronunciation |
| B. reference-flagged | 2 | talquetamab, acoramidis -- human reference clip disagrees with its own dictionary IPA; a bad score here may be the reference's fault |
| C. gemini-worst-quartile | 20 | bottom of the real 171-item gemini-2.5-flash-tts v4 run |
| D. structurally-hard-asr | 6 | Cloud STT can't transcribe even a human saying the name (`runs/reference-grounded-v1`) -- naming difficulty independent of any TTS |
| E. dual-reference-coverage | 7 | has both Drugs.com and Merriam-Webster clips -- exercises best-of-N scoring itself |
| F. ceiling-control | 8 | top of the same v4 run -- confirms a candidate CAN pass at all, so a model that fails everything doesn't look identical to one that only fails on hard items |
| G. no-reference-coverage | 8 | no human clip at all -- can't be scored by Path 2 today; tracked rather than silently dropped |

## Reading this correctly

- **This subset is deliberately not population-representative** (72% brand
  in the subset vs. 52% in the full 274 -- brand names dominate the actual
  worst-quartile pull, even though the real DOSE leaderboard shows generics
  as the harder category ON AVERAGE across all names; the two aren't
  contradictory -- most generics are fine because USAN naming conventions
  are phonetically regular by design, but the worst outliers skew brand,
  which is exactly the population this subset is built to surface). Do NOT
  report a pass-rate on this subset as if it were the model's overall DOSE
  score -- it is a fast regression/smoke check, not a leaderboard number.
- **Run the full 274-item corpus before reporting any official/comparable
  score.** The subset is for iterating quickly between candidate checkpoints
  or configs; a number meant to be compared against the real leaderboard's
  9 models needs the full corpus, scored the same way.
- Tier G items cannot be scored today -- a candidate run against them will
  come back `scoreable=False`, which is correct, not a bug to work around.
