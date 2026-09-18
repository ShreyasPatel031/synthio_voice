# Reference-audio grounding: separating ASR bias from real TTS failure

Workstream 2, 2026-09-18. Uses the human reference-pronunciation clips
Workstream 1 collected (`claude/sc-sandbox-gcp-access-arw4m8`,
`data/reference_audio/`) to correct the ASR round-trip scorer's biggest
known weakness.

## The problem this closes

The first pronunciation run (`runs/asr-roundtrip-v1-full`) compared each
synthesized clip's ASR transcript to the drug's **spelling**. That
conflates two failure modes it could not tell apart: the TTS system
mispronounced the name, or Cloud STT simply does not know the word
regardless of who says it. The example on record: "tofacitinib" came
back as "tofu Sydney" from **all four** systems, which reads far more
like an ASR gap than a shared mispronunciation, but the prior run had no
way to confirm that.

## Method

For every ingredient with a committed human reference clip, transcribe
the clip through the same Cloud STT pipeline used for synthesized audio.
That gives, per ingredient, an independent fact: can this recognizer
transcribe a **human** correctly saying this name at all? Then re-score
every synthesized clip against the reference's own transcript instead of
the spelled name (`dose_r.references.audio_grounded`).

Only Workstream 1's Drugs.com clips are used -- their Merriam-Webster and
UMich clips are gitignored on their branch as re-fetchable from public
URLs and are not committed, so not present here. 176/284 ingredients
(62%) have a usable clip; 176/176 transcribed cleanly after fixing a bug
found while running this (Cloud STT rejects stereo WAV as an error unless
`audioChannelCount` is declared; a handful of the Drugs.com clips are
stereo). 176 ingredients join to 169 of our 274 dataset rows.

## Finding: the ASR baseline, measured rather than assumed

**44.3% of reference clips (78/176) are recognized correctly when a
real human says the name.** That is close to the raw pass rates the
spelling-based scorer reported for every TTS system (44-49%) -- which is
the first direct evidence, not an inference, that a large share of that
run's "failures" were the recognizer's limits, not the model's.

Splitting synthesized-clip scores by whether their ingredient's reference
was itself ASR-recognizable makes this stark:

| Reference recognizable? | n (synth clips) | pass rate vs. reference |
| --- | ---: | ---: |
| yes (78 ingredients) | 305 | **85.2%** |
| no (98 ingredients) | 364 | 6.3% |

That is close to bimodal. On names the recognizer can hear at all, TTS
systems pass the great majority of the time. On names it cannot -- for
anyone, human included -- almost nothing passes, which is exactly what
"the recognizer doesn't know this word" predicts and "the TTS mispronounces
it" does not.

## Corrected pass rates

Restricting to the 77 ingredients (of the 169 overlapping our dataset)
whose reference is itself ASR-recognizable:

| system | n | pass % vs. reference | (for comparison: spelling-based, same subset) |
| --- | ---: | ---: | ---: |
| gtts-standard-c | 77 | 87.0% | 62.1% |
| gtts-wavenet-c | 77 | 87.0% | 63.3% |
| gtts-neural2-c | 77 | 85.7% | 62.1% |
| gtts-chirp3hd-achernar | 74 | 81.1% | 59.3% |

These land in DOSE's published range (63.1-91.2%) far more plausibly than
the earlier 44-49%. That is not a claim the correction is complete --
only that the previous number was dominated by a confound this now
measures directly instead of guessing at.

## What this does not fix

- **Coverage is 62% of ingredients, 169/274 rows**, and further
  restricted to 77 once the ASR-recognizable filter is applied. The
  remaining ~40% (dominated by the same recently-approved, no-public-source
  names flagged in the reference-layer audit) still has no grounding at
  all -- exactly where DOSE's headline finding lives, so this correction
  is weakest where the benchmark's most important result is.
- **6.3%, not 0%.** A handful of synthesized clips pass even when the
  human reference did not get recognized, and 85.2% is not 100% on the
  "easy for ASR" side either -- there is still real pass/fail variance
  inside each bucket, some of which is presumably genuine TTS quality.
- **The Standard/WaveNet/Neural2 voice-identity collision documented in
  `docs/FINDING_voice_identity_instability.md` still applies here.** These
  corrected numbers do not create new cross-tier comparisons that weren't
  already contaminated; only the Chirp3-HD/legacy split is trustworthy as
  a system comparison.
- **Still not Workstream 1's hybrid judge.** This is a second, independent,
  audio-grounded proxy -- valuable because it needs no phoneme reference
  layer and is now measured rather than assumed, not because it replaces
  the real judge once delivered.

## A pipeline strength worth noting

"Alyftrek" was misheard by ASR as "a lift trick" -- for the reference
clip itself, a real human speaking clearly. The word-boundary-insensitive
comparison (`score_pronunciation` collapses whitespace before the
phonetic check) correctly scored the synthesized clip as a match once it
was also transcribed as "a lift trick," rather than penalizing both for
an ASR word-splitting quirk that has nothing to do with pronunciation
quality.

## Artifacts

- `dose_r/references/reference_clips.py`, `audio_grounded.py` -- the
  grounding logic, offline-tested (10 new tests, 53/53 total passing).
- `scripts/score_against_reference_audio.py` -- resumable CLI; reuses the
  already-computed synth transcripts from `runs/asr-roundtrip-v1-full`,
  no re-synthesis.
- `runs/reference-grounded-v1/` -- `reference_transcripts.jsonl` (176
  entries), `results.jsonl` (1096 records, `grounded` field added),
  `summary.json`.
- `data/reference_audio/` -- pulled from Workstream 1's branch
  (`claude/sc-sandbox-gcp-access-arw4m8`) as of commit `260b5ac`: the
  176 committed Drugs.com WAVs and the manifest. Not modified here.
