# DOSE-R pronunciation evaluation: the four-path plan

Decided 2026-09-18 (Workstream 2). This is the handoff version of a planning
conversation -- written so a separate agent can pick up Path 3 research and
Path 4 without re-deriving context.

## Why four paths, not one judge

No single scorer in this repo is trustworthy alone:
- Word-level ASR (`asr_roundtrip.py`) cannot transcribe a coined drug name
  correctly even from a human, 44.3% of the time (`REFERENCE_AUDIO_GROUNDING.md`).
- An audio-LLM panel (`llm_panel.py`) works, but is a black box, can drift,
  and has a self-preference risk when judging Gemini-family TTS output.
- Nothing yet gives a deterministic, auditable, LLM-independent signal.

The plan is four independent, disagreeing paths, each reported separately,
never averaged into one number. Agreement between paths is itself the
evidence; disagreement flags exactly the items that need a closer look.

## The four paths

| # | Name | Mechanism | Needs a human reference clip? | Status |
|---|---|---|---|---|
| 1 | Audio-LLM-as-judge | audio + expected name -> LLM -> score | No | **Built, validated, committed.** `dose_r/scoring/llm_panel.py`, single judge `gemini-2.5-flash` (flash-lite disqualified -- see commit f549ebb, it hallucinates hearing whatever name it's told regardless of audio). |
| 2 | Audio-to-audio similarity | SpeechBERTScore-style: compare SSL embeddings of candidate audio vs. human reference audio, NOT raw waveform/pitch | **Yes** (~62% coverage, Drugs.com clips) | **In progress now** (this session). `dose_r/scoring/speech_similarity.py` built; validation running. |
| 3 | Audio->phoneme->edit-distance | Decode both clips to phoneme strings, compare with a phoneme-aware distance | No (but validated against human audio where available) | **Research done, not built.** See below -- handed off. |
| 4 | Phonetic-reference->audio->audio-to-audio | Render the *correct* pronunciation as audio (from WS1's IPA reference), then reuse Path 2's metric | No (extends coverage past the 62%) | **Blocked on Path 2 passing validation**, plus its own extra validation step (see below). |

## Path 1 -- done

`AudioLLMPanelScorer`, single-judge mode with `gemini-2.5-flash`. Validated by
hand: 5/5 correct pairs scored 5.0 with accurate "heard" text; 3/3 deliberately
mismatched pairs scored 0.0 correctly. `gemini-2.5-flash-lite` was tested and
rejected -- it hallucinated hearing the *labeled* drug name regardless of what
the audio actually contained (evidence in commit f549ebb). Full-corpus scoring
against all 5 systems (4 Cloud TTS tiers + the real `gemini-2.5-flash-tts`
candidate, now fully synthesized, 274/274, zero failures) is queued as Task #9.

## Path 2 -- in progress

`dose_r/scoring/speech_similarity.py`: extracts frame-level hidden states from
`facebook/wav2vec2-base` (plain self-supervised encoder, not ASR/phoneme
fine-tuned) for both clips, computes BERTScore-style greedy cosine-similarity
precision/recall/F1 between the two frame sequences, maps F1 linearly to 0-5.

Three checks must pass before this is trusted at any scale:
1. **Discrimination**: correct TTS-vs-reference pairs score high; deliberately
   mismatched pairs score low.
2. **Voice-invariance** (the check that actually answers the concern this path
   exists to address): two *different humans* saying the *same correct word*
   must still score high. Validated using paired Drugs.com + Merriam-Webster
   clips for the 79 ingredients where both exist locally (8 pulled for this
   check via MW's public per-clip URL, already recorded in
   `data/reference_audio/manifest.jsonl`).
3. **Naive-baseline comparison**: the same pairs scored with plain MFCC+DTW
   (raw acoustic distance), to show concretely -- not just argue -- that the
   naive method conflates voice identity with pronunciation and the SSL-based
   method does not.

Validation script: not yet committed (was mid-run when this doc was written).
Results to follow in the next commit.

## Path 3 -- researched, NOT built (handed off)

Do not build this until Path 2's validation is in and reviewed -- per
explicit instruction, Path 3 was deprioritized because we had no way yet to
check the phoneme-conversion model's own reliability, and building it before
that checking mechanism exists risks the same trap Path 2's validation is
designed to avoid.

### What already exists in this repo (paused, from an earlier session)
- `dose_r/scoring/phoneme_model.py` -- wraps `facebook/wav2vec2-lv-60-espeak-cv-ft`
  (a CTC model outputting ~392 espeak/IPA phoneme symbols, no English
  dictionary to fall back on). Confirmed working manually on one clip.
- `dose_r/scoring/phoneme_scorer.py` -- pure Levenshtein-based Phoneme Error
  Rate (PER), mapped to 0-5. **This is the naive version of the metric** --
  see below for what should replace or supplement it.
- 13 offline tests already passing for the pure comparison logic.
- Both modules are unregistered (not wired into any scorer registry) and
  untested against real audio at scale -- exactly the state Path 3's
  validation step is meant to resolve before either is trusted.

### The actual research findings (2026-09-18, via WebSearch)

**Phoneme-to-phoneme comparison metrics**, ranked by sophistication:

| Metric | What it is | What it fixes over plain edit distance |
|---|---|---|
| Plain Levenshtein / PER (already built) | Unweighted insert/delete/substitute edit distance | Baseline. Treats a schwa-vowel mixup the same as swapping an entire consonant -- flagged as a real weakness in `phoneme_scorer.py`'s own docstring. |
| **PanPhon-weighted edit distance** (Mortensen et al., ACL 2016; actively maintained; `pip install panphon`) | Maps every IPA symbol to ~21-24 articulatory feature vectors (voicing, place, manner, nasality, etc.); substitution cost = Hamming distance between feature vectors, not a flat 1 | Directly fixes the schwa-vs-consonant problem: acoustically-close substitutions cost less than acoustically-distant ones. |
| **Phonologically-Weighted Levenshtein Distance (PWLD)** | Same articulatory-feature-weighting idea, specifically framed for intelligibility prediction | Purpose-built for "how bad is this mispronunciation," not just transcription accuracy. |

Recommendation for whoever picks this up: **replace or supplement the current
plain-PER `phoneme_scorer.py` with a PanPhon-weighted version** before trusting
it -- the plain version is a known-weaker baseline, not the actual proposed
final metric.

**Audio-to-phoneme model candidates**, for cross-checking (do not trust one
model alone -- the whole point of Path 3's validation is catching a model that
produces confident garbage):

| Model | Type | Note |
|---|---|---|
| `wav2vec2-lv-60-espeak-cv-ft` (already downloaded) | CTC, wav2vec2 lineage | ~392-symbol espeak vocabulary |
| **Allosaurus** (`pip install allosaurus`, github.com/xinjli/allosaurus) | Different architecture -- allophone-mapping layer over a shared multilingual phone recognizer, 2000+ languages | Genuinely independent model lineage from wav2vec2; good for catching a failure mode specific to one family |
| **Charsiu / Montreal Forced Aligner** | Forced *alignment*, not blind decoding -- given audio AND the already-known expected phoneme sequence (which we have, from WS1's reference layer), aligns and scores per-phone confidence | Closer to the textbook Goodness-of-Pronunciation (GOP) approach than blind-decode-then-diff; may be more principled for this project's specific situation (we usually DO know the target) |
| ZIPA, BranchShine (2025-26 papers) | Newest research | Flagged for completeness; easy off-the-shelf (HuggingFace/pip) availability not yet verified |

### Path 3's own validation step (mirrors Path 2's discipline)

Before trusting ANY audio-to-phoneme model's output: run it on the human
reference clips (the ~62% Drugs.com coverage) and compare the resulting
phoneme string against WS1's gold IPA reference for the same ingredient
(`dose_r/references/references.jsonl` on their branch, confidence-tiered
high/medium/low -- restrict this check to `high`/`medium` entries only, since
`low` entries are themselves unverified guesses and would make this a
circular check). If the audio-to-phoneme model doesn't reproduce a
known-correct reference from real human audio, it is not ready to be trusted
on synthesized candidate audio either, and should be reported as such rather
than used.

Cross-check at least two of the candidate models against each other on the
same slice -- agreement between two independently-built model lineages
(wav2vec2-derived vs. Allosaurus, say) is much stronger evidence than either
one's confidence alone.

## Path 4 -- blocked on Path 2

Mechanism: take WS1's IPA/ARPAbet reference for a drug (available for all 284
ingredients, confidence-tiered), render it into audio via Cloud TTS's SSML
`<phoneme>` tag (already-integrated adapter, more controllable than raw
espeak since it takes the exact phoneme string rather than re-deriving it from
text), then score that synthetic "canonical" clip against the real candidate
clip using Path 2's audio-to-audio metric.

Two things must both be true before Path 4 is trusted, not just Path 2's own
validation:
1. Path 2 passes its own three checks (discrimination, voice-invariance,
   beats the naive baseline).
2. **New check specific to Path 4**: does Path 2's metric score a
   phoneme-rendered synthetic reference as "close" to a REAL human reference
   for the same word? If even a *correct* synthetic rendering doesn't match a
   human closely under Path 2's metric, Path 4 inherits a bias Path 2's own
   validation never tested for (a synthetic voice is a much bigger acoustic
   departure from "human" than two different humans are from each other), and
   Path 4 cannot be trusted regardless of how well Path 2 did on human-vs-human
   pairs alone.

Only build this once both hold.

## Reporting discipline (applies to every path)

- Every path's numbers ship with its own caveats section, not folded into a
  single headline pass rate.
- Self-preference bias (Path 1) and voice-identity contamination among
  Standard/WaveNet/Neural2 (`FINDING_voice_identity_instability.md`) apply
  regardless of which path produced a given number -- restate them, don't
  assume the reader remembers.
- None of these four paths is Workstream 1's hybrid judge. All are proxies
  built to get real numbers on the board while that judge and its
  human-verified anchor set remain outstanding.
