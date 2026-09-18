# Path 2: Gemini Flash TTS vs. human pronunciation

Workstream 2, 2026-09-18. `speech-similarity-v3` (SpeechBERTScore F1,
wavlm-large, final layer) run against the real `gemini-2.5-flash-tts`
candidate (274/274 synthesized, zero synthesis failures).

**Superseded once already: the first version of this report was built on a
broken drug-name span extraction.** The user listened to several extracted
clips and caught two real contamination bugs (fixed in commit 97bb2e9,
`dose_r/audio_span.py`): a fixed padding buffer bleeding into a neighboring
word ("esomeprazole" audibly included the tail of "start"), and Cloud STT
itself returning a single corrupted timestamp spanning 5.1 seconds for what
should have been one word ("talquetamab"). This version is re-scored with
both fixed. The human-vs-human ceiling below was never affected by this bug
(it compares two already-isolated reference clips, no span extraction
involved) and is unchanged from the first report.

## The human ceiling

Computed across all 79 ingredients with both a Drugs.com and a
Merriam-Webster human recording, scored against each other with the
identical F1 pipeline used on Gemini (`scripts/compute_human_ceiling.py`):

| | F1 | 0-5 scale |
| --- | ---: | ---: |
| mean | 0.758 | 3.79 |
| median | 0.766 | 3.83 |

Even two humans saying the same word correctly only reach ~3.8/5 under this
metric -- read every other number below relative to 3.8, not to 5.0.

## Gemini Flash TTS, corrected

| | Gemini (corrected) | Gemini (first, invalidated) | Human ceiling |
| --- | ---: | ---: | ---: |
| mean | 2.91 | 2.90 | 3.79 |
| median | 2.91 | 2.92 | 3.83 |
| scoreable | 159/274 (58%) | 171/274 (62%) | -- |

**The aggregate barely moved despite two confirmed bugs.** Worth being
straightforward about why: the bugs did not uniformly inflate scores. All
three items the user specifically listened to and flagged got WORSE, not
better, once the contamination was removed:

| Drug | Before (contaminated) | After (fixed) |
| --- | ---: | ---: |
| Eliquis | 1.86 | 1.72 |
| esomeprazole | 1.94 | 1.76 |
| Vyloy | 1.82 | 1.59 |
| talquetamab | 1.96 | **unscoreable** (correctly rejected) |

So the fix mattered a great deal for whether any INDIVIDUAL item's score (or
whether it should even have a score at all) can be trusted, even though the
aggregate mean happened to land in nearly the same place. Treat this report's
per-item numbers as trustworthy; do not assume the same was true of the
first version's.

**Coverage dropped from 62% to 58%.** 12 items that were previously scored
on a contaminated span now correctly come back `scoreable=False` ("span
extraction failed") instead of a silently-wrong number -- fewer, honest
results rather than more, wrong ones. Combined with the pre-existing 103
items with no reference clip at all, 115/274 are unscored in this run.

Brand: mean 2.95 (n=101, was 102 -- one moved to unscoreable).
Generic: mean 2.83 (n=58, was 69 -- eleven moved to unscoreable).

Hardest: Vyloy (1.59), Eliquis (1.72), esomeprazole (1.76), apremilast
(2.01), Voranigo (2.01), cariprazine (2.06).
Easiest: Valium (4.32), Zantac (3.94), alprazolam (3.70), Motrin (3.64),
Tylenol (3.63), rivaroxaban (3.60).

## Reading this correctly

- **pass@4.0 is still ~0%** and still must not be read as "fails
  everything" -- the human ceiling itself (3.79) sits below that
  inherited, uncalibrated threshold. The meaningful number is the ~0.9-point
  gap to the human ceiling.
- **Span extraction can still fail silently in ways not yet caught.** The
  two fixed bugs were found by a human listening to specific clips, not by
  an automated sweep of all 274 items' extracted audio. The duration and
  re-transcription checks are guardrails against the two confirmed failure
  modes, not a proof that every accepted span is clean -- spot-checking a
  further sample before leaning heavily on this data is warranted.
- **Still one proxy of four**, not Workstream 1's hybrid judge. The 4 Cloud
  TTS tiers have not yet been scored with speech-similarity-v3 for a
  same-metric cross-system comparison.

## Artifacts

- `runs/gemini-flash-tts-v1-speech-similarity-v3-fixed/` -- results.jsonl
  (274 records, 159 scoreable), manifest.json, report.txt, summary.json.
- `runs/human-ceiling-v1.json` -- the 79-pair human ceiling computation.
