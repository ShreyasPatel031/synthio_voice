# Path 2: Gemini Flash TTS vs. human pronunciation

Workstream 2, 2026-09-19. `speech-similarity-v4` (SpeechBERTScore F1,
wavlm-large, final layer, **forced-alignment span extraction, gap-capped**)
run against the real `gemini-2.5-flash-tts` candidate (274/274 synthesized,
zero synthesis failures).

**This is the third and current version of this report.** History, each
superseded by the next after the user caught a real bug by listening:

1. **v3 (Cloud STT timestamps).** Two contamination bugs found and
   "fixed" with guardrails, but a "fixed" esomeprazole clip still audibly
   bled into the preceding word -- Cloud STT's own timestamp can be subtly,
   not just grossly, wrong. Replaced entirely.
2. **v4, first pass (forced alignment, uncapped midpoint split).**
   Alignment against the KNOWN sentence text (`dose_r/forced_align.py`)
   removed the Cloud-STT failure modes, but "Advair" came out audibly
   truncated (0.4s) -- CTC posteriors are "peaky" (a phoneme spikes for
   1-3 frames and defaults to blank elsewhere, even during real speech), so
   the raw labeled span undershoots. Fixed by splitting each blank gap to
   its midpoint.
3. **v4, final (gap-capped).** The uncapped midpoint fix then over-extended
   on "quetiapine" (2.40s span, ~3x the corpus norm) -- confirmed by
   per-word boundary inspection that its ENTIRE sentence has large
   (0.4-1.0s) real pauses between every word (a genuinely slow, deliberate
   Gemini rendering, not a glitch -- confirmed by the user listening to the
   full clip), and splitting a real pause in half still pulls silence into
   the span. Capped recoverable half-gap at 0.15s (genuine CTC undershoot,
   confirmed on Advair, was only ~0.2-0.3s): Advair stays fixed (0.590s,
   unaffected by the cap), quetiapine drops back to 1.921s (in line with
   similar-length words). A full 274-item duration/phoneme-count sweep after
   this fix showed no remaining outliers.

The human-vs-human ceiling was never affected by any of this (it compares
two already-isolated reference clips, no span extraction involved).

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

## Gemini Flash TTS, final (v4, gap-capped forced alignment)

| | Gemini (v4 final) | Gemini (v4, uncapped) | Gemini (v3, Cloud STT) | Human ceiling |
| --- | ---: | ---: | ---: | ---: |
| mean | 3.30 | 3.18 | 2.91 | 3.79 |
| median | 3.33 | 3.21 | 2.91 | 3.83 |
| p10 | 2.69 | 2.55 | -- | -- |
| scoreable | 171/274 (62%) | 171/274 (62%) | 159/274 (58%) | -- |

Gap to the human ceiling narrowed from ~0.9 (v3) to ~0.6 (v4 uncapped) to
~0.49 (v4 final) as each extraction bug was fixed.

**Where the flagged items landed, final:**

| Drug | Rank (of 171) | Score | User's verdict by ear |
| --- | ---: | ---: | --- |
| Vyloy | #1 (worst) | 1.59 | "especially bad" -- **matches** |
| adquey | #2 | 1.86 | "very obviously wrong" -- **matches** |
| voranigo | #3 | 2.18 | "correctly very badly said" -- **matches** |
| aripiprazole | #4 | 2.29 | "actually okay, different pronunciation" -- **does NOT match** |
| acoramidis | #5 | 2.32 | "actually okay, different pronunciation" -- **does NOT match** |
| Eliquis | #13 | 2.60 | "kinda okay" -- roughly matches (well off the worst) |
| Advair | #17 | 2.69 | confirmed correct extraction after the truncation fix; not separately judged for pronunciation quality |
| esomeprazole | #28 | 2.82 | "kinda okay" -- matches |
| talquetamab | #37 | 2.92 | "especially bad" -- **does NOT match** |
| quetiapine | #82 (mid-pack) | 3.31 | confirmed a real (if unusually slow) sentence, not a glitch, by listening to the full clip |

**Two open, unresolved discrepancies, not glossed over:**
- **aripiprazole and acoramidis score as if mispronounced (top-5 worst)
  when the user's own ear says they're just different, acceptable
  pronunciation variants.** This is evidence the F1-over-wavlm metric
  cannot yet distinguish "wrong" from "different but valid" -- a real
  calibration gap, not an extraction issue (their spans were independently
  reviewed above and are not currently suspected of contamination).
- **talquetamab still doesn't rank near the bottom** despite being called
  "especially bad" by ear, on either the buggy or the fully-fixed
  extraction. Whatever's wrong with that clip's pronunciation, this metric
  isn't picking it up.

Highest 10: Valium (4.29), Zantac (4.21), loratadine (4.17), fluoxetine
(4.12), diazepam (4.12), Prozac (4.09), Entresto (4.09), Zoloft (4.06),
Otezla (4.05), alprazolam (4.05).

## Reading this correctly

- **pass@4.0 is still ~0%** and still must not be read as "fails
  everything" -- the human ceiling itself (3.79) sits below that
  inherited, uncalibrated threshold.
- **This metric conflates "wrong" with "different but valid" at least
  twice (aripiprazole, acoramidis) and misses at least one genuine problem
  (talquetamab).** Before trusting this metric's ranking for items the user
  hasn't personally checked, that calibration gap needs to be taken
  seriously -- it is the main open finding of this whole exercise, not a
  footnote.
- **Coverage is still only 62% (171/274).** The other 103 items have no
  human reference clip at all, and 99 of those 103 have no dictionary
  respelling either -- there is no ground-truth pronunciation source of any
  kind for 99/274 (36%) of the benchmark. Path 3/4 need to address this
  gap; it is not solved by anything in this report.
- **Still one proxy of four**, not Workstream 1's hybrid judge. The 4 Cloud
  TTS tiers have not yet been scored with speech-similarity-v4 for a
  same-metric cross-system comparison.

## Artifacts

- `runs/gemini-flash-tts-v1-speech-similarity-v4/` -- results.jsonl
  (274 records, 171 scoreable), manifest.json, report.txt, summary.json.
- `runs/gemini-flash-tts-v1-speech-similarity-v3-fixed/` -- prior run, kept
  for the before/after comparison, no longer the reference version.
- `runs/human-ceiling-v1.json` -- the 79-pair human ceiling computation.
- `runs/forced-align-durations-v1.json` -- the post-fix duration/phoneme
  outlier sweep (274 items, no remaining outliers).
- `dose_r/forced_align.py` -- the extraction method this run uses.
