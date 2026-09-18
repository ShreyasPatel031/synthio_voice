# Path 2: Gemini Flash TTS vs. human pronunciation

Workstream 2, 2026-09-18. `speech-similarity-v3` (SpeechBERTScore F1,
wavlm-large, final layer) run against the real `gemini-2.5-flash-tts`
candidate (274/274 synthesized, zero failures) and compared to a
human-vs-human ceiling computed the same way.

## The human ceiling (the actual baseline "how well can this metric ever score")

Computed across all 79 ingredients with both a Drugs.com and a
Merriam-Webster human recording -- two different real people saying the
same drug name -- scored against each other with the identical F1
pipeline used on Gemini:

| | F1 | 0-5 scale |
| --- | ---: | ---: |
| mean | 0.758 | 3.79 |
| median | 0.766 | 3.83 |
| p10-p90 | 0.680-0.834 | 3.40-4.17 |

This is the number that matters most for interpreting anything else: **even
two humans saying the same word correctly only reach ~3.8/5 under this
metric**, not 5.0. The metric's own ceiling is well below "perfect," so any
comparison must be read relative to ~3.8, not relative to 5.0.

## Gemini Flash TTS

| | Gemini | Human ceiling | Gap |
| --- | ---: | ---: | ---: |
| mean | 2.90 | 3.79 | -0.89 |
| median | 2.92 | 3.83 | -0.91 |
| p10 | 2.32 | 3.40 | -1.08 |
| p90 | 3.47 | 4.17 | -0.70 |

Brand: mean 2.93 (n=102). Generic: mean 2.86 (n=69) -- a real but modest
gap, smaller than the brand/generic gaps seen in earlier ASR-based scoring,
consistent with this metric measuring something different (acoustic content
match, not word-recognition success).

**Coverage: 171/274 (62%) scoreable.** The other 103 have no committed
human reference clip -- concentrated in exactly the newest, hardest,
most clinically important names (Zaynich, Sonrotoclax, Retatrutide,
Casgevy, Orzeyful, and similar 2024-26 approvals). This is the same
coverage ceiling documented in `REFERENCE_AUDIO_GROUNDING.md` and applies
here for the same reason: those names have no public pronunciation source
at all yet, not just no source we pulled.

Hardest (lowest score): Vyloy (1.82), Eliquis (1.86), esomeprazole (1.94),
talquetamab (1.96), acoramidis (2.04).
Easiest (highest score): Valium (3.97), Januvia (3.71), Tylenol (3.67),
loratadine (3.67), Zantac (3.64).

## Reading this correctly -- what this does and does not show

- **Do not read "pass@4.0 = 0%" as "Gemini fails everything."** The
  PASS_THRESHOLD=4.0 inherited from this project's other scorers assumes a
  0-5 scale where a good result clears 4.0. Under this specific metric, even
  a real human doesn't clear 4.0 on average (3.79). The threshold is
  uncalibrated for this scorer (documented in the module docstring) -- the
  meaningful comparison is Gemini's score relative to the human ceiling
  (~2.90 vs. ~3.79), not relative to an arbitrary bar neither humans nor
  models reliably clear.
- **The ~0.9-point gap is real, but this is one proxy, not a verdict.**
  Path 2 measures acoustic-content overlap with a specific reference
  recording (F1, chosen for truncation sensitivity -- see
  `speech_similarity.py`'s "Correction history"). It does not know what a
  "close enough" pronunciation is linguistically, cannot distinguish a
  genuinely wrong phoneme from an accepted regional variant, and inherits
  the reference-recording-pace sensitivity documented in that same module
  (F1 moves ~24% just from which human reference happens to be picked).
- **This is not compared against the other TTS systems yet in this
  document** -- the 4 Cloud TTS tiers (Standard/WaveNet/Neural2/Chirp3-HD)
  have not been scored with speech-similarity-v3 in this run. That
  comparison is a natural next step, not done here.
- **Not Workstream 1's hybrid judge, and not validated against human
  pronunciation-quality ratings** -- this is Path 2 of the project's
  four-path plan (`EVALUATION_PATHS_PLAN.md`), one proxy among several,
  reported on its own terms.

## Artifacts

- `runs/gemini-flash-tts-v1-speech-similarity-v3/` -- results.jsonl (274
  records, 171 scoreable), manifest.json, report.txt, summary.json.
- Human ceiling computed in-line (not yet saved as a standing script);
  raw per-ingredient F1 values available on request.
