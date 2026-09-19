# Path 2: Gemini Flash TTS vs. human pronunciation

Workstream 2, 2026-09-19. `speech-similarity-v4` (SpeechBERTScore F1,
wavlm-large, final layer, **forced-alignment span extraction**) run against
the real `gemini-2.5-flash-tts` candidate (274/274 synthesized, zero
synthesis failures).

**Superseded twice now -- read this version, not the v3 one.** The v3
report used Cloud STT word timestamps to locate the drug's audio span, with
three guardrails added after two confirmed contamination bugs (a padding
buffer bleeding into a neighboring word; Cloud STT returning one corrupted
5.1s timestamp for what should be one word). Even after those guardrails,
the user listened to the "fixed" esomeprazole clip and found it STILL
audibly cut into the tail of "start" -- Cloud STT's own timestamp can be
subtly, not just grossly, wrong, and no plausibility/retranscription check
catches that. This version replaces Cloud-STT-timestamp extraction entirely
with **forced alignment** (`dose_r/forced_align.py`): the phoneme CTC model
already used for Path 3 is run once per clip, and
`torchaudio.functional.forced_align` finds the alignment constrained to the
ALREADY-KNOWN sentence text, so there is no "recognize the right word" step
to get wrong at all. Validated on the 4 items the user specifically flagged
by ear (Eliquis, esomeprazole, Vyloy, talquetamab) plus a 250+-item
extraction-only sweep with zero misses/errors, before this full rescore was
run. The human-vs-human ceiling below was never affected by any of this (it
compares two already-isolated reference clips, no span extraction involved)
and is unchanged.

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

## Gemini Flash TTS, forced-alignment extraction (v4)

| | Gemini (v4, forced align) | Gemini (v3-fixed, Cloud STT) | Human ceiling |
| --- | ---: | ---: | ---: |
| mean | 3.18 | 2.91 | 3.79 |
| median | 3.21 | 2.91 | 3.83 |
| p10 | 2.55 | -- | -- |
| scoreable | 171/274 (62%) | 159/274 (58%) | -- |

**Both coverage and score went up.** Coverage recovered to the same 62% the
very first (buggy) run had -- but for a legitimate reason this time:
forced alignment can locate talquetamab (previously uncorrectably corrupted
by Cloud STT) and other items Cloud STT's word-recognition step simply
couldn't align at all, not because a contaminated span is being silently
accepted again. Gap to the human ceiling narrowed from ~0.9 to ~0.6 points.

**Where the 4 user-flagged items landed, before vs. after forced alignment:**

| Drug | v3-fixed score (rank) | v4 score (rank of 171) | What the user heard by ear |
| --- | ---: | ---: | --- |
| Vyloy | 1.59 (worst) | **1.41 (#1, still worst)** | "especially bad" -- matches |
| Eliquis | 1.72 (#2) | 2.45 (#13) | "kinda okay" -- matches (moved off worst-3) |
| esomeprazole | 1.76 (#3) | 2.67 (#30) | "kinda okay" -- matches (moved off worst-3) |
| talquetamab | unscoreable | 2.87 (#51) | "especially bad" -- **does NOT match**: user's ear says this should rank near the bottom, but it lands solidly mid-pack |

Three of four now track the user's own ear reasonably well. talquetamab is
a flagged, open discrepancy: either the F1 metric is insensitive to
whatever's actually wrong with that clip's pronunciation, or something
about that specific alignment/embedding pair is off in a way the aggregate
number doesn't show. Not resolved -- surfaced, not glossed over.

Brand: mean 3.15 (n=102). Generic: mean 3.22 (n=69).

Hardest (lowest 6): Vyloy (1.41), advair (1.71), voranigo (1.80), adquey
(2.01), toujeo (2.14), prademagene-zamikeracel (2.18).
Easiest (highest 6): Valium (4.28), Zantac (4.21), diazepam (4.17),
alprazolam (4.05), Prozac (4.04), Benadryl (4.04).

Full lowest-25 bucket (for manual verification): vyloy, advair, voranigo,
adquey, toujeo, prademagene-zamikeracel, aripiprazole, acoramidis, imaavy,
cariprazine, zevaskyn, ebglyss, eliquis, vabysmo, attruby, xolair,
winrevair, sitagliptin, vyvgart, aucatzyl, abilify, varenicline, tryngolza,
gepotidacin, nipocalimab-aahu.

## Reading this correctly

- **pass@4.0 is still ~0%** and still must not be read as "fails
  everything" -- the human ceiling itself (3.79) sits below that
  inherited, uncalibrated threshold. The meaningful number is the ~0.6-point
  gap to the human ceiling (down from ~0.9 under the old extraction).
- **talquetamab's mismatch with the user's own listening judgment is
  unresolved.** This is the clearest open question raised by this rescore,
  not a settled result -- worth investigating before trusting the metric's
  ranking of items the user hasn't personally checked.
- **Not every accepted span has been individually verified by ear.**
  Forced alignment removed the two confirmed Cloud-STT failure modes and
  passed a 250+-item extraction-only sweep with zero misses, which is a
  much stronger check than v3 ever had -- but it is still not proof every
  one of the 171 scored spans is clean.
- **Still one proxy of four**, not Workstream 1's hybrid judge. The 4 Cloud
  TTS tiers have not yet been scored with speech-similarity-v4 for a
  same-metric cross-system comparison.

## Artifacts

- `runs/gemini-flash-tts-v1-speech-similarity-v4/` -- results.jsonl
  (274 records, 171 scoreable), manifest.json, report.txt, summary.json.
- `runs/gemini-flash-tts-v1-speech-similarity-v3-fixed/` -- prior run, kept
  for the before/after comparison above, no longer the reference version.
- `runs/human-ceiling-v1.json` -- the 79-pair human ceiling computation.
- `dose_r/forced_align.py` -- the extraction method this run uses.
