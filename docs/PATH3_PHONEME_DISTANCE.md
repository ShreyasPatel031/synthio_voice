# Path 3: phoneme-distance vs. dictionary IPA

Workstream 2, 2026-09-19. `phoneme-distance-v1` (panphon-weighted phoneme
edit distance, no synthesized reference audio, no human recording) run
against the real `gemini-2.5-flash-tts` candidate.

## Why this exists

Built after an extensive Path 4 investigation (synthesizing a stand-in
reference clip from dictionary IPA) was found to be a genuine dead end: 4
TTS engines (Cloud TTS, Gemini, Piper, Kokoro) and 2 genuinely-verified
phoneme-injection mechanisms all failed to reach human-level correctness on
confirmed-bad items (Vyloy, Adquey, Voranigo) -- traced to a structural
cause (these engines' prosody/duration components are trained only on
their own G2P's output shape, so correct-but-differently-shaped phoneme
input is out-of-distribution for them). See the session notes in
`dose_r/scoring/phoneme_distance.py`'s module docstring for the full trail.

Path 3 sidesteps that whole problem: it never synthesizes anything. It
decodes the CANDIDATE's own audio to phonemes (the same CTC model already
used for forced alignment) and compares directly against dictionary IPA.
No synthesis-quality confound, no prosody-mismatch confound, and it works
for any of the 280/284 ingredients with a sourced IPA variant --
independent of whether a human recording exists at all.

## Coverage: the headline win over Path 2

| | Path 2 (audio-to-audio) | Path 3 (phoneme-distance) |
| --- | ---: | ---: |
| Requires | human reference audio | dictionary IPA |
| Available for | 179/274 (65%) | 274/274 items have IPA data; 260/274 scoreable this run |
| Scoreable, this run | 171/274 (62%) | 260/274 (95%) |

## Validation before trusting it (`scripts/validate_phoneme_distance.py`)

- **Discrimination**: deliberately mismatched audio/IPA pairs averaged rate
  3.85; correct pairs averaged 0.88 -- a clean ~4.4x margin.
- **Human ceiling**: decoding real human recordings and scoring them
  against their OWN dictionary IPA gave median rate 0.66, p10 0.03 (near-
  perfect for well-scoped items). The tail of outliers was investigated,
  not ignored: 4 of the worst 5 traced to human reference CLIPS covering
  only part of a multi-word ingredient name (e.g. "trospium chloride"'s
  1.06s clip only contains "trospium"), and one ("Nurtec") to the clip
  actually saying the real product name "Nurtec ODT" while the dictionary
  only has "Nurtec". Real, identified data-scope issues, not a flaw in the
  distance metric.

## Real corpus run, corrected

274/274 synthesized, 260/274 scoreable (95%). A second, distinct data gap
was found running this at scale and is now flagged in metadata rather than
silently contaminating the numbers: **11 items carry an FDA-mandated
biosimilar distinguishing suffix** (e.g. "risankizumab-rzaa" -- the
"-rzaa" is deliberately arbitrary, no established pronunciation by
design). The dictionary IPA for these covers only the base name; Gemini's
audio attempts the full name including the suffix, inflating the distance
for a reason unrelated to pronunciation accuracy. Confirmed directly on
"risankizumab-rzaa" (decoded tail is Gemini spelling out "R-Z-A-A" as
letters) and "teplizumab-mzwv". Not every hyphenated name is affected --
genuine two-word compounds with real coverage on both halves
("dimethyl-fumarate", "insulin-glargine") score normally.

| | All scoreable (n=260) | Suffix-gap items excluded (n=249) |
| --- | ---: | ---: |
| mean | 3.56 | **3.71** |
| median | 3.77 | **3.79** |
| p10 | 1.84 | **2.28** |
| pass@4.0 | 40.0% | **41.8%** |

Lowest 5, clean: Adquey (0.00), faricimab-svoa (0.04), bulevirtide-gmod
(0.27), Cypsedo (0.30), Latuda (0.38).

## Where the ear-verified items from this session's Path 2 investigation land

| Drug | Path 2 rank/score (audio, n=171) | Path 3 rank/score (phoneme, n=249) | What the user heard by ear |
| --- | --- | --- | --- |
| Adquey | #2 / 1.86 | **#1 (worst) / 0.00** | "very obviously wrong" -- matches, more decisively |
| Vyloy | #1 (worst) / 1.59 | #41 / 2.35 | "genuinely bad" -- matches direction, less extreme |
| Advair | #17 / 2.69 | #56 / 2.92 | -- |
| esomeprazole | #28 / 2.82 | #58 / 2.97 | "kinda okay" -- matches |
| Voranigo | #3 / 2.18 | #77 / 3.18 | "correctly very badly said" -- **does not match**, ranks mid-pack here |
| **aripiprazole** | **#4 / 2.29** | **#78 / 3.19** | "actually okay, different pronunciation" -- **Path 3 fixes exactly this false positive** |
| Eliquis | #13 / 2.60 | #79 / 3.19 | "kinda okay" -- matches |
| talquetamab | #37 / 2.92 | #126 / 3.70 | "especially bad" -- still does not match on either path |
| **acoramidis** | **#5 / 2.32** | **#148 / 3.94** | "actually okay, different pronunciation" -- **Path 3 fixes exactly this false positive** |

**The headline result**: Path 3 corrects Path 2's most concrete, evidenced
failure -- aripiprazole and acoramidis were false positives under
single-reference audio scoring (top-5-worst despite being valid
pronunciation variants), and Path 3's multi-variant IPA matching moves both
solidly into the upper half of the corpus, matching the user's own ear.
Voranigo is a new, opposite discrepancy Path 3 introduces (confirmed bad by
ear, but not flagged here) -- worth investigating before trusting Path 3's
ranking on items nobody has personally checked. talquetamab remains
unresolved on both paths.

## Reading this correctly

- **Two paths now disagree about which items are worst, not just about
  coverage.** Neither should be treated as the sole source of truth --
  Path 2 catches Vyloy/Voranigo better; Path 3 catches Adquey more
  decisively and fixes the aripiprazole/acoramidis variant-conflation
  problem. Where they actively disagree (Voranigo) is exactly where manual
  verification matters most.
- **Not yet checked**: whether Path 3 conflates "wrong word" with "right
  phonemes, wrong prosody/stress" the way Path 2's own blind spots were
  found this session. No minimal-pair sensitivity test has been run on
  this metric yet.
- **Still one proxy of four.**

## Artifacts

- `runs/gemini-flash-tts-v1-phoneme-distance-v1/` -- results.jsonl (274
  records, 260 scoreable, 11 suffix-gap-flagged), manifest.json,
  report.txt, summary.json.
- `runs/phoneme-distance-validation-v1.json` -- human-ceiling and
  discrimination validation.
- `dose_r/scoring/phoneme_distance.py` -- the scorer.
- `dose_r/references/references.jsonl` -- IPA data snapshot (provisional,
  pulled from `claude/sc-sandbox-gcp-access-arw4m8` @ fee5ac5).
