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
| aripiprazole | #4 / 2.29 | #78 / 3.19 | Path 2 is the score that counts. Path 3 scoring this higher is a Path 3 miss, not a correction of Path 2. |
| Eliquis | #13 / 2.60 | #79 / 3.19 | "kinda okay" -- matches |
| talquetamab | #37 / 2.92 | #126 / 3.70 | "especially bad" -- Path 3 still too high |
| acoramidis | #5 / 2.32 | #148 / 3.94 | Path 2 is the score that counts. Path 3 scoring this higher is a Path 3 miss, not a correction of Path 2. |

**Path 2 is ground truth.** It was re-confirmed against the user's ear
across this session. Path 3 is judged against Path 2. Where they disagree,
Path 3 is wrong. That includes Voranigo (Path 2 2.18, near-worst, confirmed
bad by ear; Path 3 3.18, mid-pack) and also aripiprazole and acoramidis:
scoring them higher than Path 2 is not "fixing a false positive."

## Why Path 3 does not match Path 2

The interrupted check was Goodness of Pronunciation. The original Gemini
WAVs are gitignored and were not in this checkout, so GOP could not be
finished on that exact audio. It was finished on a fresh Kore synthesis of
the same sentences, with Path 2 recomputed on those same clips so the
comparison is paired. Path 2 on that new Voranigo clip was 3.33 (fine);
GOP still scored it like a bad item. On the 9 ear-checked names, GOP vs
Path 2 was Spearman +0.38. The current edit-distance formula on the same
clips was +0.33. GOP is not the fix.

Everything else that can be tested from the stored decodes was tested
against Path 2 on the 167 overlapping items. None of it beats the current
scorer (Spearman +0.49):

| Variant | Spearman vs Path 2 |
| --- | ---: |
| Current Path 3 score | +0.49 |
| 12 panphon formulas (weighted, Levenshtein, Dolgopolsky; raw / length / sqrt) | −0.45 to −0.50 |
| Same decoder on the human clip instead of dictionary IPA | −0.46 (worse) |
| Edit distance on the CTC model's own tokens, diphthongs kept whole | −0.42 to −0.47 |
| Vowel substitutions down-weighted or dropped | −0.24 to −0.45 (worse as vowels get cheaper) |
| Semi-global alignment (ignore span-bleed insertions) | −0.41 |

Token-level distance does put the original Voranigo decode in the worst
6% (the extra `t` and the `aʊ`/`oʊ` ending count as full token errors).
That is a real local fix for that one item. It does not move the corpus
correlation. Length normalization was not the bug either: changing it
moves Spearman by less than 0.01.

A second recognizer (`vitouphy/wav2vec2-xls-r-300m-timit-phoneme`, English
phones rather than the multilingual espeak model) looked strong on the 9
dramatic names (Spearman −0.73) and then failed the broader check: −0.36
edit-distance and +0.45 GOP against Path 2 on 40 paired items. Same
ceiling.

What that rules out: the dictionary string for Voranigo, the 0–5 mapping,
and the choice of edit-distance formula. What it does not rule out, and
what every result points at: free phoneme error and Path 2's wavlm
audio-to-audio score only partly measure the same thing. The recognizer's
vowel noise is about as large as a real mispronunciation, so bad and
merely-different items land in the same band. Chantix (exact match) and
Adquey (unrecognizable) still separate. Everything in between does not,
which is why Voranigo can be horrible by ear and still score 3.18.

Path 3 is left as the current scorer. Replacing it with GOP or the English
model would not make it agree with Path 2 more than it already does.

## Reading this correctly

- **Path 3 is not a second ground truth.** It exists to reach the items
  Path 2 cannot score (no human recording). On the overlap, Path 2 wins.
- **Not yet checked**: a minimal-pair sensitivity test. No phoneme-distance
  variant tried here cleared Path 2's ranking, so that test is not what is
  blocking a better Path 3.

## Artifacts

- `runs/gemini-flash-tts-v1-phoneme-distance-v1/` -- results.jsonl (274
  records, 260 scoreable, 11 suffix-gap-flagged), manifest.json,
  report.txt, summary.json.
- `runs/phoneme-distance-validation-v1.json` -- human-ceiling and
  discrimination validation.
- `dose_r/scoring/phoneme_distance.py` -- the scorer.
- `dose_r/references/references.jsonl` -- IPA data snapshot (provisional,
  pulled from `claude/sc-sandbox-gcp-access-arw4m8` @ fee5ac5).
- `scripts/diagnose_path3_vs_path2.py`, `diagnose_path3_decoder_noise.py`,
  `diagnose_path3_gop.py` -- the metric, same-decoder, and GOP checks.
- `scripts/diagnose_path3_token_metric.py`, `diagnose_path3_alignment.py`,
  `diagnose_path3_vowel_weight.py` -- token, alignment, and vowel-cost sweeps.
- `scripts/probe_timit_paired.py`, `scripts/probe_timit_gop.py` -- English
  phoneme model, scored against Path 2 on the same new clips.
- `runs/path3-timit-probe/paired.json` -- the 40-item paired result
  (Spearman −0.36 edit distance, and the Path 2 scores GOP was checked
  against).
