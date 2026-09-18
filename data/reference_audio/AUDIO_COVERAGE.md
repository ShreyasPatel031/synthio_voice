# Reference Audio -- Coverage

Human-recorded pronunciation audio for the 284 unique ingredients across
the 274 DOSE rows, from two sources: Merriam-Webster's Medical API and
Drugs.com (collected by hand -- see `data/collected/HANDOFF_AUDIO_COLLECTION.md`,
drugs.com 403s this environment on every request). "Coverage" below means a
clip that pronounces the *whole* ingredient name, since that is what DOSE
scores; Merriam-Webster's word-level partial clips are reported separately.

## Headline

| Metric | Count | Share |
| --- | --- | --- |
| Total unique ingredients | 284 | 100% |
| Drugs.com audio | 143 | 50% |
| Merriam-Webster audio (full name) | 82 | 29% |
| Union -- any audio | 149 | 52% |
| Both sources (cross-checkable) | 76 | 27% |
| Only Drugs.com | 67 | 24% |
| Only Merriam-Webster | 6 | 2% |
| No audio anywhere | 135 | 48% |

## By name type

| Group | brand | generic | total |
| --- | --- | --- | --- |
| Both sources | 45 | 31 | 76 |
| Only Drugs.com | 54 | 13 | 67 |
| Only Merriam-Webster | 0 | 6 | 6 |
| No audio anywhere | 44 | 91 | 135 |

## By reference-layer confidence tier

Cross-referencing audio coverage against the phoneme reference layer's own
confidence tier (`dose_r/references/COVERAGE.md`): where the two disagree --
audio present but tier `low`, or tier `high` but no audio -- is exactly where
independent verification helps most.

| Tier | both sources | one source | no audio |
| --- | --- | --- | --- |
| high | 20 | 4 | 4 |
| medium | 56 | 8 | 19 |
| low | 0 | 61 | 112 |

## Merriam-Webster: reconciling row counts

The manifest carries 92 Merriam-Webster rows for 89 unique ingredients -- more rows than
ingredients because two multi-word generics only ever resolved word-by-word
and get one row per word that answered (`dimethyl fumarate`: 2 rows;
`formoterol fumarate dihydrate`: 3 rows -- 3 extra rows over 89 ingredients
= 92).

Of those 89, 82 carry a clip for the *whole* name
(`coverage: full`, unflagged) and 7 carry only a
single word of a multi-word generic (`coverage: component`, flagged as
partial -- listed below). This report counts only the 82 full clips as "Merriam-Webster audio" for scoring purposes,
since a component clip does not say the name DOSE asks about.

This does not exactly reproduce an expected external count of 87: 82 (full
only) and 89 (full + any component) bracket it, and 87 falls between the
two. The most likely reading is that 87 counts most but not all of the 7
component-only ingredients as "available" -- a judgment call this report
did not have grounds to make one way or the other without a live
`MW_MEDICAL_KEY` re-check (not set in this environment) or the original
count's own criteria. All 7 component-only ingredients are listed below so
a human can decide.

Component-only Merriam-Webster ingredients (partial, not full-name, audio):

- `copper histidinate` -- only copper answered
- `dimethyl fumarate` -- only dimethyl, fumarate answered
- `enlicitide decanoate` -- only decanoate answered
- `formoterol fumarate dihydrate` -- only dihydrate, formoterol, fumarate answered
- `insulin icodec-abae` -- only insulin answered
- `tenofovir alafenamide` -- only tenofovir answered
- `trospium chloride` -- only chloride answered

## Flagged clips

### Possible name mismatch (Drugs.com)

Duration-per-syllable outliers relative to their own batch's median (see
`audio_verify.syllable_outliers`) -- candidates for a human ear check, not
discarded. All 8 are brand-name clips running long, consistent with (but not
proof of) a brand page's audio actually pronouncing its generic, the failure
mode confirmed possible for Anktiva in the handoff doc.

| Ingredient | Duration (s) | Flag |
| --- | --- | --- |
| Advair | 1.6659 | possible name mismatch: 0.833s/syllable is outside [0.186, 0.743]s/syllable for this batch (median 0.372) |
| Journavx | 1.7692 | possible name mismatch: 0.885s/syllable is outside [0.186, 0.743]s/syllable for this batch (median 0.372) |
| Motrin | 1.6254 | possible name mismatch: 0.813s/syllable is outside [0.186, 0.743]s/syllable for this batch (median 0.372) |
| Talvey | 1.5986 | possible name mismatch: 0.799s/syllable is outside [0.186, 0.743]s/syllable for this batch (median 0.372) |
| Toujeo | 2.2523 | possible name mismatch: 1.126s/syllable is outside [0.186, 0.743]s/syllable for this batch (median 0.372) |
| Vyloy | 1.6626 | possible name mismatch: 0.831s/syllable is outside [0.186, 0.743]s/syllable for this batch (median 0.372) |
| Vyvgart | 1.4921 | possible name mismatch: 0.746s/syllable is outside [0.186, 0.743]s/syllable for this batch (median 0.372) |
| Zepbound | 1.5557 | possible name mismatch: 0.778s/syllable is outside [0.186, 0.743]s/syllable for this batch (median 0.372) |

### Partial coverage (Merriam-Webster, word-level only)

Flagged because the clip pronounces one word of a multi-word generic name,
not the whole ingredient -- listed in full in the reconciliation section
above.

| Ingredient | Word covered | Duration (s) |
| --- | --- | --- |
| copper histidinate | copper | 0.4785 |
| dimethyl fumarate | dimethyl | 0.6587 |
| dimethyl fumarate | fumarate | 0.7466 |
| enlicitide decanoate | decanoate | 1.0876 |
| formoterol fumarate dihydrate | dihydrate | 0.9617 |
| formoterol fumarate dihydrate | formoterol | 1.3003 |
| formoterol fumarate dihydrate | fumarate | 0.7466 |
| insulin icodec-abae | insulin | 0.6283 |
| tenofovir alafenamide | tenofovir | 1.0217 |
| trospium chloride | chloride | 0.8213 |

## Duplicate audio across ingredients

Byte-identical clips shared by more than one ingredient -- either a genuine
shared headword or a bug; made visible either way rather than silently
inflating coverage.

| sha256 | Ingredients |
| --- | --- |
| 1751adde366f | `dimethyl fumarate`, `formoterol fumarate dihydrate` |

`dimethyl fumarate` and `formoterol fumarate dihydrate` share the word
"fumarate" -- both resolved to the same Merriam-Webster audio file for that
one shared word, which is the correct behavior, not a bug.

## What is still missing

135 ingredients (48%) have no audio
from either source: 91 generic, 44 brand. The gap skews generic --
coined INN names are exactly what neither a general dictionary nor a
consumer drug-information site reliably records. See
`data/collected/HANDOFF_AUDIO_COLLECTION.md` for sources tried and the
paid/licensed options (USP Dictionary of USAN, a citable MedlinePlus key,
a Drugs.com data license) that would close the rest.

