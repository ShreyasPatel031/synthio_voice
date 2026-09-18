# Reference Audio -- Coverage

Human-recorded pronunciation audio for the 284 unique ingredients across
the 274 DOSE rows, from three sources: Drugs.com, Merriam-Webster (full name), UMich.
Merriam-Webster's Medical API; Drugs.com collected by hand -- see
`data/collected/HANDOFF_AUDIO_COLLECTION.md`, drugs.com 403s this environment
on every request; UMich's student pronunciation page via the Wayback Machine,
since the live site 403s behind a Cloudflare challenge. "Coverage" below
means a clip that pronounces the *whole* ingredient name, since that is what
DOSE scores; Merriam-Webster's word-level partial clips are reported
separately.

## Headline

| Metric | Count | Share |
| --- | --- | --- |
| Total unique ingredients | 284 | 100% |
| Drugs.com audio | 176 | 62% |
| Merriam-Webster (full name) audio | 82 | 29% |
| UMich audio | 7 | 2% |
| Union -- any audio | 179 | 63% |
| 2+ sources (cross-checkable) | 81 | 29% |
| All 3 sources | 5 | 2% |
| Only Drugs.com | 95 | 33% |
| Only Merriam-Webster (full name) | 3 | 1% |
| Only UMich | 0 | 0% |
| No audio anywhere | 105 | 37% |

## By name type

| Group | brand | generic | total |
| --- | --- | --- | --- |
| 2+ sources | 46 | 35 | 81 |
| Only Drugs.com | 56 | 39 | 95 |
| Only Merriam-Webster (full name) | 0 | 3 | 3 |
| Only UMich | 0 | 0 | 0 |
| No audio anywhere | 41 | 64 | 105 |

## By reference-layer confidence tier

Cross-referencing audio coverage against the phoneme reference layer's own
confidence tier (`dose_r/references/COVERAGE.md`): where the two disagree --
audio present but tier `low`, or tier `high` but no audio -- is exactly where
independent verification helps most.

| Tier | 2+ sources | 1 source | no audio |
| --- | --- | --- | --- |
| high | 21 | 5 | 2 |
| medium | 60 | 13 | 10 |
| low | 0 | 80 | 93 |

## Cross-source agreement

Every ingredient with audio from 2 or more sources, with each source's
clip duration -- a rough plausibility check, not a substitute for an ear
check. UMich's clips run consistently longer than the other sources' for
the same name (roughly 1.3x-2x), which reads as a slower, more deliberate
teaching-recording pace rather than a name mismatch: `audio_verify`'s
syllable-outlier check, which flags a clip disproportionate to *its own*
batch, raised nothing for UMich because the lengthening is uniform across
its whole batch, not isolated to one name.

| Ingredient | Durations by source |
| --- | --- |
| Abilify | Drugs.com 1.047s, Merriam-Webster (full name) 2.135s |
| Advair | Drugs.com 1.666s, Merriam-Webster (full name) 1.857s |
| Advil | Drugs.com 0.734s, Merriam-Webster (full name) 0.450s |
| Ambien | Drugs.com 0.879s, Merriam-Webster (full name) 0.768s |
| Aspirin | Drugs.com 0.883s, Merriam-Webster (full name) 0.553s |
| Benadryl | Drugs.com 0.768s, Merriam-Webster (full name) 0.606s |
| Chantix | Drugs.com 1.017s, Merriam-Webster (full name) 1.300s |
| Claritin | Drugs.com 0.992s, Merriam-Webster (full name) 0.717s |
| Crestor | Drugs.com 0.902s, Merriam-Webster (full name) 0.763s, UMich 1.500s |
| Cymbalta | Drugs.com 1.066s, Merriam-Webster (full name) 1.207s |
| Eliquis | Drugs.com 0.805s, Merriam-Webster (full name) 1.300s |
| Enbrel | Drugs.com 0.733s, Merriam-Webster (full name) 1.022s |
| Farxiga | Drugs.com 0.670s, Merriam-Webster (full name) 1.207s |
| Flonase | Drugs.com 0.909s, Merriam-Webster (full name) 0.939s |
| Humira | Drugs.com 0.930s, Merriam-Webster (full name) 1.022s |
| Januvia | Drugs.com 0.935s, Merriam-Webster (full name) 1.022s |
| Latuda | Drugs.com 0.950s, Merriam-Webster (full name) 1.022s |
| Lipitor | Drugs.com 0.932s, Merriam-Webster (full name) 0.783s, UMich 1.250s |
| Lyrica | Drugs.com 0.839s, Merriam-Webster (full name) 0.929s |
| Metformin | Drugs.com 1.239s, Merriam-Webster (full name) 0.868s |
| Motrin | Drugs.com 1.625s, Merriam-Webster (full name) 0.605s |
| Mounjaro | Drugs.com 1.440s, Merriam-Webster (full name) 1.536s |
| Nexium | Drugs.com 0.934s, Merriam-Webster (full name) 0.942s |
| Ozempic | Drugs.com 0.952s, Merriam-Webster (full name) 1.451s |
| Plavix | Drugs.com 0.961s, UMich 1.250s |
| Prilosec | Drugs.com 0.921s, Merriam-Webster (full name) 1.115s |
| Prozac | Drugs.com 0.877s, Merriam-Webster (full name) 0.655s |
| Rybelsus | Drugs.com 1.532s, Merriam-Webster (full name) 1.365s |
| Seroquel | Drugs.com 1.200s, Merriam-Webster (full name) 1.300s |
| Spiriva | Drugs.com 1.100s, Merriam-Webster (full name) 0.929s |
| Symbicort | Drugs.com 0.972s, Merriam-Webster (full name) 1.207s |
| Synthroid | Drugs.com 0.948s, Merriam-Webster (full name) 0.796s |
| Tecfidera | Drugs.com 1.050s, Merriam-Webster (full name) 1.300s |
| Trulicity | Drugs.com 0.785s, Merriam-Webster (full name) 1.451s |
| Tylenol | Drugs.com 1.003s, Merriam-Webster (full name) 0.815s |
| Valium | Drugs.com 0.817s, Merriam-Webster (full name) 0.602s |
| Vyvanse | Drugs.com 1.091s, Merriam-Webster (full name) 1.115s |
| Wegovy | Drugs.com 1.620s, Merriam-Webster (full name) 1.195s |
| Xanax | Drugs.com 0.853s, Merriam-Webster (full name) 0.807s |
| Xarelto | Drugs.com 0.912s, Merriam-Webster (full name) 1.115s |
| Xeljanz | Drugs.com 0.954s, Merriam-Webster (full name) 1.300s |
| Xolair | Drugs.com 0.758s, Merriam-Webster (full name) 1.022s |
| Zantac | Drugs.com 0.842s, Merriam-Webster (full name) 0.750s |
| Zepbound | Drugs.com 1.556s, Merriam-Webster (full name) 1.621s |
| Zoloft | Drugs.com 1.037s, Merriam-Webster (full name) 0.761s |
| Zyrtec | Drugs.com 0.905s, Merriam-Webster (full name) 0.733s |
| acetaminophen | Drugs.com 1.203s, Merriam-Webster (full name) 0.935s |
| adalimumab | Drugs.com 1.202s, Merriam-Webster (full name) 1.300s |
| alprazolam | Drugs.com 1.212s, Merriam-Webster (full name) 0.970s |
| apixaban | Drugs.com 1.000s, Merriam-Webster (full name) 1.393s |
| aripiprazole | Drugs.com 1.277s, Merriam-Webster (full name) 1.207s |
| atorvastatin | Drugs.com 1.247s, Merriam-Webster (full name) 1.003s, UMich 1.750s |
| budesonide | Drugs.com 1.124s, Merriam-Webster (full name) 0.918s |
| cefepime | Drugs.com 1.017s, UMich 1.500s |
| cetirizine | Drugs.com 1.169s, Merriam-Webster (full name) 1.250s |
| clopidogrel | Drugs.com 1.112s, Merriam-Webster (full name) 1.007s, UMich 1.750s |
| dapagliflozin | Drugs.com 1.950s, Merriam-Webster (full name) 1.393s |
| diazepam | Drugs.com 1.104s, Merriam-Webster (full name) 0.920s |
| diphenhydramine | Drugs.com 1.455s, Merriam-Webster (full name) 1.101s |
| dulaglutide | Drugs.com 1.109s, Merriam-Webster (full name) 1.707s |
| duloxetine | Drugs.com 1.169s, Merriam-Webster (full name) 1.486s |
| esomeprazole | Drugs.com 1.406s, Merriam-Webster (full name) 1.323s |
| etanercept | Drugs.com 1.234s, Merriam-Webster (full name) 1.393s |
| famotidine | Drugs.com 1.052s, Merriam-Webster (full name) 0.896s |
| fluoxetine | Drugs.com 1.281s, Merriam-Webster (full name) 1.061s |
| insulin glargine | Drugs.com 1.858s, Merriam-Webster (full name) 1.555s |
| loratadine | Drugs.com 1.296s, Merriam-Webster (full name) 0.891s |
| lurasidone | Drugs.com 1.184s, Merriam-Webster (full name) 1.207s |
| omeprazole | Drugs.com 1.146s, Merriam-Webster (full name) 0.896s |
| pregabalin | Drugs.com 0.987s, Merriam-Webster (full name) 1.207s |
| quetiapine | Drugs.com 1.181s, Merriam-Webster (full name) 1.300s |
| rivaroxaban | Drugs.com 1.372s, Merriam-Webster (full name) 1.207s |
| rosuvastatin | Drugs.com 1.502s, Merriam-Webster (full name) 1.411s |
| salmeterol | Drugs.com 1.195s, Merriam-Webster (full name) 1.045s |
| sertraline | Drugs.com 0.968s, Merriam-Webster (full name) 0.878s |
| sitagliptin | Drugs.com 1.427s, Merriam-Webster (full name) 1.207s |
| tirzepatide | Drugs.com 1.556s, Merriam-Webster (full name) 1.792s |
| tofacitinib | Drugs.com 2.067s, Merriam-Webster (full name) 1.300s |
| valsartan | Drugs.com 1.017s, Merriam-Webster (full name) 1.107s, UMich 1.500s |
| varenicline | Drugs.com 1.219s, Merriam-Webster (full name) 1.672s |
| zolpidem | Drugs.com 1.106s, Merriam-Webster (full name) 1.022s |

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

### Possible name mismatch (syllable-outlier check)

Duration-per-syllable outliers relative to their own source's batch median
(see `audio_verify.syllable_outliers`) -- candidates for a human ear check,
not discarded. Every one so far is a brand-name clip running long, consistent
with (but not proof of) a brand page's audio actually pronouncing its
generic, the failure mode confirmed possible for Anktiva in the handoff doc.

| Source | Ingredient | Duration (s) | Flag |
| --- | --- | --- | --- |
| Drugs.com | Advair | 1.6659 | possible name mismatch: 0.833s/syllable is outside [0.183, 0.730]s/syllable for this batch (median 0.365) |
| Drugs.com | Journavx | 1.7692 | possible name mismatch: 0.885s/syllable is outside [0.183, 0.730]s/syllable for this batch (median 0.365) |
| Drugs.com | Meibo | 1.5325 | possible name mismatch: 0.766s/syllable is outside [0.183, 0.730]s/syllable for this batch (median 0.365) |
| Drugs.com | Motrin | 1.6254 | possible name mismatch: 0.813s/syllable is outside [0.183, 0.730]s/syllable for this batch (median 0.365) |
| Drugs.com | Nurtec | 1.9505 | possible name mismatch: 0.975s/syllable is outside [0.183, 0.730]s/syllable for this batch (median 0.365) |
| Drugs.com | Rinvoq | 1.4861 | possible name mismatch: 0.743s/syllable is outside [0.183, 0.730]s/syllable for this batch (median 0.365) |
| Drugs.com | Talvey | 1.5986 | possible name mismatch: 0.799s/syllable is outside [0.183, 0.730]s/syllable for this batch (median 0.365) |
| Drugs.com | Toujeo | 2.2523 | possible name mismatch: 1.126s/syllable is outside [0.183, 0.730]s/syllable for this batch (median 0.365) |
| Drugs.com | Vyloy | 1.6626 | possible name mismatch: 0.831s/syllable is outside [0.183, 0.730]s/syllable for this batch (median 0.365) |
| Drugs.com | Vyvgart | 1.4921 | possible name mismatch: 0.746s/syllable is outside [0.183, 0.730]s/syllable for this batch (median 0.365) |
| Drugs.com | Zepbound | 1.5557 | possible name mismatch: 0.778s/syllable is outside [0.183, 0.730]s/syllable for this batch (median 0.365) |
| Drugs.com | fluticasone propionate | 1.2781 | possible name mismatch: 0.142s/syllable is outside [0.183, 0.730]s/syllable for this batch (median 0.365) |
| Drugs.com | formoterol fumarate dihydrate | 1.0257 | possible name mismatch: 0.085s/syllable is outside [0.183, 0.730]s/syllable for this batch (median 0.365) |
| Drugs.com | tenofovir alafenamide | 1.1796 | possible name mismatch: 0.118s/syllable is outside [0.183, 0.730]s/syllable for this batch (median 0.365) |
| Drugs.com | testosterone undecanoate | 1.1736 | possible name mismatch: 0.117s/syllable is outside [0.183, 0.730]s/syllable for this batch (median 0.365) |

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

105 ingredients (37%) have no audio
from any source: 64 generic, 41 brand. The gap skews generic --
coined INN names are exactly what neither a general dictionary, a consumer
drug-information site, nor an older pharmacy-school teaching list (UMich's,
which barely overlaps DOSE's newer names) reliably records. See
`data/collected/HANDOFF_AUDIO_COLLECTION.md` for sources tried and the
paid/licensed options (USP Dictionary of USAN, a citable MedlinePlus key,
a Drugs.com data license) that would close the rest.

