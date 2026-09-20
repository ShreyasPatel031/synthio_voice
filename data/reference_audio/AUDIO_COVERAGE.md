# Reference Audio -- Coverage

Human-recorded pronunciation audio for the 284 unique ingredients across
the 274 DOSE rows, from 5 sources: Drugs.com, Merriam-Webster (full name), UMich, ClinCalc (full name), NCI Dictionary of Cancer Terms.
Merriam-Webster's Medical API; Drugs.com collected by hand -- see
`data/collected/HANDOFF_AUDIO_COLLECTION.md`, drugs.com 403s this environment
on every request; UMich's student pronunciation page via the Wayback Machine,
since the live site 403s behind a Cloudflare challenge; ClinCalc's Top 250
Drugs pronunciation pages, fetched live -- the only source that records a
generic name and a brand name as two separate clips instead of one page's
one recording; the NCI Dictionary of Cancer Terms's own backing JSON API
(`webapis.cancer.gov/glossary/v1/`), a real hosted government recording per
term, fetched live. "Coverage" below means a clip that pronounces the
*whole* ingredient name, since that is what DOSE scores; Merriam-Webster's
and ClinCalc's word-/name-level partial clips are reported separately.

## Headline

| Metric | Count | Share |
| --- | --- | --- |
| Total unique ingredients | 284 | 100% |
| Drugs.com audio | 176 | 62% |
| Merriam-Webster (full name) audio | 82 | 29% |
| UMich audio | 7 | 2% |
| ClinCalc (full name) audio | 65 | 23% |
| NCI Dictionary of Cancer Terms audio | 50 | 18% |
| Union -- any audio | 182 | 64% |
| 2+ sources (cross-checkable) | 110 | 39% |
| All 5 sources | 2 | 1% |
| Only Drugs.com | 66 | 23% |
| Only Merriam-Webster (full name) | 3 | 1% |
| Only UMich | 0 | 0% |
| Only ClinCalc (full name) | 0 | 0% |
| Only NCI Dictionary of Cancer Terms | 3 | 1% |
| No audio anywhere | 102 | 36% |

## By name type

| Group | brand | generic | total |
| --- | --- | --- | --- |
| 2+ sources | 60 | 50 | 110 |
| Only Drugs.com | 42 | 24 | 66 |
| Only Merriam-Webster (full name) | 0 | 3 | 3 |
| Only UMich | 0 | 0 | 0 |
| Only ClinCalc (full name) | 0 | 0 | 0 |
| Only NCI Dictionary of Cancer Terms | 0 | 3 | 3 |
| No audio anywhere | 41 | 61 | 102 |

## By reference-layer confidence tier

Cross-referencing audio coverage against the phoneme reference layer's own
confidence tier (`dose_r/references/COVERAGE.md`): where the two disagree --
audio present but tier `low`, or tier `high` but no audio -- is exactly where
independent verification helps most.

| Tier | 2+ sources | 1 source | no audio |
| --- | --- | --- | --- |
| high | 88 | 12 | 12 |
| medium | 22 | 59 | 87 |
| low | 0 | 1 | 3 |

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
| Abilify | Drugs.com 1.047s, Merriam-Webster (full name) 2.135s, ClinCalc (full name) 1.848s |
| Advair | Drugs.com 1.666s, Merriam-Webster (full name) 1.857s, ClinCalc (full name) 2.214s |
| Advil | Drugs.com 0.734s, Merriam-Webster (full name) 0.450s, NCI Dictionary of Cancer Terms 0.888s |
| Ambien | Drugs.com 0.879s, Merriam-Webster (full name) 0.768s, ClinCalc (full name) 1.587s, NCI Dictionary of Cancer Terms 1.175s |
| Anktiva | Drugs.com 1.535s, NCI Dictionary of Cancer Terms 1.471s |
| Aspirin | Drugs.com 0.883s, Merriam-Webster (full name) 0.553s, ClinCalc (full name) 1.587s, NCI Dictionary of Cancer Terms 0.914s |
| Aucatzyl | Drugs.com 1.428s, NCI Dictionary of Cancer Terms 1.675s |
| Benadryl | Drugs.com 0.768s, Merriam-Webster (full name) 0.606s, ClinCalc (full name) 0.891s |
| Bizengri | Drugs.com 1.407s, NCI Dictionary of Cancer Terms 1.549s |
| Byetta | Drugs.com 0.754s, ClinCalc (full name) 0.740s |
| Chantix | Drugs.com 1.017s, Merriam-Webster (full name) 1.300s, ClinCalc (full name) 1.665s, NCI Dictionary of Cancer Terms 1.071s |
| Claritin | Drugs.com 0.992s, Merriam-Webster (full name) 0.717s |
| Crestor | Drugs.com 0.902s, Merriam-Webster (full name) 0.763s, UMich 1.500s, ClinCalc (full name) 1.482s, NCI Dictionary of Cancer Terms 1.045s |
| Cymbalta | Drugs.com 1.066s, Merriam-Webster (full name) 1.207s, ClinCalc (full name) 1.508s, NCI Dictionary of Cancer Terms 1.280s |
| Datroway | Drugs.com 1.492s, NCI Dictionary of Cancer Terms 1.589s |
| Eliquis | Drugs.com 0.805s, Merriam-Webster (full name) 1.300s, ClinCalc (full name) 1.639s |
| Elrexfio | Drugs.com 1.556s, NCI Dictionary of Cancer Terms 2.200s |
| Enbrel | Drugs.com 0.733s, Merriam-Webster (full name) 1.022s, ClinCalc (full name) 0.713s |
| Ensacove | Drugs.com 1.535s, NCI Dictionary of Cancer Terms 1.759s |
| Farxiga | Drugs.com 0.670s, Merriam-Webster (full name) 1.207s |
| Flonase | Drugs.com 0.909s, Merriam-Webster (full name) 0.939s |
| Humira | Drugs.com 0.930s, Merriam-Webster (full name) 1.022s, ClinCalc (full name) 1.404s |
| Imfinzi | Drugs.com 1.207s, NCI Dictionary of Cancer Terms 1.593s |
| Januvia | Drugs.com 0.935s, Merriam-Webster (full name) 1.022s, ClinCalc (full name) 0.976s |
| Kisqali | Drugs.com 0.836s, NCI Dictionary of Cancer Terms 1.489s |
| Latuda | Drugs.com 0.950s, Merriam-Webster (full name) 1.022s |
| Lipitor | Drugs.com 0.932s, Merriam-Webster (full name) 0.783s, UMich 1.250s, ClinCalc (full name) 1.560s, NCI Dictionary of Cancer Terms 1.045s |
| Lyrica | Drugs.com 0.839s, Merriam-Webster (full name) 0.929s, ClinCalc (full name) 0.894s, NCI Dictionary of Cancer Terms 1.019s |
| Metformin | Drugs.com 1.239s, Merriam-Webster (full name) 0.868s, ClinCalc (full name) 1.796s |
| Motrin | Drugs.com 1.625s, Merriam-Webster (full name) 0.605s, NCI Dictionary of Cancer Terms 1.045s |
| Mounjaro | Drugs.com 1.440s, Merriam-Webster (full name) 1.536s |
| Nexium | Drugs.com 0.934s, Merriam-Webster (full name) 0.942s, ClinCalc (full name) 1.743s, NCI Dictionary of Cancer Terms 1.463s |
| Ojemda | Drugs.com 1.577s, NCI Dictionary of Cancer Terms 1.399s |
| Ozempic | Drugs.com 0.952s, Merriam-Webster (full name) 1.451s |
| Plavix | Drugs.com 0.961s, UMich 1.250s, ClinCalc (full name) 0.683s |
| Prilosec | Drugs.com 0.921s, Merriam-Webster (full name) 1.115s, ClinCalc (full name) 2.005s |
| Prozac | Drugs.com 0.877s, Merriam-Webster (full name) 0.655s, ClinCalc (full name) 1.639s |
| Revuforj | Drugs.com 1.705s, NCI Dictionary of Cancer Terms 1.941s |
| Rybelsus | Drugs.com 1.532s, Merriam-Webster (full name) 1.365s |
| Seroquel | Drugs.com 1.200s, Merriam-Webster (full name) 1.300s, ClinCalc (full name) 1.508s |
| Spiriva | Drugs.com 1.100s, Merriam-Webster (full name) 0.929s, ClinCalc (full name) 1.900s |
| Symbicort | Drugs.com 0.972s, Merriam-Webster (full name) 1.207s, ClinCalc (full name) 1.717s |
| Synthroid | Drugs.com 0.948s, Merriam-Webster (full name) 0.796s, ClinCalc (full name) 1.691s |
| Tagrisso | Drugs.com 1.115s, NCI Dictionary of Cancer Terms 1.584s |
| Tecfidera | Drugs.com 1.050s, Merriam-Webster (full name) 1.300s |
| Trulicity | Drugs.com 0.785s, Merriam-Webster (full name) 1.451s |
| Tylenol | Drugs.com 1.003s, Merriam-Webster (full name) 0.815s |
| Valium | Drugs.com 0.817s, Merriam-Webster (full name) 0.602s, ClinCalc (full name) 1.482s, NCI Dictionary of Cancer Terms 1.071s |
| Voranigo | Drugs.com 1.577s, NCI Dictionary of Cancer Terms 1.949s |
| Vyloy | Drugs.com 1.663s, NCI Dictionary of Cancer Terms 1.342s |
| Vyvanse | Drugs.com 1.091s, Merriam-Webster (full name) 1.115s, ClinCalc (full name) 0.878s |
| Wegovy | Drugs.com 1.620s, Merriam-Webster (full name) 1.195s |
| Xanax | Drugs.com 0.853s, Merriam-Webster (full name) 0.807s, ClinCalc (full name) 1.613s, NCI Dictionary of Cancer Terms 1.254s |
| Xarelto | Drugs.com 0.912s, Merriam-Webster (full name) 1.115s, ClinCalc (full name) 0.994s |
| Xeljanz | Drugs.com 0.954s, Merriam-Webster (full name) 1.300s |
| Xolair | Drugs.com 0.758s, Merriam-Webster (full name) 1.022s |
| Zantac | Drugs.com 0.842s, Merriam-Webster (full name) 0.750s, ClinCalc (full name) 0.923s |
| Zepbound | Drugs.com 1.556s, Merriam-Webster (full name) 1.621s |
| Zoloft | Drugs.com 1.037s, Merriam-Webster (full name) 0.761s, ClinCalc (full name) 0.813s, NCI Dictionary of Cancer Terms 0.914s |
| Zyrtec | Drugs.com 0.905s, Merriam-Webster (full name) 0.733s |
| acetaminophen | Drugs.com 1.203s, Merriam-Webster (full name) 0.935s, NCI Dictionary of Cancer Terms 1.411s |
| adalimumab | Drugs.com 1.202s, Merriam-Webster (full name) 1.300s, ClinCalc (full name) 1.926s |
| alprazolam | Drugs.com 1.212s, Merriam-Webster (full name) 0.970s, ClinCalc (full name) 2.188s, NCI Dictionary of Cancer Terms 1.463s |
| apixaban | Drugs.com 1.000s, Merriam-Webster (full name) 1.393s, ClinCalc (full name) 1.796s |
| aripiprazole | Drugs.com 1.277s, Merriam-Webster (full name) 1.207s, ClinCalc (full name) 1.952s |
| atorvastatin | Drugs.com 1.247s, Merriam-Webster (full name) 1.003s, UMich 1.750s, ClinCalc (full name) 1.243s |
| bevacizumab-vikg | Drugs.com 1.432s, NCI Dictionary of Cancer Terms 1.384s |
| budesonide | Drugs.com 1.124s, Merriam-Webster (full name) 0.918s, ClinCalc (full name) 1.717s, NCI Dictionary of Cancer Terms 1.724s |
| cefepime | Drugs.com 1.017s, UMich 1.500s, NCI Dictionary of Cancer Terms 1.463s |
| cetirizine | Drugs.com 1.169s, Merriam-Webster (full name) 1.250s |
| clopidogrel | Drugs.com 1.112s, Merriam-Webster (full name) 1.007s, UMich 1.750s, ClinCalc (full name) 1.978s |
| dapagliflozin | Drugs.com 1.950s, Merriam-Webster (full name) 1.393s |
| datopotamab deruxtecan | Drugs.com 3.732s, NCI Dictionary of Cancer Terms 3.342s |
| diazepam | Drugs.com 1.104s, Merriam-Webster (full name) 0.920s, ClinCalc (full name) 0.964s, NCI Dictionary of Cancer Terms 1.463s |
| diphenhydramine | Drugs.com 1.455s, Merriam-Webster (full name) 1.101s, ClinCalc (full name) 1.213s, NCI Dictionary of Cancer Terms 1.620s |
| dulaglutide | Drugs.com 1.109s, Merriam-Webster (full name) 1.707s |
| duloxetine | Drugs.com 1.169s, Merriam-Webster (full name) 1.486s, ClinCalc (full name) 2.240s |
| esomeprazole | Drugs.com 1.406s, Merriam-Webster (full name) 1.323s, ClinCalc (full name) 2.083s |
| etanercept | Drugs.com 1.234s, Merriam-Webster (full name) 1.393s, ClinCalc (full name) 1.796s, NCI Dictionary of Cancer Terms 1.254s |
| exenatide | Drugs.com 1.347s, ClinCalc (full name) 1.013s |
| famotidine | Drugs.com 1.052s, Merriam-Webster (full name) 0.896s, ClinCalc (full name) 1.030s |
| fluoxetine | Drugs.com 1.281s, Merriam-Webster (full name) 1.061s, ClinCalc (full name) 1.300s, NCI Dictionary of Cancer Terms 1.698s |
| fluticasone propionate | Drugs.com 1.278s, ClinCalc (full name) 1.149s |
| formoterol fumarate dihydrate | Drugs.com 1.026s, ClinCalc (full name) 1.848s |
| ibuprofen | Drugs.com 1.184s, ClinCalc (full name) 0.969s, NCI Dictionary of Cancer Terms 1.698s |
| insulin glargine | Drugs.com 1.858s, Merriam-Webster (full name) 1.555s, ClinCalc (full name) 2.083s, NCI Dictionary of Cancer Terms 2.325s |
| levothyroxine | Drugs.com 1.542s, ClinCalc (full name) 2.423s |
| lisdexamfetamine | Drugs.com 2.039s, ClinCalc (full name) 2.240s |
| loratadine | Drugs.com 1.296s, Merriam-Webster (full name) 0.891s, ClinCalc (full name) 1.126s |
| lurasidone | Drugs.com 1.184s, Merriam-Webster (full name) 1.207s |
| nogapendekin alfa inbakicept-pmln | Drugs.com 4.348s, NCI Dictionary of Cancer Terms 3.893s |
| obecabtagene autoleucel | Drugs.com 3.240s, NCI Dictionary of Cancer Terms 3.277s |
| omeprazole | Drugs.com 1.146s, Merriam-Webster (full name) 0.896s, ClinCalc (full name) 1.900s, NCI Dictionary of Cancer Terms 1.593s |
| pregabalin | Drugs.com 0.987s, Merriam-Webster (full name) 1.207s, ClinCalc (full name) 1.848s, NCI Dictionary of Cancer Terms 1.384s |
| quetiapine | Drugs.com 1.181s, Merriam-Webster (full name) 1.300s, ClinCalc (full name) 1.665s |
| rivaroxaban | Drugs.com 1.372s, Merriam-Webster (full name) 1.207s, ClinCalc (full name) 2.214s |
| rosuvastatin | Drugs.com 1.502s, Merriam-Webster (full name) 1.411s, ClinCalc (full name) 1.136s |
| salmeterol | Drugs.com 1.195s, Merriam-Webster (full name) 1.045s, ClinCalc (full name) 0.915s |
| sertraline | Drugs.com 0.968s, Merriam-Webster (full name) 0.878s, ClinCalc (full name) 2.397s, NCI Dictionary of Cancer Terms 1.724s |
| sitagliptin | Drugs.com 1.427s, Merriam-Webster (full name) 1.207s, ClinCalc (full name) 1.030s |
| talquetamab | Drugs.com 1.599s, NCI Dictionary of Cancer Terms 1.975s |
| testosterone undecanoate | Drugs.com 1.174s, ClinCalc (full name) 2.266s |
| tiotropium bromide | Drugs.com 1.258s, ClinCalc (full name) 2.240s |
| tirzepatide | Drugs.com 1.556s, Merriam-Webster (full name) 1.792s |
| tofacitinib | Drugs.com 2.067s, Merriam-Webster (full name) 1.300s |
| tovorafenib | Drugs.com 2.238s, NCI Dictionary of Cancer Terms 2.163s |
| valsartan | Drugs.com 1.017s, Merriam-Webster (full name) 1.107s, UMich 1.500s, ClinCalc (full name) 0.965s |
| varenicline | Drugs.com 1.219s, Merriam-Webster (full name) 1.672s, ClinCalc (full name) 0.882s |
| zolbetuximab | Drugs.com 2.025s, NCI Dictionary of Cancer Terms 2.100s |
| zolpidem | Drugs.com 1.106s, Merriam-Webster (full name) 1.022s, ClinCalc (full name) 0.946s, NCI Dictionary of Cancer Terms 1.019s |

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
- `formoterol fumarate dihydrate` -- only Budesonide; formoterol, dihydrate, formoterol, fumarate answered
- `insulin icodec-abae` -- only insulin answered
- `tenofovir alafenamide` -- only Efavirenz; emtricitabine; tenofovir, Emtricitabine; tenofovir, Emtricitabine; tenofovir; elvitegravir; cobicistat, Emtricitabine; tenofovir; rilpivirine, tenofovir answered
- `trospium chloride` -- only chloride answered

## ClinCalc: reconciling row counts

The manifest carries 89 ClinCalc rows for 71 unique
ingredients. Of those, 65 carry a clip that is the *only* name
in its audio block (`coverage: full`) and 13 share a block's one
recording with one or more other names -- a combination product's several
generic components, or several brands sold for one generic, e.g. ClinCalc's
one `ibuprofen` brand clip says "Advil, Motrin" together, not either alone
(`coverage: component`, flagged as partial, listed below).

62 of the 71 carry a `respelling` note pointing to
a second ClinCalc clip on the same page for the paired generic or brand name,
where that paired name is itself another DOSE ingredient with its own row --
e.g. `atorvastatin`'s row notes ClinCalc's separate `Lipitor` clip, and
`Lipitor`'s row notes the `atorvastatin` one back. Every ClinCalc row carries
such a note when the page has both a generic and a brand block, whether or
not the other name happens to be its own DOSE ingredient -- see the
manifest's `respelling` field for the rest.

Component-only ClinCalc ingredients (partial, not full-name, audio):

- `Advil` -- shares its clip with Motrin
- `Aspirin` -- shares its clip with butalbital, caffeine, dipyridamole
- `Claritin` -- shares its clip with Alavert
- `Metformin` -- shares its clip with Glyburide, Sitagliptin
- `Motrin` -- shares its clip with Advil
- `acetaminophen` -- shares its clip with Codeine, Oxycodone, butalbital, caffeine, hydrocodone
- `budesonide` -- shares its clip with formoterol
- `emtricitabine` -- shares its clip with Efavirenz, cobicistat, elvitegravir, rilpivirine, tenofovir
- `fluticasone propionate` -- shares its clip with Fluticasone, salmeterol
- `formoterol fumarate dihydrate` -- shares its clip with Budesonide, formoterol
- `salmeterol` -- shares its clip with Fluticasone
- `sitagliptin` -- shares its clip with metformin
- `tenofovir alafenamide` -- shares its clip with Efavirenz, Emtricitabine, cobicistat, elvitegravir, emtricitabine, rilpivirine, tenofovir

## ClinCalc cross-check of previously flagged clips

ClinCalc is the first source that records a brand and a generic name as two
separate clips, so where it covers a flagged ingredient its own clip (or a
same-source sibling, like a clean single-word Merriam-Webster clip) gives an
unambiguous duration to compare the flagged clip's duration against, instead
of only a within-batch syllable estimate.

- `Advair` (Drugs.com, 1.666s, flagged) -- ClinCalc's dedicated single-name
  `Advair` brand clip runs 2.214s and Merriam-Webster's unflagged `Advair`
  clip runs 1.857s. Drugs.com's duration sits at or below both independent
  clean-word recordings, not anywhere near ClinCalc's own combined
  `Fluticasone; salmeterol` clip (2.893s) a mispronunciation as the generic
  would have to resemble. **Resolved**: the flag was a within-Drugs.com-batch
  artifact; the clip's duration is consistent with genuinely saying "Advair".
- `Motrin` (Drugs.com, 1.625s, flagged) -- ClinCalc never recorded `Motrin`
  alone (its ibuprofen page's one brand clip says "Advil, Motrin" together,
  2.736s), but Merriam-Webster's unflagged, unambiguous single-word `Motrin`
  clip runs only 0.605s. Drugs.com's `Motrin` (1.625s) is also 2.2x its own
  `Advil` clip (0.734s) despite both being two-syllable brand names recorded
  in the same batch. **Confirmed suspicious**: nothing here contradicts the
  original flag, and the size of the gap from Merriam-Webster's clean word
  makes a wrong-name clip (most likely the generic, "ibuprofen") more likely
  than a slow reading of "Motrin" alone.
- `fluticasone propionate` (Drugs.com, 1.278s, flagged) -- ClinCalc's clean,
  unflagged single-name generic clip, headed "Fluticasone (inhaled)", runs
  1.149s -- 11% off Drugs.com's duration for what both would then be the same
  bare word. **Resolved**: consistent with Drugs.com's clip pronouncing only
  the base name "fluticasone" and omitting the "propionate" salt, not with a
  different drug; the flag's syllable estimate over-counted using the full
  ingredient name's syllables against a clip that likely never spoke them all.
- `formoterol fumarate dihydrate` (Drugs.com, 1.026s, flagged) -- ClinCalc's
  clean standalone `Formoterol` clip runs 1.848s and Merriam-Webster's clean
  `formoterol` word-clip runs 1.300s; Drugs.com's 1.026s is the *shortest* of
  the three bare-"formoterol" measurements, not the longest a wrong, longer
  name would produce. **Resolved**: same reading as `fluticasone propionate`
  -- a clip of the base generic name only, not a name mismatch.

## Flagged clips

### Possible name mismatch (syllable-outlier check)

Duration-per-syllable outliers relative to their own source's batch median
(see `audio_verify.syllable_outliers`) -- candidates for a human ear check,
not discarded. Every one so far is a brand-name clip running long, consistent
with (but not proof of) a brand page's audio actually pronouncing its
generic, the failure mode confirmed possible for Anktiva in the handoff doc.

| Source | Ingredient | Duration (s) | Flag |
| --- | --- | --- | --- |
| ClinCalc (full name) | Advair | 2.2136 | possible name mismatch: 1.107s/syllable is outside [0.221, 0.885]s/syllable for this batch (median 0.443) |
| ClinCalc (full name) | Advil | 2.7361 | possible name mismatch: 1.368s/syllable is outside [0.221, 0.885]s/syllable for this batch (median 0.443) |
| ClinCalc (full name) | Motrin | 2.7361 | possible name mismatch: 1.368s/syllable is outside [0.221, 0.885]s/syllable for this batch (median 0.443) |
| ClinCalc (full name) | diphenhydramine | 1.2129 | possible name mismatch: 0.202s/syllable is outside [0.221, 0.885]s/syllable for this batch (median 0.443) |
| ClinCalc (full name) | exenatide | 1.0134 | possible name mismatch: 0.203s/syllable is outside [0.221, 0.885]s/syllable for this batch (median 0.443) |
| ClinCalc (full name) | famotidine | 1.0302 | possible name mismatch: 0.206s/syllable is outside [0.221, 0.885]s/syllable for this batch (median 0.443) |
| ClinCalc (full name) | formoterol fumarate dihydrate | 1.7002 | possible name mismatch: 0.154s/syllable is outside [0.221, 0.885]s/syllable for this batch (median 0.443) |
| ClinCalc (full name) | formoterol fumarate dihydrate | 1.8479 | possible name mismatch: 0.154s/syllable is outside [0.221, 0.885]s/syllable for this batch (median 0.443) |
| ClinCalc (full name) | varenicline | 0.8825 | possible name mismatch: 0.176s/syllable is outside [0.221, 0.885]s/syllable for this batch (median 0.443) |
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
| Advil | Advil; Motrin | 2.7361 |
| Aspirin | Aspirin; butalbital; caffeine | 3.0234 |
| Aspirin | Aspirin; dipyridamole | 1.7463 |
| Claritin | Claritin; Alavert | 1.8012 |
| Metformin | Glyburide; metformin | 2.2136 |
| Metformin | Sitagliptin; metformin | 2.3965 |
| Motrin | Advil; Motrin | 2.7361 |
| acetaminophen | Acetaminophen; butalbital; caffeine | 2.5104 |
| acetaminophen | Acetaminophen; hydrocodone | 2.1038 |
| acetaminophen | Codeine; acetaminophen | 3.1279 |
| acetaminophen | Oxycodone; acetaminophen | 2.945 |
| budesonide | Budesonide; formoterol | 1.7002 |
| copper histidinate | copper | 0.4785 |
| dimethyl fumarate | dimethyl | 0.6587 |
| dimethyl fumarate | fumarate | 0.7466 |
| emtricitabine | Efavirenz; emtricitabine; tenofovir | 4.5385 |
| emtricitabine | Emtricitabine; tenofovir | 2.6838 |
| emtricitabine | Emtricitabine; tenofovir; elvitegravir; cobicistat | 4.5627 |
| emtricitabine | Emtricitabine; tenofovir; rilpivirine | 4.5646 |
| enlicitide decanoate | decanoate | 1.0876 |
| fluticasone propionate | Fluticasone; salmeterol | 2.8928 |
| formoterol fumarate dihydrate | Budesonide; formoterol | 1.7002 |
| formoterol fumarate dihydrate | dihydrate | 0.9617 |
| formoterol fumarate dihydrate | formoterol | 1.3003 |
| formoterol fumarate dihydrate | fumarate | 0.7466 |
| insulin icodec-abae | insulin | 0.6283 |
| salmeterol | Fluticasone; salmeterol | 2.8928 |
| sitagliptin | Sitagliptin; metformin | 2.3965 |
| tenofovir alafenamide | Efavirenz; emtricitabine; tenofovir | 4.5385 |
| tenofovir alafenamide | Emtricitabine; tenofovir | 2.6838 |
| tenofovir alafenamide | Emtricitabine; tenofovir; elvitegravir; cobicistat | 4.5627 |
| tenofovir alafenamide | Emtricitabine; tenofovir; rilpivirine | 4.5646 |
| tenofovir alafenamide | tenofovir | 1.0217 |
| trospium chloride | chloride | 0.8213 |

## Duplicate audio across ingredients

Byte-identical clips shared by more than one ingredient -- either a genuine
shared headword or a bug; made visible either way rather than silently
inflating coverage.

| sha256 | Ingredients |
| --- | --- |
| 88e03d9e8357 | `Advil`, `Motrin` |
| e0eaab854155 | `Metformin`, `sitagliptin` |
| a61b0f14d595 | `budesonide`, `formoterol fumarate dihydrate` |
| 1751adde366f | `dimethyl fumarate`, `formoterol fumarate dihydrate` |
| 8e33e8d839fb | `emtricitabine`, `tenofovir alafenamide` |
| 954dd3a56a3e | `emtricitabine`, `tenofovir alafenamide` |
| b4a839d59c8e | `emtricitabine`, `tenofovir alafenamide` |
| 5b9046087c4a | `emtricitabine`, `tenofovir alafenamide` |
| 7ac6a6001c29 | `fluticasone propionate`, `salmeterol` |

`dimethyl fumarate` and `formoterol fumarate dihydrate` share the word
"fumarate" -- both resolved to the same Merriam-Webster audio file for that
one shared word, which is the correct behavior, not a bug. Likewise every
ClinCalc pair here (`Advil`/`Motrin`, `Metformin`/`sitagliptin`, and the rest)
shares one page's one `coverage: component` clip that names both -- also
correct, not a bug; see the ClinCalc reconciliation section above.

## What is still missing

102 ingredients (36%) have no audio
from any source: 61 generic, 41 brand. The gap skews generic --
coined INN names are exactly what neither a general dictionary, a consumer
drug-information site, an older pharmacy-school teaching list (UMich's,
which barely overlaps DOSE's newer names), a commonly-prescribed-drugs
pronunciation page (ClinCalc's, which skews the same way), nor a cancer-
specific dictionary (NCI's, whose real gain was cross-checking names other
sources already had, not covering brand-new non-oncology names) reliably
records. See
`data/collected/HANDOFF_AUDIO_COLLECTION.md` for sources tried and the
paid/licensed options (USP Dictionary of USAN, a citable MedlinePlus key,
a Drugs.com data license) that would close the rest.

