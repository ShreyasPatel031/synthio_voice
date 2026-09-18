# Strata reconstruction — era and difficulty

DOSE reports splits (128/146 established-vs-new, 63/102/109
easy/medium/hard) that are not present in the public columns. Both
are reconstructed here from measurable signals and fit to the
published marginal counts. **Matching those counts shows the tier
sizes agree with DOSE's, not that any individual row carries the
tier DOSE assigned it.** Treat both strata as an approximation for
reporting shape (e.g. "systems drop N points on new names"), not as
ground truth for any one drug.

## Era: established vs newly-approved

**Method.** Look up each row's ingredient(s) against openFDA's
Drugs@FDA endpoint (`fda_lookup.py`). A row's anchor date is the
*latest* approval date among its ingredients (a combination enters
clinical speech only once its newest component does). Era is
`established` if that date is before the cutoff, else `new`.

**Cutoff: `2021-08-27`.** 5 years before 2026-08-27, the most recent approval date openFDA returned for any ingredient in this dataset -- a trailing window anchored to the data rather than to today's real-world date, so it does not silently drift as time passes. See STRATA.md for the sensitivity of the 128/146 split to this choice.

**Result:** 127 established / 147 new
(target: 128 / 146,
off by 1).

**openFDA hit rate.** Querying only the harmonized `openfda.*` fields
(the partial implementation this build started from) resolved 252/284
ingredients (88.7%) and left 32/274 rows with no date at all --
including Eliquis, Benadryl, Biktarvy, Ubrelvy and Wegovy, none of
which are remotely new. `openfda.*` is a harmonized enrichment block
openFDA computes after the fact and it is silently absent on real,
long-approved applications. Adding the raw `products.brand_name` /
`products.active_ingredients.name` fields as a fallback (this build)
raised ingredient-level coverage to 270/284 (95.1%) and row-level
coverage to 260/274 (94.9%), and fixed all five names above.

The remaining 14 rows (5.1%) have no FDA
match under either field set. Per `fda_lookup.py`'s own contract, a
genuine miss is informative (unapproved implies not established), so
these are labelled `era=new`, `era_confidence=low`,
`era_source=heuristic_no_fda_match` -- never given a fabricated date.
Manual spot-check: this bucket is a mix of very recent 2025-2026
approvals not yet indexed and cell/gene therapies (Casgevy and its
generic name, exagamglogene autotemcel) that Drugs@FDA's NDA/BLA-drug
endpoint is known to cover incompletely relative to CBER's biologics
review track -- both cases point toward genuine novelty, but this is
not certain for every name in the list:

- dose-011: Aucatzyl (brand)
- dose-022: Casgevy (brand)
- dose-088: exagamglogene autotemcel (generic)
- dose-114: obecabtagene autoleucel (generic)
- dose-126: prademagene zamikeracel (generic)
- dose-158: troriluzole (generic)
- dose-198: Meibo (brand)
- dose-205: Obicetrapib (brand)
- dose-216: Retatrutide (brand)
- dose-249: Vyglxia (brand)
- dose-264: Zaiidra (brand)
- dose-268: Zevaskyn (brand)
- dose-269: Zipalertinib (brand)
- dose-271: Zorevunersen (brand)

**Confidence.** high=241, low=14, medium=19
`high` = matched an `openfda.*` field directly; `medium` = matched
only via the `products.*` fallback or a modifier-stripped name, or a
combination row where a co-ingredient is unresolved; `low` = the
no-match heuristic above.

**Sensitivity to the cutoff date.** The choice above sits in a wide
plateau (2020-10 through 2021-06 all give the same split), which is
itself informative: this dataset does not densely populate that
window, so the era split's resolution is coarse, on the order of a
year, not a day.

| cutoff | established | new |
| --- | --- | --- |
| 2015-01-01 | 85 | 189 |
| 2017-01-01 | 96 | 178 |
| 2018-01-01 | 104 | 170 |
| 2019-01-01 | 109 | 165 |
| 2020-01-01 | 122 | 152 |
| 2020-10-01 | 126 | 148 |
| 2021-01-01 | 126 | 148 |
| 2021-08-27 | 127 | 147 |
| 2021-09-01 | 127 | 147 |
| 2022-01-01 | 131 | 143 |
| 2023-01-01 | 138 | 136 |

## Difficulty: easy / medium / hard

**Method.** A composite score per row:

```
score = phoneme_count(row)          # summed ARPABET phonemes across
                                     # the row's ingredient(s), from
                                     # the gold reference layer
      + 0.02 * len(row.name)         # tiebreaker within a phoneme band
      + 2.0   if usan_stem or biologic_suffix else 0
```

USAN stems checked: `-umab`, `-tinib`/`-inib`, `-tide`, `-zole` --
the brief's counts (8/7/5/4) are approximate; the actual counts in
this dataset's reference layer are:

- `-umab`: 8
- `-tinib`: 8
- `-tide`: 6
- `-zole`: 4

The brief's other measured signal -- "generics average 24 chars and
7-8 syllables, 12 exceed 10 syllables" -- **does not match this
staged dataset.** Measured directly from `data/dose_v1.jsonl` against
the gold reference layer: generic rows average 14.5 characters and
5.4 syllables (row-level, combination rows summed), and only 5 rows
exceed 10 syllables, not 12. Brand rows do match closely (7.3 chars,
2.9 syllables vs the brief's 7 chars / 2-3 syllables), which is a
useful sanity check that the syllabifier and reference layer are
sound -- it is specifically the generic-side prior that is off, most
likely because DOSE-R's 131 generic rows include a number of short,
well-known comparator generics (aspirin, ibuprofen, metformin) that a
pure INN-stem sample would not. **This build used the measured
values, not the brief's,** and flags the mismatch rather than either
silently reproducing it or silently ignoring it.

**Cutoffs:** easy if score ≤ 6.16, medium if score ≤
10.18, else hard. Chosen by grid search over cutoff pairs
to minimize the summed absolute deviation from (63, 102, 109); see
`build_strata.difficulty_sensitivity` for the search.

**Result:** 55 easy / 102 medium /
117 hard (target: 63 / 102 / 109; summed absolute
deviation = 16).
This is a *materially worse* fit than the era split, and the search
space is genuinely limited: raw phoneme/character counts on this
dataset cluster too tightly to produce three well-separated,
63/102/109-sized bands no matter how the weights are tuned -- the
closest the grid search found anywhere was within 16 of the target
sum, not 0.

**Tie sensitivity.** The two cutoffs sit inside bands where dozens of
rows share near-identical scores (short brand names cluster at
phoneme_count=6, mid-length generics at phoneme_count=9-10). Rows
within ±0.1 of the easy/medium cutoff: 38.
Rows within ±0.1 of the medium/hard cutoff: 19.
Every one of those rows' tier assignment would flip under a
trivially different weighting -- treat any single row's difficulty
label near these bands as a coin flip, not a fact.

**Weight sensitivity.** Varying the length weight and stem bonus:

| length_weight | stem_bonus | easy | medium | hard |
| --- | --- | --- | --- | --- |
| 0.0 | 1.0 | 55 | 126 | 93 |
| 0.0 | 2.0 | 55 | 121 | 98 |
| 0.0 | 3.0 | 55 | 121 | 98 |
| 0.01 | 1.0 | 55 | 125 | 94 |
| 0.01 | 2.0 | 55 | 120 | 99 |
| 0.01 | 3.0 | 55 | 120 | 99 |
| 0.02 | 1.0 | 55 | 104 | 115 |
| 0.02 | 2.0 | 55 | 102 | 117 |
| 0.02 | 3.0 | 55 | 102 | 117 |
| 0.05 | 1.0 | 17 | 138 | 119 |
| 0.05 | 2.0 | 17 | 138 | 119 |
| 0.05 | 3.0 | 17 | 138 | 119 |

## Reference-layer confidence, carried through per row

Each row also carries `reference_confidence`: the *pronunciation* gold
layer's own confidence tier (`dose_r/references/references.jsonl`,
built and documented separately in `COVERAGE.md`), taken as the
weakest tier across the row's ingredient(s). This is unrelated to
`era_confidence` above -- one grades an openFDA lookup, the other
grades a pronunciation source -- and the two must not be conflated;
an earlier draft of this build did exactly that; see `test_strata.py`.

| tier | rows |
| --- | --- |
| high | 18 |
| medium | 76 |
| low | 180 |

This is the number the fidelity report's power analysis
(`dose_r/report/power.py`) uses for `effective_sample_size` and for
propagating reference uncertainty into a pass-rate confidence
interval -- not the era-confidence heuristic bucket, which is much
smaller and answers a different question.

## Bottom line

The era split reproduces DOSE's marginal counts closely (off by 1 in
each direction) on a mechanism with real, checkable inputs (openFDA
approval dates, a documented cutoff, a documented and inspectable
heuristic bucket). Treat it as a workable proxy for the
established-vs-new *shape* DOSE reports, with the caveat that a
specific row's label near 2021-08-27 is only as reliable as a
single ingredient's openFDA record.

The difficulty split is the weaker of the two: it fits the marginal
counts only loosely (55/102/117 achieved vs 63/102/109 target) and
rests on tie-heavy cutoffs. Use it for coarse comparisons (hard vs
easy) and do not lean on it for anything that depends on the exact
count in any one tier.

