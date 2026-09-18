# USAN Stem Engine -- Back-test Results

Method: for every generic in `references.jsonl` with a real external
source (confidence `high`/`medium`) whose name ends in a recognized
USAN/INN stem, its real source is ignored and both the stem engine
(`usan_stems.py`) and the plain generic fallback (`g2p.py`) are scored
blind against the real, sourced pronunciation, using the judge's own
phonetic-distance scorer (`dose_r/judge/distance.py` +
`phonetic_scorer.py`) -- the identical PEU metric a TTS system is
scored with. A positive improvement means the stem engine produced a
lower (better) normalized PEU error than plain `g2p.py`.

`empagliflozin` excluded: sourced reference is the brand name's (Jardiance) pronunciation, not the generic's.

## Per-stem results

| Stem | n | avg g2p PEU | avg stem PEU | avg improvement | verdict |
| --- | --- | --- | --- | --- | --- |
| -gliflozin | 2 | 0.7307 | 0.6350 | +0.0957 | kept |
| -prazole | 3 | 0.6629 | 0.4515 | +0.2114 | kept |
| -sartan | 1 | 0.5000 | 0.2000 | +0.3000 | kept |
| -statin | 2 | 1.2451 | 0.7605 | +0.4846 | kept |
| -tide | 5 | 1.2166 | 1.1735 | +0.0431 | kept |
| -tinib | 4 | 1.1333 | 1.1195 | +0.0137 | kept |
| -umab | 4 | 1.8258 | 1.9989 | -0.1731 | DROPPED -- no measured benefit |
| -vir | 3 | 2.2572 | 2.3478 | -0.0906 | DROPPED -- no measured benefit |
| -zumab | 3 | 1.6954 | 2.0375 | -0.3421 | DROPPED -- no measured benefit |

## Aggregate, kept stems only

17 sourced examples across 6 stems.
Plain `g2p.py` average normalized error: 1.0033 PEU.
Stem engine average normalized error: 0.8642 PEU.
Average improvement: +0.1391 PEU (14% reduction).

## Examples

### -gliflozin

| Ingredient | g2p PEU | stem PEU | improvement |
| --- | --- | --- | --- |
| dapagliflozin | 0.9750 | 0.8232 | +0.1518 |
| sotagliflozin | 0.4864 | 0.4468 | +0.0396 |

### -prazole

| Ingredient | g2p PEU | stem PEU | improvement |
| --- | --- | --- | --- |
| aripiprazole | 0.7529 | 0.6131 | +0.1398 |
| esomeprazole | 0.7773 | 0.6375 | +0.1398 |
| omeprazole | 0.4584 | 0.1038 | +0.3546 |

### -sartan

| Ingredient | g2p PEU | stem PEU | improvement |
| --- | --- | --- | --- |
| valsartan | 0.5000 | 0.2000 | +0.3000 |

### -statin

| Ingredient | g2p PEU | stem PEU | improvement |
| --- | --- | --- | --- |
| atorvastatin | 1.4518 | 0.6129 | +0.8390 |
| rosuvastatin | 1.0384 | 0.9082 | +0.1302 |

### -tide

| Ingredient | g2p PEU | stem PEU | improvement |
| --- | --- | --- | --- |
| bulevirtide-gmod | 2.3106 | 2.7960 | -0.4855 |
| dulaglutide | 1.1171 | 0.7814 | +0.3357 |
| exenatide | 1.5123 | 1.3884 | +0.1238 |
| semaglutide | 0.5390 | 0.4215 | +0.1175 |
| tirzepatide | 0.6042 | 0.4804 | +0.1238 |

### -tinib

| Ingredient | g2p PEU | stem PEU | improvement |
| --- | --- | --- | --- |
| ensartinib | 0.2175 | 0.1000 | +0.1175 |
| osimertinib | 1.6039 | 1.6919 | -0.0880 |
| tofacitinib | 0.6011 | 0.5830 | +0.0182 |
| upadacitinib | 2.1105 | 2.1033 | +0.0072 |

### -umab

| Ingredient | g2p PEU | stem PEU | improvement |
| --- | --- | --- | --- |
| adalimumab | 1.3445 | 1.2848 | +0.0597 |
| canakinumab | 3.1704 | 3.3098 | -0.1394 |
| durvalumab | 1.3964 | 1.6590 | -0.2626 |
| secukinumab | 1.3919 | 1.7420 | -0.3501 |

### -vir

| Ingredient | g2p PEU | stem PEU | improvement |
| --- | --- | --- | --- |
| bictegravir | 1.1076 | 0.7779 | +0.3298 |
| ensitrelvir | 1.0104 | 0.5000 | +0.5104 |
| islatravir | 4.6537 | 5.7656 | -1.1119 |

### -zumab

| Ingredient | g2p PEU | stem PEU | improvement |
| --- | --- | --- | --- |
| bevacizumab-vikg | 3.6729 | 4.7913 | -1.1184 |
| omalizumab | 0.6959 | 0.5377 | +0.1583 |
| teplizumab-mzwv | 0.7175 | 0.7837 | -0.0662 |

## Dropped stems

- `-umab`: average improvement -0.1731 PEU over 4 example(s) -- the stem engine did not measurably help (or measurably hurt), so it is not applied by `build.py` even though it stays documented in `usan_stems.py` for reference.
- `-vir`: average improvement -0.0906 PEU over 3 example(s) -- the stem engine did not measurably help (or measurably hurt), so it is not applied by `build.py` even though it stays documented in `usan_stems.py` for reference.
- `-zumab`: average improvement -0.3421 PEU over 3 example(s) -- the stem engine did not measurably help (or measurably hurt), so it is not applied by `build.py` even though it stays documented in `usan_stems.py` for reference.

## Untested stems

No generic in this dataset ends in these stems with a real external
source, so there is nothing to back-test them against. They stay in
`usan_stems.py`, documented and cited, and are applied by `build.py`
when they match -- but their confidence notes say plainly that they
are unverified in this dataset, not silently treated the same as a
back-tested stem.

- `-ciclovir`: USAN stem '-ciclovir', nucleoside antiviral analogues (e.g. acyclovir, ganciclovir): rhymes with 'fir', stress before the stem. Not present in this dataset; included for completeness.
- `-dronate`: USAN stem '-dronate', bisphosphonates (e.g. alendronate, risedronate): stress before the stem. Not present in this dataset; included for completeness.
- `-inib`: USAN stem '-inib' (kinase inhibitors not using the '-tinib' variant); same convention as '-tinib'.
- `-olol`: USAN stem '-olol', beta blockers (e.g. propranolol, atenolol): stress before the stem. Not present in this dataset; included for completeness.
- `-omab`: USAN substem '-omab', murine monoclonal antibodies (e.g. muromonab): stress on the syllable before it. Not present in this dataset.
- `-pril`: USAN stem '-pril', ACE inhibitors (e.g. captopril, lisinopril): stress before the stem. Not present in this dataset; included for completeness.
- `-ximab`: USAN substem '-ximab', chimeric monoclonal antibodies (e.g. rituximab, infliximab): the AMA/USAN pronunciation key notates the 'x' as /z/; stress on the syllable before it. Not present in this dataset with a source to test against.
- `-zole`: USAN stem '-azole', antifungal/other azole-ring drugs not using the '-prazole' PPI variant (e.g. riluzole): 'zole' rhymes with 'hole', stress on the syllable before it.

