# Plain TTS vs human — where the gap actually is

> Closing this gap is **not** "convert the dictionary string to IPA."
> That converter is banned. See `README.md`.

Voice: Cloud TTS `en-US-Standard-C`. Metric: wavlm-large F1 vs the same
human clip (Path 2). **n = 174** names with both a clip and a respelling.
This run scored **Drugs.com clips only** (Merriam-Webster is preferred by
`available_clips()` but 0 MW files were on disk; 81/174 have an MW record
in the manifest).

A “good” match on this metric is about **0.80+** (Januvia 0.848, p75 =
0.794). Mean plain F1 is **0.730**. That **−0.070** mean deficit to the
good band is not a blanket-format problem. **70% of names already sit at
0.70+.** The usable signal is a tail, plus a few structural slices.

## Size of the gap

| Band | n | Share of n | Share of deficit to 0.80 |
| --- | ---: | ---: | ---: |
| ≥ 0.80 | 38 | 22% | 0% — already matches |
| 0.70–0.80 | 84 | 48% | 29% — close; not a new alphabet |
| 0.60–0.70 | 34 | 20% | **38%** — mild miss; some letter-to-sound |
| **< 0.60** | **18** | **10%** | **33%** — the sharp tail |

- Mean 0.730, median 0.750, p10 **0.599**, stdev 0.082.
- Lowest: **acoramidis 0.473** (0.33 below a good match). Highest: Januvia 0.848.
- The 18 names below 0.60 average **0.564**. The other 156 average **0.749**.
- Raising only the lowest 25 to 0.80 would lift the mean **+0.032** (to 0.761).
  That is the maximum this holdout can buy. Oracle `max(plain, spaced)`
  already tried on all 174 only lifts **+0.016**.

### Where the tail lives (means)

| Slice | n | Mean F1 |
| --- | ---: | ---: |
| established | 105 | 0.754 |
| new | 63 | **0.689** |
| brand | 101 | 0.743 |
| generic | 67 | 0.710 |
| one word | 163 | 0.738 |
| multi-word | 11 | **0.617** |
| not biosimilar | 169 | 0.734 |
| biosimilar suffix (`-vikg`, `-csrk`, …) | 5 | **0.587** |

Those slices say *where* to look. They are not a reason to rewrite the 105
established brands.

## Why IPA / dictionary respelling cannot blanket-help

Dictionary respelling is a **label**, not a TTS encoding. There is no
canonical phonetic replica of `ak-oh-RAM-id-is` that every engine will
read as one word:

- Spaces → several English words with pauses (mean duration 1.07s plain vs
  1.59s spaced; Pearson(dur_ratio, ΔF1) = −0.56).
- Hyphens ≈ spaces on Standard-C.
- CAPS may be spelled letter-by-letter.
- Invented graphemes (`soe`, `ue`, `sye`) go through English G2P, not a
  dictionary.

That is why blanket spaced respelling **loses −0.080** mean, and why IPA
SSML never became a peer of Path 2: both are trying to ship a transcription
through a channel that does not preserve it.

The only format that has moved F1 is **2–4 ordinary English syllables with
spaces**, and only on names whose **letters** the engine already misreads
(`Xeljanz`→ex-el-janz, `Wegovy`→weg-oh-vee, `Vraylar`→vuh-ray-lar).

## The 25 largest gaps (lowest plain F1)

Two buckets. Spaced dictionary respelling was already tried on all of them.
Machine copy: `plain_tts_gap_holdout.json`.

### A — letter-to-sound (16/25). Short English syllables already help.

Mean plain **0.571** → spaced **0.677** (Δ **+0.106**).

| Plain | Spaced | Name | Fed as | Note |
| ---: | ---: | --- | --- | --- |
| 0.473 | 0.559 | acoramidis | `ak oh ram id is` | clip may be a valid alternate; still the lowest score |
| 0.537 | **0.713** | atogepant | `a toe je pant` | |
| 0.539 | 0.630 | fluticasone propionate | `floo tik uh sohn` | clip likely stem-only; second word dropped |
| 0.544 | 0.560 | bevacizumab-vikg | `beh vuh sih zoo mab` | suffix not in the string |
| 0.559 | **0.752** | sotatercept-csrk | `soe tat er sept` | suffix dropped; stem letters were the bug |
| 0.560 | 0.663 | prademagene zamikeracel | `pra dem a jeen zam i ker a sel` | |
| 0.568 | 0.651 | Aucatzyl | `aw kat zil` | |
| 0.570 | 0.666 | lebrikizumab-lbkz | `leb ri kiz ue mab` | |
| 0.577 | 0.642 | datopotamab deruxtecan | `da toe poe tah mab der ux tee kan` | |
| 0.579 | **0.798** | Vraylar | `vray lar` | concat `vraylar` erases the win |
| 0.587 | 0.625 | tovorafenib | `toe voe raf en ib` | |
| 0.588 | **0.719** | Xeljanz | `zel jans` | |
| 0.599 | 0.696 | fezolinetant | `fez oh lin e tant` | |
| 0.615 | 0.694 | nipocalimab-aahu | `nip oh kal i mab` | |
| 0.620 | **0.737** | Wegovy | `wee goh vee` | |
| 0.627 | **0.734** | Meibo | `my boh` | clip flagged long, but letters also fail |

Same mechanism just above the cutoff (not in the 25): Bizengri 0.643→0.791,
Alhemo 0.636→0.748, Vyvgart 0.640→0.743. Include those if the bakeoff needs
more coined-brand signal.

### B — spaced respelling does not help (9/25). Different failure.

Mean plain **0.596** → spaced **0.534** (Δ **−0.062**).

| Plain | Spaced | Name | Likely issue |
| ---: | ---: | --- | --- |
| 0.542 | 0.480 | trospium chloride | two words; `chloride` left un-respelt |
| 0.566 | 0.553 | Advair | human 1.67s vs TTS 0.80s; clip flagged, resolved as slow Advair |
| 0.589 | 0.554 | Nurtec | human 1.95s vs TTS 0.81s; clip flagged, unresolved |
| 0.589 | 0.483 | tenofovir alafenamide | converter split `vi r`; clip likely stem-only |
| 0.591 | 0.432 | testosterone undecanoate | two words; clip flagged short |
| 0.607 | 0.534 | remibrutinib | 5-token `rem i broo tin ib` worse than the spelling |
| 0.618 | 0.625 | Imaavy | human 1.34s vs TTS 0.79s |
| 0.628 | 0.590 | upadacitinib | `sye` token + 6 syllables |
| 0.635 | 0.560 | Vabysmo | `vah bye smo` vs a 3-syllable brand |

Six of the 25 used a Drugs.com clip the audio pipeline already flagged.
Three of those (`fluticasone propionate`, `tenofovir alafenamide`,
`testosterone undecanoate`) look like **the clip omits the second word**.
A new phonetic format cannot beat a reference that did not say the name.

## What I would do next

Work **only on these 25** (plus the five extra coined brands if needed).
Do not convert to IPA. Do not feed the canonical dictionary string as TTS
except as one arm.

1. Tag clip-suspect rows (Advair, Nurtec, the three short multi-word clips,
   acoramidis) so a format “loss” is not treated as a G2P loss.
2. On the remaining letter-to-sound names, try **four ASCII inputs**, same
   voice, same human clip:

   | Arm | Wegovy | acoramidis |
   | --- | --- | --- |
   | plain | `Wegovy` | `acoramidis` |
   | short syllables (best so far) | `wee goh vee` | `ak oh ram id is` |
   | compact (no spaces) | `weegohvee` | `akohramidiss` |
   | English words only | `wee go vee` | `ack oh ram ih diss` |

   For multi-word / `-csrk` names: a fifth arm that **says the stem only**
   (`sotatercept`, `bevacizumab`, `fluticasone`) matching what the clip
   actually contains.

3. Keep the dictionary respelling as the gold *label*. Use it as a TTS
   string only when it is already short (2–4 syllables) and the letters
   are the bug. That is bucket A, and we already know that shape moves F1
   a lot (Vraylar 0.58→0.80, atogepant 0.54→0.71). Bucket B needs ears and
   cleaner strings, not a new alphabet.
