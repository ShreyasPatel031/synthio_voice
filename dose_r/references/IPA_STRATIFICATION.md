# Stratified IPA vs plain — which switch actually beats plain

> **BANNED:** These IPA sidecars were G2P of dictionary respelling. Do not
> reuse `pronunciations.jsonl` `ipa` fields; they are now empty. See `README.md`.

Always-on IPA does **not** beat plain (0.723 vs 0.730). These switches
do, and none of them need a human clip.

Scored set: n = 174, Path 2 wavlm F1, Standard-C. Policy = IPA sidecar if
the condition holds, else plain spelling. Same sidecar as
`pronunciations.jsonl`.

## Winner

**`era=new` OR biologic suffix** → **0.741 (+0.011 vs plain)**.

`era=new` alone is **0.740 (+0.010)** and is the simpler rule. The extra
0.001 is four biosimilar suffixes (`-vikg`, `-csrk`, …), one of which is
established (`bevacizumab-vikg` 0.544→0.763).

| Policy | Mean | vs plain | n given IPA | Holdout-25 |
| --- | ---: | ---: | ---: | ---: |
| **new OR biologic_suffix** | **0.741** | **+0.011** | 64 | 0.663 |
| new OR multiword | 0.741 | +0.011 | 69 | 0.660 |
| **era=new** | **0.740** | **+0.010** | 63 | 0.654 |
| new+generic | 0.737 | +0.007 | 24 | 0.641 |
| year ≥ 2020 | 0.737 | +0.007 | 62 | 0.643 |
| plain | 0.730 | — | 0 | 0.580 |
| generic (all) | 0.729 | −0.001 | 67 | 0.656 |
| **always IPA** | **0.723** | **−0.007** | 174 | **0.681** |

Always-on still wins the tail. It loses the mean because established
brands go **0.764 → 0.740**.

## Where IPA helps, on the scored set

| Slice | n | Plain | IPA | Δ |
| --- | ---: | ---: | ---: | ---: |
| **new / generic** | 24 | 0.655 | 0.705 | **+0.050** |
| new / brand | 39 | 0.711 | 0.724 | **+0.014** |
| biologic suffix | 5 | 0.587 | 0.723 | **+0.137** |
| established / brand | 62 | 0.764 | 0.740 | −0.024 |
| established / generic | 43 | 0.740 | 0.710 | −0.030 |

That is the whole story. IPA is a win on **new generics**. It is noise or
a loss on household names the engine already says.

## Audio 60% vs no-audio 40%

Dose n = 274. Clips on disk: **169**. No clip: **105**.

| | With clip (the 60% we scored) | No clip (the 40% we will test) |
| --- | --- | --- |
| n | 169 | 105 |
| era=new | 63 (37%) | **84 (80%)** |
| established | 106 | 21 |
| approved 2025+ | 18 | **60** |
| have a sidecar IPA | 168 | 92 |

The holdout we can score is the **old** set. The 40% is mostly 2025
brands and new generics — the cell where IPA was **+0.050**.

Projected mean Δ on the 105 no-clip names, transferring the scored
`era × type` delta (not a new Path 2 number; those names have no clip):

| Policy | Applied | Expected Δ on no-clip |
| --- | ---: | ---: |
| plain | 0 | 0 |
| always IPA | 92 | **+0.019** |
| **era=new** | 75 | **+0.024** |
| new OR bio | 77 | +0.023 |
| new+generic | 41 | +0.020 |

On the 60% with audio, the same transfer says era=new **+0.010** (matches
the measured 0.740). Always-on is **−0.006** (matches 0.723).

So: selective IPA **does improve** the set we will actually test next,
and by more than it improved the scored 60%, because that 40% is newer.
Always-on IPA would also help the 40% and still hurt the 60%.

Suggested default: **attach IPA for every name; apply it when
`era=new` or `biologic_suffix`.** That is computable for all 274
(OpenFDA date + INN suffix). No audio gate.

## IPA check — what was fucked, what changed

The sidecar is **not** `references.jsonl` variant[0]. That snapshot still
has the old converter. The sidecar rebuilds IPA from the canonical
respelling with the ye-fix (`wiki_notation`: syllable-final `ye` → /aɪ/).

| Name | Stored first-IPA (old) | Sidecar now | Why |
| --- | --- | --- | --- |
| tofacitinib | `toʊfæˈsjɛtɪnɪb` | `toʊfæˈsaɪtɪnɪb` | `sye` was /jɛ/, now /aɪ/ |
| omalizumab | `oʊmɑːljɛˈzuːmæb` | `oʊmɑːlaɪˈzuːmæb` | `lye` same bug |
| upadacitinib | `juːpædæˈsjɛtɪnɪb` | `juːpædæˈsaɪtɪnɪb` | `sye` |
| Vabysmo | `vɑːˈbjɛsmɒ` | `vɑːˈbaɪsmɒ` | `bye` |
| Zycubo / Zaiidra | `zjɛ…` | `zaɪ…` | `zye` |

14 stored first-IPAs still contain `jɛ`. **0** of the 10 `sye`/`lye`/`zye`
canonicals do in the sidecar (all have `aɪ`). 80 of 278 strings changed;
198 already matched.

What the ye-fix did not catch (source string / letter-by-letter G2P).
The CTC pass below fixed the first three; Humira is still a bad
DailyMed string (`HU-mare-ah` stresses the wrong syllable).

- **Nuzolvence** was `vence` → `vɛnsɛ` (now `vɛns`)
- **Revuforj** was `you`/`forge` → `jɒʌ`/`ɡɛ` (now `juː`/`dʒ`)
- **famotidine** MW lists both `dīn` and `dēn`. We now keep official
  `deen` (`fʌˈmoʊtʌdiːn`). IPA F1 0.618 → 0.711, still below plain 0.801.
- **Humira** DailyMed `HU-mare-ah` still stresses the wrong syllable

Always-on still loses because leftover source-string junk lands on names
plain already said well. The ye-bug that made Path 4 look insane on
tofacitinib / omalizumab is fixed in this field.

## One CTC iteration (wav2vec2-lv-60-espeak on the human clip)

Not a gold IPA. Used only to catch converter bugs. CTC also heard
`Advair Diskus` and extra tails on Nurtec — those are clip problems, not
rules to copy.

| Name | Old sidecar | CTC human | New sidecar | IPA F1 |
| --- | --- | --- | --- | ---: |
| Nuzolvence | `…vɛnsɛ` | `nuːzoʊvɛnts` | `…vɛns` | 0.595 → **0.735** |
| Revuforj | `ɹɛvjɒʌfɔːrɡɛ` | `ɹavifoːdʒ` | `ɹɛvjuːfɔːrdʒ` | 0.532 → **0.757** |
| Ubrelvy | `jɒʌ…` | — | `juː…` | 0.692 → **0.780** |
| famotidine | `djɛn` → `daɪn` | `diːn` | `diːn` (official `deen`) | 0.618 → **0.711** (plain 0.801) |

Rules added: syllable `you` → /juː/; trailing `nce` silent-e; trailing
`rge`/`Vge` → /dʒ/; `CyeC` (`dyen`) → /aɪ/. Always-on IPA moved
**0.720 → 0.723** (−0.010 → −0.007). Still below plain 0.730.
`era=new` / `new|bio` remain the mean-winning switches (0.740 / 0.741).

Machine copy: `ipa_stratification.json`.
