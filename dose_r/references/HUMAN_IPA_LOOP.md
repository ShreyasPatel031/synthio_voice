# Human → IPA → name+sidecar loop (5 rounds)

> **BANNED as a way to invent IPA.** Do not convert respelling to IPA and
> do not paste CTC as gold. Use the original source string. See `README.md`.

Path 2 gold. Spoken text stays the real name. IPA is only a sidecar.
CTC (`wav2vec2-lv-60-espeak`) proposes phones; a proposal is **kept
only if wavlm F1 vs the human clip rises by >0.005**. Clip tails are
windowed to the sidecar length so Advair Diskus is not copied.

This can run only on names that have a clip. It cannot seed the 40%
with no audio.

## Set

n = 35: the 25-name plain-vs-human holdout plus the 10 worst
IPA-vs-plain losses (Humira, Nexium, Adquey, …).

| | Mean F1 |
| --- | ---: |
| Plain | 0.643 |
| Current sidecar (ipa0) | 0.675 |
| Best of 5 loop rounds | **0.708** (+0.033 vs ipa0) |
| Holdout plain | 0.580 |
| Holdout ipa0 | 0.681 |
| Holdout loop | **0.701** (+0.021 vs ipa0) |

13 / 35 improved. 22 did not, including the names the holdout was
built for: acoramidis, Advair, Nurtec, atogepant, Adquey,
acetaminophen.

## What actually moved

Real phone fixes (human and a plausible English string):

| Name | ipa0 | Loop | Plain | New IPA |
| --- | ---: | ---: | ---: | --- |
| Humira | 0.626 | **0.785** | 0.803 | `huːmɛrə` (DailyMed `HU-mare-ah` was wrong) |
| Nexium | 0.654 | **0.785** | 0.824 | `nɛksiʌm` (isolated `e` is /i/, not /ɛ/) |
| Dupixent | 0.589 | **0.704** | 0.715 | `duːpɛksɛnt` |
| Eliquis | 0.659 | **0.757** | 0.772 | `æləkwɪs` |
| Jardiance | 0.667 | **0.721** | 0.793 | `dʒɑːdiəns` |
| Xeljanz | 0.689 | **0.728** | 0.588 | `sɛldʒænz` |
| tovorafenib | 0.712 | **0.725** | 0.587 | stress moved |

Those close the sidecar-vs-human hole. They still lose to plain on
household names the engine already says (Humira, Nexium, Dupixent,
Eliquis, Jardiance).

## What looks like a win and is not

The clip is only the first word. CTC windowed to that word. F1 jumped
because the sidecar had been scored against a two-word name the
speaker never said:

| Name | ipa0 → loop | What CTC wrote |
| --- | --- | --- |
| testosterone undecanoate | 0.550 → 0.802 | `tɛstɑːstərroʊn` (no undecanoate) |
| trospium chloride | 0.554 → 0.724 | `trɛsbiʌm` (no chloride) |

CTC junk the keep-if-F1-up rule still accepted:

| Name | New IPA |
| --- | --- |
| Vabysmo | `tpˈaɪsmaʊ` |
| aripiprazole | `ðˈærəpɪpərzəll` |

Those are not written into `pronunciations.jsonl`.

## Always-on 174

| Policy | Mean |
| --- | ---: |
| Plain | **0.730** |
| Current sidecar | 0.723 |
| Swap all 35 loop-bests (includes clip-span) | 0.730 (tie) |
| Swap only the 8 real phone fixes | 0.727 |

The tie is the two-word clip artifact. The real phone fixes do **not**
beat plain.

## What this is for

Use the human clip to **choose or repair a sidecar on names that have
audio**. Do not treat CTC as an IPA author, and do not run this as
the field for the 40% with no clip.

`python3 scripts/iter_human_ipa.py --rounds 5`
Machine copy: `runs/human-ipa-loop/results.json` (gitignored).

## Pass 2 — residual + official variants

Same loop, plus: Cloud sanitize (drop vowel-less junk, keep multiword
spaces, retry schwa 400s), one-phone residual, and **official IPA
variants from `references.jsonl` as proposals**. Human clip only
selects. n = 94 names where sidecar still lost to plain or sat below
0.70.

| | Mean F1 |
| --- | ---: |
| Those 94, plain | 0.740 |
| Those 94, sidecar | 0.702 |
| Those 94, loop-best | **0.751** |

Household names closed by picking the official string the clip already
used (not CTC authoring):

| Name | Sidecar | Official pick | Plain |
| --- | ---: | ---: | ---: |
| acetaminophen | 0.687 | **0.821** `əˌsiːtəˈmɪnəfən` | 0.821 |
| Januvia | 0.730 | **0.848** `dʒəˈnuːviːə` | 0.848 |
| Aspirin | 0.726 | **0.818** | 0.783 |
| Spiriva | 0.708 | **0.780** `spɪˈriːvə` | 0.820 |
| aripiprazole | 0.672 | **0.830** | 0.804 |
| Advair | 0.522 | **0.566** `ˈædˌvɛr` | 0.566 |

The hard holdout did **not** close:

| Name | Best | Plain | Why |
| --- | ---: | ---: | --- |
| acoramidis | 0.711 | 0.473 | sidecar already wins; human CTC is `/dʒ/` junk |
| Nurtec | 0.591 | 0.589 | clip tail; proposals worse |
| atogepant | 0.647 | 0.537 | USAN already in sidecar |
| Adquey | 0.623 | 0.783 | source `AD-kee`, clip `/adkwaɪ/` |
| Imaavy / Aucatzyl / Wegovy | unchanged | — | residual did not beat sidecar |

## Always-on 174 after both passes

| Policy | Mean |
| --- | ---: |
| Plain | 0.730 |
| Current sidecar | 0.723 |
| Official-variant picks only (49 names) | **0.743** |
| Oracle any-keep (includes CTC junk) | 0.755 |
| Holdout-25 official+real | 0.706 (was 0.681) |

**Yes, always-on can beat plain**, if the sidecar is the official
variant the human clip selects. That is not a new alphabet and not
CTC. It still cannot run on the 40% with no clip (those names keep
the first official string). The coined-name tail (acoramidis already
0.711; Nurtec/Adquey clip fights) is not closed by more iterations.

Pass 2 machine copy: `runs/human-ipa-loop2/results.json`.
`python3 scripts/iter_human_ipa.py --rounds 5` (open-gap default).
