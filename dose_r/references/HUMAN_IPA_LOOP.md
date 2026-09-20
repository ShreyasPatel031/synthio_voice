# Human → IPA → name+sidecar loop (5 rounds)

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
