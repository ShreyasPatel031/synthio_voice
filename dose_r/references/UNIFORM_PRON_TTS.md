# Uniform pronunciation field vs plain TTS

Path 2 gold, Cloud TTS `en-US-Standard-C`, n = 174 names with a human clip.
The 40% without clips cannot use a score-gated switch, so both arms here are
derived only from the canonical dictionary respelling.

The spoken *text* on the IPA arm is still the real spelling. Compact replaces
the spelling with a one-token ASCII string.

| Arm | What is synthesized | Mean F1 | vs plain | Holdout-25 mean |
| --- | --- | ---: | ---: | ---: |
| Plain spelling | `Wegovy` | **0.730** | — | 0.580 |
| Spaced respelling (previous) | `wee goh vee` | 0.650 | −0.080 | (pauses) |
| Compact ASCII | `weegohvee` | 0.695 | −0.035 | 0.627 |
| **IPA sidecar** | `Wegovy` + `wiːˈɡoʊviː` | **0.723** | **−0.007** | **0.681** |

Duration: plain 1.07s, compact 1.07s, IPA 1.07s, spaced 1.59s. The sidecar
does not insert syllable pauses. That is why it is the first pronunciation
channel that does not destroy the names the engine already knows *by
length*. The remaining −0.007 mean is leftover conversion noise (Humira,
acetaminophen) plus names the engine already says (famotidine official
`/diːn/` sidecar 0.711 vs plain 0.801). One CTC pass on the human clips
fixed `you`/`vence`/`forge` (Revuforj 0.532→0.757).

## The worst 25 (the actual gap)

IPA sidecar **+0.100** on this set. 18 / 25 improve by >0.05. Names still
below 0.60: 18 → 5.

| Plain | Compact | IPA | Name |
| ---: | ---: | ---: | --- |
| 0.473 | 0.502 | **0.711** | acoramidis |
| 0.537 | 0.620 | 0.647 | atogepant |
| 0.539 | 0.676 | **0.717** | fluticasone propionate |
| 0.542 | 0.542 | 0.554 | trospium chloride |
| 0.544 | 0.652 | **0.763** | bevacizumab-vikg |
| 0.559 | **0.752** | 0.738 | sotatercept-csrk |
| 0.560 | 0.588 | **0.695** | prademagene zamikeracel |
| 0.566 | 0.566 | 0.522 | Advair (clip/pace) |
| 0.568 | 0.572 | **0.674** | Aucatzyl |
| 0.570 | 0.621 | **0.755** | lebrikizumab-lbkz |
| 0.577 | 0.612 | **0.686** | datopotamab deruxtecan |
| 0.579 | 0.579 | **0.753** | Vraylar |
| 0.587 | 0.593 | **0.712** | tovorafenib |
| 0.588 | 0.657 | **0.689** | Xeljanz |
| 0.589 | 0.590 | 0.591 | Nurtec (clip/pace) |
| 0.589 | 0.552 | 0.580 | tenofovir alafenamide |
| 0.591 | 0.601 | 0.550 | testosterone undecanoate |
| 0.599 | 0.594 | **0.745** | fezolinetant |
| 0.607 | **0.777** | 0.751 | remibrutinib |
| 0.615 | 0.621 | **0.763** | nipocalimab-aahu |
| 0.618 | 0.632 | 0.663 | Imaavy |
| 0.620 | **0.744** | 0.687 | Wegovy |
| 0.627 | **0.734** | 0.726 | Meibo |
| 0.628 | 0.569 | 0.636 | upadacitinib |
| 0.635 | **0.730** | 0.705 | Vabysmo |

Advair / Nurtec / the two-word salt clips still do not move. Those are
reference-clip problems, not a missing alphabet.

## What to feed the model

Keep the clinical sentence (or the ingredient name) as written. Attach one
IPA string per name. On Cloud TTS that is
`customPronunciations.pronunciations[].pronunciation` with
`phoneticEncoding=PHONETIC_ENCODING_IPA` and `phrase` equal to the spelling.
The same IPA string is the field for the 40% with no audio.

Do **not** put the dictionary respelling in the sentence. Spaces and hyphens
are word boundaries to this engine.

`dose_r/references/pronunciations.jsonl` has `ipa` / `ipa_cloud` / `compact`
for 278 / 284 names (the same six unsourced names stay empty). `ipa_cloud`
only folds British `ɪə`/`ʊə` so en-US customPronunciations will accept the
string (Lyrica `ˈlɪərɪkɑː` → `ˈlɪrɪkɑː`).

Rebuild: `python3 scripts/build_tts_pronunciations.py`
Eval: `python3 scripts/eval_uniform_pron_tts.py`

## If the mean must not fall

Always-on IPA costs −0.007 because converted IPA overrides a good built-in
G2P on household names. A switch that does **not** need audio: apply the
sidecar only when strata `era=new` (OpenFDA approval date). That is available
for the 40% too. Measured ranking of those switches is in
`IPA_STRATIFICATION.md`.

| Policy | Mean | Holdout-25 |
| --- | ---: | ---: |
| always plain | 0.730 | 0.580 |
| always IPA | 0.723 | **0.681** |
| IPA only on `era=new` (n=63) | **0.740** | 0.654 |

`era=new` is the only uniform, no-audio policy that beats plain on the mean.
It misses established-but-opaque brands (Vraylar, Xeljanz, Wegovy). Always-on
IPA is the one that actually closes the tail.
