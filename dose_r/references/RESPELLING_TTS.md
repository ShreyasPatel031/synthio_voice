# Cloud TTS + dictionary respelling vs plain spelling

Path 2 gold: `microsoft/wavlm-large` SpeechBERTScore F1 vs the same human
clip. Voice: Cloud TTS `en-US-Standard-C`. No IPA.

Overlap: **174** ingredients with both a human clip and a canonical
respelling (`dose_r/references/respellings.jsonl`).

## Verdict

**Dictionary respelling does not beat plain spelling as a blanket TTS input.**

| Arm | Mean F1 vs human | vs plain |
| --- | ---: | ---: |
| Plain spelling (`Nuzolvence`) | **0.730** | — |
| Respell, hyphens → spaces (`nu zol vence`) | 0.650 | **−0.080** |
| Respell, lowercase hyphens (`nu-zol-vence`) | 0.646 | −0.084 |

Spaced beats plain by >0.01 on **37 / 174**. It loses by >0.01 on **125 / 174**.
Hyphen and spaced F1 are identical on 144/174: this voice treats `-` like a
space.

SSML `<sub alias="…">` and keeping ALL-CAPS stress (`PLAV iks`) were tried on
the 12 worst losses. Neither beat spaced. Caps was slightly worse.

## Why it loses

Two stacked effects.

### 1. Syllable pauses (the big one)

Mean duration: human 1.35s, **plain 1.07s**, spaced **1.59s**.
Pearson(duration_ratio, ΔF1) = **−0.56**.

Cloud TTS G2P reads `plav iks` / `uh see tuh mih nuh fen` as **several English
words**, with pauses. The human clip is one word. Wavlm F1 punishes the extra
length even when the vowels are closer.

Concatenating the syllables (`plaviks`, `aspihrin`) on the 12 worst losses
almost restores plain:

| Name | Plain | Spaced | Concat (no spaces) |
| --- | ---: | ---: | ---: |
| Plavix | 0.791 | 0.363 | **0.788** |
| Aspirin | 0.783 | 0.470 | 0.732 |
| aripiprazole | 0.804 | 0.560 | **0.804** (concat is the spelling) |

So those losses are mostly **how we fed the string**, not a wrong
pronunciation in the dictionary.

### 2. The TTS lexicon already knows common names

Household brands (Aspirin, Plavix, Xanax, …): mean Δ **−0.15**.
NCI/MW-primary generics (acetaminophen, fluoxetine): mean Δ **−0.13 to −0.15**.
The engine's own G2P for those spellings is already close to the human clip.
A lay respelling can only add pauses and odd graphemes (`fuh moht uh dyen`).

## Where it *does* help

Opaque / newly coined names whose **letters** the G2P misreads. Top gains:

| Name | Plain | Spaced | Fed as |
| --- | ---: | ---: | ---: |
| Vraylar | 0.579 | **0.798** | `vray lar` |
| sotatercept-csrk | 0.559 | **0.752** | `soe tat er sept` |
| atogepant | 0.537 | **0.713** | `a toe je pant` |
| Xeljanz | 0.588 | **0.719** | `zel jans` |
| Wegovy | 0.620 | **0.737** | `wee goh vee` |
| Meibo | 0.627 | **0.734** | `my boh` |

Concatenating those wins often **erases** them: `vraylar` scores 0.579 again
(the spelling). The help is specifically “read these English syllables,” not
“paste the letters together.”

## What this means

Dictionary respelling is the right **gold string** (matches what DOSE
publishes). It is the wrong **universal Cloud TTS control**.

- Do not replace every name with `uh see tuh mih nuh fen`.
- Do inject a spaced respelling when plain G2P is the failure (coined brands).
- IPA SSML is a different, still-broken channel (see earlier Path 4).
- A compact one-token respelling (`ayripiprayzole`) is closer to what Kokoro
  accepted; concatenating NCI syllables is not that (it becomes
  `uhseetuhmihnuhfen`).

Rebuild: `python3 scripts/eval_respelling_tts.py`
Output: `runs/plain-vs-respelling-tts/results.json` (gitignored audio).
Listen: `runs/listen-plain-vs-respelling/index.html`.
