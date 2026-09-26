# Pronunciation references — original source only

**Banned: converting DailyMed / USAN / NCI / canonical ASCII respelling
into IPA.** That is how this repo produced garbage sidecar IPA and wasted
eval time.

## What happened

Wikipedia's respelling key is not DailyMed's. Feeding `DU-pix-ent` through
it maps `u` → `/ʌ/` ("duh"). The PPI means DU like "do" / "dual".
`ah-troo-be` became `ˈɑːtruːbɛ` (wrong stress, `/ɑː/`, final `/ɛ/`). Google
`æˈtruːbi` / `əˈtɹuːbi` and `ˈduːpɪksɛnt` beat that converter on every
tested name (see `runs/listen-google-ipa/`).

`respell_to_arpabet_ipa` and `ipa_from_canonical` now raise
`RespellToIpaBanned`. Do not un-raise them. Do not write a replacement G2P.

## What to use

**Injection and finetune:** `data/gold_gemini_ipa/` only (validated IPA + Gemini 3.1 Kore wav). That folder is write-locked. Never regenerate the wavs or edit the IPA unless the user explicitly approves that change and a reason is appended to `changes.jsonl`. Do not use experiment `runs/` copies or the old Cloud finetune pack.


| Source published | Store and feed the model |
| --- | --- |
| IPA (`/ˈduːpɪksɛnt/`, `{{IPA}}`, Wiktionary) | That IPA, unmodified except Cloud en-US inventory folds |
| Respell (`DU-pix-ent`, `ah-troo-be`) | That respelling. Not IPA invented from it |
| Nothing | Leave empty. Do not invent |

When you look something up (Gemini, Google), **ask for IPA**. Do not take
`fa-MOE-ti-deen` and convert it.

## Files

- `respellings.jsonl` — homogenized **original** dictionary respelling. Keep.
- `pronunciations.jsonl` — `ipa` / `ipa_cloud` are empty on purpose.
- `references.jsonl` — snapshot; many `ipa_variants` were produced by the
  banned converter. **Do not inject them into TTS.** Do not treat them as
  gold IPA. Prefer `sources[].raw` (what the page actually printed).
- `wiki_notation.py` — may still *parse* source-published IPA (`{{IPA}}`,
  `{{IPAc-en}}`). Must not G2P respelling.
