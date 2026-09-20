# Drugs.com audio collection — handoff

Reference pronunciation audio has to be collected from a machine that can reach
drugs.com. The cloud workspace cannot: both the pages and the audio files return
403 there, to curl with a full browser header set and to a real headless
Chromium alike. The block is on the network path rather than the session, so
signing in does not change it.

This is worth the manual effort because the audio is the strongest reference we
can get. Everything else in the pipeline compares *phoneme transcriptions* — a
dictionary respelling we convert to ARPABET, against a transcription of the TTS
output — and both conversions inject error of our own making that is then
indistinguishable from model error. A human recording removes that layer. DOSE
describes its own method as "automated audio comparison", so this also moves the
rebuild onto the same footing as the thing it is replicating.

## Where the first pass got to

274 DOSE rows contain 284 unique ingredients. Of those, 246 were attempted:

| Outcome | Count | Meaning |
| --- | --- | --- |
| Audio collected | **127** | All valid WAV, 0.30–4.00s. Nothing suspect. |
| HTTP 404 | **76** | **Not absent — wrong URL guessed.** Retry these. |
| 200, no audio on page | 43 | Genuinely no recording. Mostly 2024–26 approvals. |
| Never attempted | 38 | Run ended before reaching them. |

The 127 are already ingested and verified. The 76 are the recoverable prize.

## What went wrong, and the fix

The first collector only ever tried `https://www.drugs.com/{slug}.html`. Many
drugs are not there. Osimertinib, for example, lives at
`https://www.drugs.com/mtm/osimertinib.html`, and generic INN names skew heavily
toward `/mtm/`. Every one of those 76 was recorded as a 404 without the other
paths being tried at all.

`scripts/drugscom_collect_v2.js` tries each known path in turn — root, then
`mtm/`, `pro/`, `cdi/`, `monograph/`, `npp/`, `international/` — and falls back
to the site search, following the first drug result. It targets the 114 names
that are worth another attempt (the 76 plus the 38 never reached) and skips
everything already collected.

## Running it

Open drugs.com signed in, then DevTools console (F12), and paste the whole file.

- It logs which path each hit came from (`AUDIO via mtm/`), which will show
  quickly whether one path dominates.
- **Nothing is written to disk until you ask for it.** Results accumulate in
  `window.DOSER`. Run `dumpDoseR()` at any point — including while the loop is
  still running — to download a snapshot. Do this every 30–50 drugs; if the tab
  closes or navigates away, everything in memory is lost.
- Keep `DELAY_MS` at 1500. A burst of rapid requests is what trips bot detection,
  and the account doing the browsing is the one at risk.
- 114 names against up to 7 paths each puts this around 20–30 minutes.

Send back the downloaded JSON. The importer already handles this shape, verifies
every clip, and merges into the shared manifest.

## Rules that must hold

**No pronunciation may ever be generated, guessed, or synthesised.** This corpus
is human recordings only. Once written down, a plausible guess is
indistinguishable from a real one, and everything downstream treats this layer as
ground truth. A name with no recording stays empty and is reported as a gap. That
is the wanted outcome, not a failure.

**Audio must speak the name the row asks about.** DOSE scores the drug name as it
appears in the carrier sentence, so a brand-name row needs the brand spoken, not
its generic. This was confirmed by ear for Anktiva, whose page also lists
"nogapendekin alfa inbakicept" — the clip says "Anktiva". That check has not been
repeated across the set, so treat any clip whose duration looks wrong for the
name as suspect rather than assuming.

## Known dead ends — do not spend time here

- **DrugBank** — 403, same class of block.
- **FDA labels** (openFDA, DailyMed) — reachable, and carry no pronunciation data
  of any kind. Useful only for approval dates.
- **NCI Drug Dictionary** — a single-page app whose backing API host does not
  resolve publicly.
- **AMA USAN Drug Finder** — a single-page app with no pronunciation data in its
  bundle.
- **WHO INN lists** — name and CAS registries only, no phonetics.
- **MedlinePlus** — this one *does* carry pronunciations (`pronounced as (met
  for' min)`) and is reachable, but it uses its own lay respelling notation with
  no published key. Decoding it by eye would mean inventing a letter-to-phoneme
  mapping and labelling the result "sourced". Blocked on NLM or ASHP documenting
  that convention; not on effort.

## After this pass

Remaining coverage gaps for names with no recording anywhere fall to the paid or
licensed sources, in order of what they would close:

1. **USP Dictionary of USAN and International Drug Names** — compiled official
   pronunciations for nearly every INN generic, using the AMA/USAN key, which is
   itself public. Hits precisely the coined names that dominate the gap.
2. **A citable key for the MedlinePlus/AHFS respelling**, which unlocks a free
   and already-reachable source.
3. **A Drugs.com or DrugBank data licence**, which would replace this manual
   collection entirely.
