# DOSE-R sign-off criteria — pre-registered

Written before any real TTS score exists in this repo, against the state
verified on 2026-09-18: dataset staged, gold references built (284
ingredients, confidence breakdown in `COVERAGE.md`), judge built and
stress-tested on synthetic perturbations only, strata reconstructed
(`STRATA.md`), fidelity harness built and tested on synthetic runs only
(`FIDELITY.md`), zero real TTS audio. These thresholds are fixed now so that
nobody sets them after looking at a number DOSE-R produces. If a future
version of this document changes a threshold, it must say what result
prompted the change and treat that as a red flag on the process, not just a
number update.

## What "trustworthy enough for Workstream 2" means here

Workstream 2 spends real money fine-tuning against DOSE-R's judge. Accepting
DOSE-R means accepting that a number it produces reflects real pronunciation
quality, not noise, not an artifact of an unverified reference, and not an
artifact of the aggregation or cutoff choices made in Workstreams 1a-1d.
Three independent gates, all numeric, all must pass.

## Gate 1 — the judge is calibrated against real, human-verified audio

**Current state.** The phonetic scorer separates known-good from known-bad
at AUC 0.998 (`artifacts/stress_test.json`), but every trial is a *synthetic*
perturbation of a reference (`dose_r/judge/stress_test.py`), not real speech.
This number calibrates nothing about real audio. `artifacts/anchor_set.jsonl`
stages exactly the right check — 50 items, stratified across difficulty
(17/17/16 easy/medium/hard), and drawn **entirely** from the lowest-confidence
reference tier, i.e. the 50 items where a wrong verdict is most likely — but
nobody has listened to them yet.

**Threshold to accept.** Once real audio exists and is scored:

- All 50 anchor-set items get a human pass/fail verdict against the actual
  audio (not against the rule-derived reference — a human listens and judges
  intelligibility/correctness directly).
- Automated-vs-human pass/fail agreement (Cohen's kappa, same statistic
  `dose_r/judge/agreement.py` already computes) is **≥ 0.60** ("substantial
  agreement" on the standard Landis-Koch scale) across all 50, and **≥ 0.60**
  within the "hard" difficulty subgroup specifically (n=16) — the judge
  passing on average while failing exactly where failures matter most would
  be a worse outcome than failing overall.
- Any item where phonetic and panel scores disagree by ≥2 or disagree on
  pass/fail (`agreement.py`'s existing `flagged` definition) that also
  disagrees with the human verdict gets manually inspected before sign-off;
  more than 5 of the 50 in that state is an automatic hold, not a judgment
  call.

**If this gate fails:** hold. Do not average around it, do not lower the
threshold, and do not substitute the synthetic AUC as a stand-in — that
number has already been computed and is not what this gate asks.

## Gate 2 — an observed gap clears the noise floor before being reported as a finding

**Current state.** `dose_r/report/FIDELITY.md` computes, from this repo's own
numbers, that a two-system or two-stratum pass-rate comparison at n=274 needs
roughly an **11-point gap** to clear 80% power at α=0.05 (unpaired, so this is
a conservative upper bound), and roughly **15-16 points** for the
established/new subgroup sizes (n≈127/147) this build produced. These are not
new assumptions for sign-off — they are arithmetic already in this repo, and
sign-off is refused if a report ignores them.

**Threshold to accept a specific claim:**

- A claim that "system A beats system B" or "system A drops N points from
  established to new" is reported as a finding only if the observed gap
  exceeds the relevant minimum-detectable-gap figure from `FIDELITY.md` for
  that comparison's sample size. Below that floor, the report must say "not
  distinguishable from noise at this n" — not a softened version of the
  claim.
- DOSE's own headline shape (systems drop 15-30+ points from established to
  new) is only reproduced if this rebuild's established/new gap, on real
  data, is itself ≥ 15 points **and** in the same direction. A reproduced gap
  of, say, 6 points is not "a smaller version of the same effect" at this
  sample size — it is noise, and must be reported as such.

**If this gate fails:** the specific claim is withheld, not asterisked. A
report with more asterisks than defensible numbers is worse than a shorter
report.

## Gate 3 — the reference layer's uncertainty is either resolved or explicitly bounded

**Current state.** At last count, roughly 165-185 of 284 gold references (170
of 274 rows once aggregated to a row's weakest ingredient, at the time this
was written -- re-check `COVERAGE.md` and `strata.jsonl`'s
`reference_confidence` field, since a parallel effort is actively adding
sources and this share is moving) carry no external confirmation. This is the
single biggest systematic risk named in this project's brief, and it does not
go away by building a good judge or a good harness on top of it; a
well-calibrated judge scoring against a wrong reference produces a confident,
wrong number.

**Threshold to accept:**

- Any headline pass-rate number is reported **twice**: once over all 274
  rows, and once restricted to the rows with medium-or-high confidence
  references (109 of 274, ≈40%, at last count -- recompute via
  `power.effective_sample_size` against the current `strata.jsonl`). If the
  two numbers disagree by more than the Gate 2 noise floor for that reduced
  n (≈17 points at n=109, narrower if the confirmed share keeps growing —
  recompute, don't reuse this figure once it does), the discrepancy is the
  headline finding, not a footnote, and Workstream 2 does not proceed on the
  274-row number alone.
- `attribution_by_reference_confidence` (already implemented and tested,
  `dose_r/report/fidelity.py`) is run on every real system's output. If the
  high-vs-low confidence tier gap is statistically significant (p < 0.05)
  *and* exceeds 10 points, that system's overall pass rate is reported as
  "partly attributable to unverified references" and is not used as-is to
  judge a fine-tuned model's real improvement in Workstream 2.
- The reference-uncertainty sensitivity bracket in `FIDELITY.md` (CI
  half-width widening from ±5.4 to ±6.1-7.0 points depending on an assumed
  10-30% gold error rate) is carried into any published CI, not silently
  dropped in favor of the sampling-only number.

**What would lift this constraint.** The 65% low-confidence share is a
property of the sources tried so far (Merriam-Webster and CMUdict; Drugs.com
returns 403, FDA labels carry no pronunciation data). Two things would
actually move it, and nothing else claimed to work should be trusted to:

1. **A second independent human-checkable source** for the 185 low-confidence
   names — e.g., systematic Wikipedia/Wiktionary respellings or IPA
   transcriptions (apparently in progress elsewhere in this repo as of this
   writing — see `dose_r/references/wiki_notation.py`), which would let
   ingredients currently at "low" (rule-only) move to "medium" (one real
   source) or "high" (two agreeing sources). This is the only currently
   visible path to a materially smaller low-confidence share.
2. **A measured gold-error rate**, replacing the assumed 10/30% bracket in
   `FIDELITY.md` with a real number: have a pharmacist or trained annotator
   independently transcribe a random sample of the low-confidence
   ingredients and compare to the rule-derived reference. Even n=30-40 would
   turn a guessed bracket into an estimated rate with its own CI.

Until one of those happens, Gate 3's double-reporting and attribution
requirements stay in force, permanently, not as a one-time check.

## Overall verdict rule

**Accept** DOSE-R as trustworthy enough for Workstream 2 only if all three
gates pass on a real, live-audio run: judge calibration (Gate 1), an
established/new gap that clears the noise floor in the expected direction
(Gate 2), and a reference-confidence attribution that does not show a
significant, sizeable low-tier penalty (Gate 3) — or, if it does, restrict
Workstream 2 to training/evaluating only against the 94-row
medium-or-high-confidence subset until Gate 3's remedies land.

**Reject or hold** if any gate fails outright: a real run with unresolved
judge miscalibration, a headline gap that doesn't clear its noise floor, or a
significant reference-confidence-driven gap with no restricted-subset
fallback reported. A held sign-off is not a failure of this workstream — a
benchmark that honestly says "not yet" is doing its job; one that says "yes"
under time pressure is not.
