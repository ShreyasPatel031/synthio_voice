# Replication-fidelity report — harness and status

**This has not been run on real data. No real TTS audio exists in this repo.**
`runs/` holds only `mock-slow/`, a mock backend's output; `runs/` and
`artifacts/` are gitignored, so nothing recoverable was lost to git, but
nothing recoverable exists either. Producing real scores means a live,
billable TTS pass (Gemini, at minimum, to check the one published anchor with
a full established/new breakdown) that nobody has authorised. This document
describes a harness that is ready the moment that changes, not a result.

## What exists

| Module | Purpose |
| --- | --- |
| `stats.py` | Wilson CI, two-proportion z-test, Spearman's rho with ties, inverse-normal quantiles. No scipy dependency; validated against known values in `tests/test_fidelity.py`. |
| `fidelity.py` | Loads judged `rows.jsonl` output, computes pass rate (overall / by era / by difficulty / by name_type), gaps vs. published numbers, rank correlation, and two attribution methods. |
| `power.py` | Sampling-noise power analysis and reference-layer uncertainty propagation. |
| `synthetic.py` | Generates synthetic strata + judged runs with planted, known effects, used by every test in `tests/test_fidelity.py`. |
| `build_report.py` | CLI: `python -m dose_r.report.build_report --run system_id=path/to/rows.jsonl [--run ...]` produces one JSON report from any number of judged runs. Run with zero `--run` arguments and it refuses, loudly, rather than emitting an empty report. |
| `published_leaderboard.json` | The only three DOSE numbers this workstream actually has (see below), each with a provenance note. |

All of it is exercised only against synthetic input (see "Testing philosophy"
below) — that is the honest state of a harness with no real run to point at
yet, not a shortcut.

## The published anchor is thin: 3 of 9 systems

DOSE ranks 9 TTS systems. This workstream was given three numbers:
RxPronounce 91.2% (rank 1), Azure DragonHD 63.1% (rank 9), and Gemini TTS at
74.5% overall / 89.1% established / 61.6% new (the only system with a
published stratum breakdown). The other six systems' names and scores were
never provided and are not recorded anywhere in this repo. Consequences:

- **Rank correlation across "the 9 published systems" cannot be computed.**
  `fidelity.rank_correlation` requires at least 4 systems in common before it
  will report a Spearman rho at all (fewer than that, a correlation
  coefficient is decoration, not evidence), and only 3 are available.
- **The one number worth reproducing precisely is Gemini's established/new
  split**, not the overall 74.5% — matching the *shape* (a ~27-point drop
  on new names) is the real test of whether this rebuild's era stratum
  (`dose_r/strata/strata.jsonl`) means the same thing DOSE's does.

## Statistical power: what gap would actually mean something

Computed once, against the real numbers in this repo, so a future report
reads a "2-point gap" correctly on the day it appears rather than after the
fact.

**Whole-set comparison, sampling noise only (n=274, baseline pass rate 0.7,
α=0.05, 80% power):**

- Minimum gap distinguishable from noise between two independent ~274-item
  pass rates: **~11 percentage points**.
- 95% Wilson CI half-width on a single ~274-item pass rate near 0.7:
  **~±5.4 points** — consistent with the brief's own illustration of a ±6
  CI, which is a useful sanity check on this calculation, not a coincidence
  built into it.

**Per-era subgroup comparison** (established n≈127, new n≈147, the split this
build actually produced — see `dose_r/strata/STRATA.md`): minimum detectable
gap **~15-16 points** per subgroup at the same alpha/power. Gemini's published
89.1% vs. 61.6% (a 27.5-point gap) clears that bar comfortably; a
replication that reproduced only a 10-point established/new gap would be
reproducing noise-indistinguishable-from-zero on DOSE's own terms, not a
smaller version of the real effect.

These are **unpaired, independent-sample estimates**, which is conservative:
DOSE-R's real comparisons are paired (the same 274 items scored by each
system, or the same run split by stratum), and a paired test has more power
than this credits it with. Treat every number above as an upper bound on the
gap needed for significance — the true detectable gap is probably smaller,
but by how much depends on inter-item correlation this repo cannot estimate
without real per-item scores from more than one system.

## The reference layer's uncertainty has to be propagated, not ignored

As of this writing, 165-185 of 284 gold pronunciation references (the count
has moved during this workstream as a parallel effort adds Wikipedia/
Wiktionary sources -- see `COVERAGE.md`'s git history for the current figure)
carry no external confirmation at all. Aggregated to each row's weakest
ingredient (`reference_confidence` in `strata.jsonl`), that was 170 of 274
rows at the time this document was last regenerated. A pass/fail verdict on
one of those rows depends on a rule-derived respelling nobody has checked.
Two consequences, both implemented in `power.py`, and both **re-derived from
`strata.jsonl` at report-build time, not hardcoded** -- rerun
`power.effective_sample_size` and `power.propagate_reference_uncertainty`
against the current file rather than trusting the numbers quoted here as
frozen:

1. **Effective sample size.** Excluding every low-confidence row, this
   benchmark's *externally verifiable* sample size was **109 of 274 items**
   (40%) at last count, not 274. At n=109, the minimum detectable gap widens
   from ~11 to **~17 percentage points**, and the single-pass-rate CI
   half-width widens from ~±5.4 to **~±8.5 points**. This share has been
   improving as the reference layer gains sources and should be re-measured
   before it is quoted in a final report.
2. **Uncertainty in the labels themselves.** There is no measured error rate
   for the rule-derived low-confidence references — nothing in this project
   has checked how often they are actually wrong. `propagate_reference_uncertainty`
   sweeps three illustrative assumed error rates instead of asserting one
   (figures below at n_low=165):

   | assumed gold error rate | CI half-width, sampling only | CI half-width, with reference uncertainty |
   | --- | --- | --- |
   | 10% | ±5.4 pp | ±6.1 pp |
   | 20% | ±5.4 pp | ±6.6 pp |
   | 30% | ±5.4 pp | ±6.9 pp |

   This is a sensitivity bracket, not a calibrated correction — treat the
   right-hand column as "at least this wide," not as the true interval.

**Bottom line for anyone reading a future fidelity report:** a gap under
roughly 10-15 percentage points, on the whole set or on any one stratum, is
not distinguishable from noise at this sample size, and that floor rises
further for any claim that leans on the 66% of rows with an unverified
reference. Report gaps below that floor as "not distinguishable from noise
at n=274," not as findings.

## Attribution: judge/system vs. reference layer

Two methods, because only one of them is currently runnable:

- **`attribution_by_reference_confidence`** (runnable today): splits a
  judged run's pass rate by `reference_confidence` tier and runs a
  two-proportion z-test between the high and low tiers. A significant,
  sizeable high-vs-low gap is evidence that *some* of a run's apparent
  failure rate traces to unverified gold references rather than to the TTS
  system or the judge. This is a proxy — it cannot separate "the reference is
  wrong" from "low-confidence names happen to be harder for an unrelated
  reason" — but it is a real, checkable signal, and it is exercised in
  `tests/test_fidelity.py` against both a planted confidence effect and a
  planted null.
- **`attribution_against_hand_verified`** (the method the brief actually
  specifies, not runnable yet): compares the judge's automated verdict
  against a human-confirmed verdict on a stratified sample.
  `artifacts/anchor_set.jsonl` already stages exactly such a sample — built
  by Workstream 1b, stratified by difficulty and confidence — but per
  `HANDOFF.md`, nobody has listened through it yet. The moment that happens,
  its verdicts are the `hand_verified` input this function expects.

## Testing philosophy

Every test in `tests/test_fidelity.py` runs against `synthetic.py`'s
generated strata and runs, with planted effects (a known era gap, a known
confidence-tier gap, a known null) that the harness must recover within a
stated tolerance. This is deliberate: it proves the *arithmetic* is correct
(Wilson intervals, the z-test, Spearman, the attribution split) independent
of whether any real system's numbers ever look like the synthetic ones. It is
not, and is not presented as, evidence about any real TTS system.

## Running it once real data exists

```bash
python -m dose_r.report.build_report \
    --run rxpronounce=artifacts/rxpronounce/rows.jsonl \
    --run gemini_tts=artifacts/gemini_tts/rows.jsonl \
    --out artifacts/fidelity_report.json
```

Each `rows.jsonl` is `dose_r.judge.pipeline.write_artifacts`'s row-level
output for one system's judged run. The command refuses to run with zero
`--run` arguments rather than silently producing an empty report.
