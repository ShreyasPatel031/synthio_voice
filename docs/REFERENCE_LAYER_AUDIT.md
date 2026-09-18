# Audit: the gold pronunciation layer is not reliable yet

Workstream 2 audit of `dose_r/references/references.jsonl`
(branch `claude/sc-sandbox-gcp-access-arw4m8`, 284 ingredients), run
2026-09-18. Every number below was measured from the file, and every
claim about an external source was tested against that source directly.

## Short answer

**No, we do not have reliable pronunciations for all 284 names.** We have
them for 19.

| Confidence | Count | Share | What it actually means |
| --- | ---: | ---: | --- |
| high | 19 | 7% | Two independent sources agree. Trustworthy. |
| medium | 80 | 28% | Exactly one external source answered. Usable, unverified. |
| low | 185 | 65% | **No external source. Guessed from spelling by rule.** |

`WORKSTREAM_1_TASKS.md` sets a success criterion of >=60% high
confidence. The layer is at 7%.

## The 185 are not merely unsourced -- they are wrong

"Derived from spelling by rule" reads like a reasonable fallback. It is
not, and this is checkable rather than a matter of opinion. The clearest
case is the `-tide` peptide stem:

| Ingredient | Layer says | Reads as | Correct |
| --- | --- | --- | --- |
| exenatide | `ˌiksˈineɪtɪd` | eeks-EE-nay-**tid** | ek-SEN-uh-**tide** |
| Retatrutide | `ɹˌitˈætɹutɪd` | ree-TAT-roo-**tid** | re-ta-TROO-**tide** |
| navepegritide | `nˌeɪvipˈɛɡɹaɪtɪd` | nay-vee-PEG-rye-**tid** | -- ends **-tide** |

`exenatide` settles it. It is a long-marketed drug (Byetta) with an
unambiguous pronunciation, and the rule gets it wrong three ways: the
initial vowel, the stress placement, and the stem. The `-tide` stem is
/taɪd/ in every peptide INN; the rule renders it /tɪd/ **every time**.

A reference that is wrong in a consistent direction is worse than a
missing one. A missing reference makes an item unscoreable and visible. A
systematically wrong one makes every TTS system look wrong in the same
direction, which is exactly the systematic bias the 1d fidelity report
exists to detect -- and it cannot detect it if the bias is in the
yardstick.

## What is and is not the cause

**Tested: the source puller is not buggy.** The obvious hypothesis was
that Merriam-Webster has these words and the puller missed them. It does
not. Fetched directly:

- `apixaban` -> HTTP 200 (MW has it)
- `exenatide`, `atogepant`, `acoramidis`, `resmetirom` -> HTTP 404

So the coverage gap is real, not an extraction failure. Re-running the
pullers will not close it. Workstream 1's read here was correct.

**Not done: the USAN/INN stem rules the plan specified.** The original
design called for "USAN/INN stem pronunciation rules -- stems like
*-tinib*, *-mab*, *-gliflozin* have documented stress patterns" as the
first candidate source for generics. What exists instead is a generic
English spelling-to-sound rule with no stem awareness, which is why
`-tide` collapses to /tɪd/.

**40 of the 185 (22%) carry a stem with a documented, regular
pronunciation** and are fixable deterministically, with no LLM and no
new network source:

| Stem | Class | Count |
| --- | --- | ---: |
| -mab | monoclonal antibody | 18 |
| -tinib | kinase inhibitor | 7 |
| -tide | peptide | 4 |
| -gepant | CGRP antagonist | 3 |
| -vir | antiviral | 3 |
| -cept | receptor fusion protein | 2 |
| -gliflozin | SGLT2 inhibitor | 2 |
| -ciclib | CDK inhibitor | 1 |

## Why this is the project's dominant risk

DOSE's headline finding is that every system drops 15-30 points on
**newly-approved** names. Those are precisely the coined names no
dictionary lists -- precisely the 185. The reference layer is weakest
exactly where the benchmark's most important signal lives, so the error
is not spread evenly across strata; it is concentrated on the stratum
that carries the result.

Any pass rate computed against this layer today is bounded by the
yardstick, not by the TTS systems.

## What "LLM arbitration" means, plainly

For a name no public source lists, an LLM panel is asked to propose a
pronunciation from the name's morphology and stem, several judges are
polled, and the result is recorded with its provenance. It is an
*informed inference*, not a citation, so an arbitrated entry should be
promoted to `medium`, never to `high`. `high` should continue to require
two independent external sources.

## Recommended order

1. **Implement the USAN/INN stem rules.** Deterministic, auditable, no
   spend, fixes 40 entries and removes a systematic class of error. This
   is the plan's own step 1 and it was skipped.
2. **LLM arbitration over the remaining ~145**, promoting to `medium`
   with provenance recorded.
3. **Human-verify the anchor set** (40-60 items, stratified). Nothing is
   calibrated until this exists, and it is the only true ground truth
   available while DOSE's real gold stays private.
4. **Stratify every reported number by reference confidence.** Until the
   layer improves, a pass rate over low-confidence items measures our
   guesses, not the model. Report high/medium and low separately, always.

## Note on sequencing

Workstream 2 is not waiting on any of this. The 1,088 synthesized WAVs
that Workstream 1's handoff lists as the hard blocker already exist at
`runs/standin-v1/audio/` on branch `claude/eloquent-mayer-0i2a52` (4
systems x 274 items, already paid for). Scoring can proceed against the
current layer as long as results are stratified by confidence and read as
directional rather than DOSE-comparable.
