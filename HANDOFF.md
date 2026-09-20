# DOSE-R Handoff

> **Pronunciation:** Never convert dictionary respelling to IPA. Use the
> original source string. Converter is banned (`RespellToIpaBanned`).
> See `dose_r/references/README.md`.

Last verified against the repo on 2026-09-18, branch
`claude/sc-sandbox-gcp-access-arw4m8`. Where this file and the planning
documents disagree, this file was checked against the code and they were not.

## Is Workstream 2 unblocked?

**No.** Workstream 2 is gated on a Workstream 1d sign-off that does not exist
yet, and 1d in turn cannot finish without scored audio that this repo does not
contain. Three things stand between here and that gate, in order:

1. **1d has not produced its deliverables.** Strata, the fidelity report, and
   the pre-registered `SIGNOFF.md` are all outstanding. In progress.
2. **The judge is not calibrated.** `artifacts/anchor_set.jsonl` holds a
   stratified sample for a human to verify, but nobody has verified it, so the
   scorer's weighting has never been tuned against ground truth.
3. **No real TTS audio exists.** This is the hard blocker — see below.

Nothing in Workstream 2 should start before that sign-off. Racing to "beat
Gemini" on an unvalidated judge produces a number nobody can defend.

## The audio blocker

Planning docs refer to 274 Gemini WAVs in `runs/full-cheap-tier-v1/`. **They are
not in this repo.** `runs/` holds only `mock-slow/`, and `runs/` is gitignored,
so the files are not recoverable from any branch — they live in whatever
container produced them and are likely gone.

Producing them again means a live, billable Gemini TTS pass, which has not been
authorised. Until someone either locates those files or approves the spend, the
fidelity report can be built and tested but cannot be run on real data, and the
sign-off cannot be granted.

Vertex AI itself is confirmed reachable (see below), so this is a spend
decision, not a technical one.

## What is done

| Piece | State |
| --- | --- |
| Dataset staging | Complete — `data/dose_v1.jsonl` |
| 1a gold references | Complete — `dose_r/references/references.jsonl` |
| 1b judge | Built and stress-tested; **not calibrated** |
| 1c adapters + runner | Complete; mock-validated, never run live |
| 1d strata + report | In progress |
| Workstream 2 | Correctly gated, not started |

54 tests pass: `python3 -m pytest tests/ -q`

## Things the planning docs get wrong

- **GCP was never blocked.** The docs claim `googleapis.com` is 403'd with no
  credentials. What actually happens is that the system `cryptography` install
  is missing `_cffi_backend`, which takes down all of `google.auth.crypt` and
  makes every auth attempt fail like a network error. `pip install cffi` fixes
  it. Vertex AI then answers: HTTP 200 for `gemini-2.5-flash-preview-tts`, from
  service account `dev-env-sa@project-amer-scs-sandbox`, which holds
  `aiplatform.admin`. The key is read from `GOOGLE_APPLICATION_CREDENTIALS_JSON`
  in memory and must never be written to disk.
- **The unit of work is 286 ingredient spans, not 274 rows** (284 unique
  ingredients). Nine generic rows are combination products naming two or three
  ingredients each.
- **The raw `drug` column is not a pronunciation target.** Every generic row
  carries a literal `" (generic)"` suffix that never appears in the sentence,
  and no brand row has a `(brand)` counterpart. Staging strips it.

## The dominant risk: the reference layer is thin

1a covers all 284 ingredients, but only 99 rest on an external source:

| Tier | Count | Meaning |
| --- | --- | --- |
| high | 19 | two independent sources agree |
| medium | 80 | one external source answered |
| low | 185 | no external source; derived from spelling by rule |

Source reality, measured rather than assumed:

- **Merriam-Webster** is the only external source that answered, and it supplies
  real pronunciations plus accepted variants.
- **CMUdict** covers 13 of 284. Almost every DOSE name is a coined trade or INN
  name no general dictionary lists, so there is no dictionary shortcut here.
- **Drugs.com** returns HTTP 403 from this environment.
- **FDA labels** (openFDA, DailyMed) are reachable but contain no pronunciation
  respellings at all. They remain useful for approval dates.

`WORKSTREAM_1_TASKS.md` sets a success criterion of **>=60% high confidence**.
The layer is at **7% high, 35% sourced**. That is a real miss. It matters
because a wrong reference makes every system look wrong in the same direction —
the exact systematic bias the fidelity report exists to detect — so the gap
cannot be averaged away and must be stratified on in any result.

## Highest-leverage next step

**LLM arbitration over the 185 low-confidence entries.** It needs no new
infrastructure, Vertex AI is confirmed reachable, and it attacks the project's
dominant risk directly. Each such entry carries a `TODO` in its `notes` field,
so the work list is already machine-readable. Raising even half of them to
medium would change what the fidelity report is able to conclude.

After that: human-verify the anchor set and calibrate the judge; then resolve
the audio blocker; then 1d signs off or does not.

## Ground rules worth keeping

- Spawn subagents on **Sonnet**, not Opus. An earlier round ran four Opus agents
  concurrently and exhausted the session limit, losing all of their work.
- 1a is the critical path. Everything downstream, in both workstreams, consumes
  `references.jsonl`. It was built last; it should have been built first.
- Never write the service-account key to disk.
- `runs/`, `artifacts/`, and the source caches are gitignored on purpose; they
  are large and rebuildable.

## Commands

```bash
python3 -m pytest tests/ -q                      # full suite
python3 -m dose_r.references.build               # rebuild gold layer (cached)
python3 -m dose_r.judge.stress_test \
    --references dose_r/references/references.jsonl   # judge separation check
python3 scripts/stage_dataset.py                 # re-stage from Hugging Face
```
