---
license: cc-by-4.0
task_categories:
  - text-to-speech
language:
  - en
tags:
  - text-to-speech
  - tts
  - pronunciation
  - healthcare
  - pharma
  - drug-names
  - benchmark
pretty_name: DOSE v1.0 (Drug-name Oral Synthesis Evaluation)
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*.parquet
---

# DOSE v1.0: Drug-name Oral Synthesis Evaluation

**Can your voice model say retatrutide?**

DOSE is a benchmark measuring how accurately text-to-speech (TTS) systems
pronounce pharmaceutical drug names. It covers established brands,
newly-approved names, and generic/INN names, each tested inside a real
clinical sentence rather than spoken in isolation.

**Version: DOSE v1.0.** Live leaderboard and five public audio samples:
https://synthiolabs.com/synthio-dose-benchmark

This public dataset is the **held-out test set only**: drug name, name type,
and clinical sentence. Reference pronunciations, evaluation audio, and
per-system scores are not released, so the set cannot be used as an
answer key. Email [rohit@synthiolabs.com](mailto:rohit@synthiolabs.com?subject=DOSE%20model%20submission)
to submit a model for scoring.

## Why this matters

Mispronounced drug names in voice AI (IVR systems, clinical voice agents,
pharmacy assistants) are a patient-safety concern analogous to LASA
(Look-Alike/Sound-Alike) drug name confusion — a recognized WHO/ISMP/FDA
error category. As voice AI is deployed further into clinical workflows,
TTS mispronunciation of a drug name is a new, largely unmeasured extension
of that same risk category.

## Quickstart

```python
from datasets import load_dataset

ds = load_dataset("SynthioLabs/dose-benchmark", split="train")

row = ds[0]
print(row["drug"], row["name_type"], row["sentence"])
```

Filter by name type:

```python
generic_names = ds.filter(lambda r: r["name_type"] == "generic")
brand_names = ds.filter(lambda r: r["name_type"] == "brand")
```

## Dataset

- **274 drug names**: 128 traditional (established) + 146 new (recently
  approved) names. One row per name.
- **Brand and generic/INN names** (`name_type` column).
- **9 TTS systems** were evaluated on these sentences (see leaderboard).
  Those recordings and scores are not in this download.
- Every system read the **exact same clinical sentence** containing the
  drug name, with no special pronunciation hints given to any system.

## How DOSE is scored

**Reference pronunciations.** Each drug’s reference comes from public
pharmaceutical sources, including labels, manufacturer materials, and
established medical references. It is then checked against spoken clinical
usage so the target reflects the accepted pronunciation — not a model’s
guess. The reference strings themselves are held out.

**Judge.** Each system’s recording is scored 0–5 against that same
reference by automated audio comparison. Several independent judgments are
combined into one result; **4 or higher counts as a pass**. Human
reviewers are not in the scoring loop.

## Columns

| Column | Type | Description |
|---|---|---|
| `drug` | string | Drug name (brand or generic/INN) |
| `name_type` | class label | `brand` or `generic` |
| `sentence` | string | The clinical sentence to be read aloud |

## Leaderboard (overall pass rate, score >= 4)

| Rank | System | Model | Pass rate |
|------|--------|-------|-----------|
| 1 | Synthio | Synthio RxPronounce | 91.2% |
| 2 | Cartesia | sonic-3.6 (beta) | 80.3% |
| 3 | ElevenLabs | eleven_v3 | 79.2% |
| 4 | OpenAI TTS | gpt-4o-mini-tts | 77.4% |
| 5 | xAI | Grok TTS | 77.0% |
| 6 | Google (Gemini TTS) | gemini-3.1-flash-tts-preview | 74.5% |
| 7 | Deepgram | aura-2-thalia-en | 72.6% |
| 8 | Cartesia | sonic-3.5 | 69.0% |
| 9 | Microsoft Azure | DragonHD Neural | 63.1% |

## Submit your model

Email us at [rohit@synthiolabs.com](mailto:rohit@synthiolabs.com?subject=DOSE%20model%20submission)
to submit a model for evaluation.

## Citation

```bibtex
@misc{synthio2026dosev1,
  title   = {DOSE v1.0: Drug-name Oral Synthesis Evaluation},
  author  = {Pari, Rohit Gangadhar and {Synthio Labs}},
  year    = {2026},
  url     = {https://synthiolabs.com/synthio-dose-benchmark}
}
```
