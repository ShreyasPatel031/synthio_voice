# Kokoro + Misaki handoff

## Run the benchmark

One endpoint. Plain sentences. Kokoro + Misaki is already loaded. No voice, speed, or phone-inject config.

```bash
curl -sS https://kokoro-misaki-347838016394.us-east4.run.app/
python scripts/benchmark_cloudrun_misaki.py
```

`GET /` must show `"lexicon": 284`. The script reads each carrier sentence from `data/dose_v1.jsonl` and posts `{"input": "<full sentence>"}` to `POST /v1/audio/speech`. It crops the drug span and scores WavLM F1 against `data/gold_gemini_ipa/wavs`. Results: `runs/cloudrun-misaki-bench/cloud-rank.json`. Dashboard column to compare: `items[].scores["kokoro-misaki"]`, mean 0.7533.

Do not send `[Name](/phones/)`.

## What this is

Two artifacts:

1. The comparison dashboard, for reading the benchmark and pulling per-name data.
2. The public Cloud Run service, for synthesizing speech from this lexicon.

The number on both is WavLM-large SpeechBERTScore F1. The hypothesis clip is the forced-align crop of the drug name inside the carrier sentence. The reference is the locked Cloud Standard-C gold for that name. Speed is 1.0. This is not the public DOSE pass@4 leaderboard. `docs/CANDIDATE_MODEL_EVAL_RUNBOOK.md` scores a different question (best match against a human clip).

Do not convert a dictionary respelling into IPA. The Misaki strings in the lexicon are the phones that were scored. See `dose_r/references/README.md`.

## Dashboard

```bash
python dashboard/server.py
```

Open http://127.0.0.1:8771/dashboard/index.html

`dashboard/server.py` also serves the layer-score API. A plain static file server will load the board and the audio, and the "Why F1 moves" panel will 404.

The board file is `dashboard/board.json`. Kokoro + Misaki is the default system: **284 names, mean 0.7533**.

| System id | What it is | Scores | Clips mounted |
| --- | --- | ---: | ---: |
| `kokoro-misaki` | Kokoro-82M + Misaki lexicon, voice `af_heart` | 284 | 284 sentence crops |
| `kokoro-plain` | Same Kokoro, no lexicon | 250 | 284 |
| `qwen-wav` | Qwen + IPA mp3 | 284 | 94 |
| `qwen-ft` | Qwen fine-tune | 250 | 284 |
| `qwen-base` | Qwen, no injection | 284 | 284 |

Slice charts (era, brand/generic, difficulty) use the 265 names that join to `dose_r/strata/strata.jsonl`. The other 19 stay in the overall mean. Kokoro + Misaki slices on the current board: established 0.752 (n=124), new 0.755 (n=141), brand 0.746 (n=143), generic 0.764 (n=122).

Each `items[]` row:

| Field | Use |
| --- | --- |
| `slug`, `drug` | Join key and display name |
| `sentence` | Carrier sentence to synthesize |
| `f1` | Kokoro + Misaki vs Cloud Standard-C |
| `scores` | Same metric for every system that has a score |
| `human_f1`, `human_source` | F1 against a Drugs.com clip when one exists (176 names). Secondary. The official column is `f1`. |
| `era`, `name_type`, `difficulty` | From `dose_r/strata/strata.jsonl`. Null on 19 combo-only rows. |
| `audio.gold` | Cloud Standard-C reference clip |
| `audio.model` | Kokoro + Misaki sentence crop |
| `audio.human` | Human clip (183 names) |
| `audio.by_system` | Clip path per system id |

Layer scores, on demand, for one name and one system:

```text
GET /dashboard/api/layers?slug=xolair&system=kokoro-misaki
```

The response is F1 at each WavLM layer against the gold clip, plus the same pass against the human clip when that file exists. The headline F1 on the board is the final layer only.

## Cloud Run

Checked against the live service.

| Field | Value |
| --- | --- |
| Project | `project-amer-scs-sandbox` |
| Region | `us-east4` |
| Service | `kokoro-misaki` |
| URL | https://kokoro-misaki-347838016394.us-east4.run.app |
| Alt URL | https://kokoro-misaki-uevtdub7oa-uk.a.run.app |
| Revision | `kokoro-misaki-00002-nns` (100% of traffic) |
| Image | `us-central1-docker.pkg.dev/project-amer-scs-sandbox/synthio-voice/kokoro-misaki:20260925-181753` |
| Auth | Public |
| GPU | 1× NVIDIA L4, zonal redundancy off |
| CPU / RAM | 4 vCPU, 16 GiB |
| Concurrency | 1 |
| Timeout | 300s |
| Max instances | 2 |
| Port | 8080 |
| Env | `KOKORO_VOICE=af_heart`, `KOKORO_SPEED=1.0`, `KOKORO_SPEED_LOCK=1` |

Liveness:

```bash
curl -sS https://kokoro-misaki-347838016394.us-east4.run.app/
```

`GET /` returns `{"service":"kokoro-misaki","model":"hexgrad/Kokoro-82M","lexicon":284,...}`.

### Synthesize

Send the full carrier sentence as plain text. The lexicon is already loaded. Misaki looks the drug up. Do not wrap the name in `[Name](/phones/)`. Do not pass voice, speed, or a lexicon file. One endpoint, one field.

```bash
curl -sS -X POST "https://kokoro-misaki-347838016394.us-east4.run.app/v1/audio/speech" \
  -H 'Content-Type: application/json' \
  -d '{"input":"We are starting Xolair today.","voice":"af_heart","speed":1.0}' \
  --output out.wav
```

`voice` and `speed` can be omitted. The server locks both: `af_heart`, `1.0`. Response is a 24 kHz wav. One request at a time. The first request after idle loads the model onto the L4.

### Score the endpoint

Same sentences as the dashboard, sent as plain text:

```bash
python scripts/benchmark_cloudrun_misaki.py --only xolair
python scripts/benchmark_cloudrun_misaki.py
```

The script posts `{"input": "<sentence>"}` and nothing else. The full run is sequential because concurrency is 1. Wavs land in `runs/cloudrun-misaki-bench/sent/`. Scores land in `runs/cloudrun-misaki-bench/cloud-rank.json`. Compare `rows[].ctc_f1` to `dashboard/board.json` `items[].scores["kokoro-misaki"]`. The board mean for that column is 0.7533.

Gold wavs are `data/gold_gemini_ipa/wavs/<slug>.wav`. Leave that directory as it is. The script only reads it.

## Data for research and training

| Path | What it is |
| --- | --- |
| `dashboard/board.json` | Per-name scores, strata, sentences, and audio paths for every system on the board |
| `data/dose_v1.jsonl` | Carrier sentences and character spans |
| `data/kokoro_misaki_lexicon.json` | 284 scored Misaki strings. `word`, `misaki`, `f1`. Mean 0.7533 |
| `dose_r/strata/strata.jsonl` | `era`, `name_type`, `difficulty` |
| `runs/misaki-iter/lexicon-rank.json` | Official Kokoro + Misaki crop scores behind the board |
| `runs/misaki-iter/lexicon-vs-human.json` | The `human_f1` column |
| `runs/misaki-iter/lexicon-span/<slug>.wav` | Kokoro + Misaki drug crops |
| `data/gold_gemini_ipa/wavs/` | Locked Cloud Standard-C gold. Read, do not regenerate |
| `runs/listen-hack-vs-gold/mp3/` | Listen clips: `gold/`, `human/`, `kokoro-plain/`, `qwen-wav/`, `qwen-ft/` |
| `runs/main-bench-cloud/qwen-plain/span/` | Qwen base crops |
| `runs/main-bench-cloud/qwen-ft/span/` | Qwen fine-tune crops that are not in the mp3 folder |
| `data/reference_audio/manifest.jsonl` | Human-clip provenance |

Rebuild the board after a new score file with `python scripts/build_dashboard_board.py`. The inputs it expects are listed at the top of that script. Qwen + IPA mp3 scores can be present for all 284 names while only 94 listen clips are on disk. Train and evaluate from the score field, and check `audio.by_system` before assuming a wav exists.

A training target for "say this name the way the benchmark rewards" is the gold wav plus the lexicon `misaki` string, inside the `sentence` from the board. The gold IPA was published by the teacher. Do not fill it in by converting a respelling.
