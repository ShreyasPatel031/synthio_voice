# Kokoro-82M + scored Misaki lexicon

Stock `hexgrad/Kokoro-82M`, voice `af_heart`, speed 1.0. No finetune.
The only change is the 284-name lexicon (mean F1 0.7533 vs Cloud Standard-C)
loaded into `pipeline.g2p.lexicon.golds`.

Team guide: `docs/TEAM_HANDOFF.md`.

## Live service

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
| Env | `KOKORO_VOICE=af_heart`, `KOKORO_SPEED=1.0`, `KOKORO_SPEED_LOCK=1` |

`GET /` returns `lexicon: 284`.

Send the full sentence as plain text. The drug is looked up in the lexicon. No `[Name](/phones/)` inject.

```bash
curl -sS -X POST "https://kokoro-misaki-347838016394.us-east4.run.app/v1/audio/speech" \
  -H 'Content-Type: application/json' \
  -d '{"input":"We are starting Xolair today.","voice":"af_heart","speed":1.0}' \
  --output out.wav
```

Benchmark, plain sentences only:

```bash
python scripts/benchmark_cloudrun_misaki.py
```

## Deploy a new revision

`./services/kokoro_misaki/deploy_cloudrun.sh` builds a new image and replaces
the service above.
