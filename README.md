# Synthio voice

Stock [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), voice `af_heart`, speed `1.0`. No finetune. The only change is a 284-name Misaki lexicon (mean F1 **0.7533** vs Cloud Standard-C) loaded into `pipeline.g2p.lexicon.golds`.

Send a full sentence. The lexicon is already on the server. Do not wrap names in `[Name](/phones/)`. Do not pass a voice, speed, or lexicon.

## Speak

```bash
curl -sS -X POST "https://kokoro-misaki-347838016394.us-east4.run.app/v1/audio/speech" \
  -H 'Content-Type: application/json' \
  -d '{"input":"We are starting Xolair today."}' \
  --output out.wav
```

That returns a 24 kHz wav. One request at a time. The first call after idle loads the model and can take a few minutes.

Check it is up:

```bash
curl -sS https://kokoro-misaki-347838016394.us-east4.run.app/
```

`GET /` returns `"model": "hexgrad/Kokoro-82M"` and `"lexicon": 284`.

| | |
| --- | --- |
| URL | https://kokoro-misaki-347838016394.us-east4.run.app |
| Alt URL | https://kokoro-misaki-uevtdub7oa-uk.a.run.app |
| Project | `project-amer-scs-sandbox` |
| Region | `us-east4` |
| Revision | `kokoro-misaki-00002-nns` |
| Image | `us-central1-docker.pkg.dev/project-amer-scs-sandbox/synthio-voice/kokoro-misaki:20260925-181753` |
| GPU | 1× NVIDIA L4 |
| Auth | Public |

`voice` and `speed` can be omitted. The server locks them to `af_heart` and `1.0`.

## Score it

The benchmark posts each carrier sentence in `data/dose_v1.jsonl` as plain text, crops the drug name, and scores WavLM SpeechBERTScore F1 against the locked Cloud Standard-C gold.

```bash
python scripts/benchmark_cloudrun_misaki.py --only xolair
python scripts/benchmark_cloudrun_misaki.py
```

Wavs land in `runs/cloudrun-misaki-bench/sent/`. Scores land in `runs/cloudrun-misaki-bench/cloud-rank.json`. The published Kokoro + Misaki mean on this lexicon is **0.7533**.

Gold clips are `data/gold_gemini_ipa/wavs/<slug>.wav`. They are not in git. The script only reads them. Do not regenerate them, and do not convert a dictionary respelling into IPA.
