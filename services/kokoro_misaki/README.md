# Kokoro Misaki TTS (OpenAI-compatible)

Public Cloud Run service for Synthio DoSE benchmarking.

## API

```bash
curl -sS -X POST "$URL/v1/audio/speech" \
  -H 'Content-Type: application/json' \
  -d '{"input":"We are starting Xolair today.","voice":"af_heart","speed":1.0}' \
  --output out.wav
```

- Misaki injects work: `[Xolair](/zˈOlɛəɹ/)`
- Plain drug names in `pins.json` are auto-injected
- Speed is locked at 1.0 unless `KOKORO_SPEED_LOCK=0`

## Deploy

```bash
./services/kokoro_misaki/deploy_cloudrun.sh
```
