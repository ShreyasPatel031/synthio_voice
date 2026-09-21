# Official DoSE benchmark

This is the pronunciation benchmark. Use it. Do not invent another gold set.

## What you score

A candidate says a **DoSE carrier sentence** (`data/dose_v1.jsonl`). You do not score the whole sentence.

1. Forced-align the sentence (`dose_r.forced_align.extract_drug_span_forced_align`).
2. Cut out the drug name.
3. Compare that cut to the **isolated** Gemini teacher for that name with SpeechBERTScore F1 (WavLM).

```bash
python scripts/score_dose_ctc_vs_gemini_ipa.py \
  --wav-dir runs/oss-eval/kokoro-full-plain \
  --condition kokoro-full-plain \
  --scope full
```

`--scope full` is every DoSE ingredient span (the benchmark). `--scope hard` is the 54-name subset only.

## Gold — do not touch the IPA

The only pronunciation source is `data/gold_gemini_ipa/`.

- 284 names
- `manifest.jsonl` and `ipa.json` — the IPA strings that were kept
- `wavs/{slug}.wav` — Gemini 3.1 Flash TTS, voice Kore, of that IPA

Do not edit the IPA. Do not convert a respelling into IPA. Do not fill `ipa` from Wikipedia, DailyMed, USAN, or a G2P tool. Do not swap in Cloud Standard-C, `runs/gemini31-ipa-vs-cloud`, or any other teacher. The scorer reads `data/gold_gemini_ipa` only.

## Listen

`benchmark/examples/kokoro-plain/` — a few Kokoro plain rows: full sentence, CTC cut, and the official gold wav. Open `index.html`.
