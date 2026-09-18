# Finding: the requested voice is not reliably the voice you get

Measured 2026-09-18 against Google Cloud Text-to-Speech from
`project-amer-scs-sandbox`, while auditing why three architecturally
different voice tiers produced near-identical pronunciation scores.

## Observation

Synthesizing the **same sentence** through the **same adapter** with the
**same requested voice**, four times in a row:

```
en-US-Standard-C  ->  8752781f, a03e8e43, a03e8e43, 8752781f   (2 distinct)
en-US-Wavenet-C   ->  8752781f, 8752781f, 8752781f, 8752781f   (1 distinct)
```

Two things are wrong here:

1. **`Standard-C` is not deterministic.** Identical requests return two
   different audio payloads.
2. **`Standard-C` sometimes returns the exact bytes `Wavenet-C` returns.**
   `8752781f` is the same MD5 in both rows.

Across the full 274-item set, byte-identical audio between tiers:

| Pair | Identical | Share |
| --- | ---: | ---: |
| Standard-C vs Wavenet-C | 101/274 | 37% |
| Wavenet-C vs Neural2-C | 109/274 | 40% |
| Standard-C vs Neural2-C | 52/274 | 19% |
| all three | 12/274 | 4% |
| Chirp3-HD vs Standard-C | 0/266 | 0% |

## This is not a bug in our adapter

Ruled out directly. The adapter sends distinct, correct voice names
(`en-US-Standard-C`, `en-US-Wavenet-C`, `en-US-Neural2-C`) on every call;
this was verified by printing the outbound `voice.name` alongside each
response hash. A client-side bug would collapse **all** requests to one
voice, giving ~100% identical output. What we see is partial (19-40%),
varies per item, varies **between repeat calls of the same request**, and
disappears entirely for Chirp3-HD (0/266).

A concatenative Standard voice and a neural WaveNet voice cannot produce
byte-identical PCM by coincidence. The behaviour is server-side.

**Hypothesis (not confirmed):** a response cache keyed on input text that
does not fully key on the requested voice, so a repeated sentence can be
served audio synthesized for a different tier. We cannot inspect Google's
internals to confirm the mechanism; the observation above stands
regardless of what causes it.

## Why it matters

**For this benchmark.** System identity is not stable, so a score cannot
be reliably attributed to a named voice tier. The finding that 91% of
items scored identically across Standard/WaveNet/Neural2 is therefore not
a weakness of the ASR judge -- it is contamination in the audio those
scores were computed from. Any Standard-vs-WaveNet-vs-Neural2 comparison
in `runs/asr-roundtrip-v1-full/` should be treated as measuring one
partially-shared backend, not three systems. The Chirp3-HD/legacy split
is the only system-level contrast in that run that survives.

**For DOSE itself, and this is the larger point.** DOSE ranks nine hosted
TTS APIs on a public leaderboard. If hosted TTS endpoints can serve
cached or cross-tier audio, then a benchmark of a hosted API is measuring
an endpoint's behaviour on a given day, not a fixed model -- and a
leaderboard gap of a few points between two systems may not be stable
under re-run. Nothing here shows that DOSE's specific nine endpoints do
this; what it shows is that the assumption of a stable target is
testable, usually untested, and false for at least one major provider.

## What we should change

1. **Fingerprint every generated clip** (hash + duration) in the run log,
   and report per-system distinct-audio counts alongside pass rates. A
   system whose audio collides with another's is a data-quality flag, not
   a result.
2. **Re-run a sample N times per item** and report within-system variance.
   A benchmark that reports one number per system per item is assuming a
   determinism that does not hold here.
3. **Prefer architecturally distinct systems** for headline comparisons
   until identity is verified; do not report fine-grained rankings among
   tiers that share a backend.
4. **Raise it in the 2a metric critique.** "Is the system under test
   stable across identical requests?" belongs in the evaluation blueprint
   as a precondition, alongside accuracy, latency and cost. DOSE does not
   appear to test it.
