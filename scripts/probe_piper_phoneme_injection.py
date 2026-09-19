#!/usr/bin/env python
"""Real, verified phoneme-injection synthesis via Piper TTS (VITS-based,
local, CPU, free), and score against human reference audio.

Why this is different from the earlier Cloud TTS / Gemini probes
-------------------------------------------------------------------
Cloud TTS's SSML `<phoneme alphabet="ipa">` tag was found to be silently
ignored for 3 of 13 test words (byte-identical to plain-text synthesis, no
error). espeak-ng's `[[...]]` phoneme markup is documented to expect its OWN
internal ASCII phoneme mnemonics, not standard IPA -- feeding it raw dictionary
IPA is undocumented, unverified behavior.

Piper exposes its synthesis pipeline as separable steps:
`phonemize(text) -> list[list[str]]` (one phoneme per list entry) ->
`phonemes_to_ids(phonemes) -> list[int]` -> `phoneme_ids_to_audio(ids)`.
Its phoneme vocabulary is confirmed `espeak`-type (`config["phoneme_type"]`)
and every character in our dictionary IPA strings is present in its
`phoneme_id_map` -- checked directly, not assumed. This lets us BYPASS
Piper's own internal espeak G2P and feed the dictionary's IPA directly,
character-by-character, with a verified (not hoped-for) vocabulary match.

Mechanism
---------
1. `ipa_to_piper_phonemes(ipa)`: split the IPA string into individual
   Unicode characters (matching Piper's own phonemize() granularity, which
   also treats stress marks ˈˌ as individual "phoneme" tokens in the
   sequence -- confirmed by inspecting its own output on plain text).
2. `phonemes_to_ids` -> `phoneme_ids_to_audio` -- direct model inference,
   no G2P involved for the injected variant.
3. Score against the human reference clip with the same F1 pipeline as
   every other synthesis method this session.
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from piper.voice import PiperVoice  # noqa: E402

from dose_r import dataset  # noqa: E402
from dose_r.forced_align import extract_drug_span_forced_align  # noqa: E402
from dose_r.references.reference_clips import available_clips  # noqa: E402
from dose_r.scoring.speech_similarity import extract_frame_embeddings, speech_bertscore  # noqa: E402

VOICE_PATH = "/tmp/piper_voices/en/en_US/lessac/medium/en_US-lessac-medium.onnx"

# Same 13 items + IPA as scripts/probe_synthetic_reference.py's TARGETS/IPA.
TARGETS = ["vyloy", "advair", "eliquis", "esomeprazole", "aripiprazole",
           "acoramidis", "talquetamab", "xolair", "winrevair", "imaavy",
           "cariprazine", "sitagliptin", "varenicline"]
IPA = {
    "vyloy": "ˈvaɪlɔɪ", "advair": "ˈædˌvɛr", "eliquis": "ˈɛləkwəs",
    "esomeprazole": "ɛsoʊˈmɛpræzoʊl", "aripiprazole": "ɑːrɪˈpɪpræzoʊl",
    "acoramidis": "ækoʊˈræmɪdɪs", "talquetamab": "tælˈkwɛtæmæb",
    "xolair": "ˈzoʊˌlɛr", "winrevair": "ˈwɪnrɛvɛər", "imaavy": "ɪmˈɑːviː",
    "cariprazine": "kɑːrˈɪpræziːn", "sitagliptin": "sɪtæˈɡlɪptɪn",
    "varenicline": "vɑːrˈɛnɪkliːn",
}


def ipa_to_piper_phonemes(ipa: str, phoneme_id_map: dict) -> list[str]:
    """Split into individual characters, matching Piper's own phonemize()
    granularity. Any character genuinely outside the model's vocabulary is
    dropped rather than crashing -- checked empirically to not happen for
    our 13 test items, but a real gap for the full 280-item reference set.
    """
    return [c for c in ipa if c in phoneme_id_map]


def synthesize_from_ipa(voice: PiperVoice, ipa: str) -> np.ndarray:
    phonemes = ipa_to_piper_phonemes(ipa, voice.config.phoneme_id_map)
    ids = voice.phonemes_to_ids(phonemes)
    audio = voice.phoneme_ids_to_audio(ids)
    if isinstance(audio, tuple):
        audio = audio[0]
    return audio


def synthesize_from_text(voice: PiperVoice, text: str) -> np.ndarray:
    chunks = list(voice.synthesize(text))
    return np.concatenate([c.audio_float_array for c in chunks])


def to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    import io
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()


def main() -> int:
    voice = PiperVoice.load(VOICE_PATH)
    sr = voice.config.sample_rate
    clips = available_clips()
    items_by_id = {i.item_id: i for i in dataset.load_items()}

    RUN_DIR = REPO_ROOT / "runs" / "gemini-flash-tts-v1"
    recs = {json.loads(l)["item_id"]: json.loads(l)
            for l in (RUN_DIR / "results.jsonl").read_text().splitlines() if l.strip()}

    OUT_DIR = REPO_ROOT.parent / "scratch_audio"
    OUT_DIR.mkdir(exist_ok=True)

    results = {}
    for name in TARGETS:
        item = items_by_id[name]
        clip = clips.get(item.drug)
        if clip is None:
            continue
        ipa = IPA[name]

        piper_ipa_audio = synthesize_from_ipa(voice, ipa)
        piper_ipa_wav = to_wav_bytes(piper_ipa_audio, sr)

        piper_text_audio = synthesize_from_text(voice, item.drug)
        piper_text_wav = to_wav_bytes(piper_text_audio, sr)

        gemini_full = (RUN_DIR / recs[name]["audio_path"]).read_bytes()
        gemini_span = extract_drug_span_forced_align(gemini_full, item.sentence, item.drug)

        feats_ref = extract_frame_embeddings(clip.path)
        row = {}
        for tag, wav_bytes in [("piper-ipa-injected", piper_ipa_wav),
                                ("piper-text-default", piper_text_wav),
                                ("gemini-baseline", gemini_span)]:
            feats = extract_frame_embeddings(wav_bytes)
            f1 = speech_bertscore(feats, feats_ref)["f1"]
            row[tag] = f1
            with wave.open(__import__("io").BytesIO(wav_bytes), "rb") as w:
                dur = w.getnframes() / w.getframerate()
            print(f"{name:16s} {tag:20s} f1={f1:.3f}  dur={dur:.2f}s")

        (OUT_DIR / f"{name}_piper_ipa.wav").write_bytes(piper_ipa_wav)
        (OUT_DIR / f"{name}_piper_text.wav").write_bytes(piper_text_wav)
        results[name] = row
        print()

    print("=" * 80)
    print("SUMMARY")
    methods = ["gemini-baseline", "piper-text-default", "piper-ipa-injected"]
    for m in methods:
        vals = [r[m] for r in results.values() if m in r]
        print(f"{m:20s} mean={sum(vals)/len(vals):.3f}  n={len(vals)}")

    (REPO_ROOT / "runs" / "piper-phoneme-probe-v1.json").write_text(json.dumps(results, indent=2))
    print(f"\naudio written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
