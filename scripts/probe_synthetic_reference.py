#!/usr/bin/env python
"""Quick, cheap probe: can we synthesize a stand-in reference clip for drug
names that have NO human recording, using an IPA pronunciation as the
target? Tests several synthesis strategies against a SMALL set of items that
DO have a human reference (so we have ground truth to check against),
picking specifically the items already flagged as worst/most-informative
from the real Gemini Flash TTS run -- not a full corpus validation, a fast
iteration to see which strategy is even worth validating properly.

Strategies tried
----------------
1. gemini-baseline   -- the drug's span already extracted from the real
                         gemini-flash-tts-v1 run (Gemini reading the SPELLED
                         word in its natural sentence context). This is the
                         thing we're trying to find something BETTER than
                         for the no-audio items, so it's the control.
2. cloudtts-plain     -- Cloud TTS Standard voice reading the bare spelled
                         word, no phonetic hint at all. Control for whether
                         phoneme injection helps at all vs. just the letters.
3. cloudtts-ssml-ipa  -- Cloud TTS Standard voice with SSML
                         <phoneme alphabet="ipa" ph="...">word</phoneme>,
                         using the first (primary) IPA variant. This is the
                         approach the project owner is skeptical of
                         ("historically pretty bad") -- tested here rather
                         than assumed.
4. gemini-ipa-text    -- Gemini TTS given the bare IPA string (in slashes)
                         AS the input text, no spelling at all. Untested
                         territory: does the underlying LLM in Gemini's TTS
                         path recognize IPA notation and read it phonetically?
5. gemini-ipa-hint    -- Gemini TTS given BOTH the spelled word and its IPA
                         together in the input text, e.g. "Vyloy (IPA:
                         ˈvaɪlɔɪ)" -- testing whether the model uses the IPA
                         as a pronunciation hint rather than reading it aloud
                         as if it were more English text.

Each synthesized clip is scored against the item's REAL human reference clip
using the exact same SpeechBERTScore F1 pipeline as Path 2
(scoring/speech_similarity.py) -- the same methodology already trusted for
the human-vs-human ceiling. A strategy that reaches near that ceiling is a
credible stand-in for items with no human recording; one that doesn't, isn't.
"""

from __future__ import annotations

import base64
import json
import sys
import wave
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import requests  # noqa: E402

from dose_r import auth, dataset  # noqa: E402
from dose_r.forced_align import extract_drug_span_forced_align  # noqa: E402
from dose_r.references.reference_clips import available_clips  # noqa: E402
from dose_r.scoring.speech_similarity import extract_frame_embeddings, speech_bertscore  # noqa: E402

_TTS_ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"
_GEMINI_URL = (
    "https://us-central1-aiplatform.googleapis.com/v1/projects/{project}"
    "/locations/us-central1/publishers/google/models/gemini-2.5-flash-preview-tts"
    ":generateContent"
)
_CLOUD_VOICE = "en-US-Standard-C"
_SAMPLE_RATE = 24_000

TARGETS = ["vyloy", "advair", "eliquis", "esomeprazole", "aripiprazole",
           "acoramidis", "talquetamab",
           "xolair", "winrevair", "imaavy",         # brand -- arbitrary, coined
           "cariprazine", "sitagliptin", "varenicline"]  # generic -- USAN-regular

# Pulled from references.jsonl (currently only on another branch -- read
# directly here rather than depending on a cross-branch import).
IPA = {
    "vyloy": "ˈvaɪlɔɪ",
    "advair": "ˈædˌvɛr",
    "eliquis": "ˈɛləkwəs",
    "esomeprazole": "ɛsoʊˈmɛpræzoʊl",
    "aripiprazole": "ɑːrɪˈpɪpræzoʊl",
    "acoramidis": "ækoʊˈræmɪdɪs",
    "talquetamab": "tælˈkwɛtæmæb",
    "xolair": "ˈzoʊˌlɛr",
    "winrevair": "ˈwɪnrɛvɛər",
    "imaavy": "ɪmˈɑːviː",
    "cariprazine": "kɑːrˈɪpræziːn",
    "sitagliptin": "sɪtæˈɡlɪptɪn",
    "varenicline": "vɑːrˈɛnɪkliːn",
}

# Tagged so the summary can test: does "plain spelling works" track with
# generic (USAN-regular affixes, e.g. -azole/-mab/-nib) vs. brand (arbitrary,
# marketing-coined, no regularity a G2P could ever learn)?
NAME_TYPE = {
    "vyloy": "brand", "advair": "brand", "eliquis": "brand",
    "esomeprazole": "generic", "aripiprazole": "generic",
    "acoramidis": "generic", "talquetamab": "generic",
    "xolair": "brand", "winrevair": "brand", "imaavy": "brand",
    "cariprazine": "generic", "sitagliptin": "generic", "varenicline": "generic",
}


def _wav_header(pcm_len: int, sample_rate: int) -> bytes:
    import struct
    byte_rate = sample_rate * 2
    return (
        b"RIFF" + struct.pack("<I", 36 + pcm_len) + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, byte_rate, 2, 16)
        + b"data" + struct.pack("<I", pcm_len)
    )


def cloudtts_synthesize(*, text: str | None = None, ssml: str | None = None,
                        session: requests.Session) -> bytes:
    body: dict = {
        "input": {"ssml": ssml} if ssml else {"text": text},
        "voice": {"languageCode": "en-US", "name": _CLOUD_VOICE},
        "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": _SAMPLE_RATE},
    }
    resp = session.post(_TTS_ENDPOINT, headers=auth.auth_headers(), json=body, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Cloud TTS HTTP {resp.status_code}: {resp.text[:300]}")
    return base64.b64decode(resp.json()["audioContent"])


def gemini_synthesize(text: str, *, project: str, session: requests.Session) -> bytes:
    body = {
        "contents": [{"role": "user", "parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}},
        },
    }
    url = _GEMINI_URL.format(project=project)
    resp = session.post(url, headers=auth.auth_headers(), json=body, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini TTS HTTP {resp.status_code}: {resp.text[:300]}")
    parts = resp.json()["candidates"][0]["content"]["parts"]
    inline = next((p["inlineData"] for p in parts if "inlineData" in p), None)
    if inline is None:
        raise RuntimeError(f"no audio part: {str(parts)[:300]}")
    pcm = base64.b64decode(inline["data"])
    return _wav_header(len(pcm), _SAMPLE_RATE) + pcm


def score_against_reference(clip_bytes: bytes, ref_path: Path) -> float:
    feats_a = extract_frame_embeddings(clip_bytes)
    feats_b = extract_frame_embeddings(ref_path)
    result = speech_bertscore(feats_a, feats_b)
    return result["f1"]


def clip_duration_s(clip_bytes: bytes) -> float:
    with wave.open(BytesIO(clip_bytes), "rb") as w:
        return w.getnframes() / w.getframerate()


def main() -> int:
    from dose_r.config import GCP_PROJECT

    session = requests.Session()
    clips = available_clips()

    RUN_DIR = REPO_ROOT / "runs" / "gemini-flash-tts-v1"
    recs = {json.loads(l)["item_id"]: json.loads(l)
            for l in (RUN_DIR / "results.jsonl").read_text().splitlines() if l.strip()}
    items_by_id = {i.item_id: i for i in dataset.load_items()}

    results: dict[str, dict[str, float]] = {}

    for name in TARGETS:
        ref_clip = clips.get(items_by_id[name].drug)
        if ref_clip is None:
            print(f"{name}: NO human reference clip, skipping")
            continue
        ipa = IPA[name]
        item = items_by_id[name]
        row: dict[str, float] = {}
        print(f"\n=== {name}  (IPA: {ipa}, ref: {ref_clip.source}) ===")

        # 1. gemini-baseline: reuse the already-extracted span from the real run
        try:
            rec = recs[name]
            full_audio = (RUN_DIR / rec["audio_path"]).read_bytes()
            span = extract_drug_span_forced_align(full_audio, item.sentence, item.drug)
            f1 = score_against_reference(span, ref_clip.path)
            row["gemini-baseline"] = f1
            print(f"  gemini-baseline    f1={f1:.3f}  dur={clip_duration_s(span):.2f}s")
        except Exception as exc:
            print(f"  gemini-baseline    FAILED: {exc}")

        # 2. cloudtts-plain
        try:
            clip = cloudtts_synthesize(text=item.drug, session=session)
            f1 = score_against_reference(clip, ref_clip.path)
            row["cloudtts-plain"] = f1
            print(f"  cloudtts-plain     f1={f1:.3f}  dur={clip_duration_s(clip):.2f}s")
        except Exception as exc:
            print(f"  cloudtts-plain     FAILED: {exc}")

        # 3. cloudtts-ssml-ipa
        try:
            ssml = f'<speak><phoneme alphabet="ipa" ph="{xml_escape(ipa)}">{xml_escape(item.drug)}</phoneme></speak>'
            clip = cloudtts_synthesize(ssml=ssml, session=session)
            f1 = score_against_reference(clip, ref_clip.path)
            row["cloudtts-ssml-ipa"] = f1
            print(f"  cloudtts-ssml-ipa  f1={f1:.3f}  dur={clip_duration_s(clip):.2f}s")
        except Exception as exc:
            print(f"  cloudtts-ssml-ipa  FAILED: {exc}")

        # 4. gemini-ipa-text: bare IPA as the literal input text
        try:
            clip = gemini_synthesize(f"/{ipa}/", project=GCP_PROJECT, session=session)
            f1 = score_against_reference(clip, ref_clip.path)
            row["gemini-ipa-text"] = f1
            print(f"  gemini-ipa-text    f1={f1:.3f}  dur={clip_duration_s(clip):.2f}s")
        except Exception as exc:
            print(f"  gemini-ipa-text    FAILED: {exc}")

        # 5. gemini-ipa-hint: spelling + IPA together
        try:
            clip = gemini_synthesize(f"{item.drug} (IPA: /{ipa}/)", project=GCP_PROJECT, session=session)
            f1 = score_against_reference(clip, ref_clip.path)
            row["gemini-ipa-hint"] = f1
            print(f"  gemini-ipa-hint    f1={f1:.3f}  dur={clip_duration_s(clip):.2f}s")
        except Exception as exc:
            print(f"  gemini-ipa-hint    FAILED: {exc}")

        results[name] = row

    print("\n" + "=" * 90)
    print("SUMMARY (F1, higher = closer to this item's real human reference)")
    print("=" * 90)
    methods = ["gemini-baseline", "cloudtts-plain", "cloudtts-ssml-ipa", "gemini-ipa-text", "gemini-ipa-hint"]
    header = f"{'drug':16s}" + "".join(f"{m:>20s}" for m in methods)
    print(header)
    for name, row in results.items():
        line = f"{name:16s}" + "".join(f"{row.get(m, float('nan')):20.3f}" for m in methods)
        print(line)
    print()
    for m in methods:
        vals = [row[m] for row in results.values() if m in row]
        if vals:
            print(f"{m:20s} mean={sum(vals)/len(vals):.3f}  n={len(vals)}")

    print("\n=== brand vs generic breakdown ===")
    for grp in ("brand", "generic"):
        names_in_grp = [n for n in results if NAME_TYPE.get(n) == grp]
        print(f"\n{grp} (n={len(names_in_grp)}): {names_in_grp}")
        for m in methods:
            vals = [results[n][m] for n in names_in_grp if m in results[n]]
            if vals:
                print(f"  {m:20s} mean={sum(vals)/len(vals):.3f}")

    (REPO_ROOT / "runs" / "synthetic-reference-probe-v2.json").write_text(
        json.dumps({"results": results, "name_type": NAME_TYPE}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
