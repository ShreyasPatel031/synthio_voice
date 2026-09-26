#!/usr/bin/env python3
"""Cloud vs Gemini: sentence + vorasidenib IPA → CTC-cut Path-2 F1.

Uses source IPA from finetune pack (never G2P). Spelling stays in the
sentence; IPA is injected via customPronunciations (and Gemini prompt).
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_hf = ROOT / ".cache" / "huggingface"
_hf.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_hf))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_hf))
os.environ.setdefault("HF_HUB_CACHE", str(_hf / "hub"))

import requests
import soundfile as sf

from dose_r.forced_align import extract_drug_span_forced_align
from dose_r.references.reference_clips import available_clips_all
from dose_r.references.tts_pronunciation import custom_pronunciation
from dose_r.scoring.candidate_eval import score_against_best_reference
from dose_r.scoring.speech_similarity import extract_frame_embeddings, speech_bertscore

ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "project-amer-scs-sandbox")
CLOUD = "en-US-Standard-C"
GEMINI_MODEL = "gemini-3.1-flash-tts-preview"
GEMINI_VOICE = "Kore"
RATE = 24000
DRUG = "vorasidenib"
IPA = "vɔːrəˈsɪdənɪb"  # source IPA from data/gold_gemini_ipa
OUT = ROOT / "runs" / "vora-sent-cloud-vs-gemini"
TTS = OUT / "tts"
LISTEN = ROOT / "runs" / "listen-vora-sent-cloud-vs-gemini"

SENTENCES = [
    {
        "id": "dose",
        "text": "Let's initiate vorasidenib therapy for this patient to target the mutant IDH1 and IDH2 enzymes in the tumor.",
    },
    {
        "id": "prescribed",
        "text": "The oncologist prescribed vorasidenib for the patient's IDH-mutant glioma.",
    },
    {
        "id": "confirm",
        "text": "Please confirm the vorasidenib dose before the next clinic visit.",
    },
    {
        "id": "monitor",
        "text": "Patients taking vorasidenib need monitoring for liver enzyme elevation.",
    },
]

_token = {"value": None, "exp": 0.0}


def token() -> str:
    now = time.time()
    if _token["value"] and now < _token["exp"]:
        return _token["value"]
    import google.auth
    from google.auth.transport.requests import Request

    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    creds.refresh(Request())
    _token["value"] = creds.token
    _token["exp"] = now + 3000
    return _token["value"]


def synth_cloud(sentence: str) -> bytes:
    body = {
        "input": {
            "text": sentence,
            "customPronunciations": custom_pronunciation(DRUG, IPA),
        },
        "voice": {"languageCode": "en-US", "name": CLOUD},
        "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": RATE},
    }
    resp = requests.post(
        ENDPOINT,
        headers={
            "Authorization": f"Bearer {token()}",
            "Content-Type": "application/json",
            "x-goog-user-project": PROJECT,
        },
        json=body,
        timeout=90,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"cloud {resp.status_code}: {resp.text[:400]}")
    return base64.b64decode(resp.json()["audioContent"])


def synth_gemini(sentence: str) -> bytes:
    prompt = (
        "Read this clinical sentence naturally. "
        f"Pronounce the drug name using this IPA exactly: /{IPA}/. "
        "Do not spell the drug letter by letter."
    )
    body = {
        "input": {
            "text": sentence,
            "prompt": prompt,
            "customPronunciations": custom_pronunciation(DRUG, IPA),
        },
        "voice": {
            "languageCode": "en-US",
            "name": GEMINI_VOICE,
            "modelName": GEMINI_MODEL,
        },
        "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": RATE},
    }
    last: Exception | None = None
    for attempt in range(6):
        resp = requests.post(
            ENDPOINT,
            headers={
                "Authorization": f"Bearer {token()}",
                "Content-Type": "application/json",
                "x-goog-user-project": PROJECT,
            },
            json=body,
            timeout=120,
        )
        if resp.status_code == 429:
            time.sleep(min(2**attempt, 20))
            last = RuntimeError(resp.text[:200])
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"gemini {resp.status_code}: {resp.text[:400]}")
        return base64.b64decode(resp.json()["audioContent"])
    raise last or RuntimeError("gemini failed")


def cached(path: Path, fn, *args) -> bytes:
    if path.exists() and path.stat().st_size > 500:
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = fn(*args)
    path.write_bytes(data)
    return data


def wav_bytes_from_path(path: Path) -> bytes:
    # TTS returns LINEAR16 wav already; pass through for aligner.
    return path.read_bytes()


def score_span(span: bytes, human, teacher_emb) -> dict:
    best = score_against_best_reference(span, human)
    emb = extract_frame_embeddings(span)
    t_f1 = float(speech_bertscore(emb, teacher_emb)["f1"])
    return {
        "human_f1": round(best.best_f1, 4),
        "teacher_f1": round(t_f1, 4),
        "human_source": best.best_source,
    }


def mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def main() -> None:
    TTS.mkdir(parents=True, exist_ok=True)
    LISTEN.mkdir(parents=True, exist_ok=True)

    clips = {}
    for ing, clist in available_clips_all().items():
        clips.setdefault(ing.lower(), []).extend(clist)
    human = clips.get(DRUG.lower(), [])
    if not human:
        raise SystemExit("no human clip for vorasidenib")

    teacher = ROOT / "data/gold_gemini_ipa/wavs/vorasidenib.wav"
    teacher_emb = extract_frame_embeddings(teacher.read_bytes())

    rows = []
    for item in SENTENCES:
        sid = item["id"]
        sentence = item["text"]
        print(f"=== {sid} ===", flush=True)
        cloud_path = TTS / f"{sid}__cloud.wav"
        gem_path = TTS / f"{sid}__gemini.wav"
        cached(cloud_path, synth_cloud, sentence)
        cached(gem_path, synth_gemini, sentence)

        row = {
            "id": sid,
            "sentence": sentence,
            "ipa": IPA,
            "drug": DRUG,
        }
        for arm, path in (("cloud", cloud_path), ("gemini", gem_path)):
            raw = wav_bytes_from_path(path)
            span = extract_drug_span_forced_align(raw, sentence, DRUG)
            if span is None:
                row[f"{arm}_error"] = "align_none"
                print(arm, "align failed", flush=True)
                continue
            span_path = TTS / f"{sid}__{arm}_span.wav"
            span_path.write_bytes(span)
            scores = score_span(span, human, teacher_emb)
            row[f"{arm}_human_f1"] = scores["human_f1"]
            row[f"{arm}_teacher_f1"] = scores["teacher_f1"]
            print(
                f"{arm}: human={scores['human_f1']} teacher={scores['teacher_f1']}",
                flush=True,
            )
        if row.get("cloud_human_f1") is not None and row.get("gemini_human_f1") is not None:
            d = row["gemini_human_f1"] - row["cloud_human_f1"]
            row["delta_human_gemini_minus_cloud"] = round(d, 4)
            row["winner_human"] = (
                "gemini" if d > 0.01 else "cloud" if d < -0.01 else "tie"
            )
        rows.append(row)

    summary = {
        "ipa": IPA,
        "n": len(rows),
        "mean_cloud_human": mean([r["cloud_human_f1"] for r in rows if r.get("cloud_human_f1") is not None]),
        "mean_gemini_human": mean([r["gemini_human_f1"] for r in rows if r.get("gemini_human_f1") is not None]),
        "mean_cloud_teacher": mean([r["cloud_teacher_f1"] for r in rows if r.get("cloud_teacher_f1") is not None]),
        "mean_gemini_teacher": mean([r["gemini_teacher_f1"] for r in rows if r.get("gemini_teacher_f1") is not None]),
        "n_gemini_wins": sum(1 for r in rows if r.get("winner_human") == "gemini"),
        "n_cloud_wins": sum(1 for r in rows if r.get("winner_human") == "cloud"),
        "n_tie": sum(1 for r in rows if r.get("winner_human") == "tie"),
        "rows": rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2), flush=True)

    # listen page
    parts = [
        "<!DOCTYPE html><html><head><meta charset=utf-8>",
        "<title>vorasidenib sentences — Cloud vs Gemini + IPA</title>",
        "<style>body{font:16px/1.45 -apple-system,sans-serif;max-width:780px;margin:2rem auto;padding:0 1rem}",
        "table{border-collapse:collapse;margin:1rem 0;width:100%;font-size:.92rem}",
        "td,th{border:1px solid #ddd;padding:.35rem .5rem;text-align:left}",
        "h2{margin-top:1.6rem;font-size:1.05rem} label{display:block;margin:.45rem 0 .1rem;font-weight:600}",
        "audio{width:100%} .meta{color:#555;font-size:.9rem} blockquote{margin:.4rem 0;color:#333}</style></head><body>",
        "<h1>vorasidenib in sentences — Cloud vs Gemini (IPA sidecar)</h1>",
        f"<p class='meta'>IPA <code>/{IPA}/</code> via customPronunciations. Metric: CTC-cut Path-2 F1 vs human.</p>",
        "<table><tr><th></th><th>mean vs human</th><th>mean vs isolated teacher</th><th>wins</th></tr>",
        f"<tr><td>Cloud Standard-C + IPA</td><td>{summary['mean_cloud_human']}</td><td>{summary['mean_cloud_teacher']}</td><td>{summary['n_cloud_wins']}</td></tr>",
        f"<tr><td>Gemini 3.1 + IPA</td><td>{summary['mean_gemini_human']}</td><td>{summary['mean_gemini_teacher']}</td><td>{summary['n_gemini_wins']}</td></tr>",
        "</table>",
    ]
    # human ref
    h0 = human[0]
    hdest = LISTEN / f"0_human{Path(h0.path).suffix}"
    hdest.write_bytes(Path(h0.path).read_bytes())
    parts.append(f"<label>Human ({h0.source})</label><audio controls src='{hdest.name}'></audio>")

    for r in rows:
        sid = r["id"]
        parts.append(f"<h2>{sid} · winner={r.get('winner_human')}</h2>")
        parts.append(f"<blockquote>{r['sentence']}</blockquote>")
        parts.append(
            f"<p class='meta'>cloud human {r.get('cloud_human_f1')} · "
            f"gemini human {r.get('gemini_human_f1')} · "
            f"Δ(g−c) {r.get('delta_human_gemini_minus_cloud')}</p>"
        )
        for arm, label in (("cloud", "Cloud + IPA"), ("gemini", "Gemini + IPA")):
            full = TTS / f"{sid}__{arm}.wav"
            span = TTS / f"{sid}__{arm}_span.wav"
            if full.exists():
                dest = LISTEN / f"{sid}__{arm}.wav"
                dest.write_bytes(full.read_bytes())
                parts.append(f"<label>{label} (full)</label><audio controls src='{dest.name}'></audio>")
            if span.exists():
                dest = LISTEN / f"{sid}__{arm}_span.wav"
                dest.write_bytes(span.read_bytes())
                parts.append(
                    f"<label>{label} CTC span "
                    f"(human {r.get(f'{arm}_human_f1')})</label>"
                    f"<audio controls src='{dest.name}'></audio>"
                )
    parts.append("</body></html>")
    (LISTEN / "index.html").write_text("\n".join(parts))
    print("listen", LISTEN / "index.html", flush=True)


if __name__ == "__main__":
    main()
