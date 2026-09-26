#!/usr/bin/env python3
"""Synth Gemini+IPA teachers for vorasidenib sentence hillclimb pack."""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import requests

from dose_r.references.tts_pronunciation import custom_pronunciation

ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "project-amer-scs-sandbox")
MODEL = "gemini-3.1-flash-tts-preview"
VOICE = "Kore"
RATE = 24000
DRUG = "vorasidenib"
IPA = "vɔːrəˈsɪdənɪb"
TEACHER_DIR = ROOT / "runs" / "ft-vora-sent-hillclimb" / "teachers"
OLD = ROOT / "runs/vora-sent-cloud-vs-gemini/tts"

# Keep in sync with ft_vora_sent_hillclimb.ALL_SENTENCES
ALL_SENTENCES = [
    {"id": "dose", "text": "Let's initiate vorasidenib therapy for this patient to target the mutant IDH1 and IDH2 enzymes in the tumor.", "eval": True},
    {"id": "prescribed", "text": "The oncologist prescribed vorasidenib for the patient's IDH-mutant glioma.", "eval": True},
    {"id": "confirm", "text": "Please confirm the vorasidenib dose before the next clinic visit.", "eval": True},
    {"id": "monitor", "text": "Patients taking vorasidenib need monitoring for liver enzyme elevation.", "eval": True},
    {"id": "start", "text": "We will start vorasidenib this week if labs remain stable.", "eval": False},
    {"id": "discuss", "text": "I want to discuss the risks and benefits of vorasidenib with you today.", "eval": False},
    {"id": "oral", "text": "Vorasidenib is taken by mouth once daily with or without food.", "eval": False},
    {"id": "switch", "text": "If side effects worsen we may hold vorasidenib and reassess.", "eval": False},
    {"id": "idh", "text": "Because the tumor carries an IDH mutation, vorasidenib is a reasonable option.", "eval": False},
    {"id": "pharmacy", "text": "Please counsel the patient on how to store and take vorasidenib correctly.", "eval": False},
    {"id": "followup", "text": "At follow-up we will review imaging and decide whether to continue vorasidenib.", "eval": False},
    {"id": "combo", "text": "Do not combine vorasidenib with strong CYP inducers without checking interactions.", "eval": False},
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
        "voice": {"languageCode": "en-US", "name": VOICE, "modelName": MODEL},
        "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": RATE},
    }
    last = None
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
            raise RuntimeError(f"{resp.status_code}: {resp.text[:400]}")
        return base64.b64decode(resp.json()["audioContent"])
    raise last or RuntimeError("gemini failed")


def main() -> None:
    TEACHER_DIR.mkdir(parents=True, exist_ok=True)
    meta = []
    for item in ALL_SENTENCES:
        dest = TEACHER_DIR / f"{item['id']}__gemini.wav"
        old = OLD / f"{item['id']}__gemini.wav"
        if dest.exists() and dest.stat().st_size > 500:
            print("have", item["id"], flush=True)
        elif old.exists() and item["id"] in {"dose", "prescribed", "confirm", "monitor"}:
            dest.write_bytes(old.read_bytes())
            print("copy", item["id"], flush=True)
        else:
            print("synth", item["id"], flush=True)
            dest.write_bytes(synth_gemini(item["text"]))
        meta.append({"id": item["id"], "text": item["text"], "eval": item["eval"], "wav": str(dest)})
    (TEACHER_DIR / "manifest.json").write_text(json.dumps(meta, indent=2))
    print("teachers", len(meta), TEACHER_DIR, flush=True)


if __name__ == "__main__":
    main()
