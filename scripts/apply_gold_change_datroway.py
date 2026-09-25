#!/usr/bin/env python3
"""Apply user-approved Datroway gold change: IPA + Cloud Standard-C wav."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold_gemini_ipa"
WAV = GOLD / "wavs" / "datroway.wav"
MANIFEST = GOLD / "manifest.jsonl"
IPA_JSON = GOLD / "ipa.json"
CHANGES = GOLD / "changes.jsonl"
CLOUD_GOLD = ROOT / "runs" / "misaki-iter" / "cloud-gold" / "datroway.wav"

INGREDIENT = "Datroway"
FROM_IPA = "/ˈdæt.roʊ.weɪ/"
TO_IPA = "/ˈdætɹoʊweɪ/"
IPA_USED = "ˈdætɹoʊweɪ"
VOICE = "en-US-Standard-C"
ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"
PROJECT = __import__("os").environ.get("GOOGLE_CLOUD_PROJECT", "project-amer-scs-sandbox")

CHANGE_LINE = {
    "at": "2026-09-21",
    "ingredient": INGREDIENT,
    "action": "ipa_and_audio",
    "from_ipa": FROM_IPA,
    "to_ipa": TO_IPA,
    "teacher": VOICE,
    "suggest_respelling": "DAT-roe-way",
    "suggest_source": ["dailymed", "drugs.com human", "nci"],
    "why": (
        "User: switch Datroway gold to Cloud ˈdætɹoʊweɪ (DAT stress, no syllable dots). "
        "Same DAT-roe-way intent as /ˈdæt.roʊ.weɪ/, but dots hurt Cloud synth."
    ),
    "user_ok": True,
}


def token() -> str:
    try:
        import google.auth
        from google.auth.transport.requests import Request

        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds.refresh(Request())
        return creds.token
    except Exception:
        return subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()


def synth_cloud_iso() -> bytes:
    tok = token()
    body = {
        "input": {
            "text": INGREDIENT,
            "customPronunciations": {
                "pronunciations": [
                    {
                        "phrase": INGREDIENT,
                        "phoneticEncoding": "PHONETIC_ENCODING_IPA",
                        "pronunciation": IPA_USED,
                    }
                ]
            },
        },
        "voice": {"languageCode": "en-US", "name": VOICE},
        "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": 24000},
    }
    resp = requests.post(
        ENDPOINT,
        headers={
            "Authorization": f"Bearer {tok}",
            "Content-Type": "application/json",
            "x-goog-user-project": PROJECT,
        },
        json=body,
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Cloud TTS {resp.status_code}: {resp.text[:400]}")
    return base64.b64decode(resp.json()["audioContent"])


def chmod_writable() -> None:
    for p in (WAV, MANIFEST, IPA_JSON, CHANGES):
        if p.exists():
            subprocess.check_call(["chmod", "u+w", str(p)])


def chmod_locked() -> None:
    for p in (WAV, MANIFEST, IPA_JSON, CHANGES):
        if p.exists():
            subprocess.check_call(["chmod", "444", str(p)])


def append_change() -> None:
    line = json.dumps(CHANGE_LINE, ensure_ascii=False)
    if CHANGES.exists() and line in CHANGES.read_text():
        print("changes.jsonl already has Datroway entry", flush=True)
        return
    with CHANGES.open("a") as f:
        f.write(line + "\n")
    print("appended changes.jsonl", flush=True)


def update_manifest() -> None:
    rows = []
    for line in MANIFEST.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("ingredient") == INGREDIENT:
            rec["ipa"] = IPA_USED
            rec["ipa_used"] = IPA_USED
            rec["teacher"] = VOICE
            rec["voice"] = VOICE
        rows.append(rec)
    MANIFEST.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")


def update_ipa_json() -> None:
    data = json.loads(IPA_JSON.read_text())
    for row in data:
        if row.get("ingredient") == INGREDIENT:
            row["ipa"] = TO_IPA
    IPA_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def main() -> int:
    chmod_writable()
    append_change()
    audio = synth_cloud_iso()
    WAV.write_bytes(audio)
    update_manifest()
    update_ipa_json()
    chmod_locked()
    CLOUD_GOLD.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["chmod", "u+w", str(CLOUD_GOLD)])
    CLOUD_GOLD.write_bytes(audio)
    subprocess.check_call(["chmod", "444", str(CLOUD_GOLD)])
    print(f"WROTE {WAV} sha256={hashlib.sha256(audio).hexdigest()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
