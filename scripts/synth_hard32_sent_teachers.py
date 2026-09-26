#!/usr/bin/env python3
"""New clinical sentences for the 32 hard names, Gemini 3.1 + locked gold IPA.

Does not edit data/gold_gemini_ipa. Teachers land in runs/ft-hard32-sent/.
Spelling stays in the sentence; IPA is the gold string via customPronunciations.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import requests

from dose_r.references.tts_pronunciation import custom_pronunciation

BASE_SCORES = ROOT / "runs" / "dose-ctc-vs-gemini-ipa" / "pron-gemini-base-full" / "scores.jsonl"

ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "project-amer-scs-sandbox")
MODEL = "gemini-3.1-flash-tts-preview"
VOICE = "Kore"
RATE = 24000
WORK = ROOT / "runs" / "ft-hard32-sent"
TEACHER_DIR = WORK / "teachers"
GOLD_MANIFEST = ROOT / "data" / "gold_gemini_ipa" / "manifest.jsonl"
DOSE = ROOT / "data" / "dose_v1.jsonl"
WORKERS = 4

# New sentences. None of these is a DoSE carrier.
FRAMES = [
    ("start", "The clinic will start {name} this week if labs remain stable."),
    ("confirm", "Please confirm the {name} dose before the next visit."),
    ("monitor", "Patients taking {name} need monitoring for side effects."),
    ("discuss", "I want to discuss the risks and benefits of {name} with you today."),
    ("counsel", "Pharmacy should counsel the patient on how to take {name} correctly."),
    ("hold", "If side effects worsen we may hold {name} and reassess."),
    ("continue", "We will review labs before deciding whether to continue {name}."),
    ("interact", "Do not combine {name} with strong CYP inducers without checking interactions."),
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


def strip_ipa(ipa: str) -> str:
    return (ipa or "").strip().strip("/")


def load_gold() -> dict[str, dict]:
    out = {}
    for line in GOLD_MANIFEST.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["ingredient"].lower()] = rec
    return out


def lowest(n_names: int, gold: dict) -> list[str]:
    rows = []
    for line in BASE_SCORES.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("gemini_f1") is None:
            continue
        rows.append(rec)
    rows.sort(key=lambda r: r["gemini_f1"])
    names, seen = [], set()
    for rec in rows:
        key = rec["drug"].lower()
        if key in seen or key not in gold:
            continue
        seen.add(key)
        names.append(rec["drug"])
        if len(names) >= n_names:
            break
    return names


def dose_sentences() -> set[str]:
    out = set()
    if not DOSE.exists():
        return out
    for line in DOSE.read_text().splitlines():
        if line.strip():
            out.add(json.loads(line)["sentence"].strip())
    return out


def pick_frames(name: str) -> list[tuple[str, str]]:
    digest = hashlib.sha256(name.lower().encode()).digest()
    order = list(range(len(FRAMES)))
    # Deterministic shuffle from name hash so each drug gets a different mix.
    for i in range(len(order) - 1, 0, -1):
        j = digest[i % len(digest)] % (i + 1)
        order[i], order[j] = order[j], order[i]
    return [FRAMES[i] for i in order[:4]]


def slug(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def synth_gemini(sentence: str, phrase: str, ipa: str) -> bytes:
    prompt = (
        "Read this clinical sentence naturally. "
        f"Pronounce {phrase} using this IPA exactly: /{ipa}/. "
        "Do not spell the drug letter by letter."
    )
    body = {
        "input": {
            "text": sentence,
            "prompt": prompt,
            "customPronunciations": custom_pronunciation(phrase, ipa),
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
    gold = load_gold()
    dose = dose_sentences()
    names = lowest(32, gold)
    TEACHER_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name in names:
        g = gold.get(name.lower())
        if g is None:
            raise SystemExit(f"missing gold IPA for {name}")
        ipa = strip_ipa(g.get("ipa_used") or g["ipa"])
        phrase = name
        for sid, tmpl in pick_frames(name):
            text = tmpl.format(name=phrase)
            if phrase not in text:
                raise SystemExit(f"phrase {phrase!r} not in {text!r}")
            if text.strip() in dose:
                raise SystemExit(f"accidentally reused DoSE carrier: {text}")
            dest = TEACHER_DIR / f"{slug(name)}__{sid}.wav"
            jobs.append({
                "name": name,
                "id": sid,
                "text": text,
                "phrase": phrase,
                "ipa": ipa,
                "wav": dest,
            })
    (WORK / "sentences.json").write_text(json.dumps(
        [{k: (str(v) if k == "wav" else v) for k, v in j.items()} for j in jobs],
        indent=2,
    ))
    print(f"synth {len(jobs)} sentences for {len(names)} names", flush=True)
    token()  # warm ADC before workers

    def one(job: dict) -> dict:
        dest: Path = job["wav"]
        if dest.exists() and dest.stat().st_size > 500:
            print("have", dest.name, flush=True)
            return job
        print("synth", dest.name, flush=True)
        dest.write_bytes(synth_gemini(job["text"], job["phrase"], job["ipa"]))
        return job

    meta = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(one, j) for j in jobs]
        for fut in as_completed(futs):
            meta.append(fut.result())
    meta.sort(key=lambda r: (r["name"].lower(), r["id"]))
    (TEACHER_DIR / "manifest.json").write_text(json.dumps(
        [{**{k: v for k, v in r.items() if k != "wav"}, "wav": str(r["wav"])} for r in meta],
        indent=2,
    ))
    print("wrote", len(meta), "teachers", TEACHER_DIR, flush=True)


if __name__ == "__main__":
    main()
