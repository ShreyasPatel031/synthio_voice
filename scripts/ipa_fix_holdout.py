#!/usr/bin/env python
"""Holdout: worst Cloud-TTS-IPA-vs-plain names, after the AMA `ye`=/aɪ/ fix.

Compares wavlm(F1) of the human clip against:
  plain          Cloud TTS reading the spelling
  ipa_old        SSML injection of stored first-IPA (broken `ye` -> /jɛ/)
  ipa_fixed      SSML injection of the same source string, reconverted
  ipa_mw         SSML injection of the MW IPA variant, when one exists
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

import base64
import requests

WS = Path("/Users/shreyaspatel/Projects/synthio_voice")
PATH3 = Path("/tmp/path3-wt")
sys.path.insert(0, str(WS))

from dose_r.references.build import _respelling_text_to_variant  # noqa: E402
from dose_r.references.ipa_references import load_ipa_references  # noqa: E402

HOLD = PATH3 / "runs" / "gemini-regen-holdout"
CACHE = HOLD / "human-vs-plain-vs-ipa" / "tts"
OUT = HOLD / "ipa-fix-holdout"
ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"
VOICE = "en-US-Standard-C"
RATE = 24000
PROJECT = "project-amer-scs-sandbox"
# Worst stored-IPA vs plain from the 165-item sweep.
HOLDOUT = [
    "nuzolvence", "omalizumab", "adquey", "acetaminophen", "tofacitinib",
    "dupixent", "jardiance", "fluoxetine", "alprazolam", "attruby",
    "aripiprazole", "zycubo",
]


def token():
    import google.auth
    from google.auth.transport.requests import Request
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(Request())
    return creds.token


def synth(tok, *, text=None, ssml=None) -> bytes:
    body = {
        "input": {"ssml": ssml} if ssml else {"text": text},
        "voice": {"languageCode": "en-US", "name": VOICE},
        "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": RATE},
    }
    resp = requests.post(
        ENDPOINT,
        headers={
            "Authorization": f"Bearer {tok}",
            "Content-Type": "application/json",
            "x-goog-user-project": PROJECT,
        },
        json=body,
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"{resp.status_code}: {resp.text[:300]}")
    return base64.b64decode(resp.json()["audioContent"])


def cached(path: Path, tok, **kw) -> bytes:
    if path.exists() and path.stat().st_size > 1000:
        return path.read_bytes()
    audio = synth(tok, **kw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audio)
    return audio


def f1(a, b) -> float:
    from dose_r.scoring.speech_similarity import extract_frame_embeddings, speech_bertscore
    return speech_bertscore(extract_frame_embeddings(a), extract_frame_embeddings(b))["f1"]


def load_clips():
    """MW-preferred clips from the path3 worktree (workspace has no MW files)."""
    pri = {"merriam-webster": 0, "drugs.com": 1, "umich": 2}
    by = {}
    for line in (PATH3 / "data" / "reference_audio" / "manifest.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("coverage") != "full":
            continue
        p = PATH3 / r["local_path"]
        if not p.exists():
            continue
        by.setdefault(r["ingredient"].lower(), []).append((pri.get(r["source"], 9), p, r["source"]))
    return {k: min(v) for k, v in by.items()}


def reconvert(raw: str) -> str | None:
    raw = raw.replace("\uf0a2", "'").replace("\u201d", '"').replace("\u2019", "'")
    hit = _respelling_text_to_variant(raw)
    return hit[1] if hit else None


def main() -> int:
    prev = {r["name"]: r for r in json.loads(
        (HOLD / "human-vs-plain-vs-ipa" / "summary.json").read_text()
    )["rows"]}
    refs = load_ipa_references(WS / "dose_r" / "references" / "references.jsonl")
    clips = load_clips()
    tok = token()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in HOLDOUT:
        rec = prev[name]
        ref = refs.get(name) or refs.get(rec["spoken"].lower())
        sources = (ref or {}).get("sources") or []
        raw = next((s["raw"] for s in sources if s.get("raw")), None)
        fixed = reconvert(raw) if raw else None
        mw = next(
            (v for s, v in zip(sources, (ref or {}).get("ipa_variants") or [])
             if "merriam-webster" in s.get("name", "")),
            None,
        )
        if mw is None and ref and len(ref.get("ipa_variants") or []) > 1:
            if any("merriam-webster" in s.get("name", "") for s in sources):
                mw = ref["ipa_variants"][1]
        ye = bool(raw and any(t in raw.lower() for t in ("lye", "sye", "zye", "rye", "dye", "tye", "bye", "vye")))
        _, clip_path, clip_src = clips[name]
        human = Path(clip_path).read_bytes()
        spoken = rec["spoken"]
        plain = cached(CACHE / f"{name}.plain.wav", tok, text=spoken)
        old = cached(CACHE / f"{name}.ipa.wav", tok, ssml=(
            f'<speak><phoneme alphabet="ipa" ph="{xml_escape(rec["ipa"])}">{xml_escape(spoken)}</phoneme></speak>'
        ))
        arms = {
            "plain": plain,
            "ipa_old": old,
        }
        if fixed:
            arms["ipa_fixed"] = cached(
                OUT / f"{name}.fixed.wav", tok,
                ssml=f'<speak><phoneme alphabet="ipa" ph="{xml_escape(fixed)}">{xml_escape(spoken)}</phoneme></speak>',
            )
        if mw and mw != rec["ipa"] and mw != fixed:
            arms["ipa_mw"] = cached(
                OUT / f"{name}.mw.wav", tok,
                ssml=f'<speak><phoneme alphabet="ipa" ph="{xml_escape(mw)}">{xml_escape(spoken)}</phoneme></speak>',
            )
        scores = {k: f1(v, human) for k, v in arms.items()}
        row = {
            "name": name,
            "clip_source": clip_src,
            "ye_item": ye,
            "raw": raw,
            "ipa_old": rec["ipa"],
            "ipa_fixed": fixed,
            "ipa_mw": mw,
            "scores": scores,
            "fixed_changed": bool(fixed and fixed != rec["ipa"]),
        }
        rows.append(row)
        bits = " ".join(f"{k}={scores[k]:.3f}" for k in scores)
        print(f"{name:16s} ye={ye} changed={row['fixed_changed']} {bits}", flush=True)
        print(f"  old   {rec['ipa']}", flush=True)
        print(f"  fixed {fixed}", flush=True)
        if mw:
            print(f"  mw    {mw}", flush=True)

    def mean(key):
        xs = [r["scores"][key] for r in rows if key in r["scores"]]
        return sum(xs) / len(xs) if xs else None

    summary = {
        "n": len(rows),
        "mean_plain": mean("plain"),
        "mean_ipa_old": mean("ipa_old"),
        "mean_ipa_fixed": mean("ipa_fixed"),
        "mean_ipa_mw": mean("ipa_mw"),
        "ye_items": [r["name"] for r in rows if r["ye_item"]],
        "fixed_beats_plain": [
            r["name"] for r in rows
            if "ipa_fixed" in r["scores"] and r["scores"]["ipa_fixed"] > r["scores"]["plain"] + 0.01
        ],
        "fixed_beats_old": [
            r["name"] for r in rows
            if "ipa_fixed" in r["scores"] and r["scores"]["ipa_fixed"] > r["scores"]["ipa_old"] + 0.01
        ],
        "rows": rows,
    }
    (OUT / "results.json").write_text(json.dumps(summary, indent=2))
    print("\nmeans",
          f"plain={summary['mean_plain']:.3f}",
          f"old={summary['mean_ipa_old']:.3f}",
          f"fixed={summary['mean_ipa_fixed']:.3f}",
          f"mw={summary['mean_ipa_mw']}")
    print("fixed beats plain:", summary["fixed_beats_plain"])
    print("fixed beats old:", summary["fixed_beats_old"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
