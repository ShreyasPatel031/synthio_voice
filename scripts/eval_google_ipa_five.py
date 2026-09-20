#!/usr/bin/env python
"""Inject the five Google IPA strings into Cloud TTS and score vs human.

One-off proof that *source IPA* beats Wikipedia-key G2P of DailyMed.
Do not rebuild a converter from this. See dose_r/references/README.md.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_hf = ROOT / ".cache" / "huggingface"
_hf.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(_hf)
os.environ["TRANSFORMERS_CACHE"] = str(_hf)
os.environ["HF_HUB_CACHE"] = str(_hf / "hub")

import requests  # noqa: E402

from dose_r.references.reference_clips import available_clips  # noqa: E402
from dose_r.references.tts_pronunciation import custom_pronunciation  # noqa: E402

ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"
VOICE = "en-US-Standard-C"
RATE = 24000
PROJECT = "project-amer-scs-sandbox"
OUT = ROOT / "runs" / "listen-google-ipa"
PRON = ROOT / "dose_r" / "references" / "pronunciations.jsonl"

# User paste, mapped to ingredient. Slashes/brackets stripped; ASCII ' → ˈ.
GOOGLE = {
    "Attruby": "æˈtruːbi",
    "Dupixent": "ˈduːpɪksɛnt",
    "Humira": "hjuːˈmɛr.ə",
    "Nexium": "ˈnɛksiəm",
    "aripiprazole": "ˌæɹ.ɪˈpɪp.ɹəˌzoʊl",
}


def token() -> str:
    import google.auth
    from google.auth.transport.requests import Request

    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    creds.refresh(Request())
    return creds.token


def cloud_ipa(ipa: str) -> str:
    s = (ipa or "").strip().strip("/[]() ")
    return s.replace("'", "ˈ")


def synth(tok: str, *, text: str, pronunciations: dict | None = None) -> bytes:
    inp: dict = {"text": text}
    if pronunciations:
        inp["customPronunciations"] = pronunciations
    body = {
        "input": inp,
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
        raise RuntimeError(f"{resp.status_code}: {resp.text[:500]}")
    return base64.b64decode(resp.json()["audioContent"])


def try_google_synth(tok: str, name: str, ipa: str) -> tuple[bytes, str]:
    """Try user IPA, then without syllable dots if Cloud 400s."""
    candidates = [cloud_ipa(ipa)]
    nodot = cloud_ipa(ipa.replace(".", ""))
    if nodot not in candidates:
        candidates.append(nodot)
    last = None
    for cand in candidates:
        try:
            wav = synth(
                tok, text=name, pronunciations=custom_pronunciation(name, cand)
            )
            return wav, cand
        except RuntimeError as exc:
            last = exc
            print(f"  Cloud reject {name} {cand!r}: {exc}", flush=True)
    raise last  # type: ignore[misc]


def f1(a: bytes, b: bytes) -> float:
    from dose_r.scoring.speech_similarity import (
        extract_frame_embeddings,
        speech_bertscore,
    )

    return float(speech_bertscore(extract_frame_embeddings(a), extract_frame_embeddings(b))["f1"])


def to_mp3(wav: Path, mp3: Path) -> None:
    import subprocess

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav), "-q:a", "4", str(mp3)],
        check=True,
        capture_output=True,
    )


def main() -> int:
    stored = {}
    for line in PRON.read_text().splitlines():
        rec = json.loads(line)
        stored[rec["ingredient"]] = rec

    clips = available_clips()
    tok = token()
    OUT.mkdir(parents=True, exist_ok=True)
    tts = OUT / "tts"
    tts.mkdir(exist_ok=True)

    rows = []
    for name, g_ipa in GOOGLE.items():
        clip = clips.get(name)
        if clip is None:
            raise SystemExit(f"no human clip for {name}")
        rec = stored[name]
        conv = rec["ipa_cloud"]
        human = clip.path.read_bytes()
        slug = name.lower()

        print(f"synth {name}", flush=True)
        p_plain = tts / f"{slug}.plain.wav"
        p_conv = tts / f"{slug}.converter.wav"
        p_google = tts / f"{slug}.google.wav"
        used_path = tts / f"{slug}.google.ipa.txt"
        if p_plain.exists() and p_plain.stat().st_size > 1000:
            plain = p_plain.read_bytes()
        else:
            plain = synth(tok, text=name)
            p_plain.write_bytes(plain)
        if p_conv.exists() and p_conv.stat().st_size > 1000:
            conv_wav = p_conv.read_bytes()
        else:
            conv_wav = synth(
                tok, text=name, pronunciations=custom_pronunciation(name, conv)
            )
            p_conv.write_bytes(conv_wav)
        if p_google.exists() and p_google.stat().st_size > 1000 and used_path.exists():
            google_wav = p_google.read_bytes()
            used = used_path.read_text().strip()
        else:
            google_wav, used = try_google_synth(tok, name, g_ipa)
            p_google.write_bytes(google_wav)
            used_path.write_text(used)
        (tts / f"{slug}.human{clip.path.suffix}").write_bytes(human)

        print(f"score {name}", flush=True)
        row = {
            "ingredient": name,
            "clip_source": clip.source,
            "human_path": str(clip.path),
            "converter_ipa": conv,
            "google_ipa_in": g_ipa,
            "google_ipa_used": used,
            "plain": round(f1(plain, human), 4),
            "converter": round(f1(conv_wav, human), 4),
            "google": round(f1(google_wav, human), 4),
        }
        row["delta_google_vs_plain"] = round(row["google"] - row["plain"], 4)
        row["delta_google_vs_converter"] = round(row["google"] - row["converter"], 4)
        rows.append(row)
        print(
            f"  plain={row['plain']:.3f} converter={row['converter']:.3f} "
            f"({rec['ipa']}) google={row['google']:.3f} ({used})",
            flush=True,
        )

    n = len(rows)
    summary = {
        "n": n,
        "voice": VOICE,
        "mean_plain": round(sum(r["plain"] for r in rows) / n, 4),
        "mean_converter": round(sum(r["converter"] for r in rows) / n, 4),
        "mean_google": round(sum(r["google"] for r in rows) / n, 4),
    }
    summary["google_beats_plain"] = sum(1 for r in rows if r["delta_google_vs_plain"] > 0.01)
    summary["google_beats_converter"] = sum(
        1 for r in rows if r["delta_google_vs_converter"] > 0.01
    )
    (OUT / "results.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n"
    )

    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Google IPA vs converter IPA vs plain</title>",
        "<style>body{font:16px/1.4 system-ui;max-width:760px;margin:2rem auto;padding:0 1rem}",
        "section{border:1px solid #ccc;border-radius:8px;padding:1rem 1.2rem;margin:1rem 0}",
        "h1{font-size:1.25rem} h2{font-size:1.05rem;margin:0 0 .4rem}",
        "p{margin:.25rem 0 .5rem} .ipa{font-family:ui-monospace,monospace}",
        "label{display:block;font-weight:600;margin:.5rem 0 .15rem} audio{width:100%}",
        ".meta{color:#444}</style>",
        "<h1>Same voice, same human clip. Google IPA vs our converter.</h1>",
        f"<p class='meta'>Means — plain {summary['mean_plain']:.3f} · "
        f"converter {summary['mean_converter']:.3f} · "
        f"google {summary['mean_google']:.3f}</p>",
    ]
    for rec in rows:
        slug = rec["ingredient"].lower()
        human_src = Path(rec["human_path"])
        dest_h = OUT / f"{slug}__human{human_src.suffix}"
        dest_h.write_bytes(human_src.read_bytes())
        for arm in ("plain", "converter", "google"):
            (OUT / f"{slug}__{arm}.wav").write_bytes(
                (tts / f"{slug}.{arm}.wav").read_bytes()
            )
        d_p = rec["delta_google_vs_plain"]
        d_c = rec["delta_google_vs_converter"]
        parts.append(f"<section id='{slug}'><h2>{rec['ingredient']}</h2>")
        parts.append(
            f"<p>converter: <span class='ipa'>{rec['converter_ipa']}</span><br>"
            f"google in: <span class='ipa'>{rec['google_ipa_in']}</span><br>"
            f"google used: <span class='ipa'>{rec['google_ipa_used']}</span></p>"
        )
        parts.append(
            f"<p class='meta'>plain {rec['plain']:.3f} &nbsp; "
            f"converter {rec['converter']:.3f} &nbsp; "
            f"<b>google {rec['google']:.3f}</b> "
            f"(vs plain {d_p:+.3f}, vs converter {d_c:+.3f})</p>"
        )
        parts.append(f"<label>Human ({rec['clip_source']})</label>")
        parts.append(f"<audio controls src='{dest_h.name}'></audio>")
        parts.append("<label>Plain spelling</label>")
        parts.append(f"<audio controls src='{slug}__plain.wav'></audio>")
        parts.append("<label>Our converter IPA</label>")
        parts.append(f"<audio controls src='{slug}__converter.wav'></audio>")
        parts.append("<label>Google IPA</label>")
        parts.append(f"<audio controls src='{slug}__google.wav'></audio></section>")
    (OUT / "index.html").write_text("\n".join(parts))
    print("MEANS", json.dumps(summary), flush=True)
    print("listen", OUT / "index.html", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
