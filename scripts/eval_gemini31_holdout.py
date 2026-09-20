#!/usr/bin/env python
"""Gemini 3.1 Flash TTS as a third voice — holdout only.

Input is the original source string (plain spelling or dictionary
respelling). Never G2P respelling to IPA.

Variants:
  plain    ingredient spelling
  hyphen   published respelling as-is (lin-e-RIX-i-bat)
  spaced   lowercase syllables (lin e rix i bat)
  compact  hyphens stripped (linerixibat)

Score wavlm F1 vs the human clip and vs Cloud Standard-C + Google IPA.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_hf = ROOT / ".cache" / "huggingface"
_hf.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_hf))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_hf))
os.environ.setdefault("HF_HUB_CACHE", str(_hf / "hub"))

import requests  # noqa: E402

from dose_r.references.reference_clips import available_clips  # noqa: E402
from dose_r.references.tts_pronunciation import compact_ascii  # noqa: E402

ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"
MODEL = "gemini-3.1-flash-tts-preview"
VOICE = "Kore"
RATE = 24000
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "project-amer-scs-sandbox")
PROMPT = "Pronounce this US drug name clearly as a single name."
HOLDOUT = ROOT / "dose_r" / "references" / "plain_tts_gap_holdout.json"
VERIFY_TTS = ROOT / "runs" / "verify-google-ipa" / "tts"
RESPELL_TTS = ROOT / "runs" / "ipa-vs-respell" / "tts"
OUT = ROOT / "runs" / "gemini31-holdout"
TTS = OUT / "tts"
LISTEN = ROOT / "runs" / "listen-gemini31-holdout"
WORKERS = int(os.environ.get("TTS_WORKERS", "4"))

_lock = threading.Lock()
_token = {"value": None, "exp": 0.0}
_EMB: dict[int, object] = {}

VARIANTS = ("plain", "hyphen", "spaced", "compact")


def token() -> str:
    now = time.time()
    with _lock:
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


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def verify_slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def cloud_ipa_path(name: str) -> Path | None:
    s = verify_slug(name)
    for p in (VERIFY_TTS / f"{s}.ipa.wav", RESPELL_TTS / f"{s}.ipa.wav"):
        if p.exists() and p.stat().st_size > 500:
            return p
    return None


def spaced(canonical: str) -> str:
    return " ".join(canonical.replace("-", " ").split()).lower()


def texts(name: str, canonical: str) -> dict[str, str]:
    return {
        "plain": name,
        "hyphen": canonical,
        "spaced": spaced(canonical),
        "compact": compact_ascii(canonical) or name,
    }


def synth_gemini(text: str) -> bytes:
    body = {
        "input": {"text": text, "prompt": PROMPT},
        "voice": {
            "languageCode": "en-US",
            "name": VOICE,
            "modelName": MODEL,
        },
        "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": RATE},
    }
    last: Exception | None = None
    for attempt in range(6):
        try:
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
        except (requests.Timeout, requests.ConnectionError) as exc:
            last = exc
            time.sleep(min(2**attempt, 20))
            continue
        if resp.status_code == 429:
            time.sleep(min(2**attempt, 20))
            last = RuntimeError(f"429 {resp.text[:200]}")
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"{resp.status_code}: {resp.text[:400]}")
        return base64.b64decode(resp.json()["audioContent"])
    raise last or RuntimeError("gemini synth failed")


def cached_synth(path: Path, text: str) -> bytes:
    if path.exists() and path.stat().st_size > 500:
        return path.read_bytes()
    wav = synth_gemini(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(wav)
    return wav


def f1(a: Path | bytes, b: Path | bytes) -> float:
    from dose_r.scoring.speech_similarity import (
        extract_frame_embeddings,
        speech_bertscore,
    )

    def emb(data: Path | bytes):
        if isinstance(data, Path):
            key = ("p", str(data), data.stat().st_size)
        else:
            key = ("b", hash(bytes(data)), len(data))
        if key not in _EMB:
            _EMB[key] = extract_frame_embeddings(data)
        return _EMB[key]

    return float(speech_bertscore(emb(a), emb(b))["f1"])


def mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def write_listen(rows: list[dict], clips: dict) -> Path:
    LISTEN.mkdir(parents=True, exist_ok=True)
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Gemini 3.1 Flash TTS holdout</title>",
        "<style>body{font:16px/1.4 system-ui;max-width:820px;margin:2rem auto;padding:0 1rem}",
        "section{border:1px solid #ccc;border-radius:8px;padding:1rem 1.2rem;margin:1rem 0}",
        "h1{font-size:1.25rem} h2{font-size:1.05rem;margin:0 0 .35rem}",
        "p{margin:.2rem 0 .5rem} .ipa{font-family:ui-monospace,monospace}",
        "label{display:block;font-weight:600;margin:.45rem 0 .1rem} audio{width:100%}",
        ".meta{font-size:.9rem;color:#555}</style>",
        "<h1>Gemini 3.1 Flash TTS — Path 2 holdout</h1>",
        "<p>Third voice vs human and vs Cloud Standard-C + Google IPA. "
        "Respelling is the original source string, not G2P to IPA.</p>",
    ]
    for rec in rows:
        s = slug(rec["ingredient"])
        parts.append(f"<section id='{s}'><h2>{rec['ingredient']}</h2>")
        parts.append(
            f"<p>respelling <span class='ipa'>{rec.get('respelling') or '—'}</span></p>"
        )
        scores = " · ".join(
            f"{k} vs human {rec['vs_human'].get(k)}"
            f" / vs IPA {rec['vs_ipa'].get(k)}"
            for k in VARIANTS
            if rec.get("vs_human", {}).get(k) is not None
        )
        parts.append(f"<p class='meta'>{scores}</p>")
        clip = clips.get(rec["ingredient"])
        if clip is not None:
            dest = LISTEN / f"{s}__human{clip.path.suffix}"
            dest.write_bytes(clip.path.read_bytes())
            parts.append(
                f"<label>human ({clip.source})</label>"
                f"<audio controls src='{dest.name}'></audio>"
            )
        ipa_src = cloud_ipa_path(rec["ingredient"])
        if ipa_src is not None:
            dest = LISTEN / f"{s}__cloud_ipa.wav"
            dest.write_bytes(ipa_src.read_bytes())
            cip = rec.get("cloud_ipa_vs_human")
            extra = f" vs human {cip}" if cip is not None else ""
            parts.append(
                f"<label>Cloud Standard-C + Google IPA{extra}</label>"
                f"<audio controls src='{dest.name}'></audio>"
            )
        for kind in VARIANTS:
            src = TTS / f"{s}.{kind}.wav"
            dest = LISTEN / f"{s}__{kind}.wav"
            if src.exists():
                dest.write_bytes(src.read_bytes())
                fed = rec.get("texts", {}).get(kind, "")
                parts.append(
                    f"<label>Gemini 3.1 {kind} "
                    f"(<span class='ipa'>{fed}</span>)</label>"
                    f"<audio controls src='{dest.name}'></audio>"
                )
        parts.append("</section>")
    dest = LISTEN / "index.html"
    dest.write_text("\n".join(parts) + "\n")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    hold = json.loads(HOLDOUT.read_text())
    items = hold["core_holdout"]
    if args.only:
        want = {n.lower() for n in args.only}
        items = [r for r in items if r["ingredient"].lower() in want]
    clips = available_clips()
    TTS.mkdir(parents=True, exist_ok=True)

    jobs = []
    for rec in items:
        name = rec["ingredient"]
        canon = rec.get("respelling") or ""
        if not canon:
            continue
        fed = texts(name, canon)
        s = slug(name)
        for kind, text in fed.items():
            jobs.append((name, kind, text, TTS / f"{s}.{kind}.wav"))

    print(f"{len(items)} holdout names, {len(jobs)} synths", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {
            ex.submit(cached_synth, path, text): (name, kind, text)
            for name, kind, text, path in jobs
        }
        for fut in as_completed(futs):
            name, kind, text = futs[fut]
            try:
                fut.result()
                print(f"  ok {name} {kind} {text!r}", flush=True)
            except Exception as exc:
                print(f"  FAIL {name} {kind}: {exc}", flush=True)

    rows = []
    for rec in items:
        name = rec["ingredient"]
        clip = clips.get(name)
        if clip is None:
            print(f"skip no clip {name}", flush=True)
            continue
        human = clip.path
        ipa_path = cloud_ipa_path(name)
        fed = texts(name, rec.get("respelling") or "")
        s = slug(name)
        vs_h: dict[str, float | None] = {}
        vs_i: dict[str, float | None] = {}
        for kind in VARIANTS:
            p = TTS / f"{s}.{kind}.wav"
            if not p.exists():
                vs_h[kind] = None
                vs_i[kind] = None
                continue
            vs_h[kind] = round(f1(p, human), 4)
            vs_i[kind] = round(f1(p, ipa_path), 4) if ipa_path else None
            print(
                f"  {name} {kind}: human={vs_h[kind]} ipa={vs_i[kind]}",
                flush=True,
            )
        cloud_ipa_human = None
        if ipa_path:
            cloud_ipa_human = round(f1(ipa_path, human), 4)
        rows.append(
            {
                "ingredient": name,
                "respelling": rec.get("respelling"),
                "texts": fed,
                "clip_source": clip.source,
                "cloud_ipa_vs_human": cloud_ipa_human,
                "vs_human": vs_h,
                "vs_ipa": vs_i,
            }
        )

    summary = {
        "n": len(rows),
        "model": MODEL,
        "voice": VOICE,
        "prompt": PROMPT,
        "cloud_ipa_vs_human": mean(
            [r["cloud_ipa_vs_human"] for r in rows if r.get("cloud_ipa_vs_human") is not None]
        ),
        "by_variant": {},
    }
    for kind in VARIANTS:
        h = [r["vs_human"][kind] for r in rows if r["vs_human"].get(kind) is not None]
        i = [r["vs_ipa"][kind] for r in rows if r["vs_ipa"].get(kind) is not None]
        summary["by_variant"][kind] = {
            "n_human": len(h),
            "mean_vs_human": mean(h),
            "n_ipa": len(i),
            "mean_vs_cloud_ipa": mean(i),
        }
    ranked = sorted(
        summary["by_variant"].items(),
        key=lambda kv: (
            kv[1]["mean_vs_human"] or 0,
            kv[1]["mean_vs_cloud_ipa"] or 0,
        ),
        reverse=True,
    )
    summary["best_vs_human"] = ranked[0][0] if ranked else None
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False)
        + "\n"
    )
    html = write_listen(rows, clips)
    print(json.dumps(summary, indent=2), flush=True)
    print(html, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
