#!/usr/bin/env python
"""Holdout: keep published respelling text, vary only the Gemini prompt.

The hyphen string is the external reference. Do not Title-case, lowercase,
or otherwise rewrite ALL-CAPS stress. Score Path 2 F1 vs the human clip.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
_hf = ROOT / ".cache" / "huggingface"
_hf.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_hf))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_hf))
os.environ.setdefault("HF_HUB_CACHE", str(_hf / "hub"))

import librosa  # noqa: E402
import requests  # noqa: E402

from eval_gemini31_holdout import (  # noqa: E402
    ENDPOINT,
    HOLDOUT,
    MODEL,
    PROJECT,
    RATE,
    VOICE,
    cloud_ipa_path,
    f1,
    mean,
    slug,
    token,
)
from dose_r.references.reference_clips import available_clips  # noqa: E402

HOLD_TTS = ROOT / "runs" / "gemini31-holdout" / "tts"
OUT = ROOT / "runs" / "gemini31-holdout-prompt"
TTS = OUT / "tts"
LISTEN = ROOT / "runs" / "listen-gemini31-holdout-prompt"
WORKERS = int(os.environ.get("TTS_WORKERS", "4"))
GAP = 0.03

PROMPTS = {
    "default": "Pronounce this US drug name clearly as a single name.",
    "syllables_not_acronym": (
        "Say this dictionary respelling as one spoken drug name. "
        "Hyphens mark syllables. Capital letters mark stress, not an acronym. "
        "Never spell letters."
    ),
    "one_word": (
        "You are saying a single US drug name out loud to a patient. "
        "Read it as one word. Do not spell it. Do not pause on capital letters."
    ),
}


def synth(text: str, prompt: str) -> bytes:
    import base64
    import time

    body = {
        "input": {"text": text, "prompt": prompt},
        "voice": {"languageCode": "en-US", "name": VOICE, "modelName": MODEL},
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
            last = RuntimeError(resp.text[:200])
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"{resp.status_code}: {resp.text[:400]}")
        return base64.b64decode(resp.json()["audioContent"])
    raise last or RuntimeError("synth failed")


def cached(path: Path, text: str, prompt: str, reuse: Path | None = None) -> bytes:
    if path.exists() and path.stat().st_size > 500:
        return path.read_bytes()
    if reuse is not None and reuse.exists() and reuse.stat().st_size > 500:
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(reuse, path)
        return path.read_bytes()
    wav = synth(text, prompt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(wav)
    return wav


def dur(path: Path) -> float:
    audio, sr = librosa.load(str(path), sr=None, mono=True)
    return len(audio) / sr


def main() -> int:
    items = json.loads(HOLDOUT.read_text())["core_holdout"]
    clips = available_clips()
    TTS.mkdir(parents=True, exist_ok=True)

    jobs = []
    for rec in items:
        name = rec["ingredient"]
        text = (rec.get("respelling") or "").strip()
        if not text:
            continue
        s = slug(name)
        for key, prompt in PROMPTS.items():
            path = TTS / f"{s}.{key}.wav"
            reuse = HOLD_TTS / f"{s}.hyphen.wav" if key == "default" else None
            jobs.append((name, key, text, prompt, path, reuse))

    print(f"{len(items)} holdout names, {len(jobs)} synths, text=published respelling", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {
            ex.submit(cached, path, text, prompt, reuse): (name, key, text)
            for name, key, text, prompt, path, reuse in jobs
        }
        done = 0
        for fut in as_completed(futs):
            name, key, text = futs[fut]
            done += 1
            try:
                fut.result()
                print(f"  ok {done}/{len(jobs)} {name} {key} {text!r}", flush=True)
            except Exception as exc:
                print(f"  FAIL {name} {key}: {exc}", flush=True)

    rows = []
    for rec in items:
        name = rec["ingredient"]
        clip = clips.get(name)
        if clip is None:
            continue
        text = rec.get("respelling") or ""
        s = slug(name)
        ipa = cloud_ipa_path(name)
        vs: dict = {}
        durs: dict = {}
        vs_ipa: dict = {}
        for key in PROMPTS:
            p = TTS / f"{s}.{key}.wav"
            if not p.exists():
                vs[key] = None
                continue
            vs[key] = round(f1(p, clip.path), 4)
            durs[key] = round(dur(p), 3)
            vs_ipa[key] = round(f1(p, ipa), 4) if ipa else None
            print(
                f"  {name} {key}: F1={vs[key]} dur={durs[key]} ipa={vs_ipa[key]}",
                flush=True,
            )
        cloud = round(f1(ipa, clip.path), 4) if ipa else None
        rows.append(
            {
                "ingredient": name,
                "respelling": text,
                "vs_human": vs,
                "vs_ipa": vs_ipa,
                "dur": durs,
                "cloud_ipa_vs_human": cloud,
            }
        )

    summary = {
        "n": len(rows),
        "text": "published respelling, unchanged",
        "prompts": PROMPTS,
        "by_prompt": {},
    }
    for key in PROMPTS:
        h = [r["vs_human"][key] for r in rows if r["vs_human"].get(key) is not None]
        i = [r["vs_ipa"][key] for r in rows if r["vs_ipa"].get(key) is not None]
        ds = [r["dur"][key] for r in rows if r["dur"].get(key) is not None]
        summary["by_prompt"][key] = {
            "n": len(h),
            "mean_vs_human": mean(h),
            "mean_vs_ipa": mean(i),
            "mean_dur": mean(ds),
        }
    base = "default"
    for key in PROMPTS:
        if key == base:
            continue
        wins = ties = losses = 0
        for r in rows:
            a, b = r["vs_human"].get(base), r["vs_human"].get(key)
            if a is None or b is None:
                continue
            if b >= a + GAP:
                wins += 1
            elif a >= b + GAP:
                losses += 1
            else:
                ties += 1
        summary["by_prompt"][key]["vs_default"] = {
            "wins": wins,
            "ties": ties,
            "losses": losses,
        }
    ranked = sorted(
        summary["by_prompt"].items(),
        key=lambda kv: kv[1]["mean_vs_human"] or 0,
        reverse=True,
    )
    summary["best_vs_human"] = ranked[0][0]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False)
        + "\n"
    )

    LISTEN.mkdir(parents=True, exist_ok=True)
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Holdout prompt-only hyphen</title>",
        "<style>body{font:16px/1.4 system-ui;max-width:820px;margin:2rem auto;padding:0 1rem}",
        "section{border:1px solid #ccc;border-radius:8px;padding:1rem 1.2rem;margin:1rem 0}",
        "h1{font-size:1.2rem} label{display:block;font-weight:600;margin:.4rem 0 .1rem}",
        "audio{width:100%} .ipa{font-family:ui-monospace,monospace}",
        ".meta{font-size:.9rem;color:#555}</style>",
        "<h1>Holdout: published hyphen text, prompt only</h1>",
        f"<p class='meta'>{json.dumps(summary['by_prompt'])}</p>",
    ]
    for rec in rows:
        s = slug(rec["ingredient"])
        clip = clips[rec["ingredient"]]
        parts.append(f"<section><h2>{rec['ingredient']}</h2>")
        parts.append(
            f"<p>unchanged <span class='ipa'>{rec['respelling']}</span></p>"
        )
        dest = LISTEN / f"{s}__human{clip.path.suffix}"
        if not dest.exists():
            dest.write_bytes(clip.path.read_bytes())
        parts.append(
            f"<label>human</label><audio controls src='{dest.name}'></audio>"
        )
        ipa = cloud_ipa_path(rec["ingredient"])
        if ipa is not None:
            d2 = LISTEN / f"{s}__ipa.wav"
            if not d2.exists():
                shutil.copy2(ipa, d2)
            parts.append(
                f"<label>Cloud IPA vs human {rec.get('cloud_ipa_vs_human')}</label>"
                f"<audio controls src='{d2.name}'></audio>"
            )
        for key in PROMPTS:
            src = TTS / f"{s}.{key}.wav"
            if not src.exists():
                continue
            dest = LISTEN / f"{s}__{key}.wav"
            shutil.copy2(src, dest)
            parts.append(
                f"<label>{key} F1 {rec['vs_human'].get(key)} "
                f"dur {rec['dur'].get(key)}s</label>"
                f"<audio controls src='{dest.name}'></audio>"
            )
        parts.append("</section>")
    html = LISTEN / "index.html"
    html.write_text("\n".join(parts) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(html, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
