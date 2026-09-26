#!/usr/bin/env python
"""Gemini 3.1 Flash TTS + source IPA vs Cloud Standard-C + same IPA.

Same published IPA string Cloud already accepted (`ipa_used`). Never G2P.
Asks: does Gemini honor IPA, and is that teacher better than Cloud for
finetuning (wavlm F1 vs human clip).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
import time
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

import requests  # noqa: E402

from dose_r.references.reference_clips import available_clips  # noqa: E402
from dose_r.references.tts_pronunciation import (  # noqa: E402
    custom_pronunciation,
    spoken_text,
)
from eval_gemini31_holdout import (  # noqa: E402
    MODEL,
    RATE,
    VOICE,
    cloud_ipa_path,
    f1,
    mean,
    slug,
    token,
)

ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "project-amer-scs-sandbox")
VERIFY = ROOT / "runs" / "verify-google-ipa" / "results.json"
OUT = ROOT / "runs" / "gemini31-ipa-vs-cloud"
TTS = OUT / "tts"
LISTEN = ROOT / "runs" / "listen-gemini31-ipa-vs-cloud"
WORKERS = int(os.environ.get("TTS_WORKERS", "4"))
GAP = 0.03
PROMPT_PLAIN = "Pronounce this US drug name clearly as a single name."
PROMPT_IPA = (
    "Pronounce this US drug name using the given IPA exactly. "
    "Do not spell letters. IPA: {ipa}"
)

_EMB_NOTE = 0  # keep import path stable; f1 uses holdout cache


def synth_gemini(
    text: str,
    prompt: str,
    pronunciations: dict | None = None,
) -> bytes:
    inp: dict = {"text": text, "prompt": prompt}
    if pronunciations:
        inp["customPronunciations"] = pronunciations
    body = {
        "input": inp,
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
            timeout=90,
        )
        if resp.status_code == 429:
            time.sleep(min(2**attempt, 20))
            last = RuntimeError(f"429 {resp.text[:200]}")
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"{resp.status_code}: {resp.text[:400]}")
        return base64.b64decode(resp.json()["audioContent"])
    raise last or RuntimeError("gemini synth failed")


def cached(path: Path, fn, *args) -> bytes:
    if path.exists() and path.stat().st_size > 500:
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = fn(*args)
    path.write_bytes(data)
    return data


def classify(g_h: float | None, c_h: float | None, g_c: float | None) -> str:
    if g_h is None and c_h is None:
        return "no_human"
    if g_h is None:
        return "no_gemini"
    if c_h is None:
        return "no_cloud"
    if g_h >= c_h + GAP:
        return "gemini_ipa_better"
    if c_h >= g_h + GAP:
        return "cloud_ipa_better"
    return "tie"


def write_listen(rows: list[dict], clips: dict) -> Path:
    LISTEN.mkdir(parents=True, exist_ok=True)
    ranked = sorted(
        rows,
        key=lambda r: (
            0 if r["bucket"] == "gemini_ipa_better" else 1 if r["bucket"] == "cloud_ipa_better" else 2,
            -(abs((r.get("gemini_ipa_vs_human") or 0) - (r.get("cloud_ipa_vs_human") or 0))),
        ),
    )
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Gemini 3.1 + IPA vs Cloud + IPA</title>",
        "<style>body{font:16px/1.4 system-ui;max-width:820px;margin:2rem auto;padding:0 1rem}",
        "section{border:1px solid #ccc;border-radius:8px;padding:1rem 1.2rem;margin:1rem 0}",
        "h1{font-size:1.25rem} h2{font-size:1.05rem;margin:0 0 .35rem}",
        "p{margin:.2rem 0 .5rem} .ipa{font-family:ui-monospace,monospace}",
        "label{display:block;font-weight:600;margin:.45rem 0 .1rem} audio{width:100%}",
        ".meta{font-size:.9rem;color:#555} .tag{display:inline-block;background:#eee;",
        "padding:.1rem .45rem;border-radius:4px;margin-right:.3rem;font-size:.85rem}</style>",
        "<h1>Gemini 3.1 Flash TTS + IPA vs Cloud Standard-C + IPA</h1>",
        "<p class='meta'>Same source IPA Cloud already accepted. Kore vs Standard-C.</p>",
    ]
    for rec in ranked:
        s = slug(rec["ingredient"])
        parts.append(f"<section id='{s}'><h2>{rec['ingredient']}</h2>")
        parts.append(f"<span class='tag'>{rec['bucket']}</span>")
        parts.append(
            f"<p>IPA <span class='ipa'>{rec.get('ipa_used') or rec.get('ipa')}</span></p>"
        )
        parts.append(
            "<p class='meta'>"
            f"Gemini+IPA vs human <b>{rec.get('gemini_ipa_vs_human')}</b> · "
            f"Cloud+IPA vs human <b>{rec.get('cloud_ipa_vs_human')}</b> · "
            f"Gemini+IPA vs Cloud+IPA <b>{rec.get('gemini_ipa_vs_cloud_ipa')}</b> · "
            f"Gemini+IPA vs Gemini plain <b>{rec.get('gemini_ipa_vs_plain')}</b></p>"
        )
        clip = clips.get(rec["ingredient"])
        if clip is not None:
            dest = LISTEN / f"{s}__human{clip.path.suffix}"
            if not dest.exists():
                dest.write_bytes(clip.path.read_bytes())
            parts.append(
                f"<label>Human ({clip.source})</label>"
                f"<audio controls src='{dest.name}'></audio>"
            )
        ipa_src = cloud_ipa_path(rec["ingredient"])
        if ipa_src is not None:
            dest = LISTEN / f"{s}__cloud_ipa.wav"
            shutil.copy2(ipa_src, dest)
            parts.append(
                "<label>Cloud Standard-C + IPA</label>"
                f"<audio controls src='{dest.name}'></audio>"
            )
        for kind, label in (
            ("sidecar", "Gemini 3.1 + customPronunciations IPA"),
            ("prompt", "Gemini 3.1 + IPA in prompt"),
            ("plain", "Gemini 3.1 plain spelling"),
        ):
            src = TTS / f"{s}.{kind}.wav"
            if src.exists():
                dest = LISTEN / f"{s}__{kind}.wav"
                shutil.copy2(src, dest)
                parts.append(f"<label>{label}</label><audio controls src='{dest.name}'></audio>")
        parts.append("</section>")
    dest = LISTEN / "index.html"
    dest.write_text("\n".join(parts) + "\n")
    return dest


def load_items(only: list[str] | None) -> list[dict]:
    rows = json.loads(VERIFY.read_text())["rows"]
    items = []
    for rec in rows:
        name = rec["ingredient"]
        ipa = rec.get("ipa_used") or ""
        if not ipa:
            continue
        if cloud_ipa_path(name) is None:
            continue
        items.append(rec)
    if only:
        want = {n.lower() for n in only}
        items = [r for r in items if r["ingredient"].lower() in want]
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    items = load_items(args.only)
    clips = available_clips()
    TTS.mkdir(parents=True, exist_ok=True)
    print(f"{len(items)} names with Cloud IPA wav + ipa_used", flush=True)

    jobs = []
    for rec in items:
        name = rec["ingredient"]
        ipa = rec["ipa_used"]
        text = spoken_text(name)
        phrase = text if " " not in text else name
        s = slug(name)
        jobs.append((name, ipa, text, phrase, s))

    def one(name, ipa, text, phrase, s):
        out = {"sidecar_ok": False, "prompt_ok": False, "plain_ok": False}
        try:
            cached(
                TTS / f"{s}.sidecar.wav",
                synth_gemini,
                text,
                PROMPT_IPA.format(ipa=ipa),
                custom_pronunciation(phrase if phrase in text else text.split()[0], ipa),
            )
            out["sidecar_ok"] = True
        except Exception as exc:
            out["sidecar_err"] = str(exc)[:240]
        try:
            cached(
                TTS / f"{s}.prompt.wav",
                synth_gemini,
                text,
                PROMPT_IPA.format(ipa=ipa),
                None,
            )
            out["prompt_ok"] = True
        except Exception as exc:
            out["prompt_err"] = str(exc)[:240]
        try:
            cached(TTS / f"{s}.plain.wav", synth_gemini, text, PROMPT_PLAIN, None)
            out["plain_ok"] = True
        except Exception as exc:
            out["plain_err"] = str(exc)[:240]
        return out

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {
            ex.submit(one, name, ipa, text, phrase, s): name
            for name, ipa, text, phrase, s in jobs
        }
        done = 0
        for fut in as_completed(futs):
            name = futs[fut]
            done += 1
            try:
                info = fut.result()
                print(f"  {done}/{len(jobs)} {name} {info}", flush=True)
            except Exception as exc:
                print(f"  FAIL {name}: {exc}", flush=True)

    rows = []
    for rec in items:
        name = rec["ingredient"]
        s = slug(name)
        clip = clips.get(name)
        cloud = cloud_ipa_path(name)
        sidecar = TTS / f"{s}.sidecar.wav"
        promptp = TTS / f"{s}.prompt.wav"
        plain = TTS / f"{s}.plain.wav"
        gemini_ipa = sidecar if sidecar.exists() and sidecar.stat().st_size > 500 else None
        if gemini_ipa is None and promptp.exists() and promptp.stat().st_size > 500:
            gemini_ipa = promptp
        g_h = c_h = g_c = g_p = None
        if gemini_ipa is not None and clip is not None:
            g_h = round(f1(gemini_ipa, clip.path), 4)
        if cloud is not None and clip is not None:
            c_h = round(f1(cloud, clip.path), 4)
        if gemini_ipa is not None and cloud is not None:
            g_c = round(f1(gemini_ipa, cloud), 4)
        if gemini_ipa is not None and plain.exists() and plain.stat().st_size > 500:
            g_p = round(f1(gemini_ipa, plain), 4)
        sidecar_vs_prompt = None
        if (
            sidecar.exists()
            and promptp.exists()
            and sidecar.stat().st_size > 500
            and promptp.stat().st_size > 500
        ):
            sidecar_vs_prompt = round(f1(sidecar, promptp), 4)
        row = {
            "ingredient": name,
            "ipa": rec.get("ipa"),
            "ipa_used": rec.get("ipa_used"),
            "clip_source": clip.source if clip is not None else None,
            "gemini_ipa_path": gemini_ipa.name if gemini_ipa is not None else None,
            "gemini_ipa_vs_human": g_h,
            "cloud_ipa_vs_human": c_h,
            "gemini_ipa_vs_cloud_ipa": g_c,
            "gemini_ipa_vs_plain": g_p,
            "sidecar_vs_prompt": sidecar_vs_prompt,
            "bucket": classify(g_h, c_h, g_c),
        }
        rows.append(row)
        print(
            f"  {name} {row['bucket']} G-H={g_h} C-H={c_h} G-C={g_c} G-plain={g_p}",
            flush=True,
        )

    scored = [
        r
        for r in rows
        if r.get("gemini_ipa_vs_human") is not None
        and r.get("cloud_ipa_vs_human") is not None
    ]
    buckets = {}
    for r in rows:
        buckets[r["bucket"]] = buckets.get(r["bucket"], 0) + 1
    honor = [
        r["gemini_ipa_vs_plain"]
        for r in rows
        if r.get("gemini_ipa_vs_plain") is not None
    ]
    summary = {
        "n": len(rows),
        "n_vs_human": len(scored),
        "model": MODEL,
        "voice": VOICE,
        "cloud_voice": "en-US-Standard-C",
        "buckets": buckets,
        "mean_gemini_ipa_vs_human": mean([r["gemini_ipa_vs_human"] for r in scored]),
        "mean_cloud_ipa_vs_human": mean([r["cloud_ipa_vs_human"] for r in scored]),
        "mean_gemini_ipa_vs_cloud_ipa": mean(
            [r["gemini_ipa_vs_cloud_ipa"] for r in rows if r.get("gemini_ipa_vs_cloud_ipa") is not None]
        ),
        "mean_gemini_ipa_vs_plain": mean(honor),
        "mean_sidecar_vs_prompt": mean(
            [r["sidecar_vs_prompt"] for r in rows if r.get("sidecar_vs_prompt") is not None]
        ),
        "gemini_beats_cloud": sum(1 for r in scored if r["bucket"] == "gemini_ipa_better"),
        "cloud_beats_gemini": sum(1 for r in scored if r["bucket"] == "cloud_ipa_better"),
        "ties": sum(1 for r in scored if r["bucket"] == "tie"),
    }
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
