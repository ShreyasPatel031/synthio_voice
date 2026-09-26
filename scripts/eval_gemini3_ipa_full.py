#!/usr/bin/env python
"""Full-set name + sidecar from runs/gemini3-ipa-query/ipa.json.

Source IPA (not G2P). Cloud en-US-Standard-C. Arms:

  plain   spelling only
  ipa     PHONETIC_ENCODING_IPA sidecar
  xsampa  PHONETIC_ENCODING_X_SAMPA rewrite of the same IPA
          https://docs.cloud.google.com/text-to-speech/docs/reference/rest/Shared.Types/PhoneticEncoding

Path 2 F1 vs human clip where a clip exists. All 284 names are synthesized.
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from dose_r.references.tts_pronunciation import (  # noqa: E402
    custom_pronunciation,
    ipa_to_xsampa,
)

ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"
VOICE = "en-US-Standard-C"
RATE = 24000
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "project-amer-scs-sandbox")
IPA_IN = ROOT / "runs" / "gemini3-ipa-query" / "ipa.json"
OUT = ROOT / "runs" / "gemini3-ipa-tts"
TTS = OUT / "tts"
WORKERS = int(os.environ.get("TTS_WORKERS", "10"))

_lock = threading.Lock()
_token = {"value": None, "exp": 0.0}


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


def strip_ipa(raw: str) -> str:
    s = (raw or "").strip()
    if s.lower() in {"", "n/a", "none", "null", "unknown"}:
        return ""
    s = s.strip("/[]() ")
    s = s.replace("'", "ˈ").replace("ˈˈ", "ˈ")
    return s


def looks_like_ipa(s: str) -> bool:
    if not s:
        return False
    if re.search(r"[A-Za-z]{3,}.*['\"]", s) and not re.search(
        r"[ɪʊɛæɑɒɔʌəɚɝʃʒθðŋɡːˈˌ]", s
    ):
        return False
    if re.search(r"[ɪʊɛæɑɒɔʌəɚɝʃʒθðŋɡːˈˌ]", s):
        return True
    # ASCII-only IPA still has vowels; reject hyphenated respelling.
    if "-" in s or " " in s:
        return False
    return bool(re.search(r"[aeiouyæ]", s, re.I))


def cloud_ipa(ipa: str) -> str:
    return strip_ipa(ipa)


def synth(
    text: str,
    pronunciations: dict | None = None,
) -> bytes:
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
            "Authorization": f"Bearer {token()}",
            "Content-Type": "application/json",
            "x-goog-user-project": PROJECT,
        },
        json=body,
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"{resp.status_code}: {resp.text[:400]}")
    return base64.b64decode(resp.json()["audioContent"])


def cached_write(path: Path, fn) -> bytes:
    if path.exists() and path.stat().st_size > 500:
        return path.read_bytes()
    data = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def try_ipa(name: str, raw: str) -> tuple[bytes, str]:
    base = cloud_ipa(raw)
    cands = [base]
    nodot = base.replace(".", "")
    if nodot not in cands:
        cands.append(nodot)
    last = None
    for cand in cands:
        if not cand:
            continue
        try:
            wav = synth(
                name,
                custom_pronunciation(name, cand, "PHONETIC_ENCODING_IPA"),
            )
            return wav, cand
        except RuntimeError as exc:
            last = exc
    raise last or RuntimeError("no IPA candidate")


def try_xsampa(name: str, raw: str) -> tuple[bytes, str]:
    ipa = cloud_ipa(raw).replace(".", "")
    xs = ipa_to_xsampa(ipa)
    cands = [xs]
    spaced = xs.replace('"', '" ').replace("%", "% ")
    if spaced not in cands:
        cands.append(spaced)
    last = None
    for cand in cands:
        if not cand:
            continue
        try:
            wav = synth(
                name,
                custom_pronunciation(name, cand, "PHONETIC_ENCODING_X_SAMPA"),
            )
            return wav, cand
        except RuntimeError as exc:
            last = exc
    raise last or RuntimeError("no X-SAMPA candidate")


def f1(a: bytes, b: bytes) -> float:
    from dose_r.scoring.speech_similarity import (
        extract_frame_embeddings,
        speech_bertscore,
    )

    return float(
        speech_bertscore(extract_frame_embeddings(a), extract_frame_embeddings(b))["f1"]
    )


def mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def write_listen(rows: list[dict], clips: dict) -> None:
    scored = [r for r in rows if r.get("plain") is not None]
    worst = sorted(
        [r for r in scored if r.get("ipa_f1") is not None],
        key=lambda r: (r["ipa_f1"], r.get("delta_ipa") or 0),
    )[:25]
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Gemini IPA vs human — worst 25</title>",
        "<style>body{font:15px/1.4 system-ui;max-width:820px;margin:2rem auto;padding:0 1rem}",
        "section{border:1px solid #ccc;border-radius:8px;padding:1rem;margin:1rem 0}",
        ".ipa{font-family:ui-monospace,monospace} audio{width:100%}",
        "label{display:block;margin:.4rem 0 .1rem;font-weight:600}</style>",
        "<h1>Name + sidecar IPA vs human. Lowest IPA F1 first.</h1>",
        f"<p>Voice {VOICE}. Showing 25 furthest from the human clip.</p>",
    ]
    for rec in worst:
        slug = rec["slug"]
        clip = clips[rec["ingredient"]]
        dest_h = OUT / f"{slug}__human{clip.path.suffix}"
        if not dest_h.exists():
            dest_h.write_bytes(clip.path.read_bytes())
        for arm in ("plain", "ipa", "xsampa"):
            src = TTS / f"{slug}.{arm}.wav"
            dest = OUT / f"{slug}__{arm}.wav"
            if src.exists() and not dest.exists():
                dest.write_bytes(src.read_bytes())
        d_i = rec.get("delta_ipa")
        d_x = rec.get("delta_xsampa")
        parts.append(f"<section id='{slug}'><h2>{rec['ingredient']}</h2>")
        parts.append(
            f"<p class='ipa'>source {rec['ipa_raw']}<br>"
            f"IPA used {rec.get('ipa_used')}<br>"
            f"X-SAMPA used {rec.get('xsampa_used')}</p>"
        )
        parts.append(
            f"<p>plain {rec['plain']:.3f} &nbsp; "
            f"IPA {rec.get('ipa_f1') if rec.get('ipa_f1') is None else f'{rec[\"ipa_f1\"]:.3f}'} "
            f"({'' if d_i is None else f'{d_i:+.3f}'}) &nbsp; "
            f"X-SAMPA {rec.get('xsampa_f1') if rec.get('xsampa_f1') is None else f'{rec[\"xsampa_f1\"]:.3f}'} "
            f"({'' if d_x is None else f'{d_x:+.3f}'})</p>"
        )
        parts.append(f"<label>Human ({clip.source})</label>")
        parts.append(f"<audio controls src='{dest_h.name}'></audio>")
        parts.append("<label>Plain</label>")
        parts.append(f"<audio controls src='{slug}__plain.wav'></audio>")
        if rec.get("ipa_ok"):
            parts.append("<label>IPA sidecar</label>")
            parts.append(f"<audio controls src='{slug}__ipa.wav'></audio>")
        if rec.get("xsampa_ok"):
            parts.append("<label>X-SAMPA sidecar</label>")
            parts.append(f"<audio controls src='{slug}__xsampa.wav'></audio>")
        parts.append("</section>")
    (OUT / "index.html").write_text("\n".join(parts))


def main() -> int:
    payload = json.loads(IPA_IN.read_text())
    clips = available_clips()
    OUT.mkdir(parents=True, exist_ok=True)
    TTS.mkdir(parents=True, exist_ok=True)

    rows = []
    for rec in payload:
        name = rec["ingredient"]
        raw = rec.get("ipa") or ""
        usable = looks_like_ipa(raw)
        rows.append(
            {
                "ingredient": name,
                "slug": name.lower().replace(" ", "_"),
                "ipa_raw": raw,
                "url": rec.get("url") or "",
                "usable_ipa": usable,
                "has_clip": name in clips,
            }
        )

    print(
        f"n={len(rows)} usable_ipa={sum(1 for r in rows if r['usable_ipa'])} "
        f"clips={sum(1 for r in rows if r['has_clip'])} voice={VOICE}",
        flush=True,
    )

    t0 = time.time()
    done = 0

    def one(rec: dict) -> dict:
        name = rec["ingredient"]
        slug = rec["slug"]
        rec["plain_ok"] = False
        rec["ipa_ok"] = False
        rec["xsampa_ok"] = False
        try:
            cached_write(TTS / f"{slug}.plain.wav", lambda: synth(name))
            rec["plain_ok"] = True
        except Exception as exc:
            rec["plain_error"] = str(exc)[:240]
        if rec["usable_ipa"]:
            try:
                wav, used = try_ipa(name, rec["ipa_raw"])
                (TTS / f"{slug}.ipa.wav").write_bytes(wav)
                rec["ipa_used"] = used
                rec["ipa_ok"] = True
            except Exception as exc:
                rec["ipa_error"] = str(exc)[:240]
            try:
                wav, used = try_xsampa(name, rec["ipa_raw"])
                (TTS / f"{slug}.xsampa.wav").write_bytes(wav)
                rec["xsampa_used"] = used
                rec["xsampa_ok"] = True
            except Exception as exc:
                rec["xsampa_error"] = str(exc)[:240]
        else:
            rec["ipa_error"] = "not IPA (n/a or respelling)"
            rec["xsampa_error"] = "not IPA (n/a or respelling)"
        return rec

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(one, r): r["ingredient"] for r in rows}
        for fut in as_completed(futs):
            rec = fut.result()
            done += 1
            if done % 20 == 0 or done == len(rows):
                print(
                    f"synth [{done}/{len(rows)} +{time.time()-t0:.0f}s] "
                    f"{rec['ingredient']} ipa={rec['ipa_ok']} xs={rec['xsampa_ok']}",
                    flush=True,
                )

    scored = [r for r in rows if r["has_clip"] and r["plain_ok"]]
    print(f"scoring wavlm F1 on {len(scored)} clips...", flush=True)
    t1 = time.time()
    for i, rec in enumerate(scored, 1):
        human = clips[rec["ingredient"]].path.read_bytes()
        plain = (TTS / f"{rec['slug']}.plain.wav").read_bytes()
        rec["plain"] = round(f1(plain, human), 4)
        rec["clip_source"] = clips[rec["ingredient"]].source
        if rec["ipa_ok"]:
            rec["ipa_f1"] = round(
                f1((TTS / f"{rec['slug']}.ipa.wav").read_bytes(), human), 4
            )
            rec["delta_ipa"] = round(rec["ipa_f1"] - rec["plain"], 4)
        if rec["xsampa_ok"]:
            rec["xsampa_f1"] = round(
                f1((TTS / f"{rec['slug']}.xsampa.wav").read_bytes(), human), 4
            )
            rec["delta_xsampa"] = round(rec["xsampa_f1"] - rec["plain"], 4)
        if i % 20 == 0 or i == len(scored):
            print(f"  scored {i}/{len(scored)} +{time.time()-t1:.0f}s", flush=True)

    ipa_rows = [r for r in scored if r.get("ipa_f1") is not None]
    xs_rows = [r for r in scored if r.get("xsampa_f1") is not None]
    summary = {
        "voice": VOICE,
        "n_names": len(rows),
        "n_usable_ipa": sum(1 for r in rows if r["usable_ipa"]),
        "n_ipa_ok": sum(1 for r in rows if r["ipa_ok"]),
        "n_xsampa_ok": sum(1 for r in rows if r["xsampa_ok"]),
        "n_scored": len(scored),
        "n_scored_ipa": len(ipa_rows),
        "n_scored_xsampa": len(xs_rows),
        "mean_plain": mean([r["plain"] for r in scored]),
        "mean_ipa": mean([r["ipa_f1"] for r in ipa_rows]),
        "mean_xsampa": mean([r["xsampa_f1"] for r in xs_rows]),
        "ipa_beats_plain": sum(1 for r in ipa_rows if (r.get("delta_ipa") or 0) > 0.01),
        "ipa_loses_plain": sum(1 for r in ipa_rows if (r.get("delta_ipa") or 0) < -0.01),
        "xsampa_beats_plain": sum(
            1 for r in xs_rows if (r.get("delta_xsampa") or 0) > 0.01
        ),
        "xsampa_loses_plain": sum(
            1 for r in xs_rows if (r.get("delta_xsampa") or 0) < -0.01
        ),
        "worst_ipa_vs_human": sorted(
            ipa_rows, key=lambda r: r["ipa_f1"]
        )[:15],
        "worst_ipa_vs_plain": sorted(
            ipa_rows, key=lambda r: r.get("delta_ipa") or 0
        )[:15],
        "best_ipa_vs_plain": sorted(
            ipa_rows, key=lambda r: r.get("delta_ipa") or 0, reverse=True
        )[:10],
    }
    slim_keys = (
        "ingredient",
        "ipa_raw",
        "ipa_used",
        "plain",
        "ipa_f1",
        "delta_ipa",
        "xsampa_f1",
        "delta_xsampa",
        "ipa_error",
    )
    for key in ("worst_ipa_vs_human", "worst_ipa_vs_plain", "best_ipa_vs_plain"):
        summary[key] = [{k: r.get(k) for k in slim_keys} for r in summary[key]]

    (OUT / "results.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False)
        + "\n"
    )
    write_listen(rows, clips)
    print("SUMMARY", json.dumps({k: summary[k] for k in summary if not k.startswith("worst") and not k.startswith("best")}), flush=True)
    print("listen", OUT / "index.html", flush=True)
    print("\nFURTHEST FROM HUMAN (IPA F1)", flush=True)
    for r in summary["worst_ipa_vs_human"]:
        print(
            f"  {r['ipa_f1']:.3f}  plain {r['plain']:.3f}  {r['ingredient']:28s}  {r['ipa_raw']}",
            flush=True,
        )
    print("\nBIGGEST IPA LOSS VS PLAIN", flush=True)
    for r in summary["worst_ipa_vs_plain"]:
        print(
            f"  {r['delta_ipa']:+.3f}  ipa {r['ipa_f1']:.3f}  {r['ingredient']:28s}  {r['ipa_raw']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
