#!/usr/bin/env python
"""Verify Gemini Google-query IPA with Cloud TTS vs human clips.

Cheaper/faster than Gemini TTS: Cloud en-US-Standard-C (same Path 2 voice).
Also flags brand/generic mixups (Dupixent IPA that is actually dupilumab).
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
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_hf = ROOT / ".cache" / "huggingface"
_hf.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(_hf)
os.environ["TRANSFORMERS_CACHE"] = str(_hf)
os.environ["HF_HUB_CACHE"] = str(_hf / "hub")

import requests

from dose_r.references.reference_clips import available_clips
from dose_r.references.tts_pronunciation import custom_pronunciation, is_source_ipa

ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"
VOICE = "en-US-Standard-C"
RATE = 24000
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "project-amer-scs-sandbox")
IPA_IN = ROOT / "runs" / "gemini3-ipa-query" / "ipa.json"
PRON = ROOT / "dose_r" / "references" / "pronunciations.jsonl"
MANIFEST = ROOT / "data" / "reference_audio" / "manifest.jsonl"
OUT = ROOT / "runs" / "verify-google-ipa"
TTS = OUT / "tts"
WORKERS = int(os.environ.get("TTS_WORKERS", "12"))

_lock = threading.Lock()
_token = {"value": None, "exp": 0.0}

# Extra brand → generic when ClinCalc didn't list them.
KNOWN_GENERIC = {
    "Dupixent": "dupilumab",
    "Humira": "adalimumab",
    "Nexium": "esomeprazole",
    "Wegovy": "semaglutide",
    "Ozempic": "semaglutide",
    "Mounjaro": "tirzepatide",
    "Skyrizi": "risankizumab",
    "Jardiance": "empagliflozin",
    "Eliquis": "apixaban",
    "Xeljanz": "tofacitinib",
    "Rinvoq": "upadacitinib",
    "Cosentyx": "secukinumab",
    "Enbrel": "etanercept",
    "Keytruda": "pembrolizumab",
    "Imfinzi": "durvalumab",
    "Vabysmo": "faricimab",
}


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


def load_ipa() -> dict[str, dict]:
    rows = json.loads(IPA_IN.read_text())
    return {r["ingredient"]: r for r in rows}


def load_types() -> dict[str, str]:
    out = {}
    for line in PRON.read_text().splitlines():
        rec = json.loads(line)
        out[rec["ingredient"]] = rec.get("name_type") or ""
    return out


def clincalc_pairs() -> dict[str, str]:
    rx = re.compile(
        r"separate generic-name clip on this page: '([^']+)'", re.I
    )
    out = dict(KNOWN_GENERIC)
    for line in MANIFEST.read_text().splitlines():
        rec = json.loads(line)
        m = rx.search(rec.get("respelling") or "")
        if not m:
            continue
        other = m.group(1).split(";")[0].strip()
        if other and other.lower() != rec["ingredient"].lower():
            out.setdefault(rec["ingredient"], other)
    return out


def strip_ipa(raw: str) -> str:
    s = (raw or "").strip()
    if s.lower() in {"", "n/a", "none", "null", "unknown"}:
        return ""
    s = s.strip("/[]() ")
    s = s.replace("'", "ˈ").replace("ˈˈ", "ˈ")
    s = s.replace(".", "")
    s = re.sub(r"\s+", "", s)
    return s


def looks_like_ipa(s: str) -> bool:
    return is_source_ipa(s)


def fold(s: str) -> str:
    s = strip_ipa(s).lower()
    s = s.replace("ɹ", "r").replace("ɡ", "g")
    for ch in "ˈˌː. /-[]()":
        s = s.replace(ch, "")
    return s


def edit(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(
                min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            )
        prev = cur
    return prev[-1]


def unwrap_ipa(ipa: str) -> str:
    s = (ipa or "").strip()
    if s.lower() in {"", "n/a", "none", "null", "unknown"}:
        return ""
    s = s.strip("/[]() ")
    s = s.replace("'", "ˈ").replace("ˈˈ", "ˈ")
    return s


def cloud_ipa(ipa: str) -> str:
    """Source IPA for Cloud. No phone rewrites."""
    return unwrap_ipa(ipa)


def try_ipa_synth(tok: str, name: str, ipa: str) -> tuple[bytes, str]:
    exact = unwrap_ipa(ipa)
    if not exact:
        raise RuntimeError("no IPA candidate")
    # Dots/spaces are syllable marks, not phones. Try the published string
    # first; only drop marks if Cloud 400s. Never rewrite ɪə/ə/ɹ.
    candidates = [exact]
    compact = exact.replace(".", "").replace(" ", "")
    if compact and compact not in candidates:
        candidates.append(compact)
    last = None
    for cand in candidates:
        try:
            wav = synth(
                tok, text=name, pronunciations=custom_pronunciation(name, cand)
            )
            return wav, cand
        except RuntimeError as exc:
            last = exc
    raise last or RuntimeError("no IPA candidate")


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
            "Authorization": f"Bearer {tok() if callable(tok) else tok}",
            "Content-Type": "application/json",
            "x-goog-user-project": PROJECT,
        },
        json=body,
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"{resp.status_code}: {resp.text[:400]}")
    return base64.b64decode(resp.json()["audioContent"])


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def url_other_names(name: str, url: str, names: list[str]) -> list[str]:
    if not url:
        return []
    blob = slug(urlparse(url).path + " " + urlparse(url).netloc)
    self = slug(name)
    hits = []
    for other in names:
        o = slug(other)
        if len(o) < 6 or o == self:
            continue
        if o in blob and o not in self:
            hits.append(other)
    return hits


def mixup_flags(name: str, rec: dict, ipa_map: dict, types: dict, pairs: dict) -> list[str]:
    flags = []
    ipa = rec.get("ipa") or ""
    url = rec.get("url") or ""
    cleaned = strip_ipa(ipa)
    if not cleaned:
        flags.append("no_ipa")
        return flags
    if not looks_like_ipa(ipa):
        flags.append("not_ipa_glyphs")
    others = url_other_names(name, url, list(ipa_map))
    if others:
        flags.append("url_other:" + ",".join(others[:3]))
    gen = pairs.get(name)
    if gen:
        g_ipa = ""
        for key in ipa_map:
            if key.lower() == gen.lower():
                g_ipa = ipa_map[key].get("ipa") or ""
                break
        if g_ipa:
            fa, fb = fold(ipa), fold(g_ipa)
            if fa and fb:
                d = edit(fa, fb)
                n = max(len(fa), len(fb))
                if n and d / n <= 0.35:
                    flags.append(f"ipa_matches_generic:{gen}")
        if slug(gen) and slug(gen) in slug(ipa + url):
            if f"ipa_matches_generic:{gen}" not in flags:
                flags.append(f"citation_generic:{gen}")
    if "howtopronounce.com" in url:
        flags.append("howtopronounce")
    if types.get(name) == "brand" and re.search(r"mæb|juːmæb|umab", cleaned):
        if "mab" not in name.lower() and "umab" not in name.lower():
            flags.append("mab_ipa_on_non_mab_brand")
    return flags


def f1(a: bytes, b: bytes) -> float:
    from dose_r.scoring.speech_similarity import (
        extract_frame_embeddings,
        speech_bertscore,
    )

    return float(
        speech_bertscore(extract_frame_embeddings(a), extract_frame_embeddings(b))["f1"]
    )


def write_listen(rows: list[dict]) -> None:
    scored = [r for r in rows if r.get("plain") is not None]
    scored.sort(key=lambda r: (0 if r.get("flags") else 1, r.get("ipa_f1") or 9))
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Google IPA TTS vs human</title>",
        "<style>body{font:15px/1.4 system-ui;max-width:820px;margin:2rem auto;padding:0 1rem}",
        "section{border:1px solid #ccc;border-radius:8px;padding:1rem;margin:1rem 0}",
        ".ipa{font-family:ui-monospace,monospace} .bad{color:#a00;font-weight:700}",
        "audio{width:100%} label{display:block;margin:.4rem 0 .1rem;font-weight:600}</style>",
        "<h1>Google-query IPA injected into Cloud TTS vs human</h1>",
        f"<p>Voice {VOICE}. Scored {len(scored)} / {len(rows)} names with a clip. "
        "Flagged rows first.</p>",
    ]
    for rec in scored:
        slug_n = rec["ingredient"].lower().replace(" ", "_")
        flags = rec.get("flags") or []
        cls = "bad" if flags else ""
        parts.append(f"<section id='{slug_n}'><h2 class='{cls}'>{rec['ingredient']}</h2>")
        parts.append(
            f"<p class='ipa'>IPA {rec.get('ipa')}<br>used {rec.get('ipa_used')}"
            f"<br>{rec.get('url')}</p>"
        )
        if flags:
            parts.append(f"<p class='bad'>FLAGS: {', '.join(flags)}</p>")
        if rec.get("plain") is not None:
            ipa_s = rec.get("ipa_f1")
            if ipa_s is None:
                parts.append(
                    f"<p>plain {rec['plain']:.3f} &nbsp; IPA synth failed "
                    f"({rec.get('ipa_error','')})</p>"
                )
            else:
                d = rec.get("delta") or 0
                parts.append(
                    f"<p>plain {rec['plain']:.3f} &nbsp; IPA {ipa_s:.3f} "
                    f"&nbsp; Δ {d:+.3f}</p>"
                )
        parts.append(f"<label>Human ({rec.get('clip_source')})</label>")
        parts.append(f"<audio controls src='{slug_n}__human{rec['human_ext']}'></audio>")
        parts.append("<label>Plain</label>")
        parts.append(f"<audio controls src='{slug_n}__plain.wav'></audio>")
        if rec.get("has_ipa_wav"):
            parts.append("<label>Google IPA sidecar</label>")
            parts.append(f"<audio controls src='{slug_n}__ipa.wav'></audio>")
        parts.append("</section>")
    (OUT / "index.html").write_text("\n".join(parts))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    TTS.mkdir(parents=True, exist_ok=True)
    ipa_map = load_ipa()
    types = load_types()
    pairs = clincalc_pairs()
    clips = available_clips()
    names = list(ipa_map)

    rows = []
    for name, rec in ipa_map.items():
        flags = mixup_flags(name, rec, ipa_map, types, pairs)
        rows.append(
            {
                "ingredient": name,
                "name_type": types.get(name, ""),
                "ipa": rec.get("ipa") or "",
                "url": rec.get("url") or "",
                "generic_pair": pairs.get(name, ""),
                "flags": flags,
                "has_clip": name in clips,
            }
        )

    with_clip = [r for r in rows if r["has_clip"]]
    print(
        f"names={len(rows)} clips={len(with_clip)} "
        f"flagged={sum(1 for r in rows if r['flags'])} voice={VOICE}",
        flush=True,
    )

    tok = token()
    t0 = time.time()

    def one(rec: dict) -> dict:
        name = rec["ingredient"]
        clip = clips[name]
        slug_n = name.lower().replace(" ", "_")
        human = clip.path.read_bytes()
        dest_h = OUT / f"{slug_n}__human{clip.path.suffix}"
        dest_h.write_bytes(human)
        rec["clip_source"] = clip.source
        rec["human_ext"] = clip.path.suffix
        p_plain = TTS / f"{slug_n}.plain.wav"
        p_ipa = TTS / f"{slug_n}.ipa.wav"
        try:
            if p_plain.exists() and p_plain.stat().st_size > 500:
                plain = p_plain.read_bytes()
            else:
                plain = synth(tok, text=name)
                p_plain.write_bytes(plain)
            (OUT / f"{slug_n}__plain.wav").write_bytes(plain)
            rec["plain_ok"] = True
        except Exception as exc:
            rec["plain_ok"] = False
            rec["plain_error"] = str(exc)[:200]
            return rec
        ipa = strip_ipa(rec["ipa"])
        if not ipa or not looks_like_ipa(rec["ipa"]):
            rec["ipa_synth"] = False
            rec["has_ipa_wav"] = False
            rec["ipa_error"] = "skipped: not usable IPA"
            rec["_plain_bytes"] = plain
            rec["_human_bytes"] = human
            return rec
        try:
            wav, used = try_ipa_synth(tok, name, rec["ipa"])
            p_ipa.write_bytes(wav)
            (OUT / f"{slug_n}__ipa.wav").write_bytes(wav)
            rec["ipa_used"] = used
            rec["ipa_synth"] = True
            rec["has_ipa_wav"] = True
            rec["_ipa_bytes"] = wav
        except Exception as exc:
            rec["ipa_synth"] = False
            rec["has_ipa_wav"] = False
            rec["ipa_error"] = str(exc)[:240]
        rec["_plain_bytes"] = plain
        rec["_human_bytes"] = human
        return rec

    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(one, r): r["ingredient"] for r in with_clip}
        for fut in as_completed(futs):
            rec = fut.result()
            done += 1
            print(
                f"synth [{done}/{len(with_clip)} +{time.time()-t0:.0f}s] "
                f"{rec['ingredient']} ipa={rec.get('ipa_synth')}",
                flush=True,
            )

    print("scoring wavlm F1...", flush=True)
    t1 = time.time()
    scored = 0
    for rec in with_clip:
        human = rec.pop("_human_bytes", None)
        plain = rec.pop("_plain_bytes", None)
        ipa_wav = rec.pop("_ipa_bytes", None)
        if human is None or plain is None:
            continue
        rec["plain"] = round(f1(plain, human), 4)
        if ipa_wav is not None:
            rec["ipa_f1"] = round(f1(ipa_wav, human), 4)
            rec["delta"] = round(rec["ipa_f1"] - rec["plain"], 4)
            if rec["delta"] <= -0.03:
                rec["flags"] = list(rec.get("flags") or []) + ["ipa_worse_than_plain"]
        else:
            rec["ipa_f1"] = None
            rec["delta"] = None
        scored += 1
        if scored % 20 == 0:
            print(f"  scored {scored}/{len(with_clip)} +{time.time()-t1:.0f}s", flush=True)

    for rec in rows:
        rec.pop("_human_bytes", None)
        rec.pop("_plain_bytes", None)
        rec.pop("_ipa_bytes", None)

    n = sum(1 for r in with_clip if r.get("ipa_f1") is not None)
    summary = {
        "voice": VOICE,
        "n_names": len(rows),
        "n_clips": len(with_clip),
        "n_scored_ipa": n,
        "n_flagged": sum(1 for r in rows if r.get("flags")),
        "mean_plain": round(
            sum(r["plain"] for r in with_clip if r.get("plain") is not None)
            / max(1, sum(1 for r in with_clip if r.get("plain") is not None)),
            4,
        ),
        "mean_ipa": round(
            sum(r["ipa_f1"] for r in with_clip if r.get("ipa_f1") is not None) / max(1, n),
            4,
        )
        if n
        else None,
    }
    if n:
        summary["ipa_beats_plain"] = sum(
            1 for r in with_clip if (r.get("delta") or 0) > 0.01
        )
        summary["ipa_loses_to_plain"] = sum(
            1 for r in with_clip if (r.get("delta") or 0) < -0.01
        )
    (OUT / "results.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False)
        + "\n"
    )
    write_listen(rows)
    print("SUMMARY", json.dumps(summary), flush=True)
    print("listen", OUT / "index.html", flush=True)
    mix = [
        r
        for r in rows
        if any(
            f.startswith("url_other")
            or f.startswith("ipa_matches_generic")
            or f == "mab_ipa_on_non_mab_brand"
            for f in (r.get("flags") or [])
        )
    ]
    print(f"mixup-like {len(mix)}", flush=True)
    for r in mix:
        print(
            f"  {r['ingredient']}: {r['ipa']} :: {r['flags']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
