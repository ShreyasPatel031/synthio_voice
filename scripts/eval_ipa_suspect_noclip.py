#!/usr/bin/env python
"""No-clip IPA suspects: two Gemini spellings vs Cloud IPA, plus Gemini 3 Flash IPA variants.

Spelling arms (source respelling unchanged except compact = hyphens stripped):
  one_word  published hyphen + best prompt (do not spell)
  compact   hyphens stripped

Gemini 3 Flash proposes 3 IPA strings (asked for IPA, never G2P). Those are
replacement *candidates*. Does not write pronunciations.jsonl.
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import sys
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
from dose_r.references.tts_pronunciation import (  # noqa: E402
    compact_ascii,
    custom_pronunciation,
    is_source_ipa,
    spoken_text,
)

TTS_EP = "https://texttospeech.googleapis.com/v1/text:synthesize"
GEMINI_TTS = "gemini-3.1-flash-tts-preview"
FLASH = "gemini-3-flash-preview"
KORE = "Kore"
CLOUD = "en-US-Standard-C"
RATE = 24000
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "project-amer-scs-sandbox")
ONE_WORD = (
    "You are saying a single US drug name out loud to a patient. "
    "Read it as one word. Do not spell it. Do not pause on capital letters."
)
PRON = ROOT / "dose_r" / "references" / "pronunciations.jsonl"
VERIFY = ROOT / "runs" / "verify-google-ipa" / "results.json"
COMPACT_TTS = ROOT / "runs" / "gemini31-respell-all" / "tts"
HOLD_PROMPT = ROOT / "runs" / "gemini31-holdout-prompt" / "tts"
VERIFY_TTS = ROOT / "runs" / "verify-google-ipa" / "tts"
RESPELL_TTS = ROOT / "runs" / "ipa-vs-respell" / "tts"
OUT = ROOT / "runs" / "ipa-suspect-noclip"
TTS = OUT / "tts"
FLASH_CACHE = OUT / "flash_cache"
LISTEN = ROOT / "runs" / "listen-ipa-suspect-20"
WORKERS = int(os.environ.get("TTS_WORKERS", "4"))
LISTEN_N = 20
FLASH_N = 25
_token = {"value": None, "exp": 0.0}
_EMB: dict = {}


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


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def cloud_ipa_wav(name: str) -> Path | None:
    s = slug(name)
    for p in (VERIFY_TTS / f"{s}.ipa.wav", RESPELL_TTS / f"{s}.ipa.wav"):
        if p.exists() and p.stat().st_size > 500:
            return p
    return None


def mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def synth_gemini(text: str, prompt: str) -> bytes:
    body = {
        "input": {"text": text, "prompt": prompt},
        "voice": {"languageCode": "en-US", "name": KORE, "modelName": GEMINI_TTS},
        "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": RATE},
    }
    last: Exception | None = None
    for attempt in range(6):
        try:
            resp = requests.post(
                TTS_EP,
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
            raise RuntimeError(f"{resp.status_code}: {resp.text[:300]}")
        return base64.b64decode(resp.json()["audioContent"])
    raise last or RuntimeError("gemini tts failed")


def synth_cloud_ipa(name: str, ipa: str, part_ipas: list[tuple[str, str]] | None = None) -> bytes:
    """Cloud Standard-C + IPA. Multi-word: one customPronunciation phrase per word."""
    from dose_r.references.tts_pronunciation import (
        custom_pronunciations_for_parts,
        spoken_parts,
    )

    spoken = spoken_text(name)
    parts = spoken_parts(name)
    if part_ipas:
        pairs = [(p, i.strip().strip("/")) for p, i in part_ipas if p and i]
    elif len(parts) > 1 and ("/" in (ipa or "") and (ipa or "").count("/") >= 4):
        # "/foo/ /bar/" → per-word
        import re

        chunks = [c.strip() for c in re.findall(r"/([^/]+)/", ipa or "") if c.strip()]
        pairs = list(zip(parts, chunks)) if len(chunks) == len(parts) else []
    else:
        pairs = []
    if len(parts) > 1 and pairs and len(pairs) == len(parts):
        custom = custom_pronunciations_for_parts(pairs)
    else:
        phrase = spoken if " " not in spoken else (spoken.split()[0] if spoken.split() else name)
        custom = custom_pronunciation(phrase, ipa.strip().strip("/"))
    body = {
        "input": {
            "text": spoken,
            "customPronunciations": custom,
        },
        "voice": {"languageCode": "en-US", "name": CLOUD},
        "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": RATE},
    }
    resp = requests.post(
        TTS_EP,
        headers={
            "Authorization": f"Bearer {token()}",
            "Content-Type": "application/json",
            "x-goog-user-project": PROJECT,
        },
        json=body,
        timeout=90,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"{resp.status_code}: {resp.text[:300]}")
    return base64.b64decode(resp.json()["audioContent"])


def cached_write(path: Path, fn, *args) -> bytes:
    if path.exists() and path.stat().st_size > 500:
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = fn(*args)
    path.write_bytes(data)
    return data


def f1(a: Path, b: Path) -> float:
    from dose_r.scoring.speech_similarity import (
        extract_frame_embeddings,
        speech_bertscore,
    )

    def emb(p: Path):
        key = (str(p), p.stat().st_size)
        if key not in _EMB:
            _EMB[key] = extract_frame_embeddings(p)
        return _EMB[key]

    return float(speech_bertscore(emb(a), emb(b))["f1"])


def strip_ipa(raw: str) -> str:
    s = (raw or "").strip().strip("/[]() ")
    s = s.replace("'", "ˈ")
    return s


def flash_variations(name: str, respelling: str, current_ipa: str) -> list[dict]:
    FLASH_CACHE.mkdir(parents=True, exist_ok=True)
    cache = FLASH_CACHE / f"{slug(name)}.json"
    if cache.exists():
        try:
            hit = json.loads(cache.read_text())
            if isinstance(hit, list):
                return hit[:3]
        except json.JSONDecodeError:
            pass
    prompt = (
        f'US drug name: {name}\n'
        f'Published dictionary respelling (do not convert this to IPA yourself): {respelling}\n'
        f'Current Google/Cloud IPA candidate (may be wrong): {current_ipa or "(none)"}\n\n'
        "Search for source-published US English IPA for this exact drug name. "
        "Give 3 likely IPA pronunciations, sorted by confidence descending. "
        "Quote IPA a real page printed when you can. "
        "Return JSON only:\n"
        '{"variations":[{"ipa":"/.../","confidence":0.0,"source":"url or guess"}]}\n'
        "confidence is 0-1. No markdown."
    )
    url = (
        f"https://aiplatform.googleapis.com/v1/projects/{PROJECT}"
        f"/locations/global/publishers/google/models/{FLASH}:generateContent"
    )
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {"temperature": 0.2},
    }
    last = ""
    for attempt in range(4):
        try:
            resp = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {token()}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=120,
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            last = str(exc)
            time.sleep(min(2**attempt, 16))
            continue
        if resp.status_code == 429:
            time.sleep(min(2**attempt, 16))
            continue
        if resp.status_code != 200:
            last = resp.text[:300]
            time.sleep(1)
            continue
        text = ""
        try:
            parts = resp.json()["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text") or "" for p in parts)
        except Exception:
            last = resp.text[:300]
            continue
        blob = text.strip()
        m = re.search(r"\{.*\}", blob, re.S)
        if not m:
            last = blob[:300]
            continue
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            last = blob[:300]
            continue
        out = []
        for rec in data.get("variations") or []:
            ipa = strip_ipa(rec.get("ipa") or "")
            if not ipa or not is_source_ipa(ipa):
                continue
            try:
                conf = float(rec.get("confidence") or 0)
            except (TypeError, ValueError):
                conf = 0.0
            out.append(
                {
                    "ipa": ipa,
                    "confidence": conf,
                    "source": rec.get("source") or "",
                }
            )
        out.sort(key=lambda r: r["confidence"], reverse=True)
        out = out[:3]
        cache.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
        return out
    print(f"  flash fail {name}: {last}", flush=True)
    cache.write_text("[]\n")
    return []


def load_pron() -> list[dict]:
    rows = []
    for line in PRON.read_text().splitlines():
        rec = json.loads(line)
        if (rec.get("respelling") or "").strip():
            rows.append(rec)
    return rows


def write_listen(rows: list[dict], clips: dict) -> Path:
    LISTEN.mkdir(parents=True, exist_ok=True)
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Worst 20 no-clip IPA suspects</title>",
        "<style>body{font:16px/1.4 system-ui;max-width:880px;margin:2rem auto;padding:0 1rem}",
        "section{border:1px solid #ccc;border-radius:8px;padding:1rem 1.2rem;margin:1rem 0}",
        "h1{font-size:1.2rem} h2{font-size:1.05rem;margin:0 0 .3rem}",
        ".ipa{font-family:ui-monospace,monospace;font-size:.95rem}",
        "label{display:block;font-weight:600;margin:.4rem 0 .1rem} audio{width:100%}",
        ".meta{font-size:.9rem;color:#555} table{border-collapse:collapse;margin:.4rem 0 0.8rem}",
        "td,th{border:1px solid #ddd;padding:.2rem .45rem;text-align:left}</style>",
        "<h1>Worst 20: Cloud IPA vs Gemini hyphen (one-word prompt) and compact</h1>",
        "<p>Published respelling is unchanged. Gemini 3 Flash IPA variants are "
        "replacement candidates — not written to pronunciations.jsonl.</p>",
    ]
    for rec in rows:
        s = slug(rec["ingredient"])
        parts.append(f"<section id='{s}'><h2>{rec['rank']}. {rec['ingredient']}</h2>")
        parts.append(
            f"<p>respelling <span class='ipa'>{rec.get('respelling')}</span><br>"
            f"current IPA <span class='ipa'>{rec.get('cloud_ipa') or '—'}</span></p>"
        )
        parts.append(
            "<p class='meta'>"
            f"div(v1 vs spellings) {rec.get('div_v1')} · "
            f"one_word vs IPA {rec.get('one_word_vs_ipa')} · "
            f"compact vs IPA {rec.get('compact_vs_ipa')} · "
            f"one_word vs compact {rec.get('one_word_vs_compact')}</p>"
        )
        if rec.get("variations"):
            parts.append("<table><tr><th>#</th><th>IPA</th><th>conf</th><th>F1 vs one_word</th><th>F1 vs compact</th></tr>")
            for i, v in enumerate(rec["variations"], 1):
                parts.append(
                    f"<tr><td>v{i}</td><td class='ipa'>{v.get('ipa')}</td>"
                    f"<td>{v.get('confidence')}</td>"
                    f"<td>{v.get('f1_one_word')}</td>"
                    f"<td>{v.get('f1_compact')}</td></tr>"
                )
            parts.append("</table>")
        clip = clips.get(rec["ingredient"])
        if clip is not None:
            dest = LISTEN / f"{s}__human{clip.path.suffix}"
            if not dest.exists():
                dest.write_bytes(clip.path.read_bytes())
            parts.append(
                f"<label>human ({clip.source})</label>"
                f"<audio controls src='{dest.name}'></audio>"
            )
        for key, label in (
            ("one_word", "Gemini 3.1 hyphen + one-word prompt"),
            ("compact", "Gemini 3.1 compact"),
            ("cloud_ipa", "Cloud Standard-C + current Google IPA"),
            ("v1", "Gemini 3 Flash IPA v1 (highest conf)"),
            ("v2", "Gemini 3 Flash IPA v2"),
            ("v3", "Gemini 3 Flash IPA v3"),
        ):
            src = TTS / f"{s}.{key}.wav"
            if not src.exists():
                continue
            dest = LISTEN / f"{s}__{key}.wav"
            if not dest.exists():
                shutil.copy2(src, dest)
            extra = ""
            if key.startswith("v") and rec.get("variations"):
                idx = int(key[1]) - 1
                if idx < len(rec["variations"]):
                    extra = f" <span class='ipa'>{rec['variations'][idx].get('ipa')}</span>"
            parts.append(
                f"<label>{label}{extra}</label>"
                f"<audio controls src='{dest.name}'></audio>"
            )
        parts.append("</section>")
    dest = LISTEN / "index.html"
    dest.write_text("\n".join(parts) + "\n")
    return dest


def main() -> int:
    clips = available_clips()
    verify = {}
    if VERIFY.exists():
        verify = {r["ingredient"]: r for r in json.loads(VERIFY.read_text())["rows"]}
    items = [
        r
        for r in load_pron()
        if r["ingredient"] not in clips and cloud_ipa_wav(r["ingredient"]) is not None
    ]
    TTS.mkdir(parents=True, exist_ok=True)
    print(f"{len(items)} no-clip names with Cloud IPA wav", flush=True)

    jobs = []
    for rec in items:
        name = rec["ingredient"]
        text = rec["respelling"].strip()
        s = slug(name)
        ow = TTS / f"{s}.one_word.wav"
        reuse = HOLD_PROMPT / f"{s}.one_word.wav"
        jobs.append((name, text, ow, reuse))
        cp_src = COMPACT_TTS / f"{s}.compact.wav"
        cp_dst = TTS / f"{s}.compact.wav"
        if cp_src.exists() and not cp_dst.exists():
            shutil.copy2(cp_src, cp_dst)
        ipa_src = cloud_ipa_wav(name)
        ipa_dst = TTS / f"{s}.cloud_ipa.wav"
        if ipa_src is not None and not ipa_dst.exists():
            shutil.copy2(ipa_src, ipa_dst)

    def one_word_job(name, text, path, reuse):
        if path.exists() and path.stat().st_size > 500:
            return path
        if reuse.exists() and reuse.stat().st_size > 500:
            shutil.copy2(reuse, path)
            return path
        return cached_write(path, synth_gemini, text, ONE_WORD)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {
            ex.submit(one_word_job, name, text, path, reuse): name
            for name, text, path, reuse in jobs
        }
        done = 0
        for fut in as_completed(futs):
            name = futs[fut]
            done += 1
            try:
                fut.result()
                print(f"  one_word {done}/{len(jobs)} {name}", flush=True)
            except Exception as exc:
                print(f"  FAIL one_word {name}: {exc}", flush=True)

    scored = []
    for rec in items:
        name = rec["ingredient"]
        s = slug(name)
        ow, cp, ipa = TTS / f"{s}.one_word.wav", TTS / f"{s}.compact.wav", TTS / f"{s}.cloud_ipa.wav"
        if not (ow.exists() and cp.exists() and ipa.exists()):
            continue
        ow_ipa = round(f1(ow, ipa), 4)
        cp_ipa = round(f1(cp, ipa), 4)
        ow_cp = round(f1(ow, cp), 4)
        vs_spell = mean([ow_ipa, cp_ipa])
        v = verify.get(name) or {}
        scored.append(
            {
                "ingredient": name,
                "respelling": rec["respelling"],
                "cloud_ipa": v.get("ipa_used") or v.get("ipa") or "",
                "one_word_vs_ipa": ow_ipa,
                "compact_vs_ipa": cp_ipa,
                "one_word_vs_compact": ow_cp,
                "ipa_vs_spellings": vs_spell,
            }
        )
        print(
            f"  score {name} ow-ipa={ow_ipa} c-ipa={cp_ipa} ow-c={ow_cp} mean={vs_spell}",
            flush=True,
        )

    scored.sort(key=lambda r: r["ipa_vs_spellings"] or 9)
    flash_set = scored[:FLASH_N]
    print(f"Gemini 3 Flash IPA variants for top {len(flash_set)} divergences", flush=True)
    for rec in flash_set:
        name = rec["ingredient"]
        vars_ = flash_variations(name, rec["respelling"], rec["cloud_ipa"])
        rec["variations"] = vars_
        print(f"  flash {name} {[(v['ipa'], v['confidence']) for v in vars_]}", flush=True)
        s = slug(name)
        name_parts = spoken_text(name).split()
        for i, var in enumerate(vars_, 1):
            path = TTS / f"{s}.v{i}.wav"
            try:
                ipa = var["ipa"]
                part_ipas = None
                bits = (ipa or "").strip().split()
                if len(name_parts) > 1 and len(bits) == len(name_parts):
                    part_ipas = list(zip(name_parts, bits))
                cached_write(path, synth_cloud_ipa, name, ipa, part_ipas)
                var["synth_ok"] = True
                if part_ipas:
                    var["ipa_parts"] = [
                        {"phrase": p, "ipa": x} for p, x in part_ipas
                    ]
            except Exception as exc:
                var["synth_ok"] = False
                var["synth_error"] = str(exc)[:200]
                print(f"  FAIL ipa v{i} {name}: {exc}", flush=True)

    for rec in flash_set:
        s = slug(name := rec["ingredient"])
        ow, cp = TTS / f"{s}.one_word.wav", TTS / f"{s}.compact.wav"
        for var, i in zip(rec.get("variations") or [], range(1, 4)):
            vp = TTS / f"{s}.v{i}.wav"
            if not vp.exists():
                continue
            var["f1_one_word"] = round(f1(vp, ow), 4)
            var["f1_compact"] = round(f1(vp, cp), 4)
            var["f1_vs_spellings"] = mean(
                [x for x in (var["f1_one_word"], var["f1_compact"]) if x is not None]
            )
        v1 = (rec.get("variations") or [None])[0]
        rec["div_v1"] = (
            round(1 - v1["f1_vs_spellings"], 4)
            if v1 and v1.get("f1_vs_spellings") is not None
            else round(1 - (rec["ipa_vs_spellings"] or 0), 4)
        )

    flash_set.sort(
        key=lambda r: (
            -(r.get("div_v1") or 0),
            r.get("ipa_vs_spellings") or 9,
        )
    )
    worst = flash_set[:LISTEN_N]
    for i, rec in enumerate(worst, 1):
        rec["rank"] = i
    summary = {
        "n_no_clip_with_ipa": len(scored),
        "n_flash": len(flash_set),
        "n_listen": len(worst),
        "spelling_arms": ["one_word hyphen (published, unchanged)", "compact"],
        "flash_model": FLASH,
        "worst20": [
            {
                "rank": r["rank"],
                "ingredient": r["ingredient"],
                "respelling": r["respelling"],
                "cloud_ipa": r["cloud_ipa"],
                "ipa_vs_spellings": r["ipa_vs_spellings"],
                "div_v1": r.get("div_v1"),
                "one_word_vs_ipa": r["one_word_vs_ipa"],
                "compact_vs_ipa": r["compact_vs_ipa"],
                "variations": r.get("variations"),
            }
            for r in worst
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps({"summary": summary, "scored": scored, "flash": flash_set}, indent=2)
        + "\n"
    )
    html = write_listen(worst, clips)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(html, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
