#!/usr/bin/env python
"""Find human pronunciation audio for names with no gold clip.

Gold sources (NCI, MW, Drugs.com, UMich) stay preferred. Hits here are
*secondary*: Wiktionary/Wikipedia/Commons recordings, plus Google-found
hosted audio. They only become Path 2 gold when nothing higher exists.

Does not invent audio. Does not download TTS sites (howtopronounce robot
voice). Does not G2P respelling.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dose_r.adapters.base import sha256_hex
from dose_r.references import audio_manifest
from dose_r.references.reference_clips import available_clips
from dose_r.references.usan_stems import _split_fda_suffix

VERIFY = ROOT / "runs" / "verify-google-ipa" / "results.json"
PRON = ROOT / "dose_r" / "references" / "pronunciations.jsonl"
OUT_DIR = ROOT / "runs" / "secondary-audio"
AUDIO_DIR = ROOT / "data" / "reference_audio" / "secondary"
UA = {
    "User-Agent": "DOSE-R-research-bot/1.0 "
    "(gold pronunciation reference; contact: shreyas.patel@searce.com)"
}
TIMEOUT = 25
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "project-amer-scs-sandbox")
MODEL = "gemini-3-flash-preview"
WORKERS = int(os.environ.get("AUDIO_SEARCH_WORKERS", "8"))

AUDIO_EXT = (".ogg", ".oga", ".mp3", ".wav", ".opus", ".webm")
TTS_HOSTS = (
    "howtopronounce.com",
    "translate.google",
    "text-to-speech",
    "responsivevoice",
    "voicerss",
    "ttsmp3",
)
SKIP_HOSTS = TTS_HOSTS + ("google.com/search",)

GEMINI_PROMPT = """Use Google Search. Drug name: "{name}".

Find a HUMAN pronunciation recording of this exact name (not a different
drug, not a TTS robot). Run:
- {name} pronunciation audio
- {name} wiktionary
- {name} how to pronounce

Prefer in this order:
1. Wiktionary / Wikimedia Commons .ogg/.mp3
2. MedlinePlus or other NIH/FDA-hosted audio
3. Forvo human recording
4. YouTube of a person saying the name (not a TTS channel)

JSON only:
{{"audio_url":"https://...","page_url":"https://...","source":"wiktionary|commons|medlineplus|forvo|youtube|other","notes":""}}

audio_url must be a playable file or a YouTube watch URL. Empty string if none.
Never return howtopronounce.com or Google Translate TTS.
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_types() -> dict[str, str]:
    out = {}
    for line in PRON.read_text().splitlines():
        rec = json.loads(line)
        out[rec["ingredient"]] = rec.get("name_type") or ""
    return out


def no_clip_names() -> list[str]:
    clips = available_clips()
    rows = json.loads(VERIFY.read_text())["rows"]
    return [r["ingredient"] for r in rows if r["ingredient"] not in clips]


def fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return resp.read()


def wiki_get(url: str) -> dict:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8", "ignore"))


def probe(path: Path) -> dict:
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration,format_name:stream=codec_name,sample_rate,channels",
                "-of",
                "json",
                str(path),
            ],
            timeout=15,
        )
        info = json.loads(out)
        fmt = info.get("format") or {}
        streams = info.get("streams") or [{}]
        st = streams[0]
        dur = float(fmt.get("duration") or 0)
        return {
            "ok": True,
            "format": (fmt.get("format_name") or path.suffix.lstrip(".")).split(",")[0],
            "duration_s": round(dur, 4),
            "sample_rate_hz": int(st.get("sample_rate") or 0) or None,
            "channels": int(st.get("channels") or 0) or None,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}


def looks_like_tts(url: str) -> bool:
    u = (url or "").lower()
    return any(h in u for h in TTS_HOSTS)


def classify_source(url: str, hinted: str = "") -> str:
    u = (url or "").lower()
    if hinted in {
        "wiktionary",
        "wikipedia",
        "commons",
        "medlineplus",
        "forvo",
        "youtube",
        "merriam-webster",
    }:
        return hinted
    if "wiktionary" in u:
        return "wiktionary"
    if "wikipedia" in u or "wikimedia" in u or "upload.wikimedia.org" in u:
        return "commons"
    if "medlineplus" in u or "nlm.nih.gov" in u:
        return "medlineplus"
    if "forvo.com" in u:
        return "forvo"
    if "youtube.com" in u or "youtu.be" in u:
        return "youtube"
    if "merriam-webster" in u:
        return "merriam-webster"
    if "nci-media" in u or "cancer.gov" in u:
        return "nci"
    return "web"


def titles_to_try(name: str) -> list[str]:
    stem, _ = _split_fda_suffix(name)
    out = []
    for t in (name, stem, name.replace("-", " "), stem.replace("-", " ")):
        t = t.strip()
        if t and t not in out:
            out.append(t)
            cap = t[:1].upper() + t[1:]
            if cap not in out:
                out.append(cap)
    return out


def wiki_audio_files(domain: str, title: str) -> list[dict]:
    api = f"https://{domain}/w/api.php"
    q = urllib.parse.urlencode(
        {
            "action": "query",
            "titles": title,
            "prop": "images",
            "imlimit": 50,
            "format": "json",
        }
    )
    data = wiki_get(f"{api}?{q}")
    pages = (data.get("query") or {}).get("pages") or {}
    files = []
    for page in pages.values():
        if "missing" in page:
            continue
        for im in page.get("images") or []:
            fname = im.get("title") or ""
            low = fname.lower()
            if not any(low.endswith(ext) for ext in AUDIO_EXT):
                continue
            files.append(fname)
    return files


def file_info(domain: str, title: str) -> dict | None:
    q = urllib.parse.urlencode(
        {
            "action": "query",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url|mime|size",
            "format": "json",
        }
    )
    data = wiki_get(f"https://{domain}/w/api.php?{q}")
    pages = (data.get("query") or {}).get("pages") or {}
    for page in pages.values():
        info = (page.get("imageinfo") or [None])[0]
        if info and info.get("url"):
            return {"title": title, **info}
    return None


def commons_search(name: str) -> list[str]:
    q = urllib.parse.urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": f"{name} pronunciation",
            "srnamespace": 6,
            "srlimit": 8,
            "format": "json",
        }
    )
    data = wiki_get(f"https://commons.wikimedia.org/w/api.php?{q}")
    hits = []
    for row in (data.get("query") or {}).get("search") or []:
        title = row.get("title") or ""
        low = title.lower()
        if any(low.endswith(ext) for ext in AUDIO_EXT):
            hits.append(title)
    return hits


def name_in_title(name: str, title: str) -> bool:
    t = re.sub(r"[^a-z0-9]+", "", title.lower())
    tokens = [x for x in re.split(r"[^a-z0-9]+", name.lower()) if len(x) >= 4]
    if not tokens:
        tokens = [re.sub(r"[^a-z0-9]+", "", name.lower())]
    return any(tok in t for tok in tokens)


def prefer_us(files: list[str]) -> list[str]:
    def key(title: str) -> tuple:
        t = title.lower()
        return (
            0 if "en-us" in t or "en_us" in t else 1,
            0 if "pronunciation" in t or "pron" in t else 1,
            len(t),
        )

    return sorted(files, key=key)


def gemini_audio(name: str) -> dict:
    import requests
    import google.auth
    from google.auth.transport.requests import Request

    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    creds.refresh(Request())
    url = (
        f"https://aiplatform.googleapis.com/v1/projects/{PROJECT}"
        f"/locations/global/publishers/google/models/{MODEL}:generateContent"
    )
    body = {
        "contents": [
            {"role": "user", "parts": [{"text": GEMINI_PROMPT.format(name=name)}]}
        ],
        "tools": [{"google_search": {}}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 512,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {creds.token}",
            "Content-Type": "application/json",
            "x-goog-user-project": PROJECT,
        },
        json=body,
        timeout=90,
    )
    if resp.status_code != 200:
        return {"error": f"{resp.status_code}: {resp.text[:200]}"}
    parts = (resp.json().get("candidates") or [{}])[0].get("content", {}).get(
        "parts"
    ) or []
    text = "".join(p.get("text", "") for p in parts if "text" in p)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def save_clip(
    *,
    ingredient: str,
    name_type: str,
    source: str,
    url: str,
    page_url: str,
    data: bytes,
    ext: str,
    coverage: str = "full",
    flags: list[str] | None = None,
) -> dict | None:
    if looks_like_tts(url) or looks_like_tts(page_url):
        return None
    dest_dir = AUDIO_DIR / source
    dest_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", ingredient.lower()).strip("_")
    dest = dest_dir / f"{slug}{ext}"
    dest.write_bytes(data)
    info = probe(dest)
    flags = list(flags or [])
    if not info.get("ok"):
        flags.append(f"undecodable: {info.get('error')}")
    dur = info.get("duration_s")
    if dur is not None and (dur < 0.25 or dur > 8.0):
        flags.append(f"duration {dur}s outside 0.25-8s")
        if dur > 20:
            dest.unlink(missing_ok=True)
            return None
    rec = {
        "ingredient": ingredient,
        "name_type": name_type,
        "source": source,
        "source_name": f"secondary/{source}",
        "query": ingredient,
        "coverage": coverage,
        "headword": ingredient,
        "source_url": url,
        "page_url": page_url,
        "local_path": str(dest.relative_to(ROOT)),
        "sha256": sha256_hex(data),
        "bytes": len(data),
        "format": info.get("format") or ext.lstrip("."),
        "duration_s": dur,
        "sample_rate_hz": info.get("sample_rate_hz"),
        "channels": info.get("channels"),
        "status": "ok" if info.get("ok") and not flags else "flagged",
        "flags": flags,
        "fetched_at": now(),
        "role": "secondary",
    }
    return rec


def download_media(url: str) -> tuple[bytes, str] | None:
    if looks_like_tts(url):
        return None
    if "youtube.com" in url or "youtu.be" in url:
        return None
    path = urllib.parse.urlparse(url).path.lower()
    ext = next((e for e in AUDIO_EXT if path.endswith(e)), "")
    try:
        data = fetch_bytes(url)
    except Exception:
        return None
    if len(data) < 800:
        return None
    if not ext:
        if data.startswith(b"OggS"):
            ext = ".ogg"
        elif data[:3] == b"ID3" or data[:2] == b"\xff\xfb":
            ext = ".mp3"
        elif data[:4] == b"RIFF":
            ext = ".wav"
        else:
            return None
    return data, ext


def search_wiki(name: str) -> list[dict]:
    hits = []
    seen = set()
    for domain, source in (
        ("en.wiktionary.org", "wiktionary"),
        ("en.wikipedia.org", "wikipedia"),
    ):
        for title in titles_to_try(name):
            try:
                files = wiki_audio_files(domain, title)
            except Exception:
                continue
            for fname in prefer_us(files):
                if not name_in_title(name, fname):
                    continue
                if fname in seen:
                    continue
                seen.add(fname)
                hits.append({"source": source, "domain": domain, "file": fname, "page": title})
    try:
        for fname in prefer_us(commons_search(name)):
            if not name_in_title(name, fname):
                continue
            if fname not in seen:
                seen.add(fname)
                hits.append({"source": "commons", "domain": "commons.wikimedia.org", "file": fname, "page": name})
    except Exception:
        pass
    return hits[:4]


def recover_missing_mw(types: dict[str, str]) -> list[dict]:
    """Manifest already lists MW full clips whose files were never fetched."""
    clips = available_clips()
    out = []
    for rec in audio_manifest.load():
        if rec.get("source") != "merriam-webster":
            continue
        if rec.get("coverage") != "full":
            continue
        if rec["ingredient"] in clips:
            continue
        url = rec.get("source_url") or ""
        if not url:
            continue
        try:
            data = fetch_bytes(url)
        except Exception as exc:
            print(f"  MW miss {rec['ingredient']}: {exc}", flush=True)
            continue
        dest = ROOT / rec["local_path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        print(f"  MW recovered {rec['ingredient']} {dest} {len(data)}b", flush=True)
        out.append(rec)
    return out


def upsert(existing: list[dict], fresh: list[dict]) -> list[dict]:
    keys = {audio_manifest.record_key(r) for r in fresh}
    kept = [r for r in existing if audio_manifest.record_key(r) not in keys]
    return kept + fresh


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    types = load_types()
    names = no_clip_names()
    print(f"no-clip names: {len(names)}", flush=True)

    print("=== recover missing MW full files ===", flush=True)
    recover_missing_mw(types)
    names = no_clip_names()
    print(f"still no-clip after MW recover: {len(names)}", flush=True)

    print("=== wiktionary / wikipedia / commons ===", flush=True)
    wiki_map: dict[str, list[dict]] = {}

    def wiki_one(name: str):
        return name, search_wiki(name)

    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(wiki_one, n) for n in names]
        for i, fut in enumerate(as_completed(futs), 1):
            name, hits = fut.result()
            wiki_map[name] = hits
            if hits:
                print(f"  wiki [{i}/{len(names)}] {name}: {hits[0]['file']}", flush=True)
            elif i % 20 == 0:
                print(f"  wiki [{i}/{len(names)}] …", flush=True)

    print("=== google audio URLs for remaining ===", flush=True)
    need_google = [n for n in names if not wiki_map.get(n)]
    google_map: dict[str, dict] = {}

    def g_one(name: str):
        return name, gemini_audio(name)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(g_one, n) for n in need_google]
        for i, fut in enumerate(as_completed(futs), 1):
            name, hit = fut.result()
            google_map[name] = hit
            print(
                f"  google [{i}/{len(need_google)}] {name}: "
                f"{(hit or {}).get('audio_url') or (hit or {}).get('error') or 'none'}",
                flush=True,
            )

    print("=== download ===", flush=True)
    clips: list[dict] = []
    misses: list[dict] = []
    log = []

    for name in names:
        name_type = types.get(name, "")
        kept = False
        for hit in wiki_map.get(name) or []:
            info = file_info(hit.get("domain") or "commons.wikimedia.org", hit["file"])
            if not info:
                continue
            got = download_media(info["url"])
            if not got:
                continue
            data, ext = got
            stem, _ = _split_fda_suffix(name)
            coverage = "full"
            flags = []
            if stem.lower() != name.lower() and stem.lower() in hit["file"].lower():
                coverage = "component"
                flags.append("audio is the unsuffixed stem, not the full token")
            rec = save_clip(
                ingredient=name,
                name_type=name_type,
                source=hit["source"],
                url=info["url"],
                page_url=f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(hit['file'])}",
                data=data,
                ext=ext,
                coverage=coverage,
                flags=flags,
            )
            if rec:
                clips.append(rec)
                kept = True
                print(f"KEEP wiki {name} {hit['source']} {rec['duration_s']}s", flush=True)
                break
        if kept:
            continue
        g = google_map.get(name) or {}
        url = (g.get("audio_url") or "").strip()
        page = (g.get("page_url") or "").strip()
        log.append({"ingredient": name, "wiki": wiki_map.get(name), "google": g})
        if not url or looks_like_tts(url):
            misses.append(
                {
                    "ingredient": name,
                    "reason": "no_human_audio_url",
                    "google": g,
                }
            )
            continue
        if "youtube.com" in url or "youtu.be" in url:
            misses.append(
                {
                    "ingredient": name,
                    "reason": "youtube_url_not_downloaded",
                    "audio_url": url,
                    "page_url": page,
                }
            )
            continue
        got = download_media(url)
        if not got:
            misses.append(
                {
                    "ingredient": name,
                    "reason": "download_failed",
                    "audio_url": url,
                }
            )
            continue
        data, ext = got
        rec = save_clip(
            ingredient=name,
            name_type=name_type,
            source=classify_source(url, g.get("source") or ""),
            url=url,
            page_url=page or url,
            data=data,
            ext=ext,
        )
        if rec:
            clips.append(rec)
            print(f"KEEP google {name} {rec['source']} {rec['duration_s']}s", flush=True)
        else:
            misses.append({"ingredient": name, "reason": "rejected", "audio_url": url})

    existing = audio_manifest.load()
    merged = upsert(existing, clips)
    audio_manifest.save(merged)

    summary = {
        "n_no_clip_start": 105,
        "n_still_no_clip_before_secondary": len(names),
        "n_kept": len(clips),
        "n_miss": len(misses),
        "by_source": {},
        "kept": [
            {
                "ingredient": c["ingredient"],
                "source": c["source"],
                "duration_s": c["duration_s"],
                "coverage": c["coverage"],
                "status": c["status"],
            }
            for c in clips
        ],
        "misses": misses,
    }
    from collections import Counter

    summary["by_source"] = dict(Counter(c["source"] for c in clips))
    (OUT_DIR / "hits.json").write_text(
        json.dumps({"log": log, "summary": summary}, indent=2, ensure_ascii=False)
        + "\n"
    )
    still = no_clip_names()
    print(
        f"kept {len(clips)} secondary clips; still no clip: {len(still)}",
        flush=True,
    )
    print("by source", summary["by_source"], flush=True)
    print("wrote", OUT_DIR / "hits.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
