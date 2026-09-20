#!/usr/bin/env python
"""Google-grounded pronunciations for every DOSE name via Gemini 3 Flash.

Ask for **IPA**. Do not convert DailyMed respelling into IPA yourself.
See dose_r/references/README.md.
"""
from __future__ import annotations

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

import requests

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "project-amer-scs-sandbox")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
LOCATION = os.environ.get("GEMINI_LOCATION", "global")
NAMES = ROOT / "dose_r" / "references" / "pronunciations.jsonl"
OUT_DIR = Path(os.environ.get("OUT_DIR", str(ROOT / "runs" / "gemini3-web-pronunciations")))
RAW_DIR = OUT_DIR / "raw"
OUT = OUT_DIR / "pronunciations.jsonl"
WORKERS = int(os.environ.get("GEMINI_WORKERS", "24"))

PROMPT = """Use Google Search with this exact query and no other query:

{name} ipa pronunciation

That is the whole task. Copy the IPA Google shows for "{name}"
(AI overview, knowledge panel, Wiktionary, first result).

JSON only, no markdown:

{{"ingredient":"{name}","ipa":"/ˈnɛksiəm/","url":"https://..."}}

- ipa: the IPA string. Keep stress marks. Slashes or brackets as shown.
- url: the page or Google result it came from.
- If Google shows IPA, return it. Do not refuse just because DailyMed printed a respelling instead.
"""

_JSON_RE = re.compile(r"\{.*\}", re.S)
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


def endpoint() -> str:
    return (
        f"https://aiplatform.googleapis.com/v1/projects/{PROJECT}"
        f"/locations/{LOCATION}/publishers/google/models/{MODEL}:generateContent"
    )


def extract_text(result: dict) -> str:
    cands = result.get("candidates") or []
    if not cands:
        return ""
    parts = (cands[0].get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts if "text" in p)


def parse_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        m = _JSON_RE.search(text)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None


def grounding(result: dict) -> dict:
    meta = (result.get("candidates") or [{}])[0].get("groundingMetadata") or {}
    chunks = []
    for ch in meta.get("groundingChunks") or []:
        web = ch.get("web") or {}
        if web.get("uri") or web.get("title"):
            chunks.append(
                {
                    "uri": web.get("uri") or "",
                    "title": web.get("title") or "",
                    "domain": web.get("domain") or "",
                }
            )
    html = (meta.get("searchEntryPoint") or {}).get("renderedContent") or ""
    chips = re.findall(r'href="(https://[^"]+)"', html)
    return {
        "chunks": chunks,
        "web_search_queries": meta.get("webSearchQueries") or [],
        "search_chip_urls": chips,
    }


def call(name: str, kind: str) -> dict:
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": PROMPT.format(name=name, kind=kind)}],
            }
        ],
        "tools": [{"google_search": {}}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 1024,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    last_err = None
    for attempt in range(8):
        try:
            resp = requests.post(
                endpoint(),
                headers={
                    "Authorization": f"Bearer {token()}",
                    "Content-Type": "application/json",
                    "x-goog-user-project": PROJECT,
                },
                json=body,
                timeout=90,
            )
            if resp.status_code == 429:
                wait = min(2**attempt, 30)
                time.sleep(wait)
                last_err = f"429 {resp.text[:200]}"
                continue
            if resp.status_code >= 500:
                time.sleep(min(2**attempt, 20))
                last_err = f"{resp.status_code} {resp.text[:200]}"
                continue
            if resp.status_code != 200:
                return {
                    "ingredient": name,
                    "ok": False,
                    "error": f"{resp.status_code}: {resp.text[:800]}",
                }
            result = resp.json()
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            safe = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            (RAW_DIR / f"{safe}.json").write_text(
                json.dumps(result, ensure_ascii=False)
            )
            text = extract_text(result)
            parsed = parse_json(text) or {}
            return {
                "ingredient": name,
                "name_type": kind,
                "ok": True,
                "model": MODEL,
                "parsed": parsed,
                "raw_text": text,
                "grounding": grounding(result),
                "finish_reason": (result.get("candidates") or [{}])[0].get(
                    "finishReason"
                ),
            }
        except Exception as exc:
            last_err = str(exc)
            time.sleep(min(2**attempt, 20))
    return {"ingredient": name, "ok": False, "error": last_err or "unknown"}


def load_done() -> set[str]:
    if not OUT.exists():
        return set()
    done = set()
    for line in OUT.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("ok"):
            done.add(rec["ingredient"])
    return done


def append(rec: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with _lock:
        with OUT.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_names() -> list[tuple[str, str]]:
    rows = []
    for line in NAMES.read_text().splitlines():
        rec = json.loads(line)
        rows.append((rec["ingredient"], rec.get("name_type") or "drug"))
    only = os.environ.get("ONLY")
    if only:
        want = {x.strip() for x in only.split(",") if x.strip()}
        rows = [r for r in rows if r[0] in want]
    return rows


def main() -> int:
    names = load_names()
    if os.environ.get("LIMIT"):
        names = names[: int(os.environ["LIMIT"])]
    done = load_done()
    todo = [(n, k) for n, k in names if n not in done]
    print(
        f"model={MODEL} loc={LOCATION} workers={WORKERS} "
        f"todo={len(todo)} done={len(done)} of {len(names)}",
        flush=True,
    )
    if not todo:
        return 0
    ok = fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(call, n, k): n for n, k in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            name = futs[fut]
            try:
                rec = fut.result()
            except Exception as exc:
                rec = {"ingredient": name, "ok": False, "error": str(exc)}
            append(rec)
            if rec.get("ok"):
                ok += 1
                parsed = rec.get("parsed") or {}
                ipa = parsed.get("ipa")
                if isinstance(ipa, list):
                    vals = [
                        (x.get("value") if isinstance(x, dict) else str(x) or "").strip()
                        for x in ipa
                    ]
                    ipa = " | ".join(v for v in vals if v)
                flag = f"IPA {ipa or 'MISSING'} {(parsed.get('url') or '')[:60]}"
            else:
                fail += 1
                flag = rec.get("error", "fail")[:80]
            print(
                f"[{i}/{len(todo)} +{time.time()-t0:.0f}s] {name}: {flag}",
                flush=True,
            )
    print(f"DONE ok={ok} fail={fail} -> {OUT}", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
