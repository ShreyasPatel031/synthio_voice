#!/usr/bin/env python
"""Fix names missing usable IPA / IPA TTS; score vs Gemini 3.1 one_word + compact.

Asks Gemini for source-published IPA only (never G2P respelling).
Does not write pronunciations.jsonl ipa fields.
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
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_hf = ROOT / ".cache" / "huggingface"
_hf.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_hf))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_hf))
os.environ.setdefault("HF_HUB_CACHE", str(_hf / "hub"))

import requests  # noqa: E402

from dose_r.references.tts_pronunciation import (  # noqa: E402
    compact_ascii,
    custom_pronunciation,
    custom_pronunciations_for_parts,
    is_source_ipa,
    page_is_for_word,
    spoken_parts,
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
IPA_JSON = ROOT / "runs" / "gemini3-ipa-query" / "ipa.json"
WEB = ROOT / "runs" / "gemini3-web-pronunciations" / "pronunciations.jsonl"
COMPACT_TTS = ROOT / "runs" / "gemini31-respell-all" / "tts"
HOLD_PROMPT = ROOT / "runs" / "gemini31-holdout-prompt" / "tts"
OUT = ROOT / "runs" / "fix-missing-ipa"
TTS = OUT / "tts"
CACHE = OUT / "flash_cache"
LISTEN = ROOT / "runs" / "listen-fix-missing-ipa"
WORKERS = int(os.environ.get("TTS_WORKERS", "4"))
FLASH_WORKERS = int(os.environ.get("FLASH_WORKERS", "6"))

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
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def strip_ipa(raw: str) -> str:
    s = (raw or "").strip().strip("/[]() ")
    s = s.replace("'", "ˈ").replace("ˈˈ", "ˈ")
    return s


def cloud_fold(ipa: str) -> str:
    """Synth-only folds so Cloud en-US accepts the string. Does not G2P."""
    s = strip_ipa(ipa)
    s = s.replace("ɪər", "ɪr").replace("ɪə", "ɪ")
    s = s.replace("ʊər", "ʊr").replace("ʊə", "ʊ")
    s = s.replace("ᵻ", "ɪ").replace("ɝ", "ɜr").replace("ɚ", "ər")
    s = s.replace("ɹ", "r").replace("ɡ", "g")
    s = re.sub(r"\([^)]*\)", "", s)  # drop optional (ʊ)
    s = s.replace("̯", "").replace("̩", "").replace("ːː", "ː")
    s = re.sub(r"\s+", " ", s.replace(".", "")).strip()
    return s


def has_ipa_wav(name: str) -> Path | None:
    s = slug(name)
    for p in (
        TTS / f"{s}.cloud_ipa.wav",
        ROOT / "runs/ipa-suspect-noclip/tts" / f"{s}.cloud_ipa.wav",
        ROOT / "runs/verify-google-ipa/tts" / f"{s}.ipa.wav",
        ROOT / "runs/ipa-vs-respell/tts" / f"{s}.ipa.wav",
    ):
        if p.exists() and p.stat().st_size > 500:
            return p
    return None


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


def _cloud_custom(name: str, ipa_raw: str, cand: str) -> tuple[str, dict]:
    """Build spoken text + customPronunciations for one IPA candidate."""
    text = spoken_text(name)
    parts = spoken_parts(name)
    chunks = re.findall(r"/([^/]+)/", ipa_raw or "")
    if len(parts) > 1 and len(chunks) == len(parts):
        pairs = [(p, cloud_fold(c)) for p, c in zip(parts, chunks)]
        return text, custom_pronunciations_for_parts(pairs)
    bits = [b for b in cand.split() if b]
    if len(parts) > 1 and len(bits) == len(parts):
        return text, custom_pronunciations_for_parts(list(zip(parts, bits)))
    phrase = parts[0] if len(parts) == 1 else text
    # Single IPA blob on multi-word: phrase must be substring of text.
    if len(parts) > 1 and " " not in cand:
        phrase = parts[0]
    return text, custom_pronunciation(phrase, cand.replace(" ", ""))


def synth_cloud_ipa(name: str, ipa: str) -> tuple[bytes, str]:
    raw = strip_ipa(ipa)
    candidates = [raw]
    folded = cloud_fold(raw)
    if folded and folded not in candidates:
        candidates.append(folded)
    nodot = raw.replace(".", "").replace(" ", "")
    if nodot and nodot not in candidates:
        candidates.append(nodot)
    last: Exception | None = None
    for cand in candidates:
        try:
            text, custom = _cloud_custom(name, ipa, cand)
            body = {
                "input": {"text": text, "customPronunciations": custom},
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
            return base64.b64decode(resp.json()["audioContent"]), cand
        except Exception as exc:
            last = exc
            continue
    raise last or RuntimeError("cloud ipa synth failed")


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


def load_pron() -> dict[str, dict]:
    out = {}
    for line in PRON.read_text().splitlines():
        rec = json.loads(line)
        out[rec["ingredient"]] = rec
    return out


def load_web_ipas() -> dict[str, list[tuple[str, str]]]:
    out: dict[str, list[tuple[str, str]]] = {}
    if not WEB.exists():
        return out
    for line in WEB.read_text().splitlines():
        rec = json.loads(line)
        name = rec["ingredient"]
        parsed = rec.get("parsed") or {}
        items = parsed.get("ipa") or []
        if isinstance(items, str):
            items = [{"value": items}]
        good: list[tuple[str, str]] = []
        for item in items:
            if isinstance(item, str):
                val, url = item, ""
            else:
                val = item.get("value") or ""
                url = item.get("url") or ""
            val = strip_ipa(val)
            if not val or not is_source_ipa(val):
                continue
            if "howtopronounce" in url.lower() or "scribd.com" in url.lower():
                continue
            if "[link]" in val.lower():
                continue
            stem = name.split()[0].split("-")[0]
            if url and ("wikipedia" in url or "wiktionary" in url):
                if not page_is_for_word(stem, url):
                    continue
            good.append((val, url))
        # prefer wiki / ama / fda / cambridge
        def rank(pair: tuple[str, str]) -> tuple:
            u = pair[1].lower()
            score = 0
            if "wiktionary" in u:
                score += 5
            if "wikipedia" in u:
                score += 4
            if "ama-assn" in u or "usan" in u:
                score += 4
            if "fda.gov" in u:
                score += 3
            if "cambridge" in u:
                score += 3
            if "leskoff" in u:
                score += 1
            return (-score, len(pair[0]))

        good.sort(key=rank)
        out[name] = good
    return out


def flash_ipa(name: str, respelling: str, bad_current: str) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{slug(name)}.json"
    if cache.exists():
        try:
            hit = json.loads(cache.read_text())
            if hit.get("ipa"):
                return hit
        except json.JSONDecodeError:
            pass
    prompt = (
        f'US drug / brand name: "{name}"\n'
        f'Published dictionary respelling (DO NOT convert this to IPA): {respelling or "(none)"}\n'
        f'Bad prior IPA candidate to ignore: {bad_current or "(none)"}\n\n'
        "Search the web for source-published US English IPA for this exact name.\n"
        "Prefer: Wiktionary {{IPA}}, Wikipedia IPA, FDA proprietary-name review, "
        "USAN/AMA statement, Cambridge, Merriam-Webster IPA.\n"
        "Do NOT use howtopronounce.com. Do NOT invent IPA from the respelling.\n"
        "If no real IPA page exists, return ipa null.\n"
        'Return JSON only: {"ipa":"/.../","url":"...","confidence":0.0}\n'
        "No markdown."
    )
    url = (
        f"https://aiplatform.googleapis.com/v1/projects/{PROJECT}"
        f"/locations/global/publishers/google/models/{FLASH}:generateContent"
    )
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {"temperature": 0.1},
    }
    last = ""
    for attempt in range(5):
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
        if resp.status_code in (429, 500, 503):
            time.sleep(min(2**attempt, 16))
            last = resp.text[:200]
            continue
        if resp.status_code != 200:
            last = resp.text[:300]
            time.sleep(1)
            continue
        try:
            parts = resp.json()["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text") or "" for p in parts)
        except Exception:
            last = resp.text[:300]
            continue
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            last = text[:300]
            continue
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            last = text[:300]
            continue
        ipa = strip_ipa(data.get("ipa") or "")
        if ipa.lower() in {"", "null", "none", "n/a"}:
            out = {"ipa": "", "url": data.get("url") or "", "confidence": 0, "note": "no_ipa"}
            cache.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
            return out
        if not is_source_ipa(ipa):
            last = f"rejected_not_ipa:{ipa}"
            continue
        src = data.get("url") or ""
        if "howtopronounce" in src.lower():
            last = "howtopronounce"
            continue
        try:
            conf = float(data.get("confidence") or 0)
        except (TypeError, ValueError):
            conf = 0.0
        out = {"ipa": ipa, "url": src, "confidence": conf}
        cache.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
        return out
    out = {"ipa": "", "url": "", "confidence": 0, "error": last[:200]}
    cache.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    return out


def names_needing_fix(verify_rows: list[dict], accepted: dict[str, str]) -> list[dict]:
    need = []
    for r in verify_rows:
        name = r["ingredient"]
        if name in accepted:
            # already user-accepted; ensure wav exists
            if has_ipa_wav(name):
                continue
        ipa = accepted.get(name) or strip_ipa(r.get("ipa_used") or r.get("ipa") or "")
        wav = has_ipa_wav(name)
        usable = bool(ipa and is_source_ipa(ipa))
        synth_ok = bool(r.get("ipa_synth")) or (wav is not None and usable)
        if synth_ok and wav is not None:
            continue
        need.append(
            {
                "ingredient": name,
                "old_ipa": r.get("ipa") or "",
                "flags": r.get("flags") or [],
                "ipa_error": (r.get("ipa_error") or "")[:120],
            }
        )
    return need


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    TTS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    LISTEN.mkdir(parents=True, exist_ok=True)

    pron = load_pron()
    verify = json.loads(VERIFY.read_text())
    verify_rows = verify["rows"]
    ipa_rows = {r["ingredient"]: r for r in json.loads(IPA_JSON.read_text())}
    web = load_web_ipas()

    accepted: dict[str, str] = {}
    for rec in ipa_rows.values():
        if rec.get("accepted") and rec["accepted"].get("ipa"):
            accepted[rec["ingredient"]] = strip_ipa(rec["accepted"]["ipa"])

    need = names_needing_fix(verify_rows, accepted)
    print(f"need fix: {len(need)}", flush=True)

    # Resolve IPA
    resolved: list[dict] = []
    to_query: list[dict] = []
    for rec in need:
        name = rec["ingredient"]
        if name in accepted and is_source_ipa(accepted[name]):
            resolved.append(
                {
                    **rec,
                    "ipa": accepted[name],
                    "source": "accepted",
                    "url": "",
                }
            )
            continue
        goods = web.get(name) or []
        if goods:
            resolved.append(
                {
                    **rec,
                    "ipa": goods[0][0],
                    "source": "gemini3-web",
                    "url": goods[0][1],
                }
            )
            continue
        # try cleaner variants of current string
        cur = strip_ipa(rec["old_ipa"])
        if is_source_ipa(cur):
            resolved.append({**rec, "ipa": cur, "source": "existing", "url": ""})
            continue
        to_query.append(rec)

    print(f"  rescued/existing: {len(resolved)}; flash query: {len(to_query)}", flush=True)

    def q(rec: dict) -> dict:
        name = rec["ingredient"]
        resp = (pron.get(name) or {}).get("respelling") or ""
        hit = flash_ipa(name, resp, rec.get("old_ipa") or "")
        return {
            **rec,
            "ipa": hit.get("ipa") or "",
            "source": "flash_query",
            "url": hit.get("url") or "",
            "confidence": hit.get("confidence"),
            "query_error": hit.get("error") or hit.get("note") or "",
        }

    with ThreadPoolExecutor(max_workers=FLASH_WORKERS) as ex:
        futs = [ex.submit(q, r) for r in to_query]
        for fut in as_completed(futs):
            rec = fut.result()
            resolved.append(rec)
            print(
                f"  query {rec['ingredient']}: /{rec['ipa']}/ "
                f"{rec.get('query_error') or rec.get('url','')[:50]}",
                flush=True,
            )

    # Synth IPA wavs
    results = []

    def synth_one(rec: dict) -> dict:
        name = rec["ingredient"]
        s = slug(name)
        out = {**rec, "respelling": (pron.get(name) or {}).get("respelling") or ""}
        ipa = strip_ipa(rec.get("ipa") or "")
        path = TTS / f"{s}.cloud_ipa.wav"
        # Accept user-accepted IPA even if is_source_ipa is strict (e.g. Loargys lˈɔː…).
        allow = bool(ipa) and (is_source_ipa(ipa) or rec.get("source") == "accepted")
        if not allow:
            out["synth_ok"] = False
            out["synth_error"] = "no_usable_ipa"
            return out
        try:
            if path.exists() and path.stat().st_size > 500:
                out["synth_ok"] = True
                out["ipa_used"] = ipa
            else:
                # Prefer existing suspect/verify wav for accepted names
                prior = has_ipa_wav(name)
                if prior is not None and rec.get("source") == "accepted":
                    shutil.copy2(prior, path)
                    out["synth_ok"] = True
                    out["ipa_used"] = ipa
                else:
                    wav, used = synth_cloud_ipa(name, ipa)
                    path.write_bytes(wav)
                    out["synth_ok"] = True
                    out["ipa_used"] = used
            vdest = ROOT / "runs/verify-google-ipa/tts" / f"{s}.ipa.wav"
            vdest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, vdest)
        except Exception as exc:
            out["synth_ok"] = False
            out["synth_error"] = str(exc)[:240]
        return out

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(synth_one, r) for r in resolved]
        for fut in as_completed(futs):
            rec = fut.result()
            results.append(rec)
            st = "ok" if rec.get("synth_ok") else f"FAIL {rec.get('synth_error','')[:80]}"
            print(f"  synth {rec['ingredient']}: /{rec.get('ipa_used') or rec.get('ipa')}/ {st}", flush=True)

    # Update ipa.json for successful synths
    for rec in results:
        if not rec.get("synth_ok"):
            continue
        name = rec["ingredient"]
        used = rec.get("ipa_used") or rec["ipa"]
        row = ipa_rows.get(name)
        if row is None:
            row = {"ingredient": name, "ipa": f"/{used}/", "url": rec.get("url") or ""}
            ipa_rows[name] = row
            # append later via list rewrite
        else:
            row["ipa"] = f"/{used}/"
            if rec.get("url"):
                row["url"] = rec["url"]
            row["fixed_missing"] = {
                "ipa": used,
                "from": rec.get("source"),
                "replaced": strip_ipa(rec.get("old_ipa") or ""),
            }
    IPA_JSON.write_text(
        json.dumps(list(ipa_rows.values()), indent=2, ensure_ascii=False) + "\n"
    )

    # Also patch verify results ipa fields
    by_name = {r["ingredient"]: r for r in results}
    for r in verify_rows:
        fix = by_name.get(r["ingredient"])
        if not fix or not fix.get("synth_ok"):
            continue
        used = fix.get("ipa_used") or fix["ipa"]
        r["ipa"] = f"/{used}/"
        r["ipa_used"] = used
        r["ipa_synth"] = True
        r["ipa_error"] = None
    VERIFY.write_text(json.dumps(verify, indent=2, ensure_ascii=False) + "\n")

    # Spelling arms + score for synth_ok
    scored = []
    ok_rows = [r for r in results if r.get("synth_ok")]
    print(f"building spelling arms for {len(ok_rows)}", flush=True)

    def arms(rec: dict) -> dict:
        name = rec["ingredient"]
        s = slug(name)
        resp = rec.get("respelling") or ""
        ow = TTS / f"{s}.one_word.wav"
        cp = TTS / f"{s}.compact.wav"
        ipa = TTS / f"{s}.cloud_ipa.wav"
        # reuse caches
        for src in (
            HOLD_PROMPT / f"{s}.one_word.wav",
            ROOT / "runs/ipa-suspect-noclip/tts" / f"{s}.one_word.wav",
        ):
            if not ow.exists() and src.exists() and src.stat().st_size > 500:
                shutil.copy2(src, ow)
        for src in (
            COMPACT_TTS / f"{s}.compact.wav",
            ROOT / "runs/ipa-suspect-noclip/tts" / f"{s}.compact.wav",
            ROOT / "runs/gemini31-respell-all/tts" / f"{s}.compact.wav",
        ):
            if not cp.exists() and src.exists() and src.stat().st_size > 500:
                shutil.copy2(src, cp)
        text_ow = resp.strip() if resp.strip() else name
        text_cp = compact_ascii(resp) if resp.strip() else name.lower()
        if not ow.exists() or ow.stat().st_size < 500:
            ow.write_bytes(synth_gemini(text_ow, ONE_WORD))
        if not cp.exists() or cp.stat().st_size < 500:
            # compact: no special prompt — brand/orthography as compact ascii
            ow_prompt_compact = (
                "You are saying a US drug name. Read the letters as a normal English word. "
                "Do not spell letter by letter."
            )
            cp.write_bytes(synth_gemini(text_cp, ow_prompt_compact))
        out = {**rec}
        out["one_word_vs_ipa"] = round(f1(ow, ipa), 4)
        out["compact_vs_ipa"] = round(f1(cp, ipa), 4)
        out["sum_f1"] = round(out["one_word_vs_ipa"] + out["compact_vs_ipa"], 4)
        out["one_word_vs_compact"] = round(f1(ow, cp), 4)
        return out

    # Eager-load WavLM before any F1 (thread-pool import races on transformers).
    _ = f1
    from dose_r.scoring.speech_similarity import extract_frame_embeddings as _efe

    _efe(TTS / f"{slug(ok_rows[0]['ingredient'])}.cloud_ipa.wav")

    for r in ok_rows:
        rec = arms(r)
        scored.append(rec)
        print(
            f"  score {rec['ingredient']} ow={rec['one_word_vs_ipa']} "
            f"cp={rec['compact_vs_ipa']} sum={rec['sum_f1']}",
            flush=True,
        )

    scored.sort(key=lambda r: (r.get("sum_f1") or 9, r.get("one_word_vs_ipa") or 9))
    for i, r in enumerate(scored, 1):
        r["rank"] = i

    failed = [r for r in results if not r.get("synth_ok")]
    summary = {
        "n_need": len(need),
        "n_synth_ok": len(ok_rows),
        "n_failed": len(failed),
        "failed": [
            {
                "ingredient": r["ingredient"],
                "old_ipa": r.get("old_ipa"),
                "ipa": r.get("ipa"),
                "error": r.get("synth_error") or r.get("query_error"),
                "source": r.get("source"),
            }
            for r in failed
        ],
        "scored": [
            {
                "rank": r["rank"],
                "ingredient": r["ingredient"],
                "ipa": r.get("ipa_used") or r.get("ipa"),
                "source": r.get("source"),
                "url": r.get("url"),
                "respelling": r.get("respelling"),
                "one_word_vs_ipa": r["one_word_vs_ipa"],
                "compact_vs_ipa": r["compact_vs_ipa"],
                "sum_f1": r["sum_f1"],
            }
            for r in scored
        ],
    }
    (OUT / "results.json").write_text(
        json.dumps({"summary": summary, "rows": results, "scored": scored}, indent=2, ensure_ascii=False)
        + "\n"
    )

    # Listen page
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Fixed missing IPA vs Gemini 3.1</title>",
        "<style>",
        "body{font:16px/1.4 system-ui;max-width:920px;margin:2rem auto;padding:0 1rem}",
        "section{border:1px solid #ccc;border-radius:8px;padding:1rem 1.2rem;margin:1rem 0}",
        "section.fail{border-color:#c62828;background:#fff5f5}",
        "h1{font-size:1.25rem} h2{font-size:1.05rem;margin:0 0 .3rem}",
        ".ipa{font-family:ui-monospace,monospace;font-size:.95rem}",
        "label{display:block;font-weight:600;margin:.4rem 0 .1rem} audio{width:100%}",
        ".meta{font-size:.9rem;color:#555}",
        "ol.toc{font-size:.95rem}",
        "</style>",
        "<h1>Fixed missing IPA — vs Gemini 3.1 one_word + compact</h1>",
        f"<p>Synth OK {len(scored)}; failed {len(failed)}. "
        "Sorted by lowest (one_word + compact) F1.</p>",
    ]
    if failed:
        parts.append("<h2>Still failed</h2><ul>")
        for r in failed:
            parts.append(
                f"<li>{escape(r['ingredient'])}: "
                f"<span class='ipa'>{escape(str(r.get('ipa') or ''))}</span> "
                f"{escape(str(r.get('synth_error') or r.get('query_error') or ''))}</li>"
            )
        parts.append("</ul>")
    parts.append("<ol class='toc'>")
    for r in scored:
        ss = slug(r["ingredient"])
        parts.append(
            f"<li><a href='#{ss}'>{escape(r['ingredient'])}</a> "
            f"sum {r['sum_f1']:.3f} "
            f"(ow {r['one_word_vs_ipa']:.3f} + cp {r['compact_vs_ipa']:.3f}) "
            f"<span class='ipa'>/{escape(r.get('ipa_used') or r.get('ipa') or '')}/</span></li>"
        )
    parts.append("</ol>")
    for r in scored:
        ss = slug(r["ingredient"])
        parts.append(f"<section id='{ss}'>")
        parts.append(f"<h2>{r['rank']}. {escape(r['ingredient'])}</h2>")
        parts.append(
            f"<p>respelling <span class='ipa'>{escape(r.get('respelling') or '')}</span><br>"
            f"IPA <span class='ipa'>/{escape(r.get('ipa_used') or r.get('ipa') or '')}/</span> "
            f"({escape(str(r.get('source') or ''))})</p>"
        )
        parts.append(
            f"<p class='meta'>sum <b>{r['sum_f1']:.4f}</b> = "
            f"one_word {r['one_word_vs_ipa']:.4f} + compact {r['compact_vs_ipa']:.4f}</p>"
        )
        for key, label in (
            ("one_word", "Gemini 3.1 hyphen + one-word prompt"),
            ("compact", "Gemini 3.1 compact"),
            ("cloud_ipa", "Cloud Standard-C + IPA"),
        ):
            srcp = TTS / f"{ss}.{key}.wav"
            if not srcp.exists():
                continue
            dest = LISTEN / f"{ss}__{key}.wav"
            shutil.copy2(srcp, dest)
            parts.append(f"<label>{label}</label><audio controls src='{dest.name}'></audio>")
        parts.append("</section>")
    html = LISTEN / "index.html"
    html.write_text("\n".join(parts) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False)[:2000], flush=True)
    print(html, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
