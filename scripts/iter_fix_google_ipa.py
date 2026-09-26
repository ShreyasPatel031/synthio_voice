#!/usr/bin/env python
"""Fix lowest-F1 Google IPA first. Re-search the brand, reject generic mixups.

Keep a candidate only if Cloud TTS F1 vs the same human clip rises by >0.005.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
_hf = ROOT / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(_hf))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_hf))
os.environ.setdefault("HF_HUB_CACHE", str(_hf / "hub"))

import requests

from dose_r.references.reference_clips import available_clips
from update_lowest20_html import write_lowest20
from verify_google_ipa_tts import (
    OUT,
    TTS,
    VOICE,
    clincalc_pairs,
    edit,
    f1,
    fold,
    looks_like_ipa,
    strip_ipa,
    token as tts_token,
    try_ipa_synth,
    write_listen,
)

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "project-amer-scs-sandbox")
MODEL = "gemini-3-flash-preview"
IPA_JSON = ROOT / "runs" / "gemini3-ipa-query" / "ipa.json"
RESULTS = OUT / "results.json"
LOG = OUT / "fix-loop.jsonl"
KEEP = 0.005
LIMIT = int(os.environ.get("FIX_LIMIT", "40"))

FIX_PROMPT = """Use Google Search. The drug name is "{name}"{generic_line}.

Run these queries (exact):
{queries}

Return JSON only:
{{"ipa":"/ˈduːpɪksɛnt/","url":"https://...","query_used":"..."}}

Hard rules:
- IPA must be for "{name}" itself, not another drug.
- If a result is clearly the generic or a different word, discard it and try the next query.
- Do not return howtopronounce stress-inside-cluster junk (nˈɜːɾɛk, lˈɪ.p.ɪɾɚ, dˈuː.p.ɪksənt) if Google/Wiktionary shows a cleaner string.
- Copy IPA glyphs exactly. Empty ipa if you cannot find one for THIS name.
"""


def gemini(prompt: str) -> dict:
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
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 512,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    for attempt in range(6):
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
        if resp.status_code == 429:
            time.sleep(min(2**attempt, 20))
            continue
        if resp.status_code != 200:
            return {"ok": False, "error": f"{resp.status_code} {resp.text[:300]}"}
        result = resp.json()
        parts = (result.get("candidates") or [{}])[0].get("content", {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if "text" in p)
        parsed = {}
        try:
            raw = text.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
            parsed = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        except Exception:
            parsed = {}
        return {"ok": True, "parsed": parsed, "text": text}
    return {"ok": False, "error": "retries"}


def queries_for(name: str, generic: str) -> list[str]:
    q = [
        f'{name} ipa pronunciation',
        f'"{name}" IPA pronunciation',
        f"{name} brand name IPA pronunciation",
        f"{name} wiktionary IPA",
    ]
    if generic:
        q.insert(1, f"{name} IPA pronunciation -{generic}")
        q.append(f'how to pronounce {name} not {generic} IPA')
    return q


def is_generic_mixup(ipa: str, generic_ipa: str, generic: str, url: str) -> bool:
    if not ipa:
        return True
    if generic:
        gslug = re.sub(r"[^a-z0-9]+", "", generic.lower())
        blob = (url or "").lower() + ipa.lower()
        if gslug and len(gslug) >= 6 and gslug in re.sub(r"[^a-z0-9]+", "", blob):
            # url/ipa names the generic
            if gslug not in re.sub(r"[^a-z0-9]+", "", ipa):
                return True
        if generic_ipa:
            fa, fb = fold(ipa), fold(generic_ipa)
            if fa and fb:
                n = max(len(fa), len(fb))
                if n and edit(fa, fb) / n <= 0.35:
                    return True
    return False


def load_state() -> tuple[list[dict], dict]:
    data = json.loads(RESULTS.read_text())
    ipa_rows = {r["ingredient"]: r for r in json.loads(IPA_JSON.read_text())}
    return data["rows"], ipa_rows


def save_state(rows: list[dict], ipa_rows: dict) -> None:
    scored = [r for r in rows if r.get("ipa_f1") is not None]
    n = len(scored)
    summary = {
        "voice": VOICE,
        "n_names": len(rows),
        "n_clips": sum(1 for r in rows if r.get("has_clip")),
        "n_scored_ipa": n,
        "n_flagged": sum(1 for r in rows if r.get("flags")),
        "mean_plain": round(
            sum(r["plain"] for r in rows if r.get("plain") is not None)
            / max(1, sum(1 for r in rows if r.get("plain") is not None)),
            4,
        ),
        "mean_ipa": round(sum(r["ipa_f1"] for r in scored) / max(1, n), 4) if n else None,
    }
    RESULTS.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False)
        + "\n"
    )
    IPA_JSON.write_text(
        json.dumps(list(ipa_rows.values()), indent=2, ensure_ascii=False) + "\n"
    )
    write_listen(rows)
    write_lowest20(rows)


def main() -> int:
    rows, ipa_rows = load_state()
    by_name = {r["ingredient"]: r for r in rows}
    pairs = clincalc_pairs()
    clips = available_clips()
    # generic IPA currently on file
    generic_ipa = {
        r["ingredient"].lower(): r.get("ipa") or "" for r in ipa_rows.values()
    }

    todo = [
        r
        for r in rows
        if r.get("has_clip")
        and (
            r.get("ipa_f1") is None
            or r.get("ipa_f1", 9) < 0.70
            or (r.get("delta") or 0) < -0.03
            or any(
                str(f).startswith(("url_other", "ipa_matches_generic", "mab_ipa"))
                for f in (r.get("flags") or [])
            )
        )
    ]
    todo.sort(
        key=lambda r: (
            r.get("ipa_f1") if r.get("ipa_f1") is not None else -1.0,
            r.get("delta") if r.get("delta") is not None else -9.0,
        )
    )
    todo = todo[:LIMIT]
    print(f"fixing {len(todo)} lowest/broken first", flush=True)

    # Seed candidates we already know Google showed for the brand.
    seeds = {
        "Dupixent": ["ˈduːpɪksɛnt"],
        "Nurtec": ["ˈnɝtɛk", "ˈnɜrtɛk", "ˈnɜːrtɛk"],
        "Lipitor": ["ˈlɪpɪtɔr", "lɪˈpɪtɔːr"],
        "Prilosec": ["praɪˈloʊsɛk", "ˈpraɪloʊsɛk"],
        "Advair": ["ˈædvɛr", "ˈædvɛər"],
        "Adquey": ["ˈædki", "ˈædkiː"],
        "Abilify": ["əˈbɪlɪfaɪ", "əˈbɪləfaɪ"],
        "acoramidis": ["ˌækoʊˈræmɪdɪs", "ækoʊˈræmɪdɪs"],
        "Farxiga": ["fɑrˈsiɡə", "fɑːˈsiːɡə"],
        "Toujeo": ["tuːˈʒeɪoʊ", "tuˈdʒeɪoʊ"],
        "Meibo": ["ˈmaɪboʊ", "maɪˈboʊ"],
        "Vraylar": ["ˈvreɪlɑr", "vreɪˈlɑr"],
        "Byetta": ["baɪˈɛtə"],
        "lurasidone": ["lʊˈræsɪdoʊn", "ljʊˈræsɪdoʊn"],
        "tofacitinib": ["ˌtoʊfəˈsɪtɪnɪb", "toʊfəˈsaɪtɪnɪb"],
        "Voranigo": ["vɔːˈrænɪɡoʊ", "vəˈrænɪɡoʊ"],
    }

    tok = tts_token()

    def search_one(rec: dict) -> dict:
        name = rec["ingredient"]
        gen = pairs.get(name) or rec.get("generic_pair") or ""
        qs = queries_for(name, gen)
        generic_line = f' (generic is {gen}; do NOT return {gen} IPA)' if gen else ""
        prompt = FIX_PROMPT.format(
            name=name,
            generic_line=generic_line,
            queries="\n".join(f"- {q}" for q in qs),
        )
        hit = gemini(prompt)
        parsed = (hit.get("parsed") or {}) if hit.get("ok") else {}
        return {
            "ingredient": name,
            "ipa": parsed.get("ipa") or "",
            "url": parsed.get("url") or "",
            "query_used": parsed.get("query_used") or "",
            "error": hit.get("error", ""),
        }

    found = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(search_one, r): r["ingredient"] for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            name = futs[fut]
            try:
                found[name] = fut.result()
            except Exception as exc:
                found[name] = {"ingredient": name, "ipa": "", "error": str(exc)}
            print(
                f"search [{i}/{len(todo)} +{time.time()-t0:.0f}s] {name}: "
                f"{found[name].get('ipa') or found[name].get('error','')}",
                flush=True,
            )

    # Score candidates lowest-first (wavlm once, sequential).
    print("scoring replacements vs human...", flush=True)
    kept = 0
    for rec in todo:
        name = rec["ingredient"]
        clip = clips.get(name)
        if clip is None:
            continue
        human = clip.path.read_bytes()
        gen = pairs.get(name) or rec.get("generic_pair") or ""
        g_ipa = ""
        if gen:
            g_ipa = generic_ipa.get(gen.lower(), "")
        cands = []
        hit = found.get(name) or {}
        if hit.get("ipa"):
            cands.append((hit["ipa"], hit.get("url") or "", hit.get("query_used") or "gemini"))
        for s in seeds.get(name, []):
            cands.append((s, "seed/google", "seed"))
        # unique by folded IPA
        uniq = []
        seen = set()
        for ipa, url, src in cands:
            if not looks_like_ipa(ipa):
                continue
            if is_generic_mixup(ipa, g_ipa, gen, url):
                print(f"  reject mixup {name} {ipa} ({url})", flush=True)
                continue
            key = fold(ipa)
            if not key or key in seen:
                continue
            if key == fold(rec.get("ipa") or ""):
                continue
            seen.add(key)
            uniq.append((ipa, url, src))
        base = rec.get("ipa_f1")
        if base is None:
            base = rec.get("plain") or 0.0
        best = None
        for ipa, url, src in uniq:
            try:
                wav, used = try_ipa_synth(tok, name, ipa)
            except Exception as exc:
                print(f"  cloud reject {name} {ipa}: {exc}", flush=True)
                continue
            score = round(f1(wav, human), 4)
            print(
                f"  try {name} {ipa} -> {score:.3f} (base {base:.3f}) {src}",
                flush=True,
            )
            if score > base + KEEP and (best is None or score > best[0]):
                best = (score, ipa, used, wav, url, src)
        if best:
            score, ipa, used, wav, url, src = best
            slug = name.lower().replace(" ", "_")
            TTS.mkdir(parents=True, exist_ok=True)
            (TTS / f"{slug}.ipa.wav").write_bytes(wav)
            (OUT / f"{slug}__ipa.wav").write_bytes(wav)
            rec["ipa"] = ipa if ipa.startswith("/") or ipa.startswith("[") else f"/{ipa}/"
            rec["ipa_used"] = used
            rec["url"] = url or rec.get("url")
            rec["ipa_f1"] = score
            rec["delta"] = round(score - (rec.get("plain") or 0), 4)
            rec["ipa_synth"] = True
            rec["has_ipa_wav"] = True
            rec["flags"] = [
                f
                for f in (rec.get("flags") or [])
                if not str(f).startswith(
                    ("url_other", "ipa_matches_generic", "mab_ipa", "ipa_worse")
                )
            ]
            rec["fix_src"] = src
            ipa_rows[name] = {
                "ingredient": name,
                "ipa": rec["ipa"],
                "url": rec["url"],
                "queries": [src],
            }
            kept += 1
            print(f"KEEP {name} {rec['ipa']} F1 {base:.3f}->{score:.3f}", flush=True)
            LOG.open("a").write(
                json.dumps(
                    {
                        "ingredient": name,
                        "old_f1": base,
                        "new_f1": score,
                        "ipa": rec["ipa"],
                        "src": src,
                    }
                )
                + "\n"
            )
            save_state(list(by_name.values()), ipa_rows)
            write_lowest20(list(by_name.values()))
        else:
            print(f"NO KEEP {name}", flush=True)

    save_state(list(by_name.values()), ipa_rows)
    print(f"kept {kept}/{len(todo)} -> {RESULTS}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
