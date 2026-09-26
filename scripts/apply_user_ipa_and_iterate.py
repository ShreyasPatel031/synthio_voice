#!/usr/bin/env python
"""Apply hand-checked IPA (per-word Google) and NCI clips, then iterate lows.

Keep a candidate only if wavlm F1 vs the (possibly swapped) human clip rises.
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

from dose_r.references.reference_clips import available_clips
from dose_r.references.tts_pronunciation import page_is_for_word
from update_lowest20_html import write_lowest20
from verify_google_ipa_tts import (
    OUT,
    TTS,
    custom_pronunciation,
    f1,
    fold,
    looks_like_ipa,
    synth,
    token as tts_token,
    try_ipa_synth,
    unwrap_ipa,
    write_listen,
)

RESULTS = OUT / "results.json"
IPA_JSON = ROOT / "runs" / "gemini3-ipa-query" / "ipa.json"
KEEP = 0.005
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "project-amer-scs-sandbox")
# Cheaper than gemini-3-flash-preview. Same Google Search tool, thinking off.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_LOCATION = os.environ.get("GEMINI_LOCATION", "us-central1")


def same_stress_ipa(a: str, b: str) -> bool:
    """Skip only if phones *and* stress match. fold() strips ˈ and hid Winrevair."""

    def norm(s: str) -> str:
        s = (s or "").strip().strip("/[]() ")
        s = s.replace(".", "").replace(" ", "")
        return s.replace("ɹ", "r").replace("ɡ", "g")

    na, nb = norm(a), norm(b)
    return bool(na) and na == nb

# User-supplied IPA. Multi-word = one IPA per word (Google separately).
USER = {
    "fluticasone propionate": ["fluːˈtɪkəˌsoʊn", "ˈproʊpiəneɪt"],
    "obecabtagene autoleucel": ["ˌoʊbəˈkæbtədʒiːn", "ˌɔːtoʊˈluːsəl"],
    "Farxiga": ["fɑːrˈziːɡə"],
    "Meibo": ["ˈmaɪboʊ"],
    "Aucatzyl": ["ɔːˈkætzɪl"],
    "upadacitinib": ["juːˌpædəˈsɪtɪnɪb"],
    "Byetta": ["baɪˈɛtə"],
    "lurasidone": ["lʊˈræsɪˌdoʊn"],
    "tofacitinib": ["toʊfəˈsɪtɪnɪb"],
    "Zevaskyn": ["ˈziːvəskɪn"],
    "Zaiidra": ["ˈzaɪdrə"],
    "dulaglutide": ["duːləˈɡluːtaɪd"],
    "Biktarvy": ["bɪkˈtɑːrvi"],
    "Winrevair": ["ˈwɪn.ɹəˌvɛəɹ"],
    "valsartan": ["vælˈsɑɹ.tən"],
    "quetiapine": ["kwɪˈtaɪ.əˌpin"],
}

# Human clip is stem-only or otherwise not the spoken name. Don't burn Google
# budget on these; they saturate as clip-wrong.
CLIP_SKIP = {"formoterol fumarate dihydrate"}

WORD_PROMPT = """Use Google Search with this exact query and no other query:

{query}

Quote each IPA transcription Google displayed for "{word}" in slashes, glyphs
unchanged (stress, dots, vowels). If Google showed one, quote one. Do not
invent IPA. Do not turn a hyphenated respelling (co-BEN-fee, DU-pix-ent)
into IPA. Name the page each string appeared on.
"""

# JSON-only answers drop groundingMetadata (see gemini_grounded.py). Pull IPA
# off cited snippets and the pages Google Search actually retrieved.
_SLASH_IPA = re.compile(r"/([^/\n]{2,48})/")
_SPAN_IPA = re.compile(
    r'class="[^"]*\bIPA\b[^"]*"[^>]*>([^<]{2,48})', re.I
)
_STRESS_RUN = re.compile(r"([^\s/<]{0,8}[ˈˌ][^\s/<,]{2,40})")
_SKIP_HOST = (
    "howtopronounce.com",
    "forvo.com",
    "youtube.com",
    "youtu.be",
    "translate.google.",
    "medspeakpro.org",
    "ipa-reader.com",
    "leskoff.com",
)


def _dedupe_ipa(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        ipa = str(item).strip().strip("/[]() ")
        if not looks_like_ipa(ipa):
            continue
        nk = ipa.replace(".", "").replace(" ", "")
        if nk in seen:
            continue
        seen.add(nk)
        out.append(ipa)
    nks = [x.replace(".", "").replace(" ", "") for x in out]
    full: list[str] = []
    for ipa, nk in zip(out, nks):
        if any(nk != other and nk in other for other in nks if len(other) > len(nk)):
            continue
        full.append(ipa)
    return full[:3]


def collect_ipa_from_text(text: str) -> list[str]:
    found: list[str] = []
    blob = text or ""
    for rx in (_SLASH_IPA, _SPAN_IPA, _STRESS_RUN):
        found.extend(m.group(1) for m in rx.finditer(blob))
    return found


_SKIP_PATH = (
    "/wiki/voiced_",
    "/wiki/voiceless_",
    "/wiki/help:ipa",
    "/wiki/international_phonetic",
    "/wiki/palatal",
    "/wiki/approximant",
)


def _skip_host(url: str) -> bool:
    u = (url or "").lower()
    if any(h in u for h in _SKIP_HOST):
        return True
    return any(p in u for p in _SKIP_PATH)


def _fetch_html(url: str) -> str | None:
    import urllib.request

    from dose_r.references.gemini_grounded import TIMEOUT, UA, _cached, _store

    cache_key = f"html::{url}"
    hit = _cached(cache_key)
    if hit is not None and hit.get("html"):
        return hit["html"]
    try:
        req = urllib.request.Request(url, headers=UA)
        html = urllib.request.urlopen(req, timeout=TIMEOUT).read().decode(
            "utf-8", "ignore"
        )
    except Exception:
        return None
    if html:
        _store(cache_key, {"html": html})
    return html


def _resolve_url(uri: str) -> str:
    import requests

    from dose_r.references.gemini_grounded import TIMEOUT, UA

    try:
        resp = requests.get(uri, headers=UA, timeout=TIMEOUT, allow_redirects=True)
        return resp.url or uri
    except Exception:
        return uri


def extract_google_ipa(result: dict, word: str = "") -> tuple[list[str], str]:
    """IPA that appeared in a citation or on a retrieved page, not authored JSON."""
    cand = (result.get("candidates") or [{}])[0]
    parts = (cand.get("content") or {}).get("parts") or []
    model_text = "".join(p.get("text", "") for p in parts if "text" in p)
    gm = cand.get("groundingMetadata") or {}
    page_blobs: list[str] = []
    urls: list[str] = []
    for chunk in gm.get("groundingChunks") or []:
        web = chunk.get("web") or {}
        uri = web.get("uri") or ""
        title = (web.get("title") or "") + " " + (web.get("domain") or "")
        if not uri:
            continue
        if _skip_host(uri) or _skip_host(title):
            continue
        resolved = _resolve_url(uri)
        if _skip_host(resolved):
            continue
        if word and not page_is_for_word(word, resolved):
            continue
        urls.append(resolved)
        html = _fetch_html(resolved)
        if html:
            page_blobs.append(html)
    cited = _dedupe_ipa(
        [x for blob in page_blobs for x in collect_ipa_from_text(blob)]
    )
    model = _dedupe_ipa(collect_ipa_from_text(model_text))
    if not urls:
        return [], ""
    if cited:
        cited_fold = {fold(x) for x in cited}
        extra = [m for m in model if fold(m) in cited_fold]
        variants = _dedupe_ipa(cited + extra)
    else:
        variants = model
    return variants, urls[0]


def gemini_word(word: str, query: str | None = None) -> dict:
    import requests
    import google.auth
    from google.auth.transport.requests import Request

    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    creds.refresh(Request())
    url = (
        f"https://{GEMINI_LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT}"
        f"/locations/{GEMINI_LOCATION}/publishers/google/models/{GEMINI_MODEL}:generateContent"
    )
    q = query or f"{word} ipa pronunciation"
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": WORD_PROMPT.format(word=word, query=q)}],
            }
        ],
        "tools": [{"google_search": {}}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 1024,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    last_err = ""
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
            last_err = f"{resp.status_code} {resp.text[:240]}"
            print(f"  gemini fail {word}: {last_err}", flush=True)
            return {}
        variants, src = extract_google_ipa(resp.json(), word=word)
        return {"ipa": variants, "url": src}
    print(f"  gemini retries {word}: {last_err}", flush=True)
    return {}


def parse_ipa_variants(hit: dict) -> list[str]:
    raw = (hit or {}).get("ipa")
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, str) and raw.strip():
        items = [raw]
    else:
        return []
    return _dedupe_ipa([str(x) for x in items])


def words_of(name: str) -> list[str]:
    from dose_r.references.tts_pronunciation import spoken_parts

    return spoken_parts(name)


def synth_per_word(tok: str, name: str, ipas: list[str]) -> tuple[bytes, str]:
    from dose_r.references.tts_pronunciation import (
        custom_pronunciations_for_parts,
        spoken_parts,
        spoken_text,
    )

    tokens = spoken_parts(name)
    text = spoken_text(name)
    ipas = [unwrap_ipa(x) for x in ipas if unwrap_ipa(x)]
    if len(ipas) == 1 and len(tokens) == 1:
        return try_ipa_synth(tok, text, ipas[0])
    if len(ipas) >= len(tokens):
        pairs = list(zip(tokens, ipas))
    else:
        pairs = list(zip(tokens, ipas + [ipas[-1]] * len(tokens))) if ipas else []
    used_pairs = [(phrase, ipa) for phrase, ipa in pairs]
    wav = synth(
        tok, text=text, pronunciations=custom_pronunciations_for_parts(used_pairs)
    )
    return wav, " ".join(f"{p}={i}" for p, i in used_pairs)


def human_bytes(name: str, clips: dict) -> tuple[bytes, str, str]:
    clip = clips[name]
    return clip.path.read_bytes(), clip.path.suffix, clip.source


def slug_file(name: str) -> str:
    return name.lower().replace(" ", "_")


def save(rows: list[dict], ipa_rows: dict) -> None:
    scored = [r for r in rows if r.get("ipa_f1") is not None]
    n = len(scored)
    summary = {
        "n_names": len(rows),
        "n_scored_ipa": n,
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


def try_keep(tok, rec, clips, ipas, src, ipa_rows, by_name) -> bool:
    name = rec["ingredient"]
    if name not in clips:
        return False
    human, ext, src_h = human_bytes(name, clips)
    slug = slug_file(name)
    dest_h = OUT / f"{slug}__human{ext}"
    dest_h.write_bytes(human)
    rec["human_ext"] = ext
    rec["clip_source"] = src_h
    try:
        wav, used = synth_per_word(tok, name, ipas)
    except Exception as exc:
        print(f"  cloud reject {name} {ipas}: {exc}", flush=True)
        return False
    score = round(f1(wav, human), 4)
    # If we swapped the human clip, also rescore plain against the new human.
    p_plain = TTS / f"{slug}.plain.wav"
    if p_plain.exists():
        rec["plain"] = round(f1(p_plain.read_bytes(), human), 4)
    base = rec.get("ipa_f1")
    if base is None:
        base = rec.get("plain") or 0.0
    print(f"  {name} {ipas} -> {score:.3f} (base {base:.3f}) {src} human={src_h}", flush=True)
    if score < base + KEEP:
        print(f"  NO KEEP {name}", flush=True)
        return False
    (TTS / f"{slug}.ipa.wav").write_bytes(wav)
    (OUT / f"{slug}__ipa.wav").write_bytes(wav)
    joined = " ".join(f"/{p}/" for p in ipas)
    rec["ipa"] = joined if len(ipas) > 1 else f"/{ipas[0]}/"
    rec["ipa_used"] = used
    rec["ipa_f1"] = score
    rec["delta"] = round(score - (rec.get("plain") or 0), 4)
    rec["ipa_synth"] = True
    rec["has_ipa_wav"] = True
    rec["fix_src"] = src
    rec["flags"] = [
        f
        for f in (rec.get("flags") or [])
        if f not in {"ipa_worse_than_plain", "howtopronounce"}
    ]
    if rec["delta"] <= -0.03:
        rec["flags"].append("ipa_worse_than_plain")
    ipa_rows[name] = {
        "ingredient": name,
        "ipa": rec["ipa"],
        "url": rec.get("url") or src,
        "queries": [src],
    }
    print(f"KEEP {name} {rec['ipa']} {base:.3f}->{score:.3f} human={src_h}", flush=True)
    save(list(by_name.values()), ipa_rows)
    return True


def try_keep_best(tok, rec, clips, candidates, src, ipa_rows, by_name) -> bool:
    """Synth every IPA variant; keep the single best if F1 rises."""
    name = rec["ingredient"]
    if name not in clips or not candidates:
        return False
    human, ext, src_h = human_bytes(name, clips)
    slug = slug_file(name)
    dest_h = OUT / f"{slug}__human{ext}"
    dest_h.write_bytes(human)
    rec["human_ext"] = ext
    rec["clip_source"] = src_h
    p_plain = TTS / f"{slug}.plain.wav"
    if p_plain.exists():
        rec["plain"] = round(f1(p_plain.read_bytes(), human), 4)
    base = rec.get("ipa_f1")
    if base is None:
        base = rec.get("plain") or 0.0
    best = None
    for ipas in candidates:
        if same_stress_ipa(" ".join(ipas), rec.get("ipa") or ""):
            print(f"  skip same stress {name} {ipas}", flush=True)
            continue
        try:
            wav, used = synth_per_word(tok, name, ipas)
        except Exception as exc:
            print(f"  cloud reject {name} {ipas}: {exc}", flush=True)
            continue
        score = round(f1(wav, human), 4)
        print(
            f"  try {name} {ipas} -> {score:.3f} (base {base:.3f}) {src}",
            flush=True,
        )
        if best is None or score > best[0]:
            best = (score, ipas, used, wav)
    if best is None or best[0] < base + KEEP:
        print(f"  NO KEEP {name}", flush=True)
        return False
    score, ipas, used, wav = best
    (TTS / f"{slug}.ipa.wav").write_bytes(wav)
    (OUT / f"{slug}__ipa.wav").write_bytes(wav)
    joined = " ".join(f"/{p}/" for p in ipas)
    rec["ipa"] = joined if len(ipas) > 1 else f"/{ipas[0]}/"
    rec["ipa_used"] = used
    rec["ipa_f1"] = score
    rec["delta"] = round(score - (rec.get("plain") or 0), 4)
    rec["ipa_synth"] = True
    rec["has_ipa_wav"] = True
    rec["fix_src"] = src
    rec["flags"] = [
        f
        for f in (rec.get("flags") or [])
        if f not in {"ipa_worse_than_plain", "howtopronounce"}
    ]
    if rec["delta"] <= -0.03:
        rec["flags"].append("ipa_worse_than_plain")
    ipa_rows[name] = {
        "ingredient": name,
        "ipa": rec["ipa"],
        "url": rec.get("url") or src,
        "queries": [src],
    }
    print(f"KEEP {name} {rec['ipa']} {base:.3f}->{score:.3f} human={src_h}", flush=True)
    save(list(by_name.values()), ipa_rows)
    return True


def rescore_against_gold(rows: list[dict], clips: dict, ipa_rows: dict) -> int:
    """If Path 2 gold changed (NCI now preferred), rescore existing wavs."""
    n = 0
    for rec in rows:
        name = rec["ingredient"]
        clip = clips.get(name)
        if clip is None:
            continue
        if rec.get("clip_source") == clip.source and rec.get("has_clip"):
            continue
        human = clip.path.read_bytes()
        slug = slug_file(name)
        dest = OUT / f"{slug}__human{clip.path.suffix}"
        dest.write_bytes(human)
        rec["has_clip"] = True
        rec["clip_source"] = clip.source
        rec["human_ext"] = clip.path.suffix
        p_plain = TTS / f"{slug}.plain.wav"
        p_ipa = TTS / f"{slug}.ipa.wav"
        if p_plain.exists():
            rec["plain"] = round(f1(p_plain.read_bytes(), human), 4)
        if p_ipa.exists() and rec.get("ipa"):
            rec["ipa_f1"] = round(f1(p_ipa.read_bytes(), human), 4)
            rec["delta"] = round(rec["ipa_f1"] - (rec.get("plain") or 0), 4)
        print(
            f"gold {name}: human={clip.source} plain={rec.get('plain')} "
            f"ipa={rec.get('ipa_f1')}",
            flush=True,
        )
        n += 1
    if n:
        save(rows, ipa_rows)
    return n


def main() -> int:
    data = json.loads(RESULTS.read_text())
    rows = data["rows"]
    by_name = {r["ingredient"]: r for r in rows}
    ipa_rows = {r["ingredient"]: r for r in json.loads(IPA_JSON.read_text())}
    clips = available_clips()
    tok = tts_token()

    print("=== user IPA ===", flush=True)
    if os.environ.get("GOOGLE_ONLY"):
        print("skip user IPA (GOOGLE_ONLY)", flush=True)
    else:
        for name, ipas in USER.items():
            rec = by_name.get(name)
            if not rec:
                print("missing", name)
                continue
            if same_stress_ipa(" ".join(ipas), rec.get("ipa") or ""):
                print(f"skip same IPA {name}", flush=True)
                continue
            try_keep(tok, rec, clips, ipas, "user-google", ipa_rows, by_name)
        rec = by_name.get("Zaiidra")
        if rec and fold(rec.get("ipa") or "") != fold("zaɪdrə"):
            try_keep(tok, rec, clips, ["zaɪdrə"], "user-google", ipa_rows, by_name)

    def search_name(rec):
        hit = gemini_word(rec["ingredient"])
        variants = parse_ipa_variants(hit)
        return rec["ingredient"], variants, hit.get("url") or ""

    # Re-Google every name where IPA still loses to plain. Three IPA variants
    # per name; keep the best. Three passes.
    WORSE = -0.005
    for pass_i in range(1, 4):
        lose = [
            r
            for r in rows
            if r.get("has_clip")
            and r["ingredient"] not in CLIP_SKIP
            and r.get("ipa_f1") is not None
            and r.get("plain") is not None
            and r["ipa_f1"] < r["plain"] + WORSE
        ]
        lose.sort(key=lambda r: r["ipa_f1"] - r["plain"])
        print(
            f"=== 3-variant pass {pass_i}/3: {len(lose)} IPA-worse-than-plain ===",
            flush=True,
        )
        if not lose:
            break
        found_map = {}
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = [ex.submit(search_name, r) for r in lose]
            for fut in as_completed(futs):
                name, variants, url = fut.result()
                found_map[name] = (variants, url)
                print(name, variants, url, flush=True)
        kept_this = 0
        for rec in lose:
            variants, url = found_map.get(rec["ingredient"]) or ([], "")
            if not variants:
                print(f"skip empty {rec['ingredient']}", flush=True)
                continue
            if url:
                rec["url"] = url
            cands = [[v] for v in variants]
            if try_keep_best(
                tok, rec, clips, cands, "re-google-3var", ipa_rows, by_name
            ):
                kept_this += 1
        print(f"pass {pass_i} kept {kept_this} / {len(lose)} still-worse", flush=True)

    # Flag likely clip-wrong vs IPA-wrong on remaining lows.
    for rec in rows:
        if rec.get("ipa_f1") is None or rec["ipa_f1"] >= 0.58:
            continue
        flags = rec.setdefault("flags", [])
        if rec.get("plain") and rec["ipa_f1"] - rec["plain"] < 0.03:
            if "clip_likely_wrong" not in flags:
                flags.append("clip_likely_wrong")
    save(rows, ipa_rows)
    print("done", RESULTS, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
