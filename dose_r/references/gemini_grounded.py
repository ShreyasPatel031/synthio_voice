"""Gemini + Google Search grounding as a retrieval tool, never as an author.

This is not "ask an LLM to guess a pronunciation" -- that is banned
everywhere in this pipeline, because a plausible model guess is
indistinguishable from a real source once it is written to a file. This is
retrieval: Gemini is asked to find a pronunciation a real webpage already
states, and every claim it returns is independently re-fetched and checked
against the actual page content before being trusted. A claim that does not
verify is discarded, not downgraded -- there is no confidence tier for "the
model said so."

Two things learned by testing directly against the API before writing this:

1. Constraining the response to strict JSON silences `groundingMetadata`
   entirely. The request must ask for natural-language prose with citations;
   structure is recovered afterwards from `groundingChunks`.
2. `groundingChunks[].web.uri` is a `vertexaisearch.cloud.google.com/
   grounding-api-redirect/...` link, not the real page. It has to be
   followed (redirects), and the destination is what gets re-fetched for
   verification.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import sources  # for _load_dotenv's side effect and UA reuse

CACHE = Path(__file__).resolve().parent / ".cache"
LOCATION = "us-central1"
MODEL = "gemini-2.5-flash"
UA = {"User-Agent": "Mozilla/5.0 (compatible; DOSE-R research benchmark)"}
TIMEOUT = 25


def _cache_path(key: str) -> Path:
    safe = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    if len(safe) > 120:
        import hashlib

        safe = safe[:60] + "_" + hashlib.sha256(key.encode()).hexdigest()[:16]
    return CACHE / f"gemini_{safe}.json"


def _cached(key: str):
    path = _cache_path(key)
    if path.exists():
        return json.loads(path.read_text())
    return None


def _store(key: str, value) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    _cache_path(key).write_text(json.dumps(value))


def _access_token() -> str:
    import base64
    import binascii

    blob = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON", "")
    if not blob:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS_JSON is not set")
    try:
        creds = json.loads(base64.b64decode(blob))
    except (binascii.Error, ValueError):
        creds = json.loads(blob)

    import subprocess

    script = f"""
const crypto = require('crypto');
const creds = {json.dumps(creds)};
const now = Math.floor(Date.now()/1000);
const h = Buffer.from(JSON.stringify({{alg:'RS256',typ:'JWT'}})).toString('base64url');
const p = Buffer.from(JSON.stringify({{iss:creds.client_email,scope:'https://www.googleapis.com/auth/cloud-platform',aud:'https://oauth2.googleapis.com/token',exp:now+3600,iat:now}})).toString('base64url');
const s = crypto.createSign('RSA-SHA256'); s.update(`${{h}}.${{p}}`);
console.log(`${{h}}.${{p}}.${{s.sign(creds.private_key,'base64url')}}`);
"""
    jwt = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout.strip()

    body = (
        "grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer"
        f"&assertion={urllib.parse.quote(jwt)}"
    ).encode()
    req = urllib.request.Request(
        "https://oauth2.googleapis.com/token",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=TIMEOUT).read())
    token = resp.get("access_token")
    if not token:
        raise RuntimeError(f"token exchange failed: {resp}")
    return token


def _project_id() -> str:
    import base64

    blob = os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
    try:
        creds = json.loads(base64.b64decode(blob))
    except Exception:
        creds = json.loads(blob)
    return os.environ.get("GOOGLE_CLOUD_PROJECT") or creds["project_id"]


@dataclass(frozen=True)
class GroundingChunk:
    redirect_uri: str
    title: str
    domain: str
    resolved_url: str | None = None


@dataclass(frozen=True)
class GroundedAnswer:
    text: str
    chunks: list[GroundingChunk]
    supports: list[dict]  # {segment_text, chunk_indices}


def _generate(name: str, retries: int = 3) -> dict | None:
    cache_key = f"raw::{name}"
    hit = _cached(cache_key)
    if hit is not None:
        return hit or None

    token = _access_token()
    project = _project_id()
    prompt = (
        f'What is the phonetic pronunciation of the drug "{name}"? '
        "Search the web for it and tell me which page you found it on."
    )
    body = json.dumps(
        {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "tools": [{"google_search": {}}],
        }
    ).encode()

    url = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{project}"
        f"/locations/{LOCATION}/publishers/google/models/{MODEL}:generateContent"
    )

    for attempt in range(retries):
        req = urllib.request.Request(
            url, data=body, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        )
        try:
            raw = urllib.request.urlopen(req, timeout=TIMEOUT).read()
            result = json.loads(raw)
            _store(cache_key, result)
            return result
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                time.sleep(2**attempt * 2)
                continue
            _store(cache_key, {})
            return None
        except Exception:
            _store(cache_key, {})
            return None
    return None


def _resolve_redirect(uri: str) -> str | None:
    cache_key = f"redirect::{uri}"
    hit = _cached(cache_key)
    if hit is not None:
        return hit.get("resolved") or None

    try:
        req = urllib.request.Request(uri, headers=UA)
        resp = urllib.request.urlopen(req, timeout=TIMEOUT)
        resolved = resp.geturl()
    except Exception:
        resolved = None
    _store(cache_key, {"resolved": resolved})
    return resolved


def _fetch_page_text(url: str) -> str | None:
    cache_key = f"page::{url}"
    hit = _cached(cache_key)
    if hit is not None:
        return hit.get("text") or None

    try:
        req = urllib.request.Request(url, headers=UA)
        html = urllib.request.urlopen(req, timeout=TIMEOUT).read().decode("utf-8", "ignore")
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text)
    except Exception:
        text = None
    _store(cache_key, {"text": text})
    return text


def grounded_answer(name: str) -> GroundedAnswer | None:
    result = _generate(name)
    if not result:
        return None

    try:
        candidate = result["candidates"][0]
        text = candidate["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        return None

    gm = candidate.get("groundingMetadata", {})
    chunks = [
        GroundingChunk(
            redirect_uri=c["web"]["uri"],
            title=c["web"].get("title", ""),
            domain=c["web"].get("domain", ""),
        )
        for c in gm.get("groundingChunks", [])
        if "web" in c
    ]
    supports = [
        {
            "segment_text": s["segment"]["text"],
            "chunk_indices": s.get("groundingChunkIndices", []),
        }
        for s in gm.get("groundingSupports", [])
    ]
    return GroundedAnswer(text=text, chunks=chunks, supports=supports)


_RESPELL_PATTERN = re.compile(
    r'"([A-Za-z]{2,}(?:[-–][A-Za-z]{2,}){1,7})"|'  # quoted "bik-TEG-ra-vir"
    r"\b([A-Za-z]{2,}(?:[-–][A-Za-z]{2,}){1,7})\b"  # bare bik-TEG-ra-vir
)


def _extract_respelling(segment_text: str) -> str | None:
    """Pull a hyphenated respelling candidate out of a claim segment.

    Stress is not always marked by capitalization ("ad-kee" is as valid a
    respelling as "bik-TEG-ra-vir") -- the only structural requirement is
    that it is spelled as hyphenated syllables, not a real English word.
    """
    for m in _RESPELL_PATTERN.finditer(segment_text):
        candidate = m.group(1) or m.group(2)
        if candidate and "-" in candidate.replace("–", "-"):
            return candidate.replace("–", "-")
    return None


def _normalize(s: str) -> str:
    return re.sub(r"[\s–-]+", "", s.lower())


@dataclass(frozen=True)
class VerifiedClaim:
    respelling: str
    source_url: str
    source_domain: str


def verified_claims(name: str) -> list[VerifiedClaim]:
    """Every grounded claim for `name` that independently verifies against
    its own cited page's real, fetched content. An unverifiable claim is
    dropped silently here; the caller decides what "no claims" means."""
    answer = grounded_answer(name)
    if not answer or not answer.chunks:
        return []

    resolved = {i: _resolve_redirect(c.redirect_uri) for i, c in enumerate(answer.chunks)}

    out: list[VerifiedClaim] = []
    seen: set[tuple[str, str]] = set()
    for support in answer.supports:
        respelling = _extract_respelling(support["segment_text"])
        if not respelling:
            continue
        target = _normalize(respelling)

        for idx in support["chunk_indices"]:
            url = resolved.get(idx)
            if not url:
                continue
            page_text = _fetch_page_text(url)
            if not page_text:
                continue
            if target not in _normalize(page_text):
                continue
            key = (target, url)
            if key in seen:
                continue
            seen.add(key)
            out.append(VerifiedClaim(respelling, url, answer.chunks[idx].domain))
    return out
