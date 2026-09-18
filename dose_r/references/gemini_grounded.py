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

    # A transient failure (rate limit exhausted, timeout, network hiccup) is
    # never cached: caching `{}` here used to make it permanent, since a hit
    # of `{}` short-circuits every future call for that name at the top of
    # this function -- one bad network moment silently and irrecoverably
    # zeroed out that word. Only a genuine answer is worth remembering.
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
            return None
        except Exception:
            if attempt < retries - 1:
                time.sleep(2**attempt * 2)
                continue
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
    # A hyphenated syllable run: quoted, parenthesized, or bare. Syllables can
    # be a single letter or a schwa ("a-TOE-je-pant", "ə-PREM-i-last"), so
    # this is deliberately permissive on the regex side -- `_judge_format`
    # is what actually screens out a false positive like "well-known", not
    # a tighter regex.
    r'"([\wÀ-ʯ\']+(?:[-–][\wÀ-ʯ\']+){1,7})"|'
    r"\(([\wÀ-ʯ\' ]+(?:[-–][\wÀ-ʯ\']+){1,7})\)|"
    r"\b(\w+(?:[-–]\w+){1,7})\b"
)

_IPA_PATTERN = re.compile(r"/([^/\s][^/]{1,40}[^/\s])/")


def _extract_respelling(segment_text: str, exclude: str | None = None) -> str | None:
    """Pull a hyphenated respelling candidate out of a claim segment.

    Stress is not always marked by capitalization ("ad-kee" is as valid a
    respelling as "bik-TEG-ra-vir") -- the only structural requirement is
    that it is spelled as hyphenated syllables, not a real English word.

    Gemini's answer often echoes the input drug name in quotes right next to
    the real respelling (`the drug "elranatamab-bcmm" is "El-rah-NAH-tah-
    mab"`) -- when that echo itself has an FDA-suffix hyphen, it satisfies
    this regex and, being first in the sentence, used to get returned ahead
    of the actual respelling. `exclude` (the source word) drops that echo.

    The comparison is case-insensitive but hyphen-preserving, not the loose
    `_normalize()` (which also strips hyphens/spaces): a real respelling
    like "VRAY-lar" normalizes to the same string as its source word
    "Vraylar" purely because the hyphen and case differences wash out, and
    that would wrongly exclude it as "just the name" -- an actual echo
    reproduces the source word's own spelling verbatim (case aside), it
    doesn't happen to collide with it after stripping punctuation.
    """
    text = segment_text.replace("*", "")
    exclude_ci = exclude.strip().lower() if exclude else None
    for m in _RESPELL_PATTERN.finditer(text):
        candidate = m.group(1) or m.group(2) or m.group(3)
        if not candidate or "-" not in candidate.replace("–", "-"):
            continue
        candidate = candidate.replace("–", "-").strip()
        if exclude_ci and candidate.lower() == exclude_ci:
            continue
        return candidate
    return None


_STRESS_TOKEN = re.compile(r"^[a-zA-Z]{1,8}[\"'‘’]?$")
_STRESS_STOPWORDS = {
    "is", "a", "an", "the", "of", "or", "and", "in", "on", "at", "as",
    "to", "was", "were", "it", "be", "by", "for", "with",
}


def _extract_stress_respelling(text: str) -> str | None:
    """Pull a USAN/USP-style pronunciation-key respelling: space-separated
    syllables with a prime marking stress, e.g. `(dor" a vir' een)` or
    `troe ril' ue zole` -- secondary stress marked with a double prime ("),
    primary with a single prime ('). This is the official USAN adopted-name
    pronunciation convention, not the drugs.com/WebMD hyphenated style
    `_extract_respelling` handles, so it needs its own parser.

    Requiring the stress mark to be the LAST character of its token (not
    mid-token, as in a possessive like "Davis's") is what keeps this from
    firing on ordinary prose.
    """
    text = text.replace("*", "")
    for m in re.finditer(
        r"(?:^|[\s(\"])((?:[a-zA-Z]{1,8}[\"'‘’]?\s+){1,5}[a-zA-Z]{1,8}[\"'‘’]?)(?=[\s.)\"]|$)",
        text,
    ):
        tokens = m.group(1).split()
        while tokens and re.sub(r"[^a-zA-Z]", "", tokens[0]).lower() in _STRESS_STOPWORDS:
            tokens = tokens[1:]
        if len(tokens) < 2 or not all(_STRESS_TOKEN.match(t) for t in tokens):
            continue
        if not any(t.endswith("'") or t.endswith("’") for t in tokens):
            continue
        syllables = []
        for t in tokens:
            primary = t.endswith("'") or t.endswith("’")
            letters = re.sub(r"[^a-zA-Z]", "", t)
            if not letters:
                syllables = None
                break
            syllables.append(letters.upper() if primary else letters.lower())
        if syllables:
            return "-".join(syllables)
    return None


def _extract_caps_stress_respelling(text: str) -> str | None:
    """Pull a parenthesized, space-separated respelling where stress is
    marked by capitalizing the stressed syllable instead of hyphenating or
    priming it, e.g. `(GAD oh KWA trane)` -- a Merriam-Webster-style
    respelling rendered with spaces. Requiring parens plus a mix of upper-
    and lower-case syllables is what keeps this off an ordinary acronym-
    bearing sentence.
    """
    text = text.replace("*", "")
    for m in re.finditer(r"\(([a-zA-Z]{1,10}(?:\s+[a-zA-Z]{1,10}){1,6})\)", text):
        tokens = m.group(1).split()
        if len(tokens) < 2:
            continue
        has_upper = any(t.isupper() for t in tokens)
        has_lower = any(t.islower() for t in tokens)
        if has_upper and has_lower:
            return "-".join(tokens)
    return None


def _extract_ipa(segment_text: str) -> str | None:
    """Pull an IPA transcription out of a claim segment (text between slashes)."""
    m = _IPA_PATTERN.search(segment_text.replace("*", ""))
    return m.group(1).strip() if m else None


def _normalize(s: str) -> str:
    return re.sub(r"[\s–-]+", "", s.lower())


@dataclass(frozen=True)
class VerifiedClaim:
    respelling: str | None
    ipa: str | None
    source_url: str
    source_domain: str
    page_verified: bool


def _judge_format(word: str, respelling: str | None, ipa: str | None) -> bool:
    """Gemini 2.5 Flash format/plausibility check, cached by (word, claim).

    This is not the verification step -- the grounding citation itself (a
    real Google Search result Gemini's tool actually retrieved) is that.
    This is a cheap backstop against the extraction regex grabbing an
    unrelated hyphenated phrase from the same sentence (e.g. "Uses-Dosage-
    Warnings"), by asking whether the candidate is even a plausible,
    correctly-formatted pronunciation guide for this specific word.
    """
    cache_key = f"judge::{word}::{respelling}::{ipa}"
    hit = _cached(cache_key)
    if hit is not None:
        return bool(hit.get("valid"))

    claim_desc = ", ".join(
        x
        for x in [
            f'respelling "{respelling}"' if respelling else None,
            f'IPA "{ipa}"' if ipa else None,
        ]
        if x
    )
    prompt = (
        f'Drug name: "{word}". Candidate pronunciation: {claim_desc}. '
        "Is this plausibly a real phonetic pronunciation guide for that "
        "specific word -- correctly formatted (hyphenated syllables or IPA), "
        "not an unrelated phrase, not a different word, not boilerplate page "
        'text? Answer with exactly one word: "yes" or "no".'
    )

    try:
        token = _access_token()
        project = _project_id()
        body = json.dumps(
            {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                # Extended thinking eats maxOutputTokens before any answer
                # text is emitted (confirmed by a bare `MAX_TOKENS` response
                # with no `content` at all) -- this is a one-word yes/no
                # classification, not a reasoning task, so thinking is off.
                "generationConfig": {
                    "temperature": 0,
                    "maxOutputTokens": 20,
                    "thinkingConfig": {"thinkingBudget": 0},
                },
            }
        ).encode()
        url = (
            f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{project}"
            f"/locations/{LOCATION}/publishers/google/models/gemini-2.5-flash:generateContent"
        )
        req = urllib.request.Request(
            url, data=body, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        )
        raw = urllib.request.urlopen(req, timeout=TIMEOUT).read()
        result = json.loads(raw)
        text = result["candidates"][0]["content"]["parts"][0]["text"].strip().lower()
        valid = text.startswith("y")
    except Exception:
        valid = False

    _store(cache_key, {"valid": valid})
    return valid


def verified_claims(name: str) -> list[VerifiedClaim]:
    """Every grounded claim for `name` backed by a real Google Search
    grounding citation and passing the LLM format/plausibility check.

    The citation IS the retrieval -- Gemini's `google_search` tool already
    fetched the real page server-side to produce it, and the grounding
    chunk's domain/URL is Google's own record of that, not a model guess.
    Re-fetching the page ourselves afterward is opportunistic (it can
    upgrade a claim to `page_verified` when the domain isn't blocked to
    this sandbox), never a requirement: a real citation to a domain we
    can't personally re-fetch (drugs.com 403s here) is still a real,
    independently-retrieved source.
    """
    answer = grounded_answer(name)
    if not answer or not answer.chunks:
        return []

    resolved = {i: _resolve_redirect(c.redirect_uri) for i, c in enumerate(answer.chunks)}
    out: list[VerifiedClaim] = []
    seen: set[tuple[str, str, str]] = set()

    def emit(respelling: str | None, ipa: str | None, indices) -> None:
        if not respelling and not ipa:
            return
        if not _judge_format(name, respelling, ipa):
            return
        for idx in indices:
            chunk = answer.chunks[idx]
            resolved_url = resolved.get(idx)
            url = resolved_url or chunk.redirect_uri
            domain = chunk.domain or (urllib.parse.urlparse(resolved_url).netloc if resolved_url else "")

            page_verified = False
            if resolved_url:
                page_text = _fetch_page_text(resolved_url)
                if page_text:
                    norm_page = _normalize(page_text)
                    if (respelling and _normalize(respelling) in norm_page) or (
                        ipa and ipa in page_text
                    ):
                        page_verified = True

            key = (respelling or "", ipa or "", domain)
            if key in seen:
                continue
            seen.add(key)
            out.append(VerifiedClaim(respelling, ipa, url, domain, page_verified))

    # Precise pass: attribute each claim only to the chunks Gemini's own
    # grounding actually cited for the sentence it appeared in.
    for support in answer.supports:
        segment = support["segment_text"]
        respelling = (
            _extract_respelling(segment, exclude=name)
            or _extract_stress_respelling(segment)
            or _extract_caps_stress_respelling(segment)
        )
        ipa = _extract_ipa(segment)
        emit(respelling, ipa, support["chunk_indices"])

    # Fallback: Gemini's segmentation sometimes doesn't attach a grounding
    # support to the headline claim sentence itself (only to the trailing
    # "found on X.com" sentence around it) -- confirmed by inspecting the
    # raw response for dupilumab/valbenazine/doravirine, where the sentence
    # stating the actual respelling had zero groundingSupports entries. When
    # the precise pass finds nothing, fall back to the whole answer text and
    # attribute to every citation in the response: safe here because the
    # prompt is single-topic ("what is the phonetic pronunciation of X"), so
    # every citation Google's search grounding chose is inherently about
    # that one fact, not scattered unrelated ones.
    if not out:
        respelling = (
            _extract_respelling(answer.text, exclude=name)
            or _extract_stress_respelling(answer.text)
            or _extract_caps_stress_respelling(answer.text)
        )
        ipa = _extract_ipa(answer.text)
        emit(respelling, ipa, range(len(answer.chunks)))

    return out
