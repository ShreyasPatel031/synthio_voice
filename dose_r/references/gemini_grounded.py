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


_TIGHT_PROMPT = (
    'What is the phonetic pronunciation of the drug "{name}"? Search the '
    "web for it, but this time restrict yourself to official or "
    "institutional sources only: a government health agency or regulator "
    "(fda.gov, dailymed.nlm.nih.gov, medlineplus.gov, cancer.gov, who.int, "
    "ema.europa.eu), the drug naming body (ama-assn.org / USAN), the "
    "manufacturer's own official prescribing information or medication "
    "guide, or a major medical institution/academic reference (a "
    "university hospital, a professional drug reference like Davis's Drug "
    "Guide, a national cancer charity). Do not cite a crowdsourced or "
    "user-submitted pronunciation site (howtopronounce.com, Forvo, a "
    "YouTube video or its comments/captions, a blog, social media, a "
    "generic name-meaning site) even if one is the only result you find -- "
    "say so instead of citing it. Tell me which page you found it on."
)

# Fallback for when sources.usan_pronunciation's own direct search of
# searchusan.ama-assn.org's index finds nothing (a name spelled differently
# than the query, a temporary API hiccup, a name genuinely missing from
# that specific index despite existing) -- ask Gemini's web search to look
# specifically for a USAN Statement on Adoption, on this domain or reported
# by a secondary source that quotes one, rather than the open-ended web
# search `_generate`'s normal prompt already tries.
_USAN_PROMPT = (
    'Search specifically for the official USAN (United States Adopted '
    'Names) Council "Statement on Adoption" or "Statement on a '
    'Nonproprietary Name" document for the drug "{name}" -- these are '
    "published by the American Medical Association, usually hosted at "
    "searchusan.ama-assn.org, and each one has a PRONUNCIATION field using "
    "USAN's own stress-mark notation (a single prime like nap' for primary "
    "stress, a double prime like nap\" for secondary stress). If you can't "
    "find the document itself, a secondary source (a medical reference "
    "site, a pharmacy database) that quotes the USAN pronunciation directly "
    "is acceptable too -- but say so if all you find is a general web "
    "pronunciation not sourced to USAN. Quote the PRONUNCIATION field "
    "exactly as written and tell me which document or page you found it on."
)

_PROMPTS = {"tight": _TIGHT_PROMPT, "usan": _USAN_PROMPT}


def _generate(name: str, retries: int = 3, mode: str = "normal") -> dict | None:
    cache_key = f"raw-{mode}::{name}" if mode != "normal" else f"raw::{name}"
    hit = _cached(cache_key)
    if hit is not None:
        return hit or None

    token = _access_token()
    project = _project_id()
    prompt = (
        _PROMPTS[mode].format(name=name)
        if mode in _PROMPTS
        else (
            f'What is the phonetic pronunciation of the drug "{name}"? '
            "Search the web for it and tell me which page you found it on."
        )
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


def grounded_answer(name: str, mode: str = "normal") -> GroundedAnswer | None:
    result = _generate(name, mode=mode)
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
    #
    # The 4th branch (a *run* of 2+ hyphen-groups separated by single
    # spaces) comes before the bare single-group branch so it wins at the
    # same starting position: a two-word generic name's answer is often one
    # hyphen-group per word ("pra-DEM-a-jeen ZAM-i-KER-a-sel" for
    # "prademagene zamikeracel") and matching only the first bare group used
    # to silently record just half the name's pronunciation as if it were
    # the whole thing.
    r'"([\wÀ-ʯ\']+(?:[-–][\wÀ-ʯ\']+){1,7})"|'
    r"\(([\wÀ-ʯ\' ]+(?:[-–][\wÀ-ʯ\']+){1,7})\)|"
    r"\b(\w+(?:[-–]\w+){1,7}(?:\s+\w+(?:[-–]\w+){1,7})+)\b|"
    r"\b(\w+(?:[-–]\w+){1,7})\b"
)

_IPA_PATTERN = re.compile(r"/([^/\s][^/]{1,40}[^/\s])/")


def _extract_all_respellings(segment_text: str, exclude: str | None = None) -> list[str]:
    """Every hyphenated respelling candidate in a claim segment, in order.

    Stress is not always marked by capitalization ("ad-kee" is as valid a
    respelling as "bik-TEG-ra-vir") -- the only structural requirement is
    that it is spelled as hyphenated syllables, not a real English word.

    Gemini's answer often echoes the input drug name in quotes right next to
    the real respelling (`the drug "elranatamab-bcmm" is "El-rah-NAH-tah-
    mab"`) -- when that echo itself has an FDA-suffix hyphen, it satisfies
    this regex and, being first in the sentence, used to get returned ahead
    of the actual respelling if only the first match were kept. `exclude`
    (the source word) drops that echo.

    The exclude comparison is case-insensitive but hyphen-preserving, not
    the loose `_normalize()` (which also strips hyphens/spaces): a real
    respelling like "VRAY-lar" normalizes to the same string as its source
    word "Vraylar" purely because the hyphen and case differences wash out,
    and that would wrongly exclude it as "just the name" -- an actual echo
    reproduces the source word's own spelling verbatim (case aside), it
    doesn't happen to collide with it after stripping punctuation.

    Returning every match, not just the first, matters for a compound
    answer like `"in-SUL-in EYE-koe-dek"` when asking specifically about
    "icodec-abae": the first hyphen-group belongs to a different word
    ("insulin") than the one being asked about, and only trying candidates
    in order until one passes the format judge (see `_candidates` in
    `verified_claims`) recovers the second, correct one.
    """
    text = segment_text.replace("*", "")
    exclude_ci = exclude.strip().lower() if exclude else None
    out: list[str] = []
    seen: set[str] = set()
    for m in _RESPELL_PATTERN.finditer(text):
        candidate = m.group(1) or m.group(2) or m.group(3) or m.group(4)
        if not candidate or "-" not in candidate.replace("–", "-"):
            continue
        candidate = candidate.replace("–", "-").strip()
        if exclude_ci and candidate.lower() == exclude_ci:
            continue
        if candidate.lower() in seen:
            continue
        seen.add(candidate.lower())
        out.append(candidate)
    return out


def _extract_respelling(segment_text: str, exclude: str | None = None) -> str | None:
    """The first candidate from `_extract_all_respellings`, or None."""
    matches = _extract_all_respellings(segment_text, exclude=exclude)
    return matches[0] if matches else None


_STRESS_MARK = r"(?:''|\"|”|″|['’′‘])"
_STRESS_TOKEN = re.compile(rf"^[a-zA-Z]{{1,8}}{_STRESS_MARK}?$")
_STRESS_STOPWORDS = {
    "is", "a", "an", "the", "of", "or", "and", "in", "on", "at", "as",
    "to", "was", "were", "it", "be", "by", "for", "with",
}
_QUOTED_WORD = re.compile(r'"([a-zA-Z]+(?:\s+[a-zA-Z]+)*)"')


def _stress_kind(token: str) -> str | None:
    """`"primary"` for a single prime -- the real Unicode prime `′` (as in
    "ten ek′ te plase" for tenecteplase, from MedlinePlus/SafeMedication),
    an ASCII apostrophe `'`, or a curly quote in either direction (`’` or
    `‘` -- pypdf renders a real USAN PDF's prime as whichever curly
    direction its font happens to use, confirmed inconsistent even within
    the same document set: "dem‘" in prademagene zamikeracel's own
    Statement, "kiz’" in risankizumab's) -- `"secondary"` for a double
    prime, rendered as the real Unicode double prime `″`, a literal `"`,
    two ASCII apostrophes `''` (Gemini's own text output), or the curly
    right-double-quote `”` (pypdf extracting a real USAN PDF's double-prime
    glyph, confirmed against the risankizumab and zolbetuximab documents:
    `ris" an kiz' ue mab`) -- `None` for no stress mark at all.
    """
    if token.endswith("''") or token.endswith('"') or token.endswith("”") or token.endswith("″"):
        return "secondary"
    if token.endswith("'") or token.endswith("’") or token.endswith("′") or token.endswith("‘"):
        return "primary"
    return None


def _quoted_words(text: str) -> set[str]:
    """Every word that appears as, or as the last word of, an ordinary
    double-quoted mention in `text` -- e.g. `{"sunirine"}` from `the drug
    "sunirine" in the name "pivekimab sunirine"`. A bare `"` is genuinely
    the USAN double-prime stress glyph in some Gemini answers ("dor\" a
    vir' een"), but Gemini also just quotes plain words constantly, and a
    token immediately before one of *those* closing quotes ("sunirine\"")
    is indistinguishable from a real double-prime syllable by punctuation
    alone. This set lets the stopword-stripping loop drop such an echoed
    word instead of misreading it as a stress-marked syllable.
    """
    words: set[str] = set()
    for phrase in _QUOTED_WORD.findall(text):
        words.add(phrase.split()[-1].lower())
    return words


def stress_tokens_to_respelling(tokens: list[str]) -> str | None:
    """USAN/USP prime-stress syllable tokens (e.g. `["am", "bel'", "vist"]`,
    from `"am bel' vist"`) -> a hyphenated respelling with the primary-
    stressed syllable uppercased (`"am-BEL-vist"`), the same shape
    `respell_to_arpabet_ipa` expects. `None` if the tokens don't actually
    carry a stress mark or don't clean down to letters -- e.g. a bare repeat
    of the drug's own name with no internal structure at all.

    Shared between `_extract_stress_respelling` (pulling a candidate out of
    Gemini's free-form prose) and `sources.dailymed_pronunciation` (an FDA
    Medication Guide's title line already isolates the respelling in
    parentheses, so there is no prose to search, just tokens to convert).
    """
    if len(tokens) < 2 or not all(_STRESS_TOKEN.match(t) for t in tokens):
        return None
    if not any(_stress_kind(t) == "primary" for t in tokens):
        return None
    syllables = []
    for t in tokens:
        primary = _stress_kind(t) == "primary"
        letters = re.sub(r"[^a-zA-Z]", "", t)
        if not letters:
            return None
        syllables.append(letters.upper() if primary else letters.lower())
    return "-".join(syllables)


def _extract_stress_respelling(text: str) -> str | None:
    """Pull a USAN/USP-style pronunciation-key respelling: space-separated
    syllables with a prime marking stress, e.g. `(dor" a vir' een)` or
    `zip'' ah ler' ti nib` -- secondary stress marked with a double prime,
    primary with a single prime. This is the official USAN adopted-name
    pronunciation convention, not the drugs.com/WebMD hyphenated style
    `_extract_respelling` handles, so it needs its own parser.

    Requiring the stress mark to be the LAST character(s) of its token (not
    mid-token, as in a possessive like "Davis's") is what keeps this from
    firing on ordinary prose.
    """
    text = text.replace("*", "")
    quoted = _quoted_words(text)
    for m in re.finditer(
        rf"(?:^|[\s(\"])((?:[a-zA-Z]{{1,8}}{_STRESS_MARK}?\s+){{1,5}}[a-zA-Z]{{1,8}}{_STRESS_MARK}?)(?=[\s.)\"]|$)",
        text,
    ):
        tokens = m.group(1).split()
        while tokens and re.sub(r"[^a-zA-Z\"]", "", tokens[0]).lower().rstrip('"') in (
            _STRESS_STOPWORDS | quoted
        ):
            tokens = tokens[1:]
        respelling = stress_tokens_to_respelling(tokens)
        if respelling:
            return respelling
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


def _flash_text(prompt: str, max_tokens: int = 20) -> str | None:
    """A single Gemini 2.5 Flash call, thinking disabled, returning the raw
    text response (stripped) or None on any failure. Shared by every small
    fast classification call in this module (`_judge_format`,
    `classify_source_trust`) -- none of them are reasoning tasks, they're
    one-word or one-phrase classifications, and extended thinking silently
    eats the whole `maxOutputTokens` budget before emitting an answer at
    all if left on (confirmed by a bare `MAX_TOKENS` response with no
    `content` whatsoever).
    """
    try:
        token = _access_token()
        project = _project_id()
        body = json.dumps(
            {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0,
                    "maxOutputTokens": max_tokens,
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
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return None


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
    text = _flash_text(prompt)
    valid = bool(text) and text.strip().lower().startswith("y")
    _store(cache_key, {"valid": valid})
    return valid


TRUST_TIERS = ("official_medical", "verified_secondary", "third_party_unverified")

# Deliberately a handful of illustrative examples per bucket, not an
# exhaustive or hardcoded domain->tier map -- the point of asking Gemini is
# that it generalizes to a domain that isn't in this list (a hospital
# system's own site, a national medicines agency other than the FDA, a
# pharmacy chain's drug-info page) the way a fixed lookup table can't.
_TRUST_TIER_PROMPT = """Classify the website domain "{domain}" into exactly \
one of these three trust tiers for citing a drug's official phonetic \
pronunciation, then answer with only the tier name, nothing else.

official_medical: a government health agency, national regulator, or the \
official naming body for drug names. Examples: fda.gov, \
dailymed.nlm.nih.gov, medlineplus.gov, cancer.gov, who.int, \
ama-assn.org (USAN Council), ema.europa.eu, a drug manufacturer's own \
official prescribing information or medication guide.

verified_secondary: an editorially-maintained medical reference, \
professional dictionary, academic institution, or major hospital system -- \
not a primary regulator, but not open to arbitrary public submissions \
either. Examples: drugs.com, webmd.com, merriam-webster.com, Wikipedia, \
Wiktionary, a university hospital's patient-education site (e.g. \
oncolink.org, a .edu cancer center), a professional nursing/pharmacy \
reference (e.g. Davis's Drug Guide), Medscape, a national cancer charity \
(e.g. cancerresearchuk.org).

third_party_unverified: crowdsourced or user-generated content with no \
editorial review of accuracy -- literally anyone can post anything, \
including a wrong guess at how to say a word. Examples: \
howtopronounce.com, forvo.com, YouTube video comments or auto-generated \
captions, a personal blog or Substack, social media (Facebook, Reddit, \
X/Twitter), a generic "baby names" or "name meaning" site (e.g. names.org).

Answer with exactly one of: official_medical, verified_secondary, \
third_party_unverified"""


def classify_source_trust(domain: str) -> str:
    """Which of TRUST_TIERS `domain` belongs in, per Gemini 2.5 Flash --
    cached by domain (a domain's trust classification doesn't depend on
    which drug it's cited for, so this is a one-time cost per distinct
    domain, not per citation).

    Unrecognized Gemini output or a failed call defaults to
    "third_party_unverified": an unclassifiable source should never be
    silently trusted as if it had been vetted.
    """
    domain = domain.strip().lower()
    cache_key = f"trust::{domain}"
    hit = _cached(cache_key)
    if hit is not None:
        return hit.get("tier", "third_party_unverified")

    text = _flash_text(_TRUST_TIER_PROMPT.format(domain=domain), max_tokens=15)
    tier = "third_party_unverified"
    if text:
        cleaned = text.strip().lower()
        for candidate in TRUST_TIERS:
            if candidate in cleaned:
                tier = candidate
                break

    _store(cache_key, {"tier": tier})
    return tier


_SPECULATION_PATTERN = re.compile(
    r"was not (?:\w+\s+){0,2}found|not found in the search|"
    r"no direct pronunciation|highly probable|would likely be|"
    r"is likely (?:to be|pronounced)|probably (?:pronounced|follows)|"
    r"could not find|couldn't find|unable to find|"
    r"does not appear to have|no specific pronunciation|i couldn't find|"
    r"was not available|not available within|did not list|"
    r"was not (?:\w+\s+){0,2}available",
    re.IGNORECASE,
)


def _is_speculative(text: str) -> bool:
    """True when Gemini's own answer admits it found no direct source for
    the specific word asked about and is instead guessing by analogy, or
    reporting a DIFFERENT word's pronunciation as if it answered the
    question.

    Confirmed on two real cases:
    - "histidinate": answered by inferring from "histidine" (a different,
      if related, word) with "It is highly probable that... would likely
      be **HIS-ti-dih-nate**", citing sources that were for "histidine".
    - "Wakix" (tight mode): "a phonetic pronunciation for the drug 'Wakix'
      was not explicitly found... the phonetic pronunciation for the brand
      name 'Wakix' was not available" -- but mentioned in passing that
      WebMD gives its generic ingredient pitolisant's pronunciation, which
      the extraction regex duly grabbed and the format judge, only checking
      "is this a plausible phonetic guide" and not "is this a guide *for
      this specific word*", didn't catch. `was not (?:\w+\s+){0,2}found`
      tolerates the paraphrase ("not explicitly found") the original
      literal `was not found` phrase missed.

    In both cases the citations were real; the claim they were cited for
    was not what they said. This pipeline's whole premise is retrieval, not
    inference or substitution, so an answer that admits to either must
    never be treated as if it cited something for the word actually asked
    about.
    """
    return bool(_SPECULATION_PATTERN.search(text))


def verified_claims(name: str, mode: str = "normal") -> list[VerifiedClaim]:
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

    `mode="tight"` asks a second, differently-worded question (`_TIGHT_PROMPT`)
    that explicitly restricts Gemini's own search to official/institutional
    sources and tells it not to cite a crowdsourced site even as a last
    resort. Run only when the normal query's claims all turned out
    `third_party_unverified` (see `build._from_gemini_grounded`).

    `mode="usan"` asks Gemini's web search to look specifically for a USAN
    Statement on Adoption (`_USAN_PROMPT`). Run only when
    `sources.usan_pronunciation`'s own direct search of the AMA's index
    finds nothing for a generic name (see `build._resolve_word`) -- a
    fallback for a name spelled differently than the query, a transient API
    issue, or a name genuinely missing from that specific index.

    Neither mode replaces the normal query, both supplement it, and each is
    cached completely separately (`raw-tight::` / `raw-usan::` vs `raw::`)
    so re-running costs nothing once fetched.
    """
    answer = grounded_answer(name, mode=mode)
    if not answer or not answer.chunks or _is_speculative(answer.text):
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

    def _candidates(text: str) -> list[str]:
        """Every distinct respelling candidate across all three notation
        styles this pipeline parses, and every hyphenated match in the
        text, not just the first. Taking only the first match (the earlier
        version of this function, whether by `or`-chaining extractors or by
        `_extract_respelling` itself returning early) let an early, wrong
        match block a later, correct one from ever being tried:
          - the plain hyphen pattern matching a citation's document number
            ("USAN NO-08") ahead of the real stress-marked respelling later
            in the same sentence ("zip'' ah ler' ti nib"), for Zipalertinib.
          - a parenthetical aside that happens to end in a hyphenated word
            ("(as part of the full drug name ... inbakicept-pmln)") matching
            ahead of the real quoted answer ("in-BAK-ih-sept") later on.
          - a compound answer for a two-word name ("in-SUL-in EYE-koe-dek")
            giving up the FIRST word's respelling when asked specifically
            about the SECOND ("icodec-abae") -- only the second hyphen-group
            is a plausible answer for that word, and only trying every group
            lets the format judge find it.
        `emit` already rejects a bad candidate via the format judge, so
        there is no harm in offering it every candidate instead of just one
        regex's first opinion.
        """
        out_candidates: list[str] = []
        for candidate in (
            *_extract_all_respellings(text, exclude=name),
            _extract_stress_respelling(text),
            _extract_caps_stress_respelling(text),
        ):
            if candidate and candidate not in out_candidates:
                out_candidates.append(candidate)
        return out_candidates

    # Precise pass: attribute each claim only to the chunks Gemini's own
    # grounding actually cited for the sentence it appeared in.
    for support in answer.supports:
        segment = support["segment_text"]
        ipa = _extract_ipa(segment)
        for respelling in _candidates(segment) or [None]:
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
        ipa = _extract_ipa(answer.text)
        for respelling in _candidates(answer.text) or [None]:
            emit(respelling, ipa, range(len(answer.chunks)))

    return out
