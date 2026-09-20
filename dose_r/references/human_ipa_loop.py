"""Human-clip → IPA proposals for a name+sidecar injection loop.

CTC is a noisy decode, not gold. Proposals are trimmed to the sidecar's
length so clip tails (Advair Diskus, Nurtec extra phones) are not copied.
The eval script keeps a proposal only when Path 2 F1 rises.
"""

from __future__ import annotations

import difflib

from .tts_pronunciation import to_cloud_en_us_ipa
from .wiki_notation import IPA_TABLE, PRIMARY, SECONDARY

# Longest first. CTC extras sit beside the Wikipedia inventory.
_PHONES: list[str] = sorted(
    {g for g, _ in IPA_TABLE} | {"ɹ", "ɚ", "ɐ", "ᵻ", "eː", "ɜː", "o", "g"},
    key=len,
    reverse=True,
)

_VOWEL_START = set("aeiouæɑɒɔɛɜɪʊʌəɐᵻɚ")

_CLOUD_PHONE = {
    "ɹ": "r",
    "ɐ": "ə",
    "ᵻ": "ə",
    "ɚ": "ər",
    "eː": "eɪ",
    "g": "ɡ",
    "i": "i",
    "u": "u",
    "a": "ɑ",
    "o": "oʊ",
}


def tokenize_ipa(text: str) -> list[str]:
    """Greedy longest-match phones. Stress marks are kept as their own tokens."""
    text = (text or "").replace(" ", "")
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in (PRIMARY, SECONDARY):
            out.append(ch)
            i += 1
            continue
        hit = next((g for g in _PHONES if text.startswith(g, i)), None)
        if hit:
            out.append(hit)
            i += len(hit)
        else:
            i += 1
    return out


def ctc_phones(ctc: str) -> list[str]:
    """wav2vec2-espeak decode → phone list (spaces already segment it)."""
    raw = [p for p in (ctc or "").split() if p]
    if raw:
        return raw
    return [p for p in tokenize_ipa(ctc) if p not in (PRIMARY, SECONDARY)]


def is_vowel(phone: str) -> bool:
    return bool(phone) and phone[0] in _VOWEL_START


def strip_stress(phones: list[str]) -> list[str]:
    return [p for p in phones if p not in (PRIMARY, SECONDARY)]


def cloud_phone(phone: str) -> str:
    return _CLOUD_PHONE.get(phone, phone)


def _levenshtein(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def name_span(human: list[str], sidecar: list[str]) -> list[str]:
    """Sliding window of human phones nearest the sidecar length.

    Drops clip tails (Advair Diskus) and leading junk. If the human
    decode is not longer than the sidecar, keep it.
    """
    ref = strip_stress(sidecar)
    hum = strip_stress(human)
    if not hum:
        return ref
    if not ref:
        return hum
    if len(hum) <= len(ref) + 1:
        return hum
    best = hum[: len(ref)]
    best_d = _levenshtein(best, ref)
    for win in (len(ref) - 1, len(ref), len(ref) + 1):
        if win < 3:
            continue
        for i in range(0, len(hum) - win + 1):
            chunk = hum[i : i + win]
            d = _levenshtein(chunk, ref)
            if d < best_d:
                best, best_d = chunk, d
    return best


def apply_stress(phones: list[str], sidecar_ipa: str) -> str:
    """Put the sidecar's primary stress on the first vowel of `phones`."""
    body = "".join(cloud_phone(p) for p in strip_stress(phones) if p)
    body = to_cloud_en_us_ipa(body)
    if PRIMARY in (sidecar_ipa or "") and PRIMARY not in body:
        toks = tokenize_ipa(body)
        for i, p in enumerate(toks):
            if is_vowel(p):
                toks.insert(i, PRIMARY)
                break
        body = "".join(toks)
    return body


def from_human_ctc(sidecar_ipa: str, human_ctc: str) -> str:
    """Trimmed human CTC as Cloud IPA, stress copied from the sidecar."""
    span = name_span(ctc_phones(human_ctc), tokenize_ipa(sidecar_ipa))
    return apply_stress(span, sidecar_ipa)


def _align_take(
    sidecar_ipa: str,
    human_ctc: str,
    *,
    vowels_only: bool = False,
    consonants_only: bool = False,
) -> str:
    side = strip_stress(tokenize_ipa(sidecar_ipa))
    hum = name_span(ctc_phones(human_ctc), tokenize_ipa(sidecar_ipa))
    if not side:
        return from_human_ctc(sidecar_ipa, human_ctc)
    matcher = difflib.SequenceMatcher(a=side, b=hum, autojunk=False)
    out: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            out.extend(side[i1:i2])
            continue
        if tag == "insert":
            extra = hum[j1:j2]
            if not vowels_only and not consonants_only:
                out.extend(extra)
            continue
        if tag == "delete":
            if vowels_only or consonants_only:
                out.extend(side[i1:i2])
            continue
        s_chunk, h_chunk = side[i1:i2], hum[j1:j2]
        n = max(len(s_chunk), len(h_chunk))
        for k in range(n):
            s = s_chunk[k] if k < len(s_chunk) else ""
            h = h_chunk[k] if k < len(h_chunk) else ""
            if not h:
                if s:
                    out.append(s)
                continue
            if not s:
                out.append(h)
                continue
            if vowels_only:
                out.append(h if is_vowel(s) and is_vowel(h) else s)
            elif consonants_only:
                out.append(h if (not is_vowel(s) and not is_vowel(h)) else s)
            else:
                out.append(h)
    return apply_stress(out, sidecar_ipa)


def transplant_vowels(sidecar_ipa: str, human_ctc: str) -> str:
    return _align_take(sidecar_ipa, human_ctc, vowels_only=True)


def transplant_consonants(sidecar_ipa: str, human_ctc: str) -> str:
    return _align_take(sidecar_ipa, human_ctc, consonants_only=True)


def align_replace(sidecar_ipa: str, human_ctc: str) -> str:
    return _align_take(sidecar_ipa, human_ctc)


def residual_patch(sidecar_ipa: str, human_ctc: str, synth_ctc: str) -> str:
    """Move sidecar phones toward human where the last synth still differs."""
    if not synth_ctc:
        return align_replace(sidecar_ipa, human_ctc)
    hum = name_span(ctc_phones(human_ctc), tokenize_ipa(sidecar_ipa))
    syn = name_span(ctc_phones(synth_ctc), tokenize_ipa(sidecar_ipa))
    if "".join(cloud_phone(p) for p in hum) == "".join(cloud_phone(p) for p in syn):
        return sidecar_ipa
    return align_replace(sidecar_ipa, human_ctc)


def propose_round(
    sidecar_ipa: str,
    human_ctc: str,
    synth_ctc: str | None,
    round_idx: int,
) -> str:
    """One of five distinct proposals. `round_idx` is 1..5."""
    if round_idx == 1:
        return from_human_ctc(sidecar_ipa, human_ctc)
    if round_idx == 2:
        return transplant_vowels(sidecar_ipa, human_ctc)
    if round_idx == 3:
        return transplant_consonants(sidecar_ipa, human_ctc)
    if round_idx == 4:
        return align_replace(sidecar_ipa, human_ctc)
    return residual_patch(sidecar_ipa, human_ctc, synth_ctc or "")
