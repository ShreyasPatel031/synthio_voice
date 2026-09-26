#!/usr/bin/env python
"""Two independent Cloud Standard-C voices of the same name.

IPA arm: Google `{part} ipa pronunciation` per whitespace word and per FDA
4-letter suffix; inject those strings as Cloud customPronunciations phrases
(substrings of the spoken name). Never G2P DailyMed/USAN respelling.

Respelling arm: SSML <sub alias="a ta ki sept v y m j">atacicept-vymj</sub>
from the original source respelling. Hyphens become spoken English syllables.
FDA 4-letter suffixes are not spoken. Real extra words stay.

Keep a Google IPA candidate when CTC phones of the two wavs agree more
(keep if ratio rises > +0.005). Wavlm F1 is reported but pauses in the
respelling arm make it a weak keep metric.

Does not write pronunciations.jsonl ipa / ipa_cloud.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
_hf = ROOT / ".cache" / "huggingface"
_hf.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_hf))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_hf))
os.environ.setdefault("HF_HUB_CACHE", str(_hf / "hub"))

import requests  # noqa: E402

from apply_user_ipa_and_iterate import (  # noqa: E402
    _skip_host,
    gemini_word,
    parse_ipa_variants,
)
from dose_r.references.reference_clips import available_clips  # noqa: E402
from dose_r.references.tts_pronunciation import (  # noqa: E402
    custom_pronunciations_for_parts,
    is_fda_letter_suffix,
    is_source_ipa,
    page_is_for_word,
    spoken_parts,
    spoken_text,
    respelling_alias,
)
from verify_google_ipa_tts import strip_ipa, token  # noqa: E402

ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"
VOICE = "en-US-Standard-C"
RATE = 24000
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "project-amer-scs-sandbox")
PRON = ROOT / "dose_r" / "references" / "pronunciations.jsonl"
IPA_SEED = ROOT / "runs" / "gemini3-ipa-query" / "ipa.json"
OUT = ROOT / "runs" / "ipa-vs-respell"
TTS = OUT / "tts"
LISTEN = ROOT / "runs" / "listen-ipa-vs-respell"
PARTS = OUT / "parts.json"
RESULTS = OUT / "results.json"
KEEP = 0.005
LOW_CTC = 0.85
MIXUP_CTC = 0.35
AGREE_F1 = 0.80
WORKERS = int(os.environ.get("TTS_WORKERS", "8"))
VERIFY_TTS = ROOT / "runs" / "verify-google-ipa" / "tts"
VERIFY_RESULTS = ROOT / "runs" / "verify-google-ipa" / "results.json"
CLIP_RESULTS = OUT / "clip_f1.json"
CLIP_LISTEN = ROOT / "runs" / "listen-ipa-vs-respell-clips"
_EMB: dict[int, object] = {}
_CACHE_LOCK = threading.Lock()
_GOOGLE_QUERIES = (
    "{word} ipa pronunciation",
    "{word} pronunciation IPA",
    "{word} wiktionary IPA pronunciation",
)


def load_pron() -> dict[str, dict]:
    out = {}
    for line in PRON.read_text().splitlines():
        rec = json.loads(line)
        out[rec["ingredient"]] = rec
    return out


def load_seed_ipa() -> dict[str, str]:
    if not IPA_SEED.exists():
        return {}
    out = {}
    for rec in json.loads(IPA_SEED.read_text()):
        if _skip_host(rec.get("url") or ""):
            continue
        if not page_is_for_word(rec["ingredient"], rec.get("url") or ""):
            continue
        ipa = strip_ipa(rec.get("ipa") or "")
        if ipa and is_source_ipa(ipa):
            out[rec["ingredient"]] = ipa
    return out


def load_parts_cache() -> dict[str, dict]:
    if PARTS.exists():
        return json.loads(PARTS.read_text())
    return {}


def save_parts_cache(cache: dict[str, dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PARTS.write_text(json.dumps(cache, indent=2, ensure_ascii=False) + "\n")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def synth(
    tok: str,
    *,
    text: str | None = None,
    ssml: str | None = None,
    pronunciations: dict | None = None,
) -> bytes:
    inp: dict
    if ssml:
        inp = {"ssml": ssml}
    else:
        inp = {"text": text}
        if pronunciations:
            inp["customPronunciations"] = pronunciations
    body = {
        "input": inp,
        "voice": {"languageCode": "en-US", "name": VOICE},
        "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": RATE},
    }
    resp = requests.post(
        ENDPOINT,
        headers={
            "Authorization": f"Bearer {tok() if callable(tok) else tok}",
            "Content-Type": "application/json",
            "x-goog-user-project": PROJECT,
        },
        json=body,
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"{resp.status_code}: {resp.text[:400]}")
    return base64.b64decode(resp.json()["audioContent"])


def respell_ssml(name: str, alias: str) -> str:
    return (
        "<speak><sub alias=\""
        + xml_escape(alias)
        + "\">"
        + xml_escape(name)
        + "</sub></speak>"
    )


def ctc_pair(a: bytes, b: bytes) -> tuple[float, str, str]:
    from dose_r.scoring.phoneme_distance import normalize_phonemes
    from dose_r.scoring.phoneme_model import transcribe_phonemes

    pa = normalize_phonemes(transcribe_phonemes(a))
    pb = normalize_phonemes(transcribe_phonemes(b))
    if not pa or not pb:
        return 0.0, pa, pb
    return round(SequenceMatcher(None, pa, pb).ratio(), 4), pa, pb


def google_part(part: str, cache: dict[str, dict], refresh: bool) -> dict:
    key = part.lower()
    with _CACHE_LOCK:
        if not refresh and key in cache and (
            cache[key].get("ipa") or cache[key].get("tried")
        ):
            if not _skip_host(cache[key].get("url") or ""):
                return cache[key]
    variants: list[str] = []
    url = ""
    queries: list[str] = []
    for tmpl in _GOOGLE_QUERIES:
        q = tmpl.format(word=part)
        hit = gemini_word(part, query=q) or {}
        queries.append(q)
        variants = [v for v in parse_ipa_variants(hit) if is_source_ipa(v)]
        url = hit.get("url") or url
        if variants:
            break
    rec = {
        "ipa": variants,
        "url": url,
        "tried": True,
        "queries": queries,
    }
    with _CACHE_LOCK:
        cache[key] = rec
        save_parts_cache(cache)
    print(f"  google {part}: {variants} {url}", flush=True)
    return rec


def variants_for_part(
    part: str, name: str, cache: dict[str, dict], seed: dict[str, str]
) -> list[str]:
    if is_fda_letter_suffix(part, name):
        return []
    rec = cache.get(part.lower()) or {}
    out: list[str] = []
    seen: set[str] = set()
    cache_url = rec.get("url") or ""
    if (not cache_url or page_is_for_word(part, cache_url)) and not _skip_host(cache_url):
        for ipa in rec.get("ipa") or []:
            ipa = strip_ipa(ipa)
            if ipa and is_source_ipa(ipa) and ipa not in seen:
                out.append(ipa)
                seen.add(ipa)
    parts = spoken_parts(name)
    if parts and part == parts[0] and name in seed:
        ipa = seed[name]
        if ipa not in seen and is_source_ipa(ipa):
            out.append(ipa)
            seen.add(ipa)
    return out


def pick_ipas(parts: list[str], name: str, cache: dict, seed: dict) -> list[str]:
    chosen: list[str] = []
    for part in parts:
        opts = variants_for_part(part, name, cache, seed)
        chosen.append(opts[0] if opts else "")
    return chosen


def f1(a: bytes, b: bytes) -> float:
    from dose_r.scoring.speech_similarity import (
        extract_frame_embeddings,
        speech_bertscore,
    )

    def emb(data: bytes):
        key = hash(data)
        if key not in _EMB:
            _EMB[key] = extract_frame_embeddings(data)
        return _EMB[key]

    return float(speech_bertscore(emb(a), emb(b))["f1"])


def write_wav(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_listen(rows: list[dict], clips: dict) -> Path:
    LISTEN.mkdir(parents=True, exist_ok=True)
    ranked = sorted(
        rows,
        key=lambda r: (
            0 if r.get("ctc") is not None else 1,
            r.get("ctc") if r.get("ctc") is not None else 9,
        ),
    )
    n_low = sum(1 for r in ranked if r.get("ctc") is not None and r["ctc"] < LOW_CTC)
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>IPA Cloud vs source respelling Cloud</title>",
        "<style>body{font:16px/1.4 system-ui;max-width:780px;margin:2rem auto;padding:0 1rem}",
        "section{border:1px solid #ccc;border-radius:8px;padding:1rem 1.2rem;margin:1rem 0}",
        "section.low{border-color:#a44}",
        "h1{font-size:1.25rem} h2{font-size:1.05rem;margin:0 0 .35rem}",
        "p{margin:.2rem 0 .55rem;color:#333} .ipa{font-family:ui-monospace,monospace}",
        "label{display:block;font-weight:600;margin:.5rem 0 .15rem} audio{width:100%}",
        ".meta{font-size:.9rem;color:#555}</style>",
        "<h1>Two Cloud voices of the same name</h1>",
        "<p>IPA arm: Google each word / FDA suffix separately, inject source IPA. ",
        "Respelling arm: original dictionary syllables as English ",
        "(<span class='ipa'>a ta ki sept v y m j</span>), not G2P to IPA. ",
        f"{n_low}/{len(ranked)} CTC &lt; {LOW_CTC:.2f} (need another Google pass).</p>",
    ]
    for rec in ranked:
        s = slug(rec["ingredient"])
        cls = " low" if rec.get("ctc") is not None and rec["ctc"] < LOW_CTC else ""
        parts.append(f"<section class='{cls.strip()}' id='{s}'><h2>{rec['ingredient']}</h2>")
        ctc = rec.get("ctc")
        f1v = rec.get("f1")
        parts.append(
            "<p class='meta'>"
            f"CTC phones {ctc if ctc is not None else '—'} · "
            f"wavlm F1 {f1v if f1v is not None else '—'} "
            "(F1 is low when the respelling arm has syllable pauses)</p>"
        )
        parts.append(
            f"<p>source respelling: <span class='ipa'>{rec.get('respelling') or '—'}</span><br>"
            f"spoken as: <span class='ipa'>{rec.get('alias') or '—'}</span><br>"
            f"parts: <span class='ipa'>{' · '.join(rec.get('parts') or [])}</span><br>"
            f"IPA used: <span class='ipa'>{rec.get('ipa_used') or '—'}</span></p>"
        )
        if rec.get("ctc_ipa") or rec.get("ctc_respell"):
            parts.append(
                f"<p class='ipa'>CTC IPA: {rec.get('ctc_ipa') or ''}<br>"
                f"CTC respell: {rec.get('ctc_respell') or ''}</p>"
            )
        for label, kind in (
            ("1 Google IPA (per part)", "ipa"),
            ("2 Source respelling (English syllables)", "respell"),
        ):
            src = TTS / f"{s}.{kind}.wav"
            dest = LISTEN / f"{s}__{kind}.wav"
            show_ipa = bool(rec.get("ipa_used")) and rec.get("error") not in {
                "mixup_ipa",
                "no_ipa",
            }
            if kind == "ipa" and not show_ipa:
                if dest.exists():
                    dest.unlink()
                continue
            if src.exists():
                dest.write_bytes(src.read_bytes())
                parts.append(
                    f"<label>{label}</label><audio controls src='{dest.name}'></audio>"
                )
            elif dest.exists():
                dest.unlink()
        human = clips.get(rec["ingredient"])
        if human is not None:
            dest = LISTEN / f"{s}__human{human.path.suffix}"
            dest.write_bytes(human.path.read_bytes())
            hf = rec.get("ipa_vs_human")
            parts.append(
                f"<label>human ({human.source}"
                f"{'' if hf is None else f', IPA F1 {hf}'})"
                f"</label><audio controls src='{dest.name}'></audio>"
            )
        parts.append("</section>")
    dest = LISTEN / "index.html"
    dest.write_text("\n".join(parts) + "\n")
    return dest


def target_names(
    pron: dict[str, dict],
    clips: dict,
    only: list[str] | None,
    clips_only: bool = False,
) -> list[str]:
    if only:
        return [n for n in only if n in pron]
    names = []
    for name, rec in pron.items():
        has_respell = bool(rec.get("respelling"))
        if clips_only:
            if name in clips and has_respell:
                names.append(name)
            continue
        multi = len(spoken_parts(name)) > 1 or spoken_text(name) != name
        no_clip = name not in clips
        if multi or (no_clip and has_respell):
            names.append(name)
    names.sort(key=lambda n: (0 if len(spoken_parts(n)) > 1 or spoken_text(n) != n else 1, n.lower()))
    return names


def score_name(
    tok: str,
    name: str,
    rec: dict,
    ipas: list[str],
    clips: dict,
    *,
    skip_ctc: bool = False,
    reuse: bool = False,
) -> dict:
    parts = spoken_parts(name)
    text = spoken_text(name)
    alias = respelling_alias(rec.get("respelling") or "", name) if rec.get("respelling") else ""
    row = {
        "ingredient": name,
        "respelling": rec.get("respelling") or "",
        "alias": alias,
        "parts": parts,
        "ipa_used": " ".join(f"{p}={i}" for p, i in zip(parts, ipas) if i),
        "part_ipas": list(zip(parts, ipas)),
    }
    s = slug(name)
    ipa_path = TTS / f"{s}.ipa.wav"
    resp_path = TTS / f"{s}.respell.wav"
    pairs = [(p, i) for p, i in zip(parts, ipas) if i]
    ipa_wav: bytes | None = None
    if reuse:
        if ipa_path.exists() and ipa_path.stat().st_size > 500:
            ipa_wav = ipa_path.read_bytes()
            row["ipa_wav_src"] = "cache"
        elif len(parts) == 1:
            vpath = VERIFY_TTS / f"{name.lower().replace(' ', '_')}.ipa.wav"
            if vpath.exists() and vpath.stat().st_size > 500:
                ipa_wav = vpath.read_bytes()
                write_wav(ipa_path, ipa_wav)
                row["ipa_wav_src"] = "verify"
    if ipa_wav is None:
        if not pairs:
            row["error"] = "no_ipa"
            if ipa_path.exists():
                ipa_path.unlink()
        else:
            while True:
                try:
                    ipa_wav = synth(
                        tok,
                        text=text,
                        pronunciations=custom_pronunciations_for_parts(pairs),
                    )
                    break
                except RuntimeError as exc:
                    msg = str(exc)
                    if "custom pronunciation phrases are invalid" not in msg or not pairs:
                        raise
                    dropped = None
                    for phrase, _ipa in list(pairs):
                        if phrase in msg:
                            dropped = phrase
                            pairs = [(p, i) for p, i in pairs if p != phrase]
                            break
                    if dropped is None:
                        pairs = pairs[:-1]
                    print(
                        f"  cloud drop {name} phrase={dropped or 'last'}: {msg[:160]}",
                        flush=True,
                    )
                    if not pairs:
                        raise
            row["ipa_used"] = " ".join(f"{p}={i}" for p, i in pairs)
            row["part_ipas"] = list(pairs)
            write_wav(ipa_path, ipa_wav)
            row["ipa_wav_src"] = "synth"
    if not alias:
        row["error"] = "no_respelling"
        return row
    resp_wav: bytes | None = None
    if reuse and resp_path.exists() and resp_path.stat().st_size > 500:
        resp_wav = resp_path.read_bytes()
    if resp_wav is None:
        resp_wav = synth(tok, ssml=respell_ssml(spoken_text(name), alias))
        write_wav(resp_path, resp_wav)
    if ipa_wav is None:
        return row
    if not skip_ctc:
        ratio, pa, pb = ctc_pair(ipa_wav, resp_wav)
        row["ctc"] = ratio
        row["ctc_ipa"] = pa
        row["ctc_respell"] = pb
        if ratio < MIXUP_CTC:
            row["error"] = "mixup_ipa"
            row["ipa_used"] = ""
            row["part_ipas"] = []
            if ipa_path.exists():
                ipa_path.unlink()
            return row
    row["f1"] = round(f1(ipa_wav, resp_wav), 4)
    clip = clips.get(name)
    if clip is not None:
        human = clip.path.read_bytes()
        row["clip_source"] = clip.source
        row["ipa_vs_human"] = round(f1(ipa_wav, human), 4)
        row["respell_vs_human"] = round(f1(resp_wav, human), 4)
        row["delta_ipa_respell"] = round(row["ipa_vs_human"] - row["respell_vs_human"], 4)
    return row


def winner(ipa_h: float, resp_h: float) -> str:
    if abs(ipa_h - resp_h) < KEEP:
        return "tie"
    return "ipa" if ipa_h > resp_h else "respell"


def human_calibration(rows: list[dict], verify: dict[str, dict]) -> dict:
    scored = []
    for rec in rows:
        ipa_h = rec.get("ipa_vs_human")
        resp_h = rec.get("respell_vs_human")
        if ipa_h is None or resp_h is None:
            continue
        agree = rec.get("f1")
        v = verify.get(rec["ingredient"]) or {}
        plain_h = v.get("plain")
        rec["plain_vs_human"] = plain_h
        rec["winner"] = winner(ipa_h, resp_h)
        scored.append(rec)

    def mean(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 4) if xs else None

    def rate(subset: list[dict], arm: str) -> float | None:
        n = len(subset)
        return round(sum(1 for r in subset if r["winner"] == arm) / n, 3) if n else None

    agrees = [r for r in scored if r.get("f1") is not None and r["f1"] >= AGREE_F1]
    disagrees = [r for r in scored if r.get("f1") is not None and r["f1"] < AGREE_F1]
    return {
        "n_clip_scored": len(scored),
        "agree_f1": AGREE_F1,
        "mean_ipa_vs_human": mean([r["ipa_vs_human"] for r in scored]),
        "mean_respell_vs_human": mean([r["respell_vs_human"] for r in scored]),
        "mean_plain_vs_human": mean(
            [r["plain_vs_human"] for r in scored if r.get("plain_vs_human") is not None]
        ),
        "mean_ipa_respell_f1": mean([r["f1"] for r in scored if r.get("f1") is not None]),
        "ipa_beats_respell": sum(1 for r in scored if r["winner"] == "ipa"),
        "respell_beats_ipa": sum(1 for r in scored if r["winner"] == "respell"),
        "tie": sum(1 for r in scored if r["winner"] == "tie"),
        "n_agree_ge_0.80": len(agrees),
        "n_disagree_lt_0.80": len(disagrees),
        "when_agree_winner_ipa": rate(agrees, "ipa"),
        "when_agree_winner_respell": rate(agrees, "respell"),
        "when_agree_tie": rate(agrees, "tie"),
        "when_disagree_winner_ipa": rate(disagrees, "ipa"),
        "when_disagree_winner_respell": rate(disagrees, "respell"),
        "when_disagree_tie": rate(disagrees, "tie"),
    }


def write_clip_listen(rows: list[dict], clips: dict) -> Path:
    CLIP_LISTEN.mkdir(parents=True, exist_ok=True)
    ranked = [r for r in rows if r.get("ipa_vs_human") is not None]
    ranked.sort(key=lambda r: r.get("delta_ipa_respell") or 0)
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>IPA vs source respelling vs human clip</title>",
        "<style>body{font:16px/1.4 system-ui;max-width:780px;margin:2rem auto;padding:0 1rem}",
        "section{border:1px solid #ccc;border-radius:8px;padding:1rem 1.2rem;margin:1rem 0}",
        "section.ipa{border-color:#2a6} section.resp{border-color:#a44}",
        "h1{font-size:1.25rem} h2{font-size:1.05rem;margin:0 0 .35rem}",
        "p{margin:.2rem 0 .55rem;color:#333} .ipa{font-family:ui-monospace,monospace}",
        "label{display:block;font-weight:600;margin:.5rem 0 .15rem} audio{width:100%}",
        ".meta{font-size:.9rem;color:#555}</style>",
        "<h1>Path 2 F1: Google IPA vs source respelling vs human</h1>",
        "<p>Same human clip. IPA arm is Cloud customPronunciations. "
        "Respelling arm is SSML English syllables from the original source string "
        "(not G2P to IPA). Sorted by IPA minus respelling (respelling wins first).</p>",
    ]
    for rec in ranked:
        s = slug(rec["ingredient"])
        w = rec.get("winner") or winner(rec["ipa_vs_human"], rec["respell_vs_human"])
        cls = "ipa" if w == "ipa" else ("resp" if w == "respell" else "")
        parts.append(f"<section class='{cls}' id='{s}'><h2>{rec['ingredient']}</h2>")
        parts.append(
            "<p class='meta'>"
            f"IPA vs human {rec['ipa_vs_human']:.3f} · "
            f"respell vs human {rec['respell_vs_human']:.3f} · "
            f"Δ {rec.get('delta_ipa_respell'):+.3f} · "
            f"IPA↔respell F1 {rec.get('f1') if rec.get('f1') is not None else '—'} · "
            f"winner {w}</p>"
        )
        parts.append(
            f"<p>source respelling: <span class='ipa'>{rec.get('respelling') or '—'}</span><br>"
            f"spoken as: <span class='ipa'>{rec.get('alias') or '—'}</span><br>"
            f"IPA used: <span class='ipa'>{rec.get('ipa_used') or rec.get('ipa_wav_src') or '—'}</span></p>"
        )
        human = clips.get(rec["ingredient"])
        if human is not None:
            dest = CLIP_LISTEN / f"{s}__human{human.path.suffix}"
            dest.write_bytes(human.path.read_bytes())
            parts.append(
                f"<label>human ({human.source})</label>"
                f"<audio controls src='{dest.name}'></audio>"
            )
        for label, kind in (
            ("Google IPA", "ipa"),
            ("Source respelling", "respell"),
        ):
            src = TTS / f"{s}.{kind}.wav"
            dest = CLIP_LISTEN / f"{s}__{kind}.wav"
            if src.exists():
                dest.write_bytes(src.read_bytes())
                parts.append(
                    f"<label>{label}</label><audio controls src='{dest.name}'></audio>"
                )
        parts.append("</section>")
    dest = CLIP_LISTEN / "index.html"
    dest.write_text("\n".join(parts) + "\n")
    return dest


def bump_variant(parts: list[str], name: str, cache: dict, seed: dict, used: list[str]) -> list[str] | None:
    """Advance the first part that still has an unused IPA variant."""
    for i, part in enumerate(parts):
        opts = variants_for_part(part, name, cache, seed)
        if used[i] in opts:
            idx = opts.index(used[i])
            if idx + 1 < len(opts):
                nxt = list(used)
                nxt[i] = opts[idx + 1]
                return nxt
        elif opts:
            nxt = list(used)
            nxt[i] = opts[0]
            return nxt
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--clips", action="store_true", help="Score vs human clips (Path 2 F1).")
    ap.add_argument("--refresh-google", action="store_true")
    ap.add_argument("--skip-google", action="store_true")
    args = ap.parse_args()

    pron = load_pron()
    clips = available_clips()
    seed = load_seed_ipa()
    cache = load_parts_cache()
    if args.clips and not args.refresh_google:
        args.skip_google = True
    names = target_names(pron, clips, args.only, clips_only=args.clips)
    print(
        f"{len(names)} names "
        f"({'clip+respelling F1 vs human' if args.clips else 'multi-part first'})",
        flush=True,
    )

    tok = token()
    TTS.mkdir(parents=True, exist_ok=True)

    need_google: list[str] = []
    if not args.skip_google:
        seen: set[str] = set()
        for name in names:
            for part in spoken_parts(name):
                key = part.lower()
                if key in seen:
                    continue
                seen.add(key)
                if args.refresh_google or key not in cache:
                    need_google.append(part)
        print(f"google {len(need_google)} parts", flush=True)
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = [ex.submit(google_part, p, cache, args.refresh_google) for p in need_google]
            for fut in as_completed(futs):
                fut.result()
        save_parts_cache(cache)

    rows: list[dict] = []
    for name in names:
        rec = pron[name]
        parts = spoken_parts(name)
        used = pick_ipas(parts, name, cache, seed)
        print(f"=== {name} parts={parts} ipa={used} alias={respelling_alias(rec.get('respelling') or '', name)!r}", flush=True)
        try:
            row = score_name(
                tok,
                name,
                rec,
                used,
                clips,
                skip_ctc=args.clips,
                reuse=args.clips,
            )
        except Exception as exc:
            print(f"  FAIL {name}: {exc}", flush=True)
            rows.append(
                {
                    "ingredient": name,
                    "error": str(exc)[:300],
                    "parts": parts,
                    "respelling": rec.get("respelling") or "",
                }
            )
            continue
        tries = 0
        while (
            not args.clips
            and row.get("ctc") is not None
            and row["ctc"] < LOW_CTC
            and tries < 3
            and not args.skip_google
        ):
            nxt = bump_variant(parts, name, cache, seed, used)
            if nxt is None:
                break
            tries += 1
            print(f"  retry {name} {nxt} (ctc {row['ctc']})", flush=True)
            try:
                cand = score_name(tok, name, rec, nxt, clips)
            except Exception as exc:
                print(f"  retry fail {name}: {exc}", flush=True)
                break
            if cand.get("ctc") is not None and cand["ctc"] >= (row.get("ctc") or 0) + KEEP:
                print(
                    f"  KEEP {name} {row['ctc']:.3f}->{cand['ctc']:.3f} {cand.get('ipa_used')}",
                    flush=True,
                )
                row = cand
                used = nxt
            else:
                print(
                    f"  no-keep {name} {cand.get('ctc')} vs {row.get('ctc')}",
                    flush=True,
                )
                break
        print(
            f"  {name} ctc={row.get('ctc')} f1={row.get('f1')} "
            f"ipa_h={row.get('ipa_vs_human')} resp_h={row.get('respell_vs_human')} "
            f"{row.get('ipa_used') or row.get('ipa_wav_src') or ''}",
            flush=True,
        )
        rows.append(row)

    if args.only and not args.clips and RESULTS.exists():
        prev = {
            r["ingredient"]: r for r in json.loads(RESULTS.read_text())["rows"]
        }
        for rec in rows:
            prev[rec["ingredient"]] = rec
        rows = list(prev.values())

    scored = [r for r in rows if r.get("ctc") is not None or r.get("f1") is not None]
    summary = {
        "n": len(rows),
        "n_scored": len(scored),
        "mean_ctc": round(sum(r["ctc"] for r in scored if r.get("ctc") is not None) / max(1, sum(1 for r in scored if r.get("ctc") is not None)), 4) if any(r.get("ctc") is not None for r in scored) else None,
        "mean_f1": round(sum(r["f1"] for r in scored if r.get("f1") is not None) / max(1, sum(1 for r in scored if r.get("f1") is not None)), 4) if any(r.get("f1") is not None for r in scored) else None,
        "n_low_ctc": sum(1 for r in scored if r.get("ctc") is not None and r["ctc"] < LOW_CTC),
        "keep": KEEP,
        "low_ctc": LOW_CTC,
    }
    verify = {}
    if VERIFY_RESULTS.exists():
        verify = {
            r["ingredient"]: r
            for r in json.loads(VERIFY_RESULTS.read_text())["rows"]
        }
    if args.clips:
        summary["human"] = human_calibration(rows, verify)
        OUT.mkdir(parents=True, exist_ok=True)
        CLIP_RESULTS.write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False)
            + "\n"
        )
        html = write_clip_listen(rows, clips)
        print(json.dumps(summary, indent=2), flush=True)
        print(CLIP_RESULTS, flush=True)
        print(html, flush=True)
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False) + "\n"
    )
    html = write_listen(rows, clips)
    print(json.dumps(summary), flush=True)
    print(html, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
