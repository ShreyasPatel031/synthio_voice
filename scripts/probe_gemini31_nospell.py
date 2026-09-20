#!/usr/bin/env python
"""Stop Gemini 3.1 letter-spelling ALL-CAPS respelling syllables.

`eye-va-KAF-tor` / `PLAV-iks` / `AH-wik-lee` get read as acronyms.
Try lowercase hyphens, title-case stress, and an anti-spell prompt.
Never G2P respelling to IPA.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
_hf = ROOT / ".cache" / "huggingface"
_hf.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_hf))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_hf))
os.environ.setdefault("HF_HUB_CACHE", str(_hf / "hub"))

import librosa  # noqa: E402
import requests  # noqa: E402

from eval_gemini31_holdout import (  # noqa: E402
    ENDPOINT,
    MODEL,
    PROJECT,
    RATE,
    VOICE,
    cached_synth,
    cloud_ipa_path,
    slug,
    synth_gemini,
    token,
)
from dose_r.scoring.phoneme_model import transcribe_phonemes  # noqa: E402

OUT = ROOT / "runs" / "gemini31-nospell"
TTS = OUT / "tts"
LISTEN = ROOT / "runs" / "listen-gemini31-nospell"
HOLD = ROOT / "runs" / "gemini31-respell-all" / "tts"

DEFAULT_PROMPT = "Pronounce this US drug name clearly as a single name."
NOSPELL_PROMPT = (
    "Say this as one spoken US drug name. Hyphens mark syllables. "
    "Do not spell any letters. Do not pause between letters."
)

CASES = [
    ("ivacaftor", "eye-va-KAF-tor"),
    ("Plavix", "PLAV-iks"),
    ("Awiqli", "AH-wik-lee"),
    ("Veppanu", "VEP-uh-new"),
    ("Rinvoq", "RIN-voke"),
]


def title_stress(canonical: str) -> str:
    """PLAV-iks -> Plav-iks; eye-va-KAF-tor -> eye-va-Kaf-tor."""

    def repl(m: re.Match[str]) -> str:
        w = m.group(0)
        return w[0] + w[1:].lower()

    return re.sub(r"[A-Z]{2,}", repl, canonical)


def synth_with_prompt(text: str, prompt: str) -> bytes:
    import base64
    import time

    body = {
        "input": {"text": text, "prompt": prompt},
        "voice": {"languageCode": "en-US", "name": VOICE, "modelName": MODEL},
        "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": RATE},
    }
    last: Exception | None = None
    for attempt in range(6):
        try:
            resp = requests.post(
                ENDPOINT,
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
            raise RuntimeError(f"{resp.status_code}: {resp.text[:400]}")
        return base64.b64decode(resp.json()["audioContent"])
    raise last or RuntimeError("synth failed")


def write_wav(path: Path, text: str, prompt: str) -> Path:
    if path.exists() and path.stat().st_size > 500:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(synth_with_prompt(text, prompt))
    return path


def dur(path: Path) -> float:
    audio, sr = librosa.load(str(path), sr=None, mono=True)
    return len(audio) / sr


def looks_spelled(ctc: str) -> bool:
    """Letter-name runs: A=eɪ, H=eɪ tʃ, K=k eɪ, F=ɛ f, P=p iː, L=ɛ l, V=v iː."""
    s = " " + ctc + " "
    hits = 0
    for pat in (
        r" eɪ tʃ ",
        r" k eɪ ",
        r" p iː ",
        r" ɛ l ",
        r" ɛ s ",
        r" ɛ n ",
        r" ɛ m ",
        r" ɛ f ",
        r" t iː ",
        r" d iː ",
        r" aɪ ɛ ",
        r" eɪ ɑːɹ ",
        r" v iː ",
        r" eɪ eɪ ",
    ):
        if re.search(pat, s):
            hits += 1
    return hits >= 2


def main() -> int:
    TTS.mkdir(parents=True, exist_ok=True)
    LISTEN.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, orig in CASES:
        s = slug(name)
        variants = [
            ("orig_caps", orig, DEFAULT_PROMPT),
            ("lower_hyphen", orig.lower(), DEFAULT_PROMPT),
            ("title_stress", title_stress(orig), DEFAULT_PROMPT),
            ("orig_nospell_prompt", orig, NOSPELL_PROMPT),
            ("lower_nospell_prompt", orig.lower(), NOSPELL_PROMPT),
        ]
        rec = {"ingredient": name, "orig": orig, "variants": {}}
        for kind, text, prompt in variants:
            path = TTS / f"{s}.{kind}.wav"
            if kind == "orig_caps":
                alt = HOLD / f"{s}.hyphen.wav"
                if alt.exists() and not path.exists():
                    shutil.copy2(alt, path)
            print(f"synth {name} {kind} {text!r}", flush=True)
            write_wav(path, text, prompt)
            ctc = transcribe_phonemes(path)
            d = round(dur(path), 3)
            spelled = looks_spelled(ctc)
            rec["variants"][kind] = {
                "text": text,
                "prompt": prompt,
                "dur": d,
                "ctc": ctc,
                "spelled": spelled,
            }
            print(f"  dur={d} spelled={spelled} ctc={ctc}", flush=True)
        ipa = cloud_ipa_path(name)
        if ipa is not None:
            rec["ipa_ctc"] = transcribe_phonemes(ipa)
            rec["ipa_dur"] = round(dur(ipa), 3)
            print(f"  ipa dur={rec['ipa_dur']} ctc={rec['ipa_ctc']}", flush=True)
        rows.append(rec)

    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Gemini 3.1 stop letter-spelling</title>",
        "<style>body{font:16px/1.4 system-ui;max-width:820px;margin:2rem auto;padding:0 1rem}",
        "section{border:1px solid #ccc;border-radius:8px;padding:1rem 1.2rem;margin:1rem 0}",
        "h1{font-size:1.2rem} h2{font-size:1.05rem;margin:0 0 .35rem}",
        ".ipa{font-family:ui-monospace,monospace} .bad{color:#a44} .ok{color:#161}",
        "label{display:block;font-weight:600;margin:.45rem 0 .1rem} audio{width:100%}",
        "p{margin:.2rem 0 .5rem}</style>",
        "<h1>Stop Gemini 3.1 spelling ALL-CAPS syllables</h1>",
        "<p>Same hyphenated respelling, different casing / prompt.</p>",
    ]
    for rec in rows:
        s = slug(rec["ingredient"])
        parts.append(f"<section id='{s}'><h2>{rec['ingredient']}</h2>")
        parts.append(f"<p>published <span class='ipa'>{rec['orig']}</span></p>")
        ipa = cloud_ipa_path(rec["ingredient"])
        if ipa is not None:
            dest = LISTEN / f"{s}__ipa.wav"
            if not dest.exists():
                shutil.copy2(ipa, dest)
            parts.append(
                f"<label>Cloud IPA ({rec.get('ipa_ctc','')})</label>"
                f"<audio controls src='{dest.name}'></audio>"
            )
        for kind, info in rec["variants"].items():
            src = TTS / f"{s}.{kind}.wav"
            dest = LISTEN / f"{s}__{kind}.wav"
            shutil.copy2(src, dest)
            cls = "bad" if info["spelled"] else "ok"
            parts.append(
                f"<label class='{cls}'>{kind} "
                f"<span class='ipa'>{info['text']}</span> "
                f"{info['dur']}s spelled={info['spelled']}<br>"
                f"<span class='ipa'>{info['ctc']}</span></label>"
                f"<audio controls src='{dest.name}'></audio>"
            )
        parts.append("</section>")
    html = LISTEN / "index.html"
    html.write_text("\n".join(parts) + "\n")
    (OUT / "results.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    print(html, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
