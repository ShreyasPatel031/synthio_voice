#!/usr/bin/env python3
"""Regen Kokoro from user Misaki pins and rescore vs gold teacher wavs.

Does not write data/gold_gemini_ipa. Does not touch chosen.json.

If runs/misaki-iter/gold-sync/{slug}.wav exists, score against that copy
of the locked gold (VM extract can be stale). Listen clips are isolated
markdown inject at speed 1.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_iter = _load("misaki_iter", ROOT / "scripts" / "misaki_bottom_iterate.py")
_score = _load("score_dose_ctc", ROOT / "scripts" / "score_dose_ctc_vs_gemini_ipa.py")
STORE = _iter.STORE
PINS = STORE / "user_pins.json"
OUT = STORE / "user_rescore.json"
ISO_DIR = STORE / "user-iso"
SENT_DIR = STORE / "user-sent"
SPAN_DIR = STORE / "user-span"
MP3_ISO = STORE / "user-mp3" / "iso"
MP3_SENT = STORE / "user-mp3" / "sent"
GOLD_SYNC = STORE / "gold-sync"
VOICE = _iter.VOICE
LISTEN_SPEED = 1.0


def inject(sentence: str, spoken: str, phones: str) -> str:
    if spoken not in sentence:
        raise ValueError(f"{spoken!r} not in sentence")
    return sentence.replace(spoken, f"[{spoken}](/{phones}/)", 1)


def wav_to_mp3(wav: Path, mp3: Path) -> None:
    mp3.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(wav),
            "-codec:a",
            "libmp3lame",
            "-qscale:a",
            "5",
            str(mp3),
        ]
    )


def write_audio(pipeline, text: str, dest: Path, speed: float) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = next(pipeline(text, voice=VOICE, speed=speed))
    if result.audio is None:
        raise RuntimeError(f"no audio for {text!r}")
    sf.write(dest, result.audio.detach().cpu().numpy(), 24000)


def g2p_debug(pipeline, text: str) -> list[dict]:
    graphemes, tokens = pipeline.g2p(text)
    out = [{"text": t.text, "phonemes": t.phonemes} for t in tokens]
    return [{"graphemes": graphemes, "tokens": out}]


def duration(path: Path) -> float:
    info = sf.info(str(path))
    return round(info.frames / info.samplerate, 3)


def span_wav(sent: Path, item: dict, dest: Path) -> Path | None:
    from dose_r.forced_align import extract_drug_span_forced_align

    raw = sent.read_bytes()
    span = extract_drug_span_forced_align(raw, item["sentence"], item["spoken"])
    if span is None:
        span = extract_drug_span_forced_align(raw, item["sentence"], item["drug"])
    if span is None:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(span)
    return dest


def pin_gold(key: str, drug: str) -> Path | None:
    override = GOLD_SYNC / f"{key}.wav"
    if override.exists() and override.stat().st_size > 500:
        return override
    return _iter.gold_wav(drug)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=())
    args = ap.parse_args()
    pins = json.loads(PINS.read_text())
    if args.only:
        pins = {k: v for k, v in pins.items() if k in set(args.only)}
    chosen = {}
    chosen_path = STORE / "chosen.json"
    if chosen_path.exists():
        chosen = json.loads(chosen_path.read_text())
    prev_f1 = {k: float(v) for k, v in (chosen.get("f1") or {}).items()}
    prev_phones = dict(chosen.get("phones") or {})

    items = {it["slug"]: it for it in _iter.load_full_items()}
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device="cpu")
    print(f"device=cpu listen_speed={LISTEN_SPEED} n={len(pins)}", flush=True)

    rows = []
    for key, pin in pins.items():
        it = items.get(key)
        if it is None:
            print("missing item", key, flush=True)
            continue
        gold = pin_gold(key, it["drug"])
        if gold is None:
            print("missing gold", key, flush=True)
            continue
        phones = pin["misaki"]
        spoken = it["spoken"]
        word = pin.get("word") or spoken
        iso_text = f"[{word}](/{phones}/)"
        sent_text = inject(it["sentence"], spoken, phones)
        debug = g2p_debug(pipeline, iso_text)
        print(key, "gold", gold, "g2p", json.dumps(debug, ensure_ascii=False), flush=True)

        iso = ISO_DIR / f"{key}.wav"
        sent = SENT_DIR / f"{key}.wav"
        write_audio(pipeline, iso_text, iso, LISTEN_SPEED)
        write_audio(pipeline, sent_text, sent, LISTEN_SPEED)

        iso_f1 = float(_score.score_span_vs_gemini(iso.read_bytes(), gold))
        ctc_f1 = _iter.sentence_f1(sent, it, gold)
        span = span_wav(sent, it, SPAN_DIR / f"{key}.wav")
        wav_to_mp3(iso, MP3_ISO / f"{key}.mp3")
        wav_to_mp3(sent, MP3_SENT / f"{key}.mp3")

        old = prev_f1.get(key)
        row = {
            "slug": key,
            "word": word,
            "misaki": phones,
            "inject": iso_text,
            "gold": str(gold),
            "speed": LISTEN_SPEED,
            "g2p": debug,
            "iso_dur": duration(iso),
            "gold_dur": duration(gold),
            "sent_dur": duration(sent),
            "span_dur": duration(span) if span is not None else None,
            "iso_f1": round(iso_f1, 4),
            "ctc_f1": None if ctc_f1 is None else round(float(ctc_f1), 4),
            "previous_ctc_f1": None if old is None else round(old, 4),
            "previous_misaki": prev_phones.get(key),
        }
        rows.append(row)
        _iter.log_row(
            {
                "round": "user",
                "slug": key,
                "drug": it["drug"],
                "phonemes": phones,
                "iso_f1": row["iso_f1"],
                "sentence_f1": row["ctc_f1"],
                "previous_f1": row["previous_ctc_f1"],
                "speed": LISTEN_SPEED,
                "source": "user",
                "kept": False,
            }
        )
        print(
            f"{key} iso_f1={row['iso_f1']} ctc_f1={row['ctc_f1']} "
            f"iso={row['iso_dur']}s gold={row['gold_dur']}s span={row['span_dur']}s",
            flush=True,
        )

    prev_rows = []
    if OUT.exists() and args.only:
        try:
            prev_rows = json.loads(OUT.read_text()).get("rows") or []
        except json.JSONDecodeError:
            prev_rows = []
    by_slug = {r["slug"]: r for r in prev_rows}
    for r in rows:
        by_slug[r["slug"]] = r
    OUT.write_text(
        json.dumps(
            {"speed": LISTEN_SPEED, "device": "cpu", "rows": list(by_slug.values())},
            indent=2,
        )
    )
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
