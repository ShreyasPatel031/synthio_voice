#!/usr/bin/env python3
"""Export CTC-cut wavs for combo-only ingredients into an oss-eval wav dir.

DoSE has 274 product rows / 286 spans. Synth writes one wav per product name.
Nineteen ingredients exist only inside combination sentences, so a unique-
ingredient inventory shows them missing (and Advair brand has no salmeterol
span — salmeterol lives on the fluticasone+salmeterol combo row).

This cuts each combo ingredient from its product sentence wav and writes
{ingredient-slug}.wav next to the product wavs. Existing solo product wavs
are left alone.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import wave
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dose_r.forced_align import extract_drug_span_forced_align

DOSE = ROOT / "data" / "dose_v1.jsonl"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def wav_bytes(path: Path) -> bytes:
    audio, sr = sf.read(path, dtype="int16", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1).astype("int16")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())
    return buf.getvalue()


def write_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # span bytes are already WAV from forced_align
    path.write_bytes(pcm)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", type=Path, required=True)
    ap.add_argument("--force", action="store_true", help="overwrite existing ingredient wavs")
    args = ap.parse_args()
    wav_dir: Path = args.wav_dir
    if not wav_dir.is_dir():
        raise SystemExit(f"missing wav dir: {wav_dir}")

    rows = [json.loads(l) for l in DOSE.read_text().splitlines() if l.strip()]
    product_names = {r["name"].lower() for r in rows}
    combos = [r for r in rows if r.get("is_combination")]

    # ingredients that have no dedicated product row
    need: list[tuple[dict, str]] = []
    for r in combos:
        for ing in r["ingredients"]:
            if ing.lower() in product_names:
                continue
            need.append((r, ing))

    print(f"combo-only ingredients to export: {len(need)} from {len(combos)} combos", flush=True)
    ok = fail = skip = 0
    for row, ing in need:
        dest = wav_dir / f"{slug(ing)}.wav"
        if dest.exists() and not args.force:
            print(f"skip exists {dest.name}", flush=True)
            skip += 1
            continue
        src = wav_dir / f"{slug(row['name'])}.wav"
        if not src.exists():
            print(f"FAIL missing product wav {src.name} for {ing}", flush=True)
            fail += 1
            continue
        try:
            span = extract_drug_span_forced_align(
                wav_bytes(src), row["sentence"], ing
            )
        except Exception as exc:
            print(f"FAIL align {ing}: {exc}", flush=True)
            fail += 1
            continue
        if span is None:
            print(f"FAIL align_none {ing} from {src.name}", flush=True)
            fail += 1
            continue
        write_wav(dest, span)
        print(f"ok {ing} <- {src.name} -> {dest.name} ({len(span)} bytes)", flush=True)
        ok += 1

    # coverage check: unique ingredients with a wav
    uniq = {}
    for r in rows:
        for ing in r["ingredients"]:
            uniq.setdefault(ing.lower(), ing)
    have = sum(1 for ing in uniq.values() if (wav_dir / f"{slug(ing)}.wav").exists())
    summary = {
        "wav_dir": str(wav_dir),
        "exported": ok,
        "skipped_existing": skip,
        "failed": fail,
        "unique_ingredients": len(uniq),
        "unique_with_wav": have,
        "product_wavs": len(list(wav_dir.glob("*.wav"))),
    }
    (wav_dir / "combo_span_export.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
