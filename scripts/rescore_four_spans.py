#!/usr/bin/env python3
"""Rescore the four <0.6 slugs after a forced-align span tweak."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SLUGS = ("icotyde", "idvynso", "vorasidenib", "advair")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_iter = _load("misaki_iter", ROOT / "scripts" / "misaki_bottom_iterate.py")
_seed = _load("misaki_gold_seed", ROOT / "scripts" / "misaki_gold_seed.py")
STORE = _iter.STORE
CLOUD = STORE / "cloud-gold"
PINS = STORE / "user_pins.json"
SENT = STORE / "gold-seed" / "sent"
USER_SENT = STORE / "user-sent"


def wav_for(slug: str, pins: dict) -> Path | None:
    if slug in pins:
        p = USER_SENT / f"{slug}.wav"
        return p if p.exists() else None
    p = SENT / f"{slug}-current.wav"
    return p if p.exists() else None


def main() -> int:
    pins = json.loads(PINS.read_text()) if PINS.exists() else {}
    items = {it["slug"]: it for it in _iter.load_full_items()}
    print("slug           old_f1  new_f1  delta   wav")
    for slug in SLUGS:
        it = items[slug]
        gold_locked = _iter.gold_wav(it["drug"])
        gold = CLOUD / gold_locked.name
        wav = wav_for(slug, pins)
        if wav is None:
            print(f"{slug:14s} MISSING WAV")
            continue
        old_rank = next(r for r in json.loads((STORE / "cloud-rank.json").read_text())["rows"] if r["slug"] == slug)
        new_f1 = _seed._score_sentence(wav, it, gold)
        old_f1 = old_rank["ctc_f1"]
        delta = round(new_f1 - old_f1, 4) if new_f1 is not None else None
        print(f"{slug:14s} {old_f1:.4f}  {new_f1:.4f}  {delta:+.4f}  {wav.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
