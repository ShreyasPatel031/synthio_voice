#!/usr/bin/env python3
"""Score existing speed-1 Misaki sentence wavs against Cloud gold wavs.

Does not synthesize. Does not write chosen.json or data/gold_gemini_ipa.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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
OUT = STORE / "cloud-rank.json"


def main() -> int:
    pins = json.loads(PINS.read_text()) if PINS.exists() else {}
    items = {it["slug"]: it for it in _iter.load_full_items()}
    rows = []
    missing = []
    for slug, it in items.items():
        gold_locked = _iter.gold_wav(it["drug"])
        if gold_locked is None:
            missing.append((slug, "no-locked-gold"))
            continue
        gold = CLOUD / gold_locked.name
        if not gold.exists():
            missing.append((slug, "no-cloud-copy"))
            continue
        if slug in pins:
            wav = USER_SENT / f"{slug}.wav"
            kind = "pin"
        else:
            wav = SENT / f"{slug}-current.wav"
            kind = "stored"
        if not wav.exists():
            missing.append((slug, f"no-{kind}-wav"))
            continue
        f1 = _seed._score_sentence(wav, it, gold)
        if f1 is None:
            missing.append((slug, "no-span"))
            continue
        rows.append({
            "slug": slug,
            "drug": it["drug"],
            "kind": kind,
            "ctc_f1": round(float(f1), 4),
        })
        print(f"{len(rows)} {slug} {kind} {rows[-1]['ctc_f1']}", flush=True)
    rows.sort(key=lambda r: r["ctc_f1"])
    OUT.write_text(json.dumps({"n": len(rows), "missing": missing, "rows": rows}, indent=2))
    print(f"WROTE {OUT} n={len(rows)} missing={len(missing)}", flush=True)
    print("LOWEST")
    for r in rows[:20]:
        print(f"{r['ctc_f1']:.4f} {r['slug']} {r['kind']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
