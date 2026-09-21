#!/usr/bin/env python3
"""Seed Kokoro Misaki from locked gold IPA and rescore at speed 1.0.

What was wrong:
  The Gemini + last-vowel loop mutated stored strings to raise CTC F1.
  A keep still required +0.005 F1, but the *candidates* were no longer the
  gold pronunciation (hard g for ʤ, stress on the last syllable, etc.).

This loop:
  * builds 1–3 Misaki strings from the locked gold IPA (fold + restress)
  * synthesizes at speed 1.0 on GPU
  * never calls Gemini and never swaps a vowel just to chase F1
  * freezes names in user_pins.json
  * replaces a stored string only when a gold-derived candidate wins F1
    by more than 0.005, or immediately when the stored string is not a
    gold-IPA fold (that is the drift the old loop introduced)

Does not write data/gold_gemini_ipa.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dose_r.misaki_fold import gold_misaki_candidates  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_iter = _load("misaki_iter", ROOT / "scripts" / "misaki_bottom_iterate.py")
_kokoro = _load("synth_kokoro_hard", ROOT / "scripts" / "synth_kokoro_hard.py")

STORE = _iter.STORE
WAV_ROOT = _iter.WAV_ROOT
SPEED = 1.0
KEEP_DELTA = 0.005
PINS = STORE / "user_pins.json"
GOLD_SYNC = STORE / "gold-sync"
REPORT = STORE / "gold-seed-report.json"


def pin_gold(slug: str, drug: str) -> Path | None:
    override = GOLD_SYNC / f"{slug}.wav"
    if override.exists() and override.stat().st_size > 500:
        return override
    return _iter.gold_wav(drug)


def load_pins() -> set[str]:
    if not PINS.exists():
        return set()
    return set(json.loads(PINS.read_text()))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=())
    ap.add_argument("--bottom", type=int, default=0, help="lowest N by stored CTC F1")
    ap.add_argument("--drifted", action="store_true", help="only names whose stored Misaki is not gold-derived")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    STORE.mkdir(parents=True, exist_ok=True)
    chosen_path = STORE / "chosen.json"
    chosen = json.loads(chosen_path.read_text()) if chosen_path.exists() else {"phones": {}, "f1": {}}
    best_phones = dict(chosen.get("phones") or {})
    best_f1 = {k: float(v) for k, v in (chosen.get("f1") or {}).items()}
    pins = load_pins()
    gold_ipa = _kokoro.load_gold()
    items = {it["slug"]: it for it in _iter.load_full_items()}
    symbols = _iter.vocab() | set("T ")
    best_dir = WAV_ROOT / "kokoro-full-best"
    best_dir.mkdir(parents=True, exist_ok=True)
    trial = STORE / "gold-seed"
    trial.mkdir(parents=True, exist_ok=True)

    keys = list(best_f1) or list(items)
    if args.only:
        keys = [k for k in args.only if k in items]
    elif args.bottom:
        keys = sorted(best_f1, key=lambda k: best_f1[k])[: args.bottom]
    if args.drifted:
        drifted = []
        for key in keys:
            it = items.get(key)
            if it is None:
                continue
            ipa = gold_ipa.get(it["drug"].lower(), "")
            gold_set = {p for _, p in gold_misaki_candidates(ipa, it["spoken"], symbols)}
            if best_phones.get(key, "") not in gold_set:
                drifted.append(key)
        keys = drifted
    if args.limit:
        keys = keys[: args.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=device)
    print(
        f"device={device} speed={SPEED} n={len(keys)} pins_frozen={len(pins)}",
        flush=True,
    )

    rows = []
    switched = 0
    for n, key in enumerate(keys, 1):
        it = items.get(key)
        if it is None:
            continue
        if key in pins:
            print(f"skip pin {key}", flush=True)
            rows.append({"slug": key, "decision": "pin-frozen", "misaki": best_phones.get(key)})
            continue
        ipa = gold_ipa.get(it["drug"].lower(), "")
        gold = pin_gold(key, it["drug"])
        if not ipa or gold is None:
            print("missing gold", key, flush=True)
            continue
        current = best_phones.get(key, "")
        cands = gold_misaki_candidates(ipa, it["spoken"], symbols)
        gold_set = {p for _, p in cands}
        current_is_gold = current in gold_set
        to_score: list[tuple[str, str]] = []
        # Drifted strings scored higher by gaming CTC. Do not resynth them.
        if current and current_is_gold:
            to_score.append(("current", current))
        for src, phones in cands:
            if phones != current:
                to_score.append((src, phones))
        scored: list[tuple[float, str, str, Path]] = []
        for src, phones in to_score:
            dest = trial / "sent" / f"{key}-{src}.wav"
            try:
                _iter.write_sentence(pipeline, it, phones, dest, speed=SPEED)
                f1 = _iter.sentence_f1(dest, it, gold)
            except Exception as exc:
                print("fail", key, src, phones, exc, flush=True)
                continue
            if f1 is None:
                continue
            scored.append((float(f1), src, phones, dest))
            _iter.log_row({
                "round": "gold-seed",
                "slug": key,
                "drug": it["drug"],
                "phonemes": phones,
                "sentence_f1": round(float(f1), 4),
                "previous_f1": None if key not in best_f1 else round(best_f1[key], 4),
                "speed": SPEED,
                "source": src,
                "kept": False,
            })
        if not scored:
            continue
        current_row = next((r for r in scored if r[1] == "current"), None)
        gold_rows = [r for r in scored if r[1] != "current"]
        if not gold_rows and current_is_gold:
            rows.append({
                "slug": key,
                "decision": "already-gold",
                "current": current,
                "ipa": ipa,
            })
            continue
        if not gold_rows:
            print("no gold cand", key, flush=True)
            continue
        best_gold = max(gold_rows, key=lambda r: r[0])
        current_f1 = current_row[0] if current_row else best_f1.get(key)
        if not current_is_gold:
            keep = True
            reason = "reset-drift"
        elif current_f1 is None:
            keep = True
            reason = "no-current"
        else:
            keep = best_gold[2] != current and best_gold[0] > current_f1 + KEEP_DELTA
            reason = "gold-better" if keep else "keep-current-gold"
        row = {
            "slug": key,
            "ipa": ipa,
            "current": current,
            "current_is_gold": current_is_gold,
            "current_f1": None if current_f1 is None else round(current_f1, 4),
            "gold_src": best_gold[1],
            "gold_misaki": best_gold[2],
            "gold_f1": round(best_gold[0], 4),
            "decision": reason,
            "kept": keep,
        }
        rows.append(row)
        _iter.log_row({
            "round": "gold-seed",
            "slug": key,
            "drug": it["drug"],
            "phonemes": best_gold[2] if keep else current,
            "sentence_f1": round(best_gold[0] if keep else (current_f1 or 0), 4),
            "previous_f1": None if current_f1 is None else round(current_f1, 4),
            "speed": SPEED,
            "source": reason,
            "kept": keep,
        })
        if keep:
            best_f1[key] = best_gold[0]
            best_phones[key] = best_gold[2]
            (best_dir / f"{key}.wav").write_bytes(best_gold[3].read_bytes())
            switched += 1
        elif current_row is not None:
            # Stored F1 was speed 0.5. Refresh the number at speed 1 if we kept current.
            best_f1[key] = current_row[0]
        print(
            f"{n}/{len(keys)} {key} {reason} "
            f"cur={row['current_f1']} gold={row['gold_f1']} {best_gold[2]}",
            flush=True,
        )
        if n % 10 == 0:
            chosen_path.write_text(json.dumps(
                {
                    "round": "gold-seed",
                    "speed": SPEED,
                    "phones": best_phones,
                    "f1": {k: round(v, 4) for k, v in best_f1.items()},
                },
                indent=2,
            ))

    chosen_path.write_text(json.dumps(
        {
            "round": "gold-seed",
            "speed": SPEED,
            "phones": best_phones,
            "f1": {k: round(v, 4) for k, v in best_f1.items()},
        },
        indent=2,
    ))
    REPORT.write_text(json.dumps({"speed": SPEED, "switched": switched, "rows": rows}, indent=2))
    print(f"WROTE {chosen_path} switched={switched} report={REPORT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
