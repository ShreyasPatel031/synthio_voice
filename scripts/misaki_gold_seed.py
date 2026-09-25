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
  * replaces a stored string only when a gold-derived candidate beats it
    by more than 0.005 sentence CTC F1, scored at the same speed
  * never overwrites a string that is winning on F1

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
# Locked. Speed 0.5 made clips ~2x longer so recall/F1 matched the teacher
# duration even when the vowels were wrong. Do not add a speed flag.
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
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args(argv)

    if SPEED != 1.0:
        raise SystemExit("Kokoro search speed is locked at 1.0")
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
        keys = sorted(drifted, key=lambda k: best_f1.get(k, 1.0))
    if args.limit:
        keys = keys[: args.limit]
    keys = sorted(keys, key=lambda k: best_f1.get(k, 1.0))
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise SystemExit(f"bad shard {args.shard}/{args.shards}")
    keys = [k for i, k in enumerate(keys) if i % args.shards == args.shard]
    report_path = args.report or (STORE / f"gold-seed-shard-{args.shard}.json")

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
        if current:
            to_score.append(("current", current))
        for src, phones in cands:
            if phones != current:
                to_score.append((src, phones))
        scored: list[tuple[float, str, str, Path]] = []
        for src, phones in to_score:
            dest = trial / "sent" / f"{key}-{src}.wav"
            try:
                _iter.write_sentence(pipeline, it, phones, dest, speed=SPEED)
                f1 = _score_sentence(dest, it, gold)
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
        best_gold = max(gold_rows, key=lambda r: r[0]) if gold_rows else None
        current_f1 = current_row[0] if current_row else best_f1.get(key)
        if best_gold is None or current_f1 is None:
            keep = False
            reason = "keep-current"
        else:
            keep = best_gold[2] != current and best_gold[0] > current_f1 + KEEP_DELTA
            reason = "gold-better" if keep else "keep-current"
        row = {
            "slug": key,
            "ipa": ipa,
            "current": current,
            "current_is_gold": current_is_gold,
            "current_f1": None if current_f1 is None else round(current_f1, 4),
            "gold_src": None if best_gold is None else best_gold[1],
            "gold_misaki": None if best_gold is None else best_gold[2],
            "gold_f1": None if best_gold is None else round(best_gold[0], 4),
            "decision": reason,
            "kept": keep,
        }
        rows.append(row)
        _iter.log_row({
            "round": "gold-seed",
            "slug": key,
            "drug": it["drug"],
            "phonemes": best_gold[2] if keep and best_gold else current,
            "sentence_f1": round((best_gold[0] if keep and best_gold else current_f1) or 0, 4),
            "previous_f1": None if current_f1 is None else round(current_f1, 4),
            "speed": SPEED,
            "source": reason,
            "kept": keep,
        })
        if keep and best_gold is not None:
            best_f1[key] = best_gold[0]
            best_phones[key] = best_gold[2]
            (best_dir / f"{key}.wav").write_bytes(best_gold[3].read_bytes())
            switched += 1
        elif current_row is not None:
            (best_dir / f"{key}.wav").write_bytes(current_row[3].read_bytes())
        print(
            f"{n}/{len(keys)} {key} {reason} "
            f"cur={row['current_f1']} gold={row['gold_f1']} "
            f"{(best_gold[2] if best_gold else current)}",
            flush=True,
        )
        if n % 10 == 0:
            report_path.write_text(json.dumps(
                {"speed": SPEED, "shard": args.shard, "switched": switched, "rows": rows},
                indent=2,
            ))

    # Shard reports only. chosen.json is merged after every shard finishes
    # so a half-done run cannot overwrite the stored strings.
    report_path.write_text(json.dumps(
        {"speed": SPEED, "shard": args.shard, "switched": switched, "rows": rows},
        indent=2,
    ))
    print(f"WROTE {report_path} switched={switched}", flush=True)
    return 0


_GOLD_EMB: dict[str, object] = {}


def _score_sentence(wav: Path, item: dict, gold: Path) -> float | None:
    from dose_r.forced_align import extract_drug_span_forced_align
    from dose_r.scoring.speech_similarity import extract_frame_embeddings, speech_bertscore

    raw = wav.read_bytes()
    span = extract_drug_span_forced_align(raw, item["sentence"], item["spoken"])
    if span is None:
        span = extract_drug_span_forced_align(raw, item["sentence"], item["drug"])
    if span is None:
        return None
    emb_c = extract_frame_embeddings(span)
    key = str(gold)
    emb_g = _GOLD_EMB.get(key)
    if emb_g is None:
        emb_g = extract_frame_embeddings(gold.read_bytes())
        _GOLD_EMB[key] = emb_g
    return float(speech_bertscore(emb_c, emb_g)["f1"])


if __name__ == "__main__":
    raise SystemExit(main())
