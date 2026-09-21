#!/usr/bin/env python3
"""Overnight scale: 64, 128, 192 lowest names, 1 epoch, last-4, then full CTC.

From Base each time. Gold files are read only. Mini-probe 12 names, then
official 286 after all three trains so we are not idle between evals.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import ft_light_hard_keep as m  # noqa: E402

SIZES = [64, 128, 192]
PROBE_N = 12


def lowest(n: int) -> list[str]:
    gold = m.load_gold()
    rows = list(m.load_base_rows().values())
    rows.sort(key=lambda r: r["gemini_f1"])
    names, seen = [], set()
    for rec in rows:
        key = rec["drug"].lower()
        if key in seen or key not in gold:
            continue
        seen.add(key)
        names.append(rec["drug"])
        if len(names) >= n:
            break
    return names


def train_one(n: int) -> dict:
    names = lowest(n)
    gold = m.load_gold()
    base_rows = m.load_base_rows()
    speaker = f"dose_light_a{n}"
    vdir = m.WORK / f"A-hard-{n}"
    vdir.mkdir(parents=True, exist_ok=True)
    raw, coded, ckpt = vdir / "train_raw.jsonl", vdir / "train_coded.jsonl", vdir / "ckpt"
    n_rows = m.write_train(raw, gold, base_rows, names, [])
    print(f"=== train {n} rows={n_rows} batch=8 epoch=1 last-4 lr={m.LR} ===", flush=True)
    print("names", names[:8], "...", names[-4:], flush=True)
    m.train(raw, coded, ckpt, speaker, batch_size=8, epochs=1)
    c = ckpt / "checkpoint-epoch-0"
    probe_names = names[: PROBE_N // 2] + names[-PROBE_N // 2 :]
    print(f"--- mini probe {n} ---", flush=True)
    hard_p = m.probe(c, speaker, probe_names, f"A-hard-{n}-mini", gold, base_rows)
    out = {
        "n": n,
        "speaker": speaker,
        "ckpt": str(c),
        "probe_mean_base": hard_p["mean_base"],
        "probe_mean_ft": hard_p["mean_ft"],
        "n_up": hard_p["n_up"],
        "n_down": hard_p["n_down"],
        "names": names,
        "probe_rows": hard_p["rows"],
    }
    (vdir / "mini.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in out if k not in ("names", "probe_rows")}, indent=2), flush=True)
    return out


def eval_full(block: dict) -> None:
    import subprocess
    n = block["n"]
    wav_dir = ROOT / "runs" / "oss-eval" / f"qwen-ft-light-a{n}-ep0"
    print(f"=== full 286 n={n} ===", flush=True)
    subprocess.check_call([
        sys.executable, str(ROOT / "scripts" / "eval_custom_voice_full.py"),
        block["ckpt"], block["speaker"], str(wav_dir), f"qwen-ft-light-a{n}-ep0",
    ])


def main() -> None:
    m.WORK.mkdir(parents=True, exist_ok=True)
    results = []
    for n in SIZES:
        results.append(train_one(n))
        (m.WORK / "overnight.json").write_text(json.dumps(
            [{k: r[k] for k in r if k not in ("names", "probe_rows")} for r in results],
            indent=2,
        ))
    ranked = sorted(results, key=lambda r: (r["probe_mean_ft"] or 0), reverse=True)
    print("eval order by mini probe", [(r["n"], r["probe_mean_ft"]) for r in ranked], flush=True)
    for block in ranked:
        eval_full(block)


if __name__ == "__main__":
    main()
