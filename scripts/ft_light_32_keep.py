#!/usr/bin/env python3
"""Last-4 SFT from Base: 32 lowest + 8 keep teachers. Do not overwrite A-32."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import ft_light_hard_keep as m  # noqa: E402
import ft_light_hard_n as n  # noqa: E402

m.EPOCHS = 2


def main() -> None:
    names = n.lowest(32)
    print(f"training {len(names)} hard + {len(m.KEEP)} keep: {names}", flush=True)
    gold = m.load_gold()
    base_rows = m.load_base_rows()
    vdir = m.WORK / "A-hard-32-keep"
    vdir.mkdir(parents=True, exist_ok=True)
    raw, coded, ckpt = vdir / "train_raw.jsonl", vdir / "train_coded.jsonl", vdir / "ckpt"
    speaker = "dose_light_a32k"
    n_rows = m.write_train(raw, gold, base_rows, names, m.KEEP)
    print(f"=== A-hard-32-keep rows={n_rows} batch=8 epochs=2 last-4 lr={m.LR} ===", flush=True)
    m.train(raw, coded, ckpt, speaker, batch_size=8, epochs=2)
    curve = []
    for ep in range(2):
        c = ckpt / f"checkpoint-epoch-{ep}"
        if not c.exists():
            print("missing", c, flush=True)
            break
        print(f"--- probe epoch {ep} ---", flush=True)
        hard_p = m.probe(c, speaker, names, f"A-hard-32-keep-ep{ep}-hard", gold, base_rows)
        keep_p = m.probe(c, speaker, m.KEEP, f"A-hard-32-keep-ep{ep}-keep", gold, base_rows)
        block = {
            "epoch": ep,
            "hard": {k: hard_p[k] for k in ("mean_base", "mean_ft", "n_up", "n_down")},
            "keep": {k: keep_p[k] for k in ("mean_base", "mean_ft", "n_up", "n_down")},
        }
        curve.append(block)
        print(block, flush=True)
    (vdir / "curve.json").write_text(__import__("json").dumps(curve, indent=2))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
