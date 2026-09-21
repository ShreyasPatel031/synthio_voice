#!/usr/bin/env python3
"""Last-4 SFT from Base on 32 hard names using new Gemini+IPA sentence teachers.

Isolated gold word wavs are included as extra rows (read-only gold).
Do not edit data/gold_gemini_ipa. Do not overwrite A-32.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import ft_light_hard_keep as m  # noqa: E402
import ft_light_hard_n as n  # noqa: E402

WORK = ROOT / "runs" / "ft-hard32-sent"
MANIFEST = WORK / "teachers" / "manifest.json"
SPEAKER = "dose_sent32"
REPEATS = 4
EPOCHS = 2


def write_train(path: Path, gold: dict, names: list[str]) -> int:
    teachers = json.loads(MANIFEST.read_text())
    n_rows = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in teachers:
            audio = str((WORK / "teachers" / Path(rec["wav"]).name).resolve())
            if not Path(audio).exists():
                raise SystemExit(f"missing teacher wav: {audio}")
            row = {"audio": audio, "text": rec["text"], "ref_audio": audio}
            for _ in range(REPEATS):
                f.write(json.dumps(row) + "\n")
                n_rows += 1
        for name in names:
            g = gold[name.lower()]
            audio = str((ROOT / g["audio"]).resolve())
            row = {"audio": audio, "text": g["spoken_text"], "ref_audio": audio}
            for _ in range(REPEATS):
                f.write(json.dumps(row) + "\n")
                n_rows += 1
    return n_rows


def main() -> None:
    if not MANIFEST.exists():
        raise SystemExit(f"missing teachers {MANIFEST}; run synth_hard32_sent_teachers.py")
    names = n.lowest(32)
    gold = m.load_gold()
    base_rows = m.load_base_rows()
    raw, coded, ckpt = WORK / "train_raw.jsonl", WORK / "train_coded.jsonl", WORK / "ckpt"
    n_rows = write_train(raw, gold, names)
    print(f"=== hard32-sent rows={n_rows} names={len(names)} batch=8 epochs={EPOCHS} last-4 lr={m.LR} ===", flush=True)
    m.train(raw, coded, ckpt, SPEAKER, batch_size=8, epochs=EPOCHS)
    curve = []
    for ep in range(EPOCHS):
        c = ckpt / f"checkpoint-epoch-{ep}"
        if not c.exists():
            print("missing", c, flush=True)
            break
        print(f"--- probe epoch {ep} ---", flush=True)
        hard_p = m.probe(c, SPEAKER, names, f"hard32-sent-ep{ep}-hard", gold, base_rows)
        keep_p = m.probe(c, SPEAKER, m.KEEP, f"hard32-sent-ep{ep}-keep", gold, base_rows)
        block = {
            "epoch": ep,
            "hard": {k: hard_p[k] for k in ("mean_base", "mean_ft", "n_up", "n_down")},
            "keep": {k: keep_p[k] for k in ("mean_base", "mean_ft", "n_up", "n_down")},
        }
        curve.append(block)
        print(block, flush=True)
    (WORK / "curve.json").write_text(json.dumps(curve, indent=2))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
