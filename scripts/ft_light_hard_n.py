#!/usr/bin/env python3
"""Light last-4 SFT from Base on the N lowest Base-vs-gold names.

Same recipe as A-hard (lr 2e-6, 4 repeats, 2 epochs, probe each epoch).
Stops if the trained mean falls. Gold files are read only.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import ft_light_hard_keep as m  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 32


def lowest(n: int) -> list[str]:
    gold = m.load_gold()
    rows = list(m.load_base_rows().values())
    rows.sort(key=lambda r: r["gemini_f1"])
    names = []
    seen = set()
    for rec in rows:
        key = rec["drug"].lower()
        if key in seen or key not in gold:
            continue
        seen.add(key)
        names.append(rec["drug"])
        if len(names) >= n:
            break
    return names


def main() -> None:
    m.EPOCHS = 2
    names = lowest(N)
    print(f"training {len(names)} lowest: {names}", flush=True)
    gold = m.load_gold()
    base_rows = m.load_base_rows()
    m.run_variant(f"A-hard-{len(names)}", f"dose_light_a{len(names)}", names, [], gold, base_rows)


if __name__ == "__main__":
    main()
