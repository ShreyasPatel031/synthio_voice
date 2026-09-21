#!/usr/bin/env python3
"""Continue the Misaki search. Gold-IPA seeds at speed 1.0.

The old Gemini / last-vowel loop is gone. See scripts/misaki_gold_seed.py.
Does not write data/gold_gemini_ipa.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_seed = _load("misaki_gold_seed", ROOT / "scripts" / "misaki_gold_seed.py")


if __name__ == "__main__":
    raise SystemExit(_seed.main())
