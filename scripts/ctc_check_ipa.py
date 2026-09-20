"""One-shot: decode human clips with wav2vec2-espeak CTC, compare to sidecar IPA.

Used to find systematic dictionary→IPA converter bugs, not as a Path 2 score.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_hf = ROOT / ".cache" / "huggingface"
_hf.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_hf))

from dose_r.scoring.phoneme_distance import normalize_phonemes  # noqa: E402
from dose_r.scoring.phoneme_model import transcribe_phonemes  # noqa: E402

FOCUS = [
    "acoramidis", "atogepant", "Nuzolvence", "Revuforj", "famotidine",
    "Humira", "acetaminophen", "Adquey", "Dupixent", "Nexium",
    "tofacitinib", "omalizumab", "upadacitinib", "Vraylar", "Wegovy",
    "Xeljanz", "Meibo", "Vabysmo", "Advair", "Nurtec", "Imaavy",
    "sotatercept-csrk", "bevacizumab-vikg", "fezolinetant", "remibrutinib",
    "Januvia", "Aspirin", "Plavix", "aripiprazole", "Jardiance",
]


def compact(s: str) -> str:
    return normalize_phonemes(s).replace(" ", "")


def main() -> int:
    scored = {
        r["ingredient"].lower(): r
        for r in json.loads((ROOT / "runs/uniform-pron-tts/results.json").read_text())["rows"]
    }
    print(f"{'name':22} {'dlt_ipa':>7} {'sidecar':28} {'ctc_human'}")
    for name in FOCUS:
        rec = scored.get(name.lower())
        if not rec:
            print(f"{name:22} missing from scored set")
            continue
        ctc = transcribe_phonemes(rec["human_path"])
        side = rec["ipa"]
        print(
            f"{name:22} {rec['delta_ipa']:+7.3f} {compact(side):28} {compact(ctc)}"
            f"   raw={ctc!r}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
