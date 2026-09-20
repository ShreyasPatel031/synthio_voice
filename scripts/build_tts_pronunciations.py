"""Build a per-ingredient TTS pronunciation field from the canonical respelling.

No human clip is required. Empty respellings stay empty (we do not invent).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dose_r.references.respelling import is_canonical
from dose_r.references.tts_pronunciation import (
    compact_ascii,
    ipa_from_canonical,
    to_cloud_en_us_ipa,
)

RESPELLINGS = ROOT / "dose_r" / "references" / "respellings.jsonl"
OUT = ROOT / "dose_r" / "references" / "pronunciations.jsonl"


def main() -> int:
    rows = []
    for line in RESPELLINGS.read_text().splitlines():
        rec = json.loads(line)
        canon = rec.get("respelling") or ""
        out = {
            "ingredient": rec["ingredient"],
            "name_type": rec.get("name_type", ""),
            "respelling": canon,
            "source": rec.get("source", ""),
            "compact": "",
            "ipa": "",
            "ipa_cloud": "",
        }
        if canon and is_canonical(canon):
            ipa = ipa_from_canonical(canon)
            out["compact"] = compact_ascii(canon)
            out["ipa"] = ipa
            out["ipa_cloud"] = to_cloud_en_us_ipa(ipa)
        rows.append(out)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    sourced = sum(1 for r in rows if r["ipa"])
    print(f"wrote {OUT} n={len(rows)} with_ipa={sourced}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
