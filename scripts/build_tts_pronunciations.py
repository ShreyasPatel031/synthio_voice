"""Build per-ingredient TTS fields from the canonical *original* respelling.

Never convert that respelling to IPA. `ipa` / `ipa_cloud` stay empty unless
a later pass copies source-published IPA (not Wikipedia-key G2P).
See dose_r/references/README.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dose_r.references.respelling import is_canonical
from dose_r.references.tts_pronunciation import compact_ascii

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
            "compact": compact_ascii(canon) if canon and is_canonical(canon) else "",
            # Banned converter used to fill these. Keep empty.
            "ipa": "",
            "ipa_cloud": "",
        }
        rows.append(out)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(
        f"wrote {OUT} n={len(rows)} with_ipa=0 "
        "(respelling→IPA conversion is banned; original source only)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
