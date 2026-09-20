"""Join the hard subset to carrier sentences and gold pronunciations.

The 54-item subset lives in runs/hard-subset-v1.json. Sentences live in
data/dose_v1.jsonl. IPA and ARPABET live in dose_r/references/references.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_items() -> list[dict]:
    hard = json.loads((ROOT / "runs" / "hard-subset-v1.json").read_text())["items"]
    dose = {}
    for line in (ROOT / "data" / "dose_v1.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            dose[row["name"].lower()] = row
    refs = {}
    for line in (ROOT / "dose_r" / "references" / "references.jsonl").read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            refs[rec["ingredient"].lower()] = rec

    items = []
    missing_sentence = []
    for item in hard.values():
        drug = item["drug"]
        row = dose.get(drug.lower())
        sentence = row["sentence"] if row else ""
        idx = sentence.lower().find(drug.lower()) if sentence else -1
        if row is None or idx < 0:
            missing_sentence.append(drug)
            continue
        spoken = sentence[idx:idx + len(drug)]
        rec = refs.get(drug.lower(), {})
        arpabet = list(rec.get("arpabet_variants") or [])
        ipa = list(rec.get("ipa_variants") or [])
        items.append({
            "item_id": item["item_id"],
            "drug": drug,
            "spoken": spoken,
            "name_type": item.get("name_type"),
            "tier": item["tier"],
            "sentence": sentence,
            "ipa_variants": ipa,
            "arpabet_variants": arpabet,
            "has_human_ref": item.get("has_human_ref"),
        })
    if missing_sentence:
        raise SystemExit(f"hard-subset items with no carrier sentence: {missing_sentence}")
    return items


def inject_arpabet(sentence: str, drug: str, arpabet: str) -> str:
    """Replace the drug name with CosyVoice 3 CMU tokens, e.g. [AH0] [B] [IH1]."""
    tokens = " ".join(f"[{p}]" for p in arpabet.split())
    idx = sentence.lower().find(drug.lower())
    if idx < 0:
        raise ValueError(f"{drug!r} not in sentence")
    return sentence[:idx] + tokens + sentence[idx + len(drug):]
