"""Join the hard subset to carrier sentences and gold pronunciations.

The 54-item subset lives in runs/hard-subset-v1.json. Sentences live in
data/dose_v1.jsonl. IPA, ARPABET, and source respellings live in
dose_r/references/references.jsonl.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_IPA = re.compile(r"[əɚɛæɑɒɔʊʌɪɨᵻʃʒθðŋɡɹɾʔōēīāūˈˌː/\[\]]")
_ARPABET = re.compile(r"^[A-Z]{1,3}[012]?(?:\s+[A-Z]{1,3}[012]?)+$")
_TRUST = {"official_medical": 0, "verified_secondary": 1}


def is_dictionary_respelling(raw: str) -> bool:
    """Drugs.com / DailyMed / USAN hyphenated or prime spellings.

    Rejects MW phonetic, IPA, and ARPABET. Those are not English text.
    """
    s = (raw or "").strip()
    if not s or _ARPABET.match(s) or _IPA.search(s):
        return False
    if not re.search(r"[A-Za-z]", s) or re.search(r"[0-9]", s):
        return False
    return True


def clean_respelling(raw: str) -> str:
    """Citation markup -> speakable English syllables."""
    s = unicodedata.normalize("NFKC", raw)
    s = s.replace("“", '"').replace("”", '"').replace("„", '"')
    s = s.replace("‘", "'").replace("’", "'").replace("`", "'")
    s = re.sub(r"[\"']+", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" -")
    return s


def pick_dictionary_respelling(sources: list[dict]) -> dict | None:
    cands = []
    for src in sources or []:
        raw = (src.get("raw") or "").strip()
        if not is_dictionary_respelling(raw):
            continue
        spoken = clean_respelling(raw)
        if not spoken:
            continue
        cands.append({
            "raw": raw,
            "spoken": spoken,
            "source": src.get("name"),
            "trust_tier": src.get("trust_tier"),
        })
    cands.sort(key=lambda c: _TRUST.get(c["trust_tier"], 9))
    return cands[0] if cands else None


def _slug(drug: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", drug.lower()).strip("-")


def _item_from_row(row: dict, rec: dict, tier: str, item_id: str) -> dict | None:
    drug = row["name"]
    sentence = row["sentence"]
    idx = sentence.lower().find(drug.lower())
    if idx < 0:
        return None
    spoken = sentence[idx:idx + len(drug)]
    arpabet = list(rec.get("arpabet_variants") or [])
    ipa = list(rec.get("ipa_variants") or [])
    respell = pick_dictionary_respelling(rec.get("sources") or [])
    return {
        "item_id": item_id,
        "drug": drug,
        "spoken": spoken,
        "name_type": row.get("name_type"),
        "tier": tier,
        "sentence": sentence,
        "ipa_variants": ipa,
        "arpabet_variants": arpabet,
        "respelling": respell["spoken"] if respell else None,
        "respelling_raw": respell["raw"] if respell else None,
        "respelling_source": respell["source"] if respell else None,
        "has_human_ref": None,
    }


def load_full_items() -> list[dict]:
    """All 274 DOSE rows, not the 54-item hard subset."""
    refs = {}
    for line in (ROOT / "dose_r" / "references" / "references.jsonl").read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            refs[rec["ingredient"].lower()] = rec
    items = []
    missing = []
    for line in (ROOT / "data" / "dose_v1.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rec = refs.get(row["name"].lower(), {})
        item = _item_from_row(row, rec, tier="full", item_id=_slug(row["name"]))
        if item is None:
            missing.append(row["name"])
            continue
        items.append(item)
    if missing:
        raise SystemExit(f"full-set rows with no carrier span: {missing}")
    return items


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
        respell = pick_dictionary_respelling(rec.get("sources") or [])
        items.append({
            "item_id": item["item_id"],
            "drug": drug,
            "spoken": spoken,
            "name_type": item.get("name_type"),
            "tier": item["tier"],
            "sentence": sentence,
            "ipa_variants": ipa,
            "arpabet_variants": arpabet,
            "respelling": respell["spoken"] if respell else None,
            "respelling_raw": respell["raw"] if respell else None,
            "respelling_source": respell["source"] if respell else None,
            "has_human_ref": item.get("has_human_ref"),
        })
    if missing_sentence:
        raise SystemExit(f"hard-subset items with no carrier sentence: {missing_sentence}")
    return items


def inject_text(sentence: str, drug: str, replacement: str) -> str:
    idx = sentence.lower().find(drug.lower())
    if idx < 0:
        raise ValueError(f"{drug!r} not in sentence")
    return sentence[:idx] + replacement + sentence[idx + len(drug):]


def inject_arpabet(sentence: str, drug: str, arpabet: str) -> str:
    """Replace the drug name with CosyVoice 3 CMU tokens, e.g. [AH0] [B] [IH1]."""
    tokens = " ".join(f"[{p}]" for p in arpabet.split())
    return inject_text(sentence, drug, tokens)


def inject_respelling(sentence: str, drug: str, respelling: str) -> str:
    """Replace the drug span with a dictionary respelling, leave the rest."""
    return inject_text(sentence, drug, respelling)
