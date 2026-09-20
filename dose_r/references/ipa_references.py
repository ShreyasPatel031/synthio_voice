"""Load the dictionary IPA reference data (`references.jsonl`) that Path 3
(`scoring.phoneme_distance`) scores candidates against.

`references.jsonl` is Workstream 1's deliverable, built and cleaned on
`claude/sc-sandbox-gcp-access-arw4m8` (280/284 ingredients with at least one
sourced IPA variant from USAN/DailyMed/Merriam-Webster/NCI/CMUdict/Wikipedia,
zero character-level defects per their own cleanup pass). The copy here is a
SNAPSHOT (pulled at commit fee5ac5 on that branch) so Path 3 has something
durable to depend on -- reading it out of a scratch/tmp path would silently
break on every container restart, which already happened once this session
to an in-progress run. This file should be treated as provisional and
re-synced (or dropped in favor of a real merge) once that branch's work
lands here properly; it is not this project's own source of truth for IPA
data, just a working copy of someone else's.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_REFERENCES_PATH = Path(__file__).resolve().parent / "references.jsonl"


@lru_cache(maxsize=1)
def load_ipa_references(path: Path | None = None) -> dict[str, dict]:
    """ingredient (lowercased) -> full reference record (ipa_variants,
    arpabet_variants, sources, confidence, notes). Lowercased key so lookups
    don't depend on matching brand/generic capitalization exactly.
    """
    p = path or _REFERENCES_PATH
    out: dict[str, dict] = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        out[rec["ingredient"].lower()] = rec
    return out


def ipa_variants_for(drug: str) -> list[str]:
    """Drug name -> its list of accepted IPA variants, or [] if none (either
    the ingredient isn't in the snapshot, or it's one of the 4 with no
    phonetic source found by any of the sources this data was built from).
    """
    rec = load_ipa_references().get(drug.lower())
    return rec["ipa_variants"] if rec else []
