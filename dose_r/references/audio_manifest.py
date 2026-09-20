"""The reference-audio manifest: one JSON record per verified clip, across
every source (Merriam-Webster today, Drugs.com once a collection lands).

Keyed by (ingredient, source, query) rather than just ingredient, because a
Merriam-Webster clip can cover only one word of a multi-word ingredient (see
`coverage` on each record) -- that is a second, distinct clip for the same
ingredient, not a duplicate of the whole-name one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "data" / "reference_audio" / "manifest.jsonl"


def record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (record["ingredient"], record["source"], record.get("query") or record["ingredient"])


def load(path: Path = MANIFEST_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def save(records: list[dict[str, Any]], path: Path = MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda r: record_key(r))
    with path.open("w") as f:
        for r in ordered:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def replace_source(
    existing: list[dict[str, Any]], fresh: list[dict[str, Any]], source: str
) -> list[dict[str, Any]]:
    """`fresh` replaces every prior record for `source`; other sources are untouched."""
    kept = [r for r in existing if r["source"] != source]
    return kept + fresh


def duplicate_groups(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    """sha256 -> ingredients, for every hash shared by more than one ingredient."""
    by_hash: dict[str, set[str]] = {}
    for r in records:
        if r.get("sha256"):
            by_hash.setdefault(r["sha256"], set()).add(r["ingredient"])
    return {h: sorted(ings) for h, ings in by_hash.items() if len(ings) > 1}
