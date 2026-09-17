"""Loading and staging the public DOSE v1.0 test set.

The parquet ships `name_type` as an Arrow ClassLabel (int64) whose names live in
schema metadata: 0 -> brand, 1 -> generic. Decoding that mapping from the file
rather than hardcoding it means a re-pull with a reordered label set cannot
silently invert every stratum in the report.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from . import config

EXPECTED_ROWS = 274
EXPECTED_STRATA = {"brand": 143, "generic": 131}

# Every generic row ships its `drug` field as "acetaminophen (generic)". The
# parenthetical is a label artifact duplicating `name_type`, not part of the name:
# the carrier sentence contains only "acetaminophen". Feeding the raw field to a
# scorer as the span target would make all 131 generic items unmatchable, so the
# loader strips it and validate() asserts the stripped form appears in every
# sentence.
_LABEL_SUFFIX = re.compile(r"\s*\((?:generic|brand|inn)\)\s*$", re.IGNORECASE)


def normalize_drug_name(raw: str) -> str:
    """Strip the redundant `(generic)` / `(brand)` label suffix."""
    return _LABEL_SUFFIX.sub("", raw).strip()


@dataclass(frozen=True)
class DoseItem:
    drug: str              # normalized name, as spoken in the sentence
    name_type: str
    sentence: str
    drug_raw: str = ""     # field exactly as shipped, kept for provenance

    @property
    def item_id(self) -> str:
        return self.drug.lower().replace(" ", "-").replace("/", "-")

    @property
    def char_count(self) -> int:
        """Billable characters for the synthesis request."""
        return len(self.sentence)


class DatasetValidationError(RuntimeError):
    pass


def _class_label_names(path: Path) -> list[str]:
    schema = pq.read_schema(path)
    meta = (schema.metadata or {}).get(b"huggingface")
    if not meta:
        raise DatasetValidationError(f"{path} has no huggingface schema metadata")
    feats = json.loads(meta.decode())["info"]["features"]
    names = feats.get("name_type", {}).get("names")
    if not names:
        raise DatasetValidationError("name_type ClassLabel names missing from metadata")
    return names


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load the parquet with `name_type` decoded to its string label."""
    path = path or config.DOSE_PARQUET
    if not path.exists():
        raise DatasetValidationError(
            f"{path} not found. Run scripts/stage_dataset.py --download first."
        )
    names = _class_label_names(path)
    df = pd.read_parquet(path)
    if pd.api.types.is_integer_dtype(df["name_type"]):
        df["name_type"] = df["name_type"].map(dict(enumerate(names)))
    df = df.rename(columns={"drug": "drug_raw"})
    df["drug"] = df["drug_raw"].map(normalize_drug_name)
    return df[["drug", "drug_raw", "name_type", "sentence"]]


def validate(df: pd.DataFrame) -> dict:
    """Assert the counts DOSE publishes. Returns an integrity report."""
    problems: list[str] = []
    if len(df) != EXPECTED_ROWS:
        problems.append(f"row count {len(df)} != {EXPECTED_ROWS}")

    counts = df["name_type"].value_counts().to_dict()
    for label, expected in EXPECTED_STRATA.items():
        if counts.get(label) != expected:
            problems.append(f"{label} count {counts.get(label)} != {expected}")

    dupes = df["drug"][df["drug"].duplicated()].tolist()
    if dupes:
        problems.append(f"duplicate drug names: {dupes}")

    missing = df[df[["drug", "drug_raw", "name_type", "sentence"]].isna().any(axis=1)]
    if len(missing):
        problems.append(f"{len(missing)} rows with null fields")

    # Every sentence must actually contain its drug name -- otherwise the judge
    # has no span to score and the item is silently unscoreable.
    absent = [
        r.drug for r in df.itertuples()
        if r.drug.lower() not in r.sentence.lower()
    ]
    if absent:
        problems.append(f"{len(absent)} sentences do not contain their drug name: {absent[:10]}")

    relabelled = int((df["drug"] != df["drug_raw"]).sum())

    return {
        "rows": len(df),
        "strata": counts,
        "label_suffix_stripped": relabelled,
        "duplicate_drugs": len(dupes),
        "sentences_missing_drug": len(absent),
        "sentences_missing_drug_examples": absent[:10],
        "problems": problems,
        "ok": not problems,
    }


def load_items(path: Path | None = None) -> list[DoseItem]:
    df = load_raw(path)
    return [
        DoseItem(r.drug, r.name_type, r.sentence, drug_raw=r.drug_raw)
        for r in df.itertuples()
    ]


def stage(path: Path | None = None) -> dict:
    """Write the canonical CSV plus an integrity manifest."""
    path = path or config.DOSE_PARQUET
    df = load_raw(path)
    report = validate(df)

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DOSE_CANONICAL, index=False)

    manifest = {
        "source": "SynthioLabs/dose-benchmark (CC-BY-4.0)",
        "source_parquet_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "canonical_csv": str(config.DOSE_CANONICAL.relative_to(config.REPO_ROOT)),
        "canonical_csv_sha256": hashlib.sha256(
            config.DOSE_CANONICAL.read_bytes()
        ).hexdigest(),
        "integrity": report,
    }
    (config.DATA_PROCESSED / "dose_v1_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    return manifest
