"""Stage the public DOSE v1.0 test set into the canonical DOSE-R form.

Source: SynthioLabs/dose-benchmark on Hugging Face (CC-BY-4.0), columns
`drug`, `name_type`, `sentence`.

Two properties of the raw data drive everything downstream:

1. Every generic row's `drug` carries a literal " (generic)" annotation that
   never appears in the sentence. It is metadata, not part of the name, so it
   must be stripped before the name is used as a pronunciation target.
2. Nine generic rows are combination products naming 2-3 separate ingredients.
   The scoring unit there is each ingredient span, not the row.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

HF_PARQUET = (
    "https://huggingface.co/api/datasets/SynthioLabs/dose-benchmark"
    "/parquet/default/train/0.parquet"
)

NAME_TYPE = {0: "brand", 1: "generic"}
ANNOTATION = re.compile(r"\s*\((generic|brand)\)\s*$", re.I)

EXPECTED_TOTAL = 274
EXPECTED_BY_TYPE = {"brand": 143, "generic": 131}


def strip_annotation(drug: str) -> str:
    return ANNOTATION.sub("", drug).strip()


def split_ingredients(name: str) -> list[str]:
    """Split a combination product into its component ingredient names."""
    parts = re.split(r",\s*and\s+|,\s*|\s+and\s+", name)
    return [p.strip() for p in parts if p.strip()]


def locate(ingredient: str, sentence: str) -> tuple[int, int] | None:
    """Character span of `ingredient` in `sentence`, case-insensitive."""
    m = re.search(re.escape(ingredient), sentence, re.I)
    return (m.start(), m.end()) if m else None


def build(df: pd.DataFrame) -> list[dict]:
    rows = []
    for i, r in df.iterrows():
        name = strip_annotation(r["drug"])
        ingredients = split_ingredients(name)
        spans = [locate(ing, r["sentence"]) for ing in ingredients]
        rows.append(
            {
                "id": f"dose-{i:03d}",
                "raw_drug": r["drug"],
                "name": name,
                "name_type": NAME_TYPE[int(r["name_type"])],
                "is_combination": len(ingredients) > 1,
                "ingredients": ingredients,
                "spans": spans,
                "sentence": r["sentence"],
            }
        )
    return rows


def validate(rows: list[dict]) -> list[str]:
    errors = []
    if len(rows) != EXPECTED_TOTAL:
        errors.append(f"expected {EXPECTED_TOTAL} rows, got {len(rows)}")

    counts = pd.Series([r["name_type"] for r in rows]).value_counts().to_dict()
    for t, n in EXPECTED_BY_TYPE.items():
        if counts.get(t) != n:
            errors.append(f"expected {n} {t} rows, got {counts.get(t)}")

    for r in rows:
        if ANNOTATION.search(r["name"]):
            errors.append(f"{r['id']}: annotation survived stripping")
        for ing, span in zip(r["ingredients"], r["spans"]):
            if span is None:
                errors.append(f"{r['id']}: ingredient {ing!r} absent from sentence")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=HF_PARQUET)
    ap.add_argument("--out", default="data/dose_v1.jsonl")
    args = ap.parse_args()

    rows = build(pd.read_parquet(args.parquet))

    errors = validate(rows)
    if errors:
        print("VALIDATION FAILED", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    combos = sum(r["is_combination"] for r in rows)
    print(f"staged {len(rows)} rows -> {out}")
    print(f"  brand={EXPECTED_BY_TYPE['brand']} generic={EXPECTED_BY_TYPE['generic']}")
    print(f"  combination products={combos}")
    print(f"  total ingredient spans={sum(len(r['ingredients']) for r in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
