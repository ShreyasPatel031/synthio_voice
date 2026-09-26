#!/usr/bin/env python3
"""Collapse ingredient F1 maps to the original 274 DOSE product rows.

Combo products get one vote: the mean of that row's ingredient scores.
Use this for any board that exploded multiword / combo names into slugs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOSE = ROOT / "data" / "dose_v1.jsonl"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def load_products() -> list[dict]:
    products = []
    for line in DOSE.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ingredients = row.get("ingredients") or [row["name"]]
        products.append(
            {
                "id": row["id"],
                "name": row["name"],
                "slug": slug(row["name"]),
                "ingredients": ingredients,
                "ing_slugs": [slug(ing) for ing in ingredients],
                "is_combination": bool(row.get("is_combination") or len(ingredients) > 1),
            }
        )
    return products


def collapse(slug_f1: dict[str, float]) -> list[dict]:
    rows = []
    for product in load_products():
        vals = [slug_f1[s] for s in product["ing_slugs"] if s in slug_f1]
        if not vals:
            continue
        rows.append(
            {
                "id": product["id"],
                "name": product["name"],
                "slug": product["slug"],
                "n_ingredients": len(product["ing_slugs"]),
                "n_scored": len(vals),
                "complete": len(vals) == len(product["ing_slugs"]),
                "cloud_f1": round(sum(vals) / len(vals), 4),
            }
        )
    return rows


def summarize(rows: list[dict], n_products: int = 274) -> dict:
    xs = sorted(r["cloud_f1"] for r in rows)
    mean = sum(xs) / len(xs) if xs else 0.0
    return {
        "unit": "dose-product",
        "n_products": n_products,
        "n": len(rows),
        "n_missing": n_products - len(rows),
        "n_complete": sum(1 for r in rows if r["complete"]),
        "mean_f1": round(mean, 4),
        "median_f1": round(xs[len(xs) // 2], 4) if xs else None,
        "p10": round(xs[max(0, int(0.10 * (len(xs) - 1)))], 4) if xs else None,
        "pass_ge_0_70": sum(1 for x in xs if x >= 0.70),
        "pass_rate_ge_0_70": round(sum(1 for x in xs if x >= 0.70) / len(xs), 4) if xs else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scores",
        required=True,
        help="JSON with rows[].slug + rows[].cloud_f1|ctc_f1, or slug→float map",
    )
    args = ap.parse_args()
    raw = json.loads(Path(args.scores).read_text())
    if isinstance(raw, dict) and "rows" in raw:
        slug_f1 = {}
        for r in raw["rows"]:
            f1 = r.get("cloud_f1", r.get("ctc_f1"))
            if f1 is not None:
                slug_f1[r["slug"]] = float(f1)
    else:
        slug_f1 = {k: float(v) for k, v in raw.items()}
    rows = collapse(slug_f1)
    out = {**summarize(rows), "rows": rows}
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
