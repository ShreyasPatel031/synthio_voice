"""The DOSE test set, expanded to its real scoring unit.

`data/dose_v1.jsonl` has 274 rows but 286 ingredient spans, because nine generic
rows are combination products naming two or three ingredients. The judge scores
spans; rows are an aggregation of spans (see `combination.py`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "dose_v1.jsonl"


@dataclass(frozen=True)
class Span:
    """One ingredient occurrence inside one sentence -- the unit of scoring."""

    item_id: str
    row_id: str
    ingredient: str
    name_type: str
    is_combination: bool
    span: tuple[int, int]
    sentence: str
    position: int

    @property
    def text(self) -> str:
        return self.sentence[self.span[0] : self.span[1]]


@dataclass(frozen=True)
class Row:
    row_id: str
    name: str
    name_type: str
    is_combination: bool
    sentence: str
    spans: tuple[Span, ...]


def load_rows(path: str | Path = DEFAULT_PATH) -> list[Row]:
    with Path(path).open() as f:
        objs = [json.loads(line) for line in f if line.strip()]

    rows = []
    for obj in objs:
        spans = tuple(
            Span(
                item_id=f"{obj['id']}#{k}",
                row_id=obj["id"],
                ingredient=ing,
                name_type=obj["name_type"],
                is_combination=obj["is_combination"],
                span=tuple(sp),
                sentence=obj["sentence"],
                position=k,
            )
            for k, (ing, sp) in enumerate(zip(obj["ingredients"], obj["spans"]))
        )
        rows.append(
            Row(
                row_id=obj["id"],
                name=obj["name"],
                name_type=obj["name_type"],
                is_combination=obj["is_combination"],
                sentence=obj["sentence"],
                spans=spans,
            )
        )
    return rows


def load_spans(path: str | Path = DEFAULT_PATH) -> list[Span]:
    return [span for row in load_rows(path) for span in row.spans]


def unique_ingredients(path: str | Path = DEFAULT_PATH) -> list[str]:
    seen: dict[str, str] = {}
    for span in load_spans(path):
        seen.setdefault(span.ingredient.lower(), span.ingredient)
    return sorted(seen.values(), key=str.lower)
