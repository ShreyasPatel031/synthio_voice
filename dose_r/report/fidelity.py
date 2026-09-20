"""Replication-fidelity report: how close a DOSE-R judged run comes to DOSE's
published numbers, and where a gap comes from.

This module is pure computation over already-judged data (`rows.jsonl`-style
records from `dose_r.judge.pipeline`, plus `strata.jsonl`). It has no opinion
about where that data came from -- a mock run, a synthetic run, or (once it
exists) a real one. See FIDELITY.md for why no real run exists yet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .stats import spearman, two_proportion_z, wilson_ci

PASS_SCORE = 4


@dataclass(frozen=True)
class RowResult:
    row_id: str
    score: int
    name_type: str
    is_combination: bool

    @property
    def passed(self) -> bool:
        return self.score >= PASS_SCORE


def load_rows_jsonl(path: str | Path) -> dict[str, RowResult]:
    records = (json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    out = {}
    for r in records:
        out[r["row_id"]] = RowResult(
            row_id=r["row_id"],
            score=r["score"],
            name_type=r["name_type"],
            is_combination=r["is_combination"],
        )
    return out


def load_strata(path: str | Path) -> dict[str, dict]:
    with Path(path).open() as f:
        return {r["id"]: r for line in f if line.strip() for r in [json.loads(line)]}


@dataclass(frozen=True)
class GroupStat:
    n: int
    k: int
    rate: float | None
    ci_low: float | None
    ci_high: float | None

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "k": self.k,
            "rate": self.rate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }


def group_stat(rows: Iterable[RowResult]) -> GroupStat:
    rows = list(rows)
    n = len(rows)
    if n == 0:
        return GroupStat(0, 0, None, None, None)
    k = sum(r.passed for r in rows)
    lo, hi = wilson_ci(k, n)
    return GroupStat(n, k, k / n, lo, hi)


def pass_rate(rows: dict[str, RowResult], row_ids: set[str] | None = None) -> GroupStat:
    selected = rows.values() if row_ids is None else (rows[i] for i in row_ids if i in rows)
    return group_stat(selected)


def stratified_pass_rate(
    rows: dict[str, RowResult], strata: dict[str, dict], key: str
) -> dict[str, GroupStat]:
    """Pass rate per distinct value of `strata[row_id][key]` (e.g. key="era")."""
    buckets: dict[str, list[RowResult]] = {}
    for row_id, row in rows.items():
        if row_id not in strata:
            continue
        value = strata[row_id][key]
        buckets.setdefault(value, []).append(row)
    return {value: group_stat(group) for value, group in buckets.items()}


def gap_table(observed: dict[str, float], published: dict[str, float]) -> list[dict]:
    """Absolute gap per system present in both, observed minus published."""
    common = sorted(set(observed) & set(published))
    return [
        {
            "system": s,
            "observed": observed[s],
            "published": published[s],
            "gap_pp": round((observed[s] - published[s]) * 100, 2),
        }
        for s in common
    ]


def rank_correlation(observed: dict[str, float], published: dict[str, float]) -> dict:
    rho, n = spearman(observed, published)
    return {
        "spearman_rho": rho,
        "n_common_systems": n,
        "meaningful": rho is not None,
        "note": (
            "fewer than 4 systems in common: a rank correlation coefficient "
            "over so few points is not a meaningful summary and is not reported"
            if rho is None
            else ""
        ),
    }


def attribution_by_reference_confidence(
    rows: dict[str, RowResult], strata: dict[str, dict], confidence_key: str = "reference_confidence"
) -> dict:
    """Does this system's pass rate move with reference-layer confidence?

    If a run's pass rate is markedly lower specifically on low-confidence-
    reference rows, that gap is at least partly attributable to reference-layer
    noise (a rule-derived gold pronunciation the judge is scoring against may
    itself be wrong), not to the TTS system or the judge's phonetic model. If
    the pass rate is flat across tiers, the gap more likely sits in the judge
    or the system rather than the references.

    This is a proxy, not the "re-score against a hand-verified subset"
    attribution the brief asks for -- that requires a human (or a second,
    independent source) to have actually confirmed a sample of the low-
    confidence references, which `artifacts/anchor_set.jsonl` stages for but
    nobody has done yet (see FIDELITY.md). Once that verification exists, feed
    its verdicts through `attribution_against_hand_verified` instead, which
    performs the comparison the brief specifies directly.
    """
    tiers = ("high", "medium", "low")
    stats = {}
    for tier in tiers:
        ids = {rid for rid, s in strata.items() if s.get(confidence_key) == tier}
        stats[tier] = group_stat(rows[i] for i in ids if i in rows)

    high, low = stats["high"], stats["low"]
    z, p_value = (
        two_proportion_z(high.k, high.n, low.k, low.n) if high.n and low.n else (0.0, 1.0)
    )
    return {
        "by_tier": {t: s.to_dict() for t, s in stats.items()},
        "high_vs_low": {
            "gap_pp": round(((high.rate or 0) - (low.rate or 0)) * 100, 2) if high.n and low.n else None,
            "z": z,
            "p_value": p_value,
            "significant_at_0.05": p_value < 0.05,
        },
    }


def attribution_against_hand_verified(
    rows: dict[str, RowResult], hand_verified: dict[str, bool]
) -> dict:
    """The attribution the brief actually specifies: re-score a stratified
    sample against a hand-verified subset and see how much the verdict moves.

    `hand_verified` maps row_id -> the pass/fail a human confirmed against
    audio and a trusted pronunciation, for whatever sample was checked (e.g.
    `artifacts/anchor_set.jsonl` once someone has listened through it). This
    compares the judge's automated verdict on that same sample to the human
    one; systematic disagreement is direct evidence the automated pipeline
    (reference layer, judge, or both together) is wrong on that sample, not
    just uncertain.
    """
    common = sorted(set(rows) & set(hand_verified))
    n = len(common)
    if n == 0:
        return {"n": 0, "agreement": None, "disagreements": []}
    agree = sum(rows[i].passed == hand_verified[i] for i in common)
    disagreements = [
        {"row_id": i, "automated_passed": rows[i].passed, "hand_verified_passed": hand_verified[i]}
        for i in common
        if rows[i].passed != hand_verified[i]
    ]
    return {"n": n, "agreement": agree / n, "disagreements": disagreements}
