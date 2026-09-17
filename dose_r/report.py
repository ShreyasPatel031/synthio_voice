"""Aggregation and reporting over a run log.

This is the seed of the Workstream 2a tradeoff blueprint: accuracy, latency and
cost are reported side by side per system, because a model that is 2% more accurate
but 5x slower or 10x the cost is not automatically the better clinical deployment.

The score distribution is kept, not just the pass/fail collapse -- one of the
documented critiques of DOSE's own metric design.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .scoring.base import PASS_THRESHOLD


class NotLeaderboardComparable(RuntimeError):
    """Raised when a caller asks to compare non-pronunciation scores to DOSE."""


def load_records(results_path: Path) -> list[dict]:
    records = []
    with results_path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * pct
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


@dataclass
class SystemSummary:
    system_id: str
    n_items: int
    n_ok: int
    n_failed: int

    pass_rate: float | None
    n_scored: int
    score_mean: float | None
    score_median: float | None
    score_p10: float | None
    pass_rate_by_stratum: dict[str, float | None]

    latency_mean_ms: float | None
    latency_p50_ms: float | None
    latency_p90_ms: float | None
    latency_p99_ms: float | None

    total_cost_usd: float
    cost_per_utterance_usd: float | None
    cost_estimated: bool
    price_verified: bool

    retry_rate: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def summarize(records: list[dict]) -> dict[str, SystemSummary]:
    by_system: dict[str, list[dict]] = {}
    for r in records:
        by_system.setdefault(r["system_id"], []).append(r)

    out: dict[str, SystemSummary] = {}
    for system_id, rows in sorted(by_system.items()):
        ok = [r for r in rows if r["synthesis"]["ok"]]
        latencies = [r["synthesis"]["total_ms"] for r in ok
                     if r["synthesis"].get("total_ms") is not None]

        scored = [r for r in rows
                  if r.get("score") and r["score"].get("score") is not None]
        scores = [r["score"]["score"] for r in scored]

        by_stratum: dict[str, float | None] = {}
        for stratum in ("brand", "generic"):
            sub = [r for r in scored if r["name_type"] == stratum]
            by_stratum[stratum] = (
                round(100.0 * sum(1 for r in sub
                                  if r["score"]["score"] >= PASS_THRESHOLD) / len(sub), 1)
                if sub else None
            )

        total_cost = sum(r["synthesis"].get("cost_usd") or 0.0 for r in ok)
        estimated = any(r["synthesis"].get("cost_estimated") for r in ok)
        verified = all(r["synthesis"].get("price_verified") for r in ok) if ok else False
        retries = sum(1 for r in rows if r["synthesis"].get("attempts", 1) > 1)

        out[system_id] = SystemSummary(
            system_id=system_id,
            n_items=len(rows), n_ok=len(ok), n_failed=len(rows) - len(ok),
            pass_rate=(round(100.0 * sum(1 for s in scores if s >= PASS_THRESHOLD)
                             / len(scores), 1) if scores else None),
            n_scored=len(scores),
            score_mean=round(statistics.fmean(scores), 3) if scores else None,
            score_median=round(statistics.median(scores), 3) if scores else None,
            score_p10=round(_percentile(scores, 0.10), 3) if scores else None,
            pass_rate_by_stratum=by_stratum,
            latency_mean_ms=round(statistics.fmean(latencies), 1) if latencies else None,
            latency_p50_ms=round(_percentile(latencies, 0.50), 1) if latencies else None,
            latency_p90_ms=round(_percentile(latencies, 0.90), 1) if latencies else None,
            latency_p99_ms=round(_percentile(latencies, 0.99), 1) if latencies else None,
            total_cost_usd=round(total_cost, 6),
            cost_per_utterance_usd=round(total_cost / len(ok), 8) if ok else None,
            cost_estimated=estimated, price_verified=verified,
            retry_rate=round(100.0 * retries / len(rows), 1) if rows else 0.0,
        )
    return out


def render_text(summaries: dict[str, SystemSummary], manifest: dict) -> str:
    """Human-readable tradeoff table."""
    measures = manifest.get("scorer_measures_pronunciation")
    scorer = manifest.get("scorer") or "none"

    lines = [
        "=" * 100,
        f"DOSE-R run {manifest.get('run_id')}   scorer={scorer}",
        "=" * 100,
    ]

    if measures is False:
        lines += [
            "",
            "  !! STAND-IN SCORER -- NOT A PRONUNCIATION JUDGEMENT !!",
            "  These pass rates measure audio deliverability, not whether the drug name",
            "  was pronounced correctly. They are NOT comparable to DOSE's leaderboard.",
            "  Latency and cost figures below ARE real measurements.",
            "",
        ]

    hdr = (f"{'system':26s} {'ok':>5s} {'fail':>5s} {'pass%':>7s} {'brand%':>7s} "
           f"{'gen%':>6s} {'p50ms':>8s} {'p90ms':>8s} {'$/utt':>11s} {'$total':>9s}")
    lines += [hdr, "-" * len(hdr)]

    for s in summaries.values():
        f = lambda v, w, p=1: ("-".rjust(w) if v is None else f"{v:>{w}.{p}f}")
        lines.append(
            f"{s.system_id:26s} {s.n_ok:5d} {s.n_failed:5d} {f(s.pass_rate,7)} "
            f"{f(s.pass_rate_by_stratum.get('brand'),7)} "
            f"{f(s.pass_rate_by_stratum.get('generic'),6)} "
            f"{f(s.latency_p50_ms,8,0)} {f(s.latency_p90_ms,8,0)} "
            f"{f(s.cost_per_utterance_usd,11,8)} {f(s.total_cost_usd,9,4)}"
        )

    lines += ["", "Score distribution (retained, not collapsed to pass/fail):",
              f"  {'system':26s} {'mean':>7s} {'median':>7s} {'p10':>7s} {'n':>5s}"]
    for s in summaries.values():
        f = lambda v, w, p=2: ("-".rjust(w) if v is None else f"{v:>{w}.{p}f}")
        lines.append(f"  {s.system_id:26s} {f(s.score_mean,7)} "
                     f"{f(s.score_median,7)} {f(s.score_p10,7)} {s.n_scored:5d}")

    if any(s.cost_estimated for s in summaries.values()):
        lines += ["", "Cost note: figures are list-price estimates, not billing-export",
                  "verified (see config.Pricing.verified)."]
    return "\n".join(lines)


def compare_to_dose(summaries: dict[str, SystemSummary],
                    manifest: dict, mapping: dict[str, str]) -> dict:
    """Workstream 1d comparison. Refuses to run on a non-pronunciation scorer."""
    if manifest.get("scorer_measures_pronunciation") is not True:
        raise NotLeaderboardComparable(
            f"scorer {manifest.get('scorer')!r} does not measure pronunciation; "
            "its pass rates cannot be compared to the DOSE leaderboard."
        )
    from .config import DOSE_LEADERBOARD
    out = {}
    for system_id, dose_key in mapping.items():
        s = summaries.get(system_id)
        published = DOSE_LEADERBOARD.get(dose_key)
        if s and s.pass_rate is not None and published is not None:
            out[system_id] = {
                "ours": s.pass_rate, "dose_published": published,
                "absolute_gap": round(s.pass_rate - published, 1),
            }
    return out
