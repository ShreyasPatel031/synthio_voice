"""Synthetic inputs for exercising the fidelity harness end to end.

No real TTS run exists in this repo (see FIDELITY.md). This generates
plausible row-level judge output and a matching strata file so the harness's
statistics, not its plumbing, are what tests actually exercise.
"""

from __future__ import annotations

import random

from .fidelity import RowResult

ERAS = ("established", "new")
DIFFICULTIES = ("easy", "medium", "hard")
CONFIDENCES = ("high", "medium", "low")


def synthetic_strata(n: int = 274, seed: int = 0) -> dict[str, dict]:
    rng = random.Random(seed)
    out = {}
    for i in range(n):
        row_id = f"dose-{i:03d}"
        out[row_id] = {
            "id": row_id,
            "era": rng.choices(ERAS, weights=[0.47, 0.53])[0],
            "era_confidence": rng.choices(CONFIDENCES, weights=[0.15, 0.2, 0.65])[0],
            "difficulty": rng.choices(DIFFICULTIES, weights=[0.23, 0.37, 0.4])[0],
        }
    return out


def synthetic_rows(
    strata: dict[str, dict],
    established_p: float,
    new_p: float,
    seed: int = 0,
) -> dict[str, RowResult]:
    """A run whose pass rate depends only on era, at the given per-era rates.

    Used to check that `stratified_pass_rate` recovers a known, planted split
    and that the overall rate lands near the era-weighted average.
    """
    rng = random.Random(seed)
    out = {}
    for row_id, s in strata.items():
        p = established_p if s["era"] == "established" else new_p
        passed = rng.random() < p
        score = 4 if passed else rng.choice([1, 2, 3])
        out[row_id] = RowResult(row_id=row_id, score=score, name_type="brand", is_combination=False)
    return out
