"""Statistical power for DOSE-R's fidelity comparisons.

274 items sounds like a lot until you ask what gap between two pass rates it
can actually distinguish from sampling noise, and further ask what happens to
that answer once 185 of the 284 gold references (see COVERAGE.md) carry no
external confirmation at all. Both questions are answered here, quantitatively,
before any real score exists -- so that a 2-point gap in a future report gets
read correctly the day it appears, rather than after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass

from .stats import min_detectable_gap, wilson_ci


@dataclass(frozen=True)
class PowerResult:
    n: int
    baseline_p: float
    alpha: float
    power: float
    min_gap_pp: float
    ci_halfwidth_pp: float


def sampling_power(n: int, baseline_p: float = 0.7, alpha: float = 0.05, power: float = 0.8) -> PowerResult:
    """What gap is distinguishable at this n, ignoring reference-layer noise.

    `baseline_p` matters: proportions near 0.5 are noisiest (p(1-p) is
    maximal there), so the minimum detectable gap is itself widest near a 50%
    pass rate and narrows toward the extremes. 0.7 is used as the default
    because it sits between DOSE-R's known anchors (63-91% overall, and
    Gemini's 61.6%-89.1% established/new split) -- a representative, not
    optimistic, working point.
    """
    gap = min_detectable_gap(n, baseline_p, alpha, power)
    lo, hi = wilson_ci(round(baseline_p * n), n)
    return PowerResult(
        n=n,
        baseline_p=baseline_p,
        alpha=alpha,
        power=power,
        min_gap_pp=round(gap * 100, 1),
        ci_halfwidth_pp=round((hi - lo) / 2 * 100, 1),
    )


def effective_sample_size(total: int, low_confidence: int) -> dict:
    """The plain, non-statistical half of the story: how many of the 274 rows
    rest on any external confirmation of their gold pronunciation at all.

    A row's confidence is the weakest of its ingredients' tiers (see
    `dose_r.references.build`), so a single low-confidence ingredient in a
    combination row pulls the whole row down.
    """
    confirmed = total - low_confidence
    return {
        "total": total,
        "low_confidence": low_confidence,
        "low_confidence_share": round(low_confidence / total, 3),
        "effective_n_excluding_low_confidence": confirmed,
        "effective_n_share": round(confirmed / total, 3),
    }


def propagate_reference_uncertainty(
    n_total: int, n_low_confidence: int, p_hat: float, assumed_label_error_rates: tuple[float, ...] = (0.10, 0.20, 0.30)
) -> list[dict]:
    """Widen the pass-rate CI to account for gold references that might be
    wrong, instead of reporting a CI that only reflects finite-sample noise.

    Model: each low-confidence row's pass/fail label is correct with
    probability `1 - e` and flipped with probability `e`, independently, for
    an assumed per-row gold error rate `e`. A flip changes that row's
    contribution to the pass-count sum by +/-1, contributing variance
    e*(1-e) to the sum; averaged over n_total rows, that adds
    n_low_confidence * e*(1-e) / n_total**2 to Var(p_hat) on top of the usual
    p(1-p)/n sampling term.

    `e` itself is not measured anywhere in this project -- there is no ground
    truth for how often the rule-derived low-confidence references are wrong.
    Three illustrative values are swept instead of asserting one, and the
    output is explicit about that: this is a sensitivity bracket, not a
    calibrated correction.
    """
    sampling_var = p_hat * (1 - p_hat) / n_total
    out = []
    for e in assumed_label_error_rates:
        extra_var = n_low_confidence * e * (1 - e) / (n_total ** 2)
        total_var = sampling_var + extra_var
        half_width = 1.96 * total_var ** 0.5
        out.append(
            {
                "assumed_gold_error_rate": e,
                "ci_halfwidth_pp_sampling_only": round(1.96 * sampling_var ** 0.5 * 100, 2),
                "ci_halfwidth_pp_with_reference_uncertainty": round(half_width * 100, 2),
            }
        )
    return out
