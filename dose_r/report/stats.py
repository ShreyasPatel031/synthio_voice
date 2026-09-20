"""Small self-contained statistics: no scipy in this project's dependencies.

Everything here is standard (Wilson interval, inverse normal CDF, Spearman's
rho with tie correction) and is tested against known reference values rather
than trusted on inspection.
"""

from __future__ import annotations

import math


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation).

    Accurate to about 1.15e-9 across (0, 1), which is far more precision than
    any sample-size decision here needs.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")

    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]

    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def z_for_confidence(confidence: float) -> float:
    return norm_ppf(1 - (1 - confidence) / 2)


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the naive p +/- z*sqrt(p(1-p)/n) interval because the naive
    form is badly behaved (can exceed [0, 1], undercovers) exactly in the
    small-n, extreme-p regime this project lives in.
    """
    if n == 0:
        return (0.0, 1.0)
    z = z_for_confidence(confidence)
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((center - spread) / denom, (center + spread) / denom)


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Two-sided z-test for a difference in two independent proportions.

    Returns (z, p_value). Pooled-variance null (same true proportion), the
    standard test for "is this observed gap distinguishable from noise".
    """
    if n1 == 0 or n2 == 0:
        return (0.0, 1.0)
    p1, p2 = k1 / n1, k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (p1 - p2) / se
    p_value = 2 * (1 - _norm_cdf(abs(z)))
    return (z, p_value)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def min_detectable_gap(n: int, p: float = 0.5, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest |p1 - p2| distinguishable from noise, two independent samples
    of size `n` each, centered near baseline `p`, at the given alpha/power.

    Conservative on purpose: DOSE-R's real comparisons are usually paired (the
    same 274 items scored twice, once per system or once per stratum split of
    one system's run), and a paired test has more power than this unpaired
    approximation credits it with. Treat the output as an upper bound on the
    gap needed for significance, not a tight estimate.
    """
    z_a = z_for_confidence(1 - alpha)
    z_b = norm_ppf(power)
    variance_sum = 2 * p * (1 - p)
    return (z_a + z_b) * math.sqrt(variance_sum / n)


def spearman(x: dict[str, float], y: dict[str, float]) -> tuple[float | None, int]:
    """Spearman rank correlation over the keys common to both, with average
    ranks for ties. Returns (rho, n_common); rho is None when n_common < 4,
    where a correlation coefficient is not a meaningful summary.
    """
    common = sorted(set(x) & set(y))
    n = len(common)
    if n < 4:
        return (None, n)

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg_rank
            i = j + 1
        return out

    rx = ranks([x[k] for k in common])
    ry = ranks([y[k] for k in common])
    mean_r = (n + 1) / 2
    cov = sum((a - mean_r) * (b - mean_r) for a, b in zip(rx, ry))
    var_x = sum((a - mean_r) ** 2 for a in rx)
    var_y = sum((b - mean_r) ** 2 for b in ry)
    if var_x == 0 or var_y == 0:
        return (0.0, n)
    return (cov / math.sqrt(var_x * var_y), n)
