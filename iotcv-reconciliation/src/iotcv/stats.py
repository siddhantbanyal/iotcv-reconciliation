"""Statistical primitives for the sequential detectors, in pure NumPy.

We deliberately avoid SciPy so the project keeps a minimal dependency footprint
(numpy + pandas only) and stays edge-deployable. This module provides the few
special functions and design formulas the detectors need:

  * ``gammainc_lower`` / ``gammainc_upper``: the regularized lower/upper
    incomplete gamma functions P(a,x), Q(a,x), via a power series for x < a+1
    and a Lenz continued fraction otherwise (Numerical Recipes, sec. 6.2).
  * ``norm_cdf`` / ``norm_ppf``: standard normal CDF (via erf) and its inverse
    (Acklam's rational approximation), used for CUSUM design.
  * ``cusum_threshold_from_arl``: Siegmund's average-run-length approximation
    inverted to pick the CUSUM decision interval ``h`` for a target in-control
    ARL and reference value ``k``.
  * ``sprt_bounds``: Wald's A/B log-likelihood-ratio boundaries from target
    type-I/type-II error rates.

All functions are scalar-or-array and rely only on numpy.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

_EPS = 1e-12
_ITMAX = 200


# ---------------------------------------------------------------------------
# Incomplete gamma (regularized), pure-numpy, scalar implementation.
# ---------------------------------------------------------------------------
def _gser(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a,x) by series (good for x < a+1)."""
    if x <= 0:
        return 0.0
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(_ITMAX):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * _EPS:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gcf(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a,x) by Lentz continued fraction."""
    tiny = 1e-30
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, _ITMAX + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def gammainc_lower(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x) = gamma(a,x)/Gamma(a)."""
    a = float(a)
    x = float(x)
    if x < 0 or a <= 0:
        raise ValueError("gammainc_lower requires a>0, x>=0")
    if x == 0:
        return 0.0
    if x < a + 1.0:
        return _gser(a, x)
    return 1.0 - _gcf(a, x)


def gammainc_upper(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) = 1 - P(a, x). Survival of Gamma."""
    return 1.0 - gammainc_lower(a, x)


def gamma_sf(x: float, shape: float, scale: float) -> float:
    """Survival function S(x) = P(D > x) for D ~ Gamma(shape, scale)."""
    if x <= 0:
        return 1.0
    return gammainc_upper(shape, x / scale)


# ---------------------------------------------------------------------------
# Normal CDF / inverse CDF.
# ---------------------------------------------------------------------------
def norm_cdf(z: float) -> float:
    """Standard normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if not 0.0 < p < 1.0:
        if p <= 0.0:
            return -math.inf
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ---------------------------------------------------------------------------
# CUSUM design: Siegmund's ARL approximation, inverted for the threshold h.
# ---------------------------------------------------------------------------
def cusum_arl(h: float, k: float, delta: float = 0.0) -> float:
    """Siegmund's approximation for the average run length of a one-sided CUSUM.

    ``h`` is the decision interval, ``k`` the reference value, and ``delta`` the
    standardized drift of the monitored statistic (delta=0 => in-control ARL0).
    ARL ~ (exp(-2 b Delta) + 2 b Delta - 1) / (2 Delta^2), with Delta = delta-k
    and b = h + 1.166 (the correction for the boundary overshoot).
    """
    b = h + 1.166
    Delta = delta - k
    if abs(Delta) < 1e-6:
        return b * b  # limiting form as Delta -> 0
    return (math.exp(-2.0 * b * Delta) + 2.0 * b * Delta - 1.0) / (2.0 * Delta * Delta)


def cusum_threshold_from_arl(target_arl0: float, k: float) -> float:
    """Invert Siegmund's ARL0 (in-control, delta=0) to solve for the interval h.

    Monotone in h, so a bisection is robust and dependency-free.
    """
    lo, hi = 0.1, 50.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if cusum_arl(mid, k, 0.0) < target_arl0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Wald SPRT boundaries.
# ---------------------------------------------------------------------------
def sprt_bounds(alpha: float, beta: float) -> Tuple[float, float]:
    """Wald's log-LR decision boundaries (log A lower, log B upper).

    Accept H0 (no anomaly) when the cumulative log-LR <= log B_lower;
    accept H1 (anomaly) when it >= log A_upper. Returns (logB_lower, logA_upper).
    """
    logA_upper = math.log((1.0 - beta) / alpha)
    logB_lower = math.log(beta / (1.0 - alpha))
    return logB_lower, logA_upper


def bernoulli_llr(x: int, p1: float, p0: float) -> float:
    """Per-observation log-likelihood ratio for a Bernoulli outcome x in {0,1}."""
    if x:
        return math.log(p1 / p0)
    return math.log((1.0 - p1) / (1.0 - p0))
