"""Confidence calibration for detector flags, in pure NumPy.

Each detector emits a confidence in [0,1]. For that confidence to be usable as a
probability, e.g. to rank flags for a budgeted handoff or to threshold at a
target precision, it should be *calibrated*: among flags with confidence ~q, a
fraction ~q should be true positives. Raw detector scores rarely satisfy this, so
we fit a post-hoc calibration map on held-out data and report calibration
quality.

Two standard calibrators (Platt 1999; Zadrozny & Elkan 2002):

  * ``PlattScaler``: logistic map sigma(a*s + b), fit by Newton-Raphson on the
                         held-out (score, label) pairs. Parametric, robust on
                         small samples.
  * ``IsotonicCalibrator``: nonparametric monotone fit via the pool-adjacent-
                         violators algorithm (PAVA). Flexible, no shape assumption.

Quality is summarized by the Expected Calibration Error (Naeini et al. 2015):

  ECE = sum_m (|B_m| / N) * | acc(B_m) - conf(B_m) |

over M equal-width confidence bins, plus a reliability table (per-bin mean
confidence vs empirical accuracy) for a reliability diagram.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Platt scaling: logistic calibration sigma(a*s + b).
# ---------------------------------------------------------------------------
@dataclass
class PlattScaler:
    a: float = 1.0
    b: float = 0.0

    def fit(self, scores: np.ndarray, labels: np.ndarray,
            iters: int = 100, lr: float = 0.1) -> "PlattScaler":
        s = np.asarray(scores, dtype=float)
        y = np.asarray(labels, dtype=float)
        a, b = 1.0, 0.0
        n = len(s)
        if n == 0:
            self.a, self.b = a, b
            return self
        for _ in range(iters):
            z = a * s + b
            p = 1.0 / (1.0 + np.exp(-z))
            # gradient of mean binary cross-entropy
            ga = np.mean((p - y) * s)
            gb = np.mean(p - y)
            a -= lr * ga
            b -= lr * gb
        self.a, self.b = float(a), float(b)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        z = self.a * np.asarray(scores, dtype=float) + self.b
        return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# Isotonic regression via Pool Adjacent Violators (PAVA).
# ---------------------------------------------------------------------------
@dataclass
class IsotonicCalibrator:
    x_thresh: np.ndarray = field(default_factory=lambda: np.array([]))
    y_level: np.ndarray = field(default_factory=lambda: np.array([]))

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        s = np.asarray(scores, dtype=float)
        y = np.asarray(labels, dtype=float)
        order = np.argsort(s, kind="mergesort")
        s, y = s[order], y[order]
        n = len(s)
        if n == 0:
            return self
        # PAVA: maintain blocks of (sum, weight); merge while monotonicity violated.
        vals = list(y)
        wts = [1.0] * n
        xs = list(s)
        i = 0
        blocks: List[Tuple[float, float, float]] = []  # (mean, weight, x_right)
        for k in range(n):
            cur_mean, cur_w, cur_x = vals[k], wts[k], xs[k]
            while blocks and blocks[-1][0] >= cur_mean:
                pm, pw, px = blocks.pop()
                cur_mean = (pm * pw + cur_mean * cur_w) / (pw + cur_w)
                cur_w = pw + cur_w
            blocks.append((cur_mean, cur_w, cur_x))
        self.x_thresh = np.array([b[2] for b in blocks])
        self.y_level = np.clip(np.array([b[0] for b in blocks]), 0.0, 1.0)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        s = np.asarray(scores, dtype=float)
        if len(self.x_thresh) == 0:
            return s
        idx = np.searchsorted(self.x_thresh, s, side="left")
        idx = np.clip(idx, 0, len(self.y_level) - 1)
        return self.y_level[idx]


# ---------------------------------------------------------------------------
# Expected Calibration Error + reliability table.
# ---------------------------------------------------------------------------
def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int = 10) -> Tuple[float, list]:
    """Return (ECE, reliability_rows). Each row: (bin_lo, bin_hi, count, conf, acc)."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    n = len(p)
    if n == 0:
        return float("nan"), []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows = []
    for m in range(n_bins):
        lo, hi = edges[m], edges[m + 1]
        mask = (p > lo) & (p <= hi) if m > 0 else (p >= lo) & (p <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            rows.append((float(lo), float(hi), 0, float("nan"), float("nan")))
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        ece += (cnt / n) * abs(acc - conf)
        rows.append((float(lo), float(hi), cnt, conf, acc))
    return float(ece), rows
