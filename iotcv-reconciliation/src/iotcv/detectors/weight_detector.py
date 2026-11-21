"""Weight-lane detector: heteroscedastic GLR-CUSUM on the count innovation.

The count estimate inverted from the load cell, n_hat = (w - tau)/mu_s, has a
variance that GROWS with the amount of stock on the shelf, because per-unit
weight varies (coefficient of variation c_v) and that variation compounds over
units:

    Var(n_hat) = sigma_cell^2 / mu_s^2  +  c_v^2 * n            (heteroscedastic)

A fixed unit threshold on the residual therefore over-fires on full bins and
under-fires on nearly-empty ones. We instead standardize each innovation by its
own tick-and-bin-specific standard deviation,

    z_t = ( n_bar_t - n_sys_t ) / sigma_t ,     sigma_t^2 = sigma_cell^2/mu_s^2 + c_v^2 * n_bar_t

and monitor the standardized stream with a two-sided cumulative-sum (CUSUM)
change detector (Page 1954; Basseville & Nikiforov 1993):

    S^+_t = max(0, S^+_{t-1} + z_t - k),   alarm if S^+_t > h    (surplus vs record)
    S^-_t = max(0, S^-_{t-1} - z_t - k),   alarm if S^-_t > h    (deficit vs record)

The reference value k is half the smallest shift we care to detect (in sigma
units), and the decision interval h is not hand-tuned: it is solved from a target
in-control average run length ARL0 (mean ticks between false alarms) via
Siegmund's approximation (see stats.cusum_threshold_from_arl). This is the GLR
form of the sequential change test specialized to a Gaussian innovation with a
known, count-dependent variance. It replaces the EWMA + fixed-residual +
change-point heuristic with an explicit false-alarm
design target instead of three tuned constants.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from ..handoff import Flag
from ..stats import cusum_threshold_from_arl, norm_cdf
from .base import Detector


class WeightDetector(Detector):
    lane = "weight"

    def __init__(self, world):
        super().__init__(world)
        wc = self.dcfg["weight"]
        self.alpha = wc.get("ewma_alpha", 0.3)
        self.k = wc.get("cusum_k", 0.5)
        self.target_arl0 = wc.get("target_arl0", 500)
        self.min_shift = wc.get("min_shift_units", 1.0)
        # Decision interval solved once from the ARL0 design target.
        self.h = cusum_threshold_from_arl(float(self.target_arl0), self.k)

        self.sigma_cell = world.cfg["sensors"]["weight"]["cell_noise_g"]
        self._ewma: Dict[int, float] = {}
        self._shi: Dict[int, float] = {}   # positive-shift CUSUM (surplus)
        self._slo: Dict[int, float] = {}   # negative-shift CUSUM (deficit)
        self._cooldown: Dict[int, int] = {}

    def _sigma_units(self, bin_id: int, n_plugin: float) -> float:
        """Heteroscedastic std of the count estimate at plug-in count n_plugin."""
        b = self.w.bins[bin_id]
        sku = self.w.skus[b.sku_id]
        mu_s = sku.unit_weight_g
        c_v = sku.weight_cv
        var = (self.sigma_cell / mu_s) ** 2 + (c_v ** 2) * max(n_plugin, 0.0)
        return math.sqrt(max(var, 1e-9))

    def _classify(self, est: float, system: int) -> Tuple[str, str]:
        """Map a signed residual to an anomaly type + human reason."""
        residual = est - system
        if est < 0.5:
            return "empty_bin", f"bin reads empty (est {est:.1f}) but system shows {system}"
        if residual < 0:
            return "phantom_inventory", (
                f"est {est:.1f} < system {system} by {abs(residual):.1f} units")
        if est < max(2.0, 0.15 * max(system, 1)):
            return "low_stock", f"bin low: est {est:.1f} units"
        return "wrong_item", (
            f"weight surplus {residual:+.1f} units inconsistent with record {system}")

    def _reset(self, bin_id: int) -> None:
        self._shi[bin_id] = 0.0
        self._slo[bin_id] = 0.0
        self._cooldown[bin_id] = 10

    def update(self, readings) -> List[Flag]:
        flags: List[Flag] = []
        for kk in list(self._cooldown):
            self._cooldown[kk] -= 1
            if self._cooldown[kk] <= 0:
                del self._cooldown[kk]

        for bin_id, r in readings.items():
            prev = self._ewma.get(bin_id, r.est_units)
            smoothed = self.alpha * r.est_units + (1 - self.alpha) * prev
            self._ewma[bin_id] = smoothed

            # Standardized innovation against the WMS record.
            sigma = self._sigma_units(bin_id, smoothed)
            z = (smoothed - r.system_units) / sigma

            shi = max(0.0, self._shi.get(bin_id, 0.0) + z - self.k)
            slo = max(0.0, self._slo.get(bin_id, 0.0) - z - self.k)
            self._shi[bin_id] = shi
            self._slo[bin_id] = slo

            if bin_id in self._cooldown:
                continue

            stat = max(shi, slo)
            residual = smoothed - r.system_units
            # Require both a CUSUM alarm and a materially-sized residual, so we do
            # not emit on sub-unit noise even when the ARL boundary is crossed.
            if stat <= self.h or abs(residual) < self.min_shift:
                continue

            anomaly_type, reason = self._classify(smoothed, r.system_units)
            # Confidence = calibrated tail evidence: how far past the boundary,
            # squashed through the standard-normal CDF of the excess.
            excess = (stat - self.h) / max(self.h, 1e-9)
            conf = norm_cdf(1.5 * excess)  # in (0.5, 1); monotone in excess
            b = self.w.bins[bin_id]
            flags.append(Flag(
                flag_id=self._flag_id(), tick=self.w.tick, lane=self.lane,
                anomaly_type=anomaly_type, location=f"bin:{bin_id}", zone=b.zone,
                reason=reason, confidence=round(float(conf), 3),
                evidence={
                    "smoothed_units": round(float(smoothed), 3),
                    "system_units": int(r.system_units),
                    "residual": round(float(residual), 3),
                    "z": round(float(z), 3),
                    "cusum": round(float(stat), 3),
                    "h": round(float(self.h), 3),
                    "sigma_units": round(float(sigma), 3),
                },
            ))
            self._reset(bin_id)
        return flags
