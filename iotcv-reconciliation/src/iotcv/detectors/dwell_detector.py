"""Dwell-lane detector: Gamma survival (upper-tail) test.

Dwell times are generated from a Gamma law (positive, right-skewed), so a
Gaussian "z sigma above the mean" rule is statistically inconsistent with the
data-generating process: it mis-states tail probabilities and its effective
false-alarm rate drifts with the stage's coefficient of variation. We instead
test directly on the Gamma survival function.

For stage dwell D ~ Gamma(kappa, theta) with shape kappa = 1/c_v^2 and scale
theta = mu_D / kappa (matching the simulator's parameterization), the
probability of dwelling at least as long as observed is the regularized upper
incomplete gamma

    S(d) = P(D >= d) = Q(kappa, d / theta).

An item is flagged as stranded when this tail probability falls below a target
level eps, i.e. the observed dwell is in the extreme upper tail of the *correct*
distribution:

    S(d_obs) < eps      <=>     Q(kappa, d_obs / theta) < eps.

eps plays the role of a per-item one-sided false-alarm rate and is interpretable
(e.g. eps = 0.02 flags the top 2% of the dwell distribution), unlike a raw sigma
multiplier. The flag confidence is the calibrated complement 1 - S(d_obs), the
model probability that a normal item would not have dwelled this long.
"""

from __future__ import annotations

from typing import Dict, List, Set

from ..handoff import Flag
from ..stats import gamma_sf
from .base import Detector


class DwellDetector(Detector):
    lane = "dwell"

    def __init__(self, world):
        super().__init__(world)
        self.eps = self.dcfg["dwell"].get("survival_eps", 0.02)
        self.cv = world.cfg["process"]["dwell_cv"]
        self.kappa = 1.0 / (self.cv ** 2)          # Gamma shape
        self._flagged: Set[int] = set()            # flag each item at most once

    def update(self, readings) -> List[Flag]:
        flags: List[Flag] = []
        for item_id, r in readings.items():
            if item_id in self._flagged:
                continue
            # Scale so that E[D] = kappa * theta = expected_dwell.
            theta = max(1e-9, r.expected_dwell / self.kappa)
            surv = gamma_sf(r.observed_dwell, self.kappa, theta)
            if surv >= self.eps:
                continue
            conf = 1.0 - surv
            flags.append(Flag(
                flag_id=self._flag_id(), tick=self.w.tick, lane=self.lane,
                anomaly_type="stranded", location=f"item:{item_id}", zone=-1,
                reason=(f"dwell {r.observed_dwell:.0f} in Gamma upper tail "
                        f"(survival {surv:.4f} < {self.eps}) at stage {r.stage}, "
                        f"expected {r.expected_dwell:.0f}"),
                confidence=round(float(conf), 4),
                evidence={
                    "observed_dwell": round(float(r.observed_dwell), 2),
                    "expected_dwell": round(float(r.expected_dwell), 2),
                    "survival": round(float(surv), 5),
                    "kappa": round(float(self.kappa), 3),
                    "stage": r.stage,
                },
            ))
            self._flagged.add(item_id)
        return flags
