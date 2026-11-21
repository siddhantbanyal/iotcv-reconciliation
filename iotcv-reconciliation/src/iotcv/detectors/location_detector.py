"""Location-lane detector: Wald Sequential Probability Ratio Test (SPRT).

Localization is noisy and intermittent, so a single mismatched read cannot be
trusted. Rather than wait for a hand-set number of consecutive mismatches, we
treat each informative read as a Bernoulli trial and run Wald's SPRT (Wald 1945)
between two hypotheses about the asset's placement:

    H0 (correctly placed):  P(read mismatches expected zone) = p0  (~ BLE error e_ble)
    H1 (misplaced/moved):    P(read mismatches expected zone) = p1  (>> p0)

Each read x in {0,1} (1 = observed zone != expected zone) contributes a
log-likelihood ratio, and we accumulate

    Lambda_t = sum_j  log[ p1^{x_j} (1-p1)^{1-x_j} / p0^{x_j} (1-p0)^{1-x_j} ].

We stop and declare "misplaced" when Lambda_t >= log((1-beta)/alpha), and reset
to the correct hypothesis when Lambda_t <= log(beta/(1-alpha)). Those two Wald
boundaries are set directly by the target type-I / type-II error rates
(alpha, beta); the SPRT is optimal in the expected number of reads to a decision
(Wald & Wolfowitz 1948). The old fixed "persistence window" is thus replaced by
an emergent stopping time whose false-accept / miss rates are design parameters
rather than side effects of a tuned constant.

The stopping LLR also yields a calibrated posterior for the flag confidence:
with prior odds 1 (uninformative), P(H1 | Lambda) = sigmoid(Lambda).
"""

from __future__ import annotations

import math
from typing import Dict, List

from ..handoff import Flag
from ..stats import bernoulli_llr, sprt_bounds
from .base import Detector


class LocationDetector(Detector):
    lane = "location"

    def __init__(self, world):
        super().__init__(world)
        lc = self.dcfg["location"]
        self.alpha = lc.get("sprt_alpha", 0.01)
        self.beta = lc.get("sprt_beta", 0.05)
        self.p0 = lc.get("p_mismatch_correct", 0.18)
        self.p1 = lc.get("p_mismatch_misplaced", 0.85)
        self.logB_lower, self.logA_upper = sprt_bounds(self.alpha, self.beta)

        self._llr: Dict[int, float] = {}          # cumulative log-LR per asset
        self._reads: Dict[int, int] = {}          # informative reads since reset
        self._mism: Dict[int, int] = {}           # mismatches since reset
        self._last_est: Dict[int, int] = {}
        self._cooldown: Dict[int, int] = {}

    def _reset_llr(self, asset_id: int) -> None:
        self._llr[asset_id] = 0.0
        self._reads[asset_id] = 0
        self._mism[asset_id] = 0

    def update(self, readings) -> List[Flag]:
        flags: List[Flag] = []
        for k in list(self._cooldown):
            self._cooldown[k] -= 1
            if self._cooldown[k] <= 0:
                del self._cooldown[k]

        for asset_id, r in readings.items():
            # Dropout reads carry no positional information; skip (no LLR update).
            if r.source == "dropout":
                continue

            x = 1 if r.est_zone != r.expected_zone else 0
            llr = self._llr.get(asset_id, 0.0) + bernoulli_llr(x, self.p1, self.p0)
            reads = self._reads.get(asset_id, 0) + 1
            mism = self._mism.get(asset_id, 0) + x

            prev_est = self._last_est.get(asset_id)
            self._last_est[asset_id] = r.est_zone

            # Accept H0 (correct): reset the test and move on.
            if llr <= self.logB_lower:
                self._llr[asset_id] = 0.0
                self._reads[asset_id] = 0
                self._mism[asset_id] = 0
                continue

            self._llr[asset_id] = llr
            self._reads[asset_id] = reads
            self._mism[asset_id] = mism

            if asset_id in self._cooldown:
                continue

            # Accept H1 (misplaced/moved): SPRT crossed the upper Wald boundary.
            if llr < self.logA_upper:
                continue

            a = self.w.assets[asset_id]
            if a.expected_zone == a.last_logged_zone and prev_est == r.est_zone:
                anomaly_type = "ghost_move"
                reason = (f"SPRT accepts relocation: {mism}/{reads} mismatched reads "
                          f"in zone {r.est_zone}; no WMS move (expected "
                          f"{r.expected_zone}), LLR {llr:.2f} >= {self.logA_upper:.2f}")
            else:
                anomaly_type = "zone_mismatch"
                reason = (f"SPRT accepts mismatch: asset in zone {r.est_zone} vs "
                          f"expected {r.expected_zone}, {mism}/{reads} mismatched "
                          f"reads, LLR {llr:.2f} >= {self.logA_upper:.2f}")

            # Calibrated posterior for H1 under uninformative prior odds.
            conf = 1.0 / (1.0 + math.exp(-llr))
            flags.append(Flag(
                flag_id=self._flag_id(), tick=self.w.tick, lane=self.lane,
                anomaly_type=anomaly_type, location=f"asset:{asset_id}",
                zone=r.est_zone, reason=reason, confidence=round(float(conf), 3),
                evidence={
                    "est_zone": int(r.est_zone),
                    "expected_zone": int(r.expected_zone),
                    "llr": round(float(llr), 3),
                    "llr_bound": round(float(self.logA_upper), 3),
                    "reads": int(reads),
                    "mismatches": int(mism),
                    "source": r.source,
                },
            ))
            self._cooldown[asset_id] = 15
            self._reset_llr(asset_id)
        return flags
