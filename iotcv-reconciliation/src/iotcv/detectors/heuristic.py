"""Fixed-threshold heuristic detectors (pre-sequential-test baselines).

These reproduce the conventional fixed-threshold detector family: a fixed
residual + change-point rule on the raw count (weight), a fixed persistence
window on zone mismatches (location), and a Gaussian sigma threshold on dwell.
They exist so the sequential tests can be compared against them on identical
seeds, testing whether error-rate-designed sequential tests improve on
hand-set thresholds. Kept intentionally close to
common practice; not tuned per seed.

The design constants below are the conventional defaults (residual ~ 1.5 units,
persistence ~ 3 reads, z ~ 2.5 sigma); they are read from the ``heuristic``
config block when present so the comparison is configurable.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from ..handoff import Flag
from .base import Detector

_DEF = {
    "weight": {"ewma_alpha": 0.3, "residual_units": 1.5, "changepoint_units": 2.0},
    "location": {"persistence": 3},
    "dwell": {"z_threshold": 2.5},
}


def _hc(cfg: dict, lane: str, key: str):
    return cfg.get("heuristic", {}).get(lane, {}).get(key, _DEF[lane][key])


class HeuristicWeightDetector(Detector):
    """EWMA + fixed residual threshold + change-point (homoscedastic, unit-scaled)."""

    lane = "weight"

    def __init__(self, world):
        super().__init__(world)
        self.alpha = _hc(world.cfg, "weight", "ewma_alpha")
        self.residual_units = _hc(world.cfg, "weight", "residual_units")
        self.changepoint_units = _hc(world.cfg, "weight", "changepoint_units")
        self._ewma: Dict[int, float] = {}
        self._cooldown: Dict[int, int] = {}

    def _classify(self, est: float, system: int) -> Tuple[str, str]:
        residual = est - system
        if est < 0.5:
            return "empty_bin", f"bin reads empty (est {est:.1f}) vs system {system}"
        if residual < -self.residual_units:
            return "phantom_inventory", (
                f"est {est:.1f} < system {system} by {abs(residual):.1f} units")
        if est < max(2.0, 0.15 * max(system, 1)):
            return "low_stock", f"bin low: est {est:.1f} units"
        return "wrong_item", (
            f"weight residual {residual:+.1f} inconsistent with record {system}")

    def update(self, readings) -> List[Flag]:
        flags: List[Flag] = []
        for k in list(self._cooldown):
            self._cooldown[k] -= 1
            if self._cooldown[k] <= 0:
                del self._cooldown[k]
        for bin_id, r in readings.items():
            prev = self._ewma.get(bin_id, r.est_units)
            smoothed = self.alpha * r.est_units + (1 - self.alpha) * prev
            self._ewma[bin_id] = smoothed
            if bin_id in self._cooldown:
                continue
            residual = smoothed - r.system_units
            step = abs(smoothed - prev)
            if abs(residual) <= self.residual_units and step <= self.changepoint_units:
                continue
            atype, reason = self._classify(smoothed, r.system_units)
            conf = min(1.0, abs(residual) / (3 * self.residual_units + 1e-9))
            b = self.w.bins[bin_id]
            flags.append(Flag(
                flag_id=self._flag_id(), tick=self.w.tick, lane=self.lane,
                anomaly_type=atype, location=f"bin:{bin_id}", zone=b.zone,
                reason=reason, confidence=round(float(conf), 3),
                evidence={"smoothed_units": round(float(smoothed), 3),
                          "system_units": int(r.system_units),
                          "residual": round(float(residual), 3)}))
            self._cooldown[bin_id] = 10
        return flags


class HeuristicLocationDetector(Detector):
    """Fixed persistence window on consecutive zone mismatches."""

    lane = "location"

    def __init__(self, world):
        super().__init__(world)
        self.persistence = _hc(world.cfg, "location", "persistence")
        self._streak: Dict[int, int] = {}
        self._last_est: Dict[int, int] = {}
        self._cooldown: Dict[int, int] = {}

    def update(self, readings) -> List[Flag]:
        flags: List[Flag] = []
        for k in list(self._cooldown):
            self._cooldown[k] -= 1
            if self._cooldown[k] <= 0:
                del self._cooldown[k]
        for asset_id, r in readings.items():
            if r.source == "dropout":
                continue
            if r.est_zone != r.expected_zone:
                self._streak[asset_id] = self._streak.get(asset_id, 0) + 1
            else:
                self._streak[asset_id] = 0
            prev_est = self._last_est.get(asset_id)
            self._last_est[asset_id] = r.est_zone
            if asset_id in self._cooldown:
                continue
            if self._streak.get(asset_id, 0) < self.persistence:
                continue
            a = self.w.assets[asset_id]
            if a.expected_zone == a.last_logged_zone and prev_est == r.est_zone:
                atype = "ghost_move"
                reason = (f"{self._streak[asset_id]} consecutive mismatched reads "
                          f"in zone {r.est_zone}; no WMS move")
            else:
                atype = "zone_mismatch"
                reason = (f"asset in zone {r.est_zone} vs expected {r.expected_zone} "
                          f"for {self._streak[asset_id]} reads")
            conf = min(1.0, self._streak[asset_id] / (2 * self.persistence))
            flags.append(Flag(
                flag_id=self._flag_id(), tick=self.w.tick, lane=self.lane,
                anomaly_type=atype, location=f"asset:{asset_id}",
                zone=r.est_zone, reason=reason, confidence=round(float(conf), 3),
                evidence={"est_zone": int(r.est_zone),
                          "expected_zone": int(r.expected_zone),
                          "streak": int(self._streak[asset_id])}))
            self._cooldown[asset_id] = 15
            self._streak[asset_id] = 0
        return flags


class HeuristicDwellDetector(Detector):
    """Gaussian sigma threshold on observed vs expected dwell."""

    lane = "dwell"

    def __init__(self, world):
        super().__init__(world)
        self.z = _hc(world.cfg, "dwell", "z_threshold")
        self.cv = world.cfg["process"]["dwell_cv"]
        self._flagged: Set[int] = set()

    def update(self, readings) -> List[Flag]:
        flags: List[Flag] = []
        for item_id, r in readings.items():
            if item_id in self._flagged:
                continue
            sigma = max(1e-6, self.cv * r.expected_dwell)
            if r.observed_dwell <= r.expected_dwell + self.z * sigma:
                continue
            excess = (r.observed_dwell - r.expected_dwell) / sigma
            conf = min(1.0, excess / (2 * self.z))
            flags.append(Flag(
                flag_id=self._flag_id(), tick=self.w.tick, lane=self.lane,
                anomaly_type="stranded", location=f"item:{item_id}", zone=-1,
                reason=(f"dwell {r.observed_dwell:.0f} > expected "
                        f"{r.expected_dwell:.0f} + {self.z} sigma"),
                confidence=round(float(conf), 3),
                evidence={"observed_dwell": round(float(r.observed_dwell), 2),
                          "expected_dwell": round(float(r.expected_dwell), 2),
                          "sigma_excess": round(float(excess), 2)}))
            self._flagged.add(item_id)
        return flags
