"""Load-cell shelf telemetry emulator (weight lane).

Turns a bin's true physical weight into a noisy sensor reading and an inverted
unit-count estimate. Models the realities a smart shelf actually faces:
  * additive Gaussian read noise per sample,
  * slow zero drift (handled in events; reflected via WorldState.bin_true_weight),
  * per-unit weight variability (so count inversion is imperfect near boundaries).

Parameter seeds for the noise model are motivated by the Smart Shelves literature
(see the paper's references); exact values are configured in configs/*.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from ..world import WorldState


@dataclass
class WeightReading:
    bin_id: int
    tick: int
    weight_g: float          # noisy load-cell reading
    est_units: float         # inverted unit-count estimate
    system_units: int        # WMS record at read time (for residual computation)


class WeightSensor:
    """Emits per-bin weight readings each tick."""

    def __init__(self, world: WorldState):
        self.w = world
        self.sc = world.cfg["sensors"]["weight"]

    def read(self) -> Dict[int, WeightReading]:
        """Return a reading per bin for the current tick (subject to sample_every)."""
        readings: Dict[int, WeightReading] = {}
        if self.w.tick % self.sc["sample_every"] != 0:
            return readings
        noise_sd = self.sc["cell_noise_g"]
        for b in self.w.bins.values():
            true_w = self.w.bin_true_weight(b.bin_id)
            # Per-unit variability folded into read noise scaled by count.
            unit_var = self.w.skus[b.sku_id].unit_weight_g * self.w.skus[b.sku_id].weight_cv
            content_sd = unit_var * np.sqrt(max(b.true_units, 0))
            total_sd = float(np.sqrt(noise_sd ** 2 + content_sd ** 2))
            obs = float(self.w.rng.normal(true_w, total_sd))
            est = self.w.units_from_weight(b.bin_id, obs)
            readings[b.bin_id] = WeightReading(
                bin_id=b.bin_id, tick=self.w.tick,
                weight_g=obs, est_units=est, system_units=b.system_units,
            )
        return readings
