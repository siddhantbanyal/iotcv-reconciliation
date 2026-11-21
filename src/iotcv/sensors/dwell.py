"""Process-timing emulator (dwell lane).

Observes how long each in-process item has dwelled in its current stage. The only
sensing imperfection modeled is timestamp jitter on stage entry (clocks/scan lag),
which means a naive "over expected dwell" rule will occasionally fire late or early
near the boundary. The dwell detector handles that with a sigma-based threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ..world import WorldState


@dataclass
class DwellReading:
    item_id: int
    tick: int
    stage: str
    observed_dwell: float    # observed (jittered) dwell in ticks
    expected_dwell: float    # nominal expected dwell for the stage/item


class DwellSensor:
    """Emits dwell observations for all active (not-done) process items."""

    def __init__(self, world: WorldState):
        self.w = world
        self.jitter = world.cfg["sensors"]["dwell"]["timestamp_jitter"]

    def read(self) -> Dict[int, DwellReading]:
        readings: Dict[int, DwellReading] = {}
        for it in self.w.items.values():
            if it.done:
                continue
            true_dwell = it.dwell(self.w.tick)
            if self.jitter > 0:
                j = int(self.w.rng.integers(-self.jitter, self.jitter + 1))
            else:
                j = 0
            obs = max(0.0, true_dwell + j)
            readings[it.item_id] = DwellReading(
                item_id=it.item_id, tick=self.w.tick, stage=it.stage,
                observed_dwell=obs, expected_dwell=it.expected_dwell)
        return readings
