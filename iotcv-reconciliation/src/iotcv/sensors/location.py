"""BLE/UWB asset-tag localization emulator (location lane).

Hybrid positioning model motivated by the BLE-UWB Kalman / HYMN literature
(see the paper's references): BLE is a cheap, continuous, noisy zone estimate that can drop
out; UWB provides a periodic, high-accuracy correction. The emulator reports an
estimated zone per asset, which the location detector compares against the WMS
expected zone.

We model "zone error" directly (probability the reported zone differs from the
true zone) rather than simulating raw RSSI/ToF, keeping the simulator lightweight
and defensible while preserving the key realism: localization is uncertain and
intermittent, so naive single-read comparisons produce false alarms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..world import WorldState


@dataclass
class LocationReading:
    asset_id: int
    tick: int
    est_zone: int            # localization estimate (BLE, or UWB-corrected)
    expected_zone: int       # WMS expected zone at read time
    source: str              # "ble" | "uwb" | "dropout"
    accurate: bool           # whether this read used the accurate UWB source


class LocationSensor:
    """Emits per-asset zone estimates each tick (BLE continuous, UWB periodic)."""

    def __init__(self, world: WorldState):
        self.w = world
        self.sc = world.cfg["sensors"]["location"]
        self.n_zones = world.cfg["world"]["n_zones"]

    def _wrong_zone(self, true_zone: int) -> int:
        """Pick a plausible *adjacent* wrong zone (localization confusion)."""
        if self.n_zones <= 1:
            return true_zone
        offset = 1 if self.w.rng.random() < 0.5 else -1
        return (true_zone + offset) % self.n_zones

    def read(self) -> Dict[int, LocationReading]:
        readings: Dict[int, LocationReading] = {}
        use_uwb = (self.w.tick % self.sc["uwb_every"]) == 0
        for a in self.w.assets.values():
            if use_uwb:
                err = self.sc["uwb_zone_error"]
                est = a.true_zone if self.w.rng.random() >= err else self._wrong_zone(a.true_zone)
                readings[a.asset_id] = LocationReading(
                    a.asset_id, self.w.tick, est, a.expected_zone, "uwb", True)
                continue
            # BLE path: may drop out, else noisy.
            if self.w.rng.random() < self.sc["ble_dropout"]:
                # Dropout: report last expected (no new info) flagged as dropout.
                readings[a.asset_id] = LocationReading(
                    a.asset_id, self.w.tick, a.expected_zone, a.expected_zone,
                    "dropout", False)
                continue
            err = self.sc["ble_zone_error"]
            est = a.true_zone if self.w.rng.random() >= err else self._wrong_zone(a.true_zone)
            readings[a.asset_id] = LocationReading(
                a.asset_id, self.w.tick, est, a.expected_zone, "ble", False)
        return readings
