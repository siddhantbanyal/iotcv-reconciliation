"""Ground-truth physical world for the IoT inventory-reconciliation simulation.

The simulator owns *true* physical state. A separate *system record* (the WMS)
tracks what the warehouse *believes* is true. The two diverge whenever an event
goes unlogged (theft, misplacement, mis-scan, missed putaway, ghost move). That
divergence is the ground-truth set of discrepancies every method is scored on.

Three populations, one per CV experiment family:
  * ``bins``    -> weight / load-cell lane (true_units vs system_units)
  * ``assets``  -> location lane (true_zone vs expected_zone, via BLE/UWB tags)
  * ``items``   -> dwell lane (process items flowing through stages)

This module is pure state + bookkeeping. Event dynamics live in ``events.py``;
noisy sensing lives in ``sensors/``. Keeping them separate makes the ground
truth auditable and the experiment reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SKU:
    """A stock-keeping unit with a characteristic per-unit weight."""

    sku_id: int
    unit_weight_g: float        # mean grams per unit
    weight_cv: float            # unit-to-unit coefficient of variation


@dataclass
class Bin:
    """A fixed shelf location with a load cell, holding one SKU.

    ``true_units`` is physical reality; ``system_units`` is the WMS record.
    A nonzero ``true_units - system_units`` is a live inventory discrepancy.
    """

    bin_id: int
    sku_id: int
    zone: int
    true_units: int
    system_units: int
    cell_zero_drift_g: float = 0.0   # slowly accumulating load-cell zero drift

    @property
    def discrepancy(self) -> int:
        """Signed unit discrepancy (true minus system)."""
        return self.true_units - self.system_units

    @property
    def has_discrepancy(self) -> bool:
        return self.true_units != self.system_units


@dataclass
class Asset:
    """A movable, tag-tracked asset (pallet/tote) located in a zone.

    ``true_zone`` is where it physically is; ``expected_zone`` is where the WMS
    expects it. A mismatch is a misplaced asset; an unlogged change to
    ``true_zone`` is a ghost move.
    """

    asset_id: int
    true_zone: int
    expected_zone: int
    last_logged_zone: int            # last zone the WMS actually recorded a move to

    @property
    def is_misplaced(self) -> bool:
        return self.true_zone != self.expected_zone


@dataclass
class ProcessItem:
    """An item flowing through process stages (receiving -> staging -> outbound).

    ``entry_tick`` records when the item entered its current ``stage``.
    ``stranded`` is ground-truth: a stranded item dwells far beyond expectation.
    """

    item_id: int
    stage: str
    entry_tick: int
    expected_dwell: float            # nominal expected dwell (ticks) for this stage
    stranded: bool = False
    done: bool = False

    def dwell(self, now: int) -> int:
        return now - self.entry_tick


@dataclass
class WorldState:
    """The complete physical + system state of the simulated warehouse."""

    cfg: dict
    rng: np.random.Generator
    tick: int = 0
    skus: Dict[int, SKU] = field(default_factory=dict)
    bins: Dict[int, Bin] = field(default_factory=dict)
    assets: Dict[int, Asset] = field(default_factory=dict)
    items: Dict[int, ProcessItem] = field(default_factory=dict)
    _next_item_id: int = 0

    # -- construction ------------------------------------------------------
    @classmethod
    def build(cls, cfg: dict, rng: np.random.Generator) -> "WorldState":
        """Initialize a consistent world: system record starts == true state."""
        w = cls(cfg=cfg, rng=rng)
        wc = cfg["world"]

        # SKUs with characteristic unit weights.
        for s in range(wc["n_skus"]):
            uw = rng.uniform(wc["unit_weight_min"], wc["unit_weight_max"])
            w.skus[s] = SKU(sku_id=s, unit_weight_g=uw, weight_cv=wc["unit_weight_cv"])

        # Bins: one SKU each, placed in a zone, true == system at t=0.
        for b in range(wc["n_bins"]):
            sku = int(rng.integers(0, wc["n_skus"]))
            zone = int(rng.integers(0, wc["n_zones"]))
            units = int(rng.integers(wc["init_units_min"], wc["init_units_max"] + 1))
            w.bins[b] = Bin(
                bin_id=b, sku_id=sku, zone=zone,
                true_units=units, system_units=units,
            )

        # Assets: one tag-tracked tote per bin's worth, all correctly placed.
        for a in range(wc["n_bins"]):
            zone = int(rng.integers(0, wc["n_zones"]))
            w.assets[a] = Asset(
                asset_id=a, true_zone=zone, expected_zone=zone, last_logged_zone=zone,
            )

        return w

    # -- weight lane helpers ----------------------------------------------
    def bin_true_weight(self, bin_id: int) -> float:
        """Physical scale reading the load cell *would* see (no sensor noise).

        Includes tare and accumulated zero drift; per-unit weight varies slightly.
        """
        b = self.bins[bin_id]
        sku = self.skus[b.sku_id]
        tare = self.cfg["world"]["tare_weight"]
        # Expected content weight; unit variation is applied at sensing time so
        # ground truth stays deterministic per (bin, tick) given the rng stream.
        content = b.true_units * sku.unit_weight_g
        return tare + content + b.cell_zero_drift_g

    def units_from_weight(self, bin_id: int, weight_g: float) -> float:
        """Invert a (noisy) weight reading into an estimated unit count."""
        b = self.bins[bin_id]
        sku = self.skus[b.sku_id]
        tare = self.cfg["world"]["tare_weight"]
        return max(0.0, (weight_g - tare) / sku.unit_weight_g)

    # -- dwell lane helpers -----------------------------------------------
    def spawn_item(self, stage: str, expected_dwell: float, stranded: bool) -> ProcessItem:
        item = ProcessItem(
            item_id=self._next_item_id, stage=stage, entry_tick=self.tick,
            expected_dwell=expected_dwell, stranded=stranded,
        )
        self.items[self._next_item_id] = item
        self._next_item_id += 1
        return item

    # -- ground-truth discrepancy snapshot --------------------------------
    def discrepancy_snapshot(self) -> Dict[str, set]:
        """Ground-truth set of currently-discrepant entities, by lane.

        Returns location keys (strings) so detectors/metrics can compare against
        a single canonical key space: ``bin:<id>``, ``asset:<id>``, ``item:<id>``.
        """
        weight = {f"bin:{b.bin_id}" for b in self.bins.values() if b.has_discrepancy}
        location = {f"asset:{a.asset_id}" for a in self.assets.values() if a.is_misplaced}
        dwell = {
            f"item:{i.item_id}"
            for i in self.items.values()
            if i.stranded and not i.done
        }
        return {"weight": weight, "location": location, "dwell": dwell}
