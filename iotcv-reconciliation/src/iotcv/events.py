"""Event generator: drives the per-tick dynamics that create inventory drift.

Each tick, events mutate physical reality (the ``WorldState`` true fields) and,
*when logged*, the system record. The split between logged and unlogged events is
the entire source of ground-truth discrepancy:

  logged   -> true state AND system record both update (no drift introduced)
  unlogged -> only true state updates                (drift introduced)

Drift event types modeled:
  * mis-scan       : a pick/putaway recorded with the wrong quantity (or not at all)
  * shrink         : units physically leave a bin with no record (phantom inventory)
  * misplace       : putaway lands in the wrong zone; WMS expects another zone
  * ghost move     : a tagged asset relocates with no matching WMS move
  * missed putaway : received stock is left stranded instead of being put away

The generator also emits a per-tick ``WMSLog`` of *logged* transactions, which the
detectors use to decide whether an observed change "should" have a system event.
Everything is driven by the world's seeded rng for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .world import WorldState


@dataclass
class WMSEvent:
    """A single logged warehouse transaction (what the system record knows)."""

    tick: int
    kind: str                  # 'sale' | 'replenish' | 'move'
    target: str                # canonical key, e.g. 'bin:3' or 'asset:7'
    qty: int = 0               # signed unit delta for bin events
    from_zone: int = -1        # for move events
    to_zone: int = -1


@dataclass
class TickLog:
    """All logged WMS events for one tick, indexed by target for fast lookup."""

    tick: int
    events: List[WMSEvent] = field(default_factory=list)
    _by_target: Dict[str, List[WMSEvent]] = field(default_factory=dict)

    def add(self, ev: WMSEvent) -> None:
        self.events.append(ev)
        self._by_target.setdefault(ev.target, []).append(ev)

    def for_target(self, target: str) -> List[WMSEvent]:
        return self._by_target.get(target, [])


class EventGenerator:
    """Advances the world by one tick, returning the logged WMS events for it."""

    def __init__(self, world: WorldState):
        self.w = world
        self.cfg = world.cfg
        self.rng = world.rng

    # -- main step ---------------------------------------------------------
    def step(self) -> TickLog:
        """Apply one tick of events; return the tick's logged WMS transactions."""
        log = TickLog(tick=self.w.tick)
        self._weight_events(log)
        self._location_events(log)
        self._dwell_events(log)
        self._advance_passive_state()
        self.w.tick += 1
        return log

    # -- weight lane: sales, replenishment, shrink, mis-scan ---------------
    def _weight_events(self, log: TickLog) -> None:
        ec = self.cfg["events"]
        for b in self.w.bins.values():
            key = f"bin:{b.bin_id}"

            # Sale / pick.
            if self.rng.random() < ec["p_sale"] and b.true_units > 0:
                qty = int(self.rng.integers(1, ec["sale_qty_max"] + 1))
                qty = min(qty, b.true_units)
                b.true_units -= qty
                if self.rng.random() < ec["p_mis_scan"]:
                    # Mis-scan: record a wrong quantity (off by 1..qty) or nothing.
                    logged_qty = qty - int(self.rng.integers(1, qty + 1)) if qty > 0 else 0
                    b.system_units -= logged_qty
                    if logged_qty:
                        log.add(WMSEvent(self.w.tick, "sale", key, qty=-logged_qty))
                else:
                    b.system_units -= qty
                    log.add(WMSEvent(self.w.tick, "sale", key, qty=-qty))
                b.system_units = max(0, b.system_units)

            # Replenishment / putaway.
            if self.rng.random() < ec["p_replenish"]:
                qty = int(self.rng.integers(1, ec["replenish_qty_max"] + 1))
                b.true_units += qty
                if self.rng.random() < ec["p_mis_scan"]:
                    logged_qty = qty - int(self.rng.integers(1, qty + 1))
                    b.system_units += max(0, logged_qty)
                    if logged_qty > 0:
                        log.add(WMSEvent(self.w.tick, "replenish", key, qty=logged_qty))
                else:
                    b.system_units += qty
                    log.add(WMSEvent(self.w.tick, "replenish", key, qty=qty))

            # Shrink: units physically disappear with NO system record => phantom.
            if self.rng.random() < ec["p_shrink"] and b.true_units > 0:
                loss = int(self.rng.integers(1, min(3, b.true_units) + 1))
                b.true_units -= loss
                # intentionally no log + no system_units change

    # -- location lane: misplaced putaway + ghost moves --------------------
    def _location_events(self, log: TickLog) -> None:
        ec = self.cfg["events"]
        n_zones = self.cfg["world"]["n_zones"]
        for a in self.w.assets.values():
            key = f"asset:{a.asset_id}"

            # Logged move: true + expected + last_logged all advance together.
            if self.rng.random() < ec["p_move"]:
                new_zone = int(self.rng.integers(0, n_zones))
                if self.rng.random() < ec["p_misplace"]:
                    # Misplaced putaway: physically goes to new_zone, but the WMS
                    # move is recorded to a DIFFERENT (expected) zone.
                    wrong = int(self.rng.integers(0, n_zones))
                    a.true_zone = new_zone
                    a.expected_zone = wrong
                    a.last_logged_zone = wrong
                    log.add(WMSEvent(self.w.tick, "move", key,
                                     from_zone=a.true_zone, to_zone=wrong))
                else:
                    a.true_zone = new_zone
                    a.expected_zone = new_zone
                    a.last_logged_zone = new_zone
                    log.add(WMSEvent(self.w.tick, "move", key,
                                     from_zone=a.last_logged_zone, to_zone=new_zone))

            # Ghost move: physical relocation with NO WMS event => mismatch.
            elif self.rng.random() < ec["p_ghost_move"]:
                new_zone = int(self.rng.integers(0, n_zones))
                if new_zone != a.true_zone:
                    a.true_zone = new_zone
                    # expected_zone / last_logged_zone unchanged => now misplaced

    # -- dwell lane: spawn process items, some stranded --------------------
    def _dwell_events(self, log: TickLog) -> None:
        ec = self.cfg["events"]
        pc = self.cfg["process"]
        # Receiving inflow each tick.
        if self.rng.random() < ec["p_replenish"]:
            stage = "receiving"
            base = pc["expected_dwell"][stage]
            exp = self._gamma_dwell(base, pc["dwell_cv"])
            stranded = self.rng.random() < ec["p_missed_putaway"]
            self.w.spawn_item(stage, exp, stranded)

    # -- passive per-tick state advance (dwell completion, load drift) -----
    def _advance_passive_state(self) -> None:
        pc = self.cfg["process"]
        sc = self.cfg["sensors"]["weight"]
        now = self.w.tick

        # Load-cell zero drift accumulates slowly on every bin.
        for b in self.w.bins.values():
            b.cell_zero_drift_g += sc["drift_g_per_tick"]

        # Process items complete after their (possibly stranded) dwell elapses.
        for it in self.w.items.values():
            if it.done:
                continue
            limit = it.expected_dwell * (pc["stranded_dwell_mult"] if it.stranded else 1.0)
            if it.dwell(now) >= limit:
                # Non-stranded items move on / leave the system; stranded ones
                # remain flagged as ground-truth discrepancies until inspected.
                if not it.stranded:
                    it.done = True

    # -- helpers -----------------------------------------------------------
    def _gamma_dwell(self, mean: float, cv: float) -> float:
        """Draw a positive dwell time with given mean and coefficient of variation."""
        if cv <= 0:
            return float(mean)
        shape = 1.0 / (cv * cv)
        scale = mean / shape
        return float(self.rng.gamma(shape, scale))
