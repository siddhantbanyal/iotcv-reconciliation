"""IoT simulation engine: run one world + event stream, collect the flags.

One loop covers the full sensing layer. We build a ground-truth world,
step it tick by tick, sense it with the three noisy sensor lanes, run the edge
detectors, and record two things per tick:

  * ``truth_per_tick``: the ground-truth set of discrepant locations (from the
                           simulator's true-vs-system state), the labels every
                           metric is scored against;
  * ``flags_per_tick``: the ``Flag`` objects the detectors raised, i.e. the
                           IoT->CV handoff payload.

CV is not modeled here; the output is the flag stream (exportable as JSONL).
``truth_at_tick`` carries a per-location "truly discrepant now?" lookup so the
metrics module can label each flag as a true or false alarm without re-deriving
ground truth. ``onset`` records the first tick each discrepancy episode began,
the raw material for time-to-flag and exposure.

Determinism: everything is driven by a single seeded ``numpy.random.Generator``;
the same seed reproduces the same world, events, and sensor noise exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .detectors.dwell_detector import DwellDetector
from .detectors.location_detector import LocationDetector
from .detectors.weight_detector import WeightDetector
from .events import EventGenerator
from .handoff import Flag
from .sensors.dwell import DwellSensor
from .sensors.location import LocationSensor
from .sensors.weight import WeightSensor
from .world import WorldState


def location_is_truly_discrepant(world: WorldState, location: str) -> bool:
    """Ground-truth check for a canonical location key ("bin:i"/"asset:i"/"item:i")."""
    kind, _, sid = location.partition(":")
    idx = int(sid)
    if kind == "bin":
        return world.bins[idx].has_discrepancy if idx in world.bins else False
    if kind == "asset":
        return world.assets[idx].is_misplaced if idx in world.assets else False
    if kind == "item":
        it = world.items.get(idx)
        return bool(it and it.stranded and not it.done)
    return False


def _all_truth(world: WorldState) -> set:
    snap = world.discrepancy_snapshot()
    return snap["weight"] | snap["location"] | snap["dwell"]


@dataclass
class SimRun:
    """Everything the metrics module needs from one seeded simulation run.

    All lists are indexed by tick (length == cfg["world"]["ticks"]).
    """

    cfg: dict
    truth_per_tick: List[set]              # ground-truth discrepant locs each tick
    flags_per_tick: List[List[Flag]]       # IoT flags raised each tick
    truth_at_tick: List[Dict[str, bool]]   # loc -> truly-discrepant lookup (this tick)
    onset: Dict[str, int]                  # loc -> first tick its (latest) episode began
    n_locations: int                       # inspectable locations (bins + assets)
    final_world: WorldState


def run_simulation(cfg: dict, seed: int, family: str = "sequential") -> SimRun:
    """Build one world and run sensors + detectors for the full horizon.

    ``family`` selects the detector family:
      * ``"sequential"`` (default): the CUSUM / SPRT / Gamma-survival detectors;
      * ``"heuristic"``           : the fixed-threshold baselines, for A/B
                                      comparison on identical seeds.

    Returns a :class:`SimRun` carrying the ground-truth labels and the raised
    flags. This is the single source of truth for both the detector-quality
    metrics and the exported handoff JSONL.
    """
    world = WorldState.build(cfg, np.random.default_rng(seed))
    sensors = {
        "weight": WeightSensor(world),
        "location": LocationSensor(world),
        "dwell": DwellSensor(world),
    }
    if family == "heuristic":
        from .detectors.heuristic import (HeuristicDwellDetector,
                                          HeuristicLocationDetector,
                                          HeuristicWeightDetector)
        detectors = {
            "weight": HeuristicWeightDetector(world),
            "location": HeuristicLocationDetector(world),
            "dwell": HeuristicDwellDetector(world),
        }
    else:
        detectors = {
            "weight": WeightDetector(world),
            "location": LocationDetector(world),
            "dwell": DwellDetector(world),
        }

    truth_per_tick: List[set] = []
    flags_per_tick: List[List[Flag]] = []
    truth_at_tick: List[Dict[str, bool]] = []
    onset: Dict[str, int] = {}

    for tick in range(cfg["world"]["ticks"]):
        # Sense true state -> detectors -> flags (the handoff payload).
        tick_flags: List[Flag] = []
        for lane in ("weight", "location", "dwell"):
            reads = sensors[lane].read()
            tick_flags.extend(detectors[lane].update(reads))

        truth = _all_truth(world)
        truth_per_tick.append(truth)
        flags_per_tick.append(tick_flags)

        # Track onset: first tick each discrepancy episode began. Clear when an
        # episode physically resolves so re-drift starts a fresh onset (this is
        # what makes time-to-flag measure the *current* episode).
        for loc in truth:
            onset.setdefault(loc, tick)
        for loc in list(onset):
            if loc not in truth:
                del onset[loc]

        # Per-location truth lookups for the flags raised this tick (so metrics
        # can label flags as true/false alarms) plus every truly-discrepant loc.
        lookups = {loc: True for loc in truth}
        for f in tick_flags:
            lookups[f.location] = location_is_truly_discrepant(world, f.location)
        truth_at_tick.append(lookups)

        EventGenerator(world).step()

    n_loc = len(world.bins) + len(world.assets)
    return SimRun(cfg=cfg, truth_per_tick=truth_per_tick,
                  flags_per_tick=flags_per_tick, truth_at_tick=truth_at_tick,
                  onset=onset, n_locations=n_loc, final_world=world)


def collect_flags(run: SimRun) -> List[Flag]:
    """Flatten a run's per-tick flags into a single ordered list (for JSONL export)."""
    return [f for tick_flags in run.flags_per_tick for f in tick_flags]
