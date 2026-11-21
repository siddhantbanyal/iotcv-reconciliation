"""Detector base class. A detector consumes sensor readings each tick and emits
``Flag`` objects (the IoT->CV handoff payload). Detectors are deliberately
lightweight and explainable sequential tests and thresholds; no learned models.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from typing import List

from ..handoff import Flag
from ..world import WorldState


class Detector(ABC):
    """Edge anomaly detector for one sensor lane."""

    lane: str = "base"

    def __init__(self, world: WorldState):
        self.w = world
        self.dcfg = world.cfg["detectors"]
        self._ids = itertools.count()

    def _flag_id(self) -> str:
        return f"{self.lane}-{self.w.tick}-{next(self._ids)}"

    @abstractmethod
    def update(self, readings) -> List[Flag]:
        """Ingest this tick's readings; return any new flags raised."""
        raise NotImplementedError
