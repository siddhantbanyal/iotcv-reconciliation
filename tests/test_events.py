"""Tests for EventGenerator: drift must arise only from unlogged events."""

import unittest

from conftest import rng, small_cfg

from iotcv.events import EventGenerator
from iotcv.world import WorldState


class TestEvents(unittest.TestCase):
    def test_logged_only_world_stays_consistent(self):
        """With all drift sources off, system record must never diverge."""
        cfg = small_cfg(ticks=300)
        for k in ("p_mis_scan", "p_shrink", "p_misplace", "p_ghost_move",
                  "p_missed_putaway"):
            cfg["events"][k] = 0.0
        w = WorldState.build(cfg, rng(0))
        gen = EventGenerator(w)
        for _ in range(cfg["world"]["ticks"]):
            gen.step()
        for b in w.bins.values():
            self.assertFalse(b.has_discrepancy,
                             f"bin {b.bin_id} drifted with drift disabled")
        self.assertEqual(len(w.discrepancy_snapshot()["location"]), 0)

    def test_drift_appears_when_enabled(self):
        """With drift on, at least some discrepancies must appear."""
        cfg = small_cfg(ticks=500)
        cfg["events"]["p_shrink"] = 0.05
        cfg["events"]["p_misplace"] = 0.05
        cfg["events"]["p_ghost_move"] = 0.05
        w = WorldState.build(cfg, rng(3))
        gen = EventGenerator(w)
        for _ in range(cfg["world"]["ticks"]):
            gen.step()
        snap = w.discrepancy_snapshot()
        total = sum(len(v) for v in snap.values())
        self.assertGreater(total, 0, "expected drift but found none")

    def test_units_never_negative(self):
        cfg = small_cfg(ticks=500)
        w = WorldState.build(cfg, rng(4))
        gen = EventGenerator(w)
        for _ in range(cfg["world"]["ticks"]):
            gen.step()
        for b in w.bins.values():
            self.assertGreaterEqual(b.true_units, 0)
            self.assertGreaterEqual(b.system_units, 0)

    def test_tick_advances(self):
        cfg = small_cfg(ticks=10)
        w = WorldState.build(cfg, rng(0))
        gen = EventGenerator(w)
        for i in range(10):
            log = gen.step()
            self.assertEqual(log.tick, i)
        self.assertEqual(w.tick, 10)


if __name__ == "__main__":
    unittest.main()
