"""Tests for WorldState construction and ground-truth bookkeeping."""

import unittest

from conftest import rng, small_cfg

from iotcv.world import WorldState


class TestWorld(unittest.TestCase):
    def test_build_consistent_initial_state(self):
        cfg = small_cfg()
        w = WorldState.build(cfg, rng(0))
        self.assertEqual(len(w.bins), cfg["world"]["n_bins"])
        self.assertEqual(len(w.skus), cfg["world"]["n_skus"])
        # System record must start equal to physical truth => no initial drift.
        for b in w.bins.values():
            self.assertEqual(b.true_units, b.system_units)
            self.assertFalse(b.has_discrepancy)
        # No misplaced assets at t=0.
        self.assertEqual(len(w.discrepancy_snapshot()["location"]), 0)

    def test_weight_inversion_roundtrip(self):
        cfg = small_cfg()
        w = WorldState.build(cfg, rng(1))
        b = next(iter(w.bins.values()))
        weight = w.bin_true_weight(b.bin_id)
        est = w.units_from_weight(b.bin_id, weight)
        # Exact (no sensor noise here): inverted count ~= true units.
        self.assertAlmostEqual(est, b.true_units, places=4)

    def test_discrepancy_snapshot_keys(self):
        cfg = small_cfg()
        w = WorldState.build(cfg, rng(2))
        b = next(iter(w.bins.values()))
        b.true_units += 5  # introduce a discrepancy by hand
        snap = w.discrepancy_snapshot()
        self.assertIn(f"bin:{b.bin_id}", snap["weight"])

    def test_reproducible_build(self):
        cfg = small_cfg()
        w1 = WorldState.build(cfg, rng(7))
        w2 = WorldState.build(cfg, rng(7))
        u1 = [b.true_units for b in w1.bins.values()]
        u2 = [b.true_units for b in w2.bins.values()]
        self.assertEqual(u1, u2)


if __name__ == "__main__":
    unittest.main()
