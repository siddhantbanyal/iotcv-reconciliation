"""Tests for confidence calibration (calibration.py) and VoI triage (triage.py)."""

import unittest

import numpy as np

from conftest import _ROOT  # noqa: F401  (ensures sys.path setup)

from iotcv.calibration import (IsotonicCalibrator, PlattScaler,
                               expected_calibration_error)
from iotcv.handoff import Flag
from iotcv.triage import (expected_resolved, greedy_submodular_zones,
                          select_topk)


def _flag(fid, conf, zone, atype="phantom_inventory", lane="weight", loc="bin:0"):
    return Flag(flag_id=fid, tick=0, lane=lane, anomaly_type=atype,
                location=loc, zone=zone, reason="t", confidence=conf)


class TestCalibration(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        # True prob rises with score; labels drawn from a shifted/scaled logit so
        # raw scores are MIS-calibrated and calibration has something to fix.
        self.s = rng.uniform(0, 1, size=4000)
        true_p = 1.0 / (1.0 + np.exp(-(3.0 * self.s - 1.0)))
        self.y = (rng.uniform(0, 1, size=4000) < true_p).astype(int)

    def test_platt_improves_ece(self):
        raw_ece, _ = expected_calibration_error(self.s, self.y)
        ps = PlattScaler().fit(self.s, self.y)
        cal_ece, _ = expected_calibration_error(ps.transform(self.s), self.y)
        self.assertLess(cal_ece, raw_ece)

    def test_isotonic_improves_ece(self):
        raw_ece, _ = expected_calibration_error(self.s, self.y)
        iso = IsotonicCalibrator().fit(self.s, self.y)
        cal_ece, _ = expected_calibration_error(iso.transform(self.s), self.y)
        self.assertLess(cal_ece, raw_ece)

    def test_isotonic_is_monotone(self):
        iso = IsotonicCalibrator().fit(self.s, self.y)
        xs = np.linspace(0, 1, 50)
        ys = iso.transform(xs)
        self.assertTrue(np.all(np.diff(ys) >= -1e-9))

    def test_ece_perfect_is_zero(self):
        # Labels exactly equal to (thresholded) probs -> near-zero ECE.
        p = np.linspace(0.05, 0.95, 1000)
        rng = np.random.default_rng(1)
        y = (rng.uniform(0, 1, size=1000) < p).astype(int)
        ece, rows = expected_calibration_error(p, y, n_bins=10)
        self.assertLess(ece, 0.05)
        self.assertEqual(len(rows), 10)


class TestTriage(unittest.TestCase):
    def test_topk_picks_highest_confidence(self):
        flags = [_flag("a", 0.9, 0), _flag("b", 0.3, 1), _flag("c", 0.7, 2)]
        chosen = select_topk(flags, budget=2)
        self.assertEqual({f.flag_id for f in chosen}, {"a", "c"})

    def test_topk_maximizes_additive_value(self):
        flags = [_flag(str(i), c, i) for i, c in enumerate([0.1, 0.5, 0.9, 0.4])]
        chosen = select_topk(flags, budget=2)
        # Optimal additive value = 0.9 + 0.5.
        self.assertAlmostEqual(expected_resolved(chosen), 1.4, places=6)

    def test_submodular_prefers_zone_coverage(self):
        # Two flags in zone 0 (0.8, 0.7) and one in zone 1 (0.6). With budget 2,
        # covering two zones (0.8 + 0.6) beats doubling up in zone 0.
        flags = [_flag("a", 0.8, 0), _flag("b", 0.7, 0), _flag("c", 0.6, 1)]
        chosen, val = greedy_submodular_zones(flags, budget=2)
        zones = {f.zone for f in chosen}
        self.assertEqual(zones, {0, 1})
        # value = [1-(1-0.8)] + [1-(1-0.6)] = 0.8 + 0.6
        self.assertAlmostEqual(val, 1.4, places=6)

    def test_submodular_diminishing_returns(self):
        # Second flag in same zone adds less than its raw confidence.
        flags = [_flag("a", 0.8, 0), _flag("b", 0.7, 0)]
        _, val = greedy_submodular_zones(flags, budget=2)
        # coverage of zone 0 = 1 - (1-0.8)(1-0.7) = 0.94 < 0.8 + 0.7
        self.assertAlmostEqual(val, 0.94, places=6)


if __name__ == "__main__":
    unittest.main()
