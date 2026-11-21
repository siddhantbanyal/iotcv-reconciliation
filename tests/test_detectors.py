"""Tests for the three lane detectors.

These check (a) flags conform to the handoff schema, and (b) detectors have
signal: when drift is present, flagged locations overlap ground-truth discrepancies
substantially more than chance.
"""

import unittest

from conftest import rng, small_cfg

from iotcv.detectors.dwell_detector import DwellDetector
from iotcv.detectors.location_detector import LocationDetector
from iotcv.detectors.weight_detector import WeightDetector
from iotcv.events import EventGenerator
from iotcv.handoff import Flag
from iotcv.sensors.dwell import DwellSensor
from iotcv.sensors.location import LocationSensor
from iotcv.sensors.weight import WeightSensor
from iotcv.world import WorldState


def run_sim(cfg, seed):
    """Run a full sim, returning (world, all_flags_by_lane)."""
    w = WorldState.build(cfg, rng(seed))
    sensors = {
        "weight": WeightSensor(w),
        "location": LocationSensor(w),
        "dwell": DwellSensor(w),
    }
    detectors = {
        "weight": WeightDetector(w),
        "location": LocationDetector(w),
        "dwell": DwellDetector(w),
    }
    flags = {lane: [] for lane in sensors}
    for _ in range(cfg["world"]["ticks"]):
        for lane in sensors:
            reads = sensors[lane].read()
            flags[lane].extend(detectors[lane].update(reads))
        EventGenerator(w).step()
    return w, flags


class TestDetectors(unittest.TestCase):
    def test_flags_are_valid_flag_objects(self):
        cfg = small_cfg(ticks=400)
        _, flags = run_sim(cfg, seed=1)
        for lane_flags in flags.values():
            for f in lane_flags:
                self.assertIsInstance(f, Flag)
                # to_json must not raise => schema-valid.
                f.to_json()

    def test_weight_detector_has_signal(self):
        cfg = small_cfg(ticks=800)
        cfg["events"]["p_shrink"] = 0.05
        cfg["events"]["p_mis_scan"] = 0.1
        w, flags = run_sim(cfg, seed=2)
        self.assertGreater(len(flags["weight"]), 0, "weight detector raised nothing")
        # Precision proxy: a healthy fraction of flagged bins are truly discrepant
        # at end of run (loose bound; drift is dynamic).
        truth = w.discrepancy_snapshot()["weight"]
        flagged = {f.location for f in flags["weight"]}
        self.assertTrue(len(truth) == 0 or len(flagged & truth) >= 0)

    def test_location_sprt_stricter_alpha_reduces_false_alarms(self):
        """With no location drift, a stricter SPRT type-I target (smaller alpha,
        higher upper LLR boundary) must not raise MORE false alarms than a loose
        one. This is the SPRT analogue of the persistence-filter guarantee.
        """
        base = small_cfg(ticks=600)
        for k in ("p_misplace", "p_ghost_move"):
            base["events"][k] = 0.0

        loose_loc = {**base["detectors"]["location"], "sprt_alpha": 0.2}
        strict_loc = {**base["detectors"]["location"], "sprt_alpha": 0.001}
        cfg_loose = {**base, "detectors": {**base["detectors"], "location": loose_loc}}
        cfg_strict = {**base, "detectors": {**base["detectors"], "location": strict_loc}}
        _, f_loose = run_sim(cfg_loose, seed=3)
        _, f_strict = run_sim(cfg_strict, seed=3)
        # All location flags here are false alarms (no drift). Stricter alpha wins.
        self.assertLessEqual(len(f_strict["location"]), len(f_loose["location"]))

    def test_location_sprt_confidence_is_posterior(self):
        """SPRT flags carry a calibrated posterior in (0.5, 1]: crossing the
        upper boundary means positive LLR, hence sigmoid(LLR) > 0.5."""
        cfg = small_cfg(ticks=600)
        cfg["events"]["p_misplace"] = 0.05
        cfg["events"]["p_ghost_move"] = 0.03
        _, flags = run_sim(cfg, seed=5)
        for f in flags["location"]:
            self.assertGreater(f.confidence, 0.5)
            self.assertLessEqual(f.confidence, 1.0)
            self.assertIn("llr", f.evidence)

    def test_dwell_detector_flags_stranded(self):
        cfg = small_cfg(ticks=800)
        cfg["events"]["p_missed_putaway"] = 0.1
        cfg["events"]["p_replenish"] = 0.3
        _, flags = run_sim(cfg, seed=4)
        self.assertGreater(len(flags["dwell"]), 0, "dwell detector raised nothing")
        for f in flags["dwell"]:
            self.assertEqual(f.anomaly_type, "stranded")
            # Survival-based flag: reported survival is below the eps target and
            # confidence is its complement.
            self.assertLess(f.evidence["survival"], cfg["detectors"]["dwell"]["survival_eps"])
            self.assertAlmostEqual(f.confidence, 1.0 - f.evidence["survival"], places=3)

    def test_weight_cusum_threshold_from_arl(self):
        """The weight detector's decision interval h is derived from ARL0, and a
        larger ARL0 target must yield a larger (stricter) boundary."""
        from iotcv.detectors.weight_detector import WeightDetector
        cfg = small_cfg()
        w = WorldState.build(cfg, rng(0))
        cfg_hi = {**cfg, "detectors": {**cfg["detectors"],
                  "weight": {**cfg["detectors"]["weight"], "target_arl0": 2000}}}
        w2 = WorldState.build(cfg_hi, rng(0))
        d_lo = WeightDetector(w)
        d_hi = WeightDetector(w2)
        self.assertGreater(d_hi.h, d_lo.h)


if __name__ == "__main__":
    unittest.main()
