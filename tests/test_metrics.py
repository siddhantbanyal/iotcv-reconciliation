"""Tests for detector-quality metrics on hand-built cases + IoT-side invariants."""

import unittest

import numpy as np

from conftest import load_default_cfg

from iotcv.handoff import Flag
from iotcv.metrics import (_f1, aggregate, project_post_cv, score_run,
                           score_run_by_lane)
from iotcv.runner import run_detector_eval, summarize
from iotcv.simulate import SimRun


def make_run(truth_per_tick, flags_per_tick):
    """Build a minimal SimRun from per-tick truth sets + flag lists.

    truth_at_tick is derived so a flagged location is 'truly discrepant now'
    iff it is in that tick's truth set (matching real-run semantics).
    """
    truth_at_tick = []
    for truth, flags in zip(truth_per_tick, flags_per_tick):
        lut = {loc: True for loc in truth}
        for f in flags:
            lut[f.location] = f.location in truth
        truth_at_tick.append(lut)
    onset = {}
    for t, truth in enumerate(truth_per_tick):
        for loc in truth:
            onset.setdefault(loc, t)
    return SimRun(cfg=load_default_cfg(), truth_per_tick=truth_per_tick,
                  flags_per_tick=flags_per_tick, truth_at_tick=truth_at_tick,
                  onset=onset, n_locations=10, final_world=None)


def wflag(tick, bin_id, atype="phantom_inventory"):
    return Flag(flag_id=f"weight-{tick}-{bin_id}", tick=tick, lane="weight",
                anomaly_type=atype, location=f"bin:{bin_id}", zone=0,
                reason="test", confidence=0.9)


class TestMetricsUnit(unittest.TestCase):
    def test_f1_edges(self):
        self.assertEqual(_f1(0, 0), 0.0)
        self.assertAlmostEqual(_f1(1.0, 1.0), 1.0)
        self.assertAlmostEqual(_f1(0.5, 1.0), 2 * 0.5 / 1.5)

    def test_perfect_detector(self):
        # bin:0 discrepant ticks 1..4; flagged (true positive) at tick 1.
        truth = [set(), {"bin:0"}, {"bin:0"}, {"bin:0"}, {"bin:0"}, set()]
        flags = [[], [wflag(1, 0)], [], [], [], []]
        run = make_run(truth, flags)
        m = score_run(run)
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual(m["precision"], 1.0)
        self.assertEqual(m["time_to_flag"], 0.0)   # onset tick1, flag tick1

    def test_false_flag_lowers_precision(self):
        truth = [set(), {"bin:0"}, {"bin:0"}]
        flags = [[], [wflag(1, 0), wflag(1, 9)], []]  # bin:0 real, bin:9 false
        run = make_run(truth, flags)
        m = score_run(run)
        self.assertAlmostEqual(m["precision"], 0.5)
        self.assertAlmostEqual(m["false_flag_rate"], 0.5)

    def test_time_to_flag_counts_delay(self):
        truth = [set(), {"bin:0"}, {"bin:0"}, {"bin:0"}, {"bin:0"}]
        flags = [[], [], [], [wflag(3, 0)], []]  # onset 1, flagged 3
        run = make_run(truth, flags)
        m = score_run(run)
        self.assertEqual(m["time_to_flag"], 2.0)

    def test_lane_scoping(self):
        # A weight flag must not count toward the location lane's universe.
        truth = [set(), {"bin:0", "asset:0"}, {"bin:0", "asset:0"}]
        flags = [[], [wflag(1, 0)], []]
        run = make_run(truth, flags)
        by_lane = score_run_by_lane(run)
        self.assertEqual(by_lane["weight"]["recall"], 1.0)     # bin:0 caught
        self.assertEqual(by_lane["location"]["recall"], 0.0)   # asset:0 missed
        self.assertEqual(by_lane["location"]["n_flags"], 0.0)  # no location flags

    def test_flag_load_counts(self):
        truth = [{"bin:0"}, {"bin:0"}]
        flags = [[wflag(0, 0), wflag(0, 0)], [wflag(1, 0)]]  # 3 flags over 2 ticks
        run = make_run(truth, flags)
        m = score_run(run)
        self.assertAlmostEqual(m["flags_per_tick"], 1.5)

    def test_aggregate_mean_std(self):
        agg = aggregate([{"x": 1.0}, {"x": 3.0}])
        self.assertAlmostEqual(agg["x"]["mean"], 2.0)
        self.assertAlmostEqual(agg["x"]["std"], 1.0)


class TestCVProjection(unittest.TestCase):
    """The Bayesian post-verification projection (analytic projection)."""

    def test_projection_reduces_false_alerts(self):
        # Detector precision 0.90 -> pre-FAR 0.10. A good verifier collapses it.
        proj = project_post_cv(precision=0.90, cv_accuracy=0.92,
                               cv_fp_correction=0.97)
        self.assertAlmostEqual(proj["pre_false_alert_rate"], 0.10, places=6)
        self.assertLess(proj["post_false_alert_rate"],
                        proj["pre_false_alert_rate"])
        self.assertGreater(proj["false_alert_fold_reduction"], 1.0)

    def test_projection_perfect_rejection(self):
        # f=1.0 means every false flag is rejected -> post-FAR = 0.
        proj = project_post_cv(precision=0.5, cv_accuracy=0.9,
                               cv_fp_correction=1.0)
        self.assertAlmostEqual(proj["post_false_alert_rate"], 0.0, places=6)


class TestDetectorInvariants(unittest.TestCase):
    """Core IoT-side claims, checked on a small fast config."""

    @classmethod
    def setUpClass(cls):
        cfg = load_default_cfg()
        cfg["world"]["ticks"] = 1200
        cfg["world"]["n_bins"] = 20
        df = run_detector_eval(cfg, seeds=[0, 1, 2])
        cls.s = summarize(df)

    def test_detectors_have_high_recall(self):
        # IoT detectors should catch the large majority of discrepancies.
        self.assertGreater(self.s["recall_mean"].iloc[0], 0.8)

    def test_detectors_flag_quickly(self):
        # Time-to-flag should be far below a cycle-count horizon (hundreds of ticks).
        self.assertLess(self.s["time_to_flag_mean"].iloc[0], 100.0)

    def test_precision_leaves_room_for_cv(self):
        # Raw flags carry false alarms (precision < 1): exactly what CV is for.
        self.assertLess(self.s["precision_mean"].iloc[0], 1.0)


if __name__ == "__main__":
    unittest.main()
