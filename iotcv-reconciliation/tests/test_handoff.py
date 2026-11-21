"""Tests for the IoT->CV handoff interface (Flag / CVVerdict)."""

import os
import tempfile
import unittest

from conftest import _ROOT  # noqa: F401  (ensures sys.path setup)

from iotcv.handoff import (
    Flag,
    CVVerdict,
    read_flags_jsonl,
    write_flags_jsonl,
)


class TestHandoff(unittest.TestCase):
    def _sample_flag(self) -> Flag:
        return Flag(
            flag_id="f-1", tick=42, lane="weight",
            anomaly_type="phantom_inventory", location="bin:3", zone=2,
            reason="residual 4.2 units vs system", confidence=0.81,
            evidence={"est_units": 10.0, "system_units": 14},
        )

    def test_flag_json_roundtrip(self):
        f = self._sample_flag()
        f2 = Flag.from_json(f.to_json())
        self.assertEqual(f.to_dict(), f2.to_dict())

    def test_flag_rejects_bad_lane(self):
        with self.assertRaises(ValueError):
            Flag(flag_id="x", tick=0, lane="nope",
                 anomaly_type="phantom_inventory", location="bin:0")

    def test_flag_rejects_bad_anomaly_type(self):
        with self.assertRaises(ValueError):
            Flag(flag_id="x", tick=0, lane="weight",
                 anomaly_type="made_up", location="bin:0")

    def test_flag_rejects_bad_confidence(self):
        with self.assertRaises(ValueError):
            Flag(flag_id="x", tick=0, lane="weight",
                 anomaly_type="low_stock", location="bin:0", confidence=1.5)

    def test_from_dict_ignores_unknown_keys(self):
        d = self._sample_flag().to_dict()
        d["future_field"] = "ignored"
        f = Flag.from_dict(d)  # must not raise
        self.assertEqual(f.location, "bin:3")

    def test_verdict_roundtrip(self):
        v = CVVerdict(flag_id="f-1", tick=44, verified=True, label="empty",
                      confidence=0.9)
        v2 = CVVerdict.from_json(v.to_json())
        self.assertEqual(v.to_dict(), v2.to_dict())

    def test_jsonl_file_roundtrip(self):
        flags = [self._sample_flag(),
                 Flag(flag_id="f-2", tick=43, lane="location",
                      anomaly_type="zone_mismatch", location="asset:5", zone=1,
                      confidence=0.6)]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "flags.jsonl")
            write_flags_jsonl(flags, path)
            back = read_flags_jsonl(path)
        self.assertEqual([f.to_dict() for f in flags],
                         [f.to_dict() for f in back])


if __name__ == "__main__":
    unittest.main()
