"""Tests for the pure-numpy statistical primitives (stats.py)."""

import math
import unittest

from conftest import _ROOT  # noqa: F401  (ensures sys.path setup)

from iotcv import stats as st


class TestIncompleteGamma(unittest.TestCase):
    def test_p_a1_matches_exponential(self):
        # P(1, x) = 1 - e^{-x}
        for x in (0.5, 1.0, 2.0, 5.0):
            self.assertAlmostEqual(st.gammainc_lower(1.0, x), 1 - math.exp(-x), places=9)

    def test_p_half_matches_erf(self):
        # P(1/2, x) = erf(sqrt(x))
        for x in (0.25, 1.0, 4.0):
            self.assertAlmostEqual(st.gammainc_lower(0.5, x), math.erf(math.sqrt(x)), places=8)

    def test_upper_lower_complement(self):
        for a, x in ((2.0, 3.0), (0.7, 1.2), (5.0, 4.0)):
            self.assertAlmostEqual(
                st.gammainc_lower(a, x) + st.gammainc_upper(a, x), 1.0, places=10)

    def test_gamma_survival_monotone_decreasing(self):
        prev = 1.0
        for x in (1, 5, 10, 20, 40):
            s = st.gamma_sf(x, shape=4.0, scale=3.0)
            self.assertLessEqual(s, prev + 1e-12)
            prev = s
        self.assertAlmostEqual(st.gamma_sf(0.0, 4.0, 3.0), 1.0)


class TestNormal(unittest.TestCase):
    def test_cdf_known(self):
        self.assertAlmostEqual(st.norm_cdf(0.0), 0.5, places=9)
        self.assertAlmostEqual(st.norm_cdf(1.96), 0.975, places=3)

    def test_ppf_roundtrip(self):
        for z in (-2.0, -0.5, 0.0, 1.3, 2.5):
            self.assertAlmostEqual(st.norm_ppf(st.norm_cdf(z)), z, places=4)


class TestSequentialDesign(unittest.TestCase):
    def test_cusum_threshold_hits_target_arl(self):
        for arl0, k in ((200.0, 0.5), (500.0, 0.5), (1000.0, 0.25)):
            h = st.cusum_threshold_from_arl(arl0, k)
            self.assertAlmostEqual(st.cusum_arl(h, k, 0.0), arl0, delta=arl0 * 0.02)

    def test_cusum_arl_drops_under_drift(self):
        # In-control ARL >> ARL once a real drift (delta) is present.
        h = st.cusum_threshold_from_arl(500.0, 0.5)
        self.assertGreater(st.cusum_arl(h, 0.5, 0.0), st.cusum_arl(h, 0.5, 2.0))

    def test_sprt_bounds_symmetry(self):
        lo, hi = st.sprt_bounds(0.05, 0.05)
        self.assertAlmostEqual(hi, -lo, places=6)
        self.assertGreater(hi, 0)

    def test_bernoulli_llr_sign(self):
        # A mismatch (x=1) with p1 > p0 raises the LLR; a match lowers it.
        self.assertGreater(st.bernoulli_llr(1, 0.85, 0.18), 0)
        self.assertLess(st.bernoulli_llr(0, 0.85, 0.18), 0)


if __name__ == "__main__":
    unittest.main()
