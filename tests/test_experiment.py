import math
import unittest

from campaign_sim.experiment import (alpha_spent_obf, obf_bounds,
                                     sample_size_per_arm, sequential_analysis,
                                     z_test_two_proportions)


class TestZTest(unittest.TestCase):
    def test_known_value(self):
        # 120/1000 vs 100/1000: pooled .11, se = sqrt(.11*.89*2/1000).
        z, p = z_test_two_proportions(120, 1000, 100, 1000)
        self.assertAlmostEqual(z, 0.02 / math.sqrt(0.11 * 0.89 * 0.002), places=6)
        self.assertAlmostEqual(p, 0.1529, places=3)

    def test_zero_variance_returns_no_signal(self):
        z, p = z_test_two_proportions(0, 100, 0, 100)
        self.assertEqual((z, p), (0.0, 1.0))


class TestSampleSize(unittest.TestCase):
    def test_known_ballpark(self):
        # Standard two-proportion formula for p=.10, +.015, 80% power.
        n = sample_size_per_arm(0.10, 0.015)
        self.assertTrue(6000 < n < 7500, n)

    def test_smaller_mde_needs_more(self):
        self.assertGreater(sample_size_per_arm(0.10, 0.010),
                           sample_size_per_arm(0.10, 0.020))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            sample_size_per_arm(0.10, 0.95)


class TestSpendingFunction(unittest.TestCase):
    def test_spends_all_alpha_at_t1(self):
        self.assertAlmostEqual(alpha_spent_obf(1.0, 0.05), 0.05, places=10)

    def test_monotone_increasing(self):
        vals = [alpha_spent_obf(t, 0.05) for t in (0.25, 0.5, 0.75, 1.0)]
        self.assertEqual(vals, sorted(vals))


class TestOBBounds(unittest.TestCase):
    def test_single_look_is_fixed_horizon(self):
        self.assertAlmostEqual(obf_bounds([1.0])[0], 1.96, places=2)

    def test_bounds_decrease_with_information(self):
        bs = obf_bounds([0.25, 0.5, 0.75, 1.0])
        self.assertEqual(bs, sorted(bs, reverse=True))
        self.assertGreater(bs[0], 3.0)   # very strict early
        self.assertLess(bs[-1], 2.5)     # near 1.96 by the end

    def test_tiny_information_gets_infinite_bound(self):
        bs = obf_bounds([0.01, 1.0])
        self.assertEqual(bs[0], float("inf"))

    def test_rejects_unordered_fractions(self):
        with self.assertRaises(ValueError):
            obf_bounds([0.5, 0.25])

    def test_sequential_analysis_flags_crossing(self):
        # Overwhelming effect must cross by the final look.
        looks = [(0.5, 300, 1000, 100, 1000), (1.0, 600, 2000, 200, 2000)]
        res = sequential_analysis(looks)
        self.assertTrue(res[-1].crossed)


if __name__ == "__main__":
    unittest.main()
