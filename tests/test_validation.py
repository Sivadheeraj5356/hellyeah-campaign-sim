import unittest

from campaign_sim.validation import run_validation


class TestValidation(unittest.TestCase):
    def test_smoke(self):
        # Small rep counts: checks plumbing, not calibration. Calibration is
        # asserted by the full `validate` CLI run documented in the README.
        null, alt = run_validation(reps_null=200, reps_alt=200, looks=5, seed=9)
        for v in (null.naive_peeking_fpr, null.obf_fpr, alt.obf_power):
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)
        self.assertEqual(null.looks, 5)
        self.assertIsNone(null.obf_power)
        self.assertIsNone(alt.obf_fpr)

    def test_peeking_never_beats_obf_on_fpr_in_expectation(self):
        # With 500 null reps the probability of naive < OBF is negligible
        # (naive inflates ~5x at 20 looks).
        null, _ = run_validation(reps_null=500, reps_alt=50, looks=20, seed=11)
        self.assertGreater(null.naive_peeking_fpr, null.obf_fpr)


if __name__ == "__main__":
    unittest.main()


class TestCorrelatedPaths(unittest.TestCase):
    def test_cumulative_z_paths_are_correlated_across_looks(self):
        # Guards against a regression to independent-per-look draws: within one
        # simulated experiment the cumulative z at adjacent looks shares most
        # of its data, so the correlation must be strongly positive.
        import numpy as np
        from campaign_sim.validation import _simulate_matrix, _z_matrix
        rng = np.random.default_rng(0)
        xc, nc, xt, nt = _simulate_matrix(rng, 2000, 5, 300, 0.10, 0.10)
        z = _z_matrix(xc, nc, xt, nt)
        r = np.corrcoef(z[:, 2], z[:, 3])[0, 1]
        self.assertGreater(r, 0.5)


class TestSimulatorPipelineValidation(unittest.TestCase):
    def test_smoke(self):
        from campaign_sim.validation import validate_via_simulator
        res = validate_via_simulator(days=7, looks=2, reps_null=10, reps_alt=10, seed=3)
        for k in ("null_naive_fpr", "null_obf_fpr", "alt_obf_power"):
            self.assertGreaterEqual(res[k], 0.0)
            self.assertLessEqual(res[k], 1.0)
        self.assertEqual(res["reps_null"], 10)
