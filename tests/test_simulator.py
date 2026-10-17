import unittest

from campaign_sim.simulator import CampaignSimulator, CreativeSpec, to_csv


class TestSimulator(unittest.TestCase):
    def setUp(self):
        self.specs = [CreativeSpec("A", 0.02, 0.10, daily_budget=300),
                      CreativeSpec("B", 0.02, 0.12, daily_budget=300)]

    def test_row_count_and_schema(self):
        rows = CampaignSimulator(self.specs, seed=1).run(10)
        self.assertEqual(len(rows), 20)
        for r in rows:
            self.assertGreaterEqual(r.impressions, r.clicks)
            self.assertGreaterEqual(r.clicks, r.conversions)

    def test_seed_reproducible(self):
        a = CampaignSimulator(self.specs, seed=42).run(5)
        b = CampaignSimulator(self.specs, seed=42).run(5)
        self.assertEqual([(r.impressions, r.clicks, r.conversions) for r in a],
                         [(r.impressions, r.clicks, r.conversions) for r in b])

    def test_higher_true_cvr_wins_over_long_run(self):
        rows = CampaignSimulator(self.specs, seed=3).run(60)
        conv = {"A": 0, "B": 0}; clk = {"A": 0, "B": 0}
        for r in rows:
            conv[r.creative] += r.conversions; clk[r.creative] += r.clicks
        self.assertGreater(conv["B"] / clk["B"], conv["A"] / clk["A"])

    def test_fatigue_lowers_ctr(self):
        spec = [CreativeSpec("A", 0.02, 0.10, fatigue_rate=0.10)]
        rows = CampaignSimulator(spec, seed=5).run(20)
        early = sum(r.clicks for r in rows if r.day < 5) / sum(r.impressions for r in rows if r.day < 5)
        late = sum(r.clicks for r in rows if r.day >= 15) / sum(r.impressions for r in rows if r.day >= 15)
        self.assertLess(late, early)

    def test_duplicate_names_rejected(self):
        with self.assertRaises(ValueError):
            CampaignSimulator([CreativeSpec("A", 0.02, 0.1),
                               CreativeSpec("A", 0.02, 0.1)])

    def test_csv_roundtrip_shape(self):
        rows = CampaignSimulator(self.specs, seed=1).run(3)
        csv = to_csv(rows)
        lines = csv.strip().split("\n")
        self.assertEqual(len(lines), 1 + 6)
        self.assertEqual(lines[0], "day,creative,impressions,clicks,conversions,spend")


if __name__ == "__main__":
    unittest.main()
