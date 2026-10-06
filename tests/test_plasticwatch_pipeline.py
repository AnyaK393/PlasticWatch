import unittest
from backend.plasticwatch_pipeline import run_plasticwatch


class PlasticWatchTests(unittest.TestCase):
    def test_replay_merges_reports_and_returns_explainable_scores(self):
        result = run_plasticwatch()
        self.assertEqual(result["challenge_id"], "PS-08")
        self.assertGreater(len(result["hotspots"]), 0)
        self.assertTrue(all(item["score"] > 0 for item in result["hotspots"]))
