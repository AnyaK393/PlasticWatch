import unittest

from backend.mission_pipeline import run_mission


class MissionPipelineTests(unittest.TestCase):
    def test_replay_mission_is_transparent_and_repeatable(self):
        mission = run_mission(mode="replay", forecast_hours=48)

        self.assertEqual(mission["mode"], "replay")
        self.assertIn("synthetic", mission["mode_label"].lower())
        self.assertEqual(len(mission["incidents"]), 3)
        self.assertTrue(all(item["source"] == "Synthetic replay scene" for item in mission["incidents"]))
        self.assertTrue(all(len(item["drift_track"]) == 9 for item in mission["incidents"]))
        self.assertTrue(all(item["uncertainty_km"] > 0 for item in mission["incidents"]))


    def test_forecast_length_changes_drift_track(self):
        mission = run_mission(mode="replay", forecast_hours=24)

        self.assertTrue(all(len(item["drift_track"]) == 5 for item in mission["incidents"]))
