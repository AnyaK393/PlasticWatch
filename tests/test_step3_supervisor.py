"""Test STEP 3: Gemini / heuristic supervisor agent (legacy marine)."""
import unittest
from agents.supervisor_agent import SupervisorAgent


class TestStep3Supervisor(unittest.TestCase):
    def test_supervisor_generation(self):
        agent = SupervisorAgent()
        hotspots = [{"lat": 12.5, "lon": 65.0, "fdi": 0.08, "pi": 0.065}]
        route = {
            "waypoints": hotspots,
            "segments": [{"dist_km": 450.2, "cost": 35.1, "current_boost": 0.5}],
            "total_cost": 35.1,
            "total_dist_km": 450.2,
            "land_detours": 0
        }
        stats = {"mean_fdi": 0.08, "mean_pi": 0.065}
        report = agent.generate_mission_report(
            hotspots=hotspots,
            route=route,
            region_stats=stats,
            source="live"
        )
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 10)


if __name__ == "__main__":
    unittest.main()