import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.routing_engine import plan_cleanup_route


class TestStep2Routing(unittest.TestCase):
    def test_routing_with_coastline_and_two_opt(self):
        test_hotspots = [
            {"lat": 12.5, "lon": 65.0, "fdi": 0.08},
            {"lat": 15.0, "lon": 63.5, "fdi": 0.10},
            {"lat": 18.0, "lon": 72.0, "fdi": 0.07},
            {"lat": 22.0, "lon": 68.0, "fdi": 0.09},
            {"lat": 10.0, "lon": 70.0, "fdi": 0.06},
        ]
        route = plan_cleanup_route(
            hotspots=test_hotspots,
            ship_speed=12.0,
            use_two_opt=True
        )
        self.assertIsNotNone(route)
        self.assertEqual(len(route["waypoints"]), 5)
        self.assertEqual(len(route["segments"]), 4)
        self.assertGreater(route["total_dist_km"], 0)
        self.assertGreater(route["total_cost"], 0)


if __name__ == "__main__":
    unittest.main()