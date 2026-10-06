"""Test supervisor dispatch briefing (legacy marine)."""
import os
import unittest
from agents.supervisor_agent import SupervisorAgent


class TestSupervisor(unittest.TestCase):
    def test_dispatch_briefing(self):
        agent = SupervisorAgent()
        waypoints = [
            [25.0123, -85.1234],
            [25.1542, -85.2341],
            [25.0123, -85.1234]
        ]
        briefing = agent.generate_dispatch_briefing(
            hotspots_count=2,
            total_distance=42.85,
            waypoint_list=waypoints
        )
        self.assertIsInstance(briefing, str)
        self.assertGreater(len(briefing), 10)


if __name__ == "__main__":
    unittest.main()