# tests/test_supervisor.py
import os
from agents.supervisor_agent import SupervisorAgent

print("=== Testing Step B4: Gemini Supervisor Agent ===")

# Verify API key is present before testing
if not os.environ.get("GEMINI_API_KEY"):
    print("❌ ERROR: GEMINI_API_KEY environment variable not found!")
    print("Please set it in your terminal before running this test.")
else:
    # 1. Simulate mock outputs from your DBSCAN and Haversine scripts
    mock_hotspots_count = 3
    mock_distance = 42.85
    mock_waypoints = [
        [25.0123, -85.1234],  # Starting Port
        [25.1542, -85.2341],  # Hotspot 1 (Dense patch)
        [25.3121, -85.0112],  # Hotspot 2
        [25.0123, -85.1234]   # Return to Port
    ]

    # 2. Run the Agent
    print("Sending analytical payload to Gemini API...")
    agent = SupervisorAgent()
    briefing = agent.generate_dispatch_briefing(
        hotspots_count=mock_hotspots_count,
        total_distance=mock_distance,
        waypoint_list=mock_waypoints
    )

    # 3. Print the Result
    print("\n📬 RECEIVED DISPATCH BRIEFING FROM AGENT:\n")
    print(briefing)
    print("\n==============================================")
    print("✅ SUCCESS: Step B4 completed successfully if the brief looks structured and accurate!")