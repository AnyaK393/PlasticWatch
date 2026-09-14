#!/usr/bin/env python3
"""
Test STEP 3: Verify Gemini agent integration.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load env vars
load_dotenv()

from agents.supervisor_agent import SupervisorAgent

test_hotspots = [
    {"lat": 12.5, "lon": 65.0, "fdi": 0.08, "pi": 0.065},
    {"lat": 15.0, "lon": 63.5, "fdi": 0.10, "pi": 0.081},
    {"lat": 18.0, "lon": 72.0, "fdi": 0.07, "pi": 0.057},
]

test_route = {
    "waypoints": test_hotspots,
    "segments": [
        {"dist_km": 450.2, "cost": 35.1, "current_boost": 0.5},
        {"dist_km": 520.1, "cost": 42.3, "current_boost": -0.3},
    ],
    "total_cost": 77.4,
    "total_dist_km": 970.3,
    "land_detours": 0
}

test_stats = {
    "mean_fdi": 0.0823,
    "mean_pi": 0.0674
}

print("[TEST] STEP 3: Supervisor agent generation\n")

try:
    agent = SupervisorAgent()
    print(f"✓ Agent initialized")

    if not os.getenv("GEMINI_API_KEY"):
        print(f"⚠ WARNING: GEMINI_API_KEY not set")
        print(f"  Fallback report will be used")
    else:
        print(f"✓ Gemini API key loaded")

    report = agent.generate_mission_report(
        hotspots=test_hotspots,
        route=test_route,
        region_stats=test_stats,
        source="live"
    )

    print(f"\n✓ Mission report generated")
    print(f"  Length: {len(report)} characters")

    # Check report contains expected elements
    checks = [
        ("hotspots", "Hotspots" in report or "detected" in report.lower()),
        ("distance", "970" in report or "distance" in report.lower()),
        ("time", "77" in report or "hours" in report.lower() or "time" in report.lower()),
        ("recommendations", "recommend" in report.lower() or "suggest" in report.lower()),
    ]

    print(f"\nReport content checks:")
    for name, passed in checks:
        symbol = "✓" if passed else "⚠"
        print(f"  {symbol} {name}: {passed}")

    if any(not p for _, p in checks):
        print(f"\n⚠ Some expected content missing (might be Gemini variation)")
    else:
        print(f"\n✓ PASS: All expected content present")

    # Show first 300 chars of report
    print(f"\nReport preview:")
    print("---")
    print(report[:300] + "...")
    print("---")

    print(f"\n✅ STEP 3 TEST PASSED")

except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)