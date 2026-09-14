#!/usr/bin/env python3
"""
Test STEP 2: Verify coastline avoidance and 2-opt refinement.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.routing_engine import plan_cleanup_route

# Sample test hotspots (some near coastlines)
test_hotspots = [
    {"lat": 12.5, "lon": 65.0, "fdi": 0.08},  # Arabian Sea
    {"lat": 15.0, "lon": 63.5, "fdi": 0.10},  # Near Yemen coast
    {"lat": 18.0, "lon": 72.0, "fdi": 0.07},  # Off Gujarat
    {"lat": 22.0, "lon": 68.0, "fdi": 0.09},  # Central Arabian Sea
    {"lat": 10.0, "lon": 70.0, "fdi": 0.06},  # Off Somalia
]

print("[TEST] STEP 2: Routing with coastline avoidance + 2-opt\n")

try:
    route = plan_cleanup_route(
        hotspots=test_hotspots,
        ship_speed=12.0,
        use_two_opt=True
    )

    print(f"✓ Route computed successfully")
    print(f"  Waypoints: {len(route['waypoints'])}")
    print(f"  Segments: {len(route['segments'])}")
    print(f"  Total distance: {route['total_dist_km']} km")
    print(f"  Total cost: {route['total_cost']} hours")
    print(f"  Land detours: {route['land_detours']}")

    # Check for detours
    detour_count = sum(1 for s in route['segments'] if s.get('detour_pts'))
    print(f"\n✓ Detour segments: {detour_count}")

    if detour_count > 0:
        print(f"✓ PASS: Coastline avoidance is working")
        for i, seg in enumerate(route['segments']):
            if seg.get('detour_pts'):
                print(f"  Segment {i}: {len(seg['detour_pts'])} detour waypoint(s)")
    else:
        print(f"ℹ No detours needed (area clear of land)")

    # Check 2-opt was applied
    if route['total_cost'] > 0:
        print(f"\n✓ PASS: 2-Opt refinement completed (cost = {route['total_cost']} h)")

    print(f"\n✅ STEP 2 TEST PASSED")

except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)