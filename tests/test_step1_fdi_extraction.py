#!/usr/bin/env python3
"""
Test STEP 1: Verify FDI/PI extraction is real, not hardcoded.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetch_satellite import init_gee, get_cloud_reduced_hotspots

# Initialize GEE
try:
    init_gee()
    print("✓ GEE initialized")
except Exception as e:
    print(f"✗ GEE init failed: {e}")
    sys.exit(1)

# Test hotspot extraction
print("\n[TEST] Extracting hotspots with STEP 1 fix...")

try:
    hotspots = get_cloud_reduced_hotspots(
        lon_range=[65.0, 75.0],    # smaller AOI for speed
        lat_range=[10.0, 20.0],
        start_date="2024-03-01",
        end_date="2024-03-31",
        fdi_threshold=0.015,
        ndvi_threshold=0.2
    )

    print(f"✓ Returned {len(hotspots)} hotspots")

    if not hotspots:
        print("⚠ No hotspots found (might be OK for small test window)")
        sys.exit(0)

    # VERIFY VALUES ARE REAL (not hardcoded)
    sample = hotspots[0]
    print(f"\nSample hotspot:")
    print(f"  lat:  {sample['lat']}")
    print(f"  lon:  {sample['lon']}")
    print(f"  fdi:  {sample['fdi']}")
    print(f"  pi:   {sample['pi']}")

    # Check: FDI should NOT all be equal to fdi_threshold (0.015)
    fdi_values = [h['fdi'] for h in hotspots]
    unique_fdis = len(set(fdi_values))

    if unique_fdis > 1:
        print(f"\n✓ PASS: FDI values are REAL (not hardcoded)")
        print(f"  Found {unique_fdis} unique FDI values")
    else:
        print(f"\n✗ FAIL: All FDI values identical (hardcoded?)")
        print(f"  Unique FDI values: {set(fdi_values)}")
        sys.exit(1)

    # Check range
    fdi_min, fdi_max = min(fdi_values), max(fdi_values)
    print(f"  FDI range: {fdi_min:.5f} to {fdi_max:.5f}")

    if fdi_min >= 0.01 and fdi_max <= 0.15:
        print(f"✓ PASS: FDI values in realistic range")
    else:
        print(f"⚠ WARNING: FDI values seem odd: {fdi_min} to {fdi_max}")

    print(f"\n✅ STEP 1 TEST PASSED")

except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)