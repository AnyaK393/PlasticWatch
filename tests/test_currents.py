# tests/test_currents.py
import xarray as xr
import os

print("=== Testing Step A1: Local HYCOM Ingestion ===")

# Verify file path exists
file_path = "data/hycom_data.nc"
if not os.path.exists(file_path):
    print(f"❌ ERROR: Cannot find file at {file_path}")
else:
    ds = xr.open_dataset(file_path)
    print("✅ SUCCESS: Successfully opened local NetCDF file.")
    print("Dimensions found:", dict(ds.dims))
    
    # Check if expected velocity variables exist
    u_var = 'water_u' if 'water_u' in ds else 'u'
    v_var = 'water_v' if 'water_v' in ds else 'v'
    print(f"Target vector variables mapped: Mapping Eastward to '{u_var}', Northward to '{v_var}'")