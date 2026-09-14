# tests/test_idx.py

import numpy as np
from backend.index_calculator import IndexCalculator

print("=== Testing Step A3: Plastic vs Vegetation Filtering ===")

shape = (10, 10)

# ---------------------------------------------------
# Base Ocean Reflectance
# ---------------------------------------------------

nir = np.ones(shape) * 0.02
red = np.ones(shape) * 0.03
red_edge = np.ones(shape) * 0.04
swir = np.ones(shape) * 0.01

# ---------------------------------------------------
# Plastic Patch
# HIGH FDI + LOW NDVI
# ---------------------------------------------------

plastic_y, plastic_x = 4, 5

nir[plastic_y, plastic_x] = 0.15
red[plastic_y, plastic_x] = 0.12

# ---------------------------------------------------
# Seaweed Patch
# HIGH FDI + HIGH NDVI
# ---------------------------------------------------

seaweed_y, seaweed_x = 7, 2

nir[seaweed_y, seaweed_x] = 0.18
red[seaweed_y, seaweed_x] = 0.02

# ---------------------------------------------------
# Coordinates
# ---------------------------------------------------

latitudes = np.linspace(25.0, 26.0, 10)
longitudes = np.linspace(-85.0, -84.0, 10)

# ---------------------------------------------------
# Run Calculator
# ---------------------------------------------------

calculator = IndexCalculator(
    fdi_threshold=0.02,
    ndvi_threshold=0.2
)

fdi_matrix = calculator.calculate_fdi(
    nir,
    red_edge,
    swir
)

ndvi_matrix = calculator.calculate_ndvi(
    nir,
    red
)

plastic_mask = calculator.create_plastic_mask(
    fdi_matrix,
    ndvi_matrix
)

results = calculator.extract_anomaly_coordinates(
    plastic_mask,
    fdi_matrix,
    ndvi_matrix,
    latitudes,
    longitudes
)

# ---------------------------------------------------
# Validation
# ---------------------------------------------------

print(f"Detected Objects: {len(results)}")

if len(results) == 1:

    point = results[0]

    print("✅ SUCCESS")
    print("Plastic detected correctly.")
    print("Seaweed false-positive removed.")

    print(
        f"Location: "
        f"{point['lat']:.4f}, "
        f"{point['lon']:.4f}"
    )

    print(f"FDI: {point['fdi']:.4f}")
    print(f"NDVI: {point['ndvi']:.4f}")

else:

    print("❌ ERROR")
    print("Filtering logic failed.")