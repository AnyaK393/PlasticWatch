import numpy as np

from backend.clustering_engine import (
    DebrisClusterer
)

print("=== Testing Step A1 Aggregation ===")

# ---------------------------------------------------
# Fake Plastic Mask
# ---------------------------------------------------

mask = np.zeros((100, 100), dtype=bool)

# Simulate dense debris region
mask[20:30, 40:50] = True

clusterer = DebrisClusterer(
    density_threshold=20
)

density_grid = clusterer.create_density_grid(
    mask,
    block_size=10
)

dense_regions = clusterer.extract_dense_regions(
    density_grid
)

print("Density Grid Shape:")
print(density_grid.shape)

print("Dense Region Count:")
print(np.sum(dense_regions))

if np.sum(dense_regions) > 0:
    print("✅ SUCCESS")
    print("Spatial aggregation working.")
else:
    print("❌ ERROR")