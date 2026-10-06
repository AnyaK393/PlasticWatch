"""Test density grid aggregation (marine legacy)."""
import unittest
import numpy as np

from backend.clustering_engine import DebrisClusterer


class TestDensity(unittest.TestCase):
    def test_density_aggregation(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:30, 40:50] = True
        clusterer = DebrisClusterer(density_threshold=20)
        grid = clusterer.create_density_grid(mask, block_size=10)
        dense = clusterer.extract_dense_regions(grid)
        self.assertEqual(grid.shape, (10, 10))
        self.assertGreater(int(np.sum(dense)), 0)


if __name__ == "__main__":
    unittest.main()