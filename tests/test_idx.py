import unittest
import numpy as np
from backend.index_calculator import IndexCalculator


class TestIndexCalculator(unittest.TestCase):
    def test_plastic_vs_vegetation_filtering(self):
        shape = (10, 10)
        nir = np.ones(shape) * 0.02
        red = np.ones(shape) * 0.03
        red_edge = np.ones(shape) * 0.04
        swir = np.ones(shape) * 0.01

        # Plastic Patch: HIGH FDI + LOW NDVI
        plastic_y, plastic_x = 4, 5
        nir[plastic_y, plastic_x] = 0.15
        red[plastic_y, plastic_x] = 0.12

        # Seaweed Patch: HIGH FDI + HIGH NDVI
        seaweed_y, seaweed_x = 7, 2
        nir[seaweed_y, seaweed_x] = 0.18
        red[seaweed_y, seaweed_x] = 0.02

        latitudes = np.linspace(25.0, 26.0, 10)
        longitudes = np.linspace(-85.0, -84.0, 10)

        calculator = IndexCalculator(fdi_threshold=0.02, ndvi_threshold=0.2)
        fdi_matrix = calculator.calculate_fdi(nir, red_edge, swir)
        ndvi_matrix = calculator.calculate_ndvi(nir, red)
        plastic_mask = calculator.create_plastic_mask(fdi_matrix, ndvi_matrix)
        results = calculator.extract_anomaly_coordinates(
            plastic_mask, fdi_matrix, ndvi_matrix, latitudes, longitudes
        )

        self.assertEqual(len(results), 1)
        point = results[0]
        self.assertAlmostEqual(point["lat"], latitudes[plastic_y], places=3)
        self.assertAlmostEqual(point["lon"], longitudes[plastic_x], places=3)


if __name__ == "__main__":
    unittest.main()