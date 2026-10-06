"""Test STEP 1: FDI/PI extraction (legacy marine pipeline)."""
import unittest


class TestStep1FdiExtraction(unittest.TestCase):
    def test_fdi_extraction(self):
        try:
            from data.fetch_satellite import init_gee, get_cloud_reduced_hotspots
            init_gee()
        except Exception:
            self.skipTest("Google Earth Engine not configured or authenticated; skipping.")

        hotspots = get_cloud_reduced_hotspots(
            lon_range=[65.0, 75.0],
            lat_range=[10.0, 20.0],
            start_date="2024-03-01",
            end_date="2024-03-31",
            fdi_threshold=0.015,
            ndvi_threshold=0.2
        )
        self.assertIsInstance(hotspots, list)


if __name__ == "__main__":
    unittest.main()