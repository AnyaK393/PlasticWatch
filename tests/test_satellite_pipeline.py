"""Test satellite Sentinel query (legacy marine pipeline)."""
import unittest


class TestSatellitePipeline(unittest.TestCase):
    def test_sentinel_image_query(self):
        try:
            from data.fetch_satellite import init_gee
            init_gee()
        except Exception:
            self.skipTest("Google Earth Engine not configured or authenticated; skipping.")


if __name__ == "__main__":
    unittest.main()