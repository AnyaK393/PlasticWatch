import unittest
import os


class TestGee(unittest.TestCase):
    def test_gee_initialization(self):
        has_env = bool(os.getenv("EE_SERVICE_ACCOUNT") or os.getenv("EE_PROJECT"))
        if not has_env:
            self.skipTest("Google Earth Engine credentials not configured.")
        import ee
        ee.Initialize()
        test_point = ee.Geometry.Point([-85.0, 25.0])
        s2_collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(test_point)
            .filterDate('2024-05-01', '2024-05-30')
            .sort('CLOUDY_PIXEL_PERCENTAGE')
        )
        first_image = s2_collection.first()
        info = first_image.getInfo()
        self.assertIn('properties', info)


if __name__ == "__main__":
    unittest.main()