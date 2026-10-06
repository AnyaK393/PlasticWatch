import os
import unittest
import xarray as xr


class TestCurrents(unittest.TestCase):
    def test_hycom_ingestion(self):
        file_path = "data/hycom_data.nc"
        if not os.path.exists(file_path):
            self.skipTest(f"Local HYCOM file not found at {file_path}")
        ds = xr.open_dataset(file_path)
        self.assertIsNotNone(ds)


if __name__ == "__main__":
    unittest.main()