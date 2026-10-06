"""Test dataset ingestion (legacy marine NetCDF)."""
import unittest
from pathlib import Path


class TestDataset(unittest.TestCase):
    def test_hycom_dataset(self):
        nc_path = Path("data/hycom_data.nc")
        if not nc_path.is_file():
            self.skipTest("HYCOM dataset file data/hycom_data.nc not found (marine legacy).")
        import xarray as xr
        ds = xr.open_dataset(str(nc_path), engine="netcdf4")
        self.assertIsNotNone(ds)


if __name__ == "__main__":
    unittest.main()