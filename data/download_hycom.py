import xarray as xr

print("Downloading HYCOM dataset...")

url = (
    "https://tds.hycom.org/thredds/dodsC/"
    "GLBy0.08/expt_93.0"
)

# Open remote dataset
ds = xr.open_dataset(
    url,
    engine="pydap",
    decode_times=False
)

# Extract smaller regional slice
subset = ds.sel(
    lat=slice(5, 30),
    lon=slice(60, 80)
)

# Save locally
subset.to_netcdf(
    "data/hycom_data.nc"
)

print("✅ HYCOM dataset saved locally.")