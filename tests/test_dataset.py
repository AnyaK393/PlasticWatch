import xarray as xr

ds = xr.open_dataset(
    "data/hycom_data.nc",
    engine="netcdf4"
)

print(ds)

print("\nVARIABLES:")
print(list(ds.variables))

print("\nCOORDS:")
print(list(ds.coords))