import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, UTC

# ---------------------------------------------------
# Current UTC Date
# ---------------------------------------------------

today = datetime.now(UTC).strftime("%Y-%m-%d")

print("\nAetherSea NOAA/HYCOM Connection Test")
print(f"UTC Time: {today}")

# ---------------------------------------------------
# HYCOM OpenDAP Dataset URL
# ---------------------------------------------------

url = (
    "https://tds.hycom.org/thredds/dodsC/"
    "GLBy0.08/expt_93.0"
)

print("\nConnecting to HYCOM dataset...")
print(url)

# ---------------------------------------------------
# Open Dataset Using PYDAP
# ---------------------------------------------------

try:

    ds = xr.open_dataset(
        url,
        engine="pydap",
        decode_times=False
    )

    print("\nSUCCESS: Dataset Connected!\n")

    # ---------------------------------------------------
    # Print Dataset Structure
    # ---------------------------------------------------

    print(ds)

    # ---------------------------------------------------
    # Show Available Variables
    # ---------------------------------------------------

    print("\nAvailable Variables:")
    print(list(ds.data_vars))

    # ---------------------------------------------------
    # Extract Surface Ocean Currents
    # ---------------------------------------------------

    u_surface = ds["water_u"].isel(
        time=0,
        depth=0
    )

    v_surface = ds["water_v"].isel(
        time=0,
        depth=0
    )

    print("\nSurface Ocean Layers Extracted")

    # ---------------------------------------------------
    # Arabian Sea Regional Slice
    # ---------------------------------------------------

    # Latitude: 5°N to 30°N
    # Longitude: 60°E to 80°E

    region_u = u_surface.sel(
        lat=slice(5, 30),
        lon=slice(60, 80)
    )

    region_v = v_surface.sel(
        lat=slice(5, 30),
        lon=slice(60, 80)
    )

    print("\nArabian Sea Region Extracted Successfully")

    # ---------------------------------------------------
    # Print Regional Information
    # ---------------------------------------------------

    print("\nRegional U Shape:")
    print(region_u.shape)

    print("\nRegional V Shape:")
    print(region_v.shape)

    print("\nLatitude Bounds:")
    print(float(region_u.lat.min()), "to", float(region_u.lat.max()))

    print("\nLongitude Bounds:")
    print(float(region_u.lon.min()), "to", float(region_u.lon.max()))

    # ---------------------------------------------------
    # Convert To NumPy Arrays
    # ---------------------------------------------------

    u_array = region_u.values
    v_array = region_v.values

    print("\nConverted To NumPy Arrays")

    print("\nU Array Shape:")
    print(u_array.shape)

    print("\nV Array Shape:")
    print(v_array.shape)

    # ---------------------------------------------------
    # Compute Ocean Current Magnitude
    # ---------------------------------------------------

    magnitude = np.sqrt(u_array**2 + v_array**2)

    print("\nOcean Current Magnitude Computed")

    # ---------------------------------------------------
    # Magnitude Statistics
    # ---------------------------------------------------

    print("\nMinimum Current Speed:")
    print(np.nanmin(magnitude))

    print("\nMaximum Current Speed:")
    print(np.nanmax(magnitude))

    print("\nMean Current Speed:")
    print(np.nanmean(magnitude))

    print("\nMagnitude Matrix Shape:")
    print(magnitude.shape)

    # ---------------------------------------------------
    # Visualize Current Magnitude Heatmap
    # ---------------------------------------------------

    plt.figure(figsize=(12, 8))

    heatmap = plt.imshow(
        magnitude,
        cmap="plasma",
        origin="lower"
    )

    plt.colorbar(
        heatmap,
        label="Ocean Current Speed"
    )

    plt.title("Arabian Sea Current Magnitude")

    plt.xlabel("Longitude Grid")
    plt.ylabel("Latitude Grid")

    plt.show()

    # ---------------------------------------------------
    # Quiver Plot (Ocean Direction Vectors)
    # ---------------------------------------------------

    plt.figure(figsize=(12, 8))

    step = 15

    plt.quiver(
        u_array[::step, ::step],
        v_array[::step, ::step]
    )

    plt.title("Arabian Sea Ocean Current Directions")

    plt.xlabel("Longitude Grid")
    plt.ylabel("Latitude Grid")

    plt.show()

except Exception as e:

    print("\nERROR CONNECTING TO DATASET\n")
    print(e)