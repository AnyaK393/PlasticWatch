from data.fetch_satellite import SatelliteFetcher

print("=== Testing Enhanced Step A2 ===")

fetcher = SatelliteFetcher(
    project_name="aethersea"
)

image = fetcher.query_sentinel_image(
    lon_range=[-85, -84],
    lat_range=[25, 26],
    start_date="2024-05-01",
    end_date="2024-05-30"
)

info = image.getInfo()

print("✅ SUCCESS: Enhanced Sentinel retrieval works.")

print(
    "Image ID:",
    info["properties"]["system:index"]
)

print(
    "Cloud Cover:",
    info["properties"]["CLOUDY_PIXEL_PERCENTAGE"]
)