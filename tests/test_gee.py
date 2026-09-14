# data/test_gee.py
import ee

try:
    # 1. Initialize API Connection
    print("=== Testing Step A2: Google Earth Engine ===")
    ee.Initialize()
    print("✅ SUCCESS: Google Earth Engine initialized and authenticated successfully.")
    
    # 2. Test querying a mock region matching your HYCOM data coordinates
    # Let's target a small point inside the Gulf of Mexico area (e.g., Lat 25, Lon -85)
    test_point = ee.Geometry.Point([-85.0, 25.0])
    
    # 3. Search for a cloud-free Sentinel-2 image matching your 2024 timeline
    s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(test_point)
                     .filterDate('2024-05-01', '2024-05-30')
                     .sort('CLOUDY_PIXEL_PERCENTAGE'))
    
    # Grab the top item
    first_image = s2_collection.first()
    
    # Fetch image metadata info string
    info = first_image.getInfo()
    image_id = info['properties']['system:index']
    cloud_cover = info['properties']['CLOUDY_PIXEL_PERCENTAGE']
    
    print(f"✅ SUCCESS: Found matching satellite image tile in collection.")
    print(f"   - Satellite Image ID: {image_id}")
    print(f"   - Cloud Cover Percentage: {cloud_cover:.2f}%")

except Exception as e:
    print(f"❌ ERROR: Initialization or Query failed. Reason: {e}")
    print("Tip: Run 'earthengine authenticate' in your command prompt to re-link credentials.")