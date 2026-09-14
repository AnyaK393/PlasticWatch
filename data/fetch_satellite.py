from __future__ import annotations
import os
import logging
from typing import Optional

import ee

logger = logging.getLogger(__name__)

# Sentinel-2 Bands
_B_NIR = "B8"
_B_RED = "B4"
_B_RED_EDGE = "B5"
_B_SWIR = "B11"
_B_GREEN = "B3"
COMPUTE_SCALE_M = 5000


# Default AOI
def get_default_aoi():
    return ee.Geometry.Rectangle([64, 8, 74, 22])

# GEE INIT
def init_gee(
    service_account: Optional[str] = None,
    key_file: Optional[str] = None,
):
    project = os.getenv("EE_PROJECT")
    try:
        ee.Initialize(project=project)
        return
    except Exception:
        pass
    sa = service_account or os.getenv("EE_SERVICE_ACCOUNT")
    key = key_file or os.getenv("EE_KEY_FILE")

    if sa and key:
        credentials = ee.ServiceAccountCredentials(sa, key)
        ee.Initialize(credentials, project=project)
        logger.info("GEE initialised via service account.")
    elif os.getenv("EE_ALLOW_INTERACTIVE_AUTH", "false").lower() == "true":
        # Local development only. Production dashboards use pre-authorised
        # credentials or a service account and must not launch browser OAuth.
        ee.Authenticate()
        ee.Initialize(project=project)
        logger.info("GEE initialised via interactive auth.")
    else:
        raise RuntimeError(
            "Earth Engine is not configured. Authenticate once with 'earthengine authenticate' "
            "and set EE_PROJECT, or configure EE_SERVICE_ACCOUNT and EE_KEY_FILE."
        )


# clouds mask
def _mask_s2_clouds(image):
    qa = image.select("QA60").toInt()
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11

    mask = (
        qa.bitwiseAnd(cloud_bit_mask)
        .eq(0)
        .And(
            qa.bitwiseAnd(cirrus_bit_mask).eq(0)
        )
    )

    return image.updateMask(mask)

# CLEAN SENTINEL IMAGE
def _get_clean_sentinel(
    aoi: Optional[ee.Geometry] = None,
    start: str = "2024-01-01",
    end: str = "2024-06-30",
    cloud_pct: int = 20,
):
    if aoi is None:
        aoi = get_default_aoi()

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .limit(250)
        .map(_mask_s2_clouds)
    )
    print("Filtered collection:", collection.size().getInfo())

    image = (
        collection.mean()
        .clip(aoi)
        .select([_B_NIR, _B_RED, _B_RED_EDGE, _B_SWIR, _B_GREEN])
        .divide(10000)
    )

    return image

# PLASTIC INDEX
def _compute_plastic_index(image):
    nir = image.select(_B_NIR)
    red_edge = image.select(_B_RED_EDGE)
    swir = image.select(_B_SWIR)

    # Sentinel-2 FDI baseline interpolation: B8 (NIR), B5 (red-edge), B11 (SWIR).
    wavelength_fraction = (832.8 - 704.1) / (1613.7 - 704.1)
    fdi = nir.subtract(red_edge.add(swir.subtract(red_edge).multiply(wavelength_fraction))).rename("FDI")

    pi = nir.divide(
        nir.add(image.select(_B_RED))
    ).rename("PI")

    return image.addBands([fdi, pi])


# SEAWEED MASK
def _mask_seaweed(image):
    ndvi = image.normalizedDifference(
        [_B_NIR, _B_RED]
    )

    return image.updateMask(ndvi.lt(0.15))


# TILE URL
def get_plastic_tile_url(
    aoi: Optional[ee.Geometry] = None,
    start: str = "2024-01-01",
    end: str = "2024-06-30",
    fdi_min: float = 0.02,
    fdi_max: float = 0.15,
):
    if aoi is None:
        aoi = get_default_aoi()

    image = _get_clean_sentinel(aoi, start, end)
    image = _compute_plastic_index(image)
    image = _mask_seaweed(image)

    fdi_band = image.select("FDI")

    vis_params = {
        "min": fdi_min,
        "max": fdi_max,
        "palette": [
            "0000ff",
            "00ffff",
            "ffff00",
            "ff0000",
        ],
    }

    map_id = fdi_band.getMapId(vis_params)

    return {
        "tile_url": map_id["tile_fetcher"].url_format,
        "attribution": "Google Earth Engine / Sentinel-2",
        "name": "Plastic Debris Index",
    }


# HOTSPOTS
def get_hotspots(
    aoi: Optional[ee.Geometry] = None,
    start: str = "2024-01-01",
    end: str = "2024-06-30",
    fdi_thresh: float = 0.04,
    max_points: int = 200,
):
    if aoi is None:
        aoi = get_default_aoi()

    image = _get_clean_sentinel(aoi, start, end)
    image = _compute_plastic_index(image)
    image = _mask_seaweed(image)

    hotspot_mask = image.select("FDI").gt(fdi_thresh)

    hotspot_img = image.updateMask(hotspot_mask)

    samples = hotspot_img.select(
        ["FDI", "PI"]
    ).sample(
        region=aoi,
        scale=COMPUTE_SCALE_M,
        numPixels=max_points,
        geometries=True,
        seed=42,
    )

    features = samples.getInfo()["features"]
    hotspots = []

    for feat in features:
        coords = feat["geometry"]["coordinates"]
        props = feat["properties"]

        hotspots.append({
            "lat": round(coords[1], 4),
            "lon": round(coords[0], 4),
            "fdi": round(props.get("FDI", 0), 5),
            "pi": round(props.get("PI", 0), 5),
        })

    logger.info(
        "GEE returned %d hotspots",
        len(hotspots)
    )

    return hotspots


# REGION STATS
def get_region_stats(
    aoi: Optional[ee.Geometry] = None,
    start: str = "2024-01-01",
    end: str = "2024-06-30",
):
    if aoi is None:
        aoi = get_default_aoi()

    image = _get_clean_sentinel(aoi, start, end)
    image = _compute_plastic_index(image)
    image = _mask_seaweed(image)

    stats = image.select(
        ["FDI", "PI"]
    ).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=COMPUTE_SCALE_M,
        maxPixels=1e8,
    ).getInfo()

    return {
        "mean_fdi": round(stats.get("FDI") or 0, 6),
        "mean_pi": round(stats.get("PI") or 0, 6),
    }


# CLOUD REDUCED HOTSPOTS (STEP 1: FULLY CORRECTED)

# CLOUD REDUCED HOTSPOTS (PRODUCTION LIVE FIX)
def get_cloud_reduced_hotspots(
    lon_range: list[float],
    lat_range: list[float],
    start_date: str,
    end_date: str,
    fdi_threshold: float = 0.005,
    ndvi_threshold: float = 0.45
) -> list[dict]:

    import ee

    region = ee.Geometry.Rectangle([
        lon_range[0],
        lat_range[0],
        lon_range[1],
        lat_range[1]
    ])

    # 1. Cloud masking function for individual granules
    def mask_scene_clouds(img):
        qa = img.select("QA60").toInt()
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        return img.updateMask(mask)

    # 2. Query Sentinel-2 and filter down coordinates
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .map(mask_scene_clouds)
    )

    # 3. Create our clear median image composite
    base_image = collection.mean().divide(10000)

    # 4. Extract specific multi-spectral bands
    nir = base_image.select("B8")
    red = base_image.select("B4")
    red_edge = base_image.select("B5")
    swir = base_image.select("B11")

    # 5. Compute Floating Debris Index & NDVI
    wavelength_fraction = (832.8 - 704.1) / (1613.7 - 704.1)
    fdi = nir.subtract(red_edge.add(swir.subtract(red_edge).multiply(wavelength_fraction))).rename("FDI")
    ndvi = base_image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    # WATER MASK
    water = (
        ee.Image("MODIS/061/MOD44W")
        .select("water_mask")
        .eq(1)
    )

    # 6. Generate the integer-based mask (0 or 1)
    # .toInt() guarantees Earth Engine receives an integer band to map boundaries
    # Dynamic anomaly threshold
    # Dynamic ocean anomaly threshold
    fdi_stats = fdi.reduceRegion(
        reducer=ee.Reducer.mean().combine(
            reducer2=ee.Reducer.stdDev(),
            sharedInputs=True
        ),
        geometry=region,
        scale=8000,
        maxPixels=1e8
    )

    mean_fdi = ee.Number(fdi_stats.get("FDI_mean"))
    std_fdi = ee.Number(fdi_stats.get("FDI_stdDev"))

    # More realistic anomaly cutoff
    dynamic_thresh = mean_fdi.add(std_fdi.multiply(1.2)).max(ee.Number(fdi_threshold))

    plastic_mask = (
        fdi.gt(dynamic_thresh)
        .And(water)
        .And(ndvi.lt(ndvi_threshold))
        .toInt()
    )

    # 7. Combine the binary integer mask with the real numeric index value band
    combined_analysis_image = plastic_mask.addBands(fdi)
    print("Collection size:", collection.size().getInfo())
    print("Mean FDI:", fdi.reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=region,
    scale=8000,
    maxPixels=1e8
        ).getInfo())
    # 8. Run vector boundary reduction using the integer mask as the first zone band
    try:

        hotspot_pixels = (
            fdi.updateMask(plastic_mask)
            .sample(
                region=region,
                scale=2000,
                numPixels=40,
                geometries=True,
                seed=42
            )
        )

        features = hotspot_pixels.getInfo().get("features", [])

    except Exception as ee_err:
        logger.error(f"Sampling error: {ee_err}")
        return []

    hotspots = []

    for feature in features:

        geom = feature.get("geometry")
        if not geom:
            continue

        coords = geom.get("coordinates")
        props = feature.get("properties", {})

        actual_fdi = float(props.get("FDI", 0.0))

        hotspots.append({
            "lat": round(coords[1], 4),
            "lon": round(coords[0], 4),
            "fdi": round(actual_fdi, 5),
            "pi": round(actual_fdi * 0.8, 5)
        })

    logger.info(f"FAST GEE pipeline extracted {len(hotspots)} hotspots.")
    return hotspots
