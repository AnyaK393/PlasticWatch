# backend/index_calculator.py

import numpy as np


class IndexCalculator:

    def __init__(
        self,
        fdi_threshold=0.015,
        ndvi_threshold=0.2
    ):
        """
        fdi_threshold:
            Minimum floating debris confidence.

        ndvi_threshold:
            Anything ABOVE this is likely vegetation.
        """

        self.fdi_threshold = fdi_threshold
        self.ndvi_threshold = ndvi_threshold

    # ---------------------------------------------------
    # Floating Debris Index (FDI)
    # ---------------------------------------------------

    def calculate_fdi(
        self,
        nir_band,
        red_edge_band,
        swir_band
    ):

        lambda_nir = 832.8
        lambda_red_edge = 704.1
        lambda_swir = 1613.7

        fraction = (
            (lambda_nir - lambda_red_edge)
            /
            (lambda_swir - lambda_red_edge)
        )

        baseline = (
            red_edge_band
            +
            (swir_band - red_edge_band) * fraction
        )

        fdi_matrix = nir_band - baseline

        return fdi_matrix

    # ---------------------------------------------------
    # Normalized Difference Vegetation Index (NDVI)
    # ---------------------------------------------------

    def calculate_ndvi(
        self,
        nir_band,
        red_band
    ):
        """
        High NDVI = vegetation
        Low NDVI = non-vegetation
        """

        denominator = nir_band + red_band

        # Prevent divide-by-zero errors
        denominator = np.where(denominator == 0, 1e-10, denominator)

        ndvi_matrix = (
            (nir_band - red_band)
            /
            denominator
        )

        return ndvi_matrix

    # ---------------------------------------------------
    # Plastic Detection Mask
    # ---------------------------------------------------

    def create_plastic_mask(
        self,
        fdi_matrix,
        ndvi_matrix
    ):
        """
        Plastic logic:

        HIGH FDI
        +
        LOW NDVI
        =
        probable plastic
        """

        plastic_mask = (
            (fdi_matrix > self.fdi_threshold)
            &
            (ndvi_matrix < self.ndvi_threshold)
        )

        return plastic_mask

    # ---------------------------------------------------
    # Coordinate Extraction
    # ---------------------------------------------------

    def extract_anomaly_coordinates(
        self,
        plastic_mask,
        fdi_matrix,
        ndvi_matrix,
        latitudes,
        longitudes
    ):

        y_indices, x_indices = np.where(plastic_mask)

        coordinates = []

        for y, x in zip(y_indices, x_indices):

            coordinates.append({
                "lat": float(latitudes[y]),
                "lon": float(longitudes[x]),
                "fdi": float(fdi_matrix[y, x]),
                "ndvi": float(ndvi_matrix[y, x]),
                "confidence": "high"
            })

        return coordinates