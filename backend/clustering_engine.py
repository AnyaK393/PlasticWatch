"""DBSCAN Spatial Clustering Engine for Citizen Waste Reports.

Merges nearby duplicate citizen reports within eps meters (default 50m)
into consolidated municipal hotspots to avoid duplicate dispatch.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)

EARTH_RADIUS_METERS = 6371000.0


class DBSCANClusteringEngine:
    """Consolidates scattered citizen reports using Haversine DBSCAN."""

    def __init__(self, eps_meters: float = 50.0, min_samples: int = 2):
        """
        Args:
            eps_meters: Maximum distance in meters to consider reports part of the same hotspot (default 50m).
            min_samples: Minimum number of reports to form a multi-report cluster (default 2).
        """
        self.eps_meters = eps_meters
        self.eps_radians = eps_meters / EARTH_RADIUS_METERS
        self.min_samples = min_samples

    def haversine_distance_m(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great-circle distance between two points in meters."""
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlam = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0)**2
        return 2.0 * EARTH_RADIUS_METERS * np.arcsin(np.sqrt(a))

    def cluster_reports(self, reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group citizen reports into consolidated hotspots.
        
        Args:
            reports: List of dicts, each containing at least 'lat' and 'lon',
                     plus optional 'id', 'confidence', 'severity', 'photo_url', etc.

        Returns:
            List of hotspot dicts with merged report counts, centroids, and underlying reports.
        """
        if not reports:
            return []

        coords = np.array([[r["lat"], r["lon"]] for r in reports])
        coords_rad = np.radians(coords)

        # DBSCAN clustering with exact Haversine metric in radians
        db = DBSCAN(
            eps=self.eps_radians,
            min_samples=self.min_samples,
            metric="haversine"
        ).fit(coords_rad)

        labels = db.labels_
        unique_cluster_ids = set(labels)
        unique_cluster_ids.discard(-1)

        hotspots: List[Dict[str, Any]] = []
        hotspot_idx = 1

        # 1. Process merged clusters (labels >= 0)
        for cid in sorted(unique_cluster_ids):
            indices = np.where(labels == cid)[0]
            cluster_reports = [reports[i] for i in indices]
            cluster_coords = coords[indices]

            center_lat = float(np.mean(cluster_coords[:, 0]))
            center_lon = float(np.mean(cluster_coords[:, 1]))

            # Average detection metrics
            confs = [r.get("plastic_confidence", r.get("confidence", 0.85)) for r in cluster_reports]
            sevs = [r.get("severity", 3.0) for r in cluster_reports]
            report_ids = [r.get("id", f"R-{i}") for i, r in enumerate(cluster_reports)]

            # Compute spread radius from center
            max_dist = max(
                self.haversine_distance_m(center_lat, center_lon, r["lat"], r["lon"])
                for r in cluster_reports
            )
            buffer_radius = max(self.eps_meters, round(max_dist + 15.0, 1))

            hotspots.append({
                "hotspot_id": f"HS-{hotspot_idx:02d}",
                "cluster_dbscan_id": int(cid),
                "lat": round(center_lat, 6),
                "lon": round(center_lon, 6),
                "recurrence": len(cluster_reports),
                "is_merged": True,
                "merge_summary": f"{len(cluster_reports)} citizen reports merged via DBSCAN ({self.eps_meters:.0f}m radius)",
                "report_ids": report_ids,
                "confidence": round(float(np.mean(confs)), 3),
                "severity": round(float(np.mean(sevs)), 1),
                "buffer_radius_m": buffer_radius,
                "reports": cluster_reports,
            })
            hotspot_idx += 1

        # 2. Process isolated/single reports (label -1)
        noise_indices = np.where(labels == -1)[0]
        for idx in noise_indices:
            r = reports[idx]
            conf = r.get("plastic_confidence", r.get("confidence", 0.85))
            sev = r.get("severity", 3.0)
            rep_id = r.get("id", f"R-{idx}")

            hotspots.append({
                "hotspot_id": f"HS-{hotspot_idx:02d}",
                "cluster_dbscan_id": -1,
                "lat": round(float(r["lat"]), 6),
                "lon": round(float(r["lon"]), 6),
                "recurrence": 1,
                "is_merged": False,
                "merge_summary": "1 single citizen report (no duplicate nearby)",
                "report_ids": [rep_id],
                "confidence": round(float(conf), 3),
                "severity": round(float(sev), 1),
                "buffer_radius_m": self.eps_meters,
                "reports": [r],
            })
            hotspot_idx += 1

        return hotspots

    def cluster_coordinates(self, coordinates: List[List[float]]) -> List[Dict[str, Any]]:
        """Legacy helper for raw [lat, lon] coordinates."""
        dummy_reports = [{"id": f"P-{i}", "lat": c[0], "lon": c[1]} for i, c in enumerate(coordinates)]
        results = self.cluster_reports(dummy_reports)
        return [
            {
                "cluster_id": h["cluster_dbscan_id"],
                "lat": h["lat"],
                "lon": h["lon"],
                "points_in_cluster": h["recurrence"],
            }
            for h in results if h["cluster_dbscan_id"] != -1
        ]