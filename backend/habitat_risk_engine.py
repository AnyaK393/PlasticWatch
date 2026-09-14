import numpy as np
from shapely.geometry import Point, Polygon
from typing import List, Dict, Any

class HabitatRiskEngine:
    """
    Evaluates ecological threats by projecting debris drift trajectories 
    against known vulnerable marine habitats.
    """
    def __init__(self):
        # Sample high-priority conservation zones with sensitivity multipliers (1.0 to 3.0)
        self.habitats = [
            {
                "name": "Coral Reef Sanctuary Alpha",
                "type": "Coral Reef",
                "sensitivity": 2.8,
                "center": (14.215, 80.150),
                "radius_km": 15.0
            },
            {
                "name": "Marine Protected Area (MPA) Beta",
                "type": "MPA Reserve",
                "sensitivity": 2.5,
                "center": (13.850, 80.400),
                "radius_km": 25.0
            },
            {
                "name": "Olive Ridley Nesting Beach",
                "type": "Coastal Nesting Ground",
                "sensitivity": 3.0,
                "center": (13.500, 80.080),
                "radius_km": 10.0
            }
        ]

    def haversine_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6371.0
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlam = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0)**2
        return 2 * r * np.arcsin(np.sqrt(a))

    def evaluate_cluster_risk(self, cluster: Dict[str, Any], drift_hours: int = 48) -> Dict[str, Any]:
        """
        Calculates the Habitat Threat Score (HTS):
        HTS = Debris_Density * Proximity_Factor * Ecological_Sensitivity * Drift_Convergence
        """
        c_lat = cluster.get("lat")
        c_lon = cluster.get("lon")
        density = cluster.get("fdi_intensity", 1.0)
        u_curr = cluster.get("u_current", 0.0)  # m/s eastward
        v_curr = cluster.get("v_current", 0.0)  # m/s northward

        # Predict future position after drift_hours (1 m/s ~= 0.000009 degrees/sec approx)
        delta_lat = (v_curr * 3600 * drift_hours) / 111000.0
        delta_lon = (u_curr * 3600 * drift_hours) / (111000.0 * np.cos(np.radians(c_lat)))
        projected_lat = c_lat + delta_lat
        projected_lon = c_lon + delta_lon

        highest_hts = 0.0
        threatened_habitat = "Open Ocean"
        habitat_type = "None"
        min_dist = float("inf")

        for hab in self.habitats:
            h_lat, h_lon = hab["center"]
            curr_dist = self.haversine_km(c_lat, c_lon, h_lat, h_lon)
            proj_dist = self.haversine_km(projected_lat, projected_lon, h_lat, h_lon)

            # Check if trajectory is drifting closer to habitat
            is_approaching = proj_dist < curr_dist
            drift_factor = 1.6 if is_approaching else 0.8
            proximity_factor = max(0.1, (hab["radius_km"] * 2.0) / max(1.0, proj_dist))

            hts = density * hab["sensitivity"] * proximity_factor * drift_factor
            if hts > highest_hts:
                highest_hts = hts
                threatened_habitat = hab["name"]
                habitat_type = hab["type"]
                min_dist = proj_dist

        urgency = "CRITICAL" if highest_hts > 8.0 else ("HIGH" if highest_hts > 4.0 else "MODERATE")

        return {
            "cluster_id": cluster.get("id", "CL-01"),
            "current_location": [c_lat, c_lon],
            "projected_location": [round(projected_lat, 4), round(projected_lon, 4)],
            "threatened_habitat": threatened_habitat,
            "habitat_type": habitat_type,
            "closest_distance_km": round(min_dist, 2),
            "habitat_threat_score": round(highest_hts, 2),
            "triage_priority": urgency
        }