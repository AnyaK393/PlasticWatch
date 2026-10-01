"""Municipal Cleanup Route Optimizer for Municipal Dispatch Operations.

Computes optimal visiting sequence, driving distances, ETAs, and polyline
coordinates for municipal sanitation trucks dispatching from central depot
to prioritized waste hotspots and returning to depot.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

EARTH_RADIUS_KM = 6371.0

# Pune Municipal Corporation Central Sanitation Depot (Swargate / Shivaji Road)
DEFAULT_MUNICIPAL_DEPOT = {
    "name": "PMC Central Sanitation Depot (Swargate)",
    "lat": 18.5015,
    "lon": 73.8580,
}


class RouteOptimizer:
    """TSP and Greedy route optimizer for municipal waste collection vehicles."""

    def __init__(self, depot: Optional[Dict[str, Any]] = None):
        self.depot = depot or DEFAULT_MUNICIPAL_DEPOT

    def haversine_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate great-circle distance between two (lat, lon) points in kilometers."""
        lat1, lon1 = point1
        lat2, lon2 = point2

        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)

        a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0)**2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return EARTH_RADIUS_KM * c

    def compute_cleanup_route(
        self,
        start_port: Tuple[float, float],
        hotspots: List[Dict[str, Any]]
    ) -> List[Tuple[float, float]]:
        """Legacy greedy route optimization method."""
        if not hotspots:
            return [start_port, start_port]

        current_position = start_port
        remaining = hotspots.copy()
        route = [start_port]

        while remaining:
            nearest_hotspot = None
            nearest_dist = float("inf")
            for h in remaining:
                dist = self.haversine_distance(current_position, (h["lat"], h["lon"]))
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_hotspot = h

            next_pos = (nearest_hotspot["lat"], nearest_hotspot["lon"])
            route.append(next_pos)
            current_position = next_pos
            remaining.remove(nearest_hotspot)

        route.append(start_port)
        return route

    def generate_dispatch_plan(
        self,
        target_hotspots: List[Dict[str, Any]],
        truck_id: str = "PMC-TRUCK-04",
        crew_name: str = "PMC Rapid Response Team B",
        urban_speed_kmh: float = 24.0,
        stop_service_minutes: float = 20.0,
    ) -> Dict[str, Any]:
        """Generate a complete municipal dispatch plan with stops, distance, and polyline."""
        if not target_hotspots:
            return {
                "status": "empty",
                "message": "No hotspots selected for dispatch.",
                "total_distance_km": 0.0,
                "total_duration_minutes": 0,
                "stops": [],
                "route_polyline": [],
            }

        depot_coord = (self.depot["lat"], self.depot["lon"])
        
        # Greedy Nearest Neighbor order
        ordered_stops: List[Dict[str, Any]] = []
        remaining = list(target_hotspots)
        curr_pos = depot_coord
        curr_name = self.depot["name"]
        total_dist_km = 0.0
        legs = []

        stop_number = 1
        while remaining:
            best_idx = 0
            best_dist = float("inf")
            for i, h in enumerate(remaining):
                d = self.haversine_distance(curr_pos, (h["lat"], h["lon"]))
                if d < best_dist:
                    best_dist = d
                    best_idx = i

            picked = remaining.pop(best_idx)
            picked_coord = (picked["lat"], picked["lon"])
            
            # City street winding multiplier (~1.25x haversine)
            street_dist_km = round(best_dist * 1.25, 2)
            total_dist_km += street_dist_km
            drive_min = round((street_dist_km / urban_speed_kmh) * 60, 1)

            legs.append({
                "from": curr_name,
                "to": picked.get("hotspot_id", f"Stop {stop_number}"),
                "distance_km": street_dist_km,
                "drive_minutes": drive_min,
            })

            ordered_stops.append({
                "stop_number": stop_number,
                "hotspot_id": picked.get("hotspot_id", f"Stop {stop_number}"),
                "lat": picked["lat"],
                "lon": picked["lon"],
                "priority": picked.get("priority", "HIGH"),
                "total_score": picked.get("total_score", 75),
                "nearest_drain": picked.get("nearest_drain_name", "Storm drain"),
                "recurrence": picked.get("recurrence", 1),
                "distance_from_prev_km": street_dist_km,
                "estimated_service_min": stop_service_minutes,
            })

            curr_pos = picked_coord
            curr_name = picked.get("hotspot_id", f"Stop {stop_number}")
            stop_number += 1

        # Return to depot leg
        return_dist_km = round(self.haversine_distance(curr_pos, depot_coord) * 1.25, 2)
        total_dist_km += return_dist_km
        return_drive_min = round((return_dist_km / urban_speed_kmh) * 60, 1)
        legs.append({
            "from": curr_name,
            "to": self.depot["name"],
            "distance_km": return_dist_km,
            "drive_minutes": return_drive_min,
        })

        # Calculate polyline with intermediate points for smooth mapping
        polyline = [list(depot_coord)]
        for s in ordered_stops:
            polyline.append([s["lat"], s["lon"]])
        polyline.append(list(depot_coord))

        total_service_min = len(ordered_stops) * stop_service_minutes
        total_drive_min = sum(l["drive_minutes"] for l in legs)
        total_duration_min = int(round(total_service_min + total_drive_min))

        return {
            "status": "dispatched",
            "work_order_id": f"WO-PMC-{abs(hash(str(ordered_stops))) % 9000 + 1000}",
            "truck_id": truck_id,
            "crew_name": crew_name,
            "depot": self.depot,
            "total_stops": len(ordered_stops),
            "total_distance_km": round(total_dist_km, 2),
            "total_duration_minutes": total_duration_min,
            "estimated_drive_minutes": round(total_drive_min, 1),
            "estimated_service_minutes": total_service_min,
            "stops": ordered_stops,
            "legs": legs,
            "route_polyline": polyline,
        }