import math


class RouteOptimizer:

    EARTH_RADIUS_KM = 6371

    # ---------------------------------------------------
    # HAVERSINE DISTANCE
    # ---------------------------------------------------

    def haversine_distance(
        self,
        point1,
        point2
    ):

        lat1, lon1 = point1
        lat2, lon2 = point2

        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)

        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            math.sin(dlat / 2) ** 2
            +
            math.cos(lat1)
            * math.cos(lat2)
            * math.sin(dlon / 2) ** 2
        )

        c = 2 * math.atan2(
            math.sqrt(a),
            math.sqrt(1 - a)
        )

        return self.EARTH_RADIUS_KM * c

    # ---------------------------------------------------
    # GREEDY ROUTE OPTIMIZATION
    # ---------------------------------------------------

    def compute_cleanup_route(
        self,
        start_port,
        hotspots
    ):

        current_position = start_port

        remaining_hotspots = hotspots.copy()

        route = [start_port]

        while remaining_hotspots:

            nearest_hotspot = None
            nearest_distance = float("inf")

            for hotspot in remaining_hotspots:

                hotspot_coords = (
                    hotspot["lat"],
                    hotspot["lon"]
                )

                distance = self.haversine_distance(
                    current_position,
                    hotspot_coords
                )

                if distance < nearest_distance:

                    nearest_distance = distance
                    nearest_hotspot = hotspot

            next_position = (
                nearest_hotspot["lat"],
                nearest_hotspot["lon"]
            )

            route.append(next_position)

            current_position = next_position

            remaining_hotspots.remove(
                nearest_hotspot
            )

        # Return to starting port
        route.append(start_port)

        return route