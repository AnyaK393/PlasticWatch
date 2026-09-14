"""
routing_engine.py (STEP 2: Coastline Obstacle Avoidance)
========================================================
NEW FEATURE:
  The routing engine now checks if any proposed path segment crosses
  land (mainland, islands, peninsulas). If it does, the engine
  automatically computes a safe detour via offshore waypoints.

  Uses Natural Earth 10m land polygons (free, public domain, auto-downloaded).
  Falls back gracefully if geopandas/shapely aren't installed.

ALGORITHM:
  1. Check if path A→B intersects any land polygon
  2. If yes, find safe offshore waypoints near the midpoint
  3. Build detour: A → safe_waypoint → B
  4. Recursively check sub-legs for further collisions
  5. Return complete safe path with all intermediate points
"""

from __future__ import annotations

import math
import logging
import os
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
_DATA_DIR   = Path(__file__).parent.parent / "data"
_NOAA_FILE  = _DATA_DIR / "oscar_currents.nc"

_KNOTS_TO_MS  = 0.514444
_EARTH_R_KM   = 6371.0

# Arabian Sea AOI
_AOI_LAT = (5.0,  30.0)
_AOI_LON = (60.0, 80.0)

_NE_DIR     = _DATA_DIR / "natural_earth"
_NE_LAND    = _NE_DIR   / "ne_10m_land.shp"
_NE_URL     = ("https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip")


# ══════════════════════════════════════════════════════════════════════════════
# 1. NOAA OSCAR Current Field (unchanged from previous version)
# ══════════════════════════════════════════════════════════════════════════════

class CurrentField:
    """Bilinear-interpolated NOAA OSCAR ocean surface current field."""

    def __init__(self, nc_path: Path = _NOAA_FILE):
        self._zero = False
        try:
            import netCDF4 as nc
            ds = nc.Dataset(str(nc_path))

            def _find(keys):
                for k in ds.variables:
                    if k.lower() in [x.lower() for x in keys]:
                        return k
                raise KeyError(f"None of {keys} found")

            lat_k = _find(["lat", "latitude", "y"])
            lon_k = _find(["lon", "longitude", "x"])
            u_k   = _find(["u", "uo", "eastward_sea_water_velocity"])
            v_k   = _find(["v", "vo", "northward_sea_water_velocity"])

            self.lats = np.array(ds.variables[lat_k][:]).flatten()
            self.lons = np.array(ds.variables[lon_k][:]).flatten()

            u_raw = np.array(ds.variables[u_k][:])
            v_raw = np.array(ds.variables[v_k][:])
            while u_raw.ndim > 2:
                u_raw = u_raw[0]
                v_raw = v_raw[0]

            self.u = np.ma.filled(u_raw, 0.0)
            self.v = np.ma.filled(v_raw, 0.0)
            ds.close()
            logger.info("[Routing] OSCAR currents loaded.")

        except Exception as exc:
            logger.warning("[Routing] OSCAR load failed (%s) – zero currents.", exc)
            self.lats = np.linspace(_AOI_LAT[0], _AOI_LAT[1], 50)
            self.lons = np.linspace(_AOI_LON[0], _AOI_LON[1], 40)
            self.u    = np.zeros((50, 40))
            self.v    = np.zeros((50, 40))
            self._zero = True

    def get_uv(self, lat: float, lon: float) -> tuple[float, float]:
        """Return bilinearly interpolated (u, v) in m/s at (lat, lon)."""
        if self._zero:
            return 0.0, 0.0

        lat = float(np.clip(lat, self.lats.min(), self.lats.max()))
        lon = float(np.clip(lon, self.lons.min(), self.lons.max()))

        i = int(np.clip(np.searchsorted(self.lats, lat) - 1, 0, len(self.lats) - 2))
        j = int(np.clip(np.searchsorted(self.lons, lon) - 1, 0, len(self.lons) - 2))

        dlat = (lat - self.lats[i]) / (self.lats[i+1] - self.lats[i] + 1e-12)
        dlon = (lon - self.lons[j]) / (self.lons[j+1] - self.lons[j] + 1e-12)

        def _interp(field):
            return (field[i,   j]   * (1-dlat) * (1-dlon) +
                    field[i+1, j]   *    dlat   * (1-dlon) +
                    field[i,   j+1] * (1-dlat)  *    dlon  +
                    field[i+1, j+1] *    dlat   *    dlon)

        return float(_interp(self.u)), float(_interp(self.v))


_CF: Optional[CurrentField] = None

def _get_cf() -> CurrentField:
    global _CF
    if _CF is None:
        _CF = CurrentField()
    return _CF


# ══════════════════════════════════════════════════════════════════════════════
# 2. STEP 2 NEW: Coastline Obstacle Avoidance
# ══════════════════════════════════════════════════════════════════════════════

class CoastlineChecker:
    """
    Checks if marine segments cross land and computes safe detours.
    Uses Natural Earth 10m land polygons.
    Falls back gracefully if geopandas/shapely unavailable.
    """

    def __init__(self):
        self._available = False
        self._land_union = None
        self._prepared_land = None
        self._safe_grid  = None

        try:
            import geopandas as gpd
            from shapely.geometry import box
            from shapely.prepared import prep

            shp = self._ensure_shapefile()
            if shp is None:
                return

            # Clip to AOI + buffer
            aoi_box = box(
                _AOI_LON[0] - 2, _AOI_LAT[0] - 2,
                _AOI_LON[1] + 2, _AOI_LAT[1] + 2,
            )
            world = gpd.read_file(str(shp))
            clipped = world.clip(aoi_box)

            if clipped.empty:
                logger.info("[Routing] No land in AOI.")
                self._available = True
                return

            self._land_union = clipped.unary_union
            self._prepared_land = prep(self._land_union)
            self._available  = True
            self._safe_grid  = self._build_safe_grid()
            logger.info("[Routing] Coastline checker ready.")

        except ImportError:
            logger.warning("[Routing] geopandas/shapely not installed – coastline avoidance disabled.")
        except Exception as exc:
            logger.warning("[Routing] CoastlineChecker init failed: %s", exc)

    def _ensure_shapefile(self) -> Optional[Path]:
        """Download Natural Earth shapefile if missing."""
        if _NE_LAND.exists():
            return _NE_LAND

        _NE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("[Routing] Downloading Natural Earth 10m land polygons…")

        try:
            import urllib.request, zipfile, io
            with urllib.request.urlopen(_NE_URL, timeout=30) as resp:
                zdata = resp.read()
            with zipfile.ZipFile(io.BytesIO(zdata)) as zf:
                zf.extractall(_NE_DIR)
            if _NE_LAND.exists():
                logger.info("[Routing] Shapefile downloaded.")
                return _NE_LAND
            shps = list(_NE_DIR.glob("*.shp"))
            return shps[0] if shps else None
        except Exception as exc:
            logger.warning("[Routing] Shapefile download failed: %s", exc)
            return None

    def _build_safe_grid(self) -> list[tuple[float, float]]:
        """Pre-compute offshore safe waypoints on a 1° grid."""
        from shapely.geometry import Point

        grid = []
        prepared = self._prepared_land
        for lat in np.arange(_AOI_LAT[0] + 0.5, _AOI_LAT[1], 1.0):
            for lon in np.arange(_AOI_LON[0] + 0.5, _AOI_LON[1], 1.0):
                if prepared is None or not prepared.contains(Point(lon, lat)):
                    grid.append((float(lat), float(lon)))
        logger.debug("[Routing] Safe grid: %d points", len(grid))
        return grid

    def crosses_land(self, lat1: float, lon1: float,
                     lat2: float, lon2: float) -> bool:
        """Return True if segment intersects any land polygon."""
        if not self._available or self._land_union is None:
            return False
        from shapely.geometry import LineString
        line = LineString([(lon1, lat1), (lon2, lat2)])
        if self._prepared_land is not None:
            return bool(self._prepared_land.intersects(line))
        return bool(self._land_union.intersects(line))

    def find_detour(self, lat1: float, lon1: float,
                    lat2: float, lon2: float) -> list[tuple[float, float]]:
        """Return list of intermediate waypoints to avoid land."""
        return self._detour_recursive(lat1, lon1, lat2, lon2, depth=0)

    def _detour_recursive(self, lat1, lon1, lat2, lon2, depth) -> list[tuple[float, float]]:
        """Recursively find detour waypoints (max 3 levels deep)."""
        if depth > 3 or not self.crosses_land(lat1, lon1, lat2, lon2):
            return []

        if not self._safe_grid:
            return []

        mid_lat = (lat1 + lat2) / 2
        mid_lon = (lon1 + lon2) / 2

        def score(pt):
            plat, plon = pt
            dist_to_mid = _haversine(mid_lat, mid_lon, plat, plon)
            if self.crosses_land(lat1, lon1, plat, plon):
                return 1e9
            if self.crosses_land(plat, plon, lat2, lon2):
                return 1e9
            return dist_to_mid

        best = min(self._safe_grid, key=score, default=None)
        if best is None or score(best) >= 1e9:
            return []

        plat, plon = best
        left  = self._detour_recursive(lat1, lon1, plat, plon, depth+1)
        right = self._detour_recursive(plat, plon, lat2, lon2, depth+1)
        return left + [(plat, plon)] + right


_CC: Optional[CoastlineChecker] = None

def _get_cc() -> CoastlineChecker:
    global _CC
    if _CC is None:
        _CC = CoastlineChecker()
    return _CC


# ══════════════════════════════════════════════════════════════════════════════
# 3. Geometry Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a  = math.sin(Δφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2
    return 2 * _EARTH_R_KM * math.asin(math.sqrt(a))


def _bearing_unit(lat1: float, lon1: float,
                  lat2: float, lon2: float) -> tuple[float, float]:
    """(east, north) unit vector from point-1 → point-2."""
    dlat = lat2 - lat1
    dlon = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
    mag  = math.hypot(dlat, dlon)
    if mag < 1e-12:
        return 0.0, 0.0
    return dlon / mag, dlat / mag


# 4. Edge Cost Computation (with detour support)

_EDGE_CACHE = {}

def _edge(lat1: float, lon1: float,
          lat2: float, lon2: float,
          ship_kts: float,
          cf: CurrentField,
          cc: CoastlineChecker) -> dict:
    key = (lat1, lon1, lat2, lon2, ship_kts)
    if key in _EDGE_CACHE:
        return _EDGE_CACHE[key]
  
    # Check for land crossing and compute detour if needed (STEP 2)
    detour_pts   = cc.find_detour(lat1, lon1, lat2, lon2)
    crosses_land = cc.crosses_land(lat1, lon1, lat2, lon2)

    # Build path with detour waypoints
    path = [(lat1, lon1)] + detour_pts + [(lat2, lon2)]

    total_dist  = 0.0
    total_cost  = 0.0
    boost_accum = 0.0

    # Sum across all sub-legs
    for k in range(len(path) - 1):
        a_lat, a_lon = path[k]
        b_lat, b_lon = path[k+1]

        dist = _haversine(a_lat, a_lon, b_lat, b_lon)

        # Current at midpoint
        m_lat = (a_lat + b_lat) / 2
        m_lon = (a_lon + b_lon) / 2
        u, v  = cf.get_uv(m_lat, m_lon)

        ex, ny = _bearing_unit(a_lat, a_lon, b_lat, b_lon)
        boost_ms  = u * ex + v * ny
        boost_kts = boost_ms / _KNOTS_TO_MS

        eff_kts   = max(ship_kts + boost_kts, 0.5)
        eff_kmh   = eff_kts * 1.852
        leg_cost  = dist / eff_kmh if eff_kmh > 0 else 1e9

        total_dist  += dist
        total_cost  += leg_cost
        boost_accum += boost_kts

    avg_boost = boost_accum / (len(path) - 1) if len(path) > 1 else 0.0

    res = {
        "dist_km":       round(total_dist, 2),
        "current_boost": round(avg_boost,  3),
        "cost":          round(total_cost, 4),
        "detour_pts":    detour_pts,
        "crosses_land":  crosses_land,
    }
    _EDGE_CACHE[key] = res
    return res


def _tour_cost(tour: list[int],
               nodes: list[dict],
               ship_kts: float,
               cf: CurrentField,
               cc: CoastlineChecker) -> float:
    """Total cost of a complete tour."""
    total = 0.0
    for k in range(len(tour) - 1):
        a, b = nodes[tour[k]], nodes[tour[k+1]]
        total += _edge(a["lat"], a["lon"], b["lat"], b["lon"], ship_kts, cf, cc)["cost"]
    return total


# 5. Greedy Nearest-Neighbour

def _greedy(nodes: list[dict],
            ship_kts: float,
            cf: CurrentField,
            cc: CoastlineChecker) -> list[int]:
    """Greedy tour starting from highest-FDI hotspot."""
    n     = len(nodes)
    start = max(range(n), key=lambda i: float(nodes[i].get("fdi", 0)))
    visited = [False] * n
    tour    = [start]
    visited[start] = True

    for _ in range(n - 1):
        cur = tour[-1]
        best_cost = float("inf")
        best_next = -1

        for j in range(n):
            if visited[j]:
                continue
            a, b = nodes[cur], nodes[j]
            c = _edge(a["lat"], a["lon"], b["lat"], b["lon"], ship_kts, cf, cc)["cost"]
            priority_factor = 1.0 + (1.0 - float(nodes[j].get("fdi", 0)) / 0.15) * 0.08
            if c * priority_factor < best_cost:
                best_cost = c * priority_factor
                best_next = j

        visited[best_next] = True
        tour.append(best_next)

    return tour


# 6. STEP 2 NEW: Full 2-Opt Local Search (complete implementation)

def _two_opt(tour: list[int],
             nodes: list[dict],
             ship_kts: float,
             cf: CurrentField,
             cc: CoastlineChecker,
             max_no_improve: int = 50) -> list[int]:
    n    = len(tour)
    if n < 4:
        return tour

    best      = tour[:]
    best_cost = _tour_cost(best, nodes, ship_kts, cf, cc)

    logger.info("[2-opt] Start cost: %.2f h", best_cost)

    improved = True
    while improved:
        improved = False

        for i in range(n - 1):
            for j in range(i + 2, n):
                # Reverse sub-segment [i+1…j]
                candidate = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
                c = _tour_cost(candidate, nodes, ship_kts, cf, cc)

                if c < best_cost - 1e-6:
                    best       = candidate
                    best_cost  = c
                    improved   = True
                    break
            if improved:
                break

    logger.info("[2-opt] Final cost: %.2f h", best_cost)
    return best


# 7. Public API
def plan_cleanup_route(
    hotspots:    list[dict],
    ship_speed:  float = 12.0,
    cf:          Optional[CurrentField]    = None,
    cc:          Optional[CoastlineChecker] = None,
    use_two_opt: bool = True,
) -> dict:
    if not hotspots:
        return {"waypoints": [], "segments": [], "total_cost": 0,
                "total_dist_km": 0, "land_detours": 0}

    cf = cf or _get_cf()
    cc = cc or _get_cc()

    # Deduplicate
    seen, unique = set(), []
    for h in hotspots:
        key = (round(h["lat"], 3), round(h["lon"], 3))
        if key not in seen:
            seen.add(key)
            unique.append(h)

    if len(unique) == 1:
        return {"waypoints": unique, "segments": [], "total_cost": 0,
                "total_dist_km": 0, "land_detours": 0}

    # Build tour
    tour = _greedy(unique, ship_speed, cf, cc)
    if use_two_opt:
        tour = _two_opt(tour, unique, ship_speed, cf, cc)

    ordered = [unique[i] for i in tour]

    # Build output
    segments      = []
    total_cost    = 0.0
    total_dist_km = 0.0
    land_detours  = 0

    for k in range(len(ordered) - 1):
        a = ordered[k]
        b = ordered[k+1]
        seg = _edge(a["lat"], a["lon"], b["lat"], b["lon"], ship_speed, cf, cc)
        seg["from_stop"] = k
        seg["to_stop"]   = k + 1
        segments.append(seg)
        total_cost    += seg["cost"]
        total_dist_km += seg["dist_km"]
        if seg["detour_pts"]:
            land_detours += 1

    return {
        "waypoints":      ordered,
        "segments":       segments,
        "total_cost":     round(total_cost, 2),
        "total_dist_km":  round(total_dist_km, 2),
        "land_detours":   land_detours,
    }
