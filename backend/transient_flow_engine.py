"""Temporary Floating vs. Anchored Waste: downstream drift of reported debris.

During a cloudburst, floating litter in an open channel does not stay where a citizen
photographed it. By the time a crew arrives (default 45 min) it has drifted downstream
and lodged at the next choke point (culvert inlet, junction, outfall).

For each hotspot this engine:
  1. Finds the nearest drain line and whether it is an OPEN channel or a CLOSED culvert.
  2. Classifies the waste:
       TRANSIENT_SURCHARGE  open channel (<= 30 m), rain >= 5 mm/h, mostly floatable waste
       ANCHORED             otherwise (dry weather, closed culvert, roadside, sinking waste)
  3. For transient waste, takes the debris velocity from the Manning channel model in
     simulator_engine.channel_flow_velocity, projects the drift over the crew ETA along the
     channel (in flow direction) and returns the choke point where it should be intercepted.
"""
from __future__ import annotations

from collections import Counter
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shapely.geometry import LineString, Point
from shapely.ops import nearest_points, substring

from backend.drain_risk_engine import DrainRiskEngine
from backend.simulator_engine import channel_flow_velocity

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DRAIN_GEOJSON = ROOT / "data" / "urban_drain_network.geojson"

TRANSIENT = "TRANSIENT_SURCHARGE"
ANCHORED = "ANCHORED"

MAX_IN_CHANNEL_DISTANCE_M = 30.0   # litter must be in or at the channel to float away
MOBILISING_RAIN_MMH = 5.0          # below this, floating litter lodges on banks/vegetation
MIN_FLOATABLE_SHARE = 0.5
JUNCTION_TOLERANCE_M = 40.0        # lines closer than this are treated as connected
DEFAULT_CREW_ETA_MIN = 45.0

# Buoyant in stormwater: PET bottles (capped/air-filled), film and bags, foam/cups.
# Cans fill and sink; wet cartons/paper waterlog and settle.
FLOATABLE_CLASSES = {"plastic_bottle", "plastic_bag_wrapper", "other_plastic"}

_COMPASS = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}
_LAT0, _LON0 = 18.52, 73.86
_MX = 111320.0 * math.cos(math.radians(_LAT0))
_MY = 110540.0


def _to_m(lat: float, lon: float) -> Tuple[float, float]:
    return (lon - _LON0) * _MX, (lat - _LAT0) * _MY


def _to_ll(x: float, y: float) -> Tuple[float, float]:
    return y / _MY + _LAT0, x / _MX + _LON0


class TransientFlowEngine:
    """Classifies floating vs anchored waste and projects downstream interception points."""

    def __init__(self, geojson_path: Path = DRAIN_GEOJSON):
        self.channels: List[Dict[str, Any]] = []
        self._load(geojson_path)

    # ── Network ──────────────────────────────────────────────────────────────
    def _load(self, path: Path) -> None:
        if not path.is_file():
            logger.warning("Drain network not found at %s; transient analysis disabled.", path)
            return
        try:
            features = json.loads(path.read_text(encoding="utf-8")).get("features", [])
        except Exception as exc:
            logger.error("Failed to read drain network: %s", exc)
            return
        for f in features:
            coords = f.get("geometry", {}).get("coordinates", [])
            if f.get("geometry", {}).get("type") != "LineString" or len(coords) < 2:
                continue
            p = f.get("properties", {})
            pts = [_to_m(lat, lon) for lon, lat in coords]
            # Orient the line in flow direction (first -> last vertex = downstream)
            bearing = math.degrees(math.atan2(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])) % 360
            expected = _COMPASS.get(str(p.get("flow_direction", "")).upper())
            if expected is not None and abs((bearing - expected + 180) % 360 - 180) > 90:
                pts.reverse()
            form = p.get("channel_form") or ("closed" if p.get("type") == "culvert" else "open")
            self.channels.append({
                "id": p.get("id"),
                "name": p.get("name", "Drain"),
                "form": form,
                "hydraulic_class": p.get("hydraulic_class") or ("culvert" if form == "closed" else "nallah"),
                "last_desilted": p.get("last_desilted"),
                "silt_accrual_pct_per_day": p.get("silt_accrual_pct_per_day"),
                "line": LineString(pts),
            })
        for ch in self.channels:
            ch["choke_points"] = self._choke_points(ch)

    def _choke_points(self, ch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Where floating debris on this channel gets held: culvert crossings and the outfall."""
        line: LineString = ch["line"]
        points = []
        for other in self.channels:
            if other is ch or other["form"] != "closed":
                continue
            # The channel enters a culvert only where the culvert's UPSTREAM end meets it;
            # a culvert whose downstream end touches the channel is discharging into it.
            inlet = Point(other["line"].coords[0])
            if line.distance(inlet) <= JUNCTION_TOLERANCE_M:
                along = line.project(inlet)
                if along < line.length - 1.0:
                    points.append({"along_m": along, "point": line.interpolate(along), "kind": "culvert inlet",
                                   "name": f"{other['name']} inlet"})
        end = Point(line.coords[-1])

        def continues_downstream(o: Dict[str, Any]) -> bool:
            # A receiving channel carries on past the junction (it does not also end there)
            return o["line"].project(end) < o["line"].length - JUNCTION_TOLERANCE_M

        receiving = min(
            (o for o in self.channels if o is not ch and continues_downstream(o)),
            key=lambda o: o["line"].distance(end),
            default=None,
        )
        if receiving is not None and receiving["line"].distance(end) <= 2 * JUNCTION_TOLERANCE_M:
            kind = "culvert inlet" if receiving["form"] == "closed" else (
                "river confluence" if receiving["hydraulic_class"] == "river" else "channel junction")
            name = f"Outfall of {ch['name']} into {receiving['name']}"
        else:
            kind, name = "channel outfall", f"Outfall of {ch['name']}"
        points.append({"along_m": line.length, "point": end, "kind": kind, "name": name})
        return sorted(points, key=lambda c: c["along_m"])

    def nearest_channel(self, lat: float, lon: float) -> Tuple[Optional[Dict[str, Any]], float]:
        pt = Point(_to_m(lat, lon))
        if not self.channels:
            return None, float("inf")
        best_d = min(ch["line"].distance(pt) for ch in self.channels)
        tied = [ch for ch in self.channels if ch["line"].distance(pt) <= best_d + 5.0]
        # At a junction, prefer the channel that carries the debris onward (point not at its outfall)
        onward = [ch for ch in tied if ch["line"].project(pt) < ch["line"].length - 5.0]
        best = min(onward or tied, key=lambda ch: ch["line"].distance(pt))
        return best, best["line"].distance(pt)

    # ── Assessment ───────────────────────────────────────────────────────────
    @staticmethod
    def floatable_share(hotspot: Dict[str, Any]) -> Tuple[float, Dict[str, int]]:
        classes: List[str] = []
        for r in hotspot.get("reports") or []:
            classes.extend(r.get("detected_classes") or [])
        if not classes and hotspot.get("dominant_class"):
            classes = [hotspot["dominant_class"]]
        counts = Counter(classes)
        if not counts:
            return 0.5, {}  # unknown mix: neither confirms nor rules out floating
        return sum(n for c, n in counts.items() if c in FLOATABLE_CLASSES) / sum(counts.values()), dict(counts)

    def assess(
        self,
        hotspot: Dict[str, Any],
        rainfall_intensity_mmh: float,
        crew_eta_min: float = DEFAULT_CREW_ETA_MIN,
    ) -> Dict[str, Any]:
        lat, lon = float(hotspot["lat"]), float(hotspot["lon"])
        rain = max(0.0, float(rainfall_intensity_mmh or 0.0))
        eta_min = max(1.0, min(240.0, float(crew_eta_min)))
        ch, dist = self.nearest_channel(lat, lon)
        share, mix = self.floatable_share(hotspot)

        base = {
            "classification": ANCHORED,
            "channel_name": ch["name"] if ch else None,
            "channel_form": ch["form"] if ch else None,
            "hydraulic_class": ch["hydraulic_class"] if ch else None,
            "distance_to_channel_m": round(dist, 1) if ch else None,
            "rainfall_intensity_mmh": round(rain, 1),
            "crew_eta_min": eta_min,
            "floatable_share": round(share, 2),
            "waste_mix": mix,
            "flow": None,
            "intercept": None,
            "drift_path": [],
        }
        if ch is None:
            return {**base, "reason": "No drain network loaded."}

        # Flow velocity for this channel at this rainfall (also reported for anchored sites)
        if ch["form"] == "open":
            silt = hotspot.get("drain_silt_ratio")
            if silt is None:
                silt = DrainRiskEngine.drain_silt_status(ch)["silt_ratio"]
            flow = channel_flow_velocity(ch["hydraulic_class"], rain, silt)
            base["flow"] = flow
        else:
            flow = None

        if ch["form"] == "closed":
            return {**base, "reason": f"Nearest drain is the closed {ch['name']}: debris is held at its inlet grate, so it stays put for the crew."}
        if dist > MAX_IN_CHANNEL_DISTANCE_M:
            return {**base, "reason": f"Waste is {dist:.0f} m from {ch['name']} (roadside, > {MAX_IN_CHANNEL_DISTANCE_M:.0f} m): it is not riding the channel flow."}
        if share < MIN_FLOATABLE_SHARE:
            return {**base, "reason": f"Only {share:.0%} of the reported waste floats (cans and wet paper sink and settle): treat as anchored."}
        if rain < MOBILISING_RAIN_MMH:
            storm = channel_flow_velocity(ch["hydraulic_class"], 25.0, hotspot.get("drain_silt_ratio") or 0.0)
            return {
                **base,
                "reason": (
                    f"Rain {rain:.1f} mm/h is below the {MOBILISING_RAIN_MMH:.0f} mm/h mobilising threshold: floating litter "
                    f"lodges on banks. In a 25 mm/h storm it would drift ≈{storm['debris_velocity_ms'] * eta_min * 60:,.0f} m "
                    f"before a crew arrives in {eta_min:.0f} min."
                ),
                "mobilisation_risk": True,
            }

        # TRANSIENT: project drift along the channel in flow direction
        line: LineString = ch["line"]
        start = line.project(Point(_to_m(lat, lon)))
        v = flow["debris_velocity_ms"]
        drift_m = v * eta_min * 60.0
        downstream = [c for c in ch["choke_points"] if c["along_m"] > start + 5.0]
        if not downstream:
            return {**base, "reason": "Report sits at the channel outfall: debris is already at the choke point."}

        reached = [c for c in downstream if c["along_m"] - start <= drift_m]
        target = reached[0] if reached else downstream[0]
        travel_m = target["along_m"] - start
        arrival_min = travel_m / v / 60.0 if v > 0 else float("inf")
        end_along = target["along_m"] if reached else min(line.length, start + drift_m)
        path = substring(line, start, end_along)
        path_ll = [list(_to_ll(x, y)) for x, y in (path.coords if path.geom_type == "LineString" else [path.coords[0]])]
        tlat, tlon = _to_ll(target["point"].x, target["point"].y)
        intercept = {
            "name": target["name"],
            "kind": target["kind"],
            "lat": round(tlat, 6),
            "lon": round(tlon, 6),
            "distance_downstream_m": round(travel_m, 0),
            "debris_arrival_min": round(arrival_min, 1),
            "reached_before_crew": bool(reached),
        }
        if reached:
            where = (f"reaches {target['name']} ({target['kind']}, {travel_m:,.0f} m downstream) in ≈{arrival_min:.0f} min, "
                     f"before a crew can arrive at the reported spot ({eta_min:.0f} min)")
        else:
            where = (f"drifts ≈{drift_m:,.0f} m in {eta_min:.0f} min and will reach {target['name']} "
                     f"({travel_m:,.0f} m downstream) in ≈{arrival_min:.0f} min")
        reason = (
            f"Floating waste ({share:.0%} floatable) in the open {ch['name']} at {rain:.1f} mm/h: water moves at "
            f"{flow['mean_velocity_ms']:.2f} m/s and debris at ≈{v:.2f} m/s. It {where}. Sweeping the reported spot "
            f"would find nothing; intercept at the downstream screen/grate instead."
        )
        return {
            **base,
            "classification": TRANSIENT,
            "reason": reason,
            "drift_m": round(drift_m, 0),
            "intercept": intercept,
            "drift_path": path_ll,
        }
