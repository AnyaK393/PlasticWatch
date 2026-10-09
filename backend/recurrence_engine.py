"""Recurrence & Root-Cause Intelligence Engine for PlasticWatch Urban Intelligence.

Analyzes repeat hotspot coordinates and historical cleanup dossiers to:
  1. Calculate total recurrence count & average return interval (days)
  2. Compute recurrence persistence index (0.0 to 1.0)
  3. Identify dominant waste streams (HDPE packaging, PET bottles, thermocol, etc.)
  4. Generate automated contextual root-cause hypotheses
  5. Recommend systemic civic interventions (commercial bin mandates, bar screens, etc.)
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DOSSIER_PATH = ROOT / "data" / "historical_urban_dossier.json"
LANDUSE_NODES_PATH = ROOT / "data" / "pune_landuse_nodes.json"
RAINFALL_PATH = ROOT / "data" / "pune_rainfall_2026.json"
EARTH_RADIUS_METERS = 6371000.0

LANDUSE_CAUSES = {
    "Industrial": "Packaging film and strapping runoff from manufacturing packaging zones.",
    "Commercial Produce Market": "High-volume single-use carrier bag overflow from vegetable and trade stalls.",
    "Transit Hub": "Pedestrian beverage containers and on-the-go food packaging entering storm grates.",
    "High-Density Residential Culvert": "Household domestic plastics entering uncovered feeder nallahs.",
    "Riverine Confluence": "Accumulation point for upstream wash-off during early stormwater runoff events.",
}

# Attribution strength by distance to the nearest land-use landmark
STRONG_ATTRIBUTION_M = 750.0
MODERATE_ATTRIBUTION_M = 2000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two coordinates in meters."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_METERS * math.asin(math.sqrt(a))


_landuse_cache: Optional[List[Dict[str, Any]]] = None


def load_landuse_nodes(path: Path = LANDUSE_NODES_PATH) -> List[Dict[str, Any]]:
    global _landuse_cache
    if _landuse_cache is not None and path == LANDUSE_NODES_PATH:
        return _landuse_cache
    nodes: List[Dict[str, Any]] = []
    if path.is_file():
        try:
            nodes = json.loads(path.read_text(encoding="utf-8")).get("nodes", [])
        except Exception as e:
            logger.error("Failed to load land-use nodes: %s", e)
    if path == LANDUSE_NODES_PATH:
        _landuse_cache = nodes
    return nodes


def attribute_landuse_root_cause(lat: float, lon: float) -> Dict[str, Any]:
    """Explain a hotspot by its nearest municipal land-use landmark (Haversine)."""
    nodes = load_landuse_nodes()
    if not nodes:
        return {
            "landmark": None,
            "category": None,
            "distance_m": None,
            "attribution_strength": "UNAVAILABLE",
            "hypothesis": "Land-use landmark registry unavailable; root cause requires field survey.",
        }

    nearest = min(nodes, key=lambda n: haversine_m(lat, lon, float(n["lat"]), float(n["lon"])))
    dist_m = haversine_m(lat, lon, float(nearest["lat"]), float(nearest["lon"]))
    category = nearest.get("category", "")
    cause = LANDUSE_CAUSES.get(category, "Mixed urban land-use runoff.")

    if dist_m <= STRONG_ATTRIBUTION_M:
        strength = "STRONG"
        lead = f"Within {dist_m:,.0f} m of {nearest['name']} ({category})"
    elif dist_m <= MODERATE_ATTRIBUTION_M:
        strength = "MODERATE"
        lead = f"{dist_m:,.0f} m from {nearest['name']} ({category})"
    else:
        strength = "WEAK"
        lead = (
            f"Nearest registered landmark is {nearest['name']} ({category}), {dist_m / 1000.0:.1f} km away; "
            "local field survey recommended"
        )

    return {
        "landmark": nearest["name"],
        "landmark_id": nearest.get("node_id"),
        "category": category,
        "distance_m": round(dist_m, 1),
        "attribution_strength": strength,
        "cause": cause,
        "hypothesis": f"{lead}: {cause}",
    }


# ── Multi-evidence root-cause attribution ────────────────────────────────────
# Each candidate driver has a fingerprint. The engine scores how well a site's
# 3-month logs match each fingerprint and reports the ranked drivers with evidence.
WASTE_CLASSES = ["plastic_bottle", "plastic_bag_wrapper", "can_metal", "carton_paper", "other_plastic"]
CLASS_LABELS = {
    "plastic_bottle": "PET bottles",
    "plastic_bag_wrapper": "carry bags & film",
    "can_metal": "beverage cans",
    "carton_paper": "cartons & paper",
    "other_plastic": "mixed/other plastics",
}
HOUR_BANDS = [("night", 0, 5), ("early morning", 5, 9), ("daytime", 9, 16), ("evening", 16, 21), ("late night", 21, 24)]
STREET_FOOD = "Street Food / Kiosk Strip"

# composition: expected share per WASTE_CLASSES; hours: expected share per HOUR_BANDS;
# weekend_ratio: weekend vs weekday report intensity; rain_share: expected share of the
# accumulation rate attributable to rainfall
DRIVER_FINGERPRINTS: Dict[str, Dict[str, Any]] = {
    "Industrial": {
        "composition": [0.10, 0.45, 0.05, 0.10, 0.30], "hours": [0.05, 0.15, 0.55, 0.20, 0.05],
        "weekend_ratio": 0.5, "rain_share": 0.25,
    },
    "Commercial Produce Market": {
        "composition": [0.12, 0.56, 0.03, 0.17, 0.12], "hours": [0.10, 0.55, 0.20, 0.10, 0.05],
        "weekend_ratio": 1.3, "rain_share": 0.2,
    },
    "Transit Hub": {
        "composition": [0.42, 0.20, 0.23, 0.08, 0.07], "hours": [0.03, 0.35, 0.17, 0.40, 0.05],
        "weekend_ratio": 0.65, "rain_share": 0.2,
    },
    "High-Density Residential Culvert": {
        "composition": [0.22, 0.44, 0.07, 0.12, 0.15], "hours": [0.08, 0.45, 0.15, 0.25, 0.07],
        "weekend_ratio": 1.15, "rain_share": 0.3,
    },
    "Riverine Confluence": {
        "composition": [0.18, 0.30, 0.05, 0.10, 0.37], "hours": [0.20, 0.20, 0.20, 0.20, 0.20],
        "weekend_ratio": 1.0, "rain_share": 0.6,
    },
    STREET_FOOD: {
        "composition": [0.36, 0.16, 0.27, 0.16, 0.05], "hours": [0.03, 0.08, 0.22, 0.55, 0.12],
        "weekend_ratio": 1.35, "rain_share": 0.1,
    },
}
DRIVER_CAUSES = {**LANDUSE_CAUSES, STREET_FOOD: "Takeaway cups, bottles and cans from roadside food kiosks without segregated bins."}
DRIVER_INTERVENTIONS = {
    "Industrial": [
        "Extended Producer Responsibility audit of packaging-film users in the estate",
        "Mandate covered skips and strapping-waste collection at loading bays",
    ],
    "Commercial Produce Market": [
        "Enforce the single-use carrier bag ban with weekly market-yard drives",
        "Schedule a post-market sweep before 09:00 on peak trading days",
    ],
    "Transit Hub": [
        "Install twin-bin litter points and a reverse-vending machine at the terminal",
        "Retrofit grate-level debris screens on the commuter approach roads",
    ],
    "High-Density Residential Culvert": [
        "Cover open feeder nallahs and raise door-to-door segregated collection compliance",
        "Ward-level awareness drive with housing societies on dry-waste handover",
    ],
    "Riverine Confluence": [
        "Deploy a floating trash boom at the outfall before forecast heavy-rain days",
        "Target upstream wards: pre-monsoon desilting and inlet screening",
    ],
    STREET_FOOD: [
        "Mandate kiosk-owned segregated bins with licence-renewal checks",
        "Evening litter sweep along the kiosk strip on weekends",
    ],
}
# Evidence weights: where the site is, what it collects, when it arrives, how rain drives it
EVIDENCE_WEIGHTS = {"proximity": 0.30, "composition": 0.35, "temporal": 0.20, "rain_coupling": 0.15}
NO_REGISTRY_PROXIMITY = 0.25  # neutral prior for drivers without a mapped landmark


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


def _to_ist(ts: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None
    return dt.astimezone(timezone(timedelta(hours=5, minutes=30)))


_daily_rain_cache: Optional[Dict[str, float]] = None


def load_daily_rainfall() -> Dict[str, float]:
    """Real daily Pune rainfall (mm) keyed by ISO date; empty if the file is missing."""
    global _daily_rain_cache
    if _daily_rain_cache is None:
        try:
            raw = json.loads(RAINFALL_PATH.read_text(encoding="utf-8")).get("daily", {})
            _daily_rain_cache = {k: float(v or 0.0) for k, v in raw.items()}
        except Exception:
            _daily_rain_cache = {}
    return _daily_rain_cache


def _interval_rain_coupling(cleanups: List[Dict[str, Any]]) -> Tuple[float, float, int]:
    """Relate accumulation rate (kg/day between cleanups) to mean rain over the same interval.

    Returns (pearson r, rain-attributable share of the mean rate, intervals used). The share
    comes from the least-squares slope: slope * mean_rain / mean_rate.
    """
    daily = load_daily_rainfall()
    ordered = sorted((c for c in cleanups if _to_ist(c.get("timestamp", ""))), key=lambda c: c["timestamp"])
    rates, rains = [], []
    for prev, cur in zip(ordered, ordered[1:]):
        d0, d1 = _to_ist(prev["timestamp"]).date(), _to_ist(cur["timestamp"]).date()
        gap = (d1 - d0).days
        if gap <= 0:
            continue
        days = [(d0 + timedelta(days=k)).isoformat() for k in range(gap)]
        if daily and not all(d in daily for d in days):
            continue
        mean_rain = (
            sum(daily[d] for d in days) / gap if daily
            else float(cur.get("rainfall_prev_48h_mm", 0.0)) / 2.0
        )
        rates.append(float(cur.get("weight_removed_kg", 0.0)) / gap)
        rains.append(mean_rain)
    if len(rates) < 3:
        return 0.0, 0.0, len(rates)
    mr, mx = sum(rates) / len(rates), sum(rains) / len(rains)
    sxx = sum((x - mx) ** 2 for x in rains)
    slope = sum((x - mx) * (y - mr) for x, y in zip(rains, rates)) / sxx if sxx else 0.0
    share = max(0.0, min(1.0, slope * mx / mr)) if mr else 0.0
    return _pearson(rains, rates), share, len(rates)


def extract_site_evidence(cleanups: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise the observable fingerprint of a site from its cleanup logs."""
    weights = [float(c.get("weight_removed_kg", 0.0)) for c in cleanups]
    total_w = sum(weights) or 1.0

    # Mass-weighted waste composition
    comp = [0.0] * len(WASTE_CLASSES)
    comp_mass = 0.0
    for c, w in zip(cleanups, weights):
        mix = c.get("waste_composition_pct")
        if mix:
            for i, cls in enumerate(WASTE_CLASSES):
                comp[i] += w * float(mix.get(cls, 0.0)) / 100.0
            comp_mass += w
    composition = [x / comp_mass for x in comp] if comp_mass else []

    # When accumulations are first reported (deposit timing)
    hours = [0] * len(HOUR_BANDS)
    weekend, weekday = 0, 0
    for c in cleanups:
        dt = _to_ist(c.get("first_reported_at", ""))
        if dt is None:
            continue
        for i, (_, lo, hi) in enumerate(HOUR_BANDS):
            if lo <= dt.hour < hi:
                hours[i] += 1
        if dt.weekday() >= 5:
            weekend += 1
        else:
            weekday += 1
    n_reports = sum(hours)
    hour_share = [h / n_reports for h in hours] if n_reports else []
    # Normalise by day count: 2 weekend days vs 5 weekdays
    weekend_ratio = (weekend / 2.0) / (weekday / 5.0) if weekday else 1.0

    rain_r, rain_share, rain_pairs = _interval_rain_coupling(cleanups)

    return {
        "logs_analyzed": len(cleanups),
        "total_mass_kg": round(total_w, 1),
        "composition": composition,
        "hour_share": hour_share,
        "weekend_ratio": round(weekend_ratio, 2),
        "rain_load_correlation": round(rain_r, 2),
        "rain_share_of_load": round(rain_share, 2),
        "rain_intervals_analyzed": rain_pairs,
    }


def attribute_root_cause(lat: float, lon: float, cleanups: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Rank candidate drivers by proximity, waste fingerprint, timing and rain coupling."""
    ev = extract_site_evidence(cleanups)
    nodes = load_landuse_nodes()

    nearest_by_cat: Dict[str, Tuple[Dict[str, Any], float]] = {}
    for n in nodes:
        d = haversine_m(lat, lon, float(n["lat"]), float(n["lon"]))
        cat = n.get("category", "")
        if cat not in nearest_by_cat or d < nearest_by_cat[cat][1]:
            nearest_by_cat[cat] = (n, d)

    candidates = []
    for driver, fp in DRIVER_FINGERPRINTS.items():
        if driver in nearest_by_cat:
            node, dist = nearest_by_cat[driver]
            proximity = math.exp(-dist / 1000.0)  # 0 m -> 1.0, 1 km -> 0.37, 3 km -> 0.05
            where = f"{node['name']} {dist:,.0f} m away"
        else:
            node, dist = None, None
            proximity = NO_REGISTRY_PROXIMITY
            where = "no mapped landmark (inferred from waste and timing pattern)"

        composition = max(0.0, _cosine(ev["composition"], fp["composition"])) if ev["composition"] else 0.5
        # Sharpen cosine (most real mixes score 0.8-1.0) so differences are meaningful
        composition = max(0.0, (composition - 0.75) / 0.25)
        temporal_hours = _cosine(ev["hour_share"], fp["hours"]) if ev["hour_share"] else 0.5
        weekend_fit = math.exp(-abs(math.log(max(ev["weekend_ratio"], 0.05) / fp["weekend_ratio"])))
        temporal = 0.7 * max(0.0, (temporal_hours - 0.6) / 0.4) + 0.3 * weekend_fit
        rain_fit = max(0.0, 1.0 - abs(ev["rain_share_of_load"] - fp["rain_share"]) / 0.5)

        parts = {
            "proximity": proximity,
            "composition": min(1.0, composition),
            "temporal": min(1.0, temporal),
            "rain_coupling": rain_fit,
        }
        score = sum(EVIDENCE_WEIGHTS[k] * v for k, v in parts.items())
        candidates.append({
            "driver": driver,
            "score": score,
            "evidence_scores": {k: round(v, 2) for k, v in parts.items()},
            "landmark": node["name"] if node else None,
            "distance_m": round(dist, 0) if dist is not None else None,
            "where": where,
        })

    total = sum(c["score"] for c in candidates) or 1.0
    for c in candidates:
        c["confidence_pct"] = round(100.0 * c["score"] / total, 1)
        c["score"] = round(c["score"], 3)
    candidates.sort(key=lambda c: -c["score"])
    top, runner = candidates[0], candidates[1]
    margin = top["confidence_pct"] - runner["confidence_pct"]
    certainty = "HIGH" if margin >= 6.0 else ("MEDIUM" if margin >= 3.0 else "LOW (mixed drivers)")

    # Plain-English evidence for the winning driver
    facts = []
    if ev["composition"]:
        top_two = sorted(zip(WASTE_CLASSES, ev["composition"]), key=lambda x: -x[1])[:2]
        facts.append(
            "waste mix " + " + ".join(f"{share:.0%} {CLASS_LABELS[c]}" for c, share in top_two)
            + f" (fingerprint match {top['evidence_scores']['composition']:.2f})"
        )
    if ev["hour_share"]:
        peak_band = HOUR_BANDS[max(range(len(ev["hour_share"])), key=lambda i: ev["hour_share"][i])][0]
        peak_share = max(ev["hour_share"])
        week_note = (
            "weekend-heavy" if ev["weekend_ratio"] >= 1.15
            else ("weekday-heavy" if ev["weekend_ratio"] <= 0.85 else "even across the week")
        )
        facts.append(f"{peak_share:.0%} of accumulations first reported in the {peak_band}, {week_note} (weekend ratio {ev['weekend_ratio']:.2f})")
    share, r = ev["rain_share_of_load"], ev["rain_load_correlation"]
    rain_note = "strongly rain-driven" if share >= 0.45 else ("partly rain-driven" if share >= 0.2 else "mostly local dumping, not rain wash-in")
    facts.append(
        f"{share:.0%} of the accumulation rate is attributable to rainfall, so it is {rain_note} "
        f"(r = {r:+.2f} over {ev['rain_intervals_analyzed']} cleanup intervals, real Open-Meteo rain)"
    )

    hypothesis = (
        f"Primary driver: {top['driver']} ({top['where']}), {top['confidence_pct']:.0f}% of evidence. "
        f"{DRIVER_CAUSES.get(top['driver'], '')} Evidence: " + "; ".join(facts) + ". "
        f"Secondary: {runner['driver']} ({runner['confidence_pct']:.0f}%)."
    )

    return {
        "primary_driver": top["driver"],
        "primary_confidence_pct": top["confidence_pct"],
        "certainty": certainty,
        "hypothesis": hypothesis,
        "evidence_facts": facts,
        "site_evidence": {
            **ev,
            "composition": {c: round(v * 100, 1) for c, v in zip(WASTE_CLASSES, ev["composition"])},
            "hour_share": {b[0]: round(v * 100, 1) for b, v in zip(HOUR_BANDS, ev["hour_share"])},
        },
        "candidates": candidates,
        "recommended_interventions": DRIVER_INTERVENTIONS.get(top["driver"], []),
        "evidence_weights": EVIDENCE_WEIGHTS,
    }


class RecurrenceEngine:
    """Historical recurrence analysis and automated root-cause deduction engine."""

    def __init__(self, dossier_path: Path = DOSSIER_PATH):
        self.dossier_path = dossier_path
        self._dossier_cache: Optional[Dict[str, Any]] = None

    def _load_dossier(self) -> Dict[str, Any]:
        if self._dossier_cache is not None:
            return self._dossier_cache
        if self.dossier_path.is_file():
            try:
                self._dossier_cache = json.loads(self.dossier_path.read_text(encoding="utf-8"))
                return self._dossier_cache
            except Exception as e:
                logger.error("Failed to load historical dossier: %s", e)
        return {"locations": []}

    @staticmethod
    def data_sources(raw_dossier: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Exactly which files feed this analysis and what kind of data each one is."""
        logs = [c for loc in raw_dossier.get("locations", []) for c in loc.get("cleanups", [])]
        original = [c for c in logs if c.get("record_source", "pilot_log") == "pilot_log"]
        generated = [c for c in logs if c.get("record_source") == "seeded_prototype"]
        rain = load_daily_rainfall()
        nodes = load_landuse_nodes()

        def span(rows):
            ts = sorted(c["timestamp"][:10] for c in rows)
            return f"{ts[0]} → {ts[-1]}" if ts else "—"

        return [
            {"source": "data/historical_urban_dossier.json", "kind": "Original demo dossier (team-authored)",
             "records": len(original), "covers": span(original),
             "used_for": "Cleanup dates, weights, crews, clearance %, site metadata"},
            {"source": "data/historical_urban_dossier.json (generated part)", "kind": "Seeded prototype, driven by real rainfall",
             "records": len(generated), "covers": span(generated),
             "used_for": "Extends history to 3 months (data/generate_historical_dossier.py)"},
            {"source": "data/pune_rainfall_2026.json", "kind": "REAL: Open-Meteo archive (ERA5 reanalysis)",
             "records": len(rain), "covers": f"{min(rain)} → {max(rain)}" if rain else "—",
             "used_for": "Rain coupling evidence, antecedent rainfall per cleanup"},
            {"source": "data/pune_landuse_nodes.json", "kind": "Curated landmark coordinates",
             "records": len(nodes), "covers": "Pune land-use landmarks",
             "used_for": "Proximity evidence (Haversine distance)"},
        ]

    def analyze_recurrence(
        self,
        site_id_filter: Optional[str] = None,
        active_hotspots: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Perform comprehensive recurrence and root-cause analysis across pilot locations.

        Args:
            site_id_filter: Optional filter to focus on a specific location (e.g. 'LOC-PUNE-01').
            active_hotspots: Optional live pipeline hotspots to cross-correlate recurrence count.

        Returns:
            Dossier analysis report with return rates, persistence scores, hypotheses,
            and systemic civic recommendations.
        """
        raw_dossier = self._load_dossier()
        locations = raw_dossier.get("locations", [])

        if site_id_filter:
            locations = [loc for loc in locations if loc.get("site_id") == site_id_filter]

        analyzed_sites = []
        city_total_weight_kg = 0.0
        city_total_cleanups = 0

        for loc in locations:
            sid = loc["site_id"]
            name = loc["name"]
            lat = float(loc["lat"])
            lon = float(loc["lon"])
            cleanups = loc.get("cleanups", [])
            cleanups_count = len(cleanups)
            city_total_cleanups += cleanups_count

            # Extract weights and dates
            weights = [c.get("weight_removed_kg", 0.0) for c in cleanups]
            total_loc_weight_kg = round(sum(weights), 1)
            city_total_weight_kg += total_loc_weight_kg

            clearance_rates = [c.get("clearance_effectiveness_pct", 90.0) for c in cleanups]
            mean_clearance = round(sum(clearance_rates) / max(1, len(clearance_rates)), 1)

            # Compute actual return intervals in days from chronological logs
            timestamps = []
            for c in cleanups:
                try:
                    ts = datetime.fromisoformat(c["timestamp"].replace("Z", "+00:00"))
                    timestamps.append(ts)
                except Exception:
                    pass
            timestamps.sort()

            intervals_days = []
            if len(timestamps) >= 2:
                for i in range(1, len(timestamps)):
                    diff_days = (timestamps[i] - timestamps[i - 1]).total_seconds() / 86400.0
                    intervals_days.append(round(diff_days, 1))
                avg_interval = round(sum(intervals_days) / len(intervals_days), 1)
            else:
                avg_interval = loc.get("average_return_interval_days", 4.5)

            # Recurrence Persistence Index, normalised to a 30-day rate so the
            # index stays comparable whatever the dossier window length:
            # 9 cleanups per 30 days -> ~0.84; 5 per 30 days -> ~0.47
            window_days = float(raw_dossier.get("time_window_days", 30)) or 30.0
            per_30d = cleanups_count * 30.0 / window_days
            persistence_index = round(min(1.0, max(0.2, (per_30d / 30.0) * 2.8)), 2)

            # Month-by-month view of the monitoring window
            monthly: Dict[str, Dict[str, float]] = defaultdict(lambda: {"cleanups": 0, "weight_kg": 0.0, "rain_48h_mm": 0.0})
            for c in cleanups:
                m = monthly[c["timestamp"][:7]]
                m["cleanups"] += 1
                m["weight_kg"] += float(c.get("weight_removed_kg", 0.0))
                m["rain_48h_mm"] += float(c.get("rainfall_prev_48h_mm", 0.0))
            monthly_breakdown = [
                {"month": k, "cleanups": int(v["cleanups"]), "weight_kg": round(v["weight_kg"], 1),
                 "mean_kg_per_cleanup": round(v["weight_kg"] / max(1, v["cleanups"]), 1)}
                for k, v in sorted(monthly.items())
            ]

            # Trend: least-squares slope of removed mass per 30 days across the window
            if len(timestamps) >= 3:
                t0 = timestamps[0]
                xs = [(t - t0).total_seconds() / 86400.0 for t in timestamps]
                ws = [float(c.get("weight_removed_kg", 0.0)) for c in sorted(cleanups, key=lambda c: c["timestamp"])]
                mx, my = sum(xs) / len(xs), sum(ws) / len(ws)
                sxx = sum((x - mx) ** 2 for x in xs) or 1.0
                slope_per_30d = sum((x - mx) * (w - my) for x, w in zip(xs, ws)) / sxx * 30.0
            else:
                slope_per_30d = 0.0
            trend = "WORSENING" if slope_per_30d > 1.5 else ("IMPROVING" if slope_per_30d < -1.5 else "STABLE")

            root_cause = attribute_root_cause(lat, lon, cleanups)

            # Generate trend sparkline points for UI charts
            sparkline_points = []
            for idx, c in enumerate(cleanups, 1):
                date_str = c["timestamp"].split("T")[0]
                sparkline_points.append({
                    "log_id": c["log_id"],
                    "date": date_str,
                    "sequence": idx,
                    "weight_kg": c["weight_removed_kg"],
                    "effectiveness_pct": c["clearance_effectiveness_pct"],
                    "status": c.get("verification_status", "PASS"),
                })

            # Check if live report count augments recurrence
            live_recurrence = cleanups_count
            if active_hotspots:
                for ah in active_hotspots:
                    ah_lat = float(ah.get("lat", 0.0))
                    ah_lon = float(ah.get("lon", 0.0))
                    if math.isclose(lat, ah_lat, abs_tol=0.005) and math.isclose(lon, ah_lon, abs_tol=0.005):
                        live_recurrence += int(ah.get("recurrence", 1)) - 1

            landuse = attribute_landuse_root_cause(lat, lon)

            analyzed_sites.append({
                "site_id": sid,
                "name": name,
                "lat": lat,
                "lon": lon,
                "ward": loc.get("ward", "Kasba Peth"),
                "drain_name": loc.get("drain_name", "Stormwater Channel"),
                "drain_distance_m": loc.get("drain_distance_m", 15.0),
                "total_recurrence_count": live_recurrence,
                "historical_cleanup_events": cleanups_count,
                "average_return_interval_days": avg_interval,
                "recurrence_persistence_index": persistence_index,
                "persistence_level": "CHRONIC" if persistence_index >= 0.75 else ("ELEVATED" if persistence_index >= 0.5 else "MODERATE"),
                "dominant_waste_stream": loc.get("dominant_waste_stream", "PET bottles & HDPE packaging"),
                "waste_classes": loc.get("waste_classes", ["plastic_bottle", "plastic_bag_wrapper"]),
                "total_diverted_weight_kg": total_loc_weight_kg,
                "mean_clearance_pct": mean_clearance,
                "root_cause_hypothesis": root_cause["hypothesis"],
                "root_cause_analysis": root_cause,
                "landuse_attribution": landuse,
                "field_observation_note": loc.get("root_cause_hypothesis", ""),
                "systemic_interventions": root_cause["recommended_interventions"] + loc.get("systemic_interventions", [
                    "Enforce commercial bin mandate",
                    "Install heavy-duty bar screen at culvert apron",
                ])[:2],
                "monthly_breakdown": monthly_breakdown,
                "mass_trend_kg_per_30d": round(slope_per_30d, 2),
                "mass_trend": trend,
                "cleanups_per_30d": round(per_30d, 1),
                "sample_before_image": loc.get("sample_before_image", "sample_bottles_drain.jpg"),
                "sample_after_image": loc.get("sample_after_image", "sample_cleared_drain.jpg"),
                "sample_partial_image": loc.get("sample_partial_image", "sample_partial_cleanup.jpg"),
                "sparkline_data": sparkline_points,
                "cleanups": cleanups,
                "provenance_badge": "[Inferred / Prototype Historical Dossier]",
                "provenance_category": "Inferred",
            })

        # Sort sites by recurrence count descending (most chronic choke points first)
        analyzed_sites.sort(key=lambda x: -x["total_recurrence_count"])

        # City-wide summary KPIs
        top_chronic = analyzed_sites[0]["name"] if analyzed_sites else "None"
        avg_city_interval = round(
            sum(s["average_return_interval_days"] for s in analyzed_sites) / max(1, len(analyzed_sites)), 1
        )
        high_recurrence_sites = sum(1 for s in analyzed_sites if s["recurrence_persistence_index"] >= 0.75)

        return {
            "municipality": raw_dossier.get("municipality", "Pune Municipal Corporation (PMC)"),
            "monitoring_window_days": raw_dossier.get("time_window_days", 30),
            "summary_metrics": {
                "total_monitored_sites": len(analyzed_sites),
                "total_cleanups_completed": city_total_cleanups,
                "total_diverted_plastic_kg": round(city_total_weight_kg, 1),
                "average_return_interval_days": avg_city_interval,
                "chronic_sites_count": high_recurrence_sites,
                "chronic_site_percentage": round((high_recurrence_sites / max(1, len(analyzed_sites))) * 100.0, 1),
                "top_chronic_chokepoint": top_chronic,
            },
            "locations": analyzed_sites,
            "provenance_tag": "[Inferred / Prototype Historical Dossier]",
            "methodology": (
                f"{raw_dossier.get('time_window_days', 30)}-Day Cleanup Dossier + Real Open-Meteo Rainfall + "
                "Multi-Evidence Root-Cause Attribution (proximity, waste fingerprint, timing, rain coupling)"
            ),
            "data_sources": self.data_sources(raw_dossier),
            "window_start": raw_dossier.get("start_date"),
            "window_end": raw_dossier.get("end_date"),
            "data_notes": raw_dossier.get("data_notes", ""),
        }
