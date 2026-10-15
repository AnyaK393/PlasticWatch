"""Ward assignment and drain coverage for PlasticWatch reports.

Until real PMC ward polygons (data/pune_wards.geojson) are added, a report is
assigned to the zone of the nearest vendor anchor in data/vendor_registry.json,
reusing SmartDispatchEngine.match_vendor.

Drain coverage is the share of mapped drain length (data/urban_drain_network.geojson)
that has at least one report within 50 m during a time window: how much of the
network somebody actually looked at.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from backend.recurrence_engine import haversine_m
from backend.smart_dispatch_engine import SmartDispatchEngine

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DRAIN_GEOJSON_PATH = ROOT / "data" / "urban_drain_network.geojson"

COVERAGE_RADIUS_M = 50.0      # same as the DBSCAN merge radius
COVERAGE_SAMPLE_STEP_M = 25.0  # drain lines are checked at points this far apart


@lru_cache(maxsize=1)
def _dispatch_engine() -> SmartDispatchEngine:
    return SmartDispatchEngine()


def assign_ward(lat: float, lon: float) -> Optional[str]:
    """Zone name for a coordinate, or None if the vendor registry is unavailable."""
    try:
        match = _dispatch_engine().match_vendor(float(lat), float(lon))
    except Exception as exc:  # ward is informational; never block a report on it
        logger.warning("Ward assignment failed for (%s, %s): %s", lat, lon, exc)
        return None
    return match.get("matched_zone") if match.get("matched") else None


@lru_cache(maxsize=4)
def drain_sample_points(path: Path = DRAIN_GEOJSON_PATH, step_m: float = COVERAGE_SAMPLE_STEP_M) -> Tuple[Tuple[float, float, str, float], ...]:
    """(lat, lon, drain_id, length_m) points about ``step_m`` apart along every drain line."""
    try:
        features = json.loads(Path(path).read_text(encoding="utf-8")).get("features", [])
    except (OSError, ValueError) as exc:
        logger.error("Drain network unavailable for coverage: %s", exc)
        return ()
    points: List[Tuple[float, float, str, float]] = []
    for f in features:
        geom = f.get("geometry") or {}
        lines = [geom.get("coordinates", [])] if geom.get("type") == "LineString" else (
            geom.get("coordinates", []) if geom.get("type") == "MultiLineString" else [])
        drain_id = str((f.get("properties") or {}).get("id", "drain"))
        for line in lines:
            for (lon1, lat1), (lon2, lat2) in zip(line, line[1:]):
                seg = haversine_m(lat1, lon1, lat2, lon2)
                n = max(1, int(round(seg / step_m)))
                for i in range(n):  # segment midpoints of n equal pieces
                    t = (i + 0.5) / n
                    points.append((lat1 + (lat2 - lat1) * t, lon1 + (lon2 - lon1) * t, drain_id, seg / n))
    return tuple(points)


def _parse_ts(value: Any) -> Optional[datetime]:
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def reports_in_window(reports: Iterable[Dict[str, Any]], days: Optional[int], now: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """Reports submitted within the last ``days`` (None = all time); rejected captures excluded."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days) if days else None
    out = []
    for r in reports:
        if r.get("verification") == "rejected":
            continue
        ts = _parse_ts(r.get("submitted_at"))
        if ts is None or (cutoff and not (cutoff <= ts <= now + timedelta(minutes=5))):
            continue
        out.append(r)
    return out


def drain_coverage(reports: Iterable[Dict[str, Any]], radius_m: float = COVERAGE_RADIUS_M,
                   samples: Optional[Sequence[Tuple[float, float, str, float]]] = None) -> Dict[str, Any]:
    """Share of drain length with at least one of ``reports`` within ``radius_m``.

    ``samples`` restricts the drain network (e.g. to one ward); default is the whole network.
    """
    samples = drain_sample_points() if samples is None else samples
    located = []
    for r in reports:
        try:
            located.append((float(r["lat"]), float(r["lon"])))
        except (KeyError, TypeError, ValueError):
            continue
    covered_m = 0.0
    for lat, lon, _, length_m in samples:
        # cheap bounding-box reject before the haversine (1 deg lat ~ 111 km)
        if any(abs(lat - rl) < 0.001 and abs(lon - ro) < 0.0012 and haversine_m(lat, lon, rl, ro) <= radius_m
               for rl, ro in located):
            covered_m += length_m
    total_m = sum(s[3] for s in samples)
    return {
        "drain_km": round(total_m / 1000.0, 2),
        "covered_km": round(covered_m / 1000.0, 2),
        "coverage_pct": round(100.0 * covered_m / total_m, 1) if total_m else 0.0,
        "radius_m": radius_m,
    }


# ── Ward Drain Score ─────────────────────────────────────────────────────────
# Wards are the vendor-registry zones (one per vendor, named by its first zone) until
# real PMC ward polygons are added. Every report, drain point and cleanup is assigned
# to a ward with the same nearest-anchor rule as assign_ward().
HISTORICAL_DOSSIER_PATH = ROOT / "data" / "historical_urban_dossier.json"
HISTORY_WINDOW_DAYS = 90         # clearance and recurrence use the cleanup history
RECURRENCE_WINDOW_DAYS = 30
COVERAGE_TARGET_PCT = 25.0       # weekly coverage that earns full marks
FAST_CLEAR_H, SLOW_CLEAR_H = 24.0, 120.0
CLOGS_PER_SITE_FULL_PENALTY = 10.0  # clogs per site per 30 days that scores 0
SCORE_WEIGHTS = {"coverage": 0.30, "clear_rate": 0.30, "speed": 0.20, "recurrence": 0.20}
MIN_COMPONENTS = 2


def ward_names() -> List[str]:
    return [v.get("assigned_zones", ["Unassigned"])[0] for v in _dispatch_engine().vendors]


@lru_cache(maxsize=4)
def _ward_drain_samples(samples: Tuple[Tuple[float, float, str, float], ...]) -> Dict[str, Tuple[Tuple[float, float, str, float], ...]]:
    by_ward: Dict[str, List[Tuple[float, float, str, float]]] = {}
    for sample in samples:
        by_ward.setdefault(assign_ward(sample[0], sample[1]) or "Unassigned", []).append(sample)
    return {w: tuple(v) for w, v in by_ward.items()}


def _hours_between(start: Any, end: Any) -> Optional[float]:
    a, b = _parse_ts(start), _parse_ts(end)
    if a is None or b is None or b < a:
        return None
    return (b - a).total_seconds() / 3600.0


def history_cleanups(path: Path = HISTORICAL_DOSSIER_PATH) -> List[Dict[str, Any]]:
    """Logged cleanups from the historical dossier, one row each with ward, outcome and time to clear."""
    try:
        dossier = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Cleanup history unavailable: %s", exc)
        return []
    rows = []
    for loc in dossier.get("locations", []):
        try:
            ward = assign_ward(float(loc["lat"]), float(loc["lon"])) or "Unassigned"
        except (KeyError, TypeError, ValueError):
            continue
        for c in loc.get("cleanups", []):
            at = _parse_ts(c.get("timestamp"))
            if at is None:
                continue
            rows.append({
                "ward": ward, "site": loc.get("site_id") or loc.get("name"), "at": at,
                "verified_clear": str(c.get("verification_status", "")).upper() == "PASS",
                "hours_to_clear": _hours_between(c.get("first_reported_at"), c.get("timestamp")),
                "source": "history",
            })
    return rows


def live_cleanups(rewards: Iterable[Dict[str, Any]], case_outcomes: Iterable[Dict[str, Any]] = ()) -> List[Dict[str, Any]]:
    """Cleanups decided in the live system.

    ``case_outcomes``: {case_id, lat, lon, decision, decided_at?} per verification case with a final
    decision (INCONCLUSIVE is skipped: the evidence was unusable, not the cleanup).
    Time to clear comes from first-reporter rewards: report submitted → reward released.
    """
    rows = []
    for c in case_outcomes:
        decision = str(c.get("decision") or "").upper()
        if decision not in ("PASS", "FLAGGED FOR REVIEW"):
            continue
        try:
            ward = assign_ward(float(c["lat"]), float(c["lon"])) or "Unassigned"
        except (KeyError, TypeError, ValueError):
            continue
        rows.append({"ward": ward, "site": c.get("case_id"), "at": _parse_ts(c.get("decided_at")) or datetime.now(timezone.utc),
                     "verified_clear": decision == "PASS", "hours_to_clear": None, "source": "live"})
    for r in rewards:
        if r.get("state") == "RELEASED" and r.get("role") == "first_reporter":
            hours = _hours_between(r.get("report_submitted_at"), r.get("released_at"))
            if hours is not None:
                rows.append({"ward": r.get("ward") or assign_ward(r["lat"], r["lon"]) or "Unassigned", "site": r.get("report_id"),
                             "at": _parse_ts(r.get("released_at")), "verified_clear": None, "hours_to_clear": hours, "source": "live"})
    return rows


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    v = sorted(values)
    mid = len(v) // 2
    return v[mid] if len(v) % 2 else (v[mid - 1] + v[mid]) / 2.0


def _clamp(x: float) -> float:
    return max(0.0, min(100.0, x))


def ward_scorecard(
    reports: Iterable[Dict[str, Any]],
    rewards: Iterable[Dict[str, Any]] = (),
    case_outcomes: Iterable[Dict[str, Any]] = (),
    days: int = 30,
    now: Optional[datetime] = None,
    history_path: Path = HISTORICAL_DOSSIER_PATH,
) -> Dict[str, Any]:
    """Per-ward Drain Score (0-100) and its components, ranked best first.

    Components (each 0-100, missing ones are left out and the weights re-normalised):
      coverage    drain length within 50 m of a report in the last ``days``, vs a 25% target
      clear_rate  share of cleanups (last 90 days) that passed verification
      speed       median hours from first report to clearance: <=24 h full marks, >=120 h zero
      recurrence  clogs per site in the last 30 days: 0 full marks, >=10 zero
    A ward needs at least two components for a score.
    """
    now = now or datetime.now(timezone.utc)
    rewards = list(rewards)
    window = reports_in_window(reports, days, now)
    samples_by_ward = _ward_drain_samples(drain_sample_points())
    cleanups = history_cleanups(history_path) + live_cleanups(rewards, case_outcomes)
    hist_start = now - timedelta(days=HISTORY_WINDOW_DAYS)
    rec_start = now - timedelta(days=RECURRENCE_WINDOW_DAYS)
    prev_start = rec_start - timedelta(days=RECURRENCE_WINDOW_DAYS)

    def ward_of(r: Dict[str, Any]) -> str:
        return r.get("ward") or assign_ward(r["lat"], r["lon"]) or "Unassigned"

    wards = []
    for ward in ward_names():
        mine = [r for r in window if ward_of(r) == ward]
        samples = samples_by_ward.get(ward, ())
        cov = drain_coverage(mine, samples=samples) if samples else None
        recent = [c for c in cleanups if c["ward"] == ward and c["at"] and c["at"] >= hist_start]
        judged = [c for c in recent if c["verified_clear"] is not None]
        hours = [c["hours_to_clear"] for c in recent if c["hours_to_clear"] is not None]
        sites = {c["site"] for c in recent if c["source"] == "history"}
        clogs_now = sum(1 for c in recent if c["source"] == "history" and c["at"] >= rec_start)
        clogs_prev = sum(1 for c in cleanups if c["ward"] == ward and c["source"] == "history" and c["at"] and prev_start <= c["at"] < rec_start)

        clear_rate = 100.0 * sum(c["verified_clear"] for c in judged) / len(judged) if judged else None
        median_h = _median(hours)
        clogs_per_site = clogs_now / len(sites) if sites else None
        components = {
            "coverage": _clamp(100.0 * cov["coverage_pct"] / COVERAGE_TARGET_PCT) if cov else None,
            "clear_rate": clear_rate,
            "speed": _clamp(100.0 * (SLOW_CLEAR_H - median_h) / (SLOW_CLEAR_H - FAST_CLEAR_H)) if median_h is not None else None,
            "recurrence": _clamp(100.0 * (1.0 - clogs_per_site / CLOGS_PER_SITE_FULL_PENALTY)) if clogs_per_site is not None else None,
        }
        present = {k: v for k, v in components.items() if v is not None}
        score = None
        if len(present) >= MIN_COMPONENTS:
            total_w = sum(SCORE_WEIGHTS[k] for k in present)
            score = round(sum(SCORE_WEIGHTS[k] * v for k, v in present.items()) / total_w, 1)
        wards.append({
            "ward": ward,
            "score": score,
            "components": {k: (round(v, 1) if v is not None else None) for k, v in components.items()},
            "coverage_pct": cov["coverage_pct"] if cov else None,
            "drain_km": cov["drain_km"] if cov else 0.0,
            "covered_km": cov["covered_km"] if cov else 0.0,
            "verified_clear_rate_pct": round(clear_rate, 1) if clear_rate is not None else None,
            "cleanups_judged": len(judged),
            "median_hours_to_clear": round(median_h, 1) if median_h is not None else None,
            "clogs_last_30d": clogs_now,
            "clogs_prev_30d": clogs_prev,
            "recurrence_trend_pct": round(100.0 * (clogs_now - clogs_prev) / clogs_prev, 1) if clogs_prev else None,
            "clogs_per_site_30d": round(clogs_per_site, 2) if clogs_per_site is not None else None,
            "reports": len(mine),
            "active_reporters": len({r["reporter_id"] for r in mine if r.get("reporter_id")}),
        })

    ranked = sorted(wards, key=lambda w: (w["score"] is None, -(w["score"] or 0.0)))
    rank = 0
    for w in ranked:
        w["rank"] = (rank := rank + 1) if w["score"] is not None else None
    return {
        "window_days": days,
        "history_window_days": HISTORY_WINDOW_DAYS,
        "generated_at": now.replace(microsecond=0).isoformat(),
        "weights": SCORE_WEIGHTS,
        "targets": {"coverage_pct": COVERAGE_TARGET_PCT, "fast_clear_h": FAST_CLEAR_H, "slow_clear_h": SLOW_CLEAR_H,
                    "clogs_per_site_zero_score": CLOGS_PER_SITE_FULL_PENALTY},
        "ward_method": "nearest vendor-zone anchor (data/vendor_registry.json) until PMC ward polygons are added",
        "wards": ranked,
    }


@lru_cache(maxsize=1)
def ward_polygons() -> Dict[str, Any]:
    """GeoJSON of the ward zones: Voronoi cells of the vendor anchors, merged per ward.

    These are exactly the regions assign_ward() uses, so the map matches the scoring.
    Longitude is scaled by cos(latitude) so cells follow ground distance, not degrees.
    """
    import math

    from shapely.geometry import MultiPoint, Point, box, mapping
    from shapely.affinity import scale
    from shapely.ops import unary_union, voronoi_diagram

    anchors: List[Tuple[str, float, float]] = []
    for v in _dispatch_engine().vendors:
        ward = v.get("assigned_zones", ["Unassigned"])[0]
        anchors += [(ward, float(a[0]), float(a[1])) for a in v.get("zone_anchors", [])]
    if len(anchors) < 2:
        return {"type": "FeatureCollection", "features": []}
    k = math.cos(math.radians(sum(a[1] for a in anchors) / len(anchors)))
    pts = [Point(lon * k, lat) for _, lat, lon in anchors]
    lats, lons = [a[1] for a in anchors], [a[2] for a in anchors]
    frame = box((min(lons) - 0.04) * k, min(lats) - 0.04, (max(lons) + 0.04) * k, max(lats) + 0.04)
    cells = voronoi_diagram(MultiPoint(pts), envelope=frame)
    by_ward: Dict[str, list] = {}
    for cell in cells.geoms:
        owner = next((anchors[i][0] for i, p in enumerate(pts) if cell.covers(p)), None)
        if owner:
            by_ward.setdefault(owner, []).append(cell.intersection(frame))
    features = []
    for ward, parts in by_ward.items():
        geom = scale(unary_union(parts), xfact=1.0 / k, yfact=1.0, origin=(0, 0))
        features.append({"type": "Feature", "properties": {"ward": ward}, "geometry": mapping(geom)})
    return {"type": "FeatureCollection", "features": features}
