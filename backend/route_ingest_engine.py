"""Passive dashcam ingestion: vehicles that already drive every street report drains for free.

A vehicle (PMPML bus, PMC garbage van) uploads a batch of GPS-tagged frames. Each frame
passes cheap filters before the detector runs, and only confident detections next to a
mapped drain become reports (channel="dashcam"), which then flow through the normal
pipeline: DBSCAN clustering, dispatch, verification and the incentive ledger.

Frame filters, cheapest first:
  1. near a drain     within NEAR_DRAIN_M of a mapped drain line (no YOLO run otherwise)
  2. not a repeat     no accepted frame from this vehicle within DEDUP_RADIUS_M today
  3. usable photo     passes the image-quality gate
  4. waste confirmed  detector found waste with mean confidence >= MIN_CONFIDENCE

Replay mode feeds the curated TACO sample photos along a fixed Pune route so the demo
works without a real dashcam. Replay reports are tagged provenance="simulated".
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.recurrence_engine import haversine_m
from backend.taco_adapter import detect_waste

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "data" / "taco_samples"

NEAR_DRAIN_M = 50.0
DEDUP_RADIUS_M = 50.0
DEDUP_WINDOW_H = 24.0
MIN_CONFIDENCE = 0.45
MAX_FRAMES_PER_BATCH = 60

# Decisions recorded per frame
ACCEPTED = "accepted"
SKIP_FAR = "far_from_drain"
SKIP_REPEAT = "already_reported_nearby"
SKIP_QUALITY = "poor_image_quality"
SKIP_NO_WASTE = "no_waste_detected"
SKIP_LOW_CONF = "low_confidence"
SKIP_UNREADABLE = "unreadable_frame"
DECISION_LABELS = {
    ACCEPTED: "Report created",
    SKIP_FAR: "Not near a mapped drain",
    SKIP_REPEAT: "Already reported nearby today",
    SKIP_QUALITY: "Blurry / dark frame",
    SKIP_NO_WASTE: "No waste detected",
    SKIP_LOW_CONF: "Detection below confidence threshold",
    SKIP_UNREADABLE: "Unreadable frame",
}


class UnknownVehicleError(LookupError):
    """The vehicle has no registered dashcam reporter."""


@dataclass
class Frame:
    lat: float
    lon: float
    captured_at: Optional[str] = None
    image_bytes: Optional[bytes] = None
    sample_filename: Optional[str] = None


# ── Replay routes (simulated) ────────────────────────────────────────────────
# Waypoints follow roads alongside mapped drains in data/urban_drain_network.geojson.
REPLAY_ROUTES: Dict[str, Dict[str, Any]] = {
    "PMPML-158": {
        "vehicle_id": "PMPML-BUS-158",
        "name": "PMPML Route 158 · Swargate → Nagzari Nallah → Shaniwar Wada → Shivajinagar",
        "frame_spacing_m": 200.0,
        "waypoints": [
            (18.5005, 73.8613), (18.5060, 73.8627), (18.5140, 73.8637), (18.5190, 73.8641),
            (18.5196, 73.8572), (18.5215, 73.8567), (18.5230, 73.8552), (18.5250, 73.8522),
            (18.5310, 73.8507), (18.5380, 73.8492),
        ],
    },
    "PMC-VAN-12": {
        "vehicle_id": "PMC-GARBAGE-VAN-12",
        "name": "PMC garbage van 12 · Kothrud feeder → Paud Road → Deccan Gymkhana",
        "frame_spacing_m": 220.0,
        "waypoints": [
            (18.5021, 73.7922), (18.5051, 73.8051), (18.5071, 73.8181), (18.5101, 73.8321),
            (18.5121, 73.8381), (18.5161, 73.8411), (18.5191, 73.8441),
        ],
    },
}

# A mix of dirty and clean frames, so the replay shows the filters doing their job
REPLAY_SAMPLE_SEQUENCE = (
    "sample_bottles_drain.jpg", "sample_cleared_drain.jpg", "sample_plastic_cups_gutter.jpg",
    "sample_polypropylene_sacks.jpg", "sample_cardboard_box_drain.jpg", "sample_cleared_curb.jpg",
    "sample_snack_wrappers_curb.jpg", "sample_partial_cleanup.jpg", "sample_food_wrappers_culvert.jpg",
    "sample_mixed_waste_grate.jpg", "sample_plastic_bag_curb.jpg", "sample_pet_bottle_pile.jpg",
    "sample_beverage_cans_gutter.jpg",
)


def _interpolate(waypoints: Sequence[Tuple[float, float]], spacing_m: float) -> List[Tuple[float, float]]:
    """Points every ``spacing_m`` along the polyline, starting at the first waypoint."""
    points = [tuple(waypoints[0])]
    carry = 0.0  # distance travelled since the last emitted point
    for (lat1, lon1), (lat2, lon2) in zip(waypoints, waypoints[1:]):
        seg = haversine_m(lat1, lon1, lat2, lon2)
        pos = spacing_m - carry
        while pos <= seg:
            t = pos / seg
            points.append((lat1 + (lat2 - lat1) * t, lon1 + (lon2 - lon1) * t))
            pos += spacing_m
        carry = (carry + seg) % spacing_m
    return points


def replay_frames(route_id: str, now: Optional[datetime] = None) -> List[Frame]:
    route = REPLAY_ROUTES.get(route_id)
    if route is None:
        raise KeyError(route_id)
    now = now or datetime.now(timezone.utc).replace(microsecond=0)
    points = _interpolate(route["waypoints"], route["frame_spacing_m"])
    n = len(points)
    return [
        Frame(lat=round(lat, 6), lon=round(lon, 6),
              captured_at=(now - timedelta(seconds=20 * (n - i))).isoformat(),
              sample_filename=REPLAY_SAMPLE_SEQUENCE[i % len(REPLAY_SAMPLE_SEQUENCE)])
        for i, (lat, lon) in enumerate(points)
    ]


def list_routes() -> List[Dict[str, Any]]:
    return [
        {"route_id": rid, "vehicle_id": r["vehicle_id"], "name": r["name"],
         "frames": len(_interpolate(r["waypoints"], r["frame_spacing_m"])), "waypoints": r["waypoints"]}
        for rid, r in REPLAY_ROUTES.items()
    ]


# ── Ingestion ────────────────────────────────────────────────────────────────
def reporter_for_vehicle(vehicle_id: str) -> Optional[Dict[str, Any]]:
    from backend.incentive_engine import list_reporters

    vid = (vehicle_id or "").strip().upper()
    return next((r for r in list_reporters()
                 if r.get("channel") == "dashcam" and str(r.get("vehicle_id", "")).upper() == vid), None)


def _parse_ts(value: Any) -> Optional[datetime]:
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _vehicle_label(reporter: Dict[str, Any], vehicle_id: str) -> str:
    return reporter.get("display_name") or f"{vehicle_id} dashcam"


def ingest_frames(pipeline: Any, vehicle_id: str, frames: Sequence[Frame], replay: bool = False,
                  now: Optional[datetime] = None) -> Dict[str, Any]:
    """Filter a batch of dashcam frames and create reports for confident detections near drains.

    ``pipeline`` is a PlasticWatchUrbanPipeline (its drain engine and submit_report are reused).
    Raises UnknownVehicleError for an unregistered vehicle and ValueError for an oversized batch.
    """
    reporter = reporter_for_vehicle(vehicle_id)
    if reporter is None:
        raise UnknownVehicleError(f"Vehicle {vehicle_id} has no registered dashcam reporter")
    if len(frames) > MAX_FRAMES_PER_BATCH:
        raise ValueError(f"At most {MAX_FRAMES_PER_BATCH} frames per batch")
    now = now or datetime.now(timezone.utc)
    label = _vehicle_label(reporter, vehicle_id)
    source = f"{label}{' (replay)' if replay else ''}"
    provenance = "simulated" if replay else "observed"

    # Spots this vehicle already reported in the last 24 h (repeat drives add nothing)
    since = now - timedelta(hours=DEDUP_WINDOW_H)
    seen: List[Tuple[float, float]] = [
        (float(r["lat"]), float(r["lon"])) for r in pipeline.get_reports()
        if r.get("reporter_id") == reporter["reporter_id"] and (_parse_ts(r.get("submitted_at")) or since) > since
    ]

    trace: List[Dict[str, Any]] = []
    report_ids: List[str] = []
    for i, f in enumerate(frames):
        row: Dict[str, Any] = {"frame": i, "lat": f.lat, "lon": f.lon, "captured_at": f.captured_at,
                               "photo": f.sample_filename, "confidence": None, "report_id": None}
        trace.append(row)
        dist, drain = pipeline.risk_engine.calculate_drain_distance_m(f.lat, f.lon)
        row.update(drain_distance_m=dist, drain_name=(drain or {}).get("name"))
        if dist > NEAR_DRAIN_M:
            row["decision"] = SKIP_FAR
            continue
        if any(haversine_m(f.lat, f.lon, lat, lon) <= DEDUP_RADIUS_M for lat, lon in seen):
            row["decision"] = SKIP_REPEAT
            continue
        try:
            source_img = f.image_bytes if f.image_bytes else str(SAMPLES_DIR / Path(f.sample_filename or "").name)
            det = detect_waste(source_img, filename_hint=f.sample_filename or "")
        except Exception as exc:  # one bad frame must not sink the batch
            logger.warning("Dashcam frame %d from %s unreadable: %s", i, vehicle_id, exc)
            row["decision"] = SKIP_UNREADABLE
            continue
        row["confidence"] = det.get("mean_confidence")
        if det.get("needs_review"):
            row["decision"] = SKIP_QUALITY
        elif not det.get("waste_detected"):
            row["decision"] = SKIP_NO_WASTE
        elif float(det.get("mean_confidence") or 0.0) < MIN_CONFIDENCE:
            row["decision"] = SKIP_LOW_CONF
        else:
            res = pipeline.submit_report(
                lat=f.lat, lon=f.lon,
                photo_filename=f.sample_filename or f"{vehicle_id}_frame{i}.jpg",
                notes=f"{source}: {det.get('item_count', 0)} item(s) {round(dist)} m from {(drain or {}).get('name', 'drain')}",
                image_bytes=f.image_bytes,
                reporter_id=reporter["reporter_id"],
                source=source,
                detection=det,
                extra_fields={"provenance": provenance, "vehicle_id": vehicle_id, "captured_at": f.captured_at},
            )
            row["decision"] = ACCEPTED
            row["report_id"] = res["report"]["id"]
            report_ids.append(res["report"]["id"])
            seen.append((f.lat, f.lon))

    counts = Counter(r["decision"] for r in trace)
    return {
        "status": "success",
        "vehicle_id": vehicle_id,
        "reporter_id": reporter["reporter_id"],
        "source": source,
        "provenance": provenance,
        "frames_received": len(frames),
        "near_drain": len(frames) - counts.get(SKIP_FAR, 0),
        "analysed": sum(1 for r in trace if r["confidence"] is not None or r["decision"] == SKIP_UNREADABLE),
        "accepted": counts.get(ACCEPTED, 0),
        "decisions": {k: counts.get(k, 0) for k in DECISION_LABELS},
        "report_ids": report_ids,
        "frames": trace,
        "thresholds": {"near_drain_m": NEAR_DRAIN_M, "dedup_radius_m": DEDUP_RADIUS_M,
                       "dedup_window_h": DEDUP_WINDOW_H, "min_confidence": MIN_CONFIDENCE},
    }


def replay_route(pipeline: Any, route_id: str, now: Optional[datetime] = None) -> Dict[str, Any]:
    """Drive a simulated route: sample photos along a fixed polyline, labelled provenance=simulated."""
    route = REPLAY_ROUTES.get(route_id)
    if route is None:
        raise KeyError(route_id)
    result = ingest_frames(pipeline, route["vehicle_id"], replay_frames(route_id, now), replay=True, now=now)
    return {**result, "route_id": route_id, "route_name": route["name"], "waypoints": route["waypoints"]}
