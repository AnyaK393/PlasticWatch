"""PlasticWatch Urban Pipeline: Civic Waste Ingestion, DBSCAN Clustering & Hydrological Risk.

Connects:
  1. Citizen mobile reports / photo captures
  2. TACO waste object detection & severity estimation
  3. DBSCAN spatial clustering (50m duplicate consolidation)
  4. Urban drain proximity & Open-Meteo rainfall risk engine (0-100 score)
  5. Municipal truck route optimization & human-in-the-loop dispatch
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from backend.clustering_engine import DBSCANClusteringEngine
from backend.drain_risk_engine import DrainRiskEngine
from backend.recurrence_engine import attribute_landuse_root_cause
from backend.route_optimizer import RouteOptimizer
from backend.storage import CorruptDataError, locked, read_json, write_bytes_atomic, write_json
from backend.taco_adapter import REVIEW_STATUS, detect_waste, get_sample_images

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DB_PATH = ROOT / "data" / "citizen_reports.json"
SAMPLES_DIR = ROOT / "data" / "taco_samples"
UPLOADS_DIR = ROOT / "data" / "uploads"

# Reports in these states stay out of DBSCAN clustering and the cleanup queue
EXCLUDED_FROM_QUEUE = {REVIEW_STATUS, "rejected"}


def resolve_photo_path(photo_filename: Optional[str]) -> Optional[Path]:
    """Locate a report photo among citizen uploads first, then curated samples."""
    if not photo_filename:
        return None
    name = Path(str(photo_filename)).name  # never follow directory components
    for base in (UPLOADS_DIR, SAMPLES_DIR):
        candidate = base / name
        if candidate.is_file():
            return candidate
    return None


@dataclass
class CitizenReport:
    id: str
    lat: float
    lon: float
    plastic_confidence: float
    severity: float
    photo_filename: str
    detected_classes: List[str]
    submitted_at: str
    source: str
    verification: str
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_seed_reports() -> List[CitizenReport]:
    """Default citizen reports along Pune drainage corridors for demonstration."""
    return [
        CitizenReport(
            id="CR-201",
            lat=18.5204,
            lon=73.8568,
            plastic_confidence=0.94,
            severity=4.6,
            photo_filename="sample_bottles_drain.jpg",
            detected_classes=["plastic_bottle", "plastic_bag_wrapper"],
            submitted_at="2026-10-01T10:14:00Z",
            source="Citizen Mobile App (GPS Verified)",
            verification="verified",
            notes="Severe plastic bottle buildup choking stormwater culvert inlet.",
        ),
        CitizenReport(
            id="CR-202",
            lat=18.5207,
            lon=73.8569,
            plastic_confidence=0.89,
            severity=4.2,
            photo_filename="sample_plastic_bag_curb.jpg",
            detected_classes=["plastic_bag_wrapper", "other_plastic"],
            submitted_at="2026-10-01T10:28:00Z",
            source="Citizen Mobile App (GPS Verified)",
            verification="pending",
            notes="Second report of plastic bags around same culvert curb.",
        ),
        CitizenReport(
            id="CR-203",
            lat=18.5205,
            lon=73.8566,
            plastic_confidence=0.91,
            severity=4.4,
            photo_filename="sample_mixed_waste_grate.jpg",
            detected_classes=["plastic_bottle", "other_plastic"],
            submitted_at="2026-10-01T10:45:00Z",
            source="Citizen Mobile App (GPS Verified)",
            verification="pending",
            notes="Third report: waste overflowing into storm drain opening.",
        ),
        CitizenReport(
            id="CR-204",
            lat=18.5060,
            lon=73.8625,
            plastic_confidence=0.92,
            severity=4.1,
            photo_filename="sample_food_wrappers_culvert.jpg",
            detected_classes=["plastic_bag_wrapper", "carton_paper"],
            submitted_at="2026-10-01T09:30:00Z",
            source="Citizen Mobile App (GPS Verified)",
            verification="verified",
            notes="Nagzari Nallah street grate covered with multilayer plastic packaging.",
        ),
        CitizenReport(
            id="CR-205",
            lat=18.5063,
            lon=73.8627,
            plastic_confidence=0.88,
            severity=3.8,
            photo_filename="sample_beverage_cans_gutter.jpg",
            detected_classes=["can_metal", "plastic_bottle"],
            submitted_at="2026-10-01T11:02:00Z",
            source="Citizen Mobile App (GPS Verified)",
            verification="pending",
            notes="Duplicate report 35m from CR-204 along Nagzari drainage corridor.",
        ),
        CitizenReport(
            id="CR-206",
            lat=18.5074,
            lon=73.8077,
            plastic_confidence=0.84,
            severity=3.5,
            photo_filename="sample_bottles_drain.jpg",
            detected_classes=["plastic_bottle", "can_metal"],
            submitted_at="2026-10-01T08:45:00Z",
            source="Citizen Mobile App (GPS Verified)",
            verification="verified",
            notes="Kothrud stormwater feeder roadside waste pile.",
        ),
        CitizenReport(
            id="CR-207",
            lat=18.5350,
            lon=73.8920,
            plastic_confidence=0.87,
            severity=4.5,
            photo_filename="sample_mixed_waste_grate.jpg",
            detected_classes=["plastic_bag_wrapper", "plastic_bottle", "other_plastic"],
            submitted_at="2026-10-01T11:20:00Z",
            source="Citizen Mobile App (GPS Verified)",
            verification="pending",
            notes="Mula-Mutha confluence outfall: heavy floating plastic buildup.",
        ),
        CitizenReport(
            id="CR-208",
            lat=18.5480,
            lon=73.9053,
            plastic_confidence=0.76,
            severity=2.6,
            photo_filename="sample_food_wrappers_culvert.jpg",
            detected_classes=["other_plastic", "carton_paper"],
            submitted_at="2026-10-01T07:15:00Z",
            source="Citizen Mobile App (GPS Verified)",
            verification="pending",
            notes="Yerawada side canal minor litter accumulation.",
        ),
    ]


class PlasticWatchUrbanPipeline:
    """End-to-end urban civic pipeline managing reports, clustering, risk, and routing."""

    def __init__(self):
        self.clustering_engine = DBSCANClusteringEngine(eps_meters=50.0, min_samples=2)
        self.risk_engine = DrainRiskEngine()
        self.route_optimizer = RouteOptimizer()
        self._reports_cache: Optional[List[Dict[str, Any]]] = None
        self._reports_mtime: Optional[float] = None

    def _db_mtime(self) -> Optional[float]:
        try:
            return REPORTS_DB_PATH.stat().st_mtime
        except OSError:
            return None

    def get_reports(self) -> List[Dict[str, Any]]:
        """Retrieve all active citizen reports.

        The cache is invalidated when the JSON file changes on disk, so reports
        submitted through the API process appear in the dashboard process.
        """
        if self._reports_cache is not None and self._reports_mtime == self._db_mtime():
            return self._reports_cache

        try:
            data = read_json(REPORTS_DB_PATH, None)
        except CorruptDataError as e:
            # The damaged file is preserved as *.corrupt-<timestamp>; continue from seeds
            logger.error("%s", e)
            data = None
        if isinstance(data, list):
            self._reports_cache = [r for r in data if self._valid_report(r)]
            self._reports_mtime = self._db_mtime()
            return self._reports_cache

        # Initialize with seed reports
        seeds = [r.to_dict() for r in get_seed_reports()]
        self.save_reports(seeds)
        return seeds

    @staticmethod
    def _valid_report(r: Any) -> bool:
        """Skip malformed rows instead of crashing clustering."""
        try:
            lat, lon = float(r["lat"]), float(r["lon"])
            return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0 and bool(r.get("id"))
        except (KeyError, TypeError, ValueError):
            return False

    @staticmethod
    def _next_report_id(reports: List[Dict[str, Any]]) -> str:
        nums = []
        for r in reports:
            try:
                nums.append(int(str(r.get("id", "")).split("-")[-1]))
            except ValueError:
                pass
        return f"CR-{max(nums, default=200) + 1}"

    def save_reports(self, reports: List[Dict[str, Any]]) -> None:
        """Persist reports to disk."""
        self._reports_cache = reports
        try:
            write_json(REPORTS_DB_PATH, reports)
            self._reports_mtime = self._db_mtime()
        except Exception as e:
            logger.error("Failed to save reports DB: %s", e)

    def submit_report(
        self,
        lat: float,
        lon: float,
        photo_filename: str = "sample_bottles_drain.jpg",
        notes: str = "",
        image_bytes: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """Ingest a new citizen mobile report, run TACO detection, and append to database."""
        if not (-90.0 <= float(lat) <= 90.0 and -180.0 <= float(lon) <= 180.0):
            raise ValueError("Coordinates out of range")
        notes = (notes or "")[:500]
        ran_detection = True
        upload_suffix = None

        # Run AI detection (slow: done before taking the database lock)
        if image_bytes:
            detection = detect_waste(image_bytes, filename_hint=photo_filename)
            upload_suffix = Path(photo_filename or "").suffix.lower()
            if upload_suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
                upload_suffix = ".jpg"
        else:
            sample_path = resolve_photo_path(photo_filename)
            if sample_path is not None:
                detection = detect_waste(str(sample_path), filename_hint=photo_filename)
            else:
                ran_detection = False
                detection = {
                    "mean_confidence": 0.88,
                    "severity": 3.8,
                    "detected_items": [{"class": "plastic_bottle"}, {"class": "plastic_bag_wrapper"}],
                    "hazard_level": "HIGH",
                    "needs_review": False,
                }

        classes = [item.get("class", "plastic_bottle") for item in detection.get("detected_items", [])]

        # Ambiguity routing: unreadable photos, or photos where AI could not confirm any waste
        review_reasons: List[str] = []
        if detection.get("needs_review"):
            review_reasons.extend(detection.get("image_quality", {}).get("reasons", []) or ["image quality"])
        if ran_detection and not detection.get("waste_detected", True):
            review_reasons.append("AI could not confirm waste in photo")
        needs_review = bool(review_reasons)

        # Reserve the id, store the photo and append the report atomically
        with locked(REPORTS_DB_PATH):
            self._reports_cache = None
            reports = list(self.get_reports())
            next_id = self._next_report_id(reports)
            if image_bytes:
                # Persist the real citizen photo so operators see it in the inspection drawer
                stored_name = f"{next_id}{upload_suffix}"
                try:
                    write_bytes_atomic(UPLOADS_DIR / stored_name, image_bytes)
                    photo_filename = stored_name
                except OSError as e:
                    logger.error("Failed to store uploaded photo: %s", e)
            new_rep = self._build_report(next_id, lat, lon, photo_filename, notes, detection, classes, needs_review, review_reasons)
            reports.append(new_rep)
            self.save_reports(reports)

        return self._submission_response(next_id, new_rep, detection, needs_review, review_reasons)

    @staticmethod
    def _build_report(next_id, lat, lon, photo_filename, notes, detection, classes, needs_review, review_reasons) -> Dict[str, Any]:
        new_rep = CitizenReport(
            id=next_id,
            lat=round(lat, 6),
            lon=round(lon, 6),
            plastic_confidence=detection.get("mean_confidence", 0.88),
            severity=detection.get("severity", 3.8),
            photo_filename=photo_filename,
            detected_classes=list(set(classes)),
            submitted_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            source="Citizen Mobile App (Live Ingestion)",
            verification=REVIEW_STATUS if needs_review else "pending",
            notes=notes or "Citizen mobile upload via PWA/API Feeder",
        ).to_dict()
        new_rep["status"] = REVIEW_STATUS if needs_review else "ACTIVE"
        if needs_review:
            new_rep["review_reasons"] = review_reasons
            new_rep["image_quality"] = {k: v for k, v in detection.get("image_quality", {}).items() if k != "thresholds"}
        return new_rep

    @staticmethod
    def _submission_response(next_id, new_rep, detection, needs_review, review_reasons) -> Dict[str, Any]:
        if needs_review:
            if detection.get("needs_review"):
                review_message = detection.get("review_message") or "⚠️ Photo appears blurry or dark. Submitted to Municipal Operator Review Queue."
            else:
                review_message = "⚠️ AI could not confirm waste in this photo. Submitted to Municipal Operator Review Queue."
            message = f"Report {next_id} routed to the Municipal Operator Review Queue ({'; '.join(review_reasons)})."
        else:
            review_message = ""
            message = f"Report {next_id} successfully submitted and indexed for DBSCAN clustering."

        # The annotated image is large and already returned by /api/detect
        detection_summary = {k: v for k, v in detection.items() if k not in ("annotated_image", "annotated_image_base64", "trace")}

        return {
            "status": "success",
            "report": new_rep,
            "detection": detection_summary,
            "needs_review": needs_review,
            "review_message": review_message,
            "message": message,
        }

    # ── Operator Review Queue (blurry / dark / ambiguous captures) ───────────
    def get_review_queue(self) -> List[Dict[str, Any]]:
        return [r for r in self.get_reports() if r.get("verification") == REVIEW_STATUS]

    def _set_review_outcome(self, report_id: str, verification: str, status: str, operator_note: str) -> Optional[Dict[str, Any]]:
        with locked(REPORTS_DB_PATH):
            self._reports_cache = None
            reports = list(self.get_reports())
            for r in reports:
                if r.get("id") == report_id:
                    if r.get("verification") != REVIEW_STATUS:
                        raise ValueError(f"{report_id} is not awaiting review (current: {r.get('verification')})")
                    r["verification"] = verification
                    r["status"] = status
                    r["reviewed_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
                    if operator_note:
                        r["operator_review_note"] = operator_note[:500]
                    self.save_reports(reports)
                    return r
        return None

    def approve_report(self, report_id: str, operator_note: str = "") -> Optional[Dict[str, Any]]:
        """Operator confirms the capture; it joins DBSCAN clustering and the cleanup queue."""
        return self._set_review_outcome(report_id, "verified", "APPROVED", operator_note)

    def reject_report(self, report_id: str, operator_note: str = "") -> Optional[Dict[str, Any]]:
        """Operator rejects the capture; it stays on record but out of the queue."""
        return self._set_review_outcome(report_id, "rejected", "REJECTED", operator_note)

    def run_pipeline(
        self,
        rainfall_override_mm: Optional[float] = None,
        rainfall_intensity_mmh: Optional[float] = None,
        drain_silt_ratio: Optional[float] = None,
        surface_slope_factor: float = 1.0,
    ) -> Dict[str, Any]:
        """Run the end-to-end urban intelligence pipeline.

        Args:
            rainfall_override_mm: Optional 24h forecast rainfall override (mm).
            rainfall_intensity_mmh: Rainfall intensity 0-75 mm/hr; None = live Open-Meteo current hour.
            drain_silt_ratio: Siltation override 0.0-1.0; None = each drain's own desilting record.
            surface_slope_factor: 1.0 flat, 1.25 moderate, 1.5 steep.
        """
        if rainfall_intensity_mmh is None:
            live = self.risk_engine.fetch_live_rainfall_intensity()
            rainfall_intensity_mmh = live["rainfall_intensity_mmh"]
            rain_source = live["source"]
        else:
            rain_source = "scenario override"
        all_reports = self.get_reports()
        # Unverified / rejected captures wait in the operator review queue
        raw_reports = [r for r in all_reports if r.get("verification") not in EXCLUDED_FROM_QUEUE]
        review_queue = [r for r in all_reports if r.get("verification") == REVIEW_STATUS]

        # Step 1: DBSCAN Spatial Clustering (eps=50m)
        hotspot_clusters = self.clustering_engine.cluster_reports(raw_reports)
        cluster_telemetry = self.clustering_engine.last_telemetry

        # Step 2: Urban Drain Proximity & Context Risk Scoring (0-100)
        scored_hotspots = []
        for cluster in hotspot_clusters:
            risk = self.risk_engine.compute_risk(
                cluster,
                rainfall_override_mm=rainfall_override_mm,
                rainfall_intensity_mmh=rainfall_intensity_mmh,
                drain_silt_ratio=drain_silt_ratio,
                surface_slope_factor=surface_slope_factor,
            )
            
            # Combine cluster info with risk breakdown
            combined = {**cluster, **risk}
            cluster_reports = cluster.get("reports", [])
            # Verification status of hotspot
            verified_count = sum(r.get("verification") == "verified" for r in cluster_reports)
            combined["verified_count"] = verified_count
            combined["dispatch_status"] = "PENDING"
            combined["status"] = "ACTIVE"
            # Stable identity across re-ranking (HOTSPOT-NN ids follow the live score order)
            combined["site_key"] = min(str(rid) for rid in cluster.get("report_ids", ["UNKNOWN"]))
            combined["landuse_attribution"] = attribute_landuse_root_cause(cluster["lat"], cluster["lon"])
            
            # Primary photo: most recent capture whose image is on disk (real citizen uploads first)
            ordered = sorted(cluster_reports, key=lambda r: str(r.get("submitted_at", "")), reverse=True)
            existing = [r["photo_filename"] for r in ordered if resolve_photo_path(r.get("photo_filename"))]
            rep_photos = [r.get("photo_filename") for r in cluster_reports if r.get("photo_filename")]
            combined["primary_photo"] = existing[0] if existing else (rep_photos[0] if rep_photos else "sample_bottles_drain.jpg")
            
            scored_hotspots.append(combined)

        # Sort ranked by risk score descending
        scored_hotspots.sort(key=lambda h: h["total_score"], reverse=True)

        # Renumber IDs cleanly by priority
        for idx, h in enumerate(scored_hotspots, 1):
            h["id"] = f"HOTSPOT-{idx:02d}"
            h["hotspot_id"] = f"HOTSPOT-{idx:02d}"
            h["score"] = h["total_score"]

        # Summary KPIs
        total_reports = len(raw_reports)
        total_hotspots = len(scored_hotspots)
        merged_hotspots = sum(1 for h in scored_hotspots if h.get("is_merged"))
        critical_count = sum(1 for h in scored_hotspots if h["priority"] == "CRITICAL")
        high_count = sum(1 for h in scored_hotspots if h["priority"] == "HIGH")
        moderate_count = sum(1 for h in scored_hotspots if h["priority"] == "MODERATE")
        drain_choke_risks = sum(1 for h in scored_hotspots if h["nearest_drain_distance_m"] <= 50.0)
        mean_conf = sum(h["confidence_score"] for h in scored_hotspots) / max(1, total_hotspots) / 100.0
        current_rain = scored_hotspots[0]["rainfall_forecast_mm"] if scored_hotspots else 16.5

        return {
            "challenge_id": "PS-08",
            "area": "Pune Municipal Corporation (PMC) - Smart Drain Pilot",
            "mode": "live_urban_intelligence",
            "mode_label": "Urban Civic Environmental Intelligence Engine",
            "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "rainfall_forecast_24h_mm": current_rain,
            "metrics": {
                "active_hotspots": total_hotspots,
                "merged_clusters": merged_hotspots,
                "citizen_reports": total_reports,
                "drain_proximity_alerts": drain_choke_risks,
                "critical_sites": critical_count,
                "high_priority_sites": high_count,
                "moderate_sites": moderate_count,
                "mean_detection_confidence": round(mean_conf, 2),
                "pending_review_reports": len(review_queue),
                "compression_ratio_pct": cluster_telemetry["compression_ratio_pct"],
                "prevented_duplicate_dispatches": cluster_telemetry["prevented_duplicate_dispatches"],
                "estimated_civic_savings_inr": cluster_telemetry["estimated_civic_savings_inr"],
            },
            "environment": {
                "rainfall_intensity_mmh": max(0.0, min(75.0, float(rainfall_intensity_mmh))),
                "rainfall_source": rain_source,
                "drain_silt_ratio": None if drain_silt_ratio is None else max(0.0, min(1.0, float(drain_silt_ratio))),
                "drain_silt_source": "per-drain desilting records" if drain_silt_ratio is None else "scenario override",
                "silt_range": [
                    min((h["drain_silt_ratio"] for h in scored_hotspots), default=0.0),
                    max((h["drain_silt_ratio"] for h in scored_hotspots), default=0.0),
                ],
                "surface_slope_factor": max(1.0, min(1.5, float(surface_slope_factor))),
            },
            "cluster_telemetry": cluster_telemetry,
            "hotspots": scored_hotspots,
            "reports": raw_reports,
            "review_queue": review_queue,
            "formula": (
                "0.35 * Drain Proximity x Live Env (rain intensity, siltation, slope) + "
                "0.25 * Detection Severity (1-5) + 0.20 * TACO Confidence + "
                "0.20 * Open-Meteo Rain Forecast (24h) + Recurrence"
            ),
        }

    def dispatch_work_order(self, hotspot_ids: List[str], **pipeline_kwargs: Any) -> Dict[str, Any]:
        """Dispatch municipal truck route for selected hotspots.

        Pass the same environment kwargs used for ranking (rainfall_intensity_mmh, ...)
        so HOTSPOT-NN ids resolve to the sites the operator selected.
        """
        pipeline_data = self.run_pipeline(**pipeline_kwargs)
        target_hotspots = [h for h in pipeline_data["hotspots"] if h["id"] in hotspot_ids or h.get("hotspot_id") in hotspot_ids]
        
        if not target_hotspots:
            target_hotspots = pipeline_data["hotspots"][:3]  # Default to top critical hotspots

        plan = self.route_optimizer.generate_dispatch_plan(target_hotspots)
        return plan


# Global pipeline instance
_pipeline = PlasticWatchUrbanPipeline()


def run_plasticwatch(
    mode: str = "replay",
    rainfall_override_mm: Optional[float] = None,
    rainfall_intensity_mmh: float = 0.0,
    drain_silt_ratio: float = 0.0,
    surface_slope_factor: float = 1.0,
) -> Dict[str, Any]:
    """Compatibility runner matching dashboard invocation signature."""
    return _pipeline.run_pipeline(
        rainfall_override_mm=rainfall_override_mm,
        rainfall_intensity_mmh=rainfall_intensity_mmh,
        drain_silt_ratio=drain_silt_ratio,
        surface_slope_factor=surface_slope_factor,
    )
