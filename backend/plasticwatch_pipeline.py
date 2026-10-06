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
from backend.route_optimizer import RouteOptimizer
from backend.taco_adapter import detect_waste, get_sample_images

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DB_PATH = ROOT / "data" / "citizen_reports.json"


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

    def get_reports(self) -> List[Dict[str, Any]]:
        """Retrieve all active citizen reports."""
        if self._reports_cache is not None:
            return self._reports_cache

        if REPORTS_DB_PATH.is_file():
            try:
                data = json.loads(REPORTS_DB_PATH.read_text(encoding="utf-8"))
                self._reports_cache = data
                return self._reports_cache
            except Exception as e:
                logger.error("Failed to load reports DB: %s", e)

        # Initialize with seed reports
        seeds = [r.to_dict() for r in get_seed_reports()]
        self.save_reports(seeds)
        return seeds

    def save_reports(self, reports: List[Dict[str, Any]]) -> None:
        """Persist reports to disk."""
        self._reports_cache = reports
        try:
            REPORTS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            REPORTS_DB_PATH.write_text(json.dumps(reports, indent=2), encoding="utf-8")
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
        reports = self.get_reports()
        next_id = f"CR-{len(reports) + 201}"

        # Run AI detection
        if image_bytes:
            detection = detect_waste(image_bytes, filename_hint=photo_filename)
        else:
            sample_path = ROOT / "data" / "taco_samples" / photo_filename
            if sample_path.is_file():
                detection = detect_waste(str(sample_path), filename_hint=photo_filename)
            else:
                detection = {
                    "mean_confidence": 0.88,
                    "severity": 3.8,
                    "detected_items": [{"class": "plastic_bottle"}, {"class": "plastic_bag_wrapper"}],
                    "hazard_level": "HIGH",
                }

        classes = [item.get("class", "plastic_bottle") for item in detection.get("detected_items", [])]

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
            verification="pending",
            notes=notes or "Citizen mobile upload via PWA/API Feeder",
        ).to_dict()

        reports.append(new_rep)
        self.save_reports(reports)

        return {
            "status": "success",
            "report": new_rep,
            "detection": detection,
            "message": f"Report {next_id} successfully submitted and indexed for DBSCAN clustering.",
        }

    def run_pipeline(self, rainfall_override_mm: Optional[float] = None) -> Dict[str, Any]:
        """Run the end-to-end urban intelligence pipeline."""
        raw_reports = self.get_reports()

        # Step 1: DBSCAN Spatial Clustering (eps=50m)
        hotspot_clusters = self.clustering_engine.cluster_reports(raw_reports)

        # Step 2: Urban Drain Proximity & Context Risk Scoring (0-100)
        scored_hotspots = []
        for cluster in hotspot_clusters:
            risk = self.risk_engine.compute_risk(cluster, rainfall_override_mm=rainfall_override_mm)
            
            # Combine cluster info with risk breakdown
            combined = {**cluster, **risk}
            # Verification status of hotspot
            verified_count = sum(r.get("verification") == "verified" for r in cluster.get("reports", []))
            combined["verified_count"] = verified_count
            combined["dispatch_status"] = "PENDING"
            
            # Primary sample photo
            rep_photos = [r.get("photo_filename") for r in cluster.get("reports", []) if r.get("photo_filename")]
            combined["primary_photo"] = rep_photos[0] if rep_photos else "sample_bottles_drain.jpg"
            
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
            },
            "hotspots": scored_hotspots,
            "reports": raw_reports,
            "formula": (
                "0.35 * Drain Proximity (m) + 0.25 * Detection Severity (1-5) + "
                "0.20 * TACO Confidence + 0.20 * Open-Meteo Rain Forecast (24h) + Recurrence"
            ),
        }

    def dispatch_work_order(self, hotspot_ids: List[str]) -> Dict[str, Any]:
        """Dispatch municipal truck route for selected hotspots."""
        pipeline_data = self.run_pipeline()
        target_hotspots = [h for h in pipeline_data["hotspots"] if h["id"] in hotspot_ids or h.get("hotspot_id") in hotspot_ids]
        
        if not target_hotspots:
            target_hotspots = pipeline_data["hotspots"][:3]  # Default to top critical hotspots

        plan = self.route_optimizer.generate_dispatch_plan(target_hotspots)
        return plan


# Global pipeline instance
_pipeline = PlasticWatchUrbanPipeline()


def run_plasticwatch(mode: str = "replay", rainfall_override_mm: Optional[float] = None) -> Dict[str, Any]:
    """Compatibility runner matching dashboard invocation signature."""
    return _pipeline.run_pipeline(rainfall_override_mm=rainfall_override_mm)
