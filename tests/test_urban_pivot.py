"""Unit and Integration Tests for PlasticWatch Urban Intelligence Pivot.

Verifies:
  1. DBSCAN Spatial Clustering (50m duplicate pin consolidation)
  2. Urban Drain Proximity & Hydrological Risk Scoring (0-100 Formula)
  3. TACO Waste Object Detection & Severity Estimation
  4. Municipal Truck Route Optimizer (TSP & Dispatch Plan)
  5. End-to-end Urban Pipeline & FastAPI Endpoints
"""
import json
from pathlib import Path
import unittest

from backend.clustering_engine import DBSCANClusteringEngine
from backend.drain_risk_engine import DrainRiskEngine
from backend.plasticwatch_pipeline import PlasticWatchUrbanPipeline, run_plasticwatch
from backend.route_optimizer import RouteOptimizer
from backend.taco_adapter import dataset_status, detect_waste, get_sample_images


class TestUrbanClusteringEngine(unittest.TestCase):
    def setUp(self):
        self.engine = DBSCANClusteringEngine(eps_meters=50.0, min_samples=2)

    def test_duplicate_reports_merge_within_50m(self):
        # 3 points within 35 meters of each other (Pune Shaniwar Wada)
        reports = [
            {"id": "R-1", "lat": 18.5204, "lon": 73.8568, "confidence": 0.92, "severity": 4.5},
            {"id": "R-2", "lat": 18.5206, "lon": 73.8569, "confidence": 0.90, "severity": 4.0},
            {"id": "R-3", "lat": 18.5205, "lon": 73.8567, "confidence": 0.94, "severity": 4.8},
            # 1 distant point 1.2km away
            {"id": "R-4", "lat": 18.5350, "lon": 73.8920, "confidence": 0.85, "severity": 3.0},
        ]
        hotspots = self.engine.cluster_reports(reports)
        self.assertEqual(len(hotspots), 2)

        # First hotspot must be merged with 3 reports
        merged = next(h for h in hotspots if h["is_merged"])
        self.assertEqual(merged["recurrence"], 3)
        self.assertCountEqual(merged["report_ids"], ["R-1", "R-2", "R-3"])
        self.assertAlmostEqual(merged["confidence"], 0.92, places=2)

        # Second hotspot is isolated single report
        single = next(h for h in hotspots if not h["is_merged"])
        self.assertEqual(single["recurrence"], 1)
        self.assertEqual(single["report_ids"], ["R-4"])

    def test_empty_reports_returns_empty(self):
        self.assertEqual(self.engine.cluster_reports([]), [])


class TestDrainRiskEngine(unittest.TestCase):
    def setUp(self):
        self.risk_engine = DrainRiskEngine()

    def test_nearest_drain_distance_and_scoring(self):
        # Test point right on Shaniwar Wada Stormwater Trunk (18.5204, 73.8568)
        hotspot = {
            "hotspot_id": "HS-TEST",
            "lat": 18.5204,
            "lon": 73.8568,
            "confidence": 0.90,
            "severity": 4.5,
            "recurrence": 3,
        }
        risk = self.risk_engine.compute_risk(hotspot, rainfall_override_mm=18.0)

        # Proximity checks
        self.assertLess(risk["nearest_drain_distance_m"], 25.0)
        self.assertIn("nearest_drain_name", risk)
        
        # 0-100 Score checks
        self.assertGreaterEqual(risk["total_score"], 70.0)
        self.assertEqual(risk["priority"], "CRITICAL")
        self.assertEqual(risk["priority_color"], "#ef4444")

        # Formula component breakdown weights check:
        # Drain proximity = 35% weight
        self.assertAlmostEqual(risk["drain_points"], risk["drain_score"] * 0.35, places=1)
        # Severity = 25% weight
        self.assertAlmostEqual(risk["severity_points"], risk["severity_score"] * 0.25, places=1)
        # Confidence = 20% weight
        self.assertAlmostEqual(risk["confidence_points"], risk["confidence_score"] * 0.20, places=1)
        # Rain = 20% weight
        self.assertAlmostEqual(risk["rainfall_points"], risk["rainfall_score"] * 0.20, places=1)

    def test_distant_low_waste_has_moderate_priority(self):
        # Point far from any primary drain, low severity
        hotspot = {
            "hotspot_id": "HS-DISTANT",
            "lat": 18.5600,
            "lon": 73.9500,
            "confidence": 0.70,
            "severity": 1.5,
            "recurrence": 1,
        }
        risk = self.risk_engine.compute_risk(hotspot, rainfall_override_mm=0.0)
        self.assertLess(risk["total_score"], 50.0)
        self.assertEqual(risk["priority"], "MODERATE")


class TestTacoWasteDetection(unittest.TestCase):
    def test_dataset_provenance_facts(self):
        status = dataset_status()
        self.assertTrue(status["available"])
        self.assertIn("plastic_bottle", status["target_classes"])
        self.assertIn("plastic_bag_wrapper", status["target_classes"])
        self.assertGreaterEqual(len(status["target_classes"]), 5)

    def test_detection_on_sample_images(self):
        samples = {s["filename"]: s for s in get_sample_images()}
        self.assertGreater(len(samples), 0)

        # A pile of PET bottles: YOLO proposes bottles and the material check confirms plastic
        bottles = samples["sample_pet_bottle_pile.jpg"]
        result = detect_waste(bottles["path"], filename_hint=bottles["filename"])
        self.assertEqual(result["status"], "success")
        self.assertGreater(len(result["detected_items"]), 0)
        self.assertTrue(all(c == "plastic_bottle" for c in result["detected_classes"]))
        self.assertEqual(result["evidence_level"], "model")
        self.assertGreater(result["mean_confidence"], 0.5)
        self.assertGreater(result["severity"], 2.0)
        self.assertIn("annotated_image_base64", result)
        self.assertGreater(len(result["annotated_image_base64"]), 1000)

        # Furniture/people proposed by COCO are scene context, never counted as plastic
        cans = samples["sample_beverage_cans_gutter.jpg"]
        result = detect_waste(cans["path"], filename_hint=cans["filename"])
        ignored = [t for t in result["trace"] if t["decision"] == "ignored"]
        self.assertTrue(ignored)
        self.assertNotIn("bed", [b.get("coco_class") for b in result["detected_items"]])


class TestMunicipalRouteOptimizer(unittest.TestCase):
    def setUp(self):
        self.optimizer = RouteOptimizer()

    def test_dispatch_plan_generation(self):
        hotspots = [
            {"hotspot_id": "HS-01", "lat": 18.5204, "lon": 73.8568, "priority": "CRITICAL", "total_score": 88},
            {"hotspot_id": "HS-02", "lat": 18.5060, "lon": 73.8625, "priority": "CRITICAL", "total_score": 82},
        ]
        plan = self.optimizer.generate_dispatch_plan(hotspots)
        self.assertEqual(plan["status"], "dispatched")
        self.assertEqual(plan["total_stops"], 2)
        self.assertGreater(plan["total_distance_km"], 0.0)
        self.assertGreater(plan["total_duration_minutes"], 0)
        # Polyline starts and ends at Depot: len = stops + 2
        self.assertEqual(len(plan["route_polyline"]), 4)


class TestUrbanPipelineEndToEnd(unittest.TestCase):
    def test_full_pipeline_execution(self):
        result = run_plasticwatch(rainfall_override_mm=16.5)
        self.assertEqual(result["area"], "Pune Municipal Corporation (PMC) - Smart Drain Pilot")
        self.assertGreater(len(result["hotspots"]), 0)
        self.assertGreater(result["metrics"]["merged_clusters"], 0)
        self.assertGreater(result["metrics"]["citizen_reports"], 5)

        # Check ranking order (descending by score)
        scores = [h["total_score"] for h in result["hotspots"]]
        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main()
