"""Unit and Integration Tests for PlasticWatch Advanced Municipal Features.

Validates that:
  1. ForecastingEngine produces non-null, valid bounded outputs (24-72h, kg, time window, actions).
  2. CleanupVerificationEngine computes delta severity, count reduction, effectiveness score (0-100%),
     diverted kg, PASS/FLAGGED decision status, and disclaimer.
  3. RecurrenceEngine computes recurrence counts, return intervals, persistence indices, dominant
     waste streams, automated root-cause hypotheses, and civic interventions.
  4. MonsoonSimulatorEngine models outcomes monotonically across rainfall intensity, fleet, and delay.
  5. ProvenanceLayer generates standard 4-tier auditable badges.
  6. FastAPI endpoints (/api/forecast, /api/verify-cleanup, /api/recurrence, /api/simulate,
     /api/dossier, /api/provenance) return status 200 with valid schemas.
"""
from __future__ import annotations

import unittest
from fastapi.testclient import TestClient

from backend.api import app
from backend.forecasting_engine import ForecastingEngine
from backend.provenance import PROVENANCE_TIERS, get_provenance_meta, render_provenance_badge
from backend.recurrence_engine import RecurrenceEngine
from backend.simulator_engine import MonsoonSimulatorEngine
from backend.verification_engine import CleanupVerificationEngine


class TestForecastingEngine(unittest.TestCase):
    def setUp(self):
        self.engine = ForecastingEngine()
        self.sample_hotspots = [
            {
                "id": "HOTSPOT-01",
                "lat": 18.5204,
                "lon": 73.8568,
                "severity": 4.6,
                "nearest_drain_distance_m": 18.5,
                "nearest_drain_name": "Shaniwar Wada Stormwater Trunk",
                "total_score": 88.0,
                "recurrence": 3,
            },
            {
                "id": "HOTSPOT-02",
                "lat": 18.5060,
                "lon": 73.8625,
                "severity": 3.8,
                "nearest_drain_distance_m": 42.0,
                "nearest_drain_name": "Nagzari Nallah Corridor",
                "total_score": 62.0,
                "recurrence": 1,
            },
        ]

    def test_forecast_outputs_bounded_for_all_horizons(self):
        for horizon in [24, 48, 72]:
            result = self.engine.generate_hotspot_forecast(self.sample_hotspots, horizon_hours=horizon)
            self.assertEqual(result["forecast_horizon_hours"], horizon)
            self.assertGreater(result["total_predicted_waste_kg"], 0.0)
            self.assertEqual(len(result["hotspot_forecasts"]), 2)
            self.assertEqual(result["provenance_tag"], "[Predicted / Modelled]")

            for hf in result["hotspot_forecasts"]:
                self.assertIn(hf["escalation_risk"], ["CRITICAL", "HIGH", "MODERATE"])
                self.assertGreater(hf["predicted_accumulation_kg"], 0.0)
                self.assertGreater(hf["predicted_volume_liters"], 0.0)
                self.assertTrue(hf["critical_time_window"].startswith("T+"))
                self.assertGreater(len(hf["preventive_actions"]), 0)
                self.assertIn("24h", hf["timeline"])
                self.assertIn("48h", hf["timeline"])
                self.assertIn("72h", hf["timeline"])

    def test_forecast_accumulation_increases_with_horizon(self):
        fc24 = self.engine.generate_hotspot_forecast(self.sample_hotspots, horizon_hours=24)
        fc72 = self.engine.generate_hotspot_forecast(self.sample_hotspots, horizon_hours=72)
        kg24 = fc24["total_predicted_waste_kg"]
        kg72 = fc72["total_predicted_waste_kg"]
        self.assertGreater(kg72, kg24)

    def test_custom_rainfall_override(self):
        dry = self.engine.generate_hotspot_forecast(self.sample_hotspots, horizon_hours=48, rainfall_scenario_mm=0.0)
        heavy = self.engine.generate_hotspot_forecast(self.sample_hotspots, horizon_hours=48, rainfall_scenario_mm=60.0)
        self.assertGreater(heavy["total_predicted_waste_kg"], dry["total_predicted_waste_kg"])


class TestCleanupVerificationEngine(unittest.TestCase):
    def setUp(self):
        self.engine = CleanupVerificationEngine()

    def test_verified_clearance_yields_pass(self):
        result = self.engine.verify_cleanup(
            before_image="sample_bottles_drain.jpg",
            after_image="sample_cleared_drain.jpg",
            hotspot_id="HOTSPOT-01",
            site_name="Shaniwar Wada Culvert",
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["decision_status"], "PASS")
        self.assertTrue(result["invoice_eligible"])
        self.assertGreaterEqual(result["effectiveness_score"], 85.0)
        self.assertLessEqual(result["effectiveness_score"], 100.0)
        self.assertGreater(result["metrics"]["diverted_plastic_weight_kg"], 0.0)
        self.assertGreater(result["metrics"]["count_reduction"], 0)
        self.assertGreater(result["metrics"]["delta_severity"], 0.0)
        self.assertIn("Operator sign-off required", result["disclaimer"])
        self.assertIn("before_annotated_base64", result["visuals"])
        self.assertIn("after_annotated_base64", result["visuals"])
        self.assertEqual(result["provenance_badge"], "[AI Inference / Modelled Estimate]")

    def test_partial_clearance_yields_flagged(self):
        result = self.engine.verify_cleanup(
            before_image="sample_bottles_drain.jpg",
            after_image="sample_partial_cleanup.jpg",
            hotspot_id="HOTSPOT-01",
        )
        self.assertEqual(result["decision_status"], "FLAGGED FOR REVIEW")
        self.assertFalse(result["invoice_eligible"])
        self.assertLess(result["effectiveness_score"], 85.0)
        self.assertGreaterEqual(result["effectiveness_score"], 0.0)


class TestRecurrenceEngine(unittest.TestCase):
    def setUp(self):
        self.engine = RecurrenceEngine()

    def test_recurrence_analysis_structure(self):
        report = self.engine.analyze_recurrence()
        self.assertIn("Pune Municipal Corporation", report["municipality"])
        self.assertEqual(len(report["locations"]), 4)
        summary = report["summary_metrics"]
        self.assertGreater(summary["total_cleanups_completed"], 15)
        self.assertGreater(summary["total_diverted_plastic_kg"], 500.0)
        self.assertGreater(summary["average_return_interval_days"], 0.0)
        self.assertLess(summary["average_return_interval_days"], 15.0)

    def test_location_dossier_fields(self):
        report = self.engine.analyze_recurrence()
        for loc in report["locations"]:
            self.assertGreater(loc["total_recurrence_count"], 0)
            self.assertGreater(loc["average_return_interval_days"], 0.0)
            self.assertGreaterEqual(loc["recurrence_persistence_index"], 0.0)
            self.assertLessEqual(loc["recurrence_persistence_index"], 1.0)
            self.assertIn(loc["persistence_level"], ["CHRONIC", "ELEVATED", "MODERATE"])
            self.assertGreater(len(loc["dominant_waste_stream"]), 3)
            self.assertGreater(len(loc["root_cause_hypothesis"]), 15)
            self.assertGreater(len(loc["systemic_interventions"]), 1)
            self.assertGreater(len(loc["sparkline_data"]), 0)
            self.assertGreater(len(loc["cleanups"]), 0)
            self.assertEqual(loc["provenance_badge"], "[Inferred / Prototype Historical Dossier]")


class TestMonsoonSimulatorEngine(unittest.TestCase):
    def setUp(self):
        self.simulator = MonsoonSimulatorEngine()

    def test_dry_scenario_has_zero_inundation(self):
        res = self.simulator.simulate(rainfall_intensity_mmh=0.0, available_fleet=5, action_delay_hours=0.0)
        outcomes = res["outcomes"]
        self.assertEqual(outcomes["potential_flood_inundation_area_sqm"], 0.0)
        self.assertEqual(outcomes["plastic_swept_into_rivers_kg"], 0.0)
        self.assertEqual(res["provenance_badge"], "[Simulated / Modelled Estimate]")

    def test_heavy_monsoon_increases_risk_monotonically(self):
        moderate = self.simulator.simulate(rainfall_intensity_mmh=20.0, available_fleet=4, action_delay_hours=2.0)
        severe = self.simulator.simulate(rainfall_intensity_mmh=60.0, available_fleet=4, action_delay_hours=2.0)

        self.assertGreater(
            severe["outcomes"]["plastic_mass_at_risk_kg"],
            moderate["outcomes"]["plastic_mass_at_risk_kg"],
        )
        self.assertGreater(
            severe["outcomes"]["potential_flood_inundation_area_sqm"],
            moderate["outcomes"]["potential_flood_inundation_area_sqm"],
        )
        self.assertGreater(
            severe["outcomes"]["estimated_civic_cost_of_inaction_inr"],
            moderate["outcomes"]["estimated_civic_cost_of_inaction_inr"],
        )

    def test_fleet_intervention_reduces_river_plastic(self):
        low_fleet = self.simulator.simulate(rainfall_intensity_mmh=35.0, available_fleet=1, action_delay_hours=2.0)
        high_fleet = self.simulator.simulate(rainfall_intensity_mmh=35.0, available_fleet=10, action_delay_hours=2.0)

        self.assertGreater(
            low_fleet["outcomes"]["plastic_swept_into_rivers_kg"],
            high_fleet["outcomes"]["plastic_swept_into_rivers_kg"],
        )
        self.assertGreater(
            high_fleet["outcomes"]["plastic_cleared_by_fleet_kg"],
            low_fleet["outcomes"]["plastic_cleared_by_fleet_kg"],
        )

    def test_delay_escalates_costs(self):
        immediate = self.simulator.simulate(rainfall_intensity_mmh=35.0, available_fleet=4, action_delay_hours=0.0)
        delayed = self.simulator.simulate(rainfall_intensity_mmh=35.0, available_fleet=4, action_delay_hours=12.0)

        self.assertGreater(
            delayed["outcomes"]["estimated_civic_cost_of_inaction_inr"],
            immediate["outcomes"]["estimated_civic_cost_of_inaction_inr"],
        )


class TestProvenanceLayer(unittest.TestCase):
    def test_all_tiers_available(self):
        for tier in ["observed", "derived", "inferred", "simulated"]:
            self.assertIn(tier, PROVENANCE_TIERS)
            meta = get_provenance_meta(tier)
            self.assertIn("emoji", meta)
            self.assertIn("color", meta)
            badge_html = render_provenance_badge(tier)
            self.assertIn(meta["emoji"], badge_html)


class TestFastAPIEndpointsNewFeatures(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_forecast_api(self):
        # GET
        res = self.client.get("/api/forecast?horizon_hours=48")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["forecast_horizon_hours"], 48)
        self.assertGreater(len(data["hotspot_forecasts"]), 0)

        # POST
        res_post = self.client.post("/api/forecast", json={"horizon_hours": 72, "rainfall_scenario_mm": 40.0})
        self.assertEqual(res_post.status_code, 200)
        self.assertEqual(res_post.json()["forecast_horizon_hours"], 72)

    def test_verify_cleanup_api(self):
        payload = {
            "before_image": "sample_bottles_drain.jpg",
            "after_image": "sample_cleared_drain.jpg",
            "hotspot_id": "HOTSPOT-01",
        }
        res = self.client.post("/api/verify-cleanup", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["decision_status"], "PASS")
        self.assertGreaterEqual(data["effectiveness_score"], 85.0)

    def test_recurrence_api(self):
        res = self.client.get("/api/recurrence")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["locations"]), 4)

    def test_simulate_api(self):
        # GET
        res = self.client.get("/api/simulate?rainfall_intensity=30.0&fleet=4&delay_hours=2.0")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("outcomes", data)
        self.assertGreater(data["outcomes"]["estimated_civic_cost_of_inaction_inr"], 0)

        # POST
        res_post = self.client.post("/api/simulate", json={
            "rainfall_intensity_mmh": 40.0,
            "available_fleet": 5,
            "action_delay_hours": 3.0,
        })
        self.assertEqual(res_post.status_code, 200)

    def test_dossier_api(self):
        res = self.client.get("/api/dossier")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("locations", data)
        self.assertEqual(len(data["locations"]), 4)

    def test_provenance_api(self):
        res = self.client.get("/api/provenance")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("tiers", data)


if __name__ == "__main__":
    unittest.main()
