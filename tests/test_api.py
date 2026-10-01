"""Integration tests for PlasticWatch FastAPI endpoints."""
import unittest
from fastapi.testclient import TestClient
from backend.api import app


class TestFastAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_mobile_simulator_endpoint(self):
        response = self.client.get("/mobile")
        self.assertEqual(response.status_code, 200)
        self.assertIn("PlasticWatch Civic", response.text)
        self.assertIn("Submit Citizen Report", response.text)

    def test_get_reports(self):
        response = self.client.get("/api/reports")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertGreater(data["count"], 0)

    def test_get_hotspots(self):
        response = self.client.get("/api/hotspots?rain_override_mm=16.5")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("hotspots", data)
        self.assertGreater(len(data["hotspots"]), 0)
        self.assertIn("metrics", data)

    def test_get_drains_geojson(self):
        response = self.client.get("/api/drains")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["type"], "FeatureCollection")
        self.assertGreater(len(data["features"]), 0)

    def test_dispatch_work_order(self):
        payload = {
            "hotspot_ids": ["HOTSPOT-01", "HOTSPOT-02"],
            "truck_id": "PMC-TRUCK-09",
            "crew_name": "Sanitation Rapid Crew 4"
        }
        response = self.client.post("/api/dispatch", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "dispatched")
        self.assertEqual(data["truck_id"], "PMC-TRUCK-09")
        self.assertGreater(data["total_distance_km"], 0.0)

    def test_submit_citizen_report(self):
        payload = {
            "lat": 18.5204,
            "lon": 73.8568,
            "photo_filename": "sample_bottles_drain.jpg",
            "notes": "Test report from test suite"
        }
        response = self.client.post("/api/reports", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("CR-", data["report"]["id"])


if __name__ == "__main__":
    unittest.main()
