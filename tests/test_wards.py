"""Ward Drain Score tests: ward zones, score components, re-weighting and the API.

Data files are redirected to a temporary directory (see IncentiveTestCase).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import random

from backend import ward_engine as we
from tests.test_incentives import IncentiveTestCase

NOW = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)
SHANIWAR = (18.5204, 73.8568)   # Kasba Peth zone


def _cleanup(days_ago: float, status: str = "PASS", hours_to_clear: float = 24.0):
    at = NOW - timedelta(days=days_ago)
    return {"timestamp": at.isoformat(), "verification_status": status,
            "first_reported_at": (at - timedelta(hours=hours_to_clear)).isoformat()}


class TestWardZones(IncentiveTestCase):
    def test_polygons_match_assignment_rule(self):
        from shapely.geometry import Point, shape

        polys = we.ward_polygons()["features"]
        self.assertEqual(sorted(f["properties"]["ward"] for f in polys), sorted(we.ward_names()))
        rng = random.Random(7)
        for _ in range(200):
            lat, lon = rng.uniform(18.46, 18.67), rng.uniform(73.75, 73.95)
            owners = [f["properties"]["ward"] for f in polys if shape(f["geometry"]).covers(Point(lon, lat))]
            if owners:
                self.assertEqual(owners[0], we.assign_ward(lat, lon))


class TestScorecard(IncentiveTestCase):
    def setUp(self):
        super().setUp()
        self.dossier = self.tmp / "dossier.json"
        self.dossier.write_text(json.dumps({"locations": [{
            "site_id": "LOC-T1", "lat": SHANIWAR[0], "lon": SHANIWAR[1],
            "cleanups": [_cleanup(5), _cleanup(20, "FLAGGED FOR REVIEW"),             # last 30 days: 2 clogs
                         _cleanup(35), _cleanup(40), _cleanup(45), _cleanup(50),     # previous 30 days: 4 clogs
                         _cleanup(200)],                                             # outside every window
        }]}))

    def card(self, reports=(), rewards=(), outcomes=(), days=30):
        sc = we.ward_scorecard(list(reports), list(rewards), list(outcomes), days=days, now=NOW, history_path=self.dossier)
        return {w["ward"]: w for w in sc["wards"]}, sc

    def test_components_and_weighted_score(self):
        wards, _ = self.card()
        kasba = wards["Kasba Peth"]
        self.assertAlmostEqual(kasba["verified_clear_rate_pct"], 83.3, places=1)   # 5 of 6 in 90 days
        self.assertEqual(kasba["median_hours_to_clear"], 24.0)
        self.assertEqual(kasba["components"]["speed"], 100.0)
        self.assertEqual((kasba["clogs_last_30d"], kasba["clogs_prev_30d"]), (2, 4))
        self.assertEqual(kasba["recurrence_trend_pct"], -50.0)
        self.assertEqual(kasba["components"]["recurrence"], 80.0)                   # 2 clogs/site → 80
        self.assertEqual(kasba["components"]["coverage"], 0.0)
        # 0.3*0 + 0.3*83.33 + 0.2*100 + 0.2*80
        self.assertAlmostEqual(kasba["score"], 61.0, places=1)
        self.assertEqual(kasba["rank"], 1)

    def test_wards_without_enough_data_are_not_ranked(self):
        wards, sc = self.card()
        self.assertIsNone(wards["Hadapsar"]["score"])   # coverage only
        self.assertIsNone(wards["Akurdi"]["rank"])      # no mapped drains, no history
        self.assertEqual(sc["wards"][0]["ward"], "Kasba Peth")

    def test_coverage_and_reporters_raise_the_score(self):
        report = {"id": "CR-1", "lat": SHANIWAR[0], "lon": SHANIWAR[1], "reporter_id": "RP-0001",
                  "submitted_at": (NOW - timedelta(days=1)).isoformat()}
        with_report, _ = self.card(reports=[report])
        without, _ = self.card()
        self.assertGreater(with_report["Kasba Peth"]["coverage_pct"], 0)
        self.assertGreater(with_report["Kasba Peth"]["score"], without["Kasba Peth"]["score"])
        self.assertEqual(with_report["Kasba Peth"]["active_reporters"], 1)
        stale, _ = self.card(reports=[{**report, "submitted_at": "2026-01-01T00:00:00+00:00"}])
        self.assertEqual(stale["Kasba Peth"]["reports"], 0)  # outside the 30-day window

    def test_live_verification_and_release_times_count(self):
        outcomes = [{"case_id": "VC-1", "lat": SHANIWAR[0], "lon": SHANIWAR[1], "decision": "FLAGGED FOR REVIEW"},
                    {"case_id": "VC-2", "lat": SHANIWAR[0], "lon": SHANIWAR[1], "decision": "INCONCLUSIVE"}]
        rewards = [{"state": "RELEASED", "role": "first_reporter", "report_id": "CR-9", "ward": "Kasba Peth",
                    "lat": SHANIWAR[0], "lon": SHANIWAR[1],
                    "report_submitted_at": (NOW - timedelta(hours=10)).isoformat(), "released_at": NOW.isoformat()}]
        wards, _ = self.card(rewards=rewards, outcomes=outcomes)
        kasba = wards["Kasba Peth"]
        self.assertEqual(kasba["cleanups_judged"], 7)                    # 6 history + 1 live (INCONCLUSIVE skipped)
        self.assertAlmostEqual(kasba["verified_clear_rate_pct"], 71.4, places=1)
        self.assertEqual(kasba["median_hours_to_clear"], 24.0)           # 10 h joins six 24 h values

    def test_missing_history_file_is_tolerated(self):
        sc = we.ward_scorecard([], history_path=self.tmp / "missing.json", now=NOW)
        self.assertTrue(all(w["score"] is None for w in sc["wards"]))


class TestWardApi(IncentiveTestCase):
    def test_scorecard_and_geojson_endpoints(self):
        from fastapi.testclient import TestClient
        from backend import api

        client = TestClient(api.app)
        sc = client.get("/api/wards/scorecard?days=30")
        self.assertEqual(sc.status_code, 200)
        self.assertEqual(len(sc.json()["wards"]), len(we.ward_names()))
        self.assertEqual(client.get("/api/wards/scorecard?days=0").status_code, 422)
        self.assertEqual(len(client.get("/api/wards/geojson").json()["features"]), len(we.ward_names()))


if __name__ == "__main__":
    import unittest

    unittest.main()
