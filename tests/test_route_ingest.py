"""Dashcam route ingestion tests: frame filters, replay, dedup and the ingest API.

Data files are redirected to a temporary directory (see IncentiveTestCase).
"""
from __future__ import annotations

import base64
from unittest import mock

from backend import incentive_engine as ie
from backend import plasticwatch_pipeline as pipeline_mod
from backend import route_ingest_engine as rie
from backend.recurrence_engine import haversine_m
from tests.test_incentives import IncentiveTestCase

ON_DRAIN = (18.5060, 73.8627)    # Nagzari nallah road
OFF_DRAIN = (18.5196, 73.8610)   # between Nagzari and Shaniwar Wada lines
WASTE = {"waste_detected": True, "mean_confidence": 0.6, "needs_review": False, "severity": 3.0,
         "item_count": 2, "detected_items": [{"class": "plastic_bottle"}]}


class TestReplayGeometry(IncentiveTestCase):
    def test_interpolation_spacing(self):
        pts = rie._interpolate([(18.50, 73.86), (18.51, 73.86), (18.51, 73.87)], 200.0)
        gaps = [haversine_m(*a, *b) for a, b in zip(pts, pts[1:])]
        straight = [g for g in gaps[:-1] if abs(g - 200.0) < 1.0]
        self.assertGreaterEqual(len(straight), len(gaps) - 3)  # only the corner gap is shorter
        self.assertEqual(pts[0], (18.50, 73.86))

    def test_replay_frames_are_deterministic_and_ordered(self):
        a, b = rie.replay_frames("PMPML-158"), rie.replay_frames("PMPML-158")
        self.assertEqual([f.sample_filename for f in a], [f.sample_filename for f in b])
        self.assertEqual(sorted(f.captured_at for f in a), [f.captured_at for f in a])
        with self.assertRaises(KeyError):
            rie.replay_frames("NOPE")


class TestFrameFilters(IncentiveTestCase):
    def setUp(self):
        super().setUp()
        self.pipe = pipeline_mod.PlasticWatchUrbanPipeline()

    def ingest(self, frames, detection=WASTE, side_effect=None):
        with mock.patch.object(rie, "detect_waste", return_value=detection, side_effect=side_effect) as det:
            return rie.ingest_frames(self.pipe, "PMPML-BUS-158", frames), det

    def frame(self, lat_lon=ON_DRAIN):
        return rie.Frame(lat=lat_lon[0], lon=lat_lon[1], sample_filename="sample_bottles_drain.jpg")

    def test_far_frames_never_reach_the_detector(self):
        res, det = self.ingest([self.frame(OFF_DRAIN)])
        self.assertEqual(res["decisions"][rie.SKIP_FAR], 1)
        det.assert_not_called()

    def test_each_filter(self):
        cases = [
            ({**WASTE, "needs_review": True}, rie.SKIP_QUALITY),
            ({**WASTE, "waste_detected": False}, rie.SKIP_NO_WASTE),
            ({**WASTE, "mean_confidence": 0.3}, rie.SKIP_LOW_CONF),
        ]
        for detection, expected in cases:
            res, _ = self.ingest([self.frame()], detection)
            self.assertEqual(res["frames"][0]["decision"], expected)
        res, _ = self.ingest([self.frame()], side_effect=OSError("corrupt"))
        self.assertEqual(res["frames"][0]["decision"], rie.SKIP_UNREADABLE)
        self.assertEqual(res["accepted"], 0)

    def test_accepted_frame_becomes_tagged_dashcam_report_without_second_inference(self):
        with mock.patch.object(pipeline_mod, "detect_waste") as pipeline_detect:
            res, det = self.ingest([self.frame(), self.frame((18.5062, 73.8628))])
        self.assertEqual(res["accepted"], 1)
        self.assertEqual(res["decisions"][rie.SKIP_REPEAT], 1)  # second frame is 25 m away
        self.assertEqual(det.call_count, 1)
        pipeline_detect.assert_not_called()
        report = next(r for r in self.pipe.get_reports() if r["id"] == res["report_ids"][0])
        self.assertEqual((report["channel"], report["reporter_id"], report["provenance"]), ("dashcam", "RP-0005", "observed"))
        self.assertEqual(report["source"], "PMPML Bus 158 dashcam")
        self.assertEqual(ie.get_reward(f"RW-{report['id']}")["amount_inr"], 0)

    def test_unknown_vehicle_and_batch_limit(self):
        with self.assertRaises(rie.UnknownVehicleError):
            rie.ingest_frames(self.pipe, "UBER-123", [self.frame()])
        with self.assertRaises(ValueError):
            rie.ingest_frames(self.pipe, "PMPML-BUS-158", [self.frame()] * (rie.MAX_FRAMES_PER_BATCH + 1))


class TestReplayEndToEnd(IncentiveTestCase):
    def test_replay_creates_simulated_reports_once_per_day(self):
        pipe = pipeline_mod.PlasticWatchUrbanPipeline()
        first = rie.replay_route(pipe, "PMPML-158")
        self.assertGreater(first["accepted"], 0)
        self.assertGreater(first["decisions"][rie.SKIP_FAR], 0)
        reports = {r["id"]: r for r in pipe.get_reports()}
        for rid in first["report_ids"]:
            self.assertEqual(reports[rid]["provenance"], "simulated")
            self.assertTrue(reports[rid]["source"].endswith("(replay)"))
        second = rie.replay_route(pipe, "PMPML-158")
        self.assertEqual(second["accepted"], 0)
        self.assertEqual(second["decisions"][rie.SKIP_REPEAT], first["accepted"])
        dashcam = ie.adoption_stats(pipe.get_reports(), 7)["by_channel"]["dashcam"]
        self.assertEqual(dashcam["reports"], first["accepted"])
        self.assertGreater(dashcam["coverage_pct"], 0)


class TestIngestApi(IncentiveTestCase):
    def setUp(self):
        super().setUp()
        from fastapi.testclient import TestClient
        from backend import api

        api.pipeline._reports_cache = None
        self.addCleanup(setattr, api.pipeline, "_reports_cache", None)
        self.client = TestClient(api.app)

    def test_routes_frames_and_replay_endpoints(self):
        self.assertIn("PMPML-158", [r["route_id"] for r in self.client.get("/api/ingest/routes").json()["routes"]])
        photo = base64.b64encode((pipeline_mod.SAMPLES_DIR / "sample_cardboard_box_drain.jpg").read_bytes()).decode()
        body = {"vehicle_id": "PMPML-BUS-158", "frames": [
            {"lat": ON_DRAIN[0], "lon": ON_DRAIN[1], "captured_at": "2026-10-15T08:00:00+05:30", "image_base64": photo},
            {"lat": OFF_DRAIN[0], "lon": OFF_DRAIN[1], "image_base64": photo},
        ]}
        res = self.client.post("/api/ingest/route-frames", json=body)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertEqual((data["accepted"], data["decisions"][rie.SKIP_FAR]), (1, 1))
        report = next(r for r in self.client.get("/api/reports").json()["reports"] if r["id"] == data["report_ids"][0])
        self.assertEqual((report["provenance"], report["photo_filename"]), ("observed", f"{report['id']}.jpg"))

        bad = {**body, "frames": [{**body["frames"][0], "image_base64": "not base64 at all!!"}]}
        self.assertEqual(self.client.post("/api/ingest/route-frames", json=bad).status_code, 422)
        self.assertEqual(self.client.post("/api/ingest/route-frames", json={**body, "vehicle_id": "UBER-123"}).status_code, 422)
        self.assertEqual(self.client.post("/api/ingest/replay/NOPE").status_code, 404)
        self.assertGreater(self.client.post("/api/ingest/replay/PMC-VAN-12").json()["accepted"], 0)


if __name__ == "__main__":
    import unittest

    unittest.main()
