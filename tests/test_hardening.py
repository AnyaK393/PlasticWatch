"""Hardening tests: quality gate, plastic policy, storage safety, verification evidence,
human overrides, API validation, and invariants of the decision engines.

Every test that writes data redirects the module paths to a temporary directory,
so the real data/ files are never modified.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np
from PIL import Image, ImageFilter

from backend import plasticwatch_pipeline as pipeline_mod
from backend import storage
from backend import verification_engine as ve
from backend.explainability_engine import ExplainabilityEngine
from backend.recurrence_engine import RecurrenceEngine
from backend.simulator_engine import MonsoonSimulatorEngine
from backend.smart_dispatch_engine import SmartDispatchEngine
from backend.taco_adapter import detect_waste, evaluate_image_quality, model_card, taco_annotation_stats

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "taco_samples"


def _jpeg_bytes(img: Image.Image, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def _sample(name: str) -> Image.Image:
    return Image.open(SAMPLES / name).convert("RGB")


class TestQualityGate(unittest.TestCase):
    def test_sharp_photo_passes_both_tiers(self):
        img = _sample("sample_food_wrappers_culvert.jpg")
        self.assertFalse(evaluate_image_quality(img)["needs_review"])
        self.assertFalse(evaluate_image_quality(img, strict=True)["needs_review"])

    def test_blur_is_rejected_for_evidence(self):
        img = Image.open(io.BytesIO(_jpeg_bytes(_sample("sample_pet_bottle_pile.jpg").filter(ImageFilter.GaussianBlur(1.5)))))
        q = evaluate_image_quality(img, strict=True)
        self.assertTrue(q["is_blurry"])
        self.assertTrue(q["needs_review"])

    def test_dark_overexposed_lowres_featureless(self):
        base = _sample("sample_food_wrappers_culvert.jpg")
        self.assertTrue(evaluate_image_quality(Image.eval(base, lambda v: v * 0.3))["is_dark"])
        self.assertTrue(evaluate_image_quality(Image.eval(base, lambda v: min(255, v * 2.6)), strict=True)["is_overexposed"])
        self.assertTrue(evaluate_image_quality(base.resize((300, 225)), strict=True)["is_low_resolution"])
        flat = Image.new("RGB", (1280, 960), (150, 150, 150))
        self.assertTrue(evaluate_image_quality(flat, strict=True)["is_featureless"])
        # A featureless frame is not "blurry" for citizens (clean floor), only unusable as evidence
        self.assertFalse(evaluate_image_quality(flat)["is_blurry"])


class TestPlasticPolicy(unittest.TestCase):
    def test_context_and_food_are_not_plastic(self):
        r = detect_waste(str(SAMPLES / "sample_bottles_drain.jpg"))
        decisions = {t["coco_class"]: t["decision"] for t in r["trace"]}
        self.assertEqual(decisions.get("person"), "ignored")
        self.assertEqual(decisions.get("dining table"), "ignored")
        self.assertEqual(decisions.get("pizza"), "non-plastic waste")
        self.assertIn("plastic_bottle", r["detected_classes"])
        self.assertNotIn("organic", r["detected_classes"])

    def test_clean_surfaces_score_zero(self):
        for img in (Image.new("RGB", (800, 600), (150, 150, 150)), Image.new("RGB", (800, 600), (235, 232, 228))):
            r = detect_waste(img)
            self.assertFalse(r["waste_detected"])
            self.assertEqual(r["severity"], 0.0)

    def test_missing_file_raises_instead_of_silent_placeholder(self):
        with self.assertRaises(FileNotFoundError):
            detect_waste("does_not_exist.jpg")

    def test_model_card_is_honest(self):
        card = model_card()
        self.assertEqual(card["pretraining"], "COCO 2017 (80 classes)")
        self.assertIn("taco_fine_tuned", card)
        stats = taco_annotation_stats()
        if stats.get("available"):
            self.assertEqual(stats["images"], 1500)
            self.assertGreater(stats["mapped_annotations"], 0)


class TestStorage(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_corrupt_file_is_quarantined_not_overwritten(self):
        p = self.tmp / "db.json"
        p.write_text("{not json", encoding="utf-8")
        with self.assertRaises(storage.CorruptDataError):
            storage.read_json(p, [])
        self.assertFalse(p.exists())
        self.assertEqual(len(list(self.tmp.glob("db.json.corrupt-*"))), 1)

    def test_concurrent_appends_are_not_lost(self):
        p = self.tmp / "ledger.json"

        def append(i):
            with storage.locked(p):
                data = storage.read_json(p, [])
                data.append(i)
                storage.write_json(p, data)

        threads = [threading.Thread(target=append, args=(i,)) for i in range(25)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(sorted(storage.read_json(p, [])), list(range(25)))


class TestPipelineSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.patches = [
            mock.patch.object(pipeline_mod, "REPORTS_DB_PATH", self.tmp / "reports.json"),
            mock.patch.object(pipeline_mod, "UPLOADS_DIR", self.tmp / "uploads"),
        ]
        for p in self.patches:
            p.start()
        self.pipe = pipeline_mod.PlasticWatchUrbanPipeline()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp)

    def test_concurrent_submissions_get_unique_ids(self):
        photo = (SAMPLES / "sample_pet_bottle_pile.jpg").read_bytes()
        ids, errors = [], []

        def submit():
            try:
                ids.append(self.pipe.submit_report(18.52, 73.85, "x.jpg", "t", image_bytes=photo)["report"]["id"])
            except Exception as exc:  # pragma: no cover - surfaced by assertion
                errors.append(exc)

        threads = [threading.Thread(target=submit) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertFalse(errors)
        self.assertEqual(len(set(ids)), 6)
        stored = json.loads((self.tmp / "reports.json").read_text())
        self.assertEqual(len({r["id"] for r in stored}), len(stored))

    def test_invalid_coordinates_rejected(self):
        with self.assertRaises(ValueError):
            self.pipe.submit_report(123.0, 73.85)

    def test_review_transition_requires_pending_report(self):
        self.pipe.get_reports()  # seeds
        with self.assertRaises(ValueError):
            self.pipe.approve_report("CR-201")

    def test_blurry_upload_goes_to_review_queue(self):
        blurry = _jpeg_bytes(_sample("sample_food_wrappers_culvert.jpg").filter(ImageFilter.GaussianBlur(4)))
        res = self.pipe.submit_report(18.52, 73.85, "b.jpg", image_bytes=blurry)
        self.assertTrue(res["needs_review"])
        self.assertIn(res["report"]["id"], [r["id"] for r in self.pipe.get_review_queue()])
        hotspot_ids = {rid for h in self.pipe.run_pipeline(rainfall_intensity_mmh=0.0)["hotspots"] for rid in h["report_ids"]}
        self.assertNotIn(res["report"]["id"], hotspot_ids)


class TestVerificationEvidence(unittest.TestCase):
    engine = ve.CleanupVerificationEngine()

    def test_same_scene_full_cleanup_passes(self):
        r = self.engine.verify_cleanup("sample_partial_cleanup.jpg", "sample_cleared_drain.jpg")
        self.assertEqual(r["decision_status"], "PASS")

    def test_different_scene_is_flagged(self):
        r = self.engine.verify_cleanup("sample_food_wrappers_culvert.jpg", "sample_snack_wrappers_curb.jpg")
        self.assertEqual(r["decision_status"], "FLAGGED FOR REVIEW")
        self.assertIn("Same-scene match", {c["check"] for c in r["checks"] if c["status"] == "FAIL"})

    def test_blurry_after_photo_is_inconclusive(self):
        blurry = _jpeg_bytes(_sample("sample_cleared_drain.jpg").filter(ImageFilter.GaussianBlur(3)))
        r = self.engine.verify_cleanup("sample_partial_cleanup.jpg", blurry)
        self.assertEqual(r["decision_status"], "INCONCLUSIVE")
        self.assertFalse(r["invoice_eligible"])


class TestAlignedItemClearance(unittest.TestCase):
    """Same-scene pairs are judged item by item at the original litter locations."""

    @staticmethod
    def _erase(img: Image.Image, boxes) -> bytes:
        import cv2

        arr = np.asarray(img).copy()
        mask = np.zeros(arr.shape[:2], np.uint8)
        for x0, y0, x1, y1 in boxes:
            mask[max(0, y0 - 6):y1 + 6, max(0, x0 - 6):x1 + 6] = 255
        return _jpeg_bytes(Image.fromarray(cv2.inpaint(arr, mask, 7, cv2.INPAINT_TELEA)), 92)

    def setUp(self):
        from backend.taco_adapter import _analysis_frame, litter_blobs

        self.before = _sample("sample_snack_wrappers_curb.jpg")
        frame = _analysis_frame(self.before)
        sx = self.before.width / frame.width
        self.boxes = [[int(v * sx) for v in b["box"]] for b in litter_blobs(frame)]
        self.assertGreaterEqual(len(self.boxes), 2)
        self.engine = ve.CleanupVerificationEngine()

    def test_full_cleanup_passes(self):
        r = self.engine.verify_cleanup(_jpeg_bytes(self.before, 92), self._erase(self.before, self.boxes))
        self.assertTrue(r["alignment"]["aligned"])
        self.assertEqual(r["alignment"]["items_cleared"], r["alignment"]["items_total"])
        self.assertEqual(r["decision_status"], "PASS")

    def test_partial_cleanup_is_flagged(self):
        r = self.engine.verify_cleanup(_jpeg_bytes(self.before, 92), self._erase(self.before, self.boxes[:1]))
        self.assertLess(r["alignment"]["items_cleared"], r["alignment"]["items_total"])
        self.assertEqual(r["decision_status"], "FLAGGED FOR REVIEW")

    def test_no_cleanup_is_flagged(self):
        r = self.engine.verify_cleanup(_jpeg_bytes(self.before, 92), _jpeg_bytes(self.before, 90))
        self.assertEqual(r["alignment"]["items_cleared"], 0)
        self.assertEqual(r["decision_status"], "FLAGGED FOR REVIEW")


class TestCasesAndOverrides(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "cases.json").write_text(json.dumps({"cases": [
            {"case_id": "VC-T1", "site_name": "Test", "ward": "W", "lat": 18.52, "lon": 73.85,
             "vendor_id": "V", "contractor": "C"}]}))
        self.patches = [
            mock.patch.object(ve, "CASES_DIR", self.tmp),
            mock.patch.object(ve, "CASES_MANIFEST", self.tmp / "cases.json"),
            mock.patch.object(ve, "OVERRIDES_PATH", self.tmp / "overrides.json"),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp)

    def _case(self):
        return ve.list_verification_cases()[0]

    def test_photo_lifecycle(self):
        self.assertEqual(self._case()["status"], "AWAITING_PHOTOS")
        with self.assertRaises(ValueError):
            ve.save_case_photo("VC-T1", "after", b"not an image", "x.jpg")
        ve.save_case_photo("VC-T1", "before", (SAMPLES / "sample_partial_cleanup.jpg").read_bytes(), "b.jpg")
        self.assertEqual(self._case()["status"], "AWAITING_AFTER_PHOTO")
        ve.save_case_photo("VC-T1", "after", (SAMPLES / "sample_cleared_drain.jpg").read_bytes(), "a.png")
        self.assertEqual(self._case()["status"], "READY")
        # Replacing with another extension leaves exactly one after-photo
        ve.save_case_photo("VC-T1", "after", (SAMPLES / "sample_cleared_drain.jpg").read_bytes(), "a.jpg")
        self.assertEqual(len(list(self.tmp.glob("VC-T1_after.*"))), 1)
        self.assertTrue(ve.remove_case_photo("VC-T1", "after"))
        self.assertEqual(self._case()["status"], "AWAITING_AFTER_PHOTO")

    def test_override_rules_and_binding(self):
        ve.save_case_photo("VC-T1", "before", (SAMPLES / "sample_food_wrappers_culvert.jpg").read_bytes(), "b.jpg")
        ve.save_case_photo("VC-T1", "after", (SAMPLES / "sample_snack_wrappers_curb.jpg").read_bytes(), "a.jpg")
        case = self._case()
        ai = ve.CleanupVerificationEngine().verify_cleanup(Path(case["before_path"]), Path(case["after_path"]))
        self.assertEqual(ai["decision_status"], "FLAGGED FOR REVIEW")
        fp = case["evidence_fingerprint"]
        with self.assertRaises(ValueError):  # too short
            ve.record_override("VC-T1", ai, fp, "PASS", "SCENE_VERIFIED_ON_SITE", "ok", "Inspector")
        with self.assertRaises(ValueError):  # scene failed needs on-site confirmation
            ve.record_override("VC-T1", ai, fp, "PASS", "DETECTOR_FALSE_POSITIVE", "x" * 30, "Inspector")
        with self.assertRaises(ValueError):  # must change the decision
            ve.record_override("VC-T1", ai, fp, "FLAGGED FOR REVIEW", "OTHER", "x" * 30, "Inspector")
        entry = ve.record_override("VC-T1", ai, fp, "PASS", "SCENE_VERIFIED_ON_SITE", "Inspector visited the culvert at 10:30.", "JE A. Patil")
        final = ve.effective_decision(ai, ve.get_active_override("VC-T1", fp))
        self.assertEqual(final["decision"], "PASS")
        self.assertTrue(final["invoice_eligible"])
        self.assertEqual(final["override"]["override_id"], entry["override_id"])
        # Replacing a photo voids the override
        ve.save_case_photo("VC-T1", "after", (SAMPLES / "sample_cleared_drain.jpg").read_bytes(), "a.jpg")
        self.assertIsNone(ve.get_active_override("VC-T1", self._case()["evidence_fingerprint"]))

    def test_inconclusive_cannot_be_passed(self):
        ai = {"decision_status": "INCONCLUSIVE", "checks": []}
        with self.assertRaises(ValueError):
            ve.record_override("VC-T1", ai, "f" * 40, "PASS", "PHOTO_QUALITY_ACCEPTABLE", "x" * 30, "Inspector")


class TestApiValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from backend import api

        cls.api = api
        cls.client = TestClient(api.app)

    def test_bad_coordinates_and_payloads(self):
        self.assertEqual(self.client.post("/api/reports", json={"lat": 999, "lon": 73.8}).status_code, 422)
        empty = self.client.post("/api/detect", files={"file": ("x.jpg", b"", "image/jpeg")})
        self.assertEqual(empty.status_code, 400)
        junk = self.client.post("/api/detect", files={"file": ("x.jpg", b"definitely not an image", "image/jpeg")})
        self.assertEqual(junk.status_code, 400)
        self.assertIn("HEIC", junk.json()["detail"])
        with mock.patch.object(self.api, "MAX_UPLOAD_BYTES", 1000):
            big = self.client.post("/api/detect", files={"file": ("x.jpg", (SAMPLES / "sample_pet_bottle_pile.jpg").read_bytes(), "image/jpeg")})
            self.assertEqual(big.status_code, 413)

    def test_unknown_resources(self):
        self.assertEqual(self.client.post("/api/verification-cases/NOPE/verify").status_code, 404)
        self.assertEqual(self.client.delete("/api/verification-cases/VC-01/photo/sideways").status_code, 400)
        self.assertEqual(self.client.post("/api/verification-cases/VC-01/override",
                                          json={"decision": "PASS", "reason_code": "OTHER", "justification": "short", "operator": "A"}).status_code, 422)

    def test_model_card_endpoint(self):
        r = self.client.get("/api/model-card")
        self.assertEqual(r.status_code, 200)
        self.assertIn("class_policy", r.json()["model"])


class TestDecisionInvariants(unittest.TestCase):
    def test_simulator_monotonic(self):
        sim = MonsoonSimulatorEngine()
        dry = sim.simulate(rainfall_intensity_mmh=0)["outcomes"]
        self.assertEqual(dry["potential_flood_inundation_area_sqm"], 0)
        self.assertEqual(dry["plastic_swept_into_rivers_kg"], 0)
        floods = [sim.simulate(rainfall_intensity_mmh=r)["outcomes"]["potential_flood_inundation_area_sqm"] for r in (10, 30, 60)]
        self.assertEqual(floods, sorted(floods))
        silts = [sim.simulate(drain_silt_pct=s)["outcomes"]["potential_flood_inundation_area_sqm"] for s in (0, 50, 90)]
        self.assertEqual(silts, sorted(silts))
        swept = [sim.simulate(available_fleet=f, rainfall_intensity_mmh=35)["outcomes"]["plastic_swept_into_rivers_kg"] for f in (1, 4, 10)]
        self.assertEqual(swept, sorted(swept, reverse=True))

    def test_explainability_safety_order(self):
        base = {"total_score": 90, "nearest_drain_distance_m": 5, "rainfall_intensity_mmh": 30, "reports": []}
        self.assertEqual(ExplainabilityEngine.recommend_action({**base, "confidence": 0.9})["action_type"], "IMMEDIATE_CLEANUP_DISPATCH")
        self.assertEqual(ExplainabilityEngine.recommend_action({**base, "confidence": 0.4})["action_type"], "HUMAN_VERIFICATION_REQUIRED")
        self.assertEqual(ExplainabilityEngine.recommend_action({**base, "confidence": 0.9, "rainfall_intensity_mmh": 0})["action_type"], "ROUTINE_MONITORING")

    def test_fraud_shield_window(self):
        sd = SmartDispatchEngine()
        inside = sd.check_duplicate_billing_fraud(18.5204, 73.8568, reference_date=datetime(2026, 10, 9, tzinfo=timezone.utc))
        self.assertTrue(inside["is_fraud_suspect"])
        later = sd.check_duplicate_billing_fraud(18.5204, 73.8568, reference_date=datetime(2027, 6, 1, tzinfo=timezone.utc))
        self.assertFalse(later["is_fraud_suspect"])
        far = sd.check_duplicate_billing_fraud(18.60, 73.70, reference_date=datetime(2026, 10, 9, tzinfo=timezone.utc))
        self.assertFalse(far["is_fraud_suspect"])

    def test_root_cause_recovers_known_drivers(self):
        expected = {
            "LOC-PUNE-01": "High-Density Residential Culvert",
            "LOC-PUNE-02": "Transit Hub",
            "LOC-PUNE-03": "Riverine Confluence",
            "LOC-PUNE-04": "Street Food / Kiosk Strip",
        }
        for site in RecurrenceEngine().analyze_recurrence()["locations"]:
            self.assertEqual(site["root_cause_analysis"]["primary_driver"], expected[site["site_id"]], site["site_id"])


if __name__ == "__main__":
    unittest.main()
