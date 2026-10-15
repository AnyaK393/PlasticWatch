"""Incentive engine tests: reward state machine, anti-gaming rules and pipeline/API hooks.

Every test redirects the ledger, reporter registry and report database to a
temporary directory, so the real data/ files are never modified.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from backend import incentive_engine as ie
from backend import plasticwatch_pipeline as pipeline_mod

SHANIWAR = (18.5204, 73.8568)
NAGZARI = (18.5060, 73.8625)
PASS = {"decision": "PASS", "source": "AI", "override": None}
FLAGGED = {"decision": "FLAGGED FOR REVIEW", "source": "AI", "override": None}


def _report(rid, lat_lon=SHANIWAR, reporter="RP-0001", minutes=0, verification="pending", status="ACTIVE"):
    ts = datetime(2026, 10, 10, 6, 0, tzinfo=timezone.utc) + timedelta(minutes=minutes)
    return {"id": rid, "lat": lat_lon[0], "lon": lat_lon[1], "submitted_at": ts.isoformat(),
            "verification": verification, "status": status, "reporter_id": reporter}


def _hotspot(label, *report_ids, merged=True):
    return {"id": label, "site_key": min(report_ids), "report_ids": list(report_ids), "is_merged": merged}


def _case(case_id="VC-T1", lat_lon=SHANIWAR, fingerprint="fp-1", contractor="PMC-SANITATION-04", vendor_id="VEND-PUNE-KASBA"):
    return {"case_id": case_id, "lat": lat_lon[0], "lon": lat_lon[1], "evidence_fingerprint": fingerprint,
            "site_name": "Test culvert", "contractor": contractor, "vendor_id": vendor_id}


class IncentiveTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        shutil.copy(ie.REPORTERS_PATH, self.tmp / "reporters.json")
        self.patches = [
            mock.patch.object(ie, "LEDGER_PATH", self.tmp / "reward_ledger.json"),
            mock.patch.object(ie, "REPORTERS_PATH", self.tmp / "reporters.json"),
            mock.patch.object(pipeline_mod, "REPORTS_DB_PATH", self.tmp / "reports.json"),
            mock.patch.object(pipeline_mod, "UPLOADS_DIR", self.tmp / "uploads"),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp)

    def trust(self, reporter_id):
        return ie.get_reporter(reporter_id)["trust_score"]


class TestReporters(IncentiveTestCase):
    def test_register_is_idempotent_per_phone_and_hides_hash(self):
        a, created_a = ie.register_reporter("Asha", "waste_picker", phone="+91 98220 12345")
        b, created_b = ie.register_reporter("Asha again", "citizen", phone="9822012345")
        self.assertTrue(created_a)
        self.assertFalse(created_b)
        self.assertEqual(a["reporter_id"], b["reporter_id"])
        self.assertEqual(a["program_id"], "SWACH-ROUTE")
        self.assertNotIn("phone_hash", ie.public_reporter(a))

    def test_register_validation(self):
        with self.assertRaises(ie.IncentiveError):
            ie.register_reporter("X", "citizen")
        with self.assertRaises(ie.IncentiveError):
            ie.register_reporter("Valid Name", "astronaut")
        with self.assertRaises(ie.IncentiveError):
            ie.register_reporter("Valid Name", "citizen", phone="123")

    def test_registered_channel_wins_and_unknown_reporter_rejected(self):
        reporter, channel = ie.resolve_identity("RP-0001", "citizen")
        self.assertEqual(channel, "waste_picker")
        self.assertEqual(ie.resolve_identity(None, "bogus")[1], "citizen")
        with self.assertRaises(ie.UnknownReporterError):
            ie.resolve_identity("RP-9999", None)


class TestStateMachine(IncentiveTestCase):
    def test_anonymous_reports_earn_nothing(self):
        self.assertIsNone(ie.on_report(_report("CR-1", reporter=None)))
        self.assertEqual(ie.ledger_events(), [])

    def test_happy_path_pending_escrow_release(self):
        r = ie.on_report(_report("CR-1"))
        self.assertEqual((r["state"], r["role"], r["amount_inr"]), (ie.PENDING, ie.FIRST_REPORTER, 15))
        self.assertIn("₹15 pending", ie.status_message(r))

        [esc] = ie.on_dispatch([_hotspot("HOTSPOT-01", "CR-1", merged=False)], "WO-1")
        self.assertEqual((esc["state"], esc["work_order_id"]), (ie.ESCROWED, "WO-1"))

        [rel] = ie.on_verification(_case(), PASS)
        self.assertEqual(rel["state"], ie.RELEASED)
        self.assertEqual(rel["evidence"]["verification_case"], "VC-T1")
        self.assertTrue(rel["released_at"])
        self.assertEqual([e["state"] for e in ie.ledger_events("RW-CR-1")], [ie.PENDING, ie.ESCROWED, ie.RELEASED])

    def test_hooks_are_idempotent(self):
        ie.on_report(_report("CR-1"))
        ie.on_report(_report("CR-1"))
        ie.on_dispatch([_hotspot("HOTSPOT-01", "CR-1")], "WO-1")
        self.assertEqual(ie.on_dispatch([_hotspot("HOTSPOT-01", "CR-1")], "WO-2"), [])
        ie.on_verification(_case(), PASS)
        self.assertEqual(ie.on_verification(_case(), PASS), [])
        self.assertEqual(len(ie.ledger_events("RW-CR-1")), 3)

    def test_no_payout_before_dispatch(self):
        ie.on_report(_report("CR-1"))
        self.assertEqual(ie.on_verification(_case(), PASS), [])
        self.assertEqual(ie.get_reward("RW-CR-1")["state"], ie.PENDING)

    def test_inconclusive_changes_nothing_and_far_cases_ignored(self):
        ie.on_report(_report("CR-1"))
        ie.on_dispatch([_hotspot("HOTSPOT-01", "CR-1")], "WO-1")
        self.assertEqual(ie.on_verification(_case(), {"decision": "INCONCLUSIVE"}), [])
        self.assertEqual(ie.on_verification(_case(lat_lon=NAGZARI), PASS), [])
        self.assertEqual(ie.get_reward("RW-CR-1")["state"], ie.ESCROWED)

    def test_review_queue_approve_and_reject(self):
        queued = ie.on_report(_report("CR-1", verification="PENDING_OPERATOR_VERIFICATION", status="PENDING_OPERATOR_VERIFICATION"))
        self.assertEqual(queued["state"], ie.SUBMITTED)
        approved = ie.on_review(_report("CR-1", verification="verified", status="APPROVED"))
        self.assertEqual(approved["state"], ie.PENDING)

        before = self.trust("RP-0003")
        ie.on_report(_report("CR-2", lat_lon=NAGZARI, reporter="RP-0003",
                             verification="PENDING_OPERATOR_VERIFICATION", status="PENDING_OPERATOR_VERIFICATION"))
        rejected = ie.on_review(_report("CR-2", lat_lon=NAGZARI, reporter="RP-0003", verification="rejected", status="REJECTED"))
        self.assertEqual(rejected["state"], ie.VOID)
        self.assertLess(self.trust("RP-0003"), before)

    def test_first_reporter_and_corroboration_share(self):
        ie.on_report(_report("CR-1", reporter="RP-0003", minutes=0))
        second = ie.on_report(_report("CR-2", lat_lon=(18.5206, 73.8569), reporter="RP-0001", minutes=10))
        self.assertEqual((second["role"], second["amount_inr"]), (ie.CORROBORATOR, 5))
        rows = {r["reward_id"]: r for r in ie.on_dispatch([_hotspot("HOTSPOT-02", "CR-2", "CR-1")], "WO-1")}
        self.assertEqual(rows["RW-CR-1"]["role"], ie.FIRST_REPORTER)
        self.assertEqual(rows["RW-CR-1"]["amount_inr"], 20)  # MSEDCL credit
        self.assertEqual(rows["RW-CR-2"]["role"], ie.CORROBORATOR)
        self.assertEqual(rows["RW-CR-2"]["hotspot"]["label"], "HOTSPOT-02")

    def test_dispatch_backfills_tagged_reports_missing_from_ledger(self):
        seed = _report("CR-1", reporter="RP-0003", minutes=0)
        ie.on_report(_report("CR-2", lat_lon=(18.5206, 73.8569), minutes=30))
        hotspot = {**_hotspot("HOTSPOT-01", "CR-1", "CR-2"), "reports": [seed]}
        rows = {r["reward_id"]: r for r in ie.on_dispatch([hotspot], "WO-1")}
        self.assertEqual(rows["RW-CR-1"]["role"], ie.FIRST_REPORTER)
        self.assertEqual(rows["RW-CR-2"]["role"], ie.CORROBORATOR)

    def test_flagged_verification_holds_then_override_pass_releases(self):
        ie.on_report(_report("CR-1"))
        ie.on_dispatch([_hotspot("HOTSPOT-01", "CR-1")], "WO-1")
        [held] = ie.on_verification(_case(), FLAGGED)
        self.assertEqual((held["state"], held["hold"]["kind"]), (ie.HELD, "verification"))
        override = {"decision": "PASS", "source": "operator override", "override": {"override_id": "OVR-1"}}
        [rel] = ie.on_verification(_case(), override)
        self.assertEqual(rel["state"], ie.RELEASED)
        self.assertEqual(rel["evidence"]["override_id"], "OVR-1")


class TestAntiGaming(IncentiveTestCase):
    def test_same_photo_twice_is_void(self):
        photo = b"\xff\xd8 the same bytes \xff\xd9"
        ie.on_report(_report("CR-1"), photo)
        before = self.trust("RP-0001")
        dup = ie.on_report(_report("CR-2", lat_lon=NAGZARI), photo)
        self.assertEqual(dup["state"], ie.VOID)
        self.assertLess(self.trust("RP-0001"), before)

    def test_daily_cap(self):
        with mock.patch.object(ie, "list_programs", return_value=[{**ie.get_program("SWACH-ROUTE"), "daily_cap": 2}]):
            states = [ie.on_report(_report(f"CR-{i}", lat_lon=(18.50 + i * 0.01, 73.85), minutes=i))["state"] for i in range(3)]
        self.assertEqual(states, [ie.PENDING, ie.PENDING, ie.VOID])

    def test_repeat_payout_at_same_site_is_held(self):
        ie.on_report(_report("CR-1"))
        ie.on_dispatch([_hotspot("HOTSPOT-01", "CR-1")], "WO-1")
        ie.on_verification(_case(), PASS)
        now = datetime.now(timezone.utc).isoformat()
        again = ie.on_report({**_report("CR-2", reporter="RP-0003"), "submitted_at": now})
        self.assertEqual((again["state"], again["hold"]["kind"]), (ie.HELD, "repeat_site"))

    def test_self_dealing_reporter_is_held_on_pass(self):
        crew, _ = ie.register_reporter("Crew member", "citizen", partner_org="Sahyadri Civic Desilting Consortium")
        ie.on_report(_report("CR-1", reporter=crew["reporter_id"]))
        ie.on_dispatch([_hotspot("HOTSPOT-01", "CR-1")], "WO-1")
        [held] = ie.on_verification(_case(), PASS)
        self.assertEqual((held["state"], held["hold"]["kind"]), (ie.HELD, "self_dealing"))
        # Operator can still pay it: verification passed, the conflict was reviewed
        released = ie.release_reward("RW-CR-1", "Ward Officer", "Reporter not on the crew for this job")
        self.assertEqual(released["state"], ie.RELEASED)

    def test_low_trust_reduces_rate(self):
        self.assertEqual(ie.reward_amount(ie.get_program("SWACH-ROUTE"), ie.FIRST_REPORTER, 0.5), 8)
        self.assertEqual(ie.reward_amount(ie.get_program("SWACH-ROUTE"), ie.FIRST_REPORTER, 0.2), 0)


class TestOperatorActions(IncentiveTestCase):
    def test_hold_and_release_rules(self):
        ie.on_report(_report("CR-1"))
        with self.assertRaises(ie.IncentiveError):
            ie.release_reward("RW-CR-1", "Officer", "not held, cannot release")
        with self.assertRaises(ie.IncentiveError):
            ie.hold_reward("RW-CR-1", "Officer", "short")
        with self.assertRaises(KeyError):
            ie.hold_reward("RW-NOPE", "Officer", "a long enough note")
        held = ie.hold_reward("RW-CR-1", "Officer", "Investigating repeated reports")
        self.assertEqual((held["state"], held["hold"]["from_state"]), (ie.HELD, ie.PENDING))
        # Releasing a hold without verified clearance resumes the wait; it never pays early
        resumed = ie.release_reward("RW-CR-1", "Officer", "Investigation closed, genuine")
        self.assertEqual(resumed["state"], ie.PENDING)
        ie.on_dispatch([_hotspot("HOTSPOT-01", "CR-1")], "WO-1")
        self.assertEqual(ie.on_verification(_case(), PASS)[0]["state"], ie.RELEASED)
        with self.assertRaises(ie.IncentiveError):
            ie.hold_reward("RW-CR-1", "Officer", "cannot hold a released reward")


class TestViews(IncentiveTestCase):
    def test_wallet_totals_and_timeline(self):
        ie.on_report(_report("CR-1"))
        ie.on_report(_report("CR-2", lat_lon=NAGZARI, minutes=5))
        ie.on_dispatch([_hotspot("HOTSPOT-01", "CR-1", merged=False)], "WO-1")
        ie.on_verification(_case(), PASS)
        w = ie.wallet("RP-0001")
        self.assertEqual(w["totals"]["released_inr"], 15)
        self.assertEqual(w["totals"]["pending_inr"], 15)
        released = next(r for r in w["rewards"] if r["report_id"] == "CR-1")
        self.assertEqual([s["done"] for s in released["timeline"]], [True, True, True, True])
        self.assertIsNone(ie.wallet("RP-9999"))
        s = ie.summary()
        self.assertEqual(s["by_channel"]["waste_picker"]["released_inr"], 15)
        self.assertEqual(s["inr_per_verified_clearance"], 15.0)

    def test_backfill_tags_seed_reports_once(self):
        reports = [r.to_dict() for r in pipeline_mod.get_seed_reports()]
        self.assertEqual(ie.backfill_from_reports(reports), 8)
        self.assertEqual(ie.backfill_from_reports(reports), 0)
        self.assertEqual(ie.get_reward("RW-CR-201")["role"], ie.FIRST_REPORTER)
        self.assertEqual(ie.get_reward("RW-CR-202")["role"], ie.CORROBORATOR)

    def test_corrupt_ledger_is_quarantined_not_overwritten(self):
        ie.LEDGER_PATH.write_text("{not json")
        self.assertEqual(ie.list_rewards(), [])
        self.assertTrue(list(self.tmp.glob("reward_ledger.json.corrupt-*")))


class TestAdoption(IncentiveTestCase):
    def test_drain_coverage_counts_only_length_near_reports(self):
        from backend import ward_engine as we

        line = [[73.8500, 18.5000], [73.8600, 18.5000]]  # ~1.05 km east-west
        path = self.tmp / "drains.geojson"
        path.write_text(json.dumps({"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"id": "d1"}, "geometry": {"type": "LineString", "coordinates": line}}]}))
        original = we.drain_sample_points
        with mock.patch.object(we, "drain_sample_points", lambda: original(path)):
            none = we.drain_coverage([])
            one = we.drain_coverage([{"lat": 18.5000, "lon": 73.8550}])
            far = we.drain_coverage([{"lat": 18.5100, "lon": 73.8550}])
        self.assertAlmostEqual(none["drain_km"], 1.05, delta=0.03)
        self.assertEqual(none["coverage_pct"], 0.0)
        self.assertAlmostEqual(one["covered_km"], 0.1, delta=0.03)  # +/- 50 m around the report
        self.assertEqual(far["coverage_pct"], 0.0)

    def test_window_excludes_old_and_rejected_reports(self):
        from backend.ward_engine import reports_in_window

        now = datetime(2026, 10, 15, tzinfo=timezone.utc)
        rows = [
            {"id": "a", "submitted_at": "2026-10-14T10:00:00+00:00"},
            {"id": "b", "submitted_at": "2026-09-01T10:00:00+00:00"},
            {"id": "c", "submitted_at": "2026-10-14T10:00:00+00:00", "verification": "rejected"},
            {"id": "d", "submitted_at": "garbage"},
        ]
        self.assertEqual([r["id"] for r in reports_in_window(rows, 7, now)], ["a"])
        self.assertEqual([r["id"] for r in reports_in_window(rows, None, now)], ["a", "b"])

    def test_adoption_stats_by_channel(self):
        now = datetime(2026, 10, 15, tzinfo=timezone.utc)
        rows = [
            {**_report("CR-1", reporter="RP-0001"), "channel": "waste_picker", "submitted_at": "2026-10-14T06:00:00+00:00"},
            {**_report("CR-2", reporter=None), "channel": "citizen", "submitted_at": "2026-10-14T07:00:00+00:00"},
            {**_report("CR-3", reporter="RP-0002"), "channel": "gig_rider", "submitted_at": "2026-08-01T07:00:00+00:00"},
        ]
        s = ie.adoption_stats(rows, 7, now)
        self.assertEqual((s["reports"], s["active_reporters"]), (2, 1))
        self.assertEqual(s["by_channel"]["waste_picker"]["active_reporters"], 1)
        self.assertEqual(s["by_channel"]["citizen"]["anonymous_reports"], 1)
        self.assertEqual(s["by_channel"]["gig_rider"]["reports"], 0)
        self.assertGreater(s["coverage"]["drain_km"], 10)


class TestPipelineHooks(IncentiveTestCase):
    def test_submit_dispatch_through_pipeline(self):
        pipe = pipeline_mod.PlasticWatchUrbanPipeline()
        res = pipe.submit_report(18.5700, 73.8000, "sample_bottles_drain.jpg", "hook test", reporter_id="RP-0002", channel="citizen")
        rep = res["report"]
        self.assertEqual((rep["reporter_id"], rep["channel"]), ("RP-0002", "gig_rider"))
        self.assertTrue(rep["ward"])
        self.assertEqual(res["incentive"]["state"], ie.PENDING)
        with self.assertRaises(ie.UnknownReporterError):
            pipe.submit_report(18.52, 73.85, reporter_id="RP-9999")

        hotspot = next(h for h in pipe.run_pipeline(rainfall_intensity_mmh=0.0)["hotspots"] if rep["id"] in h["report_ids"])
        plan = pipe.dispatch_work_order([hotspot["id"]], rainfall_intensity_mmh=0.0)
        self.assertIn(res["incentive"]["reward_id"], [e["reward_id"] for e in plan["incentives_escrowed"]])
        self.assertEqual(ie.get_reward(res["incentive"]["reward_id"])["state"], ie.ESCROWED)

    def test_toast_role_matches_payout_next_to_seed_reports(self):
        pipe = pipeline_mod.PlasticWatchUrbanPipeline()
        res = pipe.submit_report(*SHANIWAR, "sample_bottles_drain.jpg", reporter_id="RP-0002")
        self.assertEqual(res["incentive"]["role"], ie.CORROBORATOR)  # CR-201 (seed, RP-0001) was first
        self.assertIn("corroborating", res["incentive"]["message"])
        self.assertEqual(ie.get_reward("RW-CR-201")["state"], ie.PENDING)

    def test_anonymous_submission_unchanged(self):
        res = pipeline_mod.PlasticWatchUrbanPipeline().submit_report(18.5700, 73.8000, "sample_bottles_drain.jpg")
        self.assertNotIn("incentive", res)
        self.assertEqual(res["report"]["channel"], "citizen")

    def test_ledger_failure_never_breaks_submission(self):
        with mock.patch.object(ie, "on_report", side_effect=RuntimeError("disk full")):
            res = pipeline_mod.PlasticWatchUrbanPipeline().submit_report(18.5700, 73.8000, "sample_bottles_drain.jpg", reporter_id="RP-0001")
        self.assertEqual(res["status"], "success")
        self.assertNotIn("incentive", res)


class TestIncentiveApi(IncentiveTestCase):
    def setUp(self):
        super().setUp()
        from fastapi.testclient import TestClient
        from backend import api

        api.pipeline._reports_cache = None
        self.addCleanup(setattr, api.pipeline, "_reports_cache", None)
        self.client = TestClient(api.app)

    def test_register_report_wallet_and_payout_endpoints(self):
        reg = self.client.post("/api/reporters", json={"display_name": "Test Rider", "channel": "gig_rider", "phone": "9000000001"})
        self.assertEqual(reg.status_code, 200)
        rid = reg.json()["reporter"]["reporter_id"]
        self.assertNotIn("phone_hash", reg.json()["reporter"])
        self.assertEqual(self.client.post("/api/reporters", json={"display_name": "Bad", "channel": "pilot"}).status_code, 422)

        sub = self.client.post("/api/reports", json={"lat": 18.5700, "lon": 73.8000, "reporter_id": rid})
        self.assertEqual(sub.status_code, 200)
        self.assertEqual(sub.json()["incentive"]["amount_inr"], 15)
        self.assertEqual(self.client.post("/api/reports", json={"lat": 18.57, "lon": 73.80, "reporter_id": "RP-9999"}).status_code, 422)

        wallet = self.client.get(f"/api/reporters/{rid}/wallet").json()
        self.assertEqual(wallet["totals"]["pending_inr"], 15)
        self.assertEqual(self.client.get("/api/reporters/RP-9999/wallet").status_code, 404)

        reward_id = sub.json()["incentive"]["reward_id"]
        body = {"operator": "Ward Officer", "note": "checking a complaint"}
        self.assertEqual(self.client.post(f"/api/incentives/{reward_id}/release", json=body).status_code, 409)
        self.assertEqual(self.client.post(f"/api/incentives/{reward_id}/hold", json=body).json()["reward"]["state"], ie.HELD)
        self.assertEqual(self.client.post("/api/incentives/RW-NOPE/hold", json=body).status_code, 404)
        detail = self.client.get(f"/api/incentives/{reward_id}").json()
        self.assertEqual(len(detail["history"]), 2)

    def test_wallet_shows_report_photo_and_verified_before_after(self):
        from backend import verification_engine as ve

        (self.tmp / "cases.json").write_text(json.dumps({"cases": [{**_case(), "ward": "W"}]}))
        samples = Path(pipeline_mod.SAMPLES_DIR)
        shutil.copy(samples / "sample_bottles_drain.jpg", self.tmp / "VC-T1_before.jpg")
        shutil.copy(samples / "sample_pet_bottle_pile.jpg", self.tmp / "VC-T1_after.jpg")
        for attr, value in (("CASES_DIR", self.tmp), ("CASES_MANIFEST", self.tmp / "cases.json")):
            patcher = mock.patch.object(ve, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        sub = self.client.post("/api/reports", json={"lat": SHANIWAR[0], "lon": SHANIWAR[1], "reporter_id": "RP-0005"}).json()
        reward_id = sub["incentive"]["reward_id"]
        ie.on_dispatch([_hotspot("HOTSPOT-01", sub["report"]["id"])], "WO-1")
        ie.on_verification({**ve.list_verification_cases()[0]}, PASS)

        wallet = self.client.get("/api/reporters/RP-0005/wallet").json()
        mine = next(r for r in wallet["rewards"] if r["reward_id"] == reward_id)
        self.assertEqual(mine["photo_url"], "/static/samples/sample_bottles_drain.jpg")
        after_url = mine["verification_photos"]["after"]
        self.assertEqual(after_url, "/api/verification-cases/VC-T1/photo/after")
        self.assertEqual(self.client.get(after_url).status_code, 200)
        self.assertEqual(self.client.get("/api/verification-cases/VC-T1/photo/sideways").status_code, 400)
        self.assertEqual(self.client.get("/api/verification-cases/NOPE/photo/after").status_code, 404)

    def test_mobile_page_has_onboarding_and_wallet(self):
        html = self.client.get("/mobile").text
        for marker in ('id="onboard-sheet"', 'id="wallet-view"', "reporter_id", "/wallet"):
            self.assertIn(marker, html)

    def test_ledger_programs_summary(self):
        self.assertGreaterEqual(len(self.client.get("/api/incentives/programs").json()["programs"]), 4)
        ledger = self.client.get("/api/incentives/ledger").json()  # backfills seed personas
        self.assertGreaterEqual(ledger["count"], 8)
        self.assertEqual(self.client.get("/api/incentives/ledger?state=bogus").status_code, 422)
        summary = self.client.get("/api/incentives/summary").json()
        self.assertGreater(summary["by_channel"]["waste_picker"]["reports"], 0)
        adoption = self.client.get("/api/incentives/adoption?days=365").json()
        self.assertGreaterEqual(adoption["by_channel"]["waste_picker"]["reports"], 2)
        self.assertEqual(self.client.get("/api/incentives/adoption?days=0").status_code, 422)


if __name__ == "__main__":
    unittest.main()
