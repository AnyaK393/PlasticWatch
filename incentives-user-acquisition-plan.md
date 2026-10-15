# PlasticWatch: Incentives & User Acquisition Plan

Plan only. Nothing in the repo has been changed. Based on a read of `AnyaK393/PlasticWatch` (main, Oct 2026).

## 1. The answer to give the judges

> "We don't depend on goodwill. We recruit people who already walk or drive every street (waste pickers, delivery riders, buses and garbage vans), and we **only pay for a photo when it leads to a cleanup our AI has verified**. Households earn tax or bill credits on the same rule, and wards compete for pre-monsoon desilting funds using a public drain score."

The phrase that wins this question is **"pay on verified clearance."** It works because PlasticWatch already has the hard part: DBSCAN de-duplication, an operator review queue, contractor before/after verification, and a 90-day fraud shield. Incentives are just a ledger that listens to those existing events. Most competing teams cannot make that claim.

## 2. What already exists that incentives can reuse

| Existing piece | Where | How incentives use it |
|---|---|---|
| Citizen report ingestion + photo upload | `backend/plasticwatch_pipeline.py` `submit_report`, `POST /api/reports/upload` | Add reporter identity and channel to each report |
| Image quality gate (blur, dark, motion) | `backend/taco_adapter.py` `evaluate_image_quality` | Blocks junk photos from ever earning |
| Operator review queue (approve/reject) | `/api/review-queue`, `/approve`, `/reject` | A rejected report earns nothing; approval moves it to "pending reward" |
| DBSCAN 50 m clustering | `backend/clustering_engine.py` | De-duplicates: first reporter on a hotspot gets the main reward, later ones a small corroboration share |
| Work orders + dispatch | `/api/dispatch`, `smart_dispatch_engine.record_work_order` | "Your report triggered a cleanup" notification |
| Before/after cleanup verification with human override | `backend/verification_engine.py`, `/api/verification-cases/*` | **The payout trigger.** Reward is released only when final decision is PASS |
| 90-day duplicate-billing fraud shield | `smart_dispatch_engine.check_duplicate_billing_fraud` | Same logic reused to stop people re-reporting a site that was just cleared |
| SHA-1 evidence fingerprint | `verification_engine.evidence_fingerprint` | Reuse to reject the same photo being submitted twice |
| Recurrence engine | `backend/recurrence_engine.py` | Feeds the ward drain score (fewer repeat clogs = better score) |
| Vendor zones with anchors | `data/vendor_registry.json` | Quick way to assign a report to a zone/ward before real ward polygons exist |
| Provenance tiers | `backend/provenance.py` | Mark rewards as Observed/Derived so payouts are auditable |

**Gaps today:** reports have no reporter identity, no channel (citizen vs waste picker vs rider vs dashcam), no ward, and there is no reward/credit record anywhere. The mobile app ends at "report submitted" with nothing to come back for.

Side note found while reading: `mobile/citizen.html` posts to `/api/v1/citizen/report`, which does not exist in `backend/api.py`. `mobile/index.html` (the one served at `/mobile`) is fine.

## 3. The core building block: a reward ledger driven by the existing pipeline

One new engine, `backend/incentive_engine.py`, plus two data files. Everything else in this plan is a view on top of it.

### Data model (JSON, using the existing `storage.locked/read_json/write_json` helpers)

`data/reporters.json`
```json
{ "reporter_id": "RP-0042", "display_name": "Sunita (SWaCH)", "channel": "waste_picker",
  "partner_org": "SWaCH Pune", "ward": "Kasba-Vishrambaug", "phone_hash": "…",
  "payout_method": "SWaCH_coop_wallet", "household_link": null, "trust_score": 0.92,
  "joined_at": "2026-10-12T06:00:00Z" }
```
`channel` is one of `citizen`, `waste_picker`, `sanitary_inspector`, `gig_rider`, `dashcam`, `rwa`.
`household_link` holds a property-tax ID or electricity/water consumer number when a household opts into credits.

`data/incentive_programs.json` (rates are config, so the pitch can show different schemes)
```json
[ { "program_id": "SWACH-ROUTE", "channel": "waste_picker", "reward": "₹5 per verified drain check, ₹50 daily route bonus", "funder": "PMC SWM", "payout": "coop_wallet" },
  { "program_id": "GIG-FLOODCHOKE", "channel": "gig_rider", "reward_inr": 15, "funder": "Fleet CSR + PMC", "payout": "partner_wallet" },
  { "program_id": "MSEDCL-CREDIT", "channel": "citizen", "reward_inr": 20, "payout": "utility_bill_credit" },
  { "program_id": "SWACHHATA-TAX", "channel": "citizen", "reward": "points toward annual property-tax rebate (cap 5%)", "payout": "property_tax" } ]
```

`data/reward_ledger.json` (append-only, one row per state change)
```json
{ "entry_id": "RW-1001", "reporter_id": "RP-0042", "report_id": "CR-237", "hotspot_id": "HOTSPOT-03",
  "program_id": "SWACH-ROUTE", "role": "first_reporter", "amount_inr": 15,
  "state": "RELEASED", "evidence": { "verification_case": "HOTSPOT-03", "decision": "PASS", "fingerprint": "…" },
  "created_at": "…", "released_at": "…" }
```

### State machine (each step is an event that already happens in the code)

```
SUBMITTED ──quality gate fails / operator rejects──► VOID (₹0)
    │
    ▼  report ACTIVE (passes gate or approved in review queue)
PENDING  ──clustered into a hotspot──► first reporter = full share, others = corroboration share
    │
    ▼  hotspot included in /api/dispatch work order
ESCROWED
    │
    ▼  verification case final decision == PASS (incl. operator override)
RELEASED ──► payout export (coop wallet / partner API / bill-credit CSV / tax-rebate CSV)
    │
    └── FLAGGED FOR REVIEW or fraud hold ──► HELD (operator decides)
```

Hook points: call `incentive_engine.on_report(...)` at the end of `submit_report` and `_set_review_outcome`; `on_dispatch(...)` inside `/api/dispatch` next to `record_work_order`; `on_verification(...)` inside `/api/verification-cases/{id}/verify` and `/override`. Small, isolated changes.

### Anti-gaming rules (judges will ask "won't people dump trash to get paid?")

- **Pay only on verified clearance**, never per photo. Dumping trash and reporting it costs more effort than ₹15.
- Daily cap per reporter (e.g. 10 rewarded reports), and lower rates once a reporter's trust score drops.
- Same photo hash = rejected; same 35 m spot reported again within 90 days of a verified cleanup = flagged (reuse the fraud-shield code).
- Reporter cannot be on the crew or vendor that cleans the same site (self-dealing check against `vendor_registry.json` / crew name).
- EXIF/GPS sanity: capture time and GPS must be close to submission time and location.
- Repeated recurrence at the same site turns into a root-cause ticket (already in `recurrence_engine`), not repeated payouts.

## 4. What to build, by priority

### P0: for the hackathon demo (aim for one coherent story, about 1–2 days of work)

**P0.1 Reporter identity and channel on every report**
- `CitizenReport` dataclass: add `reporter_id`, `channel`, `ward` (optional fields, defaults keep old data valid).
- `POST /api/reports/upload`: accept `reporter_id` and `channel` form fields.
- `mobile/index.html`: a one-time "Who are you?" sheet (Resident / SWaCH waste picker / Delivery rider / Sanitary inspector) saved in `localStorage`, plus name/phone. Send with every report.
- Seed `data/reporters.json` with 6–8 personas and tag existing seed reports in `citizen_reports.json` so the dashboard looks alive.

**P0.2 Incentive engine + ledger** (section 3). Endpoints:
- `GET /api/reporters/{id}/wallet`: pending, escrowed, released ₹, and each report's stage.
- `GET /api/incentives/ledger` and `GET /api/incentives/programs`.
- `POST /api/incentives/{entry_id}/release` and `/hold`: operator approval, mirroring the existing override flow.

**P0.3 "My Impact" screen in the mobile app** (this is the retention hook)
- After submit, the toast says "₹15 pending, released when the drain is verified clear."
- A wallet card: total earned, pending, and a per-report timeline: *Reported → Merged into Hotspot-03 → Truck dispatched → Verified clear ✅ ₹15 released*.
- Show the before/after pair from verification when a report's cleanup passes. Seeing their photo cause a cleanup is the strongest reason to report again.

**P0.4 New dashboard tab "🤝 Incentives & Adoption"** in `dashboard/app.py` (add to the `st.tabs` list at line ~507)
- KPIs: active reporters by channel, reports per channel, **street/drain coverage in last 7 days** (share of drain segments in `urban_drain_network.geojson` with a report within 50 m), ₹ released, **₹ per verified clearance** (compare against cost of a flood event from the simulator).
- Payout approval queue (pending → release/hold), with the verification evidence beside it.
- Channel comparison chart: this is where you prove waste pickers and riders give far more coverage than residents.

**P0.5 Passive dashcam ingestion (demo as replay)**
- `POST /api/ingest/route-frames`: a vehicle ID (e.g. `PMPML-BUS-158`, `PMC-GARBAGE-VAN-12`) plus a batch of frames with GPS points. Runs the existing `detect_waste` on each frame; keeps only high-confidence detections near a drain; creates reports with `channel="dashcam"` and `source="PMPML Bus 158 dashcam"`.
- Demo: a "Replay bus route" button that feeds the existing `data/taco_samples` images along a polyline in Pune, and the map fills with dashcam reports. Label it Simulated in the provenance layer, like `mission_pipeline` does for its replay mode.
- This is the "you don't make people take photos" answer, live on screen.

**P0.6 Ward Drain Score**
- Ward assignment: for the demo, nearest anchor using the zone anchors already in `vendor_registry.json`; later, real PMC ward polygons (`data/pune_wards.geojson`).
- `GET /api/wards/scorecard`: per ward, coverage %, verified clear rate, median time-to-clear, recurrence trend (from `recurrence_engine`), active reporters.
- Dashboard leaderboard + map shading. Pitch line: "Top wards get priority in pre-monsoon desilting allocation."

**Suggested 3-minute demo flow:** pick "SWaCH waste picker" on the phone → snap a drain → toast shows ₹ pending → in the dashboard, dispatch the hotspot → upload the after-photo, verification passes → phone wallet flips to ₹ released with before/after → hit "Replay bus route" to show dashcam reports appearing → end on the ward leaderboard and "₹ per verified clearance vs cost of flooding".

### P1: right after the hackathon (pilot-ready)

- **Waste-picker route mode:** assign each SWaCH member their usual lanes as a list of drain points; the app shows a morning checklist ("8 of 12 points checked") and pays a daily completion bonus. Endpoint `GET /api/routes/{reporter_id}/today`. Clean reports ("drain is clear") also count, which gives the forecaster negative examples too.
- **Gig fleet partner API:** `POST /api/partner/reports` with an API key per fleet; a single-tap "flood choke on route" payload (photo + GPS + rider ID); a webhook back to the fleet when a reward is released so they credit the rider's wallet. Write a one-page integration spec for Zomato/Swiggy/Blinkit/Rapido.
- **Household credits:** link a property-tax ID or MSEDCL/water consumer number in the profile; monthly CSV exports from the ledger (`/api/incentives/export?program=MSEDCL-CREDIT&month=…`) in the format the utility or PMC tax office would upload. No payment integration needed for the pilot.
- **RWA / society accounts:** a society admin groups residents, sees a society scorecard, and downloads a **pre-monsoon drain maintenance certificate (PDF)** listing verified clearances with before/after photos and timestamps. This serves both the participatory-budgeting and the insurance ideas.
- Real authentication: phone OTP for reporters, roles for operator/partner admin (the dashboard already has a password gate to build on).
- Notifications (SMS/WhatsApp) at each stage of the timeline.

### P2: scale

- On-device model in a dashcam app (the detector badge already says "YOLOv8 Live Edge"; ship a TFLite/ONNX build) so only detections, not video, are uploaded.
- UPI payouts for individuals; direct integration with PMC property-tax and MSEDCL billing systems.
- Reporter trust score learned from history; anomaly detection on reward patterns.
- Insurance partner API: share a society's verified maintenance record (with consent) for premium discounts.
- Feed ward scores into the corporator fund / budget workflow.

## 5. Acquisition view: which channel brings what

| Channel | Why they report | Coverage | Cost per report | Build priority |
|---|---|---|---|---|
| SWaCH waste pickers / Arogya inspectors | Paid task within their existing morning route | Every lane, daily | Low (fixed daily bonus) | P0 persona, P1 route mode |
| Gig riders | ₹10–20 to their existing wallet, one tap | Arterials and low-lying chokepoints | ₹10–20, only when verified | P0 persona, P1 partner API |
| Bus / garbage van dashcams | No human effort | Fixed routes, repeated daily | Near zero after hardware | P0 replay demo, P2 edge model |
| Households | Tax rebate / bill credit | Their own neighbourhood, recurring | ₹15–25 credit or tax points | P0 persona, P1 linkage + exports |
| RWAs / wards | Desilting priority, certificates, insurance | Society boundaries | Non-cash | P0 ward score, P1 certificates |

The first three give the city-wide coverage; households and RWAs give local stake and political visibility. Lead the pitch with the first three.

## 6. Files that would change (when you decide to build)

New: `backend/incentive_engine.py`, `backend/ward_engine.py`, `backend/route_ingest_engine.py`, `data/reporters.json`, `data/incentive_programs.json`, `data/reward_ledger.json`, later `data/pune_wards.geojson`, `tests/test_incentives.py`.

Modified (small hooks): `backend/plasticwatch_pipeline.py` (report fields + ledger hook), `backend/api.py` (new endpoints + hooks in dispatch/verify/override), `mobile/index.html` (persona picker, wallet, timeline), `dashboard/app.py` (new tab, ward leaderboard layer), `backend/provenance.py` (tag rewards and dashcam replay), `README.md` and the pitch deck.
