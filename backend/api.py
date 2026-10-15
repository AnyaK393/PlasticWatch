"""FastAPI Backend Server for PlasticWatch Urban Intelligence Platform.

Exposes REST API endpoints for:
  - Mobile citizen report ingestion
  - Computer vision waste detection & severity estimation
  - DBSCAN spatial clustering of duplicate reports
  - Urban drain proximity & rainfall risk score calculation
  - Municipal truck dispatch & route optimization
  - Mobile web simulator client delivery
"""
from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.clustering_engine import DBSCANClusteringEngine
from backend.drain_risk_engine import DrainRiskEngine
from backend.explainability_engine import ExplainabilityEngine
from backend.forecasting_engine import ForecastingEngine
from backend import incentive_engine
from backend.incentive_engine import IncentiveError, UnknownReporterError
from backend.plasticwatch_pipeline import (
    UPLOADS_DIR,
    PlasticWatchUrbanPipeline,
    get_seed_reports,
    resolve_photo_path,
    run_plasticwatch,
)
from backend.provenance import PROVENANCE_TIERS, render_provenance_badge
from backend.recurrence_engine import RecurrenceEngine, load_landuse_nodes
from backend import route_ingest_engine, ward_engine
from backend.route_optimizer import RouteOptimizer
from backend.simulator_engine import MonsoonSimulatorEngine
from backend.smart_dispatch_engine import SmartDispatchEngine, list_work_orders, record_work_order
from backend.taco_adapter import dataset_status, detect_waste, get_sample_images, model_card, taco_annotation_stats
from backend.verification_engine import (
    OVERRIDE_REASONS,
    CleanupVerificationEngine,
    effective_decision,
    get_active_override,
    list_overrides,
    list_verification_cases,
    record_override,
    remove_case_photo,
    save_case_photo,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
MOBILE_HTML_PATH = ROOT / "mobile" / "index.html"
SAMPLES_DIR = ROOT / "data" / "taco_samples"
DRAIN_GEOJSON = ROOT / "data" / "urban_drain_network.geojson"

app = FastAPI(
    title="PlasticWatch Urban Civic Intelligence API",
    description="API-first pipeline for citizen waste reporting, DBSCAN clustering, and municipal cleanup dispatch.",
    version="2.0.0",
)

# Enable CORS for local cross-origin simulation and DevTools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = PlasticWatchUrbanPipeline()

MAX_UPLOAD_BYTES = 15 * 1024 * 1024


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    """Never leak a stack trace to the phone/dashboard; log it and return JSON."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {type(exc).__name__}. The request was not applied."})


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Empty photo upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Photo larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    return data


def _unreadable_detail(exc: Exception) -> str:
    return f"Unreadable image ({exc}). Use JPEG/PNG/WebP; iPhone HEIC photos must be exported as JPEG."


# ── Request / Response Models ────────────────────────────────────────────────
class ReportSubmissionRequest(BaseModel):
    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude of waste observation")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude of waste observation")
    photo_filename: str = Field("sample_bottles_drain.jpg", description="Sample photo filename or reference")
    notes: Optional[str] = Field("", max_length=500, description="Optional citizen field notes")
    reporter_id: Optional[str] = Field(None, max_length=20, description="Registered reporter (POST /api/reporters); omit for anonymous")
    channel: Optional[str] = Field(None, max_length=30, description="Reporter channel for anonymous reports (citizen, waste_picker, ...)")


class DispatchRequest(BaseModel):
    hotspot_ids: List[str] = Field(default_factory=list, description="List of hotspot IDs to dispatch")
    truck_id: Optional[str] = Field("PMC-TRUCK-04", description="Assigned municipal truck ID")
    crew_name: Optional[str] = Field("PMC Rapid Sanitation Crew 2", description="Assigned cleanup crew")
    rainfall_intensity_mmh: Optional[float] = Field(None, ge=0.0, le=75.0, description="Rainfall override used for ranking; null = live")
    drain_silt_ratio: Optional[float] = Field(None, ge=0.0, le=1.0, description="Siltation override used for ranking; null = per drain")
    surface_slope_factor: float = Field(1.0, ge=1.0, le=1.5, description="Surface slope factor used for ranking")
    crew_eta_min: float = Field(45.0, ge=5.0, le=240.0, description="Expected crew arrival (min) for floating-debris drift")


class ForecastRequest(BaseModel):
    horizon_hours: int = Field(48, ge=24, le=72, description="Forecast window (24, 48, or 72 hours)")
    rainfall_scenario_mm: Optional[float] = Field(None, description="Optional manual rainfall override in mm")


class CleanupVerificationRequest(BaseModel):
    before_image: str = Field("sample_bottles_drain.jpg", description="Pre-cleanup sample filename, path, or base64")
    after_image: str = Field("sample_cleared_drain.jpg", description="Post-cleanup clearance sample filename, path, or base64")
    hotspot_id: Optional[str] = Field("HOTSPOT-01", description="Hotspot identifier")
    site_name: Optional[str] = Field("Shaniwar Wada Culvert", description="Location name")
    contractor_id: Optional[str] = Field("PMC-SANITATION-04", description="Cleanup crew/contractor ID")
    operator_notes: Optional[str] = Field("", description="Field operator notes")


class SimulationRequest(BaseModel):
    rainfall_intensity_mmh: float = Field(25.0, ge=0.0, le=75.0, description="Rainfall intensity in mm/hr (0-75)")
    available_fleet: int = Field(4, ge=1, le=10, description="Available suction/tipper trucks (1-10)")
    action_delay_hours: float = Field(2.0, ge=0.0, le=24.0, description="Action delay in hours (0-24)")
    wind_speed_kmh: float = Field(15.0, ge=0.0, le=50.0, description="Wind drift speed in km/h (0-50)")
    drain_silt_pct: float = Field(30.0, ge=0.0, le=100.0, description="Drain siltation level in % (0-100)")


class ReviewDecisionRequest(BaseModel):
    operator_note: Optional[str] = Field("", description="Operator review remarks")


forecasting_engine = ForecastingEngine()
verification_engine = CleanupVerificationEngine()
recurrence_engine = RecurrenceEngine()
simulator_engine = MonsoonSimulatorEngine()
dispatch_engine = SmartDispatchEngine()
HISTORICAL_DOSSIER_PATH = ROOT / "data" / "historical_urban_dossier.json"


# ── Static & Mobile Client Routes ────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home_redirect():
    """Welcome page with quick links to Mobile Simulator and API documentation."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
      <title>PlasticWatch Urban Intelligence Engine</title>
      <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-slate-900 text-slate-100 flex items-center justify-center min-h-screen p-4">
      <div class="max-w-lg w-full bg-slate-800 border border-slate-700 rounded-2xl p-6 shadow-2xl">
        <div class="flex items-center space-x-3 mb-4">
          <div class="w-10 h-10 rounded-xl bg-cyan-500 flex items-center justify-center text-slate-950 font-bold text-xl">♻️</div>
          <div>
            <h1 class="text-xl font-bold text-white">PlasticWatch Urban Intelligence</h1>
            <p class="text-xs text-cyan-400">CodeCraft Civic Environmental Platform</p>
          </div>
        </div>
        <p class="text-sm text-slate-300 mb-6 leading-relaxed">
          Zero-Hardware, API-First Rollout connecting citizen mobile reporting to AI drain triage, DBSCAN clustering, and municipal cleanup dispatch.
        </p>
        <div class="space-y-3">
          <a href="/mobile" target="_blank" class="block w-full py-3 px-4 bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold rounded-xl text-center shadow-lg transition">
            📱 Launch Mobile Simulator (PWA View)
          </a>
          <a href="/docs" target="_blank" class="block w-full py-2.5 px-4 bg-slate-700 hover:bg-slate-600 text-slate-200 font-semibold rounded-xl text-center transition">
            📖 OpenAPI Interactive Docs
          </a>
          <a href="/api/hotspots" target="_blank" class="block w-full py-2.5 px-4 bg-slate-700/60 hover:bg-slate-600 text-cyan-300 font-mono text-xs rounded-xl text-center transition">
            ⚡ GET /api/hotspots (DBSCAN Clustered)
          </a>
        </div>
      </div>
    </body>
    </html>
    """


@app.get("/mobile", response_class=HTMLResponse)
async def mobile_app():
    """Serves the responsive smartphone simulator web app."""
    if MOBILE_HTML_PATH.is_file():
        return HTMLResponse(content=MOBILE_HTML_PATH.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="Mobile client HTML not found")


@app.get("/static/samples/{filename}")
async def get_sample_image(filename: str):
    """Serves sample TACO roadside waste images for mobile testing."""
    file_path = SAMPLES_DIR / filename
    if file_path.is_file():
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Sample image not found")


@app.get("/static/uploads/{filename}")
async def get_uploaded_image(filename: str):
    """Serves citizen photos stored at report submission."""
    file_path = (UPLOADS_DIR / Path(filename).name).resolve()
    if file_path.parent == UPLOADS_DIR.resolve() and file_path.is_file():
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Uploaded image not found")


# ── Core Urban Intelligence Endpoints ─────────────────────────────────────────
@app.get("/api/reports")
async def list_reports():
    """Retrieve all active citizen reports."""
    reports = pipeline.get_reports()
    return {"status": "success", "count": len(reports), "reports": reports}


@app.post("/api/reports")
async def submit_report(request: ReportSubmissionRequest):
    """Citizen mobile reporting endpoint."""
    try:
        result = pipeline.submit_report(
            lat=request.lat,
            lon=request.lon,
            photo_filename=request.photo_filename,
            notes=request.notes or "",
            reporter_id=request.reporter_id,
            channel=request.channel,
        )
    except UnknownReporterError as exc:
        raise HTTPException(status_code=422, detail=str(exc.args[0]))
    return result


@app.post("/api/reports/upload")
async def submit_report_with_photo(
    lat: float = Form(..., ge=-90.0, le=90.0),
    lon: float = Form(..., ge=-180.0, le=180.0),
    notes: str = Form("", max_length=500),
    file: UploadFile = File(...),
    reporter_id: Optional[str] = Form(None, max_length=20),
    channel: Optional[str] = Form(None, max_length=30),
):
    """Citizen mobile reporting with the real photo (multipart). Blurry/dark captures go to review."""
    contents = await _read_upload(file)
    try:
        return pipeline.submit_report(
            lat=lat,
            lon=lon,
            photo_filename=file.filename or "live_camera_capture.jpg",
            notes=notes,
            image_bytes=contents,
            reporter_id=reporter_id,
            channel=channel,
        )
    except UnknownReporterError as exc:
        raise HTTPException(status_code=422, detail=str(exc.args[0]))
    except (OSError, ValueError) as exc:  # PIL.UnidentifiedImageError subclasses OSError
        raise HTTPException(status_code=400, detail=_unreadable_detail(exc))


@app.get("/api/review-queue")
async def get_review_queue():
    """Unverified / blurry / ambiguous citizen captures awaiting operator review."""
    queue = pipeline.get_review_queue()
    return {"status": "success", "count": len(queue), "reports": queue}


@app.post("/api/reports/{report_id}/approve")
async def approve_review_report(report_id: str, request: Optional[ReviewDecisionRequest] = None):
    """Operator approves a queued capture; it joins clustering and the cleanup queue."""
    try:
        report = pipeline.approve_report(report_id, (request.operator_note if request else "") or "")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    if report is None:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
    return {"status": "success", "report": report}


@app.post("/api/reports/{report_id}/reject")
async def reject_review_report(report_id: str, request: Optional[ReviewDecisionRequest] = None):
    """Operator rejects a queued capture; it is kept for audit but never dispatched."""
    try:
        report = pipeline.reject_report(report_id, (request.operator_note if request else "") or "")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    if report is None:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
    return {"status": "success", "report": report}


@app.post("/api/detect")
async def detect_uploaded_image(file: UploadFile = File(...)):
    """Run TACO waste object detection and severity estimation on an uploaded photo."""
    contents = await _read_upload(file)
    try:
        detection = detect_waste(contents, filename_hint=file.filename or "")
    except (OSError, ValueError) as exc:  # PIL.UnidentifiedImageError (e.g. HEIC from some phones)
        raise HTTPException(status_code=400, detail=_unreadable_detail(exc))
    # annotated_image (data URI) already carries the preview; skip the duplicate raw copy
    detection.pop("annotated_image_base64", None)
    return detection


@app.get("/api/hotspots")
async def get_hotspots(
    rain_override_mm: Optional[float] = None,
    rainfall_intensity_mmh: Optional[float] = Query(None, ge=0.0, le=75.0, description="Rainfall intensity override (mm/hr); omit for live Open-Meteo"),
    drain_silt_ratio: Optional[float] = Query(None, ge=0.0, le=1.0, description="Siltation override (0-1); omit for per-drain desilting records"),
    surface_slope_factor: float = Query(1.0, ge=1.0, le=1.5, description="1.0 flat, 1.25 moderate, 1.5 steep"),
    crew_eta_min: float = Query(45.0, ge=5.0, le=240.0, description="Expected crew arrival (min) for floating-debris drift"),
):
    """Run DBSCAN spatial clustering and 0-100 urban drain risk engine with live environment factors."""
    result = pipeline.run_pipeline(
        rainfall_override_mm=rain_override_mm,
        rainfall_intensity_mmh=rainfall_intensity_mmh,
        drain_silt_ratio=drain_silt_ratio,
        surface_slope_factor=surface_slope_factor,
        crew_eta_min=crew_eta_min,
    )
    return result


@app.get("/api/hotspots/{hotspot_id}/decision-support")
async def get_hotspot_decision_support(
    hotspot_id: str,
    rain_override_mm: Optional[float] = None,
    rainfall_intensity_mmh: Optional[float] = Query(None, ge=0.0, le=75.0),
    drain_silt_ratio: Optional[float] = Query(None, ge=0.0, le=1.0),
    surface_slope_factor: float = Query(1.0, ge=1.0, le=1.5),
    crew_eta_min: float = Query(45.0, ge=5.0, le=240.0),
):
    """XAI recommendation, ranking explanation, flow dynamics, vendor match, quote and 90-day fraud check."""
    data = pipeline.run_pipeline(
        rainfall_override_mm=rain_override_mm,
        rainfall_intensity_mmh=rainfall_intensity_mmh,
        drain_silt_ratio=drain_silt_ratio,
        surface_slope_factor=surface_slope_factor,
        crew_eta_min=crew_eta_min,
    )
    hotspots = data["hotspots"]
    target = next((h for h in hotspots if hotspot_id in (h["id"], h.get("site_key"))), None)
    if target is None:
        raise HTTPException(status_code=404, detail=f"Hotspot {hotspot_id} not found")
    return {
        "status": "success",
        "hotspot_id": target["id"],
        "site_key": target.get("site_key"),
        "total_score": target["total_score"],
        "priority": target["priority"],
        "recommendation": ExplainabilityEngine.recommend_action(target),
        "ranking_explanation": ExplainabilityEngine.explain_queue_position(target, hotspots),
        "landuse_attribution": target.get("landuse_attribution"),
        "flow_dynamics": target.get("flow_dynamics"),
        "smart_dispatch": dispatch_engine.assess_hotspot(target),
    }


@app.get("/api/explain")
async def explain_hotspot_ranking(
    a: str,
    b: str,
    rainfall_intensity_mmh: Optional[float] = Query(None, ge=0.0, le=75.0),
    drain_silt_ratio: Optional[float] = Query(None, ge=0.0, le=1.0),
    surface_slope_factor: float = Query(1.0, ge=1.0, le=1.5),
):
    """Plain-English explanation of why hotspot A ranks above (or below) hotspot B."""
    hotspots = pipeline.run_pipeline(
        rainfall_intensity_mmh=rainfall_intensity_mmh,
        drain_silt_ratio=drain_silt_ratio,
        surface_slope_factor=surface_slope_factor,
    )["hotspots"]
    by_id = {h["id"]: h for h in hotspots}
    if a not in by_id or b not in by_id:
        raise HTTPException(status_code=404, detail="Unknown hotspot id")
    return {"status": "success", "explanation": ExplainabilityEngine.explain_ranking(by_id[a], by_id[b])}


@app.get("/api/vendors")
async def list_vendors():
    """Pre-approved municipal sanitation contractors and their assigned zones."""
    return {"status": "success", "vendors": dispatch_engine.vendors}


@app.get("/api/landuse")
async def list_landuse_nodes():
    """Municipal land-use landmark nodes used for root-cause attribution."""
    return {"status": "success", "nodes": load_landuse_nodes()}


@app.get("/api/drains")
async def get_drains():
    """Return urban drainage and river network GeoJSON."""
    if DRAIN_GEOJSON.is_file():
        data = json.loads(DRAIN_GEOJSON.read_text(encoding="utf-8"))
        return data
    raise HTTPException(status_code=404, detail="Drain GeoJSON not found")


@app.post("/api/dispatch")
async def dispatch_cleanup(request: DispatchRequest):
    """Human-in-the-loop municipal work order dispatch and truck route optimizer.

    Each stop carries its vendor match, quote and 90-day fraud check; suspected
    duplicate billings are listed under payment_holds for field audit.
    """
    env = {
        "rainfall_intensity_mmh": request.rainfall_intensity_mmh,
        "drain_silt_ratio": request.drain_silt_ratio,
        "surface_slope_factor": request.surface_slope_factor,
        "crew_eta_min": request.crew_eta_min,
    }
    plan = pipeline.dispatch_work_order(request.hotspot_ids, **env)
    if request.truck_id:
        plan["truck_id"] = request.truck_id
    if request.crew_name:
        plan["crew_name"] = request.crew_name

    by_id = {h["id"]: h for h in pipeline.run_pipeline(**env)["hotspots"]}
    assessments = []
    for stop in plan.get("stops", []):
        h = by_id.get(stop.get("hotspot_id"))
        if h:
            assessments.append(dispatch_engine.assess_hotspot(h))
    plan["smart_dispatch"] = assessments
    plan["payment_holds"] = [a for a in assessments if not a["dispatch_allowed"]]
    record_work_order({
        "work_order_id": plan.get("work_order_id"),
        "channel": "API",
        "hotspot_ids": [a["hotspot_id"] for a in assessments],
        "vendors": sorted({a["vendor"].get("name") for a in assessments if a["vendor"].get("name")}),
        "total_quote_inr": round(sum((a["quote"] or {}).get("total_quote_inr", 0.0) for a in assessments), 2),
        "payment_holds": [a["hotspot_id"] for a in plan["payment_holds"]],
    })
    return plan


@app.get("/api/work-orders")
async def get_work_orders():
    """Issued work orders with vendor, quote and payment-hold status."""
    orders = list_work_orders()
    return {"status": "success", "count": len(orders), "work_orders": orders}


@app.get("/api/verification-cases")
async def get_verification_cases():
    """Per-hotspot cleanup verification work orders and their photo status."""
    return {"status": "success", "cases": list_verification_cases()}


@app.post("/api/verification-cases/{case_id}/photo")
async def upload_verification_photo(case_id: str, kind: str = Form("after"), file: UploadFile = File(...)):
    """Contractor uploads a before/after photo for a verification case."""
    data = await _read_upload(file)
    try:
        path = save_case_photo(case_id, kind, data, file.filename or "")
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unreadable image: {exc}")
    return {"status": "success", "case_id": case_id, "kind": kind, "stored_as": path.name}


@app.get("/api/verification-cases/{case_id}/photo/{kind}")
async def get_verification_photo(case_id: str, kind: str):
    """Before/after evidence photo, shown to the reporter whose report triggered the cleanup."""
    if kind not in ("before", "after"):
        raise HTTPException(status_code=400, detail="kind must be 'before' or 'after'")
    case = next((c for c in list_verification_cases() if c["case_id"] == case_id), None)
    if case is None or not case.get(f"{kind}_path"):
        raise HTTPException(status_code=404, detail="No such photo")
    return FileResponse(case[f"{kind}_path"])


@app.delete("/api/verification-cases/{case_id}/photo/{kind}")
async def delete_verification_photo(case_id: str, kind: str):
    """Remove a before/after photo so it can be re-shot."""
    try:
        removed = remove_case_photo(case_id, kind)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not removed:
        raise HTTPException(status_code=404, detail="No such photo")
    return {"status": "success", "case_id": case_id, "kind": kind}


def _settle_rewards(case: Dict[str, Any], final: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Release / hold reporter rewards for a decided case; never fails the verification response."""
    try:
        changed = incentive_engine.on_verification(case, final)
    except Exception:
        logger.exception("Incentive settlement failed for case %s", case.get("case_id"))
        return []
    return [{"reward_id": r["reward_id"], "reporter_id": r["reporter_id"], "state": r["state"], "amount_inr": r["amount_inr"]}
            for r in changed]


class OverrideRequest(BaseModel):
    decision: str = Field(..., description="PASS or FLAGGED FOR REVIEW")
    reason_code: str = Field(..., description="One of the codes from GET /api/verification-overrides/reasons")
    justification: str = Field(..., min_length=20, max_length=1000)
    operator: str = Field(..., min_length=2, max_length=80)


@app.get("/api/verification-overrides/reasons")
async def get_override_reasons():
    return {"status": "success", "reasons": OVERRIDE_REASONS}


@app.get("/api/verification-overrides")
async def get_overrides(case_id: Optional[str] = None):
    return {"status": "success", "overrides": list_overrides(case_id)}


@app.post("/api/verification-cases/{case_id}/override")
async def override_case(case_id: str, request: OverrideRequest):
    """Human-in-the-loop override of the AI verdict, bound to the exact photo pair."""
    case = next((c for c in list_verification_cases() if c["case_id"] == case_id), None)
    if case is None:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")
    if case["status"] != "READY":
        raise HTTPException(status_code=409, detail="Both photos are required before a decision can be overridden")
    ai = verification_engine.verify_cleanup(
        before_image=Path(case["before_path"]), after_image=Path(case["after_path"]),
        hotspot_id=case_id, site_name=case["site_name"], contractor_id=case["contractor"],
        site_lat=case["lat"], site_lon=case["lon"],
    )
    try:
        entry = record_override(case_id, ai, case["evidence_fingerprint"], request.decision,
                                request.reason_code, request.justification, request.operator)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"status": "success", "override": entry, "incentives": _settle_rewards(case, effective_decision(ai, entry))}


@app.get("/api/model-card")
async def get_model_card():
    """What the detector is, how it decides, and TACO statistics behind the taxonomy."""
    return {"status": "success", "model": model_card(), "taco": taco_annotation_stats()}


@app.post("/api/verification-cases/{case_id}/verify")
async def verify_case(case_id: str):
    """Run before/after verification for one case."""
    case = next((c for c in list_verification_cases() if c["case_id"] == case_id), None)
    if case is None:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")
    if case["status"] != "READY":
        raise HTTPException(status_code=409, detail=f"Case not ready: {case['status']}")
    result = verification_engine.verify_cleanup(
        before_image=Path(case["before_path"]),
        after_image=Path(case["after_path"]),
        hotspot_id=case_id,
        site_name=case["site_name"],
        contractor_id=case["contractor"],
        site_lat=case["lat"],
        site_lon=case["lon"],
    )
    result.pop("visuals", None)
    result["final"] = effective_decision(result, get_active_override(case_id, case["evidence_fingerprint"]))
    result["incentives"] = _settle_rewards(case, result["final"])
    return result


@app.get("/api/samples")
async def list_sample_catalog():
    """Catalog of preloaded TACO roadside waste images."""
    return {"samples": get_sample_images()}


@app.get("/api/taco/status")
async def get_taco_status():
    """TACO dataset facts and training pipeline status."""
    return dataset_status()


# ── Advanced Municipal Intelligence Endpoints (Features 1-5) ──────────────────
@app.get("/api/forecast")
async def get_hotspot_forecast(
    horizon_hours: int = 48,
    rain_override_mm: Optional[float] = None,
):
    """Predictive Hotspot Forecasting (24-72h) with escalation risk and preventive actions."""
    pipeline_data = pipeline.run_pipeline(rainfall_override_mm=rain_override_mm)
    hotspots = pipeline_data.get("hotspots", [])
    forecast = forecasting_engine.generate_hotspot_forecast(
        hotspots=hotspots,
        horizon_hours=horizon_hours,
        rainfall_scenario_mm=rain_override_mm,
    )
    return forecast


@app.post("/api/forecast")
async def post_hotspot_forecast(request: ForecastRequest):
    """Predictive Hotspot Forecasting via POST with custom scenario payload."""
    pipeline_data = pipeline.run_pipeline(rainfall_override_mm=request.rainfall_scenario_mm)
    hotspots = pipeline_data.get("hotspots", [])
    forecast = forecasting_engine.generate_hotspot_forecast(
        hotspots=hotspots,
        horizon_hours=request.horizon_hours,
        rainfall_scenario_mm=request.rainfall_scenario_mm,
    )
    return forecast


@app.post("/api/verify-cleanup")
async def verify_cleanup_endpoint(
    request: Optional[CleanupVerificationRequest] = None,
    before_file: Optional[UploadFile] = File(None),
    after_file: Optional[UploadFile] = File(None),
    hotspot_id: Optional[str] = Form(None),
    contractor_id: Optional[str] = Form(None),
):
    """AI Before/After Cleanup Verification comparing pre vs post clearance photos."""
    if before_file and after_file:
        b_bytes = await before_file.read()
        a_bytes = await after_file.read()
        result = verification_engine.verify_cleanup(
            before_image=b_bytes,
            after_image=a_bytes,
            hotspot_id=hotspot_id or "HS-UPLOAD",
            contractor_id=contractor_id or "PMC-SANITATION-CREW",
        )
        return result

    if request:
        result = verification_engine.verify_cleanup(
            before_image=request.before_image,
            after_image=request.after_image,
            hotspot_id=request.hotspot_id,
            site_name=request.site_name,
            contractor_id=request.contractor_id,
            operator_notes=request.operator_notes,
        )
        return result

    # Default fallback demo verification
    return verification_engine.verify_cleanup(
        before_image="sample_bottles_drain.jpg",
        after_image="sample_cleared_drain.jpg",
        hotspot_id="HOTSPOT-01",
        site_name="Shaniwar Wada Culvert",
    )


@app.get("/api/recurrence")
async def get_recurrence_intelligence(site_id: Optional[str] = None):
    """Historical recurrence metrics, return intervals, and automated root-cause hypotheses."""
    pipeline_data = pipeline.run_pipeline()
    report = recurrence_engine.analyze_recurrence(
        site_id_filter=site_id,
        active_hotspots=pipeline_data.get("hotspots", []),
    )
    return report


@app.get("/api/simulate")
async def get_monsoon_simulation(
    rainfall_intensity: float = 25.0,
    fleet: int = 4,
    delay_hours: float = 2.0,
    wind_speed_kmh: float = 15.0,
    drain_silt_pct: float = 30.0,
):
    """Monsoon What-If Decision Simulator modeling flood area, plastic swept, and inaction costs."""
    pipeline_data = pipeline.run_pipeline()
    return simulator_engine.simulate(
        rainfall_intensity_mmh=rainfall_intensity,
        available_fleet=fleet,
        action_delay_hours=delay_hours,
        base_hotspots=pipeline_data.get("hotspots", []),
        wind_speed_kmh=wind_speed_kmh,
        drain_silt_pct=drain_silt_pct,
    )


@app.post("/api/simulate")
async def post_monsoon_simulation(request: SimulationRequest):
    """Monsoon What-If Decision Simulator via POST with JSON body."""
    pipeline_data = pipeline.run_pipeline()
    return simulator_engine.simulate(
        rainfall_intensity_mmh=request.rainfall_intensity_mmh,
        available_fleet=request.available_fleet,
        action_delay_hours=request.action_delay_hours,
        base_hotspots=pipeline_data.get("hotspots", []),
        wind_speed_kmh=request.wind_speed_kmh,
        drain_silt_pct=request.drain_silt_pct,
    )


@app.get("/api/dossier")
async def get_historical_dossier():
    """Returns the Pune 30-day historical cleanup dossier and pilot location logs."""
    if HISTORICAL_DOSSIER_PATH.is_file():
        return json.loads(HISTORICAL_DOSSIER_PATH.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="Historical dossier file not found")


@app.get("/api/provenance")
async def get_provenance_definitions():
    """Evidence & Provenance Layer tier taxonomy (Observed, Derived, Inferred, Simulated)."""
    return {
        "status": "success",
        "provenance_framework": "PlasticWatch Defensible Civic Intelligence",
        "tiers": PROVENANCE_TIERS,
    }


# ── Reporters & Incentive Ledger (pay on verified clearance) ─────────────────
class ReporterRegistrationRequest(BaseModel):
    display_name: str = Field(..., min_length=2, max_length=80)
    channel: str = Field("citizen", description="citizen, waste_picker, sanitary_inspector, gig_rider, dashcam or rwa")
    phone: Optional[str] = Field(None, max_length=20, description="Stored only as a salted hash; same phone = same wallet")
    partner_org: Optional[str] = Field(None, max_length=80)
    ward: Optional[str] = Field(None, max_length=80)
    household_link: Optional[str] = Field(None, max_length=40, description="Property-tax ID or utility consumer number")
    program_id: Optional[str] = Field(None, max_length=40)


class PayoutDecisionRequest(BaseModel):
    operator: str = Field(..., min_length=2, max_length=80)
    note: str = Field(..., min_length=incentive_engine.MIN_OPERATOR_NOTE_CHARS, max_length=500)


def _backfill_ledger() -> None:
    """Make sure reporter-tagged reports (seed data, pre-ledger reports) have a reward row."""
    try:
        incentive_engine.backfill_from_reports(pipeline.get_reports())
    except Exception:
        logger.exception("Incentive ledger backfill failed")


@app.post("/api/reporters")
async def register_reporter(request: ReporterRegistrationRequest):
    """One-time 'Who are you?' registration; re-registering the same phone returns the same reporter."""
    try:
        reporter, created = incentive_engine.register_reporter(
            display_name=request.display_name, channel=request.channel, phone=request.phone,
            partner_org=request.partner_org, ward=request.ward, household_link=request.household_link,
            program_id=request.program_id,
        )
    except IncentiveError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"status": "success", "created": created, "reporter": incentive_engine.public_reporter(reporter)}


@app.get("/api/reporters")
async def list_reporters(channel: Optional[str] = None):
    reporters = [incentive_engine.public_reporter(r) for r in incentive_engine.list_reporters()
                 if channel is None or r.get("channel") == channel]
    return {"status": "success", "count": len(reporters), "reporters": reporters}


@app.get("/api/reporters/{reporter_id}")
async def get_reporter(reporter_id: str):
    reporter = incentive_engine.get_reporter(reporter_id)
    if reporter is None:
        raise HTTPException(status_code=404, detail=f"Reporter {reporter_id} not found")
    return {"status": "success", "reporter": incentive_engine.public_reporter(reporter)}


@app.get("/api/reporters/{reporter_id}/wallet")
async def get_reporter_wallet(reporter_id: str):
    """Pending / escrowed / released ₹ and each report's stage: reported → hotspot → dispatched → verified."""
    _backfill_ledger()
    wallet = incentive_engine.wallet(reporter_id)
    if wallet is None:
        raise HTTPException(status_code=404, detail=f"Reporter {reporter_id} not found")
    photos = {r.get("id"): r.get("photo_filename") for r in pipeline.get_reports()}
    for reward in wallet["rewards"]:
        reward["photo_url"] = _report_photo_url(photos.get(reward["report_id"]))
        evidence = reward.get("evidence") or {}
        case_id = evidence.get("verification_case")
        reward["verification_photos"] = (
            {k: f"/api/verification-cases/{case_id}/photo/{k}" for k in ("before", "after")}
            if case_id and evidence.get("decision") == "PASS" else None
        )
    return {"status": "success", **wallet}


def _report_photo_url(photo_filename: Optional[str]) -> Optional[str]:
    path = resolve_photo_path(photo_filename)
    if path is None:
        return None
    return f"/static/{'uploads' if path.parent.name == 'uploads' else 'samples'}/{path.name}"


@app.get("/api/incentives/programs")
async def get_incentive_programs():
    return {"status": "success", "programs": incentive_engine.list_programs()}


@app.get("/api/incentives/ledger")
async def get_incentive_ledger(
    reporter_id: Optional[str] = None,
    state: Optional[str] = Query(None, description="SUBMITTED, PENDING, ESCROWED, RELEASED, HELD or VOID"),
    history: bool = Query(False, description="Return every append-only state-change row instead of current state"),
):
    _backfill_ledger()
    if state and state.upper() not in incentive_engine.STATES:
        raise HTTPException(status_code=422, detail=f"state must be one of {incentive_engine.STATES}")
    if history:
        rows = [e for e in incentive_engine.ledger_events()
                if (reporter_id is None or e.get("reporter_id") == reporter_id) and (state is None or e.get("state") == state.upper())]
    else:
        rows = incentive_engine.list_rewards(reporter_id=reporter_id, state=state.upper() if state else None)
    return {"status": "success", "count": len(rows), "entries": rows}


@app.get("/api/incentives/summary")
async def get_incentive_summary():
    """Ledger KPIs: by channel, by state, payout queue and ₹ per verified clearance."""
    _backfill_ledger()
    return {"status": "success", **incentive_engine.summary()}


@app.get("/api/incentives/adoption")
async def get_incentive_adoption(days: Optional[int] = Query(7, ge=1, le=365, description="Activity window in days")):
    """Reporters, reports and drain coverage (share of drain length within 50 m of a report) per channel."""
    return {"status": "success", **incentive_engine.adoption_stats(pipeline.get_reports(), days)}


@app.get("/api/incentives/{reward_id}")
async def get_incentive_entry(reward_id: str):
    reward = incentive_engine.get_reward(reward_id)
    if reward is None:
        raise HTTPException(status_code=404, detail=f"Reward {reward_id} not found")
    return {"status": "success", "reward": reward, "history": incentive_engine.ledger_events(reward_id),
            "timeline": incentive_engine.timeline(reward)}


def _payout_action(action, reward_id: str, request: PayoutDecisionRequest) -> Dict[str, Any]:
    try:
        reward = action(reward_id, request.operator, request.note)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Reward {reward_id} not found")
    except IncentiveError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"status": "success", "reward": reward, "message": incentive_engine.status_message(reward)}


@app.post("/api/incentives/{reward_id}/release")
async def release_incentive(reward_id: str, request: PayoutDecisionRequest):
    """Operator resolves a HELD reward. It pays only if cleanup verification passed; otherwise it resumes waiting."""
    return _payout_action(incentive_engine.release_reward, reward_id, request)


@app.post("/api/incentives/{reward_id}/hold")
async def hold_incentive(reward_id: str, request: PayoutDecisionRequest):
    """Operator freezes an open reward pending investigation."""
    return _payout_action(incentive_engine.hold_reward, reward_id, request)


# ── Passive dashcam ingestion (buses, garbage vans) ───────────────────────────
class RouteFrame(BaseModel):
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    captured_at: Optional[str] = Field(None, max_length=40, description="ISO-8601 capture time")
    image_base64: str = Field(..., min_length=16, description="JPEG/PNG frame, base64 or data URI")


class RouteFramesRequest(BaseModel):
    vehicle_id: str = Field(..., min_length=2, max_length=40, examples=["PMPML-BUS-158"])
    frames: List[RouteFrame] = Field(..., min_length=1, max_length=route_ingest_engine.MAX_FRAMES_PER_BATCH)


def _decode_frame(index: int, data: str) -> bytes:
    payload = data.split(",", 1)[1] if data.startswith("data:") else data
    try:
        raw = base64.b64decode(payload, validate=True)
    except (ValueError, TypeError):
        raise HTTPException(status_code=422, detail=f"Frame {index}: image_base64 is not valid base64")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Frame {index} larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    return raw


@app.get("/api/ingest/routes")
def list_replay_routes():
    """Simulated dashcam routes available for replay."""
    return {"status": "success", "routes": route_ingest_engine.list_routes()}


@app.post("/api/ingest/route-frames")
def ingest_route_frames(request: RouteFramesRequest):
    """A registered vehicle uploads GPS-tagged dashcam frames; confident detections near drains become reports."""
    frames = [
        route_ingest_engine.Frame(lat=f.lat, lon=f.lon, captured_at=f.captured_at, image_bytes=_decode_frame(i, f.image_base64))
        for i, f in enumerate(request.frames)
    ]
    try:
        return route_ingest_engine.ingest_frames(pipeline, request.vehicle_id, frames)
    except route_ingest_engine.UnknownVehicleError as exc:
        raise HTTPException(status_code=422, detail=str(exc.args[0]))


@app.post("/api/ingest/replay/{route_id}")
def replay_dashcam_route(route_id: str):
    """Demo: drive a simulated route with sample photos. Reports are tagged provenance=simulated."""
    try:
        return route_ingest_engine.replay_route(pipeline, route_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown route {route_id}")
    except route_ingest_engine.UnknownVehicleError as exc:
        raise HTTPException(status_code=422, detail=str(exc.args[0]))


# ── Ward Drain Score ─────────────────────────────────────────────────────────
_case_ai_cache: Dict[tuple, Dict[str, Any]] = {}


def _case_outcomes() -> List[Dict[str, Any]]:
    """Final decision per READY verification case; AI results cached per photo pair."""
    outcomes = []
    for case in list_verification_cases():
        if case["status"] != "READY":
            continue
        key = (case["case_id"], case["evidence_fingerprint"])
        ai = _case_ai_cache.get(key)
        if ai is None:
            try:
                ai = verification_engine.verify_cleanup(
                    before_image=Path(case["before_path"]), after_image=Path(case["after_path"]),
                    hotspot_id=case["case_id"], site_name=case["site_name"], contractor_id=case["contractor"],
                    site_lat=case["lat"], site_lon=case["lon"],
                )
            except Exception:
                logger.exception("Verification failed for case %s; left out of the scorecard", case["case_id"])
                continue
            ai.pop("visuals", None)
            _case_ai_cache[key] = ai
        final = effective_decision(ai, get_active_override(case["case_id"], case["evidence_fingerprint"]))
        outcomes.append({"case_id": case["case_id"], "lat": case["lat"], "lon": case["lon"], "decision": final["decision"]})
    return outcomes


@app.get("/api/wards/scorecard")
def get_ward_scorecard(days: int = Query(30, ge=1, le=365, description="Window for coverage and active reporters")):
    """Ward Drain Score: coverage, verified-clear rate, time to clear and recurrence per ward, ranked."""
    _backfill_ledger()
    return {"status": "success", **ward_engine.ward_scorecard(
        pipeline.get_reports(), incentive_engine.list_rewards(), _case_outcomes(), days=days)}


@app.get("/api/wards/geojson")
def get_ward_geojson():
    """Ward zones used for scoring (nearest vendor anchor regions) as GeoJSON polygons."""
    return ward_engine.ward_polygons()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=True)

