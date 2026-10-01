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

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.clustering_engine import DBSCANClusteringEngine
from backend.drain_risk_engine import DrainRiskEngine
from backend.plasticwatch_pipeline import (
    PlasticWatchUrbanPipeline,
    get_seed_reports,
    run_plasticwatch,
)
from backend.route_optimizer import RouteOptimizer
from backend.taco_adapter import dataset_status, detect_waste, get_sample_images

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


# ── Request / Response Models ────────────────────────────────────────────────
class ReportSubmissionRequest(BaseModel):
    lat: float = Field(..., description="Latitude of waste observation")
    lon: float = Field(..., description="Longitude of waste observation")
    photo_filename: str = Field("sample_bottles_drain.jpg", description="Sample photo filename or reference")
    notes: Optional[str] = Field("", description="Optional citizen field notes")


class DispatchRequest(BaseModel):
    hotspot_ids: List[str] = Field(default_factory=list, description="List of hotspot IDs to dispatch")
    truck_id: Optional[str] = Field("PMC-TRUCK-04", description="Assigned municipal truck ID")
    crew_name: Optional[str] = Field("PMC Rapid Sanitation Crew 2", description="Assigned cleanup crew")


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


# ── Core Urban Intelligence Endpoints ─────────────────────────────────────────
@app.get("/api/reports")
async def list_reports():
    """Retrieve all active citizen reports."""
    reports = pipeline.get_reports()
    return {"status": "success", "count": len(reports), "reports": reports}


@app.post("/api/reports")
async def submit_report(request: ReportSubmissionRequest):
    """Citizen mobile reporting endpoint."""
    result = pipeline.submit_report(
        lat=request.lat,
        lon=request.lon,
        photo_filename=request.photo_filename,
        notes=request.notes or "",
    )
    return result


@app.post("/api/detect")
async def detect_uploaded_image(file: UploadFile = File(...)):
    """Run TACO waste object detection and severity estimation on an uploaded photo."""
    contents = await file.read()
    detection = detect_waste(contents, filename_hint=file.filename or "")
    return detection


@app.get("/api/hotspots")
async def get_hotspots(rain_override_mm: Optional[float] = None):
    """Run DBSCAN spatial clustering and 0-100 urban drain risk engine."""
    result = pipeline.run_pipeline(rainfall_override_mm=rain_override_mm)
    return result


@app.get("/api/drains")
async def get_drains():
    """Return urban drainage and river network GeoJSON."""
    if DRAIN_GEOJSON.is_file():
        data = json.loads(DRAIN_GEOJSON.read_text(encoding="utf-8"))
        return data
    raise HTTPException(status_code=404, detail="Drain GeoJSON not found")


@app.post("/api/dispatch")
async def dispatch_cleanup(request: DispatchRequest):
    """Human-in-the-loop municipal work order dispatch and truck route optimizer."""
    plan = pipeline.dispatch_work_order(request.hotspot_ids)
    if request.truck_id:
        plan["truck_id"] = request.truck_id
    if request.crew_name:
        plan["crew_name"] = request.crew_name
    return plan


@app.get("/api/samples")
async def list_sample_catalog():
    """Catalog of preloaded TACO roadside waste images."""
    return {"samples": get_sample_images()}


@app.get("/api/taco/status")
async def get_taco_status():
    """TACO dataset facts and training pipeline status."""
    return dataset_status()
