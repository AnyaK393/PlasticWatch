"""PlasticWatch: Urban Civic Environmental Intelligence Command Center.

Operator decision-support dashboard featuring:
  - Left Panel: Cleanup Queue ranked by 0-100 Risk Score (CRITICAL/HIGH/MODERATE)
  - Center Panel: Leaflet map displaying urban waterways, drains, citizen report pins,
                  consolidated DBSCAN buffer rings, and dispatched truck route
  - Right Drawer: Hotspot detail with DBSCAN merged report count, exploded 0-100 score
                  breakdown, and Human-in-the-Loop "Verify & Dispatch" action
  - Citizen Mobile Simulator tab styled in paper-ish light editorial theme
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import altair as alt
import folium
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.explainability_engine import ExplainabilityEngine
from backend.forecasting_engine import ForecastingEngine
from backend.plasticwatch_pipeline import PlasticWatchUrbanPipeline, resolve_photo_path, run_plasticwatch
from backend.provenance import (
    PROVENANCE_TIERS,
    get_provenance_meta,
    render_provenance_badge,
)
from backend.recurrence_engine import RecurrenceEngine
from backend.route_optimizer import RouteOptimizer
from backend.simulator_engine import MonsoonSimulatorEngine
from backend.smart_dispatch_engine import SmartDispatchEngine, list_work_orders, record_work_order
from backend.taco_adapter import (
    dataset_status,
    detect_waste,
    evaluate_image_quality,
    get_sample_images,
    model_card,
    taco_annotation_stats,
)
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

# ── Page Configuration & Theming ─────────────────────────────────────────────
st.set_page_config(
    page_title="PlasticWatch | Urban Environmental Intelligence",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400..700;1,6..72,400..600&family=JetBrains+Mono:wght@400;500;600&display=swap');

  .stApp { 
    background-color: #07131a; 
    color: #e2e8f0; 
  }
  
  /* Ample breathing room below Streamlit top navigation bar */
  .block-container { 
    padding-top: 4.6rem !important; 
    padding-bottom: 3.5rem !important; 
    padding-left: 2.2rem !important;
    padding-right: 2.2rem !important;
    max-width: 100% !important;
  }

  /* Spacious Hero Banner */
  .hero-card {
    padding: 1.6rem 2rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #092832 0%, #0d1e26 100%);
    border: 1px solid #1a4954;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.35);
    margin-bottom: 1.4rem;
  }
  
  .hero-kicker {
    color: #38bdf8;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
  }

  .hero-title {
    margin: 0;
    font-size: 1.95rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.25;
  }

  .hero-sub {
    font-size: 0.88rem;
    color: #94a3b8;
    margin-top: 0.4rem;
    line-height: 1.5;
  }

  /* Spacious KPI Stat Cards */
  .stat-card {
    padding: 1.1rem 1.25rem;
    border-radius: 14px;
    background: #0d2029;
    border: 1px solid #1b414f;
    min-height: 98px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }
  
  .stat-label {
    font-size: 0.74rem;
    letter-spacing: 0.06em;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
  }

  .stat-val {
    font-size: 1.9rem;
    font-weight: 800;
    margin: 0.25rem 0;
    line-height: 1.1;
  }

  .stat-note {
    font-size: 0.77rem;
    color: #64748b;
  }

  /* Left Queue Cards with breathing room */
  .queue-card {
    padding: 0.9rem 1rem;
    border-radius: 12px;
    background: #0d212a;
    border: 1px solid #1c4250;
    margin-bottom: 0.7rem;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  .queue-card:hover {
    border-color: #38bdf8;
    background: #12303c;
    transform: translateY(-1px);
  }
  .queue-card-active {
    border-color: #38bdf8 !important;
    background: #153745 !important;
    box-shadow: 0 0 14px rgba(56, 189, 248, 0.3);
  }

  /* Badges */
  .badge-critical {
    background: rgba(239, 68, 68, 0.22);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.55);
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.74rem;
    font-weight: 700;
  }
  .badge-high {
    background: rgba(245, 158, 11, 0.22);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.55);
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.74rem;
    font-weight: 700;
  }
  .badge-moderate {
    background: rgba(56, 189, 248, 0.22);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.55);
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.74rem;
    font-weight: 700;
  }
  .badge-merged {
    background: rgba(168, 85, 247, 0.22);
    color: #c084fc;
    border: 1px solid rgba(168, 85, 247, 0.55);
    padding: 3px 7px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 600;
  }
  .badge-observed {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.55);
    padding: 2px 7px;
    border-radius: 6px;
    font-size: 0.74rem;
    font-weight: 700;
  }
  .badge-derived {
    background: rgba(56, 189, 248, 0.2);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.55);
    padding: 2px 7px;
    border-radius: 6px;
    font-size: 0.74rem;
    font-weight: 700;
  }
  .badge-inferred {
    background: rgba(251, 191, 36, 0.2);
    color: #fbbf24;
    border: 1px solid rgba(251, 191, 36, 0.55);
    padding: 2px 7px;
    border-radius: 6px;
    font-size: 0.74rem;
    font-weight: 700;
  }
  .badge-simulated {
    background: rgba(192, 132, 252, 0.2);
    color: #c084fc;
    border: 1px solid rgba(192, 132, 252, 0.55);
    padding: 2px 7px;
    border-radius: 6px;
    font-size: 0.74rem;
    font-weight: 700;
  }
  .disclaimer-box {
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.5);
    border-radius: 10px;
    padding: 12px 16px;
    color: #fde68a;
    font-size: 0.86rem;
    margin: 12px 0;
  }
  .feature-panel {
    background: #0d2029;
    border: 1px solid #1c4250;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
  }
  .tiny-text { font-size: 0.77rem; color: #94a3b8; }

  /* Paper Theme Mobile Simulator inside Streamlit */
  .paper-mobile-frame {
    max-width: 410px;
    margin: 0 auto;
    background: #fffdfa;
    border: 2px solid #e3d7c3;
    border-radius: 28px;
    padding: 16px;
    box-shadow: 0 18px 45px rgba(50, 40, 25, 0.18), 0 2px 8px rgba(50, 40, 25, 0.08);
    font-family: 'Newsreader', Georgia, serif;
    color: #211d1a;
  }
  .paper-card {
    background: #faf6ee;
    border: 1px solid #e5dac7;
    border-radius: 12px;
    padding: 12px;
  }
  .paper-inner-box {
    background: #fffdf9;
    border: 1px solid #e3d7c3;
    border-radius: 8px;
    padding: 8px 12px;
  }
  .font-mono-code {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
  }
</style>
""", unsafe_allow_html=True)


# ── Sidebar & Map API Key Configuration ──────────────────────────────────────
st.sidebar.title("♻️ PlasticWatch Civic")
st.sidebar.caption("Urban Environmental Intelligence • Pune Pilot")

st.sidebar.markdown("---")
st.sidebar.subheader("🗺️ Map Tiles & API Key")
map_tile_choice = st.sidebar.selectbox(
    "Active Map Basemap Layer",
    [
        "OpenStreetMap (Free & Open • No Key Needed)",
        "CARTO Dark Matter (Open CDN • High Contrast)",
        "CARTO Voyager (Open CDN • Clean Light)",
        "ESRI Satellite Aerial (Open CDN)",
    ],
    index=0,
    help="PlasticWatch runs on open public tile servers that do NOT require any API key.",
)

with st.sidebar.expander("ℹ️ Why did the map mention an API key?"):
    st.markdown("""
    **No API key is required to run PlasticWatch.**
    
    1. **Default Free Open Basemap:** We configure direct, open tile layers (**OpenStreetMap & CARTO CDN**) with zero API key or billing needed.
    2. **Where to get a key (Optional):** If your organization uses private Mapbox or Stadia styles:
       - **Mapbox:** Sign up at [mapbox.com](https://www.mapbox.com) for a free public token (`pk.eyJ1...`).
       - **Stadia Maps:** Sign up at [stadiamaps.com](https://stadiamaps.com).
    3. **Where to put it:** You can enter it in the input below or set `MAP_API_KEY` in your `.env` file.
    """)

custom_map_key = st.sidebar.text_input(
    "Optional Custom Mapbox/Stadia Token",
    type="password",
    help="Optional: Leave blank to use free OpenStreetMap/CARTO tiles.",
)
if custom_map_key:
    st.sidebar.success("Custom Map Token Registered.")
else:
    st.sidebar.info("Using Free Open Basemap (100% Key-Free).")

st.sidebar.markdown("---")
# Live environment: rainfall from Open-Meteo (current hour) and siltation from each drain's
# desilting record. The Triage tab's Storm Scenario panel can override them for drills/demos.
storm_on = bool(st.session_state.get("storm_override", False))
live_env_kwargs = {
    "rainfall_intensity_mmh": float(st.session_state.get("storm_rain", 25.0)) if storm_on else None,
    "drain_silt_ratio": (
        st.session_state.get("storm_silt", 60) / 100.0
        if storm_on and st.session_state.get("storm_silt_override", False) else None
    ),
}

st.sidebar.markdown("---")
st.sidebar.markdown("**Simulation Safeguards**")
st.sidebar.caption("AI detects waste probabilities and hydrological exposure. Human operators verify site conditions before dispatch.")


# ── Pipeline & State Management ──────────────────────────────────────────────
if "pipeline_instance" not in st.session_state:
    st.session_state.pipeline_instance = PlasticWatchUrbanPipeline()

if "selected_hotspot_id" not in st.session_state:
    st.session_state.selected_hotspot_id = "HOTSPOT-01"

if "dispatched_work_orders" not in st.session_state:
    st.session_state.dispatched_work_orders = {}

if "active_route_plan" not in st.session_state:
    st.session_state.active_route_plan = None

if "rain_override" not in st.session_state:
    st.session_state.rain_override = None

if "forecasting_engine_instance" not in st.session_state:
    st.session_state.forecasting_engine_instance = ForecastingEngine()

if "verification_engine_instance" not in st.session_state:
    st.session_state.verification_engine_instance = CleanupVerificationEngine()

if "recurrence_engine_instance" not in st.session_state:
    st.session_state.recurrence_engine_instance = RecurrenceEngine()

if "simulator_engine_instance" not in st.session_state:
    st.session_state.simulator_engine_instance = MonsoonSimulatorEngine()

if "signed_off_invoices" not in st.session_state:
    st.session_state.signed_off_invoices = set()

if "dispatch_engine_instance" not in st.session_state:
    st.session_state.dispatch_engine_instance = SmartDispatchEngine()

if "fraud_audit_overrides" not in st.session_state:
    st.session_state.fraud_audit_overrides = {}

if "selected_site_key" not in st.session_state:
    st.session_state.selected_site_key = None

forecasting_engine = st.session_state.forecasting_engine_instance
verification_engine = st.session_state.verification_engine_instance
recurrence_engine = st.session_state.recurrence_engine_instance
simulator_engine = st.session_state.simulator_engine_instance
dispatch_engine = st.session_state.dispatch_engine_instance

pipeline = st.session_state.pipeline_instance
pipeline_data = pipeline.run_pipeline(rainfall_override_mm=st.session_state.rain_override, **live_env_kwargs)
if storm_on:
    live_data = pipeline.run_pipeline(rainfall_override_mm=st.session_state.rain_override)
    live_rank = {h["site_key"]: i for i, h in enumerate(live_data["hotspots"], 1)}
    live_score = {h["site_key"]: h["total_score"] for h in live_data["hotspots"]}
    live_env = live_data["environment"]
else:
    live_rank, live_score, live_env = {}, {}, pipeline_data["environment"]
hotspots = pipeline_data["hotspots"]
reports = pipeline_data["reports"]
metrics = pipeline_data["metrics"]
cluster_telemetry = pipeline_data["cluster_telemetry"]
review_queue = pipeline_data["review_queue"]

# Keep the operator on the same physical site when live sliders re-rank HOTSPOT-NN ids
if st.session_state.selected_site_key:
    same_site = next((h for h in hotspots if h.get("site_key") == st.session_state.selected_site_key), None)
    if same_site:
        st.session_state.selected_hotspot_id = same_site["id"]

# Ensure selected hotspot exists
if not any(h["id"] == st.session_state.selected_hotspot_id for h in hotspots):
    if hotspots:
        st.session_state.selected_hotspot_id = hotspots[0]["id"]


# ── Top Hero & Key Metric Cards (Spacious & Clean) ───────────────────────────
st.markdown(f"""
<div class="hero-card">
  <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 24px;">
    <div style="flex: 1;">
      <div class="hero-kicker">
        CodeCraft • Urban Civic Environmental Intelligence • Zero-Hardware Rollout
      </div>
      <h1 class="hero-title">
        Municipal Storm Drain Clogging Command Center
      </h1>
      <div class="hero-sub">
        Pune Municipal Corporation (PMC) Pilot • Real-Time Citizen Ingestion • DBSCAN Spatial Consolidation • Hydrological Risk Prioritization
      </div>
    </div>
    <div style="text-align: right; flex-shrink: 0; padding-top: 4px;">
      <div class="badge-critical" style="font-size: 0.82rem; padding: 6px 12px; display: inline-flex; align-items: center; gap: 6px;">
        <span style="font-size: 0.95rem;">🌧️</span> <b>24h Rain Alert: {pipeline_data['rainfall_forecast_24h_mm']:.1f} mm</b>
      </div>
      <div class="tiny-text" style="margin-top: 5px;">{'⛈️ STORM SCENARIO' if storm_on else 'LIVE'} • Rain {pipeline_data['environment']['rainfall_intensity_mmh']:.1f} mm/h ({pipeline_data['environment']['rainfall_source']}) • Silt {pipeline_data['environment']['silt_range'][0]:.0%}–{pipeline_data['environment']['silt_range'][1]:.0%}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# 5 Spacious KPI Cards
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""
    <div class="stat-card">
      <div class="stat-label">Consolidated Hotspots</div>
      <div class="stat-val" style="color: #38bdf8;">{metrics['active_hotspots']}</div>
      <div class="stat-note">{metrics['merged_clusters']} merged via DBSCAN</div>
    </div>
    """, unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="stat-card">
      <div class="stat-label">Citizen Reports</div>
      <div class="stat-val" style="color: #c084fc;">{metrics['citizen_reports']}</div>
      <div class="stat-note">Mobile photo captures</div>
    </div>
    """, unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class="stat-card">
      <div class="stat-label">Drain Choke Hazards</div>
      <div class="stat-val" style="color: #f87171;">{metrics['drain_proximity_alerts']}</div>
      <div class="stat-note">Sites &le; 50 m to drain inlet</div>
    </div>
    """, unsafe_allow_html=True)
with k4:
    st.markdown(f"""
    <div class="stat-card">
      <div class="stat-label">Critical Priority</div>
      <div class="stat-val" style="color: #ef4444;">{metrics['critical_sites']}</div>
      <div class="stat-note">Score &ge; 70 (Immediate action)</div>
    </div>
    """, unsafe_allow_html=True)
with k5:
    st.markdown(f"""
    <div class="stat-card">
      <div class="stat-label">AI Detection Conf.</div>
      <div class="stat-val" style="color: #34d399;">{metrics['mean_detection_confidence']:.0%}</div>
      <div class="stat-note">TACO YOLOv8 mean score</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div style="margin-top:12px; padding:10px 16px; border-radius:12px; background:#0b1c24; border:1px solid #1a3c48;
            display:flex; flex-wrap:wrap; gap:22px; align-items:center; font-size:0.84rem; color:#cbd5e1;">
  <span style="font-weight:800; color:#c084fc;">🧬 DBSCAN Deduplication Telemetry</span>
  <span><b>{cluster_telemetry['total_reports']}</b> citizen reports → <b>{cluster_telemetry['consolidated_hotspots']}</b> hotspots</span>
  <span>Compression <b style="color:#38bdf8;">{cluster_telemetry['compression_ratio_pct']:.1f}%</b></span>
  <span>Duplicate dispatches prevented <b style="color:#f59e0b;">{cluster_telemetry['prevented_duplicate_dispatches']}</b></span>
  <span>Est. civic savings <b style="color:#34d399;">₹{cluster_telemetry['estimated_civic_savings_inr']:,.0f}</b>
    <span class="tiny-text">(₹{cluster_telemetry['cost_per_dispatch_inr']:,.0f}/redundant truck)</span></span>
  <span style="margin-left:auto;">⚠️ Awaiting review: <b style="color:#fbbf24;">{len(review_queue)}</b></span>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)


# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab_command, tab_tender, tab_forecast, tab_verify, tab_recurrence, tab_simulator, tab_provenance = st.tabs([
    "🏛️ Triage & Live Map",
    "📑 Tendering & Work Orders",
    "🔮 Predictive Forecasting (24–72h)",
    "📸 AI Cleanup Verification",
    "🔁 Recurrence & Root Cause",
    "☔ Monsoon What-If Simulator",
    "🔎 Evidence & Provenance Layer",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: 3-PANEL MUNICIPAL COMMAND CENTER
# ══════════════════════════════════════════════════════════════════════════════
with tab_command:
    # ── Live environment strip + Storm Scenario override ──────────────────────
    env_now = pipeline_data["environment"]
    sc1, sc2, sc3, sc4 = st.columns([1.15, 1.35, 1.35, 2.6], gap="medium")
    with sc1:
        st.toggle(
            "⛈️ Storm Scenario",
            key="storm_override",
            help="Off = live Open-Meteo rainfall + each drain's real siltation. On = drill a storm and watch the queue re-rank.",
        )
    with sc2:
        st.slider("🌧️ Rainfall intensity (mm/h)", 0.0, 60.0, 25.0, 2.5, key="storm_rain", disabled=not storm_on,
                  help="2.5 moderate • 7.5 heavy • 35+ very heavy burst. Amplifies drain-proximity risk by (1 + rain/20).")
    with sc3:
        st.toggle("Override siltation", key="storm_silt_override", disabled=not storm_on,
                  help="Off = each drain's own silt level from its desilting date.")
        st.slider("🍂 Siltation (%)", 0, 90, 60, 5, key="storm_silt",
                  disabled=not (storm_on and st.session_state.get("storm_silt_override", False)), label_visibility="collapsed")
    with sc4:
        if storm_on:
            moves = []
            for i, h in enumerate(hotspots, 1):
                before = live_rank.get(h["site_key"])
                if before and before != i:
                    arrow = "▲" if before > i else "▼"
                    moves.append(f"{arrow} {h['nearest_drain_name'].split(' ')[0]} {before}→{i}")
            escalated = sum(1 for h in hotspots if ExplainabilityEngine.recommend_action(h)["action_type"] == "IMMEDIATE_CLEANUP_DISPATCH")
            st.markdown(f"""
            <div style="background:rgba(239,68,68,0.10); border:1px solid #ef4444; border-radius:10px; padding:8px 12px; font-size:0.8rem; color:#fecaca;">
              <b>⛈️ STORM DRILL ACTIVE</b> • {env_now['rainfall_intensity_mmh']:.1f} mm/h • silt {'override ' + format(env_now['drain_silt_ratio'], '.0%') if env_now['drain_silt_ratio'] is not None else 'per drain'}<br>
              Re-ranking vs live: {', '.join(moves) if moves else 'order unchanged (scores rose uniformly)'} • <b>{escalated}</b> site(s) escalate to IMMEDIATE DISPATCH
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:#0b1c24; border:1px solid #1a3c48; border-radius:10px; padding:8px 12px; font-size:0.8rem; color:#cbd5e1;">
              <b style="color:#34d399;">● LIVE</b> Rain <b>{env_now['rainfall_intensity_mmh']:.1f} mm/h</b> ({env_now['rainfall_source']}) •
              Drain siltation <b>{env_now['silt_range'][0]:.0%}–{env_now['silt_range'][1]:.0%}</b> from desilting records.<br>
              <span class="tiny-text">Both feed the 35% drain-proximity × environment term; immediate dispatch triggers at ≥ 10 mm/h.</span>
            </div>
            """, unsafe_allow_html=True)

    col_left, col_center, col_right = st.columns([1.18, 2.12, 1.42], gap="large")

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 1: LEFT CLEANUP QUEUE (Ranked 0-100)
    # ──────────────────────────────────────────────────────────────────────────
    with col_left:
        queue_view = st.radio(
            "Queue View",
            ["📋 Cleanup Queue", f"⚠️ Unverified / Blurry Reports Queue ({len(review_queue)})"],
            label_visibility="collapsed",
            key="queue_view_choice",
        )
        show_review_queue = queue_view.startswith("⚠️")

        if show_review_queue:
            st.markdown("### ⚠️ Operator Review Queue")
            st.caption("Blurry, dark or AI-unconfirmed captures. Held out of DBSCAN and dispatch until approved.")
            if not review_queue:
                st.success("✅ No captures awaiting review.")
            for rq in reversed(review_queue):
                rq_id = rq["id"]
                rq_reasons = ", ".join(rq.get("review_reasons", [])) or "Flagged for operator verification"
                rq_q = rq.get("image_quality", {}) or {}
                st.markdown(f"""
                <div class="queue-card" style="border-color:#b45309;">
                  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:5px;">
                    <span style="font-weight:800; font-size:0.95rem; color:#f8fafc;">{rq_id}</span>
                    <span class="badge-high">NEEDS REVIEW</span>
                  </div>
                  <div style="font-size:0.78rem; color:#fde68a; margin-bottom:4px;">⚠️ {rq_reasons}</div>
                  <div class="tiny-text">📍 {rq['lat']:.5f}, {rq['lon']:.5f} • {rq.get('submitted_at', '')[:16].replace('T', ' ')}</div>
                  <div class="tiny-text">Luminance {rq_q.get('mean_luminance', '–')} • Sharpness {rq_q.get('blur_variance', '–')} • AI conf {rq.get('plastic_confidence', 0):.0%}</div>
                </div>
                """, unsafe_allow_html=True)
                rq_photo = resolve_photo_path(rq.get("photo_filename"))
                if rq_photo:
                    st.image(str(rq_photo), use_container_width=True)
                rq_c1, rq_c2 = st.columns(2)
                with rq_c1:
                    if st.button("✅ Approve", key=f"approve_{rq_id}", use_container_width=True):
                        pipeline.approve_report(rq_id, "Approved by operator after manual photo review")
                        st.toast(f"✅ {rq_id} approved and merged into the cleanup queue.")
                        st.rerun()
                with rq_c2:
                    if st.button("🗑️ Reject", key=f"reject_{rq_id}", use_container_width=True):
                        pipeline.reject_report(rq_id, "Rejected by operator: unusable or no waste present")
                        st.toast(f"🗑️ {rq_id} rejected and kept on record for audit.")
                        st.rerun()

        if not show_review_queue:
            st.markdown("### 📋 Cleanup Queue")
            st.caption("Ranked by 0–100 Hydrological Risk Score")

            priority_filter = st.selectbox(
                "Filter Priority",
                ["All Priorities", "CRITICAL (Score ≥ 70)", "HIGH (Score ≥ 50)", "MODERATE (Score < 50)"],
                label_visibility="collapsed",
            )
        else:
            priority_filter = "All Priorities"

        filtered_hotspots = [] if show_review_queue else hotspots
        if "CRITICAL" in priority_filter:
            filtered_hotspots = [h for h in hotspots if h["priority"] == "CRITICAL"]
        elif "HIGH" in priority_filter:
            filtered_hotspots = [h for h in hotspots if h["priority"] == "HIGH"]
        elif "MODERATE" in priority_filter:
            filtered_hotspots = [h for h in hotspots if h["priority"] == "MODERATE"]

        for h in filtered_hotspots:
            hid = h["id"]
            is_active = (hid == st.session_state.selected_hotspot_id)
            score = h["total_score"]
            priority = h["priority"]
            dist_m = h["nearest_drain_distance_m"]
            drain_name = h["nearest_drain_name"]
            is_merged = h.get("is_merged", False)
            recurrence = h.get("recurrence", 1)
            is_dispatched = hid in st.session_state.dispatched_work_orders

            badge_class = "badge-critical" if priority == "CRITICAL" else ("badge-high" if priority == "HIGH" else "badge-moderate")
            active_class = "queue-card-active" if is_active else ""
            status_tag = "🚚 DISPATCHED" if is_dispatched else "PENDING"
            score_chip = render_provenance_badge("derived", f"{score:.1f}")
            move_html = ""
            if storm_on and h.get("site_key") in live_rank:
                rank_now = hotspots.index(h) + 1
                shift = live_rank[h["site_key"]] - rank_now
                delta = score - live_score.get(h["site_key"], score)
                arrow = f"▲{shift}" if shift > 0 else (f"▼{-shift}" if shift < 0 else "•")
                move_html = f"<span style='color:#fca5a5; font-size:0.72rem; margin-left:6px;'>{arrow} ({delta:+.1f})</span>"

            st.markdown(f"""
            <div class="queue-card {active_class}">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                <span style="font-weight: 800; font-size: 0.98rem; color: #f8fafc;">{hid}{move_html}</span>
                <span class="{badge_class}">{priority} {score:.1f}</span>
              </div>
              <div style="font-size: 0.8rem; color: #cbd5e1; margin-bottom: 5px;">
                🌊 <b>{dist_m:.0f}m</b> to {drain_name} {render_provenance_badge('derived', f'{dist_m:.0f}m', '0.68rem')}
              </div>
              <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.75rem;">
                <span class="{ 'badge-merged' if is_merged else 'tiny-text' }">
                  👥 {recurrence} {'reports merged' if is_merged else 'single report'} {render_provenance_badge('observed', f'{recurrence}R', '0.68rem')}
                </span>
                <span style="color: {'#34d399' if is_dispatched else '#f59e0b'}; font-weight: 600;">
                  {status_tag}
                </span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"Inspect {hid}", key=f"btn_select_{hid}", use_container_width=True):
                st.session_state.selected_hotspot_id = hid
                st.session_state.selected_site_key = h.get("site_key")
                st.rerun()

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 2: CENTER INTERACTIVE LEAFLET MAP
    # ──────────────────────────────────────────────────────────────────────────
    with col_center:
        st.markdown("### 🗺️ Urban Drainage & Hotspot Triage Map")
        st.caption("Pune Waterways • Citizen Report Pins • DBSCAN Buffer Rings • Municipal Dispatch Route")

        active_h = next((h for h in hotspots if h["id"] == st.session_state.selected_hotspot_id), hotspots[0] if hotspots else None)
        center_lat = active_h["lat"] if active_h else 18.5204
        center_lon = active_h["lon"] if active_h else 73.8568

        # Determine basemap tiles URL (All free, open CDN, ZERO key required!)
        if "CARTO Dark" in map_tile_choice:
            tile_url = "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
            tile_attr = "&copy; OpenStreetMap contributors &copy; CARTO"
        elif "CARTO Voyager" in map_tile_choice:
            tile_url = "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
            tile_attr = "&copy; OpenStreetMap contributors &copy; CARTO"
        elif "ESRI" in map_tile_choice:
            tile_url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            tile_attr = "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
        else:
            # Default: OpenStreetMap (100% Free, Public, NEVER prompts for API key)
            tile_url = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
            tile_attr = "&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors"

        fmap = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=13,
            tiles=None,
            control_scale=True,
        )

        # Add the clean open basemap TileLayer
        folium.TileLayer(
            tiles=tile_url,
            attr=tile_attr,
            name="Base Map",
            subdomains="abc" if "openstreetmap" not in tile_url else "abc",
            max_zoom=19,
        ).add_to(fmap)

        # 1. Render Urban Drains & Waterways from GeoJSON
        drain_geojson_path = ROOT / "data" / "urban_drain_network.geojson"
        if drain_geojson_path.is_file():
            try:
                drain_data = json.loads(drain_geojson_path.read_text(encoding="utf-8"))
                for feat in drain_data.get("features", []):
                    props = feat.get("properties", {})
                    geom_type = feat.get("geometry", {}).get("type")
                    coords = feat.get("geometry", {}).get("coordinates", [])

                    if geom_type == "LineString" and len(coords) >= 2:
                        latlngs = [[c[1], c[0]] for c in coords]
                        is_river = props.get("waterway") == "river"
                        line_color = "#38bdf8" if is_river else "#06b6d4"
                        line_weight = 4 if is_river else 3
                        line_dash = None if is_river else "5, 5"

                        folium.PolyLine(
                            locations=latlngs,
                            color=line_color,
                            weight=line_weight,
                            opacity=0.88,
                            dash_array=line_dash,
                            tooltip=f"🌊 {props.get('name')} ({props.get('type')})",
                        ).add_to(fmap)
            except Exception as e:
                st.error(f"Error loading drain network: {e}")

        # 2. Render Scattered Individual Citizen Report Pins (Yellow dots)
        for r in reports:
            rep_lat = r["lat"]
            rep_lon = r["lon"]
            rep_id = r["id"]
            rep_conf = r.get("plastic_confidence", 0.88)
            folium.CircleMarker(
                location=[rep_lat, rep_lon],
                radius=4,
                color="#eab308",
                fill=True,
                fill_color="#facc15",
                fill_opacity=0.9,
                tooltip=f"📱 Citizen Report {rep_id} (Conf: {rep_conf:.0%})",
            ).add_to(fmap)

        # 3. Render Consolidated DBSCAN Hotspots (Buffer Rings & Centroids)
        for h in hotspots:
            hid = h["id"]
            h_lat = h["lat"]
            h_lon = h["lon"]
            h_score = h["total_score"]
            priority = h["priority"]
            is_active = (hid == st.session_state.selected_hotspot_id)
            is_merged = h.get("is_merged", False)
            recurrence = h.get("recurrence", 1)
            dist_m = h["nearest_drain_distance_m"]
            drain_name = h["nearest_drain_name"]

            h_color = "#ef4444" if priority == "CRITICAL" else ("#f59e0b" if priority == "HIGH" else "#38bdf8")

            # 50m DBSCAN Consolidation Buffer Halo
            folium.Circle(
                location=[h_lat, h_lon],
                radius=h.get("buffer_radius_m", 50.0),
                color="#ffffff" if is_active else h_color,
                weight=2.5 if is_active else 1.2,
                fill=True,
                fill_color=h_color,
                fill_opacity=0.22 if is_active else 0.09,
                tooltip=f"{hid} • {recurrence} reports merged via DBSCAN",
            ).add_to(fmap)

            # Center Hotspot Marker
            folium.CircleMarker(
                location=[h_lat, h_lon],
                radius=13 if is_active else 8,
                color="#ffffff" if is_active else h_color,
                weight=3 if is_active else 1.5,
                fill=True,
                fill_color=h_color,
                fill_opacity=0.95,
                tooltip=f"<b>{hid}</b>: {priority} ({h_score:.1f}/100)<br>Drain: {dist_m:.0f}m to {drain_name}<br>Reports: {recurrence}",
            ).add_to(fmap)

        # 4. Render Municipal Central Sanitation Depot
        depot_lat, depot_lon = 18.5015, 73.8580
        folium.Marker(
            location=[depot_lat, depot_lon],
            icon=folium.Icon(color="green", icon="home", prefix="fa"),
            tooltip="🏢 PMC Central Sanitation Depot (Swargate)",
        ).add_to(fmap)

        # 5. Render Dispatched Municipal Truck Route (if active)
        if st.session_state.active_route_plan:
            plan = st.session_state.active_route_plan
            route_coords = plan.get("route_polyline", [])
            if len(route_coords) >= 2:
                # Emerald Green truck route line
                folium.PolyLine(
                    locations=route_coords,
                    color="#10b981",
                    weight=5,
                    opacity=0.92,
                    tooltip=f"🚚 Cleanup Work Order {plan.get('work_order_id')} • {plan.get('total_distance_km')} km",
                ).add_to(fmap)

                # Stop order markers
                for stop in plan.get("stops", []):
                    s_num = stop["stop_number"]
                    s_id = stop["hotspot_id"]
                    folium.Marker(
                        location=[stop["lat"], stop["lon"]],
                        icon=folium.DivIcon(
                            html=f"""
                            <div style="background:#10b981; color:#0f172a; font-weight:900; font-size:11px;
                                        border:2px solid white; border-radius:50%; width:22px; height:22px;
                                        display:flex; align-items:center; justify-content:center;
                                        box-shadow:0 0 10px rgba(16,185,129,0.8);">
                              {s_num}
                            </div>
                            """
                        ),
                        tooltip=f"Stop #{s_num}: {s_id} (Est. service: {stop['estimated_service_min']}m)",
                    ).add_to(fmap)

        # Map HTML Render
        st_folium(fmap, height=520, use_container_width=True, returned_objects=[])

        # Bottom Route & Map Legend
        st.markdown("""
        <div style="display:flex; flex-wrap:wrap; gap:16px; font-size:0.77rem; color:#94a3b8; margin-top:6px; padding: 4px 6px; background:#0b1c24; border-radius:8px; border:1px solid #1a3c48;">
          <span><span style="color:#ef4444;">●</span> Critical Hotspot</span>
          <span><span style="color:#f59e0b;">●</span> High Priority</span>
          <span><span style="color:#38bdf8;">●</span> Moderate Risk</span>
          <span><span style="color:#facc15;">•</span> Citizen Report Pin</span>
          <span><span style="color:#06b6d4;">―</span> Stormwater Drain</span>
          <span><span style="color:#10b981;">―</span> Dispatched Truck Route</span>
          <span style="margin-left:auto; color:#64748b;">Tiles: OpenStreetMap (No API key needed)</span>
        </div>
        """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 3: RIGHT DRAWER (Hotspot Detail, Score Breakdown & Dispatch Action)
    # ──────────────────────────────────────────────────────────────────────────
    with col_right:
        if active_h:
            hid = active_h["id"]
            score = active_h["total_score"]
            priority = active_h["priority"]
            is_merged = active_h.get("is_merged", False)
            recurrence = active_h.get("recurrence", 1)
            dist_m = active_h["nearest_drain_distance_m"]
            drain_name = active_h["nearest_drain_name"]
            is_dispatched = hid in st.session_state.dispatched_work_orders

            st.markdown(f"### 🔍 {hid} Detail")
            
            # Hotspot Header Status
            badge_class = "badge-critical" if priority == "CRITICAL" else ("badge-high" if priority == "HIGH" else "badge-moderate")
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
              <span class="{badge_class}" style="font-size:0.85rem; padding:4px 10px;">{priority} PRIORITY</span>
              <span style="font-size:1.45rem; font-weight:800; color:{active_h['priority_color']}">{score:.1f}<span style="font-size:0.8rem; color:#64748b;">/100</span></span>
            </div>
            """, unsafe_allow_html=True)

            # DBSCAN Merged Reports Callout
            if is_merged:
                st.info(f"👥 **{recurrence} Citizen Reports Merged** via DBSCAN (50m neighborhood). Duplicate pins consolidated into single municipal work target.")
            else:
                st.caption(f"📍 Single citizen report location ({active_h.get('reports', [{}])[0].get('id', 'CR-00')}).")

            # Evidence Photo Preview (real citizen upload when available, else curated sample)
            sample_photo = active_h.get("primary_photo", "sample_bottles_drain.jpg")
            sample_path = resolve_photo_path(sample_photo)
            if sample_path is not None:
                source_tag = "Citizen Upload" if sample_path.parent.name == "uploads" else "Curated Sample"
                st.image(str(sample_path), caption=f"Evidence Capture • {source_tag} • {sample_photo}", use_container_width=True)

            # Every citizen report merged into this hotspot
            merged_reports = active_h.get("reports", [])
            with st.expander(f"👥 Inspect {len(merged_reports)} merged citizen report(s)"):
                st.caption(
                    f"Centroid {active_h['lat']:.5f}, {active_h['lon']:.5f} = geographic mean of "
                    f"{len(merged_reports)} report coordinate(s) • IDs: {', '.join(active_h.get('report_ids', []))}"
                )
                st.dataframe(pd.DataFrame([
                    {
                        "ID": r.get("id"),
                        "Submitted": str(r.get("submitted_at", ""))[:16].replace("T", " "),
                        "Lat/Lon": f"{r['lat']:.5f}, {r['lon']:.5f}",
                        "Conf": f"{float(r.get('plastic_confidence', 0)):.0%}",
                        "Severity": r.get("severity"),
                        "Classes": ", ".join(r.get("detected_classes", [])),
                        "Photo": r.get("photo_filename"),
                    }
                    for r in sorted(merged_reports, key=lambda r: str(r.get("submitted_at", "")), reverse=True)
                ]), use_container_width=True, hide_index=True)
                photo_paths = []
                for r in merged_reports:
                    rp = resolve_photo_path(r.get("photo_filename"))
                    if rp and rp not in photo_paths:
                        photo_paths.append(rp)
                if photo_paths:
                    st.image([str(pp) for pp in photo_paths[:6]], width=110, caption=[pp.name for pp in photo_paths[:6]])

            # Exploded 0-100 Score Breakdown
            st.markdown(f"#### ⚡ Exploded Risk Breakdown {render_provenance_badge('derived')}", unsafe_allow_html=True)
            
            # Drain Proximity x Live Environment Component (35%)
            d_pts = active_h.get("drain_points", 0.0)
            env_mult = active_h.get("env_multiplier", 1.0)
            st.markdown(f"**Drain Proximity × Live Env** {render_provenance_badge('derived')}: {dist_m:.0f}m to *{drain_name}*", unsafe_allow_html=True)
            st.caption(
                f"Proximity {active_h.get('drain_proximity_base_score', active_h.get('drain_score', 0)):.0f} × env {env_mult:.2f} "
                f"(rain {active_h.get('rainfall_intensity_mmh', 0):.1f} mm/h, silt {active_h.get('drain_silt_ratio', 0):.0%}, "
                f"slope ×{active_h.get('surface_slope_factor', 1.0):.2f}) → {active_h.get('drain_score', 0):.0f}/100 • "
                f"**+{d_pts:.1f} pts** (35% weight)"
            )
            st.caption(f"🍂 Siltation basis: {active_h.get('drain_silt_basis') or 'n/a'}")
            st.progress(min(1.0, active_h.get("drain_score", 0) / 100.0))

            # Waste Severity Component (25%)
            s_pts = active_h.get("severity_points", 0.0)
            raw_sev = active_h.get("severity_raw", 3.5)
            st.markdown(f"**Detection Severity** {render_provenance_badge('inferred')}: {raw_sev:.1f}/5.0 (Clog Hazard)", unsafe_allow_html=True)
            st.caption(f"Score: {active_h.get('severity_score', 0):.0f}/100 • **+{s_pts:.1f} pts** (25% weight)")
            st.progress(min(1.0, active_h.get("severity_score", 0) / 100.0))

            # AI Detection Confidence Component (20%)
            c_pts = active_h.get("confidence_points", 0.0)
            raw_conf = active_h.get("confidence_raw", 0.88)
            st.markdown(f"**AI Confidence** {render_provenance_badge('inferred')}: {raw_conf:.0%} (TACO YOLOv8)", unsafe_allow_html=True)
            st.caption(f"Score: {active_h.get('confidence_score', 0):.0f}/100 • **+{c_pts:.1f} pts** (20% weight)")
            st.progress(min(1.0, active_h.get("confidence_score", 0) / 100.0))

            # 24h Rainfall Forecast Component (20%)
            r_pts = active_h.get("rainfall_points", 0.0)
            r_mm = active_h.get("rainfall_forecast_mm", 16.5)
            st.markdown(f"**Rainfall Forecast** {render_provenance_badge('simulated')}: {r_mm:.1f} mm in next 24h", unsafe_allow_html=True)
            st.caption(f"Score: {active_h.get('rainfall_score', 0):.0f}/100 • **+{r_pts:.1f} pts** (20% weight)")
            st.progress(min(1.0, active_h.get("rainfall_score", 0) / 100.0))

            # Recurrence Bonus
            if recurrence > 1:
                r_bonus = active_h.get("recurrence_bonus", 0.0)
                st.caption(f"🔁 **Recurrence Boost**: +{r_bonus:.1f} pts from {recurrence} repeated reports")

            st.write("")

            # ── Operator Decision Support & Explainability (XAI) ──────────────
            recommendation = ExplainabilityEngine.recommend_action(active_h)
            ranking_text = ExplainabilityEngine.explain_queue_position(active_h, hotspots)
            rec_color = recommendation["color"]
            escalation_html = (
                f"<div style='font-size:0.76rem; color:#fbbf24; margin-top:6px;'>⏫ {recommendation['escalation_trigger']}</div>"
                if recommendation.get("escalation_trigger") else ""
            )
            landuse = active_h.get("landuse_attribution") or {}
            landuse_html = (
                f"<div style='font-size:0.76rem; color:#94a3b8; margin-top:8px;'>🏙️ <b>Land-use driver ({landuse.get('attribution_strength')}):</b> {landuse.get('hypothesis')}</div>"
                if landuse.get("hypothesis") else ""
            )
            st.markdown(f"""
            <div class="feature-panel" style="border-left:4px solid {rec_color}; padding:14px 16px;">
              <div style="font-size:0.78rem; font-weight:800; color:#94a3b8; text-transform:uppercase; margin-bottom:8px;">
                💡 Operator Decision Support & Explainability
              </div>
              <div style="display:inline-block; background:{rec_color}22; border:1px solid {rec_color}; color:{rec_color};
                          font-weight:800; font-size:0.86rem; padding:5px 10px; border-radius:8px; margin-bottom:8px;">
                {recommendation['badge']}
              </div>
              <div style="font-size:0.82rem; color:#e2e8f0; line-height:1.45;">{recommendation['reason_summary']}</div>
              <div style="font-size:0.78rem; color:#cbd5e1; margin-top:6px;">👷 Recommended crew: <b>{recommendation['recommended_crew']}</b></div>
              {escalation_html}
              <div style="font-size:0.8rem; color:#cbd5e1; margin-top:10px; padding-top:8px; border-top:1px solid #1a3c48; line-height:1.45;">
                ⚖️ <b>Why this rank?</b> {ranking_text}
              </div>
              {landuse_html}
            </div>
            """, unsafe_allow_html=True)

            # ── Tendering summary (full board lives in the Tendering tab) ──────
            assessment = dispatch_engine.assess_hotspot(active_h)
            vendor = assessment["vendor"]
            quote = assessment["quote"] or {}
            fraud = assessment["fraud_check"]
            site_key = active_h.get("site_key", hid)
            audit_logged = site_key in st.session_state.fraud_audit_overrides
            if fraud["is_fraud_suspect"] and not audit_logged:
                shield_html = (f"<span style='color:#f87171; font-weight:700;'>🛡️ PAYMENT HOLD</span> • billed cleanup on "
                               f"{(fraud.get('previous_cleanup') or {}).get('date')} • log a field audit in 📑 Tendering to unlock")
            elif fraud["is_fraud_suspect"]:
                shield_html = "<span style='color:#fbbf24; font-weight:700;'>🛡️ AUDIT LOGGED</span> • dispatch unlocked, payment held until sign-off"
            else:
                shield_html = "<span style='color:#34d399; font-weight:700;'>🛡️ CLEAR</span> • no billed cleanup within 35 m / 90 days"
            st.markdown(f"""
            <div style="background:#0b1c24; border:1px solid #1a3c48; border-radius:10px; padding:9px 12px; font-size:0.78rem; color:#cbd5e1; margin-bottom:10px;">
              🏗️ <b>{vendor.get('name')}</b> • quote <b>₹{quote.get('total_quote_inr', 0):,.0f}</b><br>{shield_html}
            </div>
            """, unsafe_allow_html=True)

            dispatch_blocked = fraud["is_fraud_suspect"] and not audit_logged

            # ── Human-in-the-Loop Verification & Dispatch Action ───────────────
            st.markdown("#### 🚨 Municipal Action")
            
            truck_options = [
                "PMC Rapid Tipper Truck 04 (Crew B • 4 Workers)",
                "PMC Suction & Grate Clearing Truck 02",
                "PMC Emergency Drain De-silting Unit 01",
            ]
            assigned_truck = st.selectbox("Assigned Vehicle", truck_options)

            if is_dispatched:
                st.success(f"✅ Dispatched via {st.session_state.dispatched_work_orders[hid]['work_order_id']}")
                st.caption(f"Assigned: {st.session_state.dispatched_work_orders[hid]['truck_id']}")
                if st.button("Clear Dispatch Route", key=f"clear_{hid}", use_container_width=True):
                    del st.session_state.dispatched_work_orders[hid]
                    st.session_state.active_route_plan = None
                    st.rerun()
            else:
                if dispatch_blocked:
                    st.caption("🔒 Dispatch locked by the 90-day Fraud Shield. Log a field audit to unlock.")
                if st.button("🚨 Verify & Dispatch Work Order", type="primary", use_container_width=True, key=f"dispatch_{hid}", disabled=dispatch_blocked):
                    target_ids = [hid]
                    for oth in hotspots:
                        if oth["priority"] == "CRITICAL" and oth["id"] != hid:
                            target_ids.append(oth["id"])

                    plan = pipeline.dispatch_work_order(
                        target_ids,
                        rainfall_override_mm=st.session_state.rain_override,
                        **live_env_kwargs,
                    )
                    plan["vendor_id"] = vendor.get("vendor_id")
                    plan["vendor_name"] = vendor.get("name")
                    plan["quote_inr"] = quote.get("total_quote_inr")
                    record_work_order({
                        "work_order_id": plan.get("work_order_id"),
                        "channel": "Triage drawer",
                        "hotspot_ids": [hid],
                        "site_keys": [site_key],
                        "vendors": [vendor.get("name")],
                        "total_quote_inr": quote.get("total_quote_inr"),
                        "payment_holds": [hid] if fraud["is_fraud_suspect"] else [],
                        "rain_mmh": pipeline_data["environment"]["rainfall_intensity_mmh"],
                    })
                    st.session_state.dispatched_work_orders[hid] = plan
                    st.session_state.active_route_plan = plan
                    st.toast(f"✅ Work Order {plan['work_order_id']} dispatched for {hid}! Route rendered on map.")
                    st.rerun()

            if st.session_state.active_route_plan:
                plan = st.session_state.active_route_plan
                st.markdown(f"""
                <div style="background:#06281e; border:1px solid #10b981; border-radius:10px; padding:12px; margin-top:10px;">
                  <div style="color:#10b981; font-weight:700; font-size:0.82rem;">ACTIVE DISPATCH WORK ORDER</div>
                  <div style="font-size:1.05rem; font-weight:800; color:white;">{plan.get('work_order_id')}</div>
                  <div style="font-size:0.77rem; color:#cbd5e1; margin-top:3px; line-height:1.4;">
                    Distance: <b>{plan.get('total_distance_km')} km</b> • Total Est: <b>{plan.get('total_duration_minutes')} mins</b><br>
                    Stops: <b>{plan.get('total_stops')} hotspots</b> from Swargate Central Depot<br>
                    Vendor: <b>{plan.get('vendor_name') or 'PMC in-house crew'}</b>{f" • Quote ₹{plan['quote_inr']:,.0f}" if plan.get('quote_inr') else ''}
                  </div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB: TENDERING & WORK ORDERS (vendor match, quotes, 90-day fraud shield, ledger)
# ══════════════════════════════════════════════════════════════════════════════
with tab_tender:
    st.markdown("### 📑 Tendering & Work Orders")
    st.markdown(
        f"Pre-approved ward vendors • Tonnage-based quotes • 90-day duplicate-billing Fraud Shield • Work-order ledger "
        f"{render_provenance_badge('derived', '[Derived / Registry + Dossier]')}",
        unsafe_allow_html=True,
    )
    st.caption("Every hotspot is matched to the contractor registered for its ward. A site already cleaned and billed within "
               "35 m in the last 90 days is held for field audit before any new payment.")

    tender_rows = []
    for h in hotspots:
        a_ = dispatch_engine.assess_hotspot(h)
        tender_rows.append((h, a_))

    holds = [h for h, a_ in tender_rows if a_["fraud_check"]["is_fraud_suspect"] and h.get("site_key") not in st.session_state.fraud_audit_overrides]
    ledger = list_work_orders()
    t1, t2, t3, t4, t5 = st.columns(5)
    for col, label, val, color, note in [
        (t1, "Registered Vendors", len(dispatch_engine.vendors), "#38bdf8", "Ward-zoned contractors"),
        (t2, "Open Quote Value", f"₹{sum((a_['quote'] or {}).get('total_quote_inr', 0) for _, a_ in tender_rows):,.0f}", "#34d399", f"{len(tender_rows)} hotspots incl. GST"),
        (t3, "Payment Holds", len(holds), "#ef4444", "Billed ≤ 90 days, ≤ 35 m"),
        (t4, "Field Audits Logged", len(st.session_state.fraud_audit_overrides), "#fbbf24", "Holds unlocked this session"),
        (t5, "Work Orders Issued", len(ledger), "#c084fc", "Persistent ledger"),
    ]:
        with col:
            st.markdown(f"""
            <div class="stat-card">
              <div class="stat-label">{label}</div>
              <div class="stat-val" style="color:{color};">{val}</div>
              <div class="stat-note">{note}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 🧾 Hotspot Tender Board")
    st.dataframe(pd.DataFrame([
        {
            "Hotspot": h["id"],
            "Site": h["nearest_drain_name"],
            "Priority": f"{h['priority']} {h['total_score']:.1f}",
            "Vendor": a_["vendor"].get("name"),
            "Match": "ward name" if a_["vendor"].get("match_method") == "ward_name" else f"nearest zone ({a_['vendor'].get('distance_to_zone_km')} km)",
            "Est. kg": (a_["quote"] or {}).get("predicted_weight_kg"),
            "Quote ₹": round((a_["quote"] or {}).get("total_quote_inr", 0)),
            "Fraud Shield": (
                "AUDIT LOGGED" if h.get("site_key") in st.session_state.fraud_audit_overrides and a_["fraud_check"]["is_fraud_suspect"]
                else ("HOLD" if a_["fraud_check"]["is_fraud_suspect"] else "CLEAR")
            ),
            "Last billed cleanup": (a_["fraud_check"].get("previous_cleanup") or {}).get("date") or "—",
        }
        for h, a_ in tender_rows
    ]), use_container_width=True, hide_index=True)

    td_left, td_right = st.columns([1.15, 1.0], gap="large")
    with td_left:
        tender_pick = st.selectbox(
            "Select hotspot to tender",
            range(len(tender_rows)),
            format_func=lambda i: f"{tender_rows[i][0]['id']} • {tender_rows[i][0]['nearest_drain_name']}",
            key="tender_pick",
        )
        th, ta = tender_rows[tender_pick]
        t_vendor, t_quote, t_fraud = ta["vendor"], ta["quote"] or {}, ta["fraud_check"]
        t_key = th.get("site_key", th["id"])
        t_audited = t_key in st.session_state.fraud_audit_overrides
        t_match = (
            f"zone match: {t_vendor.get('matched_zone')}" if t_vendor.get("match_method") == "ward_name"
            else f"nearest zone {t_vendor.get('matched_zone')} ({t_vendor.get('distance_to_zone_km')} km)"
        )
        st.markdown(f"""
        <div class="feature-panel" style="padding:14px 16px;">
          <div style="font-size:0.78rem; font-weight:800; color:#94a3b8; text-transform:uppercase; margin-bottom:6px;">🏗️ Matched Vendor & Quote</div>
          <div style="font-weight:800; color:#f8fafc; font-size:1rem;">{t_vendor.get('name')}</div>
          <div class="tiny-text">{t_vendor.get('vendor_id')} • {t_match} • {t_vendor.get('equipment')} • crew: {t_vendor.get('crew_type', '—')}</div>
          <div style="display:flex; justify-content:space-between; font-size:0.84rem; color:#cbd5e1; margin-top:10px;">
            <span>Mobilization fee</span><span>₹{t_quote.get('base_mobilization_inr', 0):,.0f}</span>
          </div>
          <div style="display:flex; justify-content:space-between; font-size:0.84rem; color:#cbd5e1;">
            <span>{t_quote.get('predicted_weight_kg', 0):.1f} kg ({t_quote.get('predicted_tons', 0):.3f} t) × ₹{t_quote.get('rate_per_ton_inr', 0):,.0f}/t</span><span>₹{t_quote.get('tonnage_cost_inr', 0):,.0f}</span>
          </div>
          <div style="display:flex; justify-content:space-between; font-size:0.84rem; color:#cbd5e1;">
            <span>GST 18%</span><span>₹{t_quote.get('gst_inr', 0):,.0f}</span>
          </div>
          <div style="display:flex; justify-content:space-between; font-size:1rem; font-weight:800; color:#34d399; margin-top:6px; padding-top:6px; border-top:1px solid #1a3c48;">
            <span>Work Order Quote</span><span>₹{t_quote.get('total_quote_inr', 0):,.0f}</span>
          </div>
          <div class="tiny-text" style="margin-top:6px;">Weight model: severity × 6.5 kg + drain-proximity load, +15% per repeat report (same basis as the monsoon simulator).</div>
        </div>
        """, unsafe_allow_html=True)

    with td_right:
        if t_fraud["is_fraud_suspect"]:
            prev = t_fraud.get("previous_cleanup") or {}
            st.markdown(f"""
            <div style="background:rgba(239, 68, 68, 0.12); border:2px solid {'#f59e0b' if t_audited else '#ef4444'}; border-radius:12px; padding:12px 14px; margin-bottom:10px;">
              <div style="font-weight:800; color:#f87171; font-size:0.88rem;">{t_fraud['message']}</div>
              <div style="font-size:0.78rem; color:#fde68a; margin-top:6px; line-height:1.5;">
                🧾 Previous receipt <b>{prev.get('log_id')}</b> • {prev.get('date')} ({prev.get('days_ago')} days ago) • {prev.get('site_name')}<br>
                {prev.get('weight_removed_kg')} kg removed • {prev.get('clearance_effectiveness_pct')}% clearance • {prev.get('verification_status')}<br>
                {t_fraud.get('billed_cleanups_in_window')} billed cleanup(s) within {t_fraud.get('dist_threshold_m'):.0f} m / {t_fraud.get('days_threshold')} days
              </div>
            </div>
            """, unsafe_allow_html=True)
            if t_audited:
                st.warning(f"📋 {st.session_state.fraud_audit_overrides[t_key]}. Work order unlocked; payment stays on hold until audit sign-off.")
            elif st.button("📋 Log Field Audit (fresh re-accumulation verified on site)", key=f"tender_audit_{t_key}", use_container_width=True):
                st.session_state.fraud_audit_overrides[t_key] = (
                    f"Field audit logged {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}: fresh re-accumulation verified"
                )
                st.rerun()
        else:
            st.success(t_fraud["message"])

        t_blocked = t_fraud["is_fraud_suspect"] and not t_audited
        if t_blocked:
            st.caption("🔒 Work order locked by the 90-day Fraud Shield. Log a field audit to unlock.")
        if st.button("📤 Issue Work Order to Vendor", type="primary", use_container_width=True,
                     key=f"tender_issue_{t_key}", disabled=t_blocked):
            plan = pipeline.dispatch_work_order([th["id"]], rainfall_override_mm=st.session_state.rain_override, **live_env_kwargs)
            plan["vendor_id"], plan["vendor_name"] = t_vendor.get("vendor_id"), t_vendor.get("name")
            plan["quote_inr"] = t_quote.get("total_quote_inr")
            st.session_state.dispatched_work_orders[th["id"]] = plan
            st.session_state.active_route_plan = plan
            record_work_order({
                "work_order_id": plan.get("work_order_id"),
                "channel": "Tendering tab",
                "hotspot_ids": [th["id"]],
                "site_keys": [t_key],
                "vendors": [t_vendor.get("name")],
                "total_quote_inr": t_quote.get("total_quote_inr"),
                "payment_holds": [th["id"]] if t_fraud["is_fraud_suspect"] else [],
                "rain_mmh": pipeline_data["environment"]["rainfall_intensity_mmh"],
            })
            st.toast(f"📤 {plan['work_order_id']} issued to {t_vendor.get('name')}")
            st.rerun()

    st.markdown("#### 🏢 Vendor Registry")
    st.dataframe(pd.DataFrame([
        {
            "Vendor ID": v["vendor_id"],
            "Name": v["name"],
            "Zones": ", ".join(v.get("assigned_zones", [])),
            "Equipment": v.get("equipment"),
            "₹/ton": v.get("rate_per_ton_inr"),
            "Mobilization ₹": v.get("base_mobilization_inr"),
        }
        for v in dispatch_engine.vendors
    ]), use_container_width=True, hide_index=True)

    st.markdown("#### 📒 Work-Order Ledger")
    if ledger:
        st.dataframe(pd.DataFrame([
            {
                "Issued (UTC)": str(o.get("issued_at", ""))[:16].replace("T", " "),
                "Work Order": o.get("work_order_id"),
                "Channel": o.get("channel"),
                "Hotspots": ", ".join(o.get("hotspot_ids", [])),
                "Vendor": ", ".join(o.get("vendors", []) or []),
                "Quote ₹": round(o.get("total_quote_inr") or 0),
                "Payment Hold": ", ".join(o.get("payment_holds", [])) or "—",
            }
            for o in reversed(ledger)
        ]), use_container_width=True, hide_index=True)
    else:
        st.info("No work orders issued yet. Issue one above or from the Triage drawer.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: PREDICTIVE HOTSPOT FORECASTING (24–72H)
# ══════════════════════════════════════════════════════════════════════════════
with tab_forecast:
    st.markdown("### 🔮 Predictive Hotspot Forecasting (24–72h)")
    st.markdown(
        f"Pune Stormwater Runoff Escalation Model • Hydraulic Wash-Off Dynamics • "
        f"{render_provenance_badge('simulated', '[Predicted / Modelled]')}",
        unsafe_allow_html=True,
    )
    st.caption("Forecasts multi-day plastic accumulation and drain choking risks based on Open-Meteo precipitation, historical return rates, and drain proximity.")

    fc_col1, fc_col2 = st.columns([1.6, 1.0])
    with fc_col1:
        timeline_choice = st.select_slider(
            "Forecast Timeline Horizon",
            options=[24, 48, 72],
            value=48,
            format_func=lambda x: f"T+{x}h Horizon ({'Immediate Inflow Surge' if x==24 else ('Mid-Storm Cumulative' if x==48 else 'Multi-Day Severe Inundation')})",
            help="Select prediction window (24h, 48h, or 72h) to evaluate accumulation growth and choke risks.",
        )
    with fc_col2:
        rain_slider_override = st.checkbox("Manual Rainfall Scenario Override", value=False)
        if rain_slider_override:
            rain_val = st.slider("Scenario Cumulative Rain (mm)", 0.0, 100.0, 35.0, 5.0)
        else:
            rain_val = None

    forecast_data = forecasting_engine.generate_hotspot_forecast(
        hotspots=hotspots,
        horizon_hours=timeline_choice,
        rainfall_scenario_mm=rain_val,
    )

    # 5 Spacious KPI Cards
    f1, f2, f3, f4, f5 = st.columns(5)
    with f1:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-label">Timeline Horizon</div>
          <div class="stat-val" style="color:#38bdf8;">T+{forecast_data['forecast_horizon_hours']}h</div>
          <div class="stat-note">Multi-day projection</div>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-label">Total Plastic at Risk</div>
          <div class="stat-val" style="color:#c084fc;">{forecast_data['total_predicted_waste_kg']:.1f} kg</div>
          <div class="stat-note">{render_provenance_badge('simulated', '[Predicted / Modelled]')}</div>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-label">Critical Escalations</div>
          <div class="stat-val" style="color:#ef4444;">{forecast_data['critical_escalation_count']}</div>
          <div class="stat-note">Choke probability &ge; 70%</div>
        </div>
        """, unsafe_allow_html=True)
    with f4:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-label">High Risk Sites</div>
          <div class="stat-val" style="color:#f59e0b;">{forecast_data['high_escalation_count']}</div>
          <div class="stat-note">Score &ge; 50 or &le; 45m drain</div>
        </div>
        """, unsafe_allow_html=True)
    with f5:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-label">Rainfall Forecast</div>
          <div class="stat-val" style="color:#34d399;">{forecast_data['rainfall_forecast_mm']:.1f} mm</div>
          <div class="stat-note">Open-Meteo live feed</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Grid of Hotspot Projections
    st.markdown(f"#### 📍 Hotspot Projections ({len(forecast_data['hotspot_forecasts'])} Sites Monitored)")
    
    grid_cols = st.columns(2)
    for idx, hf in enumerate(forecast_data["hotspot_forecasts"]):
        with grid_cols[idx % 2]:
            risk = hf["escalation_risk"]
            r_badge_class = "badge-critical" if risk == "CRITICAL" else ("badge-high" if risk == "HIGH" else "badge-moderate")
            tl = hf["timeline"]
            
            st.markdown(f"""
            <div class="feature-panel" style="border-left: 4px solid {hf['risk_color']};">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-weight:800; font-size:1.05rem; color:#f8fafc;">{hf['hotspot_id']} • {hf['nearest_drain_name']}</span>
                <span class="{r_badge_class}">{risk} ESCALATION</span>
              </div>
              <div style="font-size:0.82rem; color:#cbd5e1; margin-bottom:8px;">
                🌊 <b>{hf['drain_distance_m']:.0f}m</b> to drain channel • Historical Return: <b>{hf['historical_return_interval_days']} days</b>
              </div>
              <div style="display:flex; gap:16px; margin-bottom:12px; background:#081921; padding:8px 12px; border-radius:8px;">
                <div>
                  <div class="tiny-text">Predicted Mass</div>
                  <div style="font-weight:800; color:#38bdf8; font-size:1.1rem;">{hf['predicted_accumulation_kg']} kg</div>
                </div>
                <div>
                  <div class="tiny-text">Est. Volume</div>
                  <div style="font-weight:700; color:#94a3b8; font-size:1.05rem;">{hf['predicted_volume_liters']:.0f} L</div>
                </div>
                <div style="margin-left:auto; text-align:right;">
                  <div class="tiny-text">Critical Window</div>
                  <div style="font-weight:800; color:#f59e0b; font-size:0.95rem;">⏱️ {hf['critical_time_window']}</div>
                </div>
              </div>
              <div style="font-size:0.82rem; font-weight:700; color:#94a3b8; margin-bottom:6px;">
                🛠️ Specific Preventive Actions Recommended:
              </div>
            """, unsafe_allow_html=True)

            for act in hf["preventive_actions"]:
                st.markdown(f"<div style='font-size:0.8rem; color:#e2e8f0; margin-bottom:3px;'>▫️ {act}</div>", unsafe_allow_html=True)

            st.markdown(f"""
              <div style="margin-top:10px; padding-top:8px; border-top:1px solid #1a3c48; display:flex; justify-content:space-between; font-size:0.75rem; color:#94a3b8;">
                <span>T+24h: <b>{tl['24h']['accum_kg']} kg</b></span>
                <span>T+48h: <b>{tl['48h']['accum_kg']} kg</b></span>
                <span>T+72h: <b>{tl['72h']['accum_kg']} kg</b></span>
              </div>
              <div style="margin-top:8px; text-align:right;">
                {render_provenance_badge('simulated', '[Predicted / Modelled]')}
              </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: AI BEFORE/AFTER CLEANUP VERIFICATION (one work order per hotspot)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _verify_case_cached(before_path: str, after_path: str, fingerprint: str, case_id: str, site: str, contractor: str, lat: float, lon: float):
    return CleanupVerificationEngine().verify_cleanup(
        before_image=Path(before_path), after_image=Path(after_path),
        hotspot_id=case_id, site_name=site, contractor_id=contractor, site_lat=lat, site_lon=lon,
    )


@st.cache_data(show_spinner=False)
def _photo_quality_cached(path: str, mtime: float):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return evaluate_image_quality(im, strict=True, original_size=im.size)


def _case_result(case):
    if case["status"] != "READY":
        return None
    return _verify_case_cached(case["before_path"], case["after_path"], case["evidence_fingerprint"], case["case_id"],
                               case["site_name"], case["contractor"], case["lat"], case["lon"])


with tab_verify:
    st.markdown("### 📸 AI Before/After Cleanup Verification")
    st.markdown(
        f"Evidence-grade photo gate • YOLOv8 + plastic material check • Same-scene, EXIF & GPS checks • Human override with audit trail • "
        f"{render_provenance_badge('inferred', '[AI Inference / Modelled Estimate]')}",
        unsafe_allow_html=True,
    )
    st.caption("Invoices are only eligible on PASS. Photos that are blurry, dark, over-exposed, hazy, low-resolution or featureless "
               "are rejected as INCONCLUSIVE. If the AI is too strict, an operator can override it with a recorded reason.")

    cases = list_verification_cases()
    case_results = {c["case_id"]: _case_result(c) for c in cases}
    case_final = {
        c["case_id"]: effective_decision(case_results[c["case_id"]], get_active_override(c["case_id"], c["evidence_fingerprint"]))
        for c in cases if case_results[c["case_id"]]
    }
    status_label = {"READY": "✅ ready", "AWAITING_AFTER_PHOTO": "⏳ need AFTER", "AWAITING_BEFORE_PHOTO": "⏳ need BEFORE", "AWAITING_PHOTOS": "⏳ need both"}
    st.dataframe(pd.DataFrame([
        {
            "Case": c["case_id"],
            "Site": c["site_name"],
            "Contractor": c["contractor"],
            "Photos": status_label.get(c["status"], c["status"]),
            "AI verdict": case_results[c["case_id"]]["decision_status"] if case_results[c["case_id"]] else "—",
            "Final": (f"{case_final[c['case_id']]['decision']}" + (" (override)" if case_final[c["case_id"]]["source"] != "AI" else ""))
                     if c["case_id"] in case_final else "—",
            "Effectiveness": f"{case_results[c['case_id']]['effectiveness_score']:.1f}%" if case_results[c["case_id"]] else "—",
        }
        for c in cases
    ]), use_container_width=True, hide_index=True)

    pick = st.selectbox("Select verification work order", range(len(cases)),
                        format_func=lambda i: f"{cases[i]['case_id']} • {cases[i]['site_name']} ({cases[i]['ward']})", key="verify_case_pick")
    case = cases[pick]
    cid = case["case_id"]
    v_result = case_results[cid]

    def _photo_panel(kind: str, title: str):
        path = case[f"{kind}_path"]
        st.markdown(f"#### 📷 {title}")
        up = st.file_uploader(
            f"{'Replace' if path else 'Upload'} {kind.upper()} photo",
            type=["jpg", "jpeg", "png", "webp"],
            key=f"up_{cid}_{kind}_{st.session_state.get(f'upgen_{cid}_{kind}', 0)}",
        )
        if up is not None:
            try:
                save_case_photo(cid, kind, up.getvalue(), up.name)
                st.session_state[f"upgen_{cid}_{kind}"] = st.session_state.get(f"upgen_{cid}_{kind}", 0) + 1
                st.toast(f"{kind.title()} photo saved for {cid}")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not save photo: {exc}")
        if not path:
            st.info(f"No {kind} photo yet. Upload one of the same spot.")
            return
        q = _photo_quality_cached(path, Path(path).stat().st_mtime)
        annotated = v_result["visuals"][f"{kind}_annotated_base64"] if v_result else ""
        st.image(base64.b64decode(annotated) if annotated else path, use_container_width=True,
                 caption=f"{Path(path).name} • {q['resolution'][0]}×{q['resolution'][1]} px")
        if q["needs_review"]:
            st.error("❌ Not evidence-grade: " + "; ".join(q["reasons"]))
        else:
            st.success(f"✅ Evidence-grade • sharpness {q['sharpness_ratio']:.2f} • luminance {q['mean_luminance']:.0f} • contrast {q['contrast']:.0f}")
        if st.button(f"🗑️ Remove {kind} photo", key=f"rm_{cid}_{kind}", use_container_width=True):
            remove_case_photo(cid, kind)
            st.toast(f"Removed {kind} photo for {cid}")
            st.rerun()

    v_left, v_right = st.columns(2)
    with v_left:
        _photo_panel("before", "Before Cleanup")
    with v_right:
        _photo_panel("after", "After Cleanup (contractor)")

    if v_result:
        m = v_result["metrics"]
        eff = v_result["effectiveness_score"]
        final = case_final[cid]
        m1, m2, m3, m4 = st.columns(4)
        for col, label, val, color, note in [
            (m1, "Plastic Items Removed", f"-{m['count_reduction']}", "#38bdf8", f"<b>{m['count_reduction_pct']:.0f}%</b> of plastic objects"),
            (m2, "Clog-Weighted Load", f"-{m['clog_weighted_load_reduction_pct']:.0f}%", "#34d399", f"Severity Δ {m['delta_severity']:.1f}"),
            (m3, "Effectiveness Score", f"{eff:.1f}%", "#10b981" if eff >= 85 else "#f59e0b", "Standard target &ge; 85%"),
            (m4, "Diverted Plastic Mass", f"{m['diverted_plastic_weight_kg']} kg", "#c084fc", f"~{m['diverted_volume_liters']:.0f} L volume"),
        ]:
            with col:
                st.markdown(f"""
                <div class="stat-card">
                  <div class="stat-label">{label}</div>
                  <div class="stat-val" style="color:{color};">{val}</div>
                  <div class="stat-note">{note}</div>
                </div>
                """, unsafe_allow_html=True)

        dec_colors = {"PASS": "#10b981", "FLAGGED FOR REVIEW": "#f59e0b", "INCONCLUSIVE": "#c084fc"}
        ai_dec = v_result["decision_status"]
        fin_dec = final["decision"]
        fc = dec_colors.get(fin_dec, "#94a3b8")
        override_html = ""
        if final["override"]:
            o = final["override"]
            override_html = (
                f"<div style='font-size:0.82rem; color:#fde68a; margin-top:8px; padding-top:8px; border-top:1px solid {fc}55;'>"
                f"🧑‍⚖️ <b>Operator override</b> by {o['operator']} ({o['recorded_at'][:16].replace('T', ' ')} UTC): "
                f"AI said <b>{o['ai_decision']}</b> → <b>{o['override_decision']}</b> • {o['reason_label']}<br>“{o['justification']}”</div>"
            )
        st.markdown(f"""
        <div style="background:{fc}26; border:2px solid {fc}; border-radius:12px; padding:16px; margin: 14px 0 10px 0;">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="font-size:1.15rem; font-weight:800; color:{fc};">FINAL DECISION: {fin_dec}{' (operator override)' if final['override'] else ''}</div>
            {render_provenance_badge('inferred', '[AI Inference / Modelled Estimate]')}
          </div>
          <div style="font-size:0.86rem; color:#e2e8f0; margin-top:6px;">AI verdict: <b>{ai_dec}</b> ({eff:.1f}% clearance). {v_result['status_message']}</div>
          {override_html}
        </div>
        """, unsafe_allow_html=True)

        ck_col, cls_col = st.columns([1.35, 1.0], gap="large")
        with ck_col:
            st.markdown("#### 🧪 Evidence Checklist")
            icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "N/A": "➖"}
            st.dataframe(pd.DataFrame([
                {"": icon.get(c["status"], "•"), "Check": c["check"], "Result": c["status"], "Detail": c["detail"]}
                for c in v_result["checks"]
            ]), use_container_width=True, hide_index=True)
        with cls_col:
            st.markdown("#### 🧩 Per-Class Plastic Removal")
            if m["class_breakdown"]:
                st.dataframe(pd.DataFrame([
                    {"Class": r["class"].replace("_", " "), "Before": r["before"], "After": r["after"], "Removed": r["removed"]}
                    for r in m["class_breakdown"]
                ]), use_container_width=True, hide_index=True)
            else:
                st.caption("No plastic objects detected in either photo.")
            sc = v_result["scene_consistency"]
            st.caption(f"Scene matcher: {sc['status']} • {sc.get('inliers', 0)} RANSAC inliers • keypoints {sc.get('keypoints', '–')}")

        al = v_result.get("alignment", {})
        if al.get("aligned"):
            map_col, item_col = st.columns([1.0, 1.2], gap="large")
            with map_col:
                st.markdown("#### 🗺️ Item-Level Clearance Map")
                if v_result["visuals"].get("clearance_map_base64"):
                    st.image(base64.b64decode(v_result["visuals"]["clearance_map_base64"]), use_container_width=True,
                             caption="Before-photo • 🟩 cleared • 🟥 still there • 🟧 out of after-frame • 🟪 new litter in after-photo")
            with item_col:
                st.markdown(f"#### 📍 {al['items_cleared']}/{al['items_total']} items cleared at their original spots")
                st.dataframe(pd.DataFrame([
                    {"#": i + 1, "Class": r["class"].replace("_", " "), "Found by": r["source"], "Status": r["status"],
                     "Plastic px before → after": f"{r['before_signature_px']} → {r['after_signature_px']}", "Visible": f"{r['visible_share']:.0%}"}
                    for i, r in enumerate(al["items"])
                ] + [
                    {"#": "new", "Class": n["class"].replace("_", " "), "Found by": n["source"], "Status": "new litter",
                     "Plastic px before → after": "—", "Visible": "—"}
                    for n in al["new_items"]
                ]), use_container_width=True, hide_index=True)
                st.caption(f"The after-photo was warped onto the before-photo using {al['inliers']} matched features; "
                           f"an item counts as cleared when ≤ 35% of its plastic signature remains and no detector box overlaps it.")
        elif al:
            st.caption(f"Item-level alignment unavailable ({al.get('reason')}); verdict uses detection counts.")

        with st.expander("🧠 How the AI decided (every detection, accepted or rejected)"):
            for kind in ("before", "after"):
                rows = v_result["detection_trace"][kind]
                st.markdown(f"**{kind.title()} photo** — {len(rows)} detector outputs")
                if rows:
                    st.dataframe(pd.DataFrame([
                        {
                            "YOLO class": t.get("coco_class"),
                            "Det. conf": t.get("detector_confidence", "—"),
                            "Role": t.get("role"),
                            "Plastic prob.": t.get("plastic_probability", "—"),
                            "Sheen / vivid / veg": (
                                f"{t['material']['sheen']:.2f} / {t['material']['vivid']:.2f} / {t['material']['vegetation']:.2f}"
                                if isinstance(t.get("material"), dict) and "sheen" in t["material"] else "—"
                            ),
                            "Decision": t.get("decision"),
                            "Why": t.get("reason"),
                        }
                        for t in rows
                    ]).astype(str), use_container_width=True, hide_index=True)

        # ── Human-in-the-loop override ─────────────────────────────────────
        st.markdown("#### 🧑‍⚖️ Operator Override (human in the loop)")
        st.caption(
            "Use when the AI is too strict (e.g. a leaf or shadow counted as residue) or too lenient. "
            "INCONCLUSIVE photos cannot be passed (re-shoot instead); a failed same-scene check can only be passed "
            "after on-site confirmation. The override is tied to these exact photos and is voided if either is replaced."
        )
        allowed = [d for d in ("PASS", "FLAGGED FOR REVIEW") if d != ai_dec and not (ai_dec == "INCONCLUSIVE" and d == "PASS")]
        if not allowed:
            st.info("No override available for this verdict.")
        else:
            with st.form(key=f"override_form_{cid}", clear_on_submit=True):
                oc1, oc2 = st.columns([1.0, 1.6])
                with oc1:
                    o_dec = st.selectbox("Override decision to", allowed)
                    o_by = st.text_input("Operator name / ID", placeholder="e.g. JE Kasba Ward – A. Patil")
                with oc2:
                    o_reason = st.selectbox("Reason", list(OVERRIDE_REASONS.keys()), format_func=lambda k: OVERRIDE_REASONS[k])
                    o_text = st.text_area("Justification (min 20 characters, stored in the audit log)", height=80)
                if st.form_submit_button("Record override", type="primary"):
                    try:
                        record_override(cid, v_result, case["evidence_fingerprint"], o_dec, o_reason, o_text, o_by)
                        st.toast(f"Override recorded for {cid}")
                        st.rerun()
                    except ValueError as exc:
                        st.error(f"Override rejected: {exc}")

        history = list_overrides(cid)
        if history:
            with st.expander(f"📜 Override audit log for {cid} ({len(history)})"):
                st.dataframe(pd.DataFrame([
                    {
                        "When (UTC)": o["recorded_at"][:16].replace("T", " "),
                        "Operator": o["operator"],
                        "AI": o["ai_decision"],
                        "Override": o["override_decision"],
                        "Reason": o["reason_label"],
                        "Justification": o["justification"],
                        "Applies to current photos": "yes" if o["evidence_fingerprint"] == case["evidence_fingerprint"] else "no (photos changed)",
                    }
                    for o in reversed(history)
                ]), use_container_width=True, hide_index=True)

        st.markdown(f"""
        <div class="disclaimer-box">
          <b>⚠️ Disclaimer:</b> {v_result['disclaimer']}
        </div>
        """, unsafe_allow_html=True)

        invoice_key = f"{cid}_{case['evidence_fingerprint'][:12]}_{fin_dec}"
        is_signed = invoice_key in st.session_state.signed_off_invoices
        s_col1, s_col2 = st.columns([1.5, 1.0])
        with s_col1:
            st.text_input("Operator Field Verification Remarks", value="Visual drain clearance verified. Intake grate unobstructed.", key=f"notes_{invoice_key}")
        with s_col2:
            st.write("")
            st.write("")
            if is_signed:
                st.success("✅ Signed Off & Cleared for Contractor Payment")
            elif final["invoice_eligible"]:
                if st.button("✍️ Sign-Off & Approve Invoice", type="primary", use_container_width=True, key=f"sign_{invoice_key}"):
                    st.session_state.signed_off_invoices.add(invoice_key)
                    st.toast(f"✅ Contractor invoice signed off for {case['site_name']}!")
                    st.rerun()
            else:
                st.button("🔒 Invoice locked: " + ("re-shoot evidence" if fin_dec == "INCONCLUSIVE" else "field re-inspection"),
                          disabled=True, use_container_width=True, key=f"locked_{invoice_key}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: RECURRENCE & ROOT-CAUSE INTELLIGENCE (3-month monsoon window)
# ══════════════════════════════════════════════════════════════════════════════
with tab_recurrence:
    rec_analysis = recurrence_engine.analyze_recurrence()
    summary = rec_analysis["summary_metrics"]
    window_days = rec_analysis["monitoring_window_days"]

    st.markdown("### 🔁 Recurrence & Root-Cause Intelligence")
    st.markdown(
        f"{window_days}-day monsoon dossier ({str(rec_analysis.get('window_start', ''))[:10]} → {str(rec_analysis.get('window_end', ''))[:10]}) • "
        f"Real Open-Meteo rainfall • Multi-evidence attribution • "
        f"{render_provenance_badge('inferred', '[Inferred / Prototype Historical Dossier]')}",
        unsafe_allow_html=True,
    )
    st.caption(
        "Root cause is not a fixed label: each candidate driver is scored on four independent evidence streams "
        "(where the site is • what waste it collects • when litter arrives • how much rain drives it) and ranked. "
        + (rec_analysis.get("data_notes") or "")
    )

    r1, r2, r3, r4, r5 = st.columns(5)
    for col, label, val, color, note in [
        (r1, "Monitored Locations", summary["total_monitored_sites"], "#38bdf8", "Pune Municipal Pilot"),
        (r2, f"{window_days}-Day Cleanups", summary["total_cleanups_completed"], "#c084fc", "Completed work orders"),
        (r3, "Total Diverted Plastic", f"{summary['total_diverted_plastic_kg']:,.0f} kg", "#34d399", "Removed from drainage"),
        (r4, "Avg Return Interval", f"{summary['average_return_interval_days']:.1f} d", "#f59e0b", "Days until site reclogs"),
        (r5, "Chronic Choke Rate", f"{summary['chronic_site_percentage']:.0f}%", "#ef4444", "Persistence index &ge; 0.75"),
    ]:
        with col:
            st.markdown(f"""
            <div class="stat-card">
              <div class="stat-label">{label}</div>
              <div class="stat-val" style="color:{color};">{val}</div>
              <div class="stat-note">{note}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    with st.expander("📂 Where does this data come from?", expanded=False):
        st.dataframe(pd.DataFrame([
            {"File": d["source"], "What it is": d["kind"], "Records": d["records"], "Covers": d["covers"], "Used for": d["used_for"]}
            for d in rec_analysis["data_sources"]
        ]), use_container_width=True, hide_index=True)
        st.caption(
            "Flow: RecurrenceEngine reads the dossier → per site it computes return intervals, monthly totals and trend → "
            "the root-cause engine scores each candidate driver on proximity (land-use file), waste mix and timing (dossier logs) "
            "and rain coupling (real rainfall file). Only the rainfall series is external, measured data."
        )

    loc_names = [f"{loc['site_id']} - {loc['name']}" for loc in rec_analysis["locations"]]
    selected_loc_idx = st.selectbox("Select Historical Dossier Location", range(len(loc_names)), format_func=lambda i: loc_names[i])
    selected_site = rec_analysis["locations"][selected_loc_idx]
    rca = selected_site["root_cause_analysis"]
    ev = rca["site_evidence"]

    s_col_left, s_col_right = st.columns([1.25, 1.0], gap="large")
    with s_col_left:
        st.markdown(f"#### 📜 Location Dossier: {selected_site['name']}")
        trend_color = {"WORSENING": "#ef4444", "IMPROVING": "#34d399"}.get(selected_site["mass_trend"], "#94a3b8")
        st.markdown(f"""
        <div class="feature-panel">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <span style="font-size:1.1rem; font-weight:800; color:#f8fafc;">{selected_site['ward']}</span>
            <span class="badge-critical" style="background:rgba(239, 68, 68, 0.2);">{selected_site['persistence_level']} PERSISTENCE ({selected_site['recurrence_persistence_index']})</span>
          </div>
          <div style="font-size:0.86rem; color:#cbd5e1; margin-bottom:12px;">
            🌊 Connected Waterway: <b>{selected_site['drain_name']}</b> ({selected_site['drain_distance_m']}m)
          </div>
          <div style="display:flex; gap:16px; margin-bottom:14px; background:#081b24; padding:10px 14px; border-radius:10px; flex-wrap:wrap;">
            <div><div class="tiny-text">Cleanups ({window_days} d)</div><div style="font-weight:800; color:#38bdf8; font-size:1.15rem;">{selected_site['historical_cleanup_events']} ({selected_site['cleanups_per_30d']}/30 d)</div></div>
            <div><div class="tiny-text">Avg Return Interval</div><div style="font-weight:800; color:#f59e0b; font-size:1.15rem;">{selected_site['average_return_interval_days']} days</div></div>
            <div><div class="tiny-text">Diverted Plastic</div><div style="font-weight:800; color:#34d399; font-size:1.15rem;">{selected_site['total_diverted_weight_kg']} kg</div></div>
            <div><div class="tiny-text">Load Trend</div><div style="font-weight:800; color:{trend_color}; font-size:1.15rem;">{selected_site['mass_trend']} ({selected_site['mass_trend_kg_per_30d']:+.1f} kg/30d)</div></div>
          </div>
          <div style="background:rgba(245, 158, 11, 0.08); border-left:4px solid #f59e0b; padding:12px 14px; border-radius:6px; margin-bottom:12px;">
            <div style="font-size:0.82rem; font-weight:800; color:#fbbf24; text-transform:uppercase; margin-bottom:4px;">
              💡 Root Cause: {rca['primary_driver']} • {rca['primary_confidence_pct']:.0f}% of evidence • certainty {rca['certainty']}
            </div>
            <div style="font-size:0.86rem; color:#e2e8f0; line-height:1.5;">{selected_site['root_cause_hypothesis']}</div>
            <div class="tiny-text" style="margin-top:6px;">📝 Field crew note: {selected_site.get('field_observation_note') or '—'}</div>
          </div>
          <div style="font-size:0.82rem; font-weight:800; color:#38bdf8; text-transform:uppercase; margin-bottom:6px;">🏛️ Targeted Systemic Interventions</div>
        """, unsafe_allow_html=True)
        for inv in selected_site["systemic_interventions"]:
            st.markdown(f"<div style='font-size:0.84rem; color:#cbd5e1; margin-bottom:4px;'>📌 <b>{inv}</b></div>", unsafe_allow_html=True)
        st.markdown(f"""
          <div style="margin-top:12px; text-align:right;">
            {render_provenance_badge('inferred', '[Inferred / Prototype Historical Dossier]')}
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### ⚖️ How the cause was determined")
        cand_df = pd.DataFrame([
            {"Driver": c["driver"], "Evidence": k.replace("_", " "),
             "Points": round(v * rca["evidence_weights"][k] * 100, 1)}
            for c in rca["candidates"] for k, v in c["evidence_scores"].items()
        ])
        cand_chart = alt.Chart(cand_df).mark_bar().encode(
            y=alt.Y("Driver:N", sort=[c["driver"] for c in rca["candidates"]], title=None),
            x=alt.X("sum(Points):Q", title="Evidence score (weighted)"),
            color=alt.Color("Evidence:N", scale=alt.Scale(
                domain=["proximity", "composition", "temporal", "rain coupling"],
                range=["#38bdf8", "#c084fc", "#fbbf24", "#34d399"])),
            tooltip=["Driver", "Evidence", "Points"],
        ).properties(height=210)
        st.altair_chart(cand_chart, use_container_width=True)
        st.caption(
            "Weights: proximity 30% (distance to mapped land-use landmark) • waste fingerprint 35% (mass-weighted mix vs driver profile) • "
            "timing 20% (hour band + weekday/weekend pattern of first reports) • rain coupling 15% (share of accumulation rate explained by real rainfall)."
        )

    with s_col_right:
        st.markdown("#### 🧬 Evidence Fingerprint")
        e1, e2 = st.columns(2)
        with e1:
            comp_df = pd.DataFrame([{"Class": k.replace("_", " "), "Share %": v} for k, v in ev["composition"].items()])
            st.altair_chart(alt.Chart(comp_df).mark_bar(color="#c084fc").encode(
                x=alt.X("Share %:Q"), y=alt.Y("Class:N", sort="-x", title=None), tooltip=["Class", "Share %"],
            ).properties(height=150, title="Waste mix (mass-weighted)"), use_container_width=True)
        with e2:
            hour_df = pd.DataFrame([{"Band": k, "Share %": v} for k, v in ev["hour_share"].items()])
            st.altair_chart(alt.Chart(hour_df).mark_bar(color="#fbbf24").encode(
                x=alt.X("Band:N", sort=list(ev["hour_share"].keys()), title=None), y=alt.Y("Share %:Q"), tooltip=["Band", "Share %"],
            ).properties(height=150, title="When litter is first reported"), use_container_width=True)
        st.caption(
            f"Weekend/weekday report ratio **{ev['weekend_ratio']}** • rain explains **{ev['rain_share_of_load']:.0%}** of the "
            f"accumulation rate (r = {ev['rain_load_correlation']:+.2f}, {ev['rain_intervals_analyzed']} intervals)"
        )

        st.markdown("#### 📈 3-Month Accumulation vs Rainfall")
        spark_df = pd.DataFrame([
            {"date": c["timestamp"][:10], "weight_kg": c["weight_removed_kg"], "rain_48h_mm": c.get("rainfall_prev_48h_mm", 0.0),
             "source": c.get("record_source", "pilot_log"), "status": c.get("verification_status", "PASS")}
            for c in selected_site["cleanups"]
        ])
        if not spark_df.empty:
            bars = alt.Chart(spark_df).mark_bar(color="#38bdf8", cornerRadiusTopLeft=3, cornerRadiusTopRight=3, size=7).encode(
                x=alt.X("date:T", title=None), y=alt.Y("weight_kg:Q", title="Plastic removed (kg)"),
                tooltip=["date:T", "weight_kg:Q", "rain_48h_mm:Q", "source:N", "status:N"],
            )
            rain_line = alt.Chart(spark_df).mark_line(color="#34d399", point=True, strokeDash=[4, 3]).encode(
                x="date:T", y=alt.Y("rain_48h_mm:Q", title="Rain prior 48 h (mm)"),
            )
            st.altair_chart(alt.layer(bars, rain_line).resolve_scale(y="independent").properties(height=220), use_container_width=True)

        st.markdown("#### 🗓️ Month-by-Month")
        st.dataframe(pd.DataFrame([
            {"Month": mb["month"], "Cleanups": mb["cleanups"], "Removed (kg)": mb["weight_kg"], "kg / cleanup": mb["mean_kg_per_cleanup"]}
            for mb in selected_site["monthly_breakdown"]
        ]), use_container_width=True, hide_index=True)

        st.markdown(f"#### 📋 {window_days}-Day Cleanup Logs")
        st.dataframe(pd.DataFrame([
            {
                "Log ID": c["log_id"],
                "Date": c["timestamp"].split("T")[0],
                "Weight (kg)": f"{c['weight_removed_kg']:.1f}",
                "Rain 48h": f"{c.get('rainfall_prev_48h_mm', 0):.0f} mm",
                "Effectiveness": f"{c['clearance_effectiveness_pct']:.0f}%",
                "Status": c["verification_status"],
                "Crew": c["crew_id"],
                "Record": "original demo" if c.get("record_source") == "pilot_log" else "generated",
            }
            for c in reversed(selected_site["cleanups"])
        ]), use_container_width=True, hide_index=True, height=260)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: MONSOON WHAT-IF DECISION SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_simulator:
    st.markdown("### ☔ Monsoon What-If Decision Simulator")
    st.markdown(
        f"Dynamic Municipal Stress-Testing • Hydrologic Runoff Surge • Delayed Inaction Costs • "
        f"{render_provenance_badge('simulated', '[Simulated / Modelled Estimate]')}",
        unsafe_allow_html=True,
    )
    st.caption("Per inlet: runoff Q = C·i·A meets a capacity reduced by plastic blockage and siltation; trucks race the storm peak (T+4 h) in priority order. Every constant is listed under 'Per-site hydraulics & model assumptions'.")

    # 5 Dynamic Sliders
    sc_col1, sc_col2, sc_col3, sc_col4, sc_col5 = st.columns(5)
    with sc_col1:
        sim_rain = st.slider(
            "🌧️ 1. Rainfall Intensity (mm/hr)",
            min_value=0.0,
            max_value=75.0,
            value=25.0,
            step=2.5,
            help="<2.5 light • 2.5–7.5 moderate • 7.5–35 heavy • 35–60 very heavy burst • 60+ extreme. "
                 "Pune's wettest hour in the Jul–Sep 2026 Open-Meteo reanalysis was 12.8 mm/h; point gauges catch short bursts several times higher.",
        )
    with sc_col2:
        sim_fleet = st.slider(
            "🚚 2. Available Fleet (Trucks)",
            min_value=1,
            max_value=10,
            value=4,
            step=1,
            help="Number of municipal tipper/suction trucks available for rapid dispatch",
        )
    with sc_col3:
        sim_delay = st.slider(
            "⏱️ 3. Action Delay (Hours)",
            min_value=0.0,
            max_value=24.0,
            value=2.0,
            step=1.0,
            help="Elapsed hours between citizen report alert and municipal crew arrival",
        )
    with sc_col4:
        sim_wind = st.slider(
            "💨 4. Wind Drift (km/h)",
            min_value=0.0,
            max_value=50.0,
            value=15.0,
            step=1.0,
            help="Surface wind pushing loose litter into gutters (raises the flush factor)",
        )
    with sc_col5:
        sim_silt = st.slider(
            "🧱 5. Drain Siltation Level (%)",
            min_value=0.0,
            max_value=100.0,
            value=30.0,
            step=5.0,
            help="Share of drain capacity lost to silt (expands flood area and flush-out)",
        )

    # Run simulation
    sim_res = simulator_engine.simulate(
        rainfall_intensity_mmh=sim_rain,
        available_fleet=sim_fleet,
        action_delay_hours=sim_delay,
        base_hotspots=hotspots,
        wind_speed_kmh=sim_wind,
        drain_silt_pct=sim_silt,
    )
    outcomes = sim_res["outcomes"]
    hydraulics = sim_res["hydraulics"]

    # Scenario Alert Tag
    st.markdown(f"""
    <div style="background:{sim_res['alert_color']}22; border:1px solid {sim_res['alert_color']}; border-radius:10px; padding:10px 16px; margin: 10px 0; display:flex; justify-content:space-between; align-items:center;">
      <div style="font-weight:800; color:{sim_res['alert_color']}; font-size:0.95rem;">
        ⚡ {sim_res['scenario_alert']}
        <span style="font-weight:600; font-size:0.8rem; color:#cbd5e1; margin-left:12px;">
          {sim_res['inputs']['rainfall_category']} • runoff {hydraulics['runoff_per_inlet_lps']:.0f} L/s vs clean-inlet capacity {hydraulics['clean_inlet_capacity_lps']:.0f} L/s • flush factor {hydraulics['flush_factor']:.2f} • silt adds {hydraulics['silt_added_flood_m2']:,.0f} m² flooding
        </span>
      </div>
      <div>
        {render_provenance_badge('simulated', '[Simulated / Modelled Estimate]')}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 6 Modeled Outcomes KPI Cards
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    with s1:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-label">Choking Hotspots</div>
          <div class="stat-val" style="color:#ef4444;">{outcomes['critical_choking_hotspots']}</div>
          <div class="stat-note">Sites with grate overflow</div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-label">Plastic at Risk</div>
          <div class="stat-val" style="color:#38bdf8;">{outcomes['plastic_mass_at_risk_kg']:.0f} kg</div>
          <div class="stat-note">Catchment wash-off mass</div>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-label">Swept Into Rivers</div>
          <div class="stat-val" style="color:#f87171;">{outcomes['plastic_swept_into_rivers_kg']:.0f} kg</div>
          <div class="stat-note">Flushed into Mula-Mutha</div>
        </div>
        """, unsafe_allow_html=True)
    with s4:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-label">Flood Inundation</div>
          <div class="stat-val" style="color:#fbbf24;">{outcomes['potential_flood_inundation_area_sqm']:,.0f} m²</div>
          <div class="stat-note">Submerged urban footprint</div>
        </div>
        """, unsafe_allow_html=True)
    with s5:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-label">Preventable Flood</div>
          <div class="stat-val" style="color:#34d399;">{outcomes['preventable_flood_area_sqm']:,.0f} m²</div>
          <div class="stat-note">Avertable via early fleet</div>
        </div>
        """, unsafe_allow_html=True)
    with s6:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-label">Cost of Inaction</div>
          <div class="stat-val" style="color:#c084fc;">₹{outcomes['estimated_civic_cost_of_inaction_inr']:,.0f}</div>
          <div class="stat-note">Damage & dewatering penalty</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # ── Explainable AI: why the numbers are what they are ─────────────────────
    sim_xai = sim_res["explanation"]
    x_col1, x_col2 = st.columns([1.3, 1.0], gap="large")
    with x_col1:
        bullets = "".join(f"<li style='margin-bottom:6px;'>{line}</li>" for line in sim_xai["narrative"])
        st.markdown(f"""
        <div class="feature-panel" style="border-left:4px solid #38bdf8;">
          <div style="font-size:0.8rem; font-weight:800; color:#94a3b8; text-transform:uppercase; margin-bottom:8px;">💡 Why these numbers? (Explainable model)</div>
          <div style="font-size:0.95rem; font-weight:800; color:#fbbf24; margin-bottom:8px;">{sim_xai['headline']}</div>
          <ul style="font-size:0.84rem; color:#e2e8f0; line-height:1.45; padding-left:18px; margin:0;">{bullets}</ul>
        </div>
        """, unsafe_allow_html=True)
    with x_col2:
        st.markdown("##### 🎚️ What each lever is worth")
        if sim_xai["sensitivity"]:
            sens_df = pd.DataFrame(sim_xai["sensitivity"])
            st.altair_chart(alt.Chart(sens_df).mark_bar(color="#34d399", cornerRadiusEnd=4).encode(
                x=alt.X("cost_saving_inr:Q", title="Cost saved if changed alone (₹)"),
                y=alt.Y("lever:N", sort="-x", title=None),
                tooltip=["lever", "cost_saving_inr", "flood_reduction_m2", "river_plastic_reduction_kg"],
            ).properties(height=170), use_container_width=True)
            st.caption("One-lever-at-a-time sensitivity: each bar re-runs the model with only that input improved.")
        else:
            st.caption("All levers are already at their best values.")

    with st.expander("🔬 Per-site hydraulics & model assumptions"):
        st.dataframe(pd.DataFrame([
            {
                "Site": r["id"], "Drain": r["name"], "Plastic kg": r["plastic_kg"],
                "Cleared by peak": "✅" if r["cleared_before_peak"] else ("⏳ ETA T+%sh" % r["clearance_eta_h"]),
                "Blockage %": r["blockage_pct"], "Runoff L/s": r["runoff_lps"], "Capacity L/s": r["inlet_capacity_lps"],
                "Overflow L/s": r["overflow_lps"], "Flood m²": r["flood_area_m2"], "To river kg": r["swept_kg"],
            }
            for r in sim_res["site_breakdown"]
        ]), use_container_width=True, hide_index=True)
        st.dataframe(pd.DataFrame([
            {"Assumption": k.replace("_", " "), "Value": v["value"], "Unit": v["unit"], "Basis": v["basis"]}
            for k, v in sim_res["assumptions"].items()
        ]), use_container_width=True, hide_index=True)
        st.caption(f"Method: {sim_res['methodology']}.")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Comparative Scenario Cards (Selected vs Optimal vs Worst-Case)
    comp_c1, comp_c2 = st.columns([1.3, 1.0], gap="large")
    with comp_c1:
        st.markdown("#### ⚖️ Comparative Before/After Scenario Benchmarking")
        bench = sim_res["comparative_benchmarks"]
        opt = bench["optimal_immediate_dispatch"]
        worst = bench["unmitigated_worst_case"]

        st.markdown(f"""
        <div class="feature-panel">
          <table style="width:100%; border-collapse:collapse; font-size:0.86rem;">
            <thead>
              <tr style="border-bottom:1px solid #1e4253; color:#94a3b8; text-align:left;">
                <th style="padding:8px 6px;">Scenario Model</th>
                <th style="padding:8px 6px;">Fleet</th>
                <th style="padding:8px 6px;">Delay</th>
                <th style="padding:8px 6px;">Wind / Silt</th>
                <th style="padding:8px 6px;">Plastic in River</th>
                <th style="padding:8px 6px;">Inundation Area</th>
                <th style="padding:8px 6px;">Civic Cost (INR)</th>
              </tr>
            </thead>
            <tbody>
              <tr style="border-bottom:1px solid #133240; color:#34d399;">
                <td style="padding:10px 6px;"><b>🟢 Optimal (Immediate Dispatch)</b></td>
                <td style="padding:10px 6px;">10 trucks</td>
                <td style="padding:10px 6px;">0 hrs</td>
                <td style="padding:10px 6px;">{sim_wind:.0f} km/h / {sim_silt:.0f}%</td>
                <td style="padding:10px 6px;">{opt['plastic_swept_kg']:.0f} kg</td>
                <td style="padding:10px 6px;">&le; {opt.get('flood_area_sqm', 450):,.0f} m²</td>
                <td style="padding:10px 6px;"><b>₹{opt['estimated_cost_inr']:,.0f}</b></td>
              </tr>
              <tr style="border-bottom:1px solid #133240; color:#38bdf8; background:#081b24;">
                <td style="padding:10px 6px;"><b>🔵 Current Selected Scenario</b></td>
                <td style="padding:10px 6px;">{sim_fleet} trucks</td>
                <td style="padding:10px 6px;">{sim_delay:.0f} hrs</td>
                <td style="padding:10px 6px;">{sim_wind:.0f} km/h / {sim_silt:.0f}%</td>
                <td style="padding:10px 6px;"><b>{outcomes['plastic_swept_into_rivers_kg']:.0f} kg</b></td>
                <td style="padding:10px 6px;"><b>{outcomes['potential_flood_inundation_area_sqm']:,.0f} m²</b></td>
                <td style="padding:10px 6px;"><b>₹{outcomes['estimated_civic_cost_of_inaction_inr']:,.0f}</b></td>
              </tr>
              <tr style="color:#ef4444;">
                <td style="padding:10px 6px;"><b>🔴 Worst-Case (Unmitigated Inaction)</b></td>
                <td style="padding:10px 6px;">0 trucks</td>
                <td style="padding:10px 6px;">24 hrs</td>
                <td style="padding:10px 6px;">{sim_wind:.0f} km/h / {sim_silt:.0f}%</td>
                <td style="padding:10px 6px;">{worst['plastic_swept_kg']:.0f} kg</td>
                <td style="padding:10px 6px;">{worst['flood_area_sqm']:,.0f} m²</td>
                <td style="padding:10px 6px;"><b>₹{worst['estimated_cost_inr']:,.0f}</b></td>
              </tr>
            </tbody>
          </table>
          <div style="margin-top:14px; display:flex; justify-content:space-between; align-items:center; font-size:0.8rem; color:#94a3b8;">
            <span>Averting {outcomes['plastic_prevented_from_rivers_pct']:.0f}% of river plastic compared to unmitigated inaction</span>
            {render_provenance_badge('simulated', '[Simulated / Modelled Estimate]')}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with comp_c2:
        st.markdown("#### 💰 Cost of Inaction Breakdown")
        cb = outcomes["cost_breakdown_inr"]
        st.markdown(f"""
        <div class="feature-panel">
          <div style="margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; font-size:0.84rem;">
              <span style="color:#94a3b8;">Suction & Dewatering Pumping:</span>
              <span style="font-weight:700; color:#cbd5e1;">₹{cb['dewatering_pumping']:,.0f}</span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.84rem; margin-top:6px;">
              <span style="color:#94a3b8;">Traffic Disruption Economic Loss:</span>
              <span style="font-weight:700; color:#cbd5e1;">₹{cb['traffic_disruption_impact']:,.0f}</span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.84rem; margin-top:6px;">
              <span style="color:#94a3b8;">River Environmental Penalties:</span>
              <span style="font-weight:700; color:#cbd5e1;">₹{cb['river_restoration_penalty']:,.0f}</span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.84rem; margin-top:6px;">
              <span style="color:#94a3b8;">Emergency Contractor Overtime:</span>
              <span style="font-weight:700; color:#cbd5e1;">₹{cb['emergency_contractor_surcharge']:,.0f}</span>
            </div>
          </div>
          <div style="padding-top:10px; border-top:1px solid #1a3c48; display:flex; justify-content:space-between; font-weight:800; font-size:1.05rem; color:#c084fc;">
            <span>Total Civic Burden:</span>
            <span>₹{outcomes['estimated_civic_cost_of_inaction_inr']:,.0f}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7: EVIDENCE & PROVENANCE LAYER (SYSTEM-WIDE DEFENSIBILITY)
# ══════════════════════════════════════════════════════════════════════════════
with tab_provenance:
    st.markdown("### 🔎 Evidence & Provenance Layer (System-Wide Defensibility)")
    st.markdown(
        "Standardized 4-tier data provenance framework ensuring full legal, auditable, "
        "and defensible transparency for Pune Municipal Corporation (PMC) operators and executive leadership.",
    )

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.markdown(f"""
        <div class="feature-panel" style="border-top:3px solid #10b981;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <span style="font-weight:800; font-size:1.05rem; color:#10b981;">🟢 Tier 1: [Observed]</span>
            {render_provenance_badge('observed')}
          </div>
          <p style="font-size:0.85rem; color:#cbd5e1; line-height:1.45;">
            <b>Ground-Truth Observational Layer:</b> Unmodified data collected directly from field agents, citizen smartphones, and physical sensors.
          </p>
          <ul style="font-size:0.8rem; color:#94a3b8; margin-top:4px;">
            <li>Raw GPS latitude & longitude coordinates</li>
            <li>Immutable ISO-8601 upload timestamps</li>
            <li>Unprocessed citizen camera captures</li>
            <li>PMC field crew physical weigh-scale receipts</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="feature-panel" style="border-top:3px solid #fbbf24;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <span style="font-weight:800; font-size:1.05rem; color:#fbbf24;">🟡 Tier 3: [Inferred]</span>
            {render_provenance_badge('inferred')}
          </div>
          <p style="font-size:0.85rem; color:#cbd5e1; line-height:1.45;">
            <b>Machine Learning & Statistical Deduction:</b> Algorithmic inferences generated via computer vision and probabilistic models.
          </p>
          <ul style="font-size:0.8rem; color:#94a3b8; margin-top:4px;">
            <li>TACO/YOLOv8 detected object bounding boxes & confidence scores</li>
            <li>Before/After cleanup clearance ratios</li>
            <li>Automated contextual root-cause hypotheses</li>
            <li>Dominant polymer classification (HDPE, PET, EPS)</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    with p_col2:
        st.markdown(f"""
        <div class="feature-panel" style="border-top:3px solid #38bdf8;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <span style="font-weight:800; font-size:1.05rem; color:#38bdf8;">🔵 Tier 2: [Derived]</span>
            {render_provenance_badge('derived')}
          </div>
          <p style="font-size:0.85rem; color:#cbd5e1; line-height:1.45;">
            <b>Deterministic Mathematical Calculation:</b> Computed spatial metrics and deterministic formulaic transformations.
          </p>
          <ul style="font-size:0.8rem; color:#94a3b8; margin-top:4px;">
            <li>DBSCAN spatial consolidation halos (eps = 50m)</li>
            <li>Geodesic distance to nearest storm drain channel (m)</li>
            <li>0–100 Weighted Hydrological Risk Score formula</li>
            <li>Optimized TSP municipal truck route polyline</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="feature-panel" style="border-top:3px solid #c084fc;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <span style="font-weight:800; font-size:1.05rem; color:#c084fc;">🟣 Tier 4: [Simulated]</span>
            {render_provenance_badge('simulated')}
          </div>
          <p style="font-size:0.85rem; color:#cbd5e1; line-height:1.45;">
            <b>Parametric Scenario Projections:</b> Hydraulic simulations and predictive what-if scenario estimations.
          </p>
          <ul style="font-size:0.8rem; color:#94a3b8; margin-top:4px;">
            <li>Upcoming 24–72h Open-Meteo precipitation wash-off</li>
            <li>Monsoon cloudburst flood inundation footprint (sq meters)</li>
            <li>Plastic mass flushed into river basin under delay</li>
            <li>Estimated civic economic cost of inaction (INR)</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    # TACO AI Readiness status preserved
    st.markdown("#### 🧠 Detection Engine: Model Card, TACO Statistics & Live Trace")
    mc = model_card()
    taco_stats = taco_annotation_stats()
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    with mc1:
        st.metric("Detector", "YOLOv8n")
    with mc2:
        st.metric("Parameters", f"{mc['parameters'] / 1e6:.2f} M" if mc["parameters"] else "—")
    with mc3:
        st.metric("Pre-training", "COCO (80 cls)")
    with mc4:
        st.metric("TACO annotations", f"{taco_stats.get('annotations', 0):,}" if taco_stats.get("available") else "—")
    with mc5:
        st.metric("TACO fine-tuned", "Yes" if mc["taco_fine_tuned"] else "Not yet")
    st.caption(
        f"Weights {mc['weights_file']} ({mc['weights_mb']} MB, sha1 {mc['weights_sha1']}) • input {mc['input_size_px']} px • "
        f"candidate conf ≥ {mc['candidate_confidence']} • accepted conf ≥ {mc['min_detection_confidence']} • "
        f"plastic probability = {mc['plastic_probability_formula']} ≥ {mc['min_plastic_probability']}. {mc['taco_fine_tune_note']}"
    )

    st.markdown("""
    <div class="feature-panel" style="padding:12px 16px; font-size:0.84rem; color:#cbd5e1; line-height:1.55;">
      <b>Pipeline:</b> ① <b>Quality gate</b> (resolution, exposure, haze, multi-scale blur, motion blur) →
      ② <b>YOLOv8n</b> proposes objects →
      ③ <b>Class policy</b>: plastic candidates (bottle, cup, cutlery, bag…) vs non-plastic waste (food, paper) vs ignored scene context (people, vehicles, furniture) →
      ④ <b>Material check</b> on each candidate crop (specular sheen + synthetic colour vs vegetation) →
      ⑤ <b>Colour/sheen fallback</b> for weathered litter COCO cannot name (marked lower-evidence).
    </div>
    """, unsafe_allow_html=True)

    mcol1, mcol2 = st.columns([1.0, 1.15], gap="large")
    with mcol1:
        if taco_stats.get("available"):
            st.markdown(f"**TACO taxonomy behind the project classes** ({taco_stats['images']:,} images, "
                        f"{taco_stats['categories']} categories, {taco_stats['plastic_share_pct']:.0f}% of mapped labels are plastic)")
            tdf = pd.DataFrame(taco_stats["per_class"])
            st.altair_chart(alt.Chart(tdf).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("taco_annotations:Q", title="TACO annotations"),
                y=alt.Y("project_class:N", sort="-x", title=None),
                color=alt.Color("is_plastic:N", scale=alt.Scale(domain=[True, False], range=["#c084fc", "#64748b"]), legend=alt.Legend(title="Plastic")),
                tooltip=["project_class", "taco_annotations", "share_pct", "images_containing", "median_box_area_pct", "taco_labels_mapped"],
            ).properties(height=180), use_container_width=True)
            st.dataframe(tdf.rename(columns={
                "project_class": "Class", "is_plastic": "Plastic", "taco_annotations": "Annotations", "share_pct": "Share %",
                "images_containing": "Images", "median_box_area_pct": "Median box % of image", "taco_labels_mapped": "TACO labels",
            }), use_container_width=True, hide_index=True)
        with st.expander("COCO → project class policy"):
            st.dataframe(pd.DataFrame(mc["class_policy"]), use_container_width=True, hide_index=True)
            st.caption("Any COCO class not listed (person, car, bench, bed, table, plant…) is treated as scene context and never counted as waste.")
        with st.expander("Quality gate thresholds"):
            st.dataframe(pd.DataFrame(mc["quality_tiers"]).T, use_container_width=True)

    with mcol2:
        trace_samples = [s["filename"] for s in get_sample_images()]
        trace_pick = st.selectbox("Run a live trace on a sample photo", trace_samples, key="trace_sample")
        trace_up = st.file_uploader("…or upload any photo to trace", type=["jpg", "jpeg", "png", "webp"], key="trace_upload")
        try:
            trace_res = detect_waste(trace_up.getvalue() if trace_up else str(ROOT / "data" / "taco_samples" / trace_pick))
        except Exception as exc:
            trace_res = None
            st.error(f"Could not analyse photo: {exc}")
        if trace_res:
            st.image(trace_res["annotated_image"], use_container_width=True,
                     caption=f"{trace_res['item_count']} plastic item(s) • evidence: {trace_res['evidence_level']} • severity {trace_res['severity']}/5")
            q = trace_res["image_quality"]
            st.caption(("❌ " + "; ".join(q["reasons"])) if q["needs_review"] else
                       f"✅ Quality OK (citizen tier) • sharpness {q['sharpness_ratio']:.2f} • isotropy {q['edge_isotropy']:.2f} • luminance {q['mean_luminance']:.0f}")
            if trace_res["trace"]:
                st.dataframe(pd.DataFrame([
                    {"YOLO class": t.get("coco_class"), "Conf": t.get("detector_confidence", "—"), "Role": t.get("role"),
                     "Plastic prob.": t.get("plastic_probability", "—"), "Decision": t.get("decision"), "Why": t.get("reason")}
                    for t in trace_res["trace"]
                ]).astype(str), use_container_width=True, hide_index=True)
            else:
                st.caption("YOLO proposed no objects and the colour/sheen fallback found no plastic signature.")
    st.caption(dataset_status().get("license", "TACO dataset: CC BY 4.0"))
