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
from PIL import Image
import streamlit as st
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.forecasting_engine import ForecastingEngine
from backend.plasticwatch_pipeline import PlasticWatchUrbanPipeline, run_plasticwatch
from backend.provenance import (
    PROVENANCE_TIERS,
    get_provenance_meta,
    render_provenance_badge,
)
from backend.recurrence_engine import RecurrenceEngine
from backend.route_optimizer import RouteOptimizer
from backend.simulator_engine import MonsoonSimulatorEngine
from backend.taco_adapter import dataset_status, detect_waste, get_sample_images
from backend.verification_engine import CleanupVerificationEngine

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

forecasting_engine = st.session_state.forecasting_engine_instance
verification_engine = st.session_state.verification_engine_instance
recurrence_engine = st.session_state.recurrence_engine_instance
simulator_engine = st.session_state.simulator_engine_instance

pipeline = st.session_state.pipeline_instance
pipeline_data = pipeline.run_pipeline(rainfall_override_mm=st.session_state.rain_override)
hotspots = pipeline_data["hotspots"]
reports = pipeline_data["reports"]
metrics = pipeline_data["metrics"]

# Ensure selected hotspot exists
if not any(h["id"] == st.session_state.selected_hotspot_id for h in hotspots):
    if hotspots:
        st.session_state.selected_hotspot_id = hotspots[0]["id"]


# ── Top Hero & Key Metric Cards (Spacious & Clean) ───────────────────────────
st.markdown("""
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
        <span style="font-size: 0.95rem;">🌧️</span> <b>24h Rain Alert: 16.5 mm</b>
      </div>
      <div class="tiny-text" style="margin-top: 5px;">Open-Meteo Precipitation Live Feed</div>
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

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)


# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab_command, tab_forecast, tab_verify, tab_recurrence, tab_simulator, tab_mobile, tab_provenance = st.tabs([
    "🏛️ Triage & Live Map",
    "🔮 Predictive Forecasting (24–72h)",
    "📸 AI Cleanup Verification",
    "🔁 Recurrence & Root Cause",
    "☔ Monsoon What-If Simulator",
    "📱 Citizen Mobile Reporter",
    "🔎 Evidence & Provenance Layer",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: 3-PANEL MUNICIPAL COMMAND CENTER
# ══════════════════════════════════════════════════════════════════════════════
with tab_command:
    col_left, col_center, col_right = st.columns([1.18, 2.12, 1.42], gap="large")

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 1: LEFT CLEANUP QUEUE (Ranked 0-100)
    # ──────────────────────────────────────────────────────────────────────────
    with col_left:
        st.markdown("### 📋 Cleanup Queue")
        st.caption("Ranked by 0–100 Hydrological Risk Score")

        priority_filter = st.selectbox(
            "Filter Priority",
            ["All Priorities", "CRITICAL (Score ≥ 70)", "HIGH (Score ≥ 50)", "MODERATE (Score < 50)"],
            label_visibility="collapsed",
        )

        filtered_hotspots = hotspots
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

            st.markdown(f"""
            <div class="queue-card {active_class}">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                <span style="font-weight: 800; font-size: 0.98rem; color: #f8fafc;">{hid}</span>
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

            # Sample Waste Photo Preview
            sample_photo = active_h.get("primary_photo", "sample_bottles_drain.jpg")
            sample_path = ROOT / "data" / "taco_samples" / sample_photo
            if sample_path.is_file():
                st.image(str(sample_path), caption=f"Evidence Capture • {sample_photo}", use_container_width=True)

            # Exploded 0-100 Score Breakdown
            st.markdown(f"#### ⚡ Exploded Risk Breakdown {render_provenance_badge('derived')}", unsafe_allow_html=True)
            
            # Drain Proximity Component (35%)
            d_pts = active_h.get("drain_points", 0.0)
            st.markdown(f"**Drain Proximity** {render_provenance_badge('derived')}: {dist_m:.0f}m to *{drain_name}*", unsafe_allow_html=True)
            st.caption(f"Score: {active_h.get('drain_score', 0):.0f}/100 • **+{d_pts:.1f} pts** (35% weight)")
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
                if st.button("🚨 Verify & Dispatch Work Order", type="primary", use_container_width=True, key=f"dispatch_{hid}"):
                    target_ids = [hid]
                    for oth in hotspots:
                        if oth["priority"] == "CRITICAL" and oth["id"] != hid:
                            target_ids.append(oth["id"])

                    plan = pipeline.dispatch_work_order(target_ids)
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
                    Stops: <b>{plan.get('total_stops')} hotspots</b> from Swargate Central Depot
                  </div>
                </div>
                """, unsafe_allow_html=True)


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
# TAB 3: AI BEFORE/AFTER CLEANUP VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_verify:
    st.markdown("### 📸 AI Before/After Cleanup Verification")
    st.markdown(
        f"Dual-Image TACO/YOLO Computer Vision Inference • Delta Severity Quantification • "
        f"{render_provenance_badge('inferred', '[AI Inference / Modelled Estimate]')}",
        unsafe_allow_html=True,
    )
    st.caption("Verifies contractor clearance effectiveness before releasing municipal invoices. Requires >= 85% clearance score.")

    v_cases = {
        "Shaniwar Wada Culvert (Verified Clearance • PASS)": {
            "before": "sample_bottles_drain.jpg",
            "after": "sample_cleared_drain.jpg",
            "site": "Shaniwar Wada Culvert (Kasba Peth)",
            "hotspot_id": "HOTSPOT-01",
            "contractor": "PMC Sanitation Rapid Crew 1",
        },
        "Nagzari Nallah Grate (Verified Clearance • PASS)": {
            "before": "sample_food_wrappers_culvert.jpg",
            "after": "sample_cleared_drain.jpg",
            "site": "Nagzari Nallah Storm Grate (Bhavani Peth)",
            "hotspot_id": "HOTSPOT-02",
            "contractor": "PMC Sanitation Rapid Crew 2",
        },
        "Mutha River Outfall (Verified Clearance • PASS)": {
            "before": "sample_polypropylene_sacks.jpg",
            "after": "sample_cleared_curb.jpg",
            "site": "Mutha River Confluence Outfall",
            "hotspot_id": "HOTSPOT-03",
            "contractor": "PMC Heavy River Skimmer Crew 4",
        },
        "Shaniwar Wada Re-Inspection (Incomplete Cleanup • FLAGGED)": {
            "before": "sample_bottles_drain.jpg",
            "after": "sample_partial_cleanup.jpg",
            "site": "Shaniwar Wada Culvert Curb Line",
            "hotspot_id": "HOTSPOT-01",
            "contractor": "PMC Sanitation Rapid Crew 1",
        },
    }

    v_col_ctrl, v_col_meta = st.columns([1.5, 1.0])
    with v_col_ctrl:
        chosen_case_name = st.selectbox("Select Verification Work Order Inspection", list(v_cases.keys()))
        case_info = v_cases[chosen_case_name]
    with v_col_meta:
        st.markdown(f"""
        <div style="background:#0b1c24; border:1px solid #1a3c48; border-radius:10px; padding:10px 14px; font-size:0.82rem; margin-top:18px;">
          <div>🏢 <b>Site:</b> {case_info['site']}</div>
          <div>👷 <b>Contractor:</b> {case_info['contractor']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Run dual verification
    v_result = verification_engine.verify_cleanup(
        before_image=case_info["before"],
        after_image=case_info["after"],
        hotspot_id=case_info["hotspot_id"],
        site_name=case_info["site"],
        contractor_id=case_info["contractor"],
    )

    # Side-by-side Visual Photo Comparison
    v_left, v_right = st.columns(2)
    with v_left:
        st.markdown("#### 📷 Baseline Before-Cleanup Photo")
        st.caption(f"Pre-intervention optical capture • {case_info['before']}")
        b_b64 = v_result["visuals"]["before_annotated_base64"]
        if b_b64:
            st.image(base64.b64decode(b_b64), caption=f"Baseline: {v_result['metrics']['before_object_count']} waste objects detected (Severity: {v_result['metrics']['before_severity']}/5.0)", use_container_width=True)
        else:
            p = ROOT / "data" / "taco_samples" / case_info["before"]
            if p.is_file():
                st.image(str(p), use_container_width=True)

    with v_right:
        st.markdown("#### 📷 Clearance After-Cleanup Photo")
        st.caption(f"Post-intervention contractor submission • {case_info['after']}")
        a_b64 = v_result["visuals"]["after_annotated_base64"]
        if a_b64:
            st.image(base64.b64decode(a_b64), caption=f"Clearance: {v_result['metrics']['after_object_count']} objects remaining (Severity: {v_result['metrics']['after_severity']}/5.0)", use_container_width=True)
        else:
            p = ROOT / "data" / "taco_samples" / case_info["after"]
            if p.is_file():
                st.image(str(p), use_container_width=True)

    # 4 Quantification KPI Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-label">Object Count Reduction</div>
          <div class="stat-val" style="color:#38bdf8;">-{v_result['metrics']['count_reduction']}</div>
          <div class="stat-note"><b>{v_result['metrics']['count_reduction_pct']:.0f}%</b> objects eliminated</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-label">Severity Reduction (&Delta;)</div>
          <div class="stat-val" style="color:#34d399;">-{v_result['metrics']['delta_severity']:.1f}</div>
          <div class="stat-note"><b>{v_result['metrics']['severity_reduction_pct']:.0f}%</b> risk decline</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        eff = v_result['effectiveness_score']
        eff_color = "#10b981" if eff >= 85 else "#f59e0b"
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-label">Effectiveness Score</div>
          <div class="stat-val" style="color:{eff_color};">{eff:.1f}%</div>
          <div class="stat-note">Standard target &ge; 85%</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-label">Diverted Plastic Mass</div>
          <div class="stat-val" style="color:#c084fc;">{v_result['metrics']['diverted_plastic_weight_kg']} kg</div>
          <div class="stat-note">~{v_result['metrics']['diverted_volume_liters']:.0f} L volume diverted</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Adjudication Decision Banner
    dec = v_result["decision_status"]
    if dec == "PASS":
        st.markdown(f"""
        <div style="background:rgba(16, 185, 129, 0.15); border:2px solid #10b981; border-radius:12px; padding:16px; margin: 10px 0;">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="font-size:1.15rem; font-weight:800; color:#34d399;">
              ✅ DECISION STATUS: PASS ({eff:.1f}% Clearance)
            </div>
            {render_provenance_badge('inferred', '[AI Inference / Modelled Estimate]')}
          </div>
          <div style="font-size:0.86rem; color:#e2e8f0; margin-top:6px;">
            Work order satisfies municipal clearance threshold (&ge; 85%). Contractor invoice eligible for approval.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:rgba(245, 158, 11, 0.15); border:2px solid #f59e0b; border-radius:12px; padding:16px; margin: 10px 0;">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="font-size:1.15rem; font-weight:800; color:#fbbf24;">
              ⚠️ DECISION STATUS: FLAGGED FOR REVIEW ({eff:.1f}% Clearance)
            </div>
            {render_provenance_badge('inferred', '[AI Inference / Modelled Estimate]')}
          </div>
          <div style="font-size:0.86rem; color:#e2e8f0; margin-top:6px;">
            Clearance effectiveness is below the required 85% threshold. Field re-inspection required before contractor invoice clearance.
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Mandatory Legal Disclaimer Box
    st.markdown(f"""
    <div class="disclaimer-box">
      <b>⚠️ Disclaimer:</b> {v_result['disclaimer']}
    </div>
    """, unsafe_allow_html=True)

    # Operator Sign-Off Interaction
    invoice_key = f"{case_info['hotspot_id']}_{chosen_case_name}"
    is_signed = invoice_key in st.session_state.signed_off_invoices

    s_col1, s_col2 = st.columns([1.5, 1.0])
    with s_col1:
        op_notes = st.text_input("Operator Field Verification Remarks", value="Visual drain clearance verified. Intake grate unobstructed.", key=f"notes_{invoice_key}")
    with s_col2:
        st.write("")
        st.write("")
        if is_signed:
            st.success("✅ Signed Off & Cleared for Contractor Payment")
        else:
            if st.button("✍️ Sign-Off & Approve Invoice", type="primary", use_container_width=True, key=f"sign_{invoice_key}"):
                st.session_state.signed_off_invoices.add(invoice_key)
                st.toast(f"✅ Contractor invoice signed off for {case_info['site']}!")
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: RECURRENCE & ROOT-CAUSE INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_recurrence:
    st.markdown("### 🔁 Recurrence & Root-Cause Intelligence")
    st.markdown(
        f"Pune Pilot 30-Day Historical Dossier • Chronic Choke Points • Systemic Civic Interventions • "
        f"{render_provenance_badge('inferred', '[Inferred / Prototype Historical Dossier]')}",
        unsafe_allow_html=True,
    )
    st.caption("Analyzes repeat hotspot coordinates and contractor dispatches to deduce underlying root causes and suggest systemic civic interventions.")

    rec_analysis = recurrence_engine.analyze_recurrence()
    summary = rec_analysis["summary_metrics"]

    # 5 KPI Cards
    r1, r2, r3, r4, r5 = st.columns(5)
    with r1:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-label">Monitored Locations</div>
          <div class="stat-val" style="color:#38bdf8;">{summary['total_monitored_sites']}</div>
          <div class="stat-note">Pune Municipal Pilot</div>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-label">30-Day Cleanups</div>
          <div class="stat-val" style="color:#c084fc;">{summary['total_cleanups_completed']}</div>
          <div class="stat-note">Completed work orders</div>
        </div>
        """, unsafe_allow_html=True)
    with r3:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-label">Total Diverted Plastic</div>
          <div class="stat-val" style="color:#34d399;">{summary['total_diverted_plastic_kg']:.0f} kg</div>
          <div class="stat-note">Removed from drainage</div>
        </div>
        """, unsafe_allow_html=True)
    with r4:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-label">Avg Return Interval</div>
          <div class="stat-val" style="color:#f59e0b;">{summary['average_return_interval_days']:.1f} d</div>
          <div class="stat-note">Days until site reclogs</div>
        </div>
        """, unsafe_allow_html=True)
    with r5:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-label">Chronic Choke Rate</div>
          <div class="stat-val" style="color:#ef4444;">{summary['chronic_site_percentage']:.0f}%</div>
          <div class="stat-note">Persistence index &ge; 0.75</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Location Selector
    loc_names = [f"{loc['site_id']} - {loc['name']}" for loc in rec_analysis["locations"]]
    selected_loc_idx = st.selectbox("Select Historical Dossier Location", range(len(loc_names)), format_func=lambda i: loc_names[i])
    selected_site = rec_analysis["locations"][selected_loc_idx]

    # Site dossier detail layout
    s_col_left, s_col_right = st.columns([1.25, 1.0], gap="large")

    with s_col_left:
        st.markdown(f"#### 📜 Location Dossier: {selected_site['name']}")
        st.markdown(f"""
        <div class="feature-panel">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <span style="font-size:1.1rem; font-weight:800; color:#f8fafc;">{selected_site['ward']}</span>
            <span class="badge-critical" style="background:rgba(239, 68, 68, 0.2);">{selected_site['persistence_level']} PERSISTENCE ({selected_site['recurrence_persistence_index']})</span>
          </div>
          <div style="font-size:0.86rem; color:#cbd5e1; margin-bottom:12px;">
            🌊 Connected Waterway: <b>{selected_site['drain_name']}</b> ({selected_site['drain_distance_m']}m)
          </div>
          <div style="display:flex; gap:16px; margin-bottom:14px; background:#081b24; padding:10px 14px; border-radius:10px;">
            <div>
              <div class="tiny-text">Total Recurrences</div>
              <div style="font-weight:800; color:#38bdf8; font-size:1.2rem;">{selected_site['total_recurrence_count']} events</div>
            </div>
            <div>
              <div class="tiny-text">Avg Return Interval</div>
              <div style="font-weight:800; color:#f59e0b; font-size:1.2rem;">{selected_site['average_return_interval_days']} days</div>
            </div>
            <div>
              <div class="tiny-text">Diverted Plastic</div>
              <div style="font-weight:800; color:#34d399; font-size:1.2rem;">{selected_site['total_diverted_weight_kg']} kg</div>
            </div>
          </div>
          <div style="margin-bottom:12px;">
            <div style="font-size:0.82rem; font-weight:700; color:#94a3b8; text-transform:uppercase;">Dominant Waste Stream</div>
            <div style="font-size:0.95rem; font-weight:700; color:#fcd34d; margin-top:2px;">
              🛍️ {selected_site['dominant_waste_stream']}
            </div>
          </div>
          <div style="background:rgba(245, 158, 11, 0.08); border-left:4px solid #f59e0b; padding:12px 14px; border-radius:6px; margin-bottom:14px;">
            <div style="font-size:0.82rem; font-weight:800; color:#fbbf24; text-transform:uppercase; margin-bottom:4px;">
              💡 Automated Contextual Root-Cause Hypothesis
            </div>
            <div style="font-size:0.88rem; color:#e2e8f0; line-height:1.45;">
              "{selected_site['root_cause_hypothesis']}"
            </div>
          </div>
          <div>
            <div style="font-size:0.82rem; font-weight:800; color:#38bdf8; text-transform:uppercase; margin-bottom:6px;">
              🏛️ Suggested Systemic Civic Interventions
            </div>
        """, unsafe_allow_html=True)

        for inv in selected_site["systemic_interventions"]:
            st.markdown(f"<div style='font-size:0.84rem; color:#cbd5e1; margin-bottom:4px;'>📌 <b>{inv}</b></div>", unsafe_allow_html=True)

        st.markdown(f"""
          <div style="margin-top:12px; text-align:right;">
            {render_provenance_badge('inferred', '[Inferred / Prototype Historical Dossier]')}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with s_col_right:
        st.markdown("#### 📈 Return-Rate Trend Sparkline")
        spark_df = pd.DataFrame(selected_site["sparkline_data"])
        if not spark_df.empty:
            chart = alt.Chart(spark_df).mark_bar(color="#38bdf8", cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X("date:T", title="Cleanup Date"),
                y=alt.Y("weight_kg:Q", title="Plastic Removed (kg)"),
                tooltip=["date:T", "weight_kg:Q", "effectiveness_pct:Q", "status:N"],
            ).properties(height=220)
            st.altair_chart(chart, use_container_width=True)

        st.markdown("#### 📋 30-Day Historical Cleanup Logs")
        cleanups_df = pd.DataFrame([
            {
                "Log ID": c["log_id"],
                "Date": c["timestamp"].split("T")[0],
                "Weight (kg)": f"{c['weight_removed_kg']:.1f}",
                "Effectiveness": f"{c['clearance_effectiveness_pct']:.0f}%",
                "Status": c["verification_status"],
                "Crew": c["crew_id"],
            }
            for c in reversed(selected_site["cleanups"])
        ])
        st.dataframe(cleanups_df, use_container_width=True, hide_index=True)


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
    st.caption("Models urban flood inundation footprint, plastic mass flushed into river channels, and economic inaction penalties across variable rainfall intensities, truck fleets, and dispatch delays.")

    # 3 Dynamic Sliders
    sc_col1, sc_col2, sc_col3 = st.columns(3)
    with sc_col1:
        sim_rain = st.slider(
            "🌧️ 1. Rainfall Intensity (mm/hr)",
            min_value=0.0,
            max_value=75.0,
            value=25.0,
            step=2.5,
            help="0 mm/hr = Dry; 15 mm/hr = Moderate; 40 mm/hr = Heavy; 75 mm/hr = Severe Cloudburst",
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

    # Run simulation
    sim_res = simulator_engine.simulate(
        rainfall_intensity_mmh=sim_rain,
        available_fleet=sim_fleet,
        action_delay_hours=sim_delay,
        base_hotspots=hotspots,
    )
    outcomes = sim_res["outcomes"]

    # Scenario Alert Tag
    st.markdown(f"""
    <div style="background:{sim_res['alert_color']}22; border:1px solid {sim_res['alert_color']}; border-radius:10px; padding:10px 16px; margin: 10px 0; display:flex; justify-content:space-between; align-items:center;">
      <div style="font-weight:800; color:{sim_res['alert_color']}; font-size:0.95rem;">
        ⚡ {sim_res['scenario_alert']}
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
                <td style="padding:10px 6px;">{opt['plastic_swept_kg']:.0f} kg</td>
                <td style="padding:10px 6px;">&le; 450 m²</td>
                <td style="padding:10px 6px;"><b>₹{opt['estimated_cost_inr']:,.0f}</b></td>
              </tr>
              <tr style="border-bottom:1px solid #133240; color:#38bdf8; background:#081b24;">
                <td style="padding:10px 6px;"><b>🔵 Current Selected Scenario</b></td>
                <td style="padding:10px 6px;">{sim_fleet} trucks</td>
                <td style="padding:10px 6px;">{sim_delay:.0f} hrs</td>
                <td style="padding:10px 6px;"><b>{outcomes['plastic_swept_into_rivers_kg']:.0f} kg</b></td>
                <td style="padding:10px 6px;"><b>{outcomes['potential_flood_inundation_area_sqm']:,.0f} m²</b></td>
                <td style="padding:10px 6px;"><b>₹{outcomes['estimated_civic_cost_of_inaction_inr']:,.0f}</b></td>
              </tr>
              <tr style="color:#ef4444;">
                <td style="padding:10px 6px;"><b>🔴 Worst-Case (Unmitigated Inaction)</b></td>
                <td style="padding:10px 6px;">0 trucks</td>
                <td style="padding:10px 6px;">24 hrs</td>
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
# TAB 6: CITIZEN MOBILE SIMULATOR (Paper-Themed Light Editorial View)
# ══════════════════════════════════════════════════════════════════════════════
with tab_mobile:
    m_col1, m_col2 = st.columns([1.1, 1.2], gap="large")

    with m_col1:
        st.markdown("### 📱 Citizen Mobile Reporter (Paper-Themed View)")
        st.caption("Styled with warm paper tones, serif typography, and complementing botanical & terracotta accents.")

        # Light Paper-Themed Smartphone Container
        st.markdown("""
        <div class="paper-mobile-frame">
          <div style="display:flex; justify-content:space-between; align-items:center; font-size:11px; color:#6b6359; margin-bottom:10px; padding-bottom:6px; border-bottom:1px solid #ece4d4;">
            <span class="font-mono-code" style="font-weight:600;">09:41</span>
            <span style="color:#284e3a; font-weight:600;">● GPS LOCKED (PMC ZONE)</span>
            <span>100% 🔋</span>
          </div>
          <div style="text-align:center; margin-bottom:14px;">
            <div style="font-size:18px; font-weight:800; color:#1c1917; font-family:'Newsreader', Georgia, serif;">PlasticWatch Civic</div>
            <div style="font-size:11px; font-style:italic; color:#78716c;">Field Registry & Urban Drain Triage</div>
          </div>
        """, unsafe_allow_html=True)

        # Live Image Selector for Mobile
        sample_catalog = get_sample_images()
        sample_names = [s["filename"] for s in sample_catalog] if sample_catalog else ["sample_bottles_drain.jpg"]
        picked_filename = st.selectbox("Curated Evidence Photo", sample_names, index=0)

        # Auto-GPS Mock Coordinates
        preset_coords = {
            "Shaniwar Wada Culvert (0.3m to drain)": (18.5204, 73.8568),
            "Nagzari Nallah Grate (8m to drain)": (18.5060, 73.8625),
            "Mutha River Outfall (0m to river)": (18.5350, 73.8920),
            "Kothrud West Stormwater Feeder": (18.5074, 73.8077),
        }
        picked_preset = st.selectbox("Auto-GPS Drainage Preset", list(preset_coords.keys()))
        mock_lat, mock_lon = preset_coords[picked_preset]

        # Display Selected Photo & Run Detection Preview
        chosen_sample_path = ROOT / "data" / "taco_samples" / picked_filename
        if chosen_sample_path.is_file():
            detection = detect_waste(str(chosen_sample_path), filename_hint=picked_filename)
            if "annotated_image_base64" in detection:
                img_data = base64.b64decode(detection["annotated_image_base64"])
                st.image(img_data, caption=f"Field Optical Analysis • {detection['hazard_level']} • {len(detection['detected_items'])} items", use_container_width=True)

            st.markdown(f"""
            <div style="background:#faf6ee; border:1px solid #e3d7c3; border-radius:10px; padding:10px 12px; margin:10px 0; font-family:'Newsreader', serif; font-size:12px; color:#2d2824;">
              <div><b>GPS Lock:</b> <span class="font-mono-code">{mock_lat:.4f}° N, {mock_lon:.4f}° E</span> {render_provenance_badge('observed')}</div>
              <div><b>AI Confidence:</b> {detection['mean_confidence']:.0%} (TACO taxonomy) {render_provenance_badge('inferred')}</div>
              <div><b>Clog Severity:</b> <span style="color:#a63d1e; font-weight:800;">{detection['severity']}/5.0</span> {render_provenance_badge('inferred')}</div>
            </div>
            """, unsafe_allow_html=True)

        user_notes = st.text_input("Field Note", "Roadside bottles and film blinding intake grate")

        if st.button("📝 Record Citizen Field Report", type="primary", use_container_width=True):
            res = pipeline.submit_report(
                lat=mock_lat,
                lon=mock_lon,
                photo_filename=picked_filename,
                notes=user_notes,
            )
            st.toast(f"✅ {res['message']}")
            st.success(f"Report Recorded! Ref: {res['report']['id']} (Clustered into Municipal Queue)")
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with m_col2:
        st.markdown("### 🎥 Screen-Recording & DevTools Integration")
        st.write("""
        You can demo the paper-themed mobile reporting experience in two complementary ways:
        
        1. **Standalone PWA Web Client**:
           - Open **`http://localhost:8000/mobile`** in your browser.
           - Experience the warm paper parchment texture, serif typography (`Newsreader`), and deep forest green accents.
           - Open Chrome DevTools (`Ctrl+Shift+I` / `Cmd+Option+I`), toggle **Device Toolbar** (`Ctrl+Shift+M`), select **iPhone 14** or **Pixel 7**, and screen-record!
        
        2. **Embedded Simulator (Left Panel)**:
           - Use the interactive smartphone container on the left directly in Streamlit.
           - Pick sample roadside litter images and inject simulated drain coordinates.
           - Reports instantly feed the DBSCAN clustering engine and update the Municipal Queue!
        """)

        st.info("💡 **Zero-Hardware, API-First Rollout**: This paper-themed PWA directly communicates with the Python FastAPI backend, avoiding native APK compile delays.")

        # Show Live Reports Stream
        st.markdown("#### 📥 Ingested Citizen Reports Feed")
        rep_df = pd.DataFrame([
            {
                "ID": r["id"],
                "Lat/Lon": f"{r['lat']:.4f}, {r['lon']:.4f}",
                "Confidence": f"{r.get('plastic_confidence', 0.88):.0%}",
                "Severity": f"{r.get('severity', 3.5)}/5",
                "Status": r.get("verification", "pending"),
                "Source": r.get("source", "Citizen Mobile"),
            }
            for r in reversed(reports)
        ])
        st.dataframe(rep_df, use_container_width=True, hide_index=True)


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
    st.markdown("#### 🧠 TACO Model Readiness & Dataset Attribution")
    taco = dataset_status()
    if taco.get("available"):
        tc1, tc2, tc3, tc4 = st.columns(4)
        with tc1:
            st.metric("TACO Annotations", f"{taco['annotations']:,}")
        with tc2:
            st.metric("Target Classes", len(taco["target_classes"]))
        with tc3:
            st.metric("Curated Roadside Samples", taco.get("sample_images", 5))
        with tc4:
            st.metric("Inference Engine", "Active (Calibrated YOLOv8)")
        st.caption(taco.get("license", "CC BY 4.0 Attribution"))
