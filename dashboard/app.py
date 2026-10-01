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

from backend.plasticwatch_pipeline import PlasticWatchUrbanPipeline, run_plasticwatch
from backend.route_optimizer import RouteOptimizer
from backend.taco_adapter import dataset_status, detect_waste, get_sample_images

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
tab_command, tab_mobile, tab_taco = st.tabs([
    "🏛️ Municipal Command Center (3-Panel Operational Triage)",
    "📱 Citizen Mobile Reporter (Paper-Themed PWA View)",
    "🧠 AI & TACO Model Readiness",
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

            st.markdown(f"""
            <div class="queue-card {active_class}">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                <span style="font-weight: 800; font-size: 0.98rem; color: #f8fafc;">{hid}</span>
                <span class="{badge_class}">{priority} {score:.1f}</span>
              </div>
              <div style="font-size: 0.8rem; color: #cbd5e1; margin-bottom: 5px;">
                🌊 <b>{dist_m:.0f}m</b> to {drain_name}
              </div>
              <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.75rem;">
                <span class="{ 'badge-merged' if is_merged else 'tiny-text' }">
                  👥 {recurrence} {'reports merged' if is_merged else 'single report'}
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
            st.markdown("#### ⚡ Exploded Risk Breakdown")
            
            # Drain Proximity Component (35%)
            d_pts = active_h.get("drain_points", 0.0)
            st.markdown(f"**Drain Proximity**: {dist_m:.0f}m to *{drain_name}*")
            st.caption(f"Score: {active_h.get('drain_score', 0):.0f}/100 • **+{d_pts:.1f} pts** (35% weight)")
            st.progress(min(1.0, active_h.get("drain_score", 0) / 100.0))

            # Waste Severity Component (25%)
            s_pts = active_h.get("severity_points", 0.0)
            raw_sev = active_h.get("severity_raw", 3.5)
            st.markdown(f"**Detection Severity**: {raw_sev:.1f}/5.0 (Clog Hazard)")
            st.caption(f"Score: {active_h.get('severity_score', 0):.0f}/100 • **+{s_pts:.1f} pts** (25% weight)")
            st.progress(min(1.0, active_h.get("severity_score", 0) / 100.0))

            # AI Detection Confidence Component (20%)
            c_pts = active_h.get("confidence_points", 0.0)
            raw_conf = active_h.get("confidence_raw", 0.88)
            st.markdown(f"**AI Confidence**: {raw_conf:.0%} (TACO YOLOv8)")
            st.caption(f"Score: {active_h.get('confidence_score', 0):.0f}/100 • **+{c_pts:.1f} pts** (20% weight)")
            st.progress(min(1.0, active_h.get("confidence_score", 0) / 100.0))

            # 24h Rainfall Forecast Component (20%)
            r_pts = active_h.get("rainfall_points", 0.0)
            r_mm = active_h.get("rainfall_forecast_mm", 16.5)
            st.markdown(f"**Rainfall Forecast**: {r_mm:.1f} mm in next 24h")
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
# TAB 2: CITIZEN MOBILE SIMULATOR (Paper-Themed Light Editorial View)
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
              <div><b>GPS Lock:</b> <span class="font-mono-code">{mock_lat:.4f}° N, {mock_lon:.4f}° E</span></div>
              <div><b>AI Confidence:</b> {detection['mean_confidence']:.0%} (TACO taxonomy)</div>
              <div><b>Clog Severity:</b> <span style="color:#a63d1e; font-weight:800;">{detection['severity']}/5.0</span></div>
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
# TAB 3: TACO DATASET & MODEL LAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_taco:
    st.markdown("### 🧠 TACO Dataset & AI Inference Engine")
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

        st.markdown("#### 🎯 Curated Urban Waste Target Classes")
        class_cols = st.columns(len(taco["target_classes"]))
        icons = ["🍾", "🛍️", "🥫", "📦", "🥤"]
        for col, target, icon in zip(class_cols, taco["target_classes"], icons):
            with col:
                st.markdown(f"""
                <div style="background:#0e232e; border:1px solid #1f4255; border-radius:10px; padding:14px; text-align:center;">
                  <div style="font-size:2rem;">{icon}</div>
                  <div style="font-weight:700; color:#38bdf8; font-size:0.88rem; margin-top:6px;">{target.replace('_', ' ').title()}</div>
                  <div class="tiny-text" style="margin-top:2px;">TACO COCO Class</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("TACO dataset status unavailable.")
