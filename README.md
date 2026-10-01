# PlasticWatch: Urban Civic Environmental Intelligence Command Center

**PlasticWatch** is an explainable AI-GIS decision-support system that pivots municipal sanitation from reactive complaints to proactive, data-driven hydrological triage. It ingests citizen smartphone captures, performs computer vision waste detection (TACO / YOLOv8), merges duplicate scattered reports via **DBSCAN spatial clustering ($eps = 50\text{ m}$)**, evaluates risk against urban stormwater drainage networks and rainfall forecasts, and optimizes municipal cleanup truck dispatch routes.

---

## Prototype Architecture & "Zero-Hardware, API-First" Pipeline

```
[Simulated Mobile Client (Web PWA/Mock)]
   │ (Photo + Lat/Lng + Timestamp)
   ▼
[FastAPI / Python Backend]
   ├── 1. Vision: YOLOv8 / TACO detector (crops plastic, confidence score & severity)
   ├── 2. Spatial Clustering: DBSCAN (eps=50m) -> Merges nearby duplicate reports
   ├── 3. Risk Engine: OSM drain proximity + Open-Meteo rainfall -> Score (0-100)
   └── 4. Optimizer: TSP / RouteOptimizer -> Municipal truck cleanup itinerary
   ▼
[Leaflet / Streamlit Operator Command Center]
   (Hotspot map, 0-100 risk score breakdown, verify button, cleanup queue)
```

---

## Core Capabilities

1. **Source Ingestion (Citizen Mobile Simulation):**
   - Responsive HTML5/Tailwind PWA mobile client (`/mobile`) styled for smartphone viewports (iPhone 14 / Pixel 7).
   - Auto-GPS mock injecting test coordinates along urban drainage networks.
   - Live AI bounding box preview showing detected objects (`plastic_bottle`, `plastic_bag_wrapper`, `can_metal`, `carton_paper`, `other_plastic`) with instant report submission toast.

2. **AI Inference & Clog Severity Estimation (`backend/taco_adapter.py`):**
   - Built on the open-source **TACO (Trash Annotations in Context)** benchmark (1,500 COCO images, 4,784 annotations across 60 categories).
   - Maps TACO categories into 5 curated municipal classes and calculates stormwater drain obstruction severity (1.0 to 5.0).

3. **Spatial Deduplication (`backend/clustering_engine.py`):**
   - Employs **Haversine DBSCAN** with $eps = 50\text{ m}$ and $min\_samples = 2$.
   - Consolidates scattered duplicate citizen pins into single municipal hotspots, avoiding redundant crew dispatch.

4. **Context Risk Engine (`backend/drain_risk_engine.py`):**
   - Computes an explainable $0–100$ Risk Score weighted by:
     $$\text{Risk Score} = 0.35(\text{Drain Proximity Score}) + 0.25(\text{Detection Severity}) + 0.20(\text{Confidence}) + 0.20(\text{Rainfall Forecast}) + \text{Recurrence Boost}$$
   - Evaluates direct Euclidean/Haversine distance in meters to nearest stormwater drain or river channel using Shapely on OpenStreetMap/GeoJSON drainage vectors.
   - Integrates live 24-hour precipitation forecasts via Open-Meteo API.

5. **Municipal Route Optimization (`backend/route_optimizer.py`):**
   - Computes optimal traveling salesperson route (TSP) for municipal sanitation trucks departing from Swargate Central Depot to high-priority hotspots and returning to depot.
   - Calculates leg distances, driving time, estimated service time, and polyline coordinates for map rendering.

6. **Operator Decision Dashboard (`dashboard/app.py`):**
   - **Left Panel:** Cleanup Queue ranked by 0–100 Risk Score with color tags (`CRITICAL` / `HIGH` / `MODERATE`).
   - **Center Panel:** Leaflet map with dark-matter tiles, showing urban waterways/drains (cyan lines), individual citizen report pins (yellow dots), consolidated DBSCAN buffer rings (red/amber circles), and the dispatched truck cleanup route (emerald green line).
   - **Right Drawer:** Hotspot detail showing merged report counts (e.g., *"3 reports merged via DBSCAN"*), exploded score breakdown (Proximity: 50m to drain, Rain: 16.5mm expected, Conf: 0.91), and a **"Verify & Dispatch Work Order"** action button.

---

## Repository Structure

```
PlasticWatch/
├── backend/
│   ├── api.py                    # FastAPI REST backend & mobile PWA delivery
│   ├── clustering_engine.py      # Haversine DBSCAN spatial clustering (eps=50m)
│   ├── drain_risk_engine.py      # 0-100 urban drain proximity & rainfall risk engine
│   ├── route_optimizer.py        # Municipal truck cleanup route optimizer (TSP)
│   ├── taco_adapter.py           # TACO dataset adapter & waste object detection
│   ├── plasticwatch_pipeline.py  # End-to-end urban civic pipeline orchestrator
│   └── test_urban_pipeline.py    # Standalone verification script & bridge demo
├── dashboard/
│   └── app.py                    # 3-panel Leaflet/Streamlit operator command center
├── mobile/
│   └── index.html                # Responsive HTML5/Tailwind citizen mobile reporting PWA
├── data/
│   ├── urban_drain_network.geojson # High-res Pune drainage & river network (Shapely/OSM)
│   ├── taco_class_mapping.json   # 60 TACO categories -> 5 urban waste classes
│   ├── taco_samples/             # Preloaded sample roadside waste images for testing
│   └── taco/data/annotations.json# Official TACO COCO dataset annotations
└── tests/
    ├── test_urban_pivot.py       # Unit & integration tests for clustering, risk, & routing
    ├── test_api.py               # Integration tests for FastAPI endpoints
    └── test_taco_adapter.py      # TACO dataset provenance tests
```

---

## Quick Start & Running Guide

### 1. Run the Standalone Pipeline Demo
Verify all 5 stages (Ingestion -> AI Vision -> DBSCAN Clustering -> Drain Risk -> Route Optimization) directly from the command line:

```bash
python3 backend/test_urban_pipeline.py
```

### 2. Launch the 3-Panel Municipal Operator Dashboard
Starts the interactive Streamlit command center on port 8501:

```bash
streamlit run dashboard/app.py
```
Open **`http://localhost:8501`** in your browser:
- **Left:** Inspect the priority cleanup queue (ranked 0–100).
- **Center:** Explore the interactive Leaflet map with drain channels, citizen pins, and DBSCAN halos.
- **Right:** Review the exploded risk breakdown and click **"🚨 Verify & Dispatch Work Order"** to render the truck route!
- **Tab 2 (Mobile Simulator):** Test citizen report submissions and see live re-clustering.

### 3. Launch the FastAPI Backend & Standalone Mobile Simulator
Starts the REST API and mobile client on port 8000:

```bash
uvicorn backend.api:app --reload --port 8000
```
- **Mobile Simulator PWA:** Open **`http://localhost:8000/mobile`** in your browser.
  - *Screen-recording tip:* Open Chrome DevTools (`Ctrl+Shift+I` / `Cmd+Option+I`), toggle **Device Toolbar** (`Ctrl+Shift+M`), select **iPhone 14** or **Pixel 7**, and record the native app experience!
- **Interactive OpenAPI Docs:** Open **`http://localhost:8000/docs`**.
- **Clustered Hotspots API:** `curl http://localhost:8000/api/hotspots`.

### 4. Run the Automated Test Suite

```bash
# Run urban pivot unit & integration tests
python3 -m unittest tests/test_urban_pivot.py

# Run FastAPI endpoint tests
python3 -m unittest tests/test_api.py

# Run all tests
python3 -m unittest discover tests
```

---

## Explainable Prioritisation Formula

Hotspots are ranked using an auditable, transparent Multi-Criteria Decision Analysis formula scaled from $0$ to $100$:

$$\text{Risk Score} = 0.35 \cdot S_{\text{drain}} + 0.25 \cdot S_{\text{severity}} + 0.20 \cdot S_{\text{confidence}} + 0.20 \cdot S_{\text{rainfall}} + B_{\text{recurrence}}$$

Where:
* $S_{\text{drain}}$: Proximity score based on distance to nearest mapped stormwater drain or river ($0–100$, where $\le 10\text{ m} \rightarrow 100$).
* $S_{\text{severity}}$: Waste accumulation and grate obstruction hazard rating ($1.0–5.0$ normalized to $0–100$).
* $S_{\text{confidence}}$: Computer vision detector confidence ($0.0–1.0$ normalized to $0–100$).
* $S_{\text{rainfall}}$: Open-Meteo 24h expected precipitation in mm ($0–100$, where $\ge 25\text{ mm} \rightarrow 100$).
* $B_{\text{recurrence}}$: Bonus points $(+3.5\text{ pts per duplicate report})$ reflecting repeat citizen complaints.

Priority Bands:
* **CRITICAL** (Red): Score $\ge 70$ (Immediate municipal dispatch to prevent drainage choke).
* **HIGH** (Amber): Score $\ge 50$ (Scheduled cleanup today).
* **MODERATE** (Sky Blue): Score $< 50$ (Routine maintenance queue).
