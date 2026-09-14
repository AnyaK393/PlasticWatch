# 🌊 AetherSea: Autonomous Marine Habitat Protection System

> Hackathon pivot: the active dashboard is **PlasticWatch (PS-08)**, an AI-GIS decision-support prototype for urban plastic-waste hotspots. The older AetherSea modules remain in the repository as research work.

<p align="center">
  <b>An AI-powered predictive marine conservation platform for detecting, tracking, and intercepting floating plastic debris before it reaches vulnerable ecosystems.</b>
</p>


## Overview

Marine plastic pollution is one of the biggest threats to coastal biodiversity. Current cleanup operations are largely reactive, discovering debris only after it reaches shorelines and sensitive habitats.

**AetherSea** introduces a predictive, autonomous approach by combining:

- 🛰️ Multispectral satellite observation
- 🌊 Ocean current hydrodynamics
- 🤖 Multi-agent AI reasoning
- 🗺️ Geospatial machine learning
- 🚢 Autonomous fleet route optimization

The platform detects floating plastic clusters, predicts their movement using ocean currents, evaluates ecological risk, and generates optimized interception strategies for autonomous cleanup vessels.

## Demo-first architecture

The current dashboard is intentionally **provider based**. It can run without external credentials in a transparent, deterministic **Simulation Replay** mode, then use the same workflow with real providers later.

```text
ReplayObservationProvider + ReplayCurrentProvider
                 │
                 ├── run_mission() ──► one mission JSON contract ──► Streamlit console
                 │
EarthEngineObservationProvider + CopernicusCurrentProvider
                 │
                 └── same contract and dashboard when access is configured
```

Simulation data is synthetic and visibly labelled as such. It is suitable for a reproducible demo, but is not presented as a real marine-debris detection.

### Run the demo

```bash
./venv/bin/streamlit run dashboard/app.py
```

Open the displayed local URL and leave **Simulation replay** selected. The console shows:

1. a map-first mission radar with candidate observations, drift paths, uncertainty circles, and verification routes;
2. a compact evidence view for each target; and
3. a field-verification queue before a cleanup operation is authorised.

### Enable real providers later

The dashboard never substitutes replay data when live mode is selected. Configure:

- one-time local Earth Engine login: `./venv/bin/earthengine authenticate`, then `./venv/bin/earthengine set_project YOUR_GOOGLE_CLOUD_PROJECT`;
- `EE_PROJECT=YOUR_GOOGLE_CLOUD_PROJECT` for Google Earth Engine / Sentinel-2 candidates;
- `CURRENT_GRID_PATH` pointing to a timestamped Copernicus or HYCOM surface-current NetCDF subset.

Then select **Live provider scan**. Provider interfaces, schemas, replay data, drift, risk, and safe failure behaviour live in `backend/mission_pipeline.py`.

For an unattended deployment, use an approved service account and set `EE_SERVICE_ACCOUNT` and `EE_KEY_FILE` instead of browser authentication. Keep credentials outside Git and never put them in the dashboard source.

## PlasticWatch (PS-08) quick start

```bash
./venv/bin/streamlit run dashboard/app.py
```

The active dashboard uses transparent synthetic reports to demonstrate the complete PS-08 workflow: AI confidence, duplicate-report merging, severity/recurrence/drain-risk prioritisation, field verification, and a cleanup-team queue.

### TACO training data

The official TACO toolkit and COCO annotations are available in `data/taco/`. TACO contains 1,500 images, 4,784 labelled objects, and 60 waste categories in the checked annotation release. Its images are downloaded separately by its official downloader because they are hosted externally:

```bash
cd data/taco
../../venv/bin/python download.py
```

Use TACO to train/evaluate a litter detector, then map its broad classes to `plastic_bottle`, `plastic_bag_or_wrapper`, `mixed_plastic_litter`, and `uncertain_or_non_plastic`. Retain the dataset attribution and CC BY 4.0 licence notice. Do not use an unverified citizen image to assign blame or automatically issue an enforcement action.

### Reproducible PlasticWatch model preparation

`data/taco_class_mapping.json` documents the deliberate 60-category TACO reduction used by this prototype. It keeps three actionable plastic classes plus `uncertain_or_non_plastic`, and explicitly excludes the ambiguous TACO label `Plastic glooves`. This is a project-specific training map, not a claim that TACO itself has only four classes.

First download the actual images with the official TACO downloader above. The checked-in annotation file alone is not trainable. Then audit and prepare a separate YOLO-format copy; the source TACO folder is never changed:

```bash
# Confirms the category map and reports whether images are available.
./venv/bin/python data/prepare_taco_yolo.py --dry-run

# Creates data/taco_training/{images,labels}/... and dataset.yaml.
./venv/bin/python data/prepare_taco_yolo.py
```

Training has not been performed in this repository: at the time of the audit there were no downloaded TACO images and the current Python 3.13 environment did not contain PyTorch or Ultralytics. After images are available, use a fresh Python 3.10–3.12 environment, install a compatible PyTorch build and Ultralytics, then run a lightweight pretrained detector for 30–50 epochs:

```bash
yolo detect train model=yolo11n.pt data=data/taco_training/dataset.yaml epochs=40 imgsz=640
yolo detect val model=runs/detect/train/weights/best.pt data=data/taco_training/dataset.yaml
```

Record precision, recall, mAP, class-wise errors and an image-level review set before connecting a trained model to live triage. No `best.pt`, metrics, or inference output should be represented as present until those steps complete. The dashboard exposes its actual readiness state rather than simulating a trained model.

### Operational data contract and limitations

For a real pilot, each submitted report should include a consented image reference, latitude/longitude accuracy, capture time, source type, detector class/confidence, and field-verification state. The current replay intentionally lacks timestamps and detector labels, so it does not fabricate trend or composition charts. In deployment, a report adapter feeds this same contract into duplicate merging, transparent water-risk prioritisation, human verification, and before/after evidence records. It is decision support only: do not use a citizen image or model score to identify responsibility, issue enforcement, or autonomously dispatch a cleanup.

---

# 🏗️ System Architecture

AetherSea follows a closed-loop intelligence pipeline:
Satellite Observation
        ↓
Floating Debris Detection (FDI)
        ↓
Spatial Clustering (DBSCAN)
        ↓
Ocean Drift Forecasting (HYCOM)
        ↓
Habitat Threat Assessment
        ↓
Autonomous Vessel Routing (A*)
        ↓
Multi-Agent Mission Generation



---

# 🛰️ Technology Stack

| Layer | Technology |
|------|------------|
| Satellite Data | ESA Sentinel-2 MSI |
| Ocean Data | HYCOM Global Ocean Model |
| Spectral Analysis | Floating Debris Index (FDI) |
| Machine Learning | DBSCAN Clustering |
| Geospatial Processing | GeoPandas, Shapely |
| Route Optimization | A* Search Algorithm |
| AI Agents | Gemini-powered Multi-Agent Architecture |
| Backend | Python |
| Dashboard | Streamlit + Interactive Maps |
| Visualization | Plotly, Leaflet |


---

# 🛰️ How We Built It


## 1. Multi-Spectral Floating Debris Detection

AetherSea uses Sentinel-2 multispectral imagery at 10m resolution to identify floating polymer signatures over seawater.

The Floating Debris Index (FDI) is calculated as:

$$
FDI = R_{NIR} -
[
R_{RED}
+
(R_{SWIR1}-R_{RED})
\times
\frac{\lambda_{NIR}-\lambda_{RED}}
{\lambda_{SWIR1}-\lambda_{RED}}
\times 10
]
$$


Where:

- $R_{RED}$ is Band 4 reflectance ($\lambda_{RED}=665 nm$)
- $R_{NIR}$ is Band 8 reflectance ($\lambda_{NIR}=842 nm$)
- $R_{SWIR1}$ is Band 11 reflectance ($\lambda_{SWIR1}=1610 nm$)


This allows identification of anomalous spectral signatures associated with floating debris.


---

# 2. Spatial Clustering & Convex Hull Delineation

Detected debris pixels are grouped using:

**Density-Based Spatial Clustering of Applications with Noise (DBSCAN)**


Parameters:
ε = 0.04°
MinPts = 4


Each detected debris region is represented using Convex Hull geometry:


$$
Hull(C_k)=
\{
\sum_{i=1}^{|C_k|}\alpha_i x_i :
\alpha_i \geq 0,
\sum_{i=1}^{|C_k|}\alpha_i = 1
\}
$$


This converts scattered debris detections into meaningful spatial clusters.


---

# 3. Lagrangian Hydrodynamic Drift Prediction

Floating debris movement is influenced by ocean currents.

AetherSea integrates velocity fields:

$(u,v)$

from the **Hybrid Coordinate Ocean Model (HYCOM)**.


Forward trajectory prediction is performed using:


$$
x(t+\Delta t)=
x(t)+
\int_t^{t+\Delta t}
u(x,\tau)d\tau
$$


$$
y(t+\Delta t)=
y(t)+
\int_t^{t+\Delta t}
v(x,\tau)d\tau
$$


The system forecasts possible debris movement over:

- 24-hour
- 48-hour
- 72-hour


time windows.


---

# 4. Habitat Threat Scoring (HTS)

Not all debris poses the same ecological risk.

AetherSea introduces a **Habitat Threat Score (HTS)** to prioritize debris approaching sensitive marine regions.


$$
HTS=
\overline{FDI}
\times
S_{habitat}
\times
(
\frac{R_{sanctuary}\times2.5}
{max(1.0,D_{proj})}
)
\times
\gamma_{convergence}
$$


The score considers:

- Floating debris intensity
- Habitat sensitivity
- Distance from protected ecosystems
- Current convergence patterns


This enables intelligent prioritization of cleanup operations.


---

# 5. Autonomous Fleet Route Optimization

To dispatch Autonomous Surface Vessels (ASVs), AetherSea implements a grid-based A* search algorithm.


The optimization function:

$$
f(n)=g(n)+h(n)
$$


The routing engine considers:

- Water-only navigation paths
- Coastal barriers
- Shallow regions
- Optimal interception distance


---

#  Multi-Agent AI Architecture

AetherSea uses specialized AI agents that collaborate to generate autonomous mission strategies.


## Sentinel Observation Agent

Responsibilities:

- Processes satellite observations
- Extracts Floating Debris Index information
- Identifies debris clusters


---

## Hydrodynamic Drift Agent

Responsibilities:

- Processes HYCOM current data
- Simulates debris movement
- Predicts future positions


---

## Sanctuary Risk Agent

Responsibilities:

- Evaluates habitat vulnerability
- Calculates Habitat Threat Scores
- Assigns priority levels


---

## Fleet Tactical Dispatch Agent

Responsibilities:

- Generates vessel routes
- Performs A* optimization
- Creates mission instructions


---

# AetherSea Mission OS Dashboard

The interactive dashboard provides:


## Sanctuary Tactical Radar

Features:

✅ Marine protected area monitoring  
✅ Debris cluster visualization  
✅ Predicted drift trajectories  
✅ Threat prioritization  


---

## Sentinel-2 FDI Analytics Studio

Features:

✅ Pixel-level reflectance analysis  
✅ Spectral signature comparison  
✅ DBSCAN cluster analysis  


---

##  HYCOM Hydrodynamic Engine

Features:

✅ Ocean velocity visualization  
✅ Current vector analysis  
✅ 72-hour drift forecasting  


---

##  Autonomous Multi-Agent War Room

Features:

✅ Agent communication  
✅ Automated mission synthesis  
✅ Structured JSON mission generation  


# Challenges Solved


## Spectral False Positives

Problem:

Ocean glare, clouds, and foam can resemble floating debris.


Solution:

- SWIR band filtering
- Signal-to-noise analysis
- DBSCAN noise removal


---

## Satellite Processing Latency

Problem:

Live raster processing caused delays.


Solution:

- Vectorized NumPy processing
- Optimized spatial operations
- Precomputed spatial structures


---

## Coastal Navigation Constraints

Problem:

Straight-line vessel paths crossed land regions.


Solution:

- Water-mask constrained A* routing
- Collision-free navigation planning


---

## Ecological Prioritization

Problem:

Raw debris coordinates lacked environmental context.


Solution:

Created the Habitat Threat Score (HTS) framework.


---

# Datasets & Attribution


## Sentinel-2 MSI

European Space Agency (ESA)

Used for:

- Multispectral satellite imagery
- Floating debris detection


---

## HYCOM

Hybrid Coordinate Ocean Model

Used for:

- Ocean current velocity fields
- Drift simulation


---

## Natural Earth

Used for:

- Coastal boundaries
- Marine geographic features


---

## Google Gemini

Used for:

- Multi-agent AI orchestration
- Mission generation


---

# Future Improvements


- Real-time Sentinel-2 API integration
- Deep learning-based debris segmentation
- Live autonomous vessel telemetry
- Integration with marine conservation organizations
- Field validation using real-world observations


---

# Vision

AetherSea aims to transform marine conservation from a reactive cleanup process into a predictive intelligence system — enabling early detection, smarter decisions, and autonomous protection of vulnerable ocean ecosystems.
