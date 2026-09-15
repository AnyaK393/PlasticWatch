
# PlasticWatch: AI-GIS Plastic Waste Intelligence Command Center

PlasticWatch is an explainable AI-GIS decision-support system that converts citizen image submissions and street-level imagery into auditable, prioritized cleanup queues. The platform prevents reactive municipal operations by deduplicating incident reports, detecting plastic debris presence, computing stormwater drain exposure, and enforcing a human-in-the-loop verification protocol.


---

## Key Capabilities

* **Geotagged Report Ingestion:** Ingests citizen smartphone photos, civic chatbot submissions, or street camera feeds along with EXIF coordinate stamps and timestamps.


* **CV-Based Detection:** Evaluates waste presence and confidence scores using a 4-class operational taxonomy mapped from the TACO benchmark.


* **Spatio-Temporal Deduplication:** Aggregates redundant complaints of the same dumpsite into consolidated hotspots using density clustering.


* **Hydrological Proximity Scoring:** Assesses proximity to OpenStreetMap (OSM) stormwater drains and riverbanks to mitigate water contamination hazards.


* **Explainable Prioritisation (0–100):** Ranks hotspots through transparent Multi-Criteria Decision Analysis (MCDA) rather than opaque black-box scoring.


* **Closed-Loop Audit Trail:** Directs ward inspectors to confirm material composition and logs before/after photo evidence post-cleanup.



---

## System Architecture

<img width="1820" height="1103" alt="image" src="https://github.com/user-attachments/assets/328ccd29-dd15-4189-8a1e-2348384808d4" />



---

## Computer Vision Pipeline & TACO Taxonomy

The vision system adapts the open-source **TACO (Trash Annotations in Context)** dataset—containing 1,500 images and 4,784 annotations across 60 categories—into 4 operational classes suited for municipal field cleanup:

| Operational Class | Source TACO Categories Included |
| --- | --- |
| **Plastic Bottle** | Clear plastic bottles, beverage bottles, plastic jugs |
| **Bag / Wrapper** | Food packaging, snack wrappers, grocery film bags, sachets |
| **Mixed Plastic** | Bottle caps, broken crates, synthetic fragments, disposable cutlery |
| **Uncertain / Non-Plastic** | Organic debris, gravel, glare reflections, algae, non-actionable litter |

---


```
[Raw Photo] ──▶ [Resize & Normalize] ──▶ [Inference Engine] ──▶ [Filter via NMS]
                                                                        │
    ┌───────────────────────────────────────────────────────────────────┘
    ▼
[Output Array]:
  • class_label: "Plastic Bottle"[cite: 1]
  • confidence: 0.85 (85%)[cite: 1]
  • bbox: [x_min, y_min, x_max, y_max]
  • severity_index: 0.80 (Area / Screen Ratio)[cite: 1]

```

---

## Explainable Prioritisation Formula

Hotspots are ranked using a linear Multi-Criteria Decision Analysis (MCDA) model, producing a composite index between $0$ and $100$:

$$\text{Priority Score} = w_1 \cdot C_{\text{AI}} + w_2 \cdot S_{\text{waste}} + w_3 \cdot R_{\text{count}} + w_4 \cdot D_{\text{risk}}$$

Where:

* $C_{\text{AI}}$: Computer vision detection confidence ($0.0 - 1.0$).


* $S_{\text{waste}}$: Severity index based on bounding-box accumulation density ($0.0 - 1.0$).


* $R_{\text{count}}$: Cluster recurrence score based on merged duplicate submissions ($0.0 - 1.0$).


* $D_{\text{risk}}$: Drain proximity factor derived from distance to the nearest OpenStreetMap drainage/waterway vector ($0.0 - 1.0$, where distance $\le 35\text{ m} \rightarrow 1.0$).



### Pilot Benchmark (Pune Municipal Pilot Dataset)

| Hotspot ID | Reports Merged | AI Conf. | Drain Distance | Priority Score | Action Status |
| --- | --- | --- | --- | --- | --- |
| **PH-01** | 3 | 85% | 35 m | **99.1 / 100** | **CRITICAL / VERIFY NOW** |
| **PH-02** | 2 | ~80% | 20 m | **91.8 / 100** | **CRITICAL / VERIFY NOW** |
| **PH-03** | 1 | ~60% | 210 m | **46.5 / 100** | **REVIEW** |

---

## Verification Lifecycle & Safeguards

<img width="2027" height="978" alt="image" src="https://github.com/user-attachments/assets/aafd4a60-2f5a-4a14-a379-8de07f975222" />


* **Non-Enforcement Directive:** Unverified citizen imagery is handled strictly as advisory evidence, not as actionable legal or penal ground truth.


* **Human-in-the-Loop:** Field inspectors validate material classifications on-site before municipal resources are dispatched.


* **Audit Trail:** Transitions follow an explicit state machine: `Reported` $\rightarrow$ `Verified / Rejected` $\rightarrow$ `Cleaned` $\rightarrow$ `Closed`.



---

## Technology Stack

* **Frontend:** React.js / Next.js, Tailwind CSS, Lucide Icons, Leaflet.js / Mapbox GL
* **Backend:** Python (FastAPI), Uvicorn, Pydantic
* **Geospatial & Spatial DB:** PostGIS, PostgreSQL, GeoPandas, Shapely, Scikit-learn (DBSCAN)
* **Computer Vision:** PyTorch, Ultralytics YOLOv8 / Faster R-CNN, OpenCV
* **External GIS Baseline:** OpenStreetMap (OSM) Overpass API (Drainage & Watercourse LineStrings)

---

## Getting Started

### 1. Prerequisites

* Python 3.10+
* PostgreSQL 15+ with PostGIS extension enabled
* Node.js 18+

### 2. Backend Setup

```bash
# Clone the repository
git clone [https://github.com/AnyaK393/PlasticWatch.git](https://github.com/AnyaK393/PlasticWatch.git)
cd PlasticWatch/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run migrations and start API
uvicorn app.main:app --reload --port 8000

```

### 3. Frontend Setup

```bash
cd ../frontend

# Install dependencies
npm install

# Start development server
npm run dev

```

---

## Project Status & Limitations

* **Current State:** Functional decision-support prototype evaluating synthetic and transparent pilot replay data.


* **Inference Pipeline:** TACO dataset structured and annotated into a 4-class operational taxonomy for detector training.


* **System Boundaries:** PlasticWatch is a municipal decision-support tool; it does not issue automated administrative fines, dispatch autonomous vehicles, or assign personal citizen liability.

