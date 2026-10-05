# PlasticWatch ♻️
### AI-Powered Urban Environmental Intelligence for Plastic Waste & Stormwater Risk

PlasticWatch is an AI-assisted civic environmental intelligence platform designed to help municipalities detect, prioritize, prevent, and verify plastic waste accumulation around urban stormwater infrastructure.

The system converts citizen-reported waste observations into spatially consolidated hotspots, evaluates their drainage and rainfall-related risk, recommends municipal intervention, verifies cleanup using before/after AI analysis, identifies recurring root causes, forecasts future hotspots, and supports monsoon what-if planning.

---

## Problem Statement

Urban plastic waste frequently accumulates around stormwater drains, culverts, nallahs, river outfalls, and other drainage corridors.

When plastic waste blocks or obstructs these locations, it can:

- Reduce stormwater flow
- Increase localized flooding risk
- Cause plastic to enter rivers and waterways
- Create recurring sanitation problems
- Increase municipal cleanup workload
- Make it difficult for authorities to prioritize the most critical locations

Traditional citizen reporting often produces individual, disconnected complaints.

For example, multiple citizens may report plastic waste within a small geographic area, but a municipality may still have to treat them as separate incidents.

PlasticWatch addresses this problem by spatially consolidating nearby reports into operational hotspots and combining environmental and operational signals to determine which locations require attention first.

---

## Proposed Solution

PlasticWatch provides an end-to-end workflow:

```text
Citizen Report
      ↓
AI Waste Detection
      ↓
Spatial Hotspot Clustering
      ↓
Drain & Rainfall Risk Assessment
      ↓
Municipal Priority Ranking
      ↓
Cleanup Dispatch & Route Planning
      ↓
AI Before/After Cleanup Verification
      ↓
Recurrence & Root-Cause Intelligence
      ↓
24–72h Predictive Hotspot Forecasting
      ↓
Monsoon What-If Simulation
```

The platform also includes an evidence/provenance layer that distinguishes between:

- **Observed** — directly captured evidence
- **Derived** — values calculated from observed data
- **Inferred** — AI/model interpretations
- **Simulated** — what-if scenario outputs

This makes the system more transparent and suitable for responsible AI-assisted decision support.

---

# Features

## 1. 🏛️ Municipal Command Center

A centralized operational dashboard for municipal teams.

It provides:

- Cleanup priority queue
- Interactive city/drainage map
- Citizen report locations
- DBSCAN-consolidated hotspots
- Hotspot severity and confidence
- Drain proximity information
- Rainfall-aware risk scoring
- Work-order generation
- Truck assignment
- Cleanup route planning

The Command Center is designed to answer:

> **Which hotspot should the municipality act on first?**

---

## 2. 🔮 Predictive Hotspot Forecasting — 24–72 Hours

PlasticWatch extends beyond reactive reporting by estimating which locations are likely to experience increased risk in the next 24–72 hours.

The forecasting layer considers factors such as:

- Rainfall forecast
- Current accumulation
- Drain proximity
- Historical recurrence
- Time since cleanup
- Waste type
- Environmental conditions

It provides:

- Predicted escalation risk
- Expected escalation window
- Predicted plastic accumulation
- Potential flood-risk area
- Contributing factors
- Preventive intervention recommendation

**Evidence type:** `Predicted / Modelled`

> Note: The current prototype uses a deterministic spatio-temporal heuristic rather than a calibrated production probability model.

---

## 3. 📸 AI Before/After Cleanup Verification

PlasticWatch closes the cleanup loop using AI-assisted before/after image comparison.

The system can compare:

```text
Before Cleanup
      ↓
Municipal Cleanup
      ↓
After Cleanup
      ↓
AI Comparison
```

It provides:

- Before/after waste item counts
- Waste reduction percentage
- Severity reduction
- Cleanup effectiveness
- AI confidence
- Verification recommendation
- Estimated diverted plastic
- Audit notes

**Evidence type:** `AI Inference / Modelled Estimate`

The result is intended as AI-assisted verification and decision support. Operator review is required before treating it as official municipal verification.

---

## 4. 🔁 Recurrence & Root-Cause Intelligence

Some locations repeatedly become waste hotspots even after cleanup.

PlasticWatch analyzes recurrence patterns to move from:

> **Repeated cleanup**

to:

> **Root-cause intervention**

The system can provide:

- Historical report count
- Cleanup history
- Average recurrence interval
- Recurrence probability
- Persistence category
- Dominant waste types
- Root-cause hypothesis
- Confidence in the inferred cause
- Recommended systemic intervention

Example root causes include:

- Market/vendor packaging waste
- Drainage or grate defects
- Construction-material runoff
- Waste accumulation caused by runoff pathways

**Evidence type:** `Inferred / Prototype Historical Dossier`

The current historical profiles are prototype knowledge-base data and should not be interpreted as verified municipal statistics.

---

## 5. ☔ Monsoon What-If Decision Simulator

PlasticWatch allows municipal operators to test different operational scenarios before or during heavy rainfall.

Inputs include:

- Rainfall intensity
- Number of available trucks
- Response delay

The simulator estimates:

- Critical hotspots
- High-risk hotspots
- Plastic at risk
- Plastic potentially swept toward waterways
- Potential inundation area
- Preventable flood area
- Cost of delayed action
- Fleet capacity
- Operational recommendations

Example question:

> **What happens if rainfall reaches 50 mm and cleanup response is delayed by 18 hours with only two trucks available?**

**Evidence type:** `Simulated / Modelled Estimate`

These values represent scenario outputs and are not measurements of an actual flood event.

---

## 6. 🔎 Evidence & Provenance Layer

PlasticWatch explicitly distinguishes the source and certainty of important information.

### Observed

Directly captured or ingested evidence.

Examples:

- Citizen GPS
- Citizen photographs
- Report timestamps
- Citizen report IDs
- Drain network data

### Derived

Calculated from observed information.

Examples:

- DBSCAN clusters
- Report density
- Drain distance
- Risk score
- Priority ranking

### Inferred

AI/model-generated interpretation.

Examples:

- Waste detection
- Cleanup effectiveness
- Root-cause hypotheses
- Predicted escalation risk
- Estimated diverted plastic

### Simulated

What-if scenario outputs.

Examples:

- Potential inundation
- Plastic at risk
- Plastic swept toward rivers
- Cost of inaction

This provenance layer helps prevent model-generated values from being presented as observed facts.

---

# Technologies / Tech Stack Used

## Dashboard

- Python
- Streamlit
- Folium
- Streamlit-Folium
- Altair
- Pandas
- HTML/CSS

## AI / Data Intelligence

- AI-based waste detection
- TACO dataset / waste classification
- DBSCAN spatial clustering
- Haversine geographic distance
- Shapely
- Heuristic predictive modelling
- Recurrence analysis
- Scenario simulation

## Environmental / Geospatial Data

- GeoJSON
- Citizen GPS reports
- Urban drainage network
- Open-Meteo rainfall data

## Backend

- Python
- Modular intelligence engines
- Centralized PlasticWatch pipeline
- Route optimization

---

# Installation & Setup Instructions

## 1. Clone the repository

```bash
git clone <repository-url>
cd PlasticWatch
```

## 2. Create a virtual environment

```bash
python3 -m venv venv
```

## 3. Activate the virtual environment

### macOS / Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 5. Verify the project structure

Make sure the following important directories/files are present:

```text
backend/
dashboard/
data/
mobile/
agents/
tests/
requirements.txt
README.md
```

---

# How to Run the Project

Activate the virtual environment:

```bash
source venv/bin/activate
```

Run the Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

The dashboard will open in the browser.

---

# Project Structure

```text
PlasticWatch/
│
├── backend/
│   ├── plasticwatch_pipeline.py
│   ├── clustering_engine.py
│   ├── drain_risk_engine.py
│   ├── predictive_engine.py
│   ├── verification_engine.py
│   ├── recurrence_engine.py
│   ├── simulator_engine.py
│   ├── route_optimizer.py
│   └── taco_adapter.py
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── urban_drain_network.geojson
│   ├── citizen_reports.json
│   └── taco_samples/
│
├── mobile/
│
├── agents/
│
├── tests/
│
├── output/
│
├── requirements.txt
└── README.md
```

---

# System Workflow

```text
1. Citizen submits a geotagged report
                 ↓
2. AI analyzes the submitted image
                 ↓
3. Nearby reports are consolidated using DBSCAN
                 ↓
4. Nearest drainage infrastructure is identified
                 ↓
5. Rainfall and environmental risk are evaluated
                 ↓
6. Hotspots are ranked by operational priority
                 ↓
7. Municipal cleanup work order is generated
                 ↓
8. Route and truck assignment are recommended
                 ↓
9. Before/after images are compared using AI
                 ↓
10. Recurrence patterns are analyzed
                 ↓
11. Future hotspot escalation is forecast
                 ↓
12. Monsoon scenarios can be simulated
```

---

# Data & Evidence

PlasticWatch currently uses a combination of:

### Citizen Reports

Stored/seeded through the project data pipeline.

Each report can contain:

- Location
- Timestamp
- Photograph
- AI confidence
- Severity
- Detected waste classes
- Verification status
- Citizen notes

### Urban Drain Network

```text
data/urban_drain_network.geojson
```

Used for:

- Drain proximity
- Map visualization
- Risk prioritization

### TACO Samples

```text
data/taco_samples/
```

Used for AI waste detection demonstrations and cleanup verification.

### Rainfall

Rainfall information can be obtained through Open-Meteo precipitation data, with fallback values available for demonstration reliability.

---

# Risk Scoring

PlasticWatch combines multiple signals into a municipal prioritization score.

Current weighting:

| Factor | Weight |
|---|---:|
| Drain Proximity | 35% |
| Waste Severity | 25% |
| AI Detection Confidence | 20% |
| 24h Rainfall | 20% |

A recurrence-related bonus can also influence the final prioritization.

The resulting score is a **derived decision-support metric**, not an observed measurement.

---

# Screenshots / Demo Images

Screenshots and demonstration images can be added here as the project presentation is finalized.

Recommended screenshots:

1. Municipal Command Center
2. Interactive hotspot map
3. Citizen mobile reporting interface
4. AI waste detection
5. Predictive hotspot forecast
6. Before/after cleanup verification
7. Recurrence/root-cause analysis
8. Monsoon what-if simulator
9. TACO model readiness

Example:

```markdown
![PlasticWatch Command Center](path/to/screenshot.png)
```

---

# Team Members

| Name | Role |
|---|---|
| Ananya Kulkarni | Project Lead / AI & Full-Stack Development |
| Team Member 2 | Add role |
| Team Member 3 | Add role |
| Team Member 4 | Add role |

---

# Future Scope / Enhancements

Future versions of PlasticWatch can include:

- Real municipal complaint-system integration
- Persistent hotspot IDs
- Real historical municipal cleanup records
- Larger geospatial datasets
- Trained predictive forecasting models
- Calibrated risk probabilities
- Real-time IoT drain sensors
- Water-level sensors
- CCTV integration
- Satellite/drone imagery
- Automated municipal work-order APIs
- Worker mobile application
- Real measured waste weights
- Long-term recurrence learning
- Advanced flood simulation
- Multi-city deployment
- Municipal analytics and reporting
- Model monitoring and explainability
- Human-in-the-loop verification workflows

---

# Current Prototype Limitations

PlasticWatch is a prototype demonstrating the architecture and intelligence workflow.

Some outputs are modelled or simulated rather than real-world measurements.

### Predictive Forecasting

The current forecasting engine is a deterministic heuristic prototype and is not a calibrated production forecasting model.

### Cleanup Verification

AI before/after verification supports operator decision-making but does not replace physical inspection or official municipal sign-off.

### Recurrence Intelligence

Current historical profiles are prototype knowledge-base information rather than verified municipal records.

### Monsoon Simulation

Flood-area, plastic-at-risk, and cost-of-inaction values are scenario-based model estimates.

### Impact Metrics

Some prototype impact values may use synthetic demonstration assumptions and should not be interpreted as measured field outcomes.

---

# Responsible AI & Provenance

PlasticWatch follows an evidence-aware approach.

The system distinguishes:

```text
OBSERVED
↓
DERIVED
↓
INFERRED
↓
SIMULATED
```

This distinction is important because AI systems should not present predictions, hypotheses, or simulations as if they were direct observations.

The platform is therefore designed as:

> **AI-assisted municipal decision support, with human operators remaining responsible for final operational decisions.**

---

# Project Objective

The ultimate objective of PlasticWatch is to move municipal plastic-waste management from a reactive complaint-and-cleanup workflow toward a:

**Detect → Prioritize → Prevent → Act → Verify → Learn → Forecast → Simulate**

intelligence loop.

---

# Conclusion

PlasticWatch combines civic reporting, computer vision, geospatial intelligence, drainage-risk assessment, predictive analytics, operational routing, cleanup verification, recurrence analysis, and scenario simulation into a single environmental intelligence platform.

Rather than only detecting where plastic waste exists, PlasticWatch aims to help municipalities understand:

> **Where the problem is, why it matters, what will happen next, what action should be taken, whether that action worked, why the problem keeps returning, and what could happen if response is delayed.**
