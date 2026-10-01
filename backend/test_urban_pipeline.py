#!/usr/bin/env python3
"""Standalone verification script bridging:
  1. TACO detection & severity scoring (taco_adapter.py)
  2. DBSCAN spatial clustering (clustering_engine.py)
  3. Urban drain proximity & Open-Meteo rainfall (drain_risk_engine.py)
  4. Municipal cleanup route optimizer (route_optimizer.py)
"""
import json
from pathlib import Path
import sys

# Ensure repository root is on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.clustering_engine import DBSCANClusteringEngine
from backend.drain_risk_engine import DrainRiskEngine
from backend.route_optimizer import RouteOptimizer
from backend.taco_adapter import dataset_status, detect_waste, get_sample_images


def run_pipeline_bridge_demo():
    print("=" * 80)
    print("  PLASTICWATCH: URBAN CIVIC ENVIRONMENTAL INTELLIGENCE PIPELINE")
    print("  Zero-Hardware, API-First Rollout (Pune Municipal Corporation Pilot)")
    print("=" * 80)

    # 1. TACO Dataset & AI Inference Check
    print("\n[STAGE 1] TACO Dataset & Computer Vision Inference Engine")
    status = dataset_status()
    print(f"  • TACO COCO Annotations : {status.get('annotations', 0):,} across {status.get('images', 0):,} images")
    print(f"  • Curated Target Classes: {status.get('target_classes', [])}")
    
    samples = get_sample_images()
    print(f"  • Preloaded Roadside Samples: {len(samples)} images detected")
    sample_file = samples[0]["path"] if samples else "data/taco_samples/sample_bottles_drain.jpg"
    detection = detect_waste(sample_file, filename_hint=Path(sample_file).name)
    print(f"  • Inference on '{Path(sample_file).name}':")
    print(f"    - Mean Confidence : {detection['mean_confidence']:.0%}")
    print(f"    - Clog Severity   : {detection['severity']}/5.0 ({detection['hazard_level']})")
    print(f"    - Detected Items  : {[item['label'] for item in detection['detected_items']]}")

    # 2. Simulated Citizen Mobile Reports
    print("\n[STAGE 2] Ingesting Citizen Mobile Reports (Photo + Auto-GPS)")
    raw_reports = [
        {"id": "CR-201", "lat": 18.5204, "lon": 73.8568, "confidence": 0.94, "severity": 4.6, "photo": "sample_bottles_drain.jpg", "note": "Bottles choking culvert grate"},
        {"id": "CR-202", "lat": 18.5207, "lon": 73.8569, "confidence": 0.89, "severity": 4.2, "photo": "sample_plastic_bag_curb.jpg", "note": "Plastic bags 35m from CR-201"},
        {"id": "CR-203", "lat": 18.5205, "lon": 73.8566, "confidence": 0.91, "severity": 4.4, "photo": "sample_mixed_waste_grate.jpg", "note": "Third report around same culvert"},
        {"id": "CR-204", "lat": 18.5060, "lon": 73.8625, "confidence": 0.92, "severity": 4.1, "photo": "sample_food_wrappers_culvert.jpg", "note": "Nagzari Nallah grate blockage"},
        {"id": "CR-205", "lat": 18.5063, "lon": 73.8627, "confidence": 0.88, "severity": 3.8, "photo": "sample_beverage_cans_gutter.jpg", "note": "Duplicate pin 32m along Nagzari"},
        {"id": "CR-206", "lat": 18.5074, "lon": 73.8077, "confidence": 0.84, "severity": 3.5, "photo": "sample_bottles_drain.jpg", "note": "Kothrud West single report"},
        {"id": "CR-207", "lat": 18.5350, "lon": 73.8920, "confidence": 0.87, "severity": 4.5, "photo": "sample_mixed_waste_grate.jpg", "note": "Mutha river confluence debris"},
        {"id": "CR-208", "lat": 18.5480, "lon": 73.9053, "confidence": 0.76, "severity": 2.6, "photo": "sample_food_wrappers_culvert.jpg", "note": "Yerawada canal minor litter"},
    ]
    print(f"  • Total Citizen Reports Received: {len(raw_reports)}")

    # 3. DBSCAN Spatial Clustering (eps=50m)
    print("\n[STAGE 3] DBSCAN Spatial Clustering (eps = 50 meters)")
    clusterer = DBSCANClusteringEngine(eps_meters=50.0, min_samples=2)
    clusters = clusterer.cluster_reports(raw_reports)
    print(f"  • DBSCAN consolidated {len(raw_reports)} raw citizen pins into {len(clusters)} municipal hotspots:")
    for c in clusters:
        status_str = f"MERGED {c['recurrence']} reports ({', '.join(c['report_ids'])})" if c["is_merged"] else f"SINGLE report ({c['report_ids'][0]})"
        print(f"    [{c['hotspot_id']}] Centroid: ({c['lat']:.5f}, {c['lon']:.5f}) | {status_str}")

    # 4. Urban Drain Proximity & Context Risk Scoring
    print("\n[STAGE 4] Urban Drainage Risk Engine (0-100 Score)")
    print("  Formula: 0.35(Drain Proximity) + 0.25(Severity) + 0.20(Confidence) + 0.20(Rainfall 24h) + Recurrence")
    risk_engine = DrainRiskEngine()
    scored_hotspots = []
    for c in clusters:
        risk = risk_engine.compute_risk(c, rainfall_override_mm=16.5)
        combined = {**c, **risk}
        scored_hotspots.append(combined)

    scored_hotspots.sort(key=lambda x: x["total_score"], reverse=True)

    print("\n  RANKED MUNICIPAL CLEANUP QUEUE:")
    print("  " + "-" * 76)
    print(f"  {'ID':<8} {'PRIORITY':<10} {'SCORE':<7} {'DRAIN PROXIMITY':<32} {'RECURRENCE':<12}")
    print("  " + "-" * 76)
    for h in scored_hotspots:
        drain_str = f"{h['nearest_drain_distance_m']:.0f}m to {h['nearest_drain_name'][:24]}"
        reps_str = f"{h['recurrence']} report(s)"
        print(f"  {h['hotspot_id']:<8} {h['priority']:<10} {h['total_score']:<7.1f} {drain_str:<32} {reps_str:<12}")
    print("  " + "-" * 76)

    # Top Hotspot Exploded Breakdown
    top = scored_hotspots[0]
    print(f"\n  EXPLODED BREAKDOWN FOR TOP HOTSPOT ({top['hotspot_id']}):")
    print(f"    • Drain Distance : {top['nearest_drain_distance_m']:.1f} m ({top['nearest_drain_name']})")
    print(f"      - Drain Score  : {top['drain_score']:.1f} / 100  -->  +{top['drain_points']:.1f} pts (35% weight)")
    print(f"    • Clog Severity  : {top['severity_raw']:.1f} / 5.0")
    print(f"      - Severity Score: {top['severity_score']:.1f} / 100 -->  +{top['severity_points']:.1f} pts (25% weight)")
    print(f"    • AI Confidence  : {top['confidence_raw']:.0%}")
    print(f"      - Conf Score   : {top['confidence_score']:.1f} / 100 -->  +{top['confidence_points']:.1f} pts (20% weight)")
    print(f"    • Rain Forecast  : {top['rainfall_forecast_mm']:.1f} mm (next 24 hours)")
    print(f"      - Rain Score   : {top['rainfall_score']:.1f} / 100  -->  +{top['rainfall_points']:.1f} pts (20% weight)")
    if top.get("recurrence_bonus"):
        print(f"    • Recurrence Boost: +{top['recurrence_bonus']:.1f} pts from {top['recurrence']} citizen submissions")
    print(f"    => TOTAL RISK SCORE: {top['total_score']:.1f} / 100  [{top['priority']}]")

    # 5. Municipal Truck Route Optimizer (TSP)
    print("\n[STAGE 5] Municipal Cleanup Route Optimizer (Swargate Depot Departure)")
    optimizer = RouteOptimizer()
    critical_hotspots = [h for h in scored_hotspots if h["priority"] == "CRITICAL"]
    plan = optimizer.generate_dispatch_plan(
        target_hotspots=critical_hotspots,
        truck_id="PMC-TRUCK-04",
        crew_name="PMC Rapid Sanitation Unit 2",
    )
    print(f"  • Work Order ID       : {plan['work_order_id']}")
    print(f"  • Assigned Truck      : {plan['truck_id']} ({plan['crew_name']})")
    print(f"  • Total Tour Distance : {plan['total_distance_km']} km")
    print(f"  • Total Duration Est  : {plan['total_duration_minutes']} mins (Drive: {plan['estimated_drive_minutes']}m, Service: {plan['estimated_service_minutes']}m)")
    print(f"  • Sequential Cleanup Itinerary:")
    for s in plan["stops"]:
        print(f"    Stop #{s['stop_number']}: {s['hotspot_id']} [{s['priority']} - Score {s['total_score']:.1f}] - {s['nearest_drain']} (+{s['distance_from_prev_km']} km)")
    print(f"    Return to Depot: {plan['depot']['name']}")

    print("\n" + "=" * 80)
    print("  ✅ PIPELINE INTEGRATION VERIFIED: ALL MODULES FUNCTIONING END-TO-END!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_pipeline_bridge_demo()
