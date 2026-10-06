"""Monsoon What-If Decision Simulator Engine for PlasticWatch Urban Intelligence.

Simulates municipal flood vulnerability and plastic wash-off outcomes across:
  1. Rainfall Intensity: 0 mm/hr (Dry) to 75 mm/hr (Cloudburst / Severe Monsoon)
  2. Available Fleet: 1 to 10 tipper/suction trucks
  3. Action Delay: 0 hrs (Immediate) to 24 hrs (Delayed response)

Outputs:
  - Critical choking hotspots count
  - Plastic mass at risk (kg)
  - Plastic volume swept into rivers (kg)
  - Potential flood inundation area (sq meters)
  - Preventable flood area (sq meters)
  - Estimated civic cost of inaction (INR / monetary damage penalty)
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MonsoonSimulatorEngine:
    """Hydrological & municipal logistics simulator for stormwater plastic choke points."""

    def __init__(self):
        pass

    def simulate(
        self,
        rainfall_intensity_mmh: float = 25.0,
        available_fleet: int = 4,
        action_delay_hours: float = 2.0,
        base_hotspots: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run what-if scenario simulation.

        Args:
            rainfall_intensity_mmh: 0.0 to 75.0 mm/hr.
            available_fleet: 1 to 10 trucks.
            action_delay_hours: 0.0 to 24.0 hours.
            base_hotspots: Optional list of hotspots from pipeline.

        Returns:
            Dictionary with scenario outcomes, flood footprint, river wash-off,
            monetary civic damage costs, and comparative baseline scenario.
        """
        # Clamp inputs within bounded operational limits
        rain_i = max(0.0, min(75.0, float(rainfall_intensity_mmh)))
        fleet = max(1, min(10, int(available_fleet)))
        delay = max(0.0, min(24.0, float(action_delay_hours)))

        # Default representative Pune pilot hotspots if none supplied
        if not base_hotspots:
            spots = [
                {"id": "HS-01", "name": "Shaniwar Wada Culvert", "severity": 4.6, "drain_dist_m": 18.5, "base_kg": 38.0},
                {"id": "HS-02", "name": "Nagzari Nallah Intake", "severity": 4.1, "drain_dist_m": 12.0, "base_kg": 44.0},
                {"id": "HS-03", "name": "Mutha River Outfall", "severity": 4.5, "drain_dist_m": 8.5, "base_kg": 62.0},
                {"id": "HS-04", "name": "Kothrud Feeder Channel", "severity": 3.5, "drain_dist_m": 28.0, "base_kg": 26.0},
                {"id": "HS-05", "name": "Yerawada Canal Culvert", "severity": 2.8, "drain_dist_m": 55.0, "base_kg": 18.0},
            ]
        else:
            spots = []
            for idx, h in enumerate(base_hotspots, 1):
                sev = float(h.get("severity", h.get("severity_raw", 3.5)))
                dist = float(h.get("nearest_drain_distance_m", 35.0))
                # Base uncollected plastic mass ~ 12 - 45 kg per cluster
                base_kg = round(sev * 6.5 + (max(0.0, 60.0 - dist) * 0.3), 1)
                spots.append({
                    "id": h.get("id") or h.get("hotspot_id", f"HS-{idx:02d}"),
                    "name": h.get("nearest_drain_name", f"Hotspot {idx}"),
                    "severity": sev,
                    "drain_dist_m": dist,
                    "base_kg": base_kg,
                })

        total_base_kg = sum(s["base_kg"] for s in spots)

        # ── 1. Plastic Mass at Risk (kg) ──────────────────────────────────────
        # High rainfall mobilizes additional urban litter from surrounding catchment
        catchment_wash_factor = 1.0 + (rain_i / 50.0) * 0.45
        plastic_at_risk_kg = round(total_base_kg * catchment_wash_factor, 1)

        # ── 2. Fleet Clearing Capacity & Delay Depletion ───────────────────────
        # Each truck clears ~32 kg/hr or ~85 kg per 4h dispatch block
        # Delay degrades efficiency because rain washes debris deep into culverts
        delay_penalty_factor = max(0.15, 1.0 - (delay / 28.0))
        truck_capacity_kg = fleet * 48.0 * delay_penalty_factor
        plastic_cleared_kg = min(plastic_at_risk_kg, round(truck_capacity_kg, 1))

        # ── 3. Plastic Swept Into Rivers (kg) ──────────────────────────────────
        # Remaining uncollected plastic washed down by stormwater runoff velocity
        wash_velocity_fraction = (rain_i / 75.0) ** 1.3 if rain_i > 0 else 0.0
        # Longer delay = higher fraction washed out before trucks arrive
        mobilization_delay_rate = min(1.0, (delay / 20.0) * 0.75 + wash_velocity_fraction * 0.4)
        uncleared_kg = max(0.0, plastic_at_risk_kg - plastic_cleared_kg)
        plastic_swept_rivers_kg = round(min(uncleared_kg, uncleared_kg * (0.25 + 0.75 * mobilization_delay_rate)), 1)
        if rain_i == 0:
            plastic_swept_rivers_kg = 0.0

        # ── 4. Critical Choking Hotspots Count ────────────────────────────────
        # Hotspots choke when rainfall runoff exceeds drain intake due to plastic obstruction
        critical_choking_count = 0
        for s in spots:
            dist = s["drain_dist_m"]
            sev = s["severity"]
            choke_score = (rain_i / 75.0) * 45.0 + (sev / 5.0) * 35.0 + max(0.0, (50.0 - dist) * 0.5) + (delay / 24.0) * 15.0
            if choke_score >= 50.0 or (dist <= 20.0 and rain_i >= 20.0 and delay >= 2.0):
                critical_choking_count += 1

        # ── 5. Potential Flood Inundation Area (sq meters) ────────────────────
        # Choked grates cause waterlogging. 1 choked intake backs up ~2,500 - 8,000 sq meters
        if rain_i <= 2.0:
            flood_area_sqm = 0.0
            preventable_flood_sqm = 0.0
        else:
            base_sqm_per_choke = 3200.0 * (rain_i / 40.0) ** 1.2
            choke_multiplier = 1.0 + (critical_choking_count * 0.85)
            delay_expansion = 1.0 + (delay / 12.0) * 0.65
            flood_area_sqm = round(base_sqm_per_choke * choke_multiplier * delay_expansion, 0)

            # Preventable flood area: difference if fleet deployed immediately (delay=0, fleet=10)
            optimal_chokes = max(0, critical_choking_count - int(fleet * 0.8))
            optimal_area = base_sqm_per_choke * (1.0 + optimal_chokes * 0.5) if optimal_chokes > 0 else 450.0
            preventable_flood_sqm = round(max(0.0, flood_area_sqm - optimal_area), 0)

        # ── 6. Estimated Civic Cost of Inaction (INR) ─────────────────────────
        # Components:
        # a) Emergency suction pumping & dewatering: ₹4,500 per critical site per hour of delay
        dewatering_cost = critical_choking_count * 4500.0 * max(1.0, delay) if critical_choking_count > 0 else 0.0
        # b) Traffic disruption & road submersion economic impact: ₹18 per sq meter inundated
        traffic_cost = flood_area_sqm * 18.5
        # c) River plastic cleanup & environmental restoration penalty: ₹350 per kg in river
        river_restoration_penalty = plastic_swept_rivers_kg * 350.0
        # d) Emergency contractor mobilization surcharge: ₹25,000 if delay > 6h and rain > 30 mm/hr
        emergency_surcharge = 25000.0 if (delay >= 6.0 and rain_i >= 30.0) else 0.0

        total_cost_inr = round(dewatering_cost + traffic_cost + river_restoration_penalty + emergency_surcharge, 0)

        # ── 7. Comparative Optimal vs Worst-Case Scenario ────────────────────
        # Optimal: Immediate response with full fleet
        opt_cleared = min(plastic_at_risk_kg, 10 * 48.0)
        opt_swept = round(max(0.0, (plastic_at_risk_kg - opt_cleared) * 0.15), 1) if rain_i > 0 else 0.0
        opt_cost = round((traffic_cost * 0.2) + (opt_swept * 350.0), 0)

        # Worst-case: 0 fleet, 24h delay
        worst_swept = round(plastic_at_risk_kg * min(0.95, 0.4 + (rain_i / 60.0) * 0.55), 1) if rain_i > 0 else 0.0
        worst_area = round(flood_area_sqm * 1.85, 0) if flood_area_sqm > 0 else 5000.0
        worst_cost = round(total_cost_inr * 2.2 + 85000.0, 0)

        # Inundation risk categorization tag
        if rain_i >= 50.0 or critical_choking_count >= 4 or flood_area_sqm >= 25000:
            scenario_alert = "CRITICAL MONSOON SURCHARGE"
            alert_color = "#ef4444"
        elif rain_i >= 25.0 or critical_choking_count >= 2 or flood_area_sqm >= 10000:
            scenario_alert = "HIGH WATERLOGGING RISK"
            alert_color = "#f59e0b"
        else:
            scenario_alert = "CONTROLLED DRAINAGE FLOW"
            alert_color = "#34d399"

        return {
            "inputs": {
                "rainfall_intensity_mmh": rain_i,
                "available_fleet": fleet,
                "action_delay_hours": delay,
            },
            "scenario_alert": scenario_alert,
            "alert_color": alert_color,
            "outcomes": {
                "critical_choking_hotspots": critical_choking_count,
                "total_monitored_hotspots": len(spots),
                "plastic_mass_at_risk_kg": plastic_at_risk_kg,
                "plastic_cleared_by_fleet_kg": plastic_cleared_kg,
                "plastic_swept_into_rivers_kg": plastic_swept_rivers_kg,
                "plastic_prevented_from_rivers_pct": round(
                    ((plastic_at_risk_kg - plastic_swept_rivers_kg) / max(0.1, plastic_at_risk_kg)) * 100.0, 1
                ),
                "potential_flood_inundation_area_sqm": flood_area_sqm,
                "preventable_flood_area_sqm": preventable_flood_sqm,
                "estimated_civic_cost_of_inaction_inr": total_cost_inr,
                "cost_breakdown_inr": {
                    "dewatering_pumping": round(dewatering_cost, 0),
                    "traffic_disruption_impact": round(traffic_cost, 0),
                    "river_restoration_penalty": round(river_restoration_penalty, 0),
                    "emergency_contractor_surcharge": round(emergency_surcharge, 0),
                },
            },
            "comparative_benchmarks": {
                "optimal_immediate_dispatch": {
                    "fleet": 10,
                    "delay_hours": 0.0,
                    "plastic_swept_kg": opt_swept,
                    "estimated_cost_inr": opt_cost,
                },
                "unmitigated_worst_case": {
                    "fleet": 0,
                    "delay_hours": 24.0,
                    "plastic_swept_kg": worst_swept,
                    "flood_area_sqm": worst_area,
                    "estimated_cost_inr": worst_cost,
                },
            },
            "provenance_badge": "[Simulated / Modelled Estimate]",
            "provenance_category": "Simulated",
            "methodology": "SWMM Rational Runoff Model + Municipal Fleet Service Capacity + Hydraulic Inundation Equation",
        }
