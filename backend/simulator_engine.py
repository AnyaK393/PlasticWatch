"""Monsoon What-If Decision Simulator Engine for PlasticWatch Urban Intelligence.

Simulates municipal flood vulnerability and plastic wash-off outcomes across:
  1. Rainfall Intensity: 0 mm/hr (Dry) to 75 mm/hr (Cloudburst / Severe Monsoon)
  2. Available Fleet: 1 to 10 tipper/suction trucks
  3. Action Delay: 0 hrs (Immediate) to 24 hrs (Delayed response)
  4. Wind Drift: 0 to 50 km/h (pushes surface litter into gutters)
  5. Drain Siltation: 0% to 100% culvert capacity lost to silt

Model (per hotspot inlet, all parameters listed in ASSUMPTIONS):
  - Runoff reaching the inlet: rational method Q = C * i * A
  - Inlet capacity: design capacity x (1 - plastic blockage) x (1 - 0.8 * siltation)
  - Flooding: runoff above capacity ponds for the storm duration -> volume / ponding depth
  - Fleet: trucks clear sites in priority order; a site cleared before the storm peak
    loses its blockage, a site cleared mid-storm is only partly protected
  - River plastic: uncleared plastic x flush factor (rain + wind drift, amplified by silt)

Outputs:
  - Critical choking hotspots count
  - Plastic mass at risk (kg)
  - Plastic volume swept into rivers (kg)
  - Potential flood inundation area (sq meters)
  - Preventable flood area (sq meters)
  - Estimated civic cost of inaction (INR)
  - Explainability: driver narrative + one-lever-at-a-time sensitivity ranking
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Model assumptions (shown to operators so every number is traceable) ──────
ASSUMPTIONS: Dict[str, Dict[str, Any]] = {
    "catchment_area_m2": {"value": 12000.0, "unit": "m² per inlet", "basis": "≈1.2 ha paved road catchment draining to one street inlet"},
    "runoff_coefficient": {"value": 0.85, "unit": "–", "basis": "Rational-method C for asphalt/concrete urban surfaces (0.7–0.95)"},
    "inlet_design_capacity_lps": {"value": 90.0, "unit": "L/s", "basis": "Clear double-grate inlet with connected drain"},
    "max_plastic_blockage": {"value": 0.85, "unit": "fraction", "basis": "Grate blinding by film/bottles rarely exceeds ~85%"},
    "kg_for_full_blockage": {"value": 70.0, "unit": "kg", "basis": "Plastic mass that blinds ~85% of a grate"},
    "silt_capacity_loss": {"value": 0.8, "unit": "per unit silt", "basis": "Silt shrinks conveyance; 100% silt leaves ~20% capacity"},
    "storm_duration_h": {"value": 2.0, "unit": "h", "basis": "Typical Pune convective monsoon burst (1–3 h)"},
    "storm_lead_time_h": {"value": 4.0, "unit": "h", "basis": "Alert-to-peak time from nowcast/forecast"},
    "site_service_time_h": {"value": 1.1, "unit": "h per site", "basis": "≈20 min travel + 45 min clearance per truck stop"},
    "ponding_depth_m": {"value": 0.10, "unit": "m", "basis": "Average waterlogging depth on carriageway"},
    "dewatering_inr_per_site_h": {"value": 2500.0, "unit": "₹/site/h", "basis": "Diesel pump hire + operator"},
    "drain_down_h": {"value": 2.0, "unit": "h", "basis": "Pumping after the storm ends"},
    "disruption_inr_per_m2": {"value": 22.0, "unit": "₹/m² flooded", "basis": "Traffic delay + shop closure per event"},
    "river_recovery_inr_per_kg": {"value": 60.0, "unit": "₹/kg", "basis": "Boom/skimmer recovery, transport and disposal"},
    "emergency_surcharge_inr": {"value": 18000.0, "unit": "₹", "basis": "Night/overtime crew premium when delay ≥ 6 h in heavy rain"},
}


def _a(key: str) -> float:
    return float(ASSUMPTIONS[key]["value"])


# ── Open-channel flow velocity (Manning) ─────────────────────────────────────
# Rectangular-section approximation per channel class. Estimated typical Pune
# geometry, not surveyed: width b (m), bed slope S, Manning n, bank-full depth (m),
# contributing catchment (ha), runoff coefficient, dry-weather base flow (m³/s).
CHANNEL_HYDRAULICS: Dict[str, Dict[str, Any]] = {
    "river": {"width_m": 60.0, "slope": 0.0008, "manning_n": 0.035, "max_depth_m": 6.0, "catchment_ha": 2500.0,
              "runoff_c": 0.6, "base_flow_m3s": 5.0, "debris_transport": 0.55,
              "basis": "Mutha/Mula urban reach, natural bed with vegetation"},
    "nallah": {"width_m": 8.0, "slope": 0.003, "manning_n": 0.030, "max_depth_m": 2.5, "catchment_ha": 150.0,
               "runoff_c": 0.75, "base_flow_m3s": 0.25, "debris_transport": 0.45,
               "basis": "Open urban nallah, partly lined, bank vegetation snags debris"},
    "lined_drain": {"width_m": 2.5, "slope": 0.004, "manning_n": 0.018, "max_depth_m": 1.2, "catchment_ha": 25.0,
                    "runoff_c": 0.85, "base_flow_m3s": 0.03, "debris_transport": 0.70,
                    "basis": "Concrete-lined open storm drain"},
    "canal": {"width_m": 12.0, "slope": 0.001, "manning_n": 0.025, "max_depth_m": 2.0, "catchment_ha": 80.0,
              "runoff_c": 0.7, "base_flow_m3s": 0.5, "debris_transport": 0.50,
              "basis": "Earthen/lined spillway canal"},
}
SURFACE_TO_MEAN_VELOCITY = 1.15  # floating debris rides the faster surface layer


def channel_flow_velocity(hydraulic_class: str, rainfall_intensity_mmh: float, silt_ratio: float = 0.0) -> Dict[str, Any]:
    """Mean and surface velocity in an open channel for a given rainfall (Manning's equation).

    Discharge = dry-weather base flow + rational-method storm runoff from the catchment.
    Flow depth is solved by bisection on Q = (1/n)·A·R^(2/3)·S^(1/2). Silt raises bed
    roughness (n × (1 + 0.5·silt)) and takes away part of the bank-full depth.
    """
    g = CHANNEL_HYDRAULICS.get(hydraulic_class)
    if g is None:
        return {"available": False, "reason": f"no open-channel hydraulics for '{hydraulic_class}'"}
    rain = max(0.0, min(75.0, float(rainfall_intensity_mmh)))
    silt = max(0.0, min(0.9, float(silt_ratio)))
    b, s = g["width_m"], g["slope"]
    n = g["manning_n"] * (1.0 + 0.5 * silt)
    max_depth = g["max_depth_m"] * (1.0 - 0.6 * silt)
    q = g["base_flow_m3s"] + g["runoff_c"] * rain * g["catchment_ha"] * 10000.0 / 3.6e6

    def manning_q(y: float) -> float:
        area, perim = b * y, b + 2.0 * y
        return (1.0 / n) * area * (area / perim) ** (2.0 / 3.0) * math.sqrt(s)

    capacity = manning_q(max_depth)
    if q >= capacity:
        depth, surcharged = max_depth, True
        mean_v = capacity / (b * max_depth)
    else:
        lo, hi = 1e-4, max_depth
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if manning_q(mid) < q else (lo, mid)
        depth, surcharged = 0.5 * (lo + hi), False
        mean_v = q / (b * depth)
    return {
        "available": True,
        "hydraulic_class": hydraulic_class,
        "discharge_m3s": round(q, 3),
        "capacity_m3s": round(capacity, 3),
        "flow_depth_m": round(depth, 3),
        "mean_velocity_ms": round(mean_v, 3),
        "surface_velocity_ms": round(mean_v * SURFACE_TO_MEAN_VELOCITY, 3),
        "debris_transport_factor": g["debris_transport"],
        "debris_velocity_ms": round(mean_v * SURFACE_TO_MEAN_VELOCITY * g["debris_transport"], 3),
        "surcharged": surcharged,
        "manning_n_effective": round(n, 4),
    }


def rain_category(rain_mmh: float) -> str:
    if rain_mmh <= 0.0:
        return "Dry"
    if rain_mmh < 2.5:
        return "Light"
    if rain_mmh < 7.5:
        return "Moderate"
    if rain_mmh < 35.0:
        return "Heavy"
    if rain_mmh < 60.0:
        return "Very heavy burst"
    return "Extreme cloudburst-type burst"


class MonsoonSimulatorEngine:
    """Hydrological & municipal logistics simulator for stormwater plastic choke points."""

    def __init__(self):
        pass

    # ── Inputs ───────────────────────────────────────────────────────────────
    @staticmethod
    def _spots(base_hotspots: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if not base_hotspots:
            return [
                {"id": "HS-01", "name": "Shaniwar Wada Culvert", "severity": 4.6, "drain_dist_m": 18.5, "base_kg": 38.0, "priority": 92.0},
                {"id": "HS-02", "name": "Nagzari Nallah Intake", "severity": 4.1, "drain_dist_m": 12.0, "base_kg": 44.0, "priority": 88.0},
                {"id": "HS-03", "name": "Mutha River Outfall", "severity": 4.5, "drain_dist_m": 8.5, "base_kg": 62.0, "priority": 90.0},
                {"id": "HS-04", "name": "Kothrud Feeder Channel", "severity": 3.5, "drain_dist_m": 28.0, "base_kg": 26.0, "priority": 70.0},
                {"id": "HS-05", "name": "Yerawada Canal Culvert", "severity": 2.8, "drain_dist_m": 55.0, "base_kg": 18.0, "priority": 48.0},
            ]
        spots = []
        for idx, h in enumerate(base_hotspots, 1):
            sev = float(h.get("severity", h.get("severity_raw", 3.5)))
            dist = float(h.get("nearest_drain_distance_m", 35.0))
            spots.append({
                "id": h.get("id") or h.get("hotspot_id", f"HS-{idx:02d}"),
                "name": h.get("nearest_drain_name", f"Hotspot {idx}"),
                "severity": sev,
                "drain_dist_m": dist,
                # Base uncollected plastic mass ~ 12 - 45 kg per cluster
                "base_kg": round(sev * 6.5 + (max(0.0, 60.0 - dist) * 0.3), 1),
                "priority": float(h.get("total_score", 50.0)),
            })
        return spots

    # ── Core hydraulic + logistics model ─────────────────────────────────────
    @staticmethod
    def _run(spots: List[Dict[str, Any]], rain: float, fleet: int, delay: float, wind: float, silt_pct: float) -> Dict[str, Any]:
        silt = silt_pct / 100.0
        lead, duration = _a("storm_lead_time_h"), _a("storm_duration_h")
        service = _a("site_service_time_h")

        # Runoff into each inlet (L/s): C * i[mm/h] * A[m²] / 3600
        q_in = _a("runoff_coefficient") * rain * _a("catchment_area_m2") / 3600.0
        silt_factor = 1.0 - _a("silt_capacity_loss") * silt
        q_cap_clear = _a("inlet_design_capacity_lps") * silt_factor

        # Loose plastic mobilised: rain wash-in + wind drift, amplified when silted drains overtop
        flush_factor = min(1.0, (1.0 - math.exp(-0.04 * (rain + 0.1 * wind))) * (1.0 + silt))
        catchment_wash = 1.0 + (rain / 50.0) * 0.45 + wind * 0.004

        ordered = sorted(spots, key=lambda s: -s["priority"])
        site_rows = []
        for rank, s in enumerate(ordered):
            kg = s["base_kg"] * catchment_wash
            done_at = delay + (rank // fleet + 1) * service if fleet > 0 else float("inf")
            # Fraction of the storm the site spends uncleared (0 = cleared before peak)
            exposure = min(1.0, max(0.0, (done_at - lead) / duration))
            remaining_kg = kg * (0.05 + 0.95 * exposure)
            blockage = min(_a("max_plastic_blockage"), _a("max_plastic_blockage") * remaining_kg / _a("kg_for_full_blockage"))
            q_cap = q_cap_clear * (1.0 - blockage)
            excess_lps = max(0.0, q_in - q_cap)
            flood_m2 = excess_lps * duration * 3600.0 / 1000.0 / _a("ponding_depth_m")
            swept = 0.0 if rain <= 0 else (kg * exposure) * flush_factor * (0.4 + 0.6 * exposure)
            site_rows.append({
                "id": s["id"],
                "name": s["name"],
                "plastic_kg": round(kg, 1),
                "cleared_kg": round(kg - remaining_kg, 1),
                "cleared_before_peak": exposure == 0.0,
                "clearance_eta_h": None if math.isinf(done_at) else round(done_at, 1),
                "blockage_pct": round(blockage * 100.0, 0),
                "inlet_capacity_lps": round(q_cap, 1),
                "runoff_lps": round(q_in, 1),
                "overflow_lps": round(excess_lps, 1),
                "flood_area_m2": round(flood_m2, 0),
                "swept_kg": round(swept, 1),
                "choking": excess_lps > 0.0 and blockage >= 0.2,
            })

        at_risk = sum(r["plastic_kg"] for r in site_rows)
        cleared = sum(r["cleared_kg"] for r in site_rows)
        swept = sum(r["swept_kg"] for r in site_rows)
        flood = sum(r["flood_area_m2"] for r in site_rows)
        flooded_sites = sum(1 for r in site_rows if r["flood_area_m2"] >= 200.0)

        dewatering = flooded_sites * _a("dewatering_inr_per_site_h") * (duration + _a("drain_down_h") + 0.25 * delay)
        disruption = flood * _a("disruption_inr_per_m2")
        river_cost = swept * _a("river_recovery_inr_per_kg")
        surcharge = _a("emergency_surcharge_inr") if (delay >= 6.0 and rain >= 30.0) else 0.0

        return {
            "rows": site_rows,
            "q_in": q_in,
            "q_cap_clear": q_cap_clear,
            "flush_factor": flush_factor,
            "at_risk_kg": at_risk,
            "cleared_kg": cleared,
            "swept_kg": swept,
            "flood_m2": flood,
            "choking": sum(1 for r in site_rows if r["choking"]),
            "cleared_before_peak": sum(1 for r in site_rows if r["cleared_before_peak"]),
            "costs": {
                "dewatering_pumping": dewatering,
                "traffic_disruption_impact": disruption,
                "river_restoration_penalty": river_cost,
                "emergency_contractor_surcharge": surcharge,
            },
            "total_cost": dewatering + disruption + river_cost + surcharge,
        }

    # ── Explainability ───────────────────────────────────────────────────────
    def _explain(self, spots, res, rain, fleet, delay, wind, silt_pct) -> Dict[str, Any]:
        n = len(res["rows"])
        lead = _a("storm_lead_time_h")
        narrative = []
        if rain <= 0:
            narrative.append("No rain: no runoff reaches the inlets, so nothing floods and no plastic is flushed to the river.")
        else:
            narrative.append(
                f"{rain:.1f} mm/h ({rain_category(rain).lower()}) sends ≈{res['q_in']:.0f} L/s of runoff to each inlet; "
                f"a clean inlet can take {res['q_cap_clear']:.0f} L/s after {silt_pct:.0f}% siltation "
                f"(design {_a('inlet_design_capacity_lps'):.0f} L/s)."
                + (" Even clean inlets overflow at this intensity." if res["q_in"] > res["q_cap_clear"] else "")
            )
            missed = [r["id"] for r in res["rows"] if not r["cleared_before_peak"]]
            narrative.append(
                f"{fleet} truck(s) starting after {delay:.0f} h clear {res['cleared_before_peak']} of {n} sites before the storm peak "
                f"(T+{lead:.0f} h)" + (f"; still blocked: {', '.join(missed[:4])}{'…' if len(missed) > 4 else ''}." if missed else ".")
            )
            max_block = max((r["blockage_pct"] for r in res["rows"]), default=0)
            if res["choking"]:
                narrative.append(
                    f"Uncleared plastic blinds grates by up to {max_block:.0f}%, so {res['choking']} inlet(s) choke "
                    f"and ≈{res['flood_m2']:,.0f} m² waterlogs."
                )
            elif res["flood_m2"] > 0:
                narrative.append(
                    f"No inlet is plastic-choked (max blockage {max_block:.0f}%); the ≈{res['flood_m2']:,.0f} m² of waterlogging "
                    "comes from runoff exceeding the silt-reduced drain capacity."
                )
            else:
                narrative.append("All inlets keep up with the runoff, so no waterlogging is expected.")
            nallah = channel_flow_velocity("nallah", rain, silt_pct / 100.0)
            narrative.append(
                f"Open nallahs run at ≈{nallah['mean_velocity_ms']:.2f} m/s: floating litter reported now drifts "
                f"≈{nallah['debris_velocity_ms'] * delay * 3600:,.0f} m downstream before a crew {delay:.0f} h away arrives, "
                "so open-channel reports are intercepted at downstream screens (TRANSIENT_SURCHARGE)."
                if delay > 0 else
                f"Open nallahs run at ≈{nallah['mean_velocity_ms']:.2f} m/s: floating litter keeps moving, so open-channel "
                "reports are intercepted at downstream screens (TRANSIENT_SURCHARGE)."
            )
            narrative.append(
                f"Wind {wind:.0f} km/h and silt {silt_pct:.0f}% give a flush factor of {res['flush_factor']:.2f}: "
                f"≈{res['swept_kg']:.0f} kg of loose plastic is carried toward the Mula-Mutha."
            )

        # One-lever-at-a-time sensitivity: what would each fix save?
        levers = [
            ("Dispatch immediately (delay → 0 h)", dict(delay=0.0), delay > 0),
            ("Full fleet (10 trucks)", dict(fleet=10), fleet < 10),
            ("Pre-monsoon desilting (silt → 0%)", dict(silt_pct=0.0), silt_pct > 0),
            ("Calm wind (0 km/h)", dict(wind=0.0), wind > 0),
        ]
        base_args = dict(rain=rain, fleet=fleet, delay=delay, wind=wind, silt_pct=silt_pct)
        sensitivity = []
        for label, change, applicable in levers:
            if not applicable:
                continue
            alt = self._run(spots, **{**base_args, **change})
            sensitivity.append({
                "lever": label,
                "cost_saving_inr": round(res["total_cost"] - alt["total_cost"], 0),
                "flood_reduction_m2": round(res["flood_m2"] - alt["flood_m2"], 0),
                "river_plastic_reduction_kg": round(res["swept_kg"] - alt["swept_kg"], 1),
            })
        sensitivity.sort(key=lambda x: -x["cost_saving_inr"])
        top = sensitivity[0] if sensitivity and sensitivity[0]["cost_saving_inr"] > 0 else None
        headline = (
            f"Biggest lever: {top['lever']} saves ≈₹{top['cost_saving_inr']:,.0f} "
            f"and {top['flood_reduction_m2']:,.0f} m² of flooding."
            if top else "This scenario is already near the best achievable outcome for the chosen rainfall."
        )
        return {"headline": headline, "narrative": narrative, "sensitivity": sensitivity}

    # ── Public API ───────────────────────────────────────────────────────────
    def simulate(
        self,
        rainfall_intensity_mmh: float = 25.0,
        available_fleet: int = 4,
        action_delay_hours: float = 2.0,
        base_hotspots: Optional[List[Dict[str, Any]]] = None,
        wind_speed_kmh: float = 15.0,
        drain_silt_pct: float = 30.0,
    ) -> Dict[str, Any]:
        """Run what-if scenario simulation.

        Args:
            rainfall_intensity_mmh: 0.0 to 75.0 mm/hr.
            available_fleet: 1 to 10 trucks.
            action_delay_hours: 0.0 to 24.0 hours.
            base_hotspots: Optional list of hotspots from pipeline.
            wind_speed_kmh: 0.0 to 50.0 km/h surface wind drift.
            drain_silt_pct: 0.0 to 100.0 % drain capacity choked by silt.

        Returns:
            Dictionary with scenario outcomes, flood footprint, river wash-off,
            monetary civic damage costs, comparative baselines and explanations.
        """
        # Clamp inputs within bounded operational limits
        rain_i = max(0.0, min(75.0, float(rainfall_intensity_mmh)))
        fleet = max(1, min(10, int(available_fleet)))
        delay = max(0.0, min(24.0, float(action_delay_hours)))
        wind = max(0.0, min(50.0, float(wind_speed_kmh)))
        silt_pct = max(0.0, min(100.0, float(drain_silt_pct)))

        spots = self._spots(base_hotspots)
        res = self._run(spots, rain_i, fleet, delay, wind, silt_pct)
        optimal = self._run(spots, rain_i, 10, 0.0, wind, silt_pct)
        worst = self._run(spots, rain_i, 0, 24.0, wind, silt_pct)
        all_cleared = self._run(spots, rain_i, len(spots), 0.0, wind, silt_pct)
        no_silt = self._run(spots, rain_i, fleet, delay, wind, 0.0)

        flood_area_sqm = round(res["flood_m2"], 0)
        preventable_flood_sqm = round(max(0.0, res["flood_m2"] - all_cleared["flood_m2"]), 0)
        plastic_at_risk_kg = round(res["at_risk_kg"], 1)
        swept_kg = round(res["swept_kg"], 1)
        total_cost_inr = round(res["total_cost"], 0)
        critical_choking_count = res["choking"]

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
                "rainfall_category": rain_category(rain_i),
                "available_fleet": fleet,
                "action_delay_hours": delay,
                "wind_speed_kmh": wind,
                "drain_silt_pct": silt_pct,
            },
            "hydraulics": {
                "flush_factor": round(res["flush_factor"], 3),
                "runoff_per_inlet_lps": round(res["q_in"], 1),
                "clean_inlet_capacity_lps": round(res["q_cap_clear"], 1),
                # Flooding attributable to siltation: same scenario with fully desilted drains
                "silt_added_flood_m2": round(max(0.0, res["flood_m2"] - no_silt["flood_m2"]), 0),
                "drain_capacity_remaining_pct": round(100.0 * (1.0 - _a("silt_capacity_loss") * silt_pct / 100.0), 1),
                "sites_cleared_before_peak": res["cleared_before_peak"],
                "open_channel_velocity_ms": channel_flow_velocity("nallah", rain_i, silt_pct / 100.0)["mean_velocity_ms"],
            },
            "scenario_alert": scenario_alert,
            "alert_color": alert_color,
            "outcomes": {
                "critical_choking_hotspots": critical_choking_count,
                "total_monitored_hotspots": len(spots),
                "plastic_mass_at_risk_kg": plastic_at_risk_kg,
                "plastic_cleared_by_fleet_kg": round(res["cleared_kg"], 1),
                "plastic_swept_into_rivers_kg": swept_kg,
                "plastic_prevented_from_rivers_pct": round(
                    ((plastic_at_risk_kg - swept_kg) / max(0.1, plastic_at_risk_kg)) * 100.0, 1
                ),
                "potential_flood_inundation_area_sqm": flood_area_sqm,
                "preventable_flood_area_sqm": preventable_flood_sqm,
                "estimated_civic_cost_of_inaction_inr": total_cost_inr,
                "cost_breakdown_inr": {k: round(v, 0) for k, v in res["costs"].items()},
            },
            "site_breakdown": res["rows"],
            "comparative_benchmarks": {
                "optimal_immediate_dispatch": {
                    "fleet": 10,
                    "delay_hours": 0.0,
                    "plastic_swept_kg": round(optimal["swept_kg"], 1),
                    "flood_area_sqm": round(optimal["flood_m2"], 0),
                    "estimated_cost_inr": round(optimal["total_cost"], 0),
                },
                "unmitigated_worst_case": {
                    "fleet": 0,
                    "delay_hours": 24.0,
                    "plastic_swept_kg": round(worst["swept_kg"], 1),
                    "flood_area_sqm": round(worst["flood_m2"], 0),
                    "estimated_cost_inr": round(worst["total_cost"], 0),
                },
            },
            "explanation": self._explain(spots, res, rain_i, fleet, delay, wind, silt_pct),
            "assumptions": ASSUMPTIONS,
            "provenance_badge": "[Simulated / Modelled Estimate]",
            "provenance_category": "Simulated",
            "methodology": (
                "Rational-method runoff (Q = C·i·A) vs blockage- and silt-reduced inlet capacity; "
                "priority-ordered fleet clearance against storm lead time; flush-factor river wash-off"
            ),
        }
