"""Explainable Intervention Decision System (XAI) for PlasticWatch operators.

Turns the exploded 0-100 risk breakdown into:
  - plain-English comparative explanations ("why Hotspot A outranks Hotspot B")
  - rule-based intervention recommendations with the crew type to send
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

REVIEW_STATUS = "PENDING_OPERATOR_VERIFICATION"

ACTION_META = {
    "IMMEDIATE_CLEANUP_DISPATCH": {
        "badge": "🚨 IMMEDIATE CLEANUP DISPATCH",
        "color": "#ef4444",
        "crew": "Suction & Desilting Crew",
    },
    "PREVENTIVE_DRAIN_SCREEN": {
        "badge": "🛡️ PREVENTIVE DRAIN SCREEN",
        "color": "#f59e0b",
        "crew": "Drain Screen Fabrication & Installation Crew",
    },
    "HUMAN_VERIFICATION_REQUIRED": {
        "badge": "🧑‍⚖️ HUMAN VERIFICATION REQUIRED",
        "color": "#c084fc",
        "crew": "Field Inspector (Photo Re-verification)",
    },
    "ROUTINE_MONITORING": {
        "badge": "👁️ ROUTINE MONITORING",
        "color": "#38bdf8",
        "crew": "Manual Litter Sweep",
    },
}

# Recommendation thresholds
DISPATCH_SCORE = 70.0
DISPATCH_DRAIN_M = 25.0
DISPATCH_RAIN_MMH = 10.0
SCREEN_RECURRENCE = 3
SCREEN_WINDOW_DAYS = 14
VERIFY_CONFIDENCE = 0.55
MONITOR_SCORE = 50.0
MONITOR_DRAIN_M = 100.0

FILM_CLASSES = {"plastic_bag_wrapper"}


def _parse_ts(value: Any) -> Optional[datetime]:
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _label(h: Dict[str, Any]) -> str:
    return str(h.get("id") or h.get("hotspot_id") or "Hotspot")


def _components(h: Dict[str, Any]) -> List[Tuple[str, float, str]]:
    """(factor name, points contributed, human-readable evidence) for one hotspot."""
    dist = float(h.get("nearest_drain_distance_m", 0.0))
    env_mult = float(h.get("env_multiplier", 1.0))
    env_note = f" x{env_mult:.2f} live rain/silt/slope" if env_mult > 1.0 else ""
    return [
        (
            "drain proximity & live environment",
            float(h.get("env_points", h.get("drain_points", 0.0))),
            f"{dist:.0f} m from {h.get('nearest_drain_name', 'a drain')}{env_note}",
        ),
        (
            "waste severity",
            float(h.get("severity_points", 0.0)),
            f"severity {float(h.get('severity_raw', h.get('severity', 0.0))):.1f}/5",
        ),
        (
            "AI detection confidence",
            float(h.get("confidence_points", 0.0)),
            f"{float(h.get('confidence_raw', h.get('confidence', 0.0))):.0%} confidence",
        ),
        (
            "24h rainfall forecast",
            float(h.get("rainfall_points", 0.0)),
            f"{float(h.get('rainfall_forecast_mm', 0.0)):.1f} mm forecast",
        ),
        (
            "repeat citizen reports",
            float(h.get("recurrence_bonus", 0.0)),
            f"{int(h.get('recurrence', h.get('recurrence_count', 1)))} report(s)",
        ),
    ]


class ExplainabilityEngine:
    """Human-in-the-loop decision support: comparative explanations and actions."""

    @staticmethod
    def explain_ranking(hotspot_a: dict, hotspot_b: dict) -> str:
        """Compares two hotspots and explains why A outranks B."""
        score_a = float(hotspot_a.get("total_score", 0.0))
        score_b = float(hotspot_b.get("total_score", 0.0))

        # Always narrate from the higher-ranked site's perspective
        if score_a < score_b:
            hotspot_a, hotspot_b = hotspot_b, hotspot_a
            score_a, score_b = score_b, score_a
        a_id, b_id = _label(hotspot_a), _label(hotspot_b)
        gap = score_a - score_b

        comps_a = _components(hotspot_a)
        comps_b = _components(hotspot_b)
        deltas = [
            (name, pts_a - pts_b, ev_a, ev_b)
            for (name, pts_a, ev_a), (_, pts_b, ev_b) in zip(comps_a, comps_b)
        ]
        favour_a = sorted([d for d in deltas if d[1] >= 0.5], key=lambda d: -d[1])
        favour_b = sorted([d for d in deltas if d[1] <= -0.5], key=lambda d: d[1])

        if gap < 0.5:
            return (
                f"{a_id} ({score_a:.1f}) and {b_id} ({score_b:.1f}) are effectively tied; "
                "dispatch order should be decided on crew proximity and field conditions."
            )

        parts = [f"{a_id} ({score_a:.1f}/100) outranks {b_id} ({score_b:.1f}/100) by {gap:.1f} points."]
        if favour_a:
            reasons = [
                f"{name} (+{delta:.1f} pts: {ev_a} vs {ev_b})"
                for name, delta, ev_a, ev_b in favour_a[:3]
            ]
            parts.append("Decisive factors: " + "; ".join(reasons) + ".")
        if favour_b:
            name, delta, _, ev_b = favour_b[0]
            parts.append(
                f"{b_id} is stronger on {name} ({ev_b}, {abs(delta):.1f} pts), "
                "but not enough to close the gap."
            )
        return " ".join(parts)

    @staticmethod
    def explain_queue_position(hotspot: dict, ranked_hotspots: List[dict]) -> str:
        """Explain a site against its neighbour below it in the queue (or above, if last)."""
        ids = [_label(h) for h in ranked_hotspots]
        hid = _label(hotspot)
        if hid not in ids or len(ranked_hotspots) < 2:
            return f"{hid} is the only site in the active queue."
        idx = ids.index(hid)
        if idx + 1 < len(ranked_hotspots):
            return ExplainabilityEngine.explain_ranking(hotspot, ranked_hotspots[idx + 1])
        return ExplainabilityEngine.explain_ranking(ranked_hotspots[idx - 1], hotspot)

    @staticmethod
    def _recent_recurrence(hotspot: dict, now: datetime) -> int:
        reports = hotspot.get("reports") or []
        if not reports:
            return int(hotspot.get("recurrence", 1))
        cutoff = now - timedelta(days=SCREEN_WINDOW_DAYS)
        return sum(
            1 for r in reports
            if (ts := _parse_ts(r.get("submitted_at"))) is not None and cutoff <= ts <= now + timedelta(days=1)
        )

    @staticmethod
    def _dominant_waste(hotspot: dict) -> str:
        classes: List[str] = []
        for r in hotspot.get("reports") or []:
            classes.extend(r.get("detected_classes") or [])
        if not classes:
            return str(hotspot.get("dominant_class", "unknown"))
        return Counter(classes).most_common(1)[0][0]

    @staticmethod
    def recommend_action(hotspot: dict, now: Optional[datetime] = None) -> dict:
        """
        Evaluates hotspot attributes and returns:
        - action_type: 'IMMEDIATE_CLEANUP_DISPATCH' | 'PREVENTIVE_DRAIN_SCREEN' |
                       'HUMAN_VERIFICATION_REQUIRED' | 'ROUTINE_MONITORING'
        - reason_summary: Plain-English explanation
        - recommended_crew: e.g. 'Suction & Desilting Crew' vs 'Manual Litter Sweep'
        """
        now = now or datetime.now(timezone.utc)
        score = float(hotspot.get("total_score", 0.0))
        dist = float(hotspot.get("nearest_drain_distance_m", 999.0))
        rain = float(hotspot.get("rainfall_intensity_mmh", 0.0))
        conf = float(hotspot.get("confidence_raw", hotspot.get("confidence", 1.0)))
        status = str(hotspot.get("status", ""))
        recent = ExplainabilityEngine._recent_recurrence(hotspot, now)
        dominant = ExplainabilityEngine._dominant_waste(hotspot)
        escalation_trigger = None

        # Safety first: never auto-dispatch on unreadable or unconfirmed evidence
        if conf < VERIFY_CONFIDENCE or status == REVIEW_STATUS:
            action = "HUMAN_VERIFICATION_REQUIRED"
            why = []
            if conf < VERIFY_CONFIDENCE:
                why.append(f"AI confidence is only {conf:.0%} (below {VERIFY_CONFIDENCE:.0%})")
            if status == REVIEW_STATUS:
                why.append("the photo was flagged blurry/dark at ingestion")
            reason = (
                " and ".join(why).capitalize()
                + ". An operator must confirm the site before a crew or payment is committed."
            )
        elif score >= DISPATCH_SCORE and dist <= DISPATCH_DRAIN_M and rain >= DISPATCH_RAIN_MMH:
            action = "IMMEDIATE_CLEANUP_DISPATCH"
            reason = (
                f"Critical score {score:.1f}, waste only {dist:.0f} m from the drain inlet, and live rainfall at "
                f"{rain:.1f} mm/h: the inlet is likely to choke during this storm. Dispatch suction crew now."
            )
        elif recent >= SCREEN_RECURRENCE and dominant in FILM_CLASSES:
            action = "PREVENTIVE_DRAIN_SCREEN"
            reason = (
                f"{recent} citizen reports in the last {SCREEN_WINDOW_DAYS} days, dominated by plastic film/bags. "
                "Repeated sweeping treats the symptom; install a bar/mesh screen at the inlet to stop recurrence."
            )
        else:
            action = "ROUTINE_MONITORING"
            if score < MONITOR_SCORE and dist > MONITOR_DRAIN_M:
                reason = (
                    f"Moderate score {score:.1f} and {dist:.0f} m from the nearest drain: low choke risk. "
                    "Keep in the routine sweep cycle."
                )
            else:
                missing = []
                if score < DISPATCH_SCORE:
                    missing.append(f"score {score:.1f} < {DISPATCH_SCORE:.0f}")
                if dist > DISPATCH_DRAIN_M:
                    missing.append(f"drain distance {dist:.0f} m > {DISPATCH_DRAIN_M:.0f} m")
                if rain < DISPATCH_RAIN_MMH:
                    missing.append(f"live rainfall {rain:.1f} mm/h < {DISPATCH_RAIN_MMH:.0f} mm/h")
                reason = (
                    f"Score {score:.1f} with waste {dist:.0f} m from the drain, but immediate-dispatch criteria are not "
                    f"all met ({'; '.join(missing)}). Schedule in the next routine sweep and keep watching."
                )
                if score >= DISPATCH_SCORE and dist <= DISPATCH_DRAIN_M and rain < DISPATCH_RAIN_MMH:
                    escalation_trigger = (
                        f"Auto-escalates to IMMEDIATE CLEANUP DISPATCH once live rainfall reaches "
                        f"{DISPATCH_RAIN_MMH:.0f} mm/h."
                    )

        meta = ACTION_META[action]
        return {
            "action_type": action,
            "badge": meta["badge"],
            "color": meta["color"],
            "reason_summary": reason,
            "recommended_crew": meta["crew"],
            "escalation_trigger": escalation_trigger,
            "evidence": {
                "total_score": round(score, 1),
                "nearest_drain_distance_m": round(dist, 1),
                "rainfall_intensity_mmh": round(rain, 1),
                "confidence": round(conf, 3),
                "recent_reports_14d": recent,
                "dominant_waste_class": dominant,
                "status": status or "ACTIVE",
            },
        }
