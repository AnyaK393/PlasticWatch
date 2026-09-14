"""PS-08 PlasticWatch: report clustering and explainable hotspot prioritisation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import asin, cos, radians, sin, sqrt


@dataclass(frozen=True)
class WasteReport:
    id: str
    lat: float
    lon: float
    plastic_confidence: float
    severity: int
    near_drain_m: int
    source: str
    verification: str


def _distance_m(a: WasteReport, b: WasteReport) -> float:
    dlat, dlon = radians(b.lat - a.lat), radians(b.lon - a.lon)
    q = sin(dlat / 2) ** 2 + cos(radians(a.lat)) * cos(radians(b.lat)) * sin(dlon / 2) ** 2
    return 6_371_000 * 2 * asin(sqrt(q))


def replay_reports() -> list[WasteReport]:
    """Clearly-labelled sample inputs; replace with TACO model/citizen-report adapters."""
    return [
        WasteReport("R-101", 18.5205, 73.8568, .91, 5, 35, "Synthetic TACO-style street report", "verified"),
        WasteReport("R-102", 18.5209, 73.8571, .84, 4, 52, "Synthetic TACO-style street report", "pending"),
        WasteReport("R-103", 18.5212, 73.8564, .79, 3, 70, "Synthetic citizen report", "pending"),
        WasteReport("R-104", 18.5074, 73.8077, .88, 5, 20, "Synthetic TACO-style street report", "verified"),
        WasteReport("R-105", 18.5079, 73.8082, .71, 3, 95, "Synthetic citizen report", "pending"),
        WasteReport("R-106", 18.5480, 73.9053, .67, 2, 210, "Synthetic citizen report", "pending"),
    ]


def _clusters(reports: list[WasteReport], merge_distance_m: float = 180) -> list[list[WasteReport]]:
    groups: list[list[WasteReport]] = []
    for report in reports:
        for group in groups:
            if any(_distance_m(report, existing) <= merge_distance_m for existing in group):
                group.append(report)
                break
        else:
            groups.append([report])
    return groups


def run_plasticwatch(mode: str = "replay") -> dict:
    if mode != "replay":
        raise RuntimeError("Live image/report connector is not configured. Do not substitute replay reports in live mode.")
    hotspots = []
    for index, group in enumerate(_clusters(replay_reports()), 1):
        confidence = sum(x.plastic_confidence for x in group) / len(group)
        severity = sum(x.severity for x in group) / len(group)
        drain = min(x.near_drain_m for x in group)
        recurrence = len(group)
        # Explainable public-service score: evidence + recurrence + severity + drainage exposure.
        score = round(confidence * 35 + severity * 8 + min(recurrence, 4) * 7 + max(0, 20 - drain / 10), 1)
        priority = "CRITICAL" if score >= 70 else "HIGH" if score >= 50 else "VERIFY"
        hotspots.append({
            "id": f"PH-{index:02d}", "lat": round(sum(x.lat for x in group) / len(group), 6),
            "lon": round(sum(x.lon for x in group) / len(group), 6), "score": score, "priority": priority,
            "confidence": round(confidence, 2), "severity": round(severity, 1), "recurrence": recurrence,
            "nearest_drain_m": drain, "verified_count": sum(x.verification == "verified" for x in group),
            "reports": [asdict(x) for x in group],
        })
    hotspots.sort(key=lambda item: item["score"], reverse=True)
    return {
        "challenge_id": "PS-08", "mode": mode, "mode_label": "SIMULATION REPLAY - synthetic, TACO-style reports",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "area": "Pune municipal pilot", "hotspots": hotspots,
        "formula": "confidence (35%) + severity (40%) + recurrence (up to 28 points) + drain exposure (up to 20 points)",
    }
