"""Smart Dispatch & 90-Day Contractor Anti-Fraud Shield for PlasticWatch.

  1. Matches a hotspot to the pre-approved sanitation vendor for its ward/zone
     (by ward/site name first, then by nearest zone anchor coordinate).
  2. Quotes the work order from the vendor's mobilization fee and per-ton rate.
  3. Blocks duplicate billing: if the same coordinate (within 35 m) was already
     cleaned and billed within the last 90 days, payment is held for field audit.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.recurrence_engine import haversine_m

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
VENDOR_REGISTRY_PATH = ROOT / "data" / "vendor_registry.json"
HISTORICAL_DOSSIER_PATH = ROOT / "data" / "historical_urban_dossier.json"

GST_RATE = 0.18


def _parse_ts(value: str) -> Optional[datetime]:
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


class SmartDispatchEngine:
    """Vendor zone matching, work-order quoting and duplicate-billing fraud checks."""

    def __init__(
        self,
        registry_path: Path = VENDOR_REGISTRY_PATH,
        dossier_path: Path = HISTORICAL_DOSSIER_PATH,
    ):
        self.registry_path = registry_path
        self.dossier_path = dossier_path
        self.vendors: List[Dict[str, Any]] = self._load_vendors()

    def _load_vendors(self) -> List[Dict[str, Any]]:
        if not self.registry_path.is_file():
            logger.warning("Vendor registry not found at %s", self.registry_path)
            return []
        try:
            return json.loads(self.registry_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Failed to load vendor registry: %s", exc)
            return []

    # ── 1. Vendor zone matching ──────────────────────────────────────────────
    def match_vendor(self, lat: float, lon: float, ward_or_site_name: str = "") -> Dict[str, Any]:
        """Match a site to its registered ward vendor.

        A zone name appearing in the ward/site name wins; otherwise the vendor whose
        zone anchor is geographically closest to the coordinate is selected.
        """
        if not self.vendors:
            return {
                "matched": False,
                "vendor_id": None,
                "name": "No registered vendor",
                "match_method": "registry_unavailable",
            }

        label = (ward_or_site_name or "").lower()
        if label:
            for vendor in self.vendors:
                for zone in vendor.get("assigned_zones", []):
                    if zone.lower() in label:
                        return {
                            **vendor,
                            "matched": True,
                            "matched_zone": zone,
                            "match_method": "ward_name",
                            "distance_to_zone_km": None,
                        }

        best_vendor, best_dist = None, float("inf")
        for vendor in self.vendors:
            for anchor_lat, anchor_lon in vendor.get("zone_anchors", []):
                d = haversine_m(lat, lon, anchor_lat, anchor_lon)
                if d < best_dist:
                    best_vendor, best_dist = vendor, d

        if best_vendor is None:
            best_vendor, best_dist = self.vendors[0], float("nan")

        return {
            **best_vendor,
            "matched": True,
            "matched_zone": best_vendor.get("assigned_zones", ["Unassigned"])[0],
            "match_method": "nearest_zone_anchor",
            "distance_to_zone_km": round(best_dist / 1000.0, 2),
        }

    # ── 2. Work order quote ──────────────────────────────────────────────────
    @staticmethod
    def estimate_weight_kg(hotspot: Dict[str, Any]) -> float:
        """Predicted removable mass, consistent with the monsoon simulator's base model."""
        sev = float(hotspot.get("severity_raw", hotspot.get("severity", 3.5)))
        dist = float(hotspot.get("nearest_drain_distance_m", 35.0))
        recurrence = int(hotspot.get("recurrence", 1))
        base_kg = sev * 6.5 + max(0.0, 60.0 - dist) * 0.3
        return round(base_kg * (1.0 + 0.15 * max(0, recurrence - 1)), 1)

    @staticmethod
    def calculate_work_order_quote(vendor: Dict[str, Any], predicted_weight_kg: float) -> Dict[str, Any]:
        """Quote = mobilization fee + per-ton rate x predicted tonnage, plus GST."""
        base = float(vendor.get("base_mobilization_inr", 0.0))
        rate = float(vendor.get("rate_per_ton_inr", 0.0))
        weight_kg = max(0.0, float(predicted_weight_kg))
        tons = weight_kg / 1000.0
        tonnage_cost = rate * tons
        subtotal = base + tonnage_cost
        gst = subtotal * GST_RATE
        return {
            "vendor_id": vendor.get("vendor_id"),
            "vendor_name": vendor.get("name"),
            "predicted_weight_kg": round(weight_kg, 1),
            "predicted_tons": round(tons, 3),
            "base_mobilization_inr": round(base, 2),
            "rate_per_ton_inr": round(rate, 2),
            "tonnage_cost_inr": round(tonnage_cost, 2),
            "subtotal_inr": round(subtotal, 2),
            "gst_inr": round(gst, 2),
            "total_quote_inr": round(subtotal + gst, 2),
        }

    # ── 3. 90-day duplicate billing fraud shield ─────────────────────────────
    def check_duplicate_billing_fraud(
        self,
        lat: float,
        lon: float,
        historical_dossier_path: Optional[Path] = None,
        days_threshold: int = 90,
        dist_threshold_m: float = 35.0,
        reference_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Flag a coordinate already cleaned and billed within the look-back window."""
        path = Path(historical_dossier_path) if historical_dossier_path else self.dossier_path
        now = reference_date or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        clean_result = {
            "is_fraud_suspect": False,
            "message": f"✅ FRAUD SHIELD: No billed cleanup within {dist_threshold_m:.0f} m in the last {days_threshold} days.",
            "days_threshold": days_threshold,
            "dist_threshold_m": dist_threshold_m,
            "previous_cleanup": None,
            "billed_cleanups_in_window": 0,
        }

        if not path.is_file():
            return {**clean_result, "message": "Fraud shield inactive: historical dossier not found."}

        try:
            dossier = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Failed to read dossier for fraud check: %s", exc)
            return {**clean_result, "message": "Fraud shield inactive: dossier unreadable."}

        matches = []
        for loc in dossier.get("locations", []):
            loc_dist = haversine_m(lat, lon, float(loc["lat"]), float(loc["lon"]))
            if loc_dist > dist_threshold_m:
                continue
            for c in loc.get("cleanups", []):
                ts = _parse_ts(c.get("timestamp", ""))
                if ts is None:
                    continue
                age_days = (now - ts).total_seconds() / 86400.0
                if 0.0 <= age_days <= days_threshold:
                    matches.append((ts, age_days, loc, c, loc_dist))

        if not matches:
            return clean_result

        ts, age_days, loc, c, loc_dist = max(matches, key=lambda m: m[0])
        contractor = c.get("contractor") or c.get("crew_id") or "registered contractor"
        if c.get("truck_id"):
            contractor = f"{contractor} ({c['truck_id']})"
        date_str = ts.date().isoformat()

        return {
            "is_fraud_suspect": True,
            "message": (
                f"⚠️ FRAUD SHIELD: Coordinate was cleaned on {date_str} by {contractor}. "
                "Duplicate payment blocked pending field audit."
            ),
            "days_threshold": days_threshold,
            "dist_threshold_m": dist_threshold_m,
            "billed_cleanups_in_window": len(matches),
            "previous_cleanup": {
                "log_id": c.get("log_id"),
                "date": date_str,
                "days_ago": round(age_days, 1),
                "contractor": contractor,
                "site_id": loc.get("site_id"),
                "site_name": loc.get("name"),
                "distance_m": round(loc_dist, 1),
                "weight_removed_kg": c.get("weight_removed_kg"),
                "clearance_effectiveness_pct": c.get("clearance_effectiveness_pct"),
                "verification_status": c.get("verification_status"),
            },
        }

    # ── Combined assessment for the inspection drawer / API ──────────────────
    def assess_hotspot(self, hotspot: Dict[str, Any]) -> Dict[str, Any]:
        lat, lon = float(hotspot["lat"]), float(hotspot["lon"])
        landuse = hotspot.get("landuse_attribution") or {}
        labels = [hotspot.get("ward"), hotspot.get("nearest_drain_name")]
        if landuse.get("attribution_strength") in ("STRONG", "MODERATE"):
            labels.append(landuse.get("landmark"))

        vendor = None
        for label in filter(None, labels):
            candidate = self.match_vendor(lat, lon, label)
            if candidate.get("match_method") == "ward_name":
                vendor = candidate
                break
        if vendor is None:
            vendor = self.match_vendor(lat, lon, "")
        quote = self.calculate_work_order_quote(vendor, self.estimate_weight_kg(hotspot)) if vendor.get("matched") else None
        fraud = self.check_duplicate_billing_fraud(lat, lon)
        return {
            "hotspot_id": hotspot.get("id") or hotspot.get("hotspot_id"),
            "vendor": vendor,
            "quote": quote,
            "fraud_check": fraud,
            "dispatch_allowed": not fraud["is_fraud_suspect"],
        }


# ── Work-order ledger (tendering audit trail) ────────────────────────────────
WORK_ORDER_LEDGER_PATH = ROOT / "data" / "work_orders.json"


def list_work_orders() -> List[Dict[str, Any]]:
    from backend.storage import CorruptDataError, read_json

    try:
        data = read_json(WORK_ORDER_LEDGER_PATH, [])
    except CorruptDataError as exc:  # quarantined copy kept for audit
        logger.error("%s", exc)
        return []
    return data if isinstance(data, list) else []


def record_work_order(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Append an issued work order (vendor, quote, fraud status) to the ledger."""
    from backend.storage import locked, write_json

    entry = {"issued_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(), **entry}
    with locked(WORK_ORDER_LEDGER_PATH):
        ledger = list_work_orders()
        ledger.append(entry)
        write_json(WORK_ORDER_LEDGER_PATH, ledger)
    return entry
