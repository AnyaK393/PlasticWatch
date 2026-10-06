"""Evidence & Provenance Layer for PlasticWatch Urban Intelligence Platform.

Enforces standardized provenance chips/badges across all municipal data points:
  🟢 [Observed]  -> Citizen GPS coordinates, upload timestamps, raw citizen photos.
  🔵 [Derived]   -> DBSCAN clusters, distance to storm drains (m), risk score (0-100).
  🟡 [Inferred]  -> YOLO detection boxes, cleanup verification ratios, root-cause hypotheses.
  🟣 [Simulated] -> Monsoon scenario projections, estimated flood footprint, delay costs.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


PROVENANCE_TIERS: Dict[str, Dict[str, Any]] = {
    "observed": {
        "tag": "[Observed]",
        "tier_name": "Observed",
        "emoji": "🟢",
        "color": "#10b981",
        "bg_color": "rgba(16, 185, 129, 0.18)",
        "border_color": "rgba(16, 185, 129, 0.45)",
        "description": "Ground-truth sensor or citizen observation: raw GPS coordinates, timestamps, unprocessed photos.",
    },
    "derived": {
        "tag": "[Derived]",
        "tier_name": "Derived",
        "emoji": "🔵",
        "color": "#38bdf8",
        "bg_color": "rgba(56, 189, 248, 0.18)",
        "border_color": "rgba(56, 189, 248, 0.45)",
        "description": "Deterministic spatial or mathematical calculation: DBSCAN clusters, distance to storm drains (m), 0-100 risk score.",
    },
    "inferred": {
        "tag": "[Inferred]",
        "tier_name": "Inferred",
        "emoji": "🟡",
        "color": "#fbbf24",
        "bg_color": "rgba(251, 191, 36, 0.18)",
        "border_color": "rgba(251, 191, 36, 0.45)",
        "description": "Machine learning / statistical deduction: YOLO/TACO object bounding boxes, cleanup verification ratios, root-cause hypotheses.",
    },
    "simulated": {
        "tag": "[Simulated]",
        "tier_name": "Simulated",
        "emoji": "🟣",
        "color": "#c084fc",
        "bg_color": "rgba(192, 132, 252, 0.18)",
        "border_color": "rgba(192, 132, 252, 0.45)",
        "description": "Parametric what-if scenario projection: monsoon rainfall surge, flood footprint estimation, delay cost modeling.",
    },
}


def get_provenance_meta(tier: str) -> Dict[str, Any]:
    """Retrieve metadata and styling for a provenance tier."""
    key = str(tier).lower().strip().replace("[", "").replace("]", "")
    if "predict" in key:
        key = "simulated"
    return PROVENANCE_TIERS.get(key, PROVENANCE_TIERS["derived"])


def render_provenance_badge(tier: str, custom_label: Optional[str] = None, font_size: str = "0.74rem") -> str:
    """Render an inline HTML badge for Streamlit and dashboard markdown."""
    meta = get_provenance_meta(tier)
    label = custom_label if custom_label else meta["tag"]
    return (
        f'<span style="background: {meta["bg_color"]}; color: {meta["color"]}; '
        f'border: 1px solid {meta["border_color"]}; padding: 2px 7px; border-radius: 6px; '
        f'font-size: {font_size}; font-weight: 700; display: inline-flex; align-items: center; '
        f'gap: 4px; letter-spacing: 0.02em; vertical-align: middle;">'
        f'{meta["emoji"]} {label}</span>'
    )
