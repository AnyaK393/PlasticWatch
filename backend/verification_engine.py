"""AI Before/After Cleanup Verification Engine for PlasticWatch.

Performs dual-image computer vision inference using TACO/YOLO detection to:
  1. Quantify reduction in waste object count
  2. Compute severity reduction delta (Delta Severity)
  3. Calculate Cleanup Effectiveness Score (0-100%)
  4. Estimate diverted plastic weight (kg)
  5. Adjudicate contract clearance decision (PASS >= 85% or FLAGGED FOR REVIEW)
"""
from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

from backend.taco_adapter import detect_waste

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "data" / "taco_samples"


class CleanupVerificationEngine:
    """Automated dual-image verification engine for municipal cleanup work orders."""

    def __init__(self, samples_dir: Path = SAMPLES_DIR):
        self.samples_dir = samples_dir

    def _resolve_image_input(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
    ) -> Tuple[Union[Path, bytes, Image.Image], str]:
        """Resolves filename string, local path, raw bytes, or base64 into callable format."""
        if isinstance(image_input, (Path, Image.Image)):
            hint = getattr(image_input, "name", "")
            return image_input, hint

        if isinstance(image_input, (bytes, bytearray)):
            return image_input, "uploaded_image.jpg"

        if isinstance(image_input, str):
            # Check if base64 data URI
            if image_input.startswith("data:image"):
                try:
                    _, b64data = image_input.split(",", 1)
                    raw_bytes = base64.b64decode(b64data)
                    return raw_bytes, "base64_upload.jpg"
                except Exception:
                    pass

            # Check if local file path
            p = Path(image_input)
            if p.is_file():
                return p, p.name

            # Check in samples directory
            sample_p = self.samples_dir / image_input
            if sample_p.is_file():
                return sample_p, sample_p.name

            # Fallback
            return image_input, image_input

        return image_input, ""

    def verify_cleanup(
        self,
        before_image: Union[str, Path, bytes, Image.Image],
        after_image: Union[str, Path, bytes, Image.Image],
        hotspot_id: Optional[str] = None,
        site_name: Optional[str] = None,
        contractor_id: Optional[str] = "PMC-SANITATION-04",
        operator_notes: Optional[str] = "",
    ) -> Dict[str, Any]:
        """Run AI dual-inference and verify contractor clearance.

        Args:
            before_image: Pre-cleanup image (path, filename, bytes, or PIL).
            after_image: Post-cleanup clearance image.
            hotspot_id: Optional hotspot ID reference.
            site_name: Optional municipal location name.
            contractor_id: Municipal crew/contractor ID.
            operator_notes: Notes entered by operator or field crew.

        Returns:
            Verification dossier containing quantified reductions, effectiveness %,
            PASS/FLAGGED decision status, annotated images, and provenance badges.
        """
        before_in, before_hint = self._resolve_image_input(before_image)
        after_in, after_hint = self._resolve_image_input(after_image)

        # 1. Run TACO inference on baseline before-photo
        before_det = detect_waste(before_in, filename_hint=before_hint)
        # 2. Run TACO inference on post-cleanup clearance photo
        after_det = detect_waste(after_in, filename_hint=after_hint)

        before_items = before_det.get("detected_items", [])
        after_items = after_det.get("detected_items", [])

        before_count = len(before_items)
        after_count = len(after_items)

        # Quantified object count reduction
        count_reduction = max(0, before_count - after_count)
        if before_count > 0:
            count_reduction_pct = round(((before_count - after_count) / before_count) * 100.0, 1)
        else:
            count_reduction_pct = 100.0 if after_count == 0 else 0.0

        # Severity comparison (1.0 - 5.0 scale)
        before_sev = float(before_det.get("severity", 3.5))
        after_sev = float(after_det.get("severity", 0.5))
        delta_severity = round(before_sev - after_sev, 2)
        sev_reduction_pct = round((max(0.0, delta_severity) / max(0.1, before_sev)) * 100.0, 1)

        # 3. Cleanup Effectiveness Score (0 - 100%)
        if after_count == 0 and after_sev <= 1.0:
            # 100% clean site
            effectiveness_score = 98.5 if before_count > 0 else 100.0
        else:
            raw_eff = (count_reduction_pct * 0.55) + (sev_reduction_pct * 0.45)
            effectiveness_score = round(max(0.0, min(100.0, raw_eff)), 1)

        # 4. Estimated Diverted Plastic Weight in kg
        # Scaled by count reduction and delta severity impact
        base_kg_per_item = 2.4
        diverted_kg = round(max(0.5, (count_reduction * base_kg_per_item) + (max(0.0, delta_severity) * 3.5)), 1)
        diverted_volume_liters = round(diverted_kg * 8.2, 0)

        # 5. Adjudication Decision Status (Threshold: >= 85% PASS)
        if effectiveness_score >= 85.0:
            decision_status = "PASS"
            status_color = "#10b981"  # Emerald green
            status_message = "Clearance criteria satisfied (>= 85%). Contractor invoice eligible for approval."
            invoice_eligible = True
        else:
            decision_status = "FLAGGED FOR REVIEW"
            status_color = "#f59e0b"  # Amber warning
            status_message = "Clearance below required 85% threshold. Field re-inspection required before contractor payment."
            invoice_eligible = False

        # Class-level reduction breakdown
        before_classes = [it.get("class", "plastic") for it in before_items]
        after_classes = [it.get("class", "plastic") for it in after_items]

        return {
            "status": "success",
            "hotspot_id": hotspot_id or "HS-VERIFY",
            "site_name": site_name or "Municipal Storm Drain Pilot Site",
            "contractor_id": contractor_id,
            "decision_status": decision_status,
            "status_color": status_color,
            "status_message": status_message,
            "invoice_eligible": invoice_eligible,
            "effectiveness_score": effectiveness_score,
            "metrics": {
                "before_object_count": before_count,
                "after_object_count": after_count,
                "count_reduction": count_reduction,
                "count_reduction_pct": count_reduction_pct,
                "before_severity": before_sev,
                "after_severity": after_sev,
                "delta_severity": delta_severity,
                "severity_reduction_pct": sev_reduction_pct,
                "diverted_plastic_weight_kg": diverted_kg,
                "diverted_volume_liters": diverted_volume_liters,
                "before_classes": before_classes,
                "after_classes": after_classes,
            },
            "visuals": {
                "before_annotated_base64": before_det.get("annotated_image_base64", ""),
                "after_annotated_base64": after_det.get("annotated_image_base64", ""),
            },
            "disclaimer": "Operator sign-off required for final contractor invoice clearance.",
            "provenance_badge": "[AI Inference / Modelled Estimate]",
            "provenance_category": "Inferred",
            "operator_notes": operator_notes or "",
        }
