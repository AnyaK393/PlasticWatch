"""AI Before/After Cleanup Verification Engine for PlasticWatch.

Performs dual-image computer vision inference using TACO/YOLO detection to:
  1. Quantify reduction in waste object count (per class)
  2. Compute severity reduction delta (Delta Severity, clog-weighted)
  3. Calculate Cleanup Effectiveness Score (0-100%)
  4. Estimate diverted plastic weight (kg)
  5. Check the evidence itself: photo quality, same-scene match (ORB + RANSAC),
     EXIF capture order and GPS distance to the site
  6. Adjudicate: PASS / FLAGGED FOR REVIEW / INCONCLUSIVE (re-shoot required)
"""
from __future__ import annotations

import base64
from collections import Counter
from datetime import datetime
import io
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import ExifTags, Image, ImageDraw, ImageOps

from backend.taco_adapter import CLASS_SEVERITY_WEIGHTS, detect_waste, litter_blobs, plastic_signature_mask

try:
    import cv2
except ImportError:  # scene check degrades to "unverifiable"
    cv2 = None

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "data" / "taco_samples"

PASS_THRESHOLD_PCT = 85.0
SCENE_MATCH_INLIERS = 30
SCENE_MISMATCH_INLIERS = 12
SCENE_MIN_KEYPOINTS = 50
GPS_MAX_DISTANCE_M = 75.0
EARTH_RADIUS_METERS = 6371000.0


def _load_pil(image_input: Union[Path, bytes, Image.Image, str]) -> Optional[Image.Image]:
    try:
        if isinstance(image_input, Image.Image):
            return image_input
        if isinstance(image_input, (bytes, bytearray)):
            return Image.open(io.BytesIO(image_input))
        if isinstance(image_input, (str, Path)) and Path(image_input).is_file():
            return Image.open(image_input)
    except Exception:
        return None
    return None


def _exif_facts(img: Optional[Image.Image]) -> Dict[str, Any]:
    """Capture time and GPS position from EXIF, when the camera recorded them."""
    facts: Dict[str, Any] = {"captured_at": None, "lat": None, "lon": None}
    if img is None:
        return facts
    try:
        exif = img.getexif()
    except Exception:
        return facts
    if not exif:
        return facts
    sub = exif.get_ifd(0x8769) if hasattr(exif, "get_ifd") else {}
    raw_dt = sub.get(36867) or exif.get(306)  # DateTimeOriginal, else DateTime
    if raw_dt:
        try:
            facts["captured_at"] = datetime.strptime(str(raw_dt).strip(), "%Y:%m:%d %H:%M:%S")
        except ValueError:
            pass
    gps = exif.get_ifd(0x8825) if hasattr(exif, "get_ifd") else {}
    try:
        def _deg(v):
            d, m, s_ = (float(x) for x in v)
            return d + m / 60.0 + s_ / 3600.0
        if gps and 2 in gps and 4 in gps:
            lat = _deg(gps[2]) * (-1 if gps.get(1) in ("S", b"S") else 1)
            lon = _deg(gps[4]) * (-1 if gps.get(3) in ("W", b"W") else 1)
            facts["lat"], facts["lon"] = round(lat, 6), round(lon, 6)
    except Exception:
        pass
    return facts


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin(math.radians(lat2 - lat1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_METERS * math.asin(math.sqrt(a))


def scene_consistency(before: Optional[Image.Image], after: Optional[Image.Image]) -> Dict[str, Any]:
    """Are both photos of the same place? ORB keypoints + RANSAC homography inliers.

    Background structure (kerb, wall, grate, trees) survives a cleanup, so a genuine
    after-photo shares many geometrically consistent features with the before-photo.
    """
    if cv2 is None or before is None or after is None:
        return {"status": "UNVERIFIABLE", "inliers": 0, "detail": "Scene matcher unavailable."}

    def _gray(img: Image.Image) -> np.ndarray:
        g = ImageOps.exif_transpose(img).convert("L")
        scale = 800.0 / max(g.size)
        if scale < 1.0:
            g = g.resize((int(g.width * scale), int(g.height * scale)))
        return np.asarray(g)

    orb = cv2.ORB_create(nfeatures=2000)
    ka, da = orb.detectAndCompute(_gray(before), None)
    kb, db = orb.detectAndCompute(_gray(after), None)
    n_a, n_b = len(ka or []), len(kb or [])
    if da is None or db is None or min(n_a, n_b) < SCENE_MIN_KEYPOINTS:
        return {
            "status": "UNVERIFIABLE", "inliers": 0, "keypoints": [n_a, n_b],
            "detail": f"Too little texture to compare scenes ({min(n_a, n_b)} keypoints).",
        }
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m[0] for m in pairs if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    inliers = 0
    if len(good) >= 8:
        src = np.float32([ka[g.queryIdx].pt for g in good])
        dst = np.float32([kb[g.trainIdx].pt for g in good])
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
        inliers = int(mask.sum()) if mask is not None else 0
    if inliers >= SCENE_MATCH_INLIERS:
        status, detail = "MATCH", f"{inliers} geometrically consistent features: same location."
    elif inliers < SCENE_MISMATCH_INLIERS:
        status, detail = "MISMATCH", f"Only {inliers} consistent features: after-photo may show a different place."
    else:
        status, detail = "WEAK", f"{inliers} consistent features: likely same place from a different angle."
    return {"status": status, "inliers": inliers, "good_matches": len(good), "keypoints": [n_a, n_b], "detail": detail}


ALIGN_WIDTH = 640
ITEM_CLEARED_RATIO = 0.35     # an item is cleared if <= 35% of its plastic signature remains
ITEM_MIN_VISIBLE = 0.6        # share of an item's box that must be inside the after-photo


def _iou(a: List[float], b: List[float]) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def aligned_clearance(
    before: Optional[Image.Image],
    after: Optional[Image.Image],
    before_det: Dict[str, Any],
    after_det: Dict[str, Any],
) -> Dict[str, Any]:
    """Item-level clearance: warp the after-photo onto the before-photo and check every
    litter item at its original location.

    Items = YOLO plastic boxes + compact colour/sheen litter blobs from the before-photo.
    An item is cleared when its plastic signature has (almost) vanished at that spot and no
    YOLO plastic detection in the after-photo overlaps it. Litter that appears only in the
    after-photo counts against the contractor.
    """
    if cv2 is None or before is None or after is None:
        return {"aligned": False, "reason": "alignment unavailable"}

    def frame(img: Image.Image) -> Image.Image:
        img = ImageOps.exif_transpose(img).convert("RGB")
        return img.resize((ALIGN_WIDTH, max(1, int(round(img.height * ALIGN_WIDTH / img.width)))))

    fb, fa = frame(before), frame(after)
    orb = cv2.ORB_create(nfeatures=3000)
    kb, db = orb.detectAndCompute(cv2.cvtColor(np.asarray(fb), cv2.COLOR_RGB2GRAY), None)
    ka, da = orb.detectAndCompute(cv2.cvtColor(np.asarray(fa), cv2.COLOR_RGB2GRAY), None)
    if db is None or da is None:
        return {"aligned": False, "reason": "not enough texture to align"}
    good = [m[0] for m in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
            if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < 8:
        return {"aligned": False, "reason": "photos do not share enough features"}
    H, inl_mask = cv2.findHomography(np.float32([ka[g.queryIdx].pt for g in good]),
                                     np.float32([kb[g.trainIdx].pt for g in good]), cv2.RANSAC, 5.0)
    inliers = int(inl_mask.sum()) if inl_mask is not None else 0
    if H is None or inliers < SCENE_MATCH_INLIERS:
        return {"aligned": False, "reason": f"only {inliers} consistent features", "inliers": inliers}

    size = fb.size
    warped = Image.fromarray(cv2.warpPerspective(np.asarray(fa), H, size))
    valid = cv2.warpPerspective(np.ones((fa.height, fa.width), np.uint8), H, size) > 0
    mb = plastic_signature_mask(fb)
    ma = plastic_signature_mask(warped) * valid

    def scale_boxes(det: Dict[str, Any], target: Image.Image, homography=None) -> List[Dict[str, Any]]:
        fw, fh = det.get("frame_size") or [target.width, target.height]
        out = []
        for b in det.get("detected_items", []):
            if b.get("source") != "yolo":
                continue
            x0, y0, x1, y1 = (b["box"][0] * ALIGN_WIDTH / fw, b["box"][1] * target.height / fh,
                              b["box"][2] * ALIGN_WIDTH / fw, b["box"][3] * target.height / fh)
            if homography is not None:
                pts = cv2.perspectiveTransform(np.float32([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]]), homography)[:, 0, :]
                x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
                x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
            out.append({"box": [x0, y0, x1, y1], "class": b["class"], "source": "yolo"})
        return out

    before_yolo = scale_boxes(before_det, fb)
    after_yolo = scale_boxes(after_det, fa, H)

    items = list(before_yolo)
    for blob in litter_blobs(fb, mask=mb):
        if all(_iou(blob["box"], it["box"]) < 0.3 for it in items):
            items.append({"box": [float(v) for v in blob["box"]], "class": blob["class"], "source": "colour/sheen"})

    results, cleared_w, total_w = [], 0.0, 0.0
    for it in items:
        x0, y0, x1, y1 = (int(max(0, it["box"][0])), int(max(0, it["box"][1])),
                          int(min(size[0], it["box"][2])), int(min(size[1], it["box"][3])))
        if x1 <= x0 or y1 <= y0:
            continue
        visible = float(valid[y0:y1, x0:x1].mean())
        before_mass = int(mb[y0:y1, x0:x1].sum())
        after_mass = int(ma[y0:y1, x0:x1].sum())
        yolo_still = any(_iou(it["box"], ab["box"]) > 0.2 for ab in after_yolo)
        if visible < ITEM_MIN_VISIBLE:
            status = "not visible in after-photo"
        elif yolo_still:
            status = "remaining (detector)"
        elif before_mass >= 30 and after_mass > ITEM_CLEARED_RATIO * before_mass:
            status = "remaining"
        else:
            status = "cleared"
        w = CLASS_SEVERITY_WEIGHTS.get(it["class"], 1.0)
        total_w += w
        cleared_w += w if status == "cleared" else 0.0
        results.append({**it, "box": [x0, y0, x1, y1], "before_signature_px": before_mass,
                        "after_signature_px": after_mass, "visible_share": round(visible, 2), "status": status})

    # Litter present only in the after-photo
    new_items = []
    for blob in litter_blobs(warped, mask=ma):
        bx0, by0, bx1, by1 = blob["box"]
        if mb[by0:by1, bx0:bx1].sum() < ITEM_CLEARED_RATIO * blob["mass"] and all(_iou(blob["box"], r["box"]) < 0.3 for r in results):
            new_items.append({"box": blob["box"], "class": blob["class"], "source": "colour/sheen"})
    for ab in after_yolo:
        if all(_iou(ab["box"], r["box"]) < 0.2 for r in results):
            new_items.append({"box": [int(v) for v in ab["box"]], "class": ab["class"], "source": "yolo"})
    new_w = sum(CLASS_SEVERITY_WEIGHTS.get(n["class"], 1.0) for n in new_items)

    effectiveness = 100.0 * cleared_w / (total_w + new_w) if (total_w + new_w) > 0 else 0.0

    # Clearance map: before-photo with each item's verdict
    canvas = fb.copy()
    draw = ImageDraw.Draw(canvas)
    colors = {"cleared": "#22c55e", "remaining": "#ef4444", "remaining (detector)": "#ef4444", "not visible in after-photo": "#f59e0b"}
    for r in results:
        draw.rectangle(r["box"], outline=colors.get(r["status"], "#94a3b8"), width=4)
    for n in new_items:
        draw.rectangle(n["box"], outline="#e879f9", width=4)
    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=82)

    return {
        "aligned": True,
        "inliers": inliers,
        "items": results,
        "new_items": new_items,
        "items_total": len(results),
        "items_cleared": sum(1 for r in results if r["status"] == "cleared"),
        "effectiveness_pct": round(effectiveness, 1),
        "clearance_map_base64": base64.b64encode(buf.getvalue()).decode("utf-8"),
    }


def _weighted_load(items: List[Dict[str, Any]]) -> float:
    """Clog-weighted load: film and bottles block grates more than paper."""
    return sum(CLASS_SEVERITY_WEIGHTS.get(it.get("class", "other_plastic"), 1.0) * float(it.get("confidence", 0.5)) for it in items)


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
        site_lat: Optional[float] = None,
        site_lon: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run AI dual-inference and verify contractor clearance.

        Args:
            before_image: Pre-cleanup image (path, filename, bytes, or PIL).
            after_image: Post-cleanup clearance image.
            hotspot_id: Optional hotspot ID reference.
            site_name: Optional municipal location name.
            contractor_id: Municipal crew/contractor ID.
            operator_notes: Notes entered by operator or field crew.
            site_lat, site_lon: Site coordinate for the EXIF GPS check.

        Returns:
            Verification dossier containing quantified reductions, effectiveness %,
            an evidence checklist, PASS / FLAGGED FOR REVIEW / INCONCLUSIVE decision,
            annotated images, and provenance badges.
        """
        before_in, before_hint = self._resolve_image_input(before_image)
        after_in, after_hint = self._resolve_image_input(after_image)

        # 1. Run TACO inference on baseline before-photo
        before_det = detect_waste(before_in, filename_hint=before_hint, strict=True)
        # 2. Run TACO inference on post-cleanup clearance photo
        after_det = detect_waste(after_in, filename_hint=after_hint, strict=True)

        before_items = before_det.get("detected_items", [])
        after_items = after_det.get("detected_items", [])

        before_count = len(before_items)
        after_count = len(after_items)

        # Quantified object count reduction
        count_reduction = max(0, before_count - after_count)
        if before_count > 0:
            count_reduction_pct = round(max(0.0, (before_count - after_count) / before_count) * 100.0, 1)
        else:
            count_reduction_pct = 100.0 if after_count == 0 else 0.0

        # Severity comparison (0.0 - 5.0 scale)
        before_sev = float(before_det.get("severity", 3.5))
        after_sev = float(after_det.get("severity", 0.5))
        delta_severity = round(before_sev - after_sev, 2)
        sev_reduction_pct = round((max(0.0, delta_severity) / max(0.1, before_sev)) * 100.0, 1)

        # Clog-weighted load reduction (a remaining bag matters more than a remaining carton)
        before_load = _weighted_load(before_items)
        after_load = _weighted_load(after_items)
        load_reduction_pct = round(max(0.0, (before_load - after_load) / before_load) * 100.0, 1) if before_load > 0 else 0.0

        before_pil, after_pil = _load_pil(before_in), _load_pil(after_in)
        alignment = aligned_clearance(before_pil, after_pil, before_det, after_det)

        # 3. Cleanup Effectiveness Score (0 - 100%)
        if alignment.get("aligned"):
            # Same scene: judge every litter item at its own location (strongest evidence)
            effectiveness_score = alignment["effectiveness_pct"] if alignment["items_total"] else 0.0
            items_now = [r for r in alignment["items"] if r["status"] != "cleared"] + alignment["new_items"]
            before_items = alignment["items"]
            after_items = items_now
            before_count, after_count = alignment["items_total"], len(items_now)
            count_reduction = alignment["items_cleared"]
            count_reduction_pct = round(100.0 * count_reduction / before_count, 1) if before_count else 0.0
            load_reduction_pct = effectiveness_score
        elif before_count > 0 and after_count == 0 and after_sev <= 1.0:
            effectiveness_score = 98.5
        elif before_count == 0:
            effectiveness_score = 0.0
        else:
            raw_eff = (count_reduction_pct * 0.35) + (sev_reduction_pct * 0.30) + (load_reduction_pct * 0.35)
            effectiveness_score = round(max(0.0, min(100.0, raw_eff)), 1)

        # 4. Estimated Diverted Plastic Weight in kg
        # Scaled by count reduction and delta severity impact
        base_kg_per_item = 2.4
        diverted_kg = round(max(0.5, (count_reduction * base_kg_per_item) + (max(0.0, delta_severity) * 3.5)), 1)
        diverted_volume_liters = round(diverted_kg * 8.2, 0)

        # Per-class residual breakdown
        before_classes = [it.get("class", "plastic") for it in before_items]
        after_classes = [it.get("class", "plastic") for it in after_items]
        bc, ac = Counter(before_classes), Counter(after_classes)
        class_breakdown = [
            {"class": cls, "before": bc.get(cls, 0), "after": ac.get(cls, 0), "removed": max(0, bc.get(cls, 0) - ac.get(cls, 0))}
            for cls in sorted(set(bc) | set(ac))
        ]
        residual_film = ac.get("plastic_bag_wrapper", 0)
        residual_non_plastic = [it.get("coco_class", it.get("class")) for it in after_det.get("non_plastic_items", [])]

        # 5. Evidence checks
        checks: List[Dict[str, str]] = []

        def check(name: str, status: str, detail: str) -> None:
            checks.append({"check": name, "status": status, "detail": detail})

        bq, aq = before_det.get("image_quality", {}), after_det.get("image_quality", {})
        check("Before-photo quality", "FAIL" if bq.get("needs_review") else "PASS",
              ", ".join(bq.get("reasons", [])) or f"sharpness {bq.get('blur_variance', '–')}, luminance {bq.get('mean_luminance', '–')}")
        check("After-photo quality", "FAIL" if aq.get("needs_review") else "PASS",
              ", ".join(aq.get("reasons", [])) or f"sharpness {aq.get('blur_variance', '–')}, luminance {aq.get('mean_luminance', '–')}")
        check("Baseline waste present", "PASS" if before_count > 0 else "FAIL",
              f"{before_count} item(s) detected before cleanup" if before_count else "No waste detected in the before-photo: nothing to verify")

        scene = scene_consistency(before_pil, after_pil)
        check("Same-scene match", {"MATCH": "PASS", "MISMATCH": "FAIL"}.get(scene["status"], "WARN"), scene["detail"])

        bx, ax = _exif_facts(before_pil), _exif_facts(after_pil)
        if bx["captured_at"] and ax["captured_at"]:
            gap_h = (ax["captured_at"] - bx["captured_at"]).total_seconds() / 3600.0
            check("Capture order (EXIF)", "PASS" if gap_h >= 0 else "FAIL",
                  f"After-photo taken {gap_h:+.1f} h relative to before-photo")
        else:
            check("Capture order (EXIF)", "N/A", "Capture timestamps not present in both photos")
        if ax["lat"] is not None and site_lat is not None and site_lon is not None:
            dist = _haversine_m(ax["lat"], ax["lon"], site_lat, site_lon)
            check("After-photo GPS at site", "PASS" if dist <= GPS_MAX_DISTANCE_M else "FAIL",
                  f"Taken {dist:,.0f} m from the site (limit {GPS_MAX_DISTANCE_M:.0f} m)")
        else:
            check("After-photo GPS at site", "N/A", "No GPS tag in after-photo or site coordinate unknown")

        if alignment.get("aligned"):
            remaining = [r for r in alignment["items"] if r["status"] != "cleared"]
            check("Item-level clearance (aligned)", "PASS" if not remaining and not alignment["new_items"] else "FAIL",
                  f"{alignment['items_cleared']}/{alignment['items_total']} litter item(s) gone from their original spots"
                  + (f"; {len(remaining)} still there" if remaining else "")
                  + (f"; {len(alignment['new_items'])} new item(s) in after-photo" if alignment["new_items"] else "")
                  + f" ({alignment['inliers']} alignment features)")
        else:
            check("Item-level clearance (aligned)", "N/A",
                  f"Photos could not be aligned ({alignment.get('reason', 'n/a')}); judged on detection counts only")
        check("Clearance effectiveness", "PASS" if effectiveness_score >= PASS_THRESHOLD_PCT else "FAIL",
              f"{effectiveness_score:.1f}% vs {PASS_THRESHOLD_PCT:.0f}% threshold "
              f"({count_reduction_pct:.0f}% items, {load_reduction_pct:.0f}% clog-weighted load removed)")
        if residual_film:
            check("Residual film/bags", "WARN", f"{residual_film} plastic film item(s) still visible: highest grate-blinding risk")
        if residual_non_plastic:
            check("Non-plastic residue", "N/A", f"Ignored for plastic clearance: {', '.join(residual_non_plastic)}")
        if after_det.get("evidence_level") == "heuristic" and not alignment.get("aligned"):
            check("Residual detection basis", "WARN", "Remaining plastic found by colour/sheen heuristic only: confirm visually")

        # 6. Adjudication
        failed = {c["check"] for c in checks if c["status"] == "FAIL"}
        if failed & {"Before-photo quality", "After-photo quality", "Baseline waste present"}:
            decision_status = "INCONCLUSIVE"
            status_color = "#c084fc"
            status_message = "Evidence insufficient: re-shoot required before the cleanup can be judged."
        elif failed:
            decision_status = "FLAGGED FOR REVIEW"
            status_color = "#f59e0b"  # Amber warning
            status_message = "Failed: " + "; ".join(sorted(failed)) + ". Field re-inspection required before contractor payment."
        else:
            decision_status = "PASS"
            status_color = "#10b981"  # Emerald green
            status_message = f"Clearance criteria satisfied (>= {PASS_THRESHOLD_PCT:.0f}%) and evidence checks passed. Contractor invoice eligible for approval."
        invoice_eligible = decision_status == "PASS"

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
            "checks": checks,
            "scene_consistency": scene,
            "alignment": {k: v for k, v in alignment.items() if k != "clearance_map_base64"},
            "detection_trace": {"before": before_det.get("trace", []), "after": after_det.get("trace", [])},
            "image_quality": {"before": bq, "after": aq},
            "exif": {
                "before_captured_at": bx["captured_at"].isoformat() if bx["captured_at"] else None,
                "after_captured_at": ax["captured_at"].isoformat() if ax["captured_at"] else None,
                "after_gps": [ax["lat"], ax["lon"]] if ax["lat"] is not None else None,
            },
            "metrics": {
                "before_object_count": before_count,
                "after_object_count": after_count,
                "count_reduction": count_reduction,
                "count_reduction_pct": count_reduction_pct,
                "before_severity": before_sev,
                "after_severity": after_sev,
                "delta_severity": delta_severity,
                "severity_reduction_pct": sev_reduction_pct,
                "clog_weighted_load_reduction_pct": load_reduction_pct,
                "diverted_plastic_weight_kg": diverted_kg,
                "diverted_volume_liters": diverted_volume_liters,
                "before_classes": before_classes,
                "after_classes": after_classes,
                "class_breakdown": class_breakdown,
            },
            "visuals": {
                "before_annotated_base64": before_det.get("annotated_image_base64", ""),
                "after_annotated_base64": after_det.get("annotated_image_base64", ""),
                "clearance_map_base64": alignment.get("clearance_map_base64", ""),
            },
            "disclaimer": "Operator sign-off required for final contractor invoice clearance.",
            "provenance_badge": "[AI Inference / Modelled Estimate]",
            "provenance_category": "Inferred",
            "operator_notes": operator_notes or "",
        }


# ── Per-hotspot verification work orders (real contractor photos) ────────────
CASES_DIR = ROOT / "data" / "cleanup_verification"
CASES_MANIFEST = CASES_DIR / "cases.json"
PHOTO_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def _find_case_photo(case_id: str, kind: str) -> Optional[Path]:
    for ext in PHOTO_EXTS:
        for name in (f"{case_id}_{kind}{ext}", f"{case_id}_{kind}{ext.upper()}"):
            p = CASES_DIR / name
            if p.is_file():
                return p
    return None


def list_verification_cases() -> List[Dict[str, Any]]:
    """Verification cases with resolved photo paths. Both photos must come from the contractor."""
    import json

    if not CASES_MANIFEST.is_file():
        return []
    cases = json.loads(CASES_MANIFEST.read_text(encoding="utf-8")).get("cases", [])
    resolved = []
    for c in cases:
        before = _find_case_photo(c["case_id"], "before")
        after = _find_case_photo(c["case_id"], "after")
        if before and after:
            status = "READY"
        elif before:
            status = "AWAITING_AFTER_PHOTO"
        elif after:
            status = "AWAITING_BEFORE_PHOTO"
        else:
            status = "AWAITING_PHOTOS"
        resolved.append({
            **c,
            "before_path": str(before) if before else None,
            "after_path": str(after) if after else None,
            "status": status,
            "evidence_fingerprint": evidence_fingerprint(before, after) if before and after else None,
        })
    return resolved


def evidence_fingerprint(before: Path, after: Path) -> str:
    """SHA-1 over both photos: an override is only valid for the exact evidence it reviewed."""
    import hashlib

    h = hashlib.sha1()
    for p in (before, after):
        h.update(Path(p).read_bytes())
    return h.hexdigest()


MAX_CASE_PHOTO_BYTES = 15 * 1024 * 1024


def save_case_photo(case_id: str, kind: str, data: bytes, filename: str = "") -> Path:
    """Store (or replace) a contractor before/after photo for a case."""
    from backend.storage import locked, write_bytes_atomic

    if kind not in ("before", "after"):
        raise ValueError("kind must be 'before' or 'after'")
    if case_id not in {c["case_id"] for c in list_verification_cases()}:
        raise KeyError(case_id)
    if not data:
        raise ValueError("Empty upload")
    if len(data) > MAX_CASE_PHOTO_BYTES:
        raise ValueError(f"Photo larger than {MAX_CASE_PHOTO_BYTES // (1024 * 1024)} MB")
    try:
        Image.open(io.BytesIO(data)).verify()  # reject non-images early
    except Exception as exc:
        raise ValueError(f"Not a readable image ({exc}). HEIC photos must be exported as JPEG.") from exc
    ext = Path(filename).suffix.lower() if Path(filename).suffix.lower() in PHOTO_EXTS else ".jpg"
    target = CASES_DIR / f"{case_id}_{kind}{ext}"
    with locked(CASES_DIR / f"{case_id}.photos"):
        for old_ext in PHOTO_EXTS:
            old = CASES_DIR / f"{case_id}_{kind}{old_ext}"
            if old.is_file() and old != target:
                old.unlink()
        write_bytes_atomic(target, data)
    return target


def remove_case_photo(case_id: str, kind: str) -> bool:
    """Delete a case photo (e.g. to re-shoot). Returns True if a file was removed."""
    if kind not in ("before", "after"):
        raise ValueError("kind must be 'before' or 'after'")
    removed = False
    for ext in PHOTO_EXTS:
        p = CASES_DIR / f"{case_id}_{kind}{ext}"
        if p.is_file():
            p.unlink()
            removed = True
    return removed


# ── Human-in-the-loop override (auditable) ───────────────────────────────────
OVERRIDES_PATH = CASES_DIR / "verification_overrides.json"
OVERRIDE_DECISIONS = ("PASS", "FLAGGED FOR REVIEW")
OVERRIDE_REASONS = {
    "NON_PLASTIC_RESIDUE": "Remaining items are non-plastic (leaves, soil, stones, organic)",
    "DETECTOR_FALSE_POSITIVE": "AI flagged a shadow, reflection or background object as waste",
    "MINOR_RESIDUE_WITHIN_TOLERANCE": "Residue is minor and does not obstruct the drain inlet",
    "SCENE_VERIFIED_ON_SITE": "Inspector confirmed on site that both photos show the same location",
    "PHOTO_QUALITY_ACCEPTABLE": "Photo is clear enough on manual review despite the quality flag",
    "AI_MISSED_RESIDUE": "Inspector sees residual plastic the AI missed (downgrade to review)",
    "OTHER": "Other (explain in justification)",
}
MIN_JUSTIFICATION_CHARS = 20


def _load_overrides() -> List[Dict[str, Any]]:
    from backend.storage import read_json

    data = read_json(OVERRIDES_PATH, [])
    return data if isinstance(data, list) else []


def get_active_override(case_id: str, fingerprint: Optional[str]) -> Optional[Dict[str, Any]]:
    """Latest override for this case that still matches the current photos."""
    if not fingerprint:
        return None
    matches = [o for o in _load_overrides() if o.get("case_id") == case_id and o.get("evidence_fingerprint") == fingerprint]
    return matches[-1] if matches else None


def list_overrides(case_id: Optional[str] = None) -> List[Dict[str, Any]]:
    return [o for o in _load_overrides() if case_id is None or o.get("case_id") == case_id]


def record_override(
    case_id: str,
    ai_result: Dict[str, Any],
    fingerprint: str,
    new_decision: str,
    reason_code: str,
    justification: str,
    operator: str,
) -> Dict[str, Any]:
    """Validate and log an operator override of the AI verdict.

    Rules:
      - a written justification and a named operator are mandatory
      - INCONCLUSIVE (unusable evidence) cannot be overridden to PASS: re-shoot instead
      - a failed same-scene check can only be passed with SCENE_VERIFIED_ON_SITE
      - the override is bound to the exact photo pair (SHA-1); replacing a photo voids it
    """
    from datetime import datetime, timezone

    from backend.storage import locked, write_json

    new_decision = (new_decision or "").strip().upper()
    if new_decision not in OVERRIDE_DECISIONS:
        raise ValueError(f"Decision must be one of {OVERRIDE_DECISIONS}")
    if reason_code not in OVERRIDE_REASONS:
        raise ValueError("Unknown reason code")
    justification = (justification or "").strip()
    operator = (operator or "").strip()
    if len(justification) < MIN_JUSTIFICATION_CHARS:
        raise ValueError(f"Justification must be at least {MIN_JUSTIFICATION_CHARS} characters")
    if len(operator) < 2:
        raise ValueError("Operator name is required")
    ai_decision = ai_result.get("decision_status")
    if new_decision == ai_decision:
        raise ValueError("Override must change the AI decision")
    if ai_decision == "INCONCLUSIVE" and new_decision == "PASS":
        raise ValueError("INCONCLUSIVE evidence cannot be passed: upload a clearer photo instead")
    failed = {c["check"] for c in ai_result.get("checks", []) if c["status"] == "FAIL"}
    if new_decision == "PASS" and "Same-scene match" in failed and reason_code != "SCENE_VERIFIED_ON_SITE":
        raise ValueError("A failed same-scene check can only be passed with reason SCENE_VERIFIED_ON_SITE")

    entry = {
        "override_id": f"OVR-{case_id}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        "case_id": case_id,
        "evidence_fingerprint": fingerprint,
        "ai_decision": ai_decision,
        "ai_effectiveness_pct": ai_result.get("effectiveness_score"),
        "ai_failed_checks": sorted(failed),
        "override_decision": new_decision,
        "reason_code": reason_code,
        "reason_label": OVERRIDE_REASONS[reason_code],
        "justification": justification[:1000],
        "operator": operator[:80],
        "recorded_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    with locked(OVERRIDES_PATH):
        log = _load_overrides()
        log.append(entry)
        write_json(OVERRIDES_PATH, log)
    return entry


def effective_decision(ai_result: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Final decision after any valid human override."""
    if override:
        return {
            "decision": override["override_decision"],
            "source": "operator override",
            "invoice_eligible": override["override_decision"] == "PASS",
            "override": override,
        }
    return {
        "decision": ai_result.get("decision_status"),
        "source": "AI",
        "invoice_eligible": bool(ai_result.get("invoice_eligible")),
        "override": None,
    }
