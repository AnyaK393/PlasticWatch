"""Plastic waste detection engine for urban runoff triage (YOLOv8 + plastic verification).

How a photo is analysed (every stage is recorded in the returned ``trace``):
  1. Quality gate     - resolution, exposure, haze/contrast, multi-scale blur and
                        directional motion-blur checks. Two tiers: "citizen" (lenient,
                        routes doubtful photos to operator review) and "evidence"
                        (strict, used for contractor cleanup verification).
  2. Object detection - YOLOv8n, COCO-pretrained (80 classes). NOTE: the weights are
                        not fine-tuned on TACO; TACO supplies the waste taxonomy and
                        class statistics (see ``model_card``).
  3. Class policy     - each COCO class is either a plastic candidate (bottle, cup,
                        cutlery, bag...), non-plastic waste (food/organic, paper) or
                        scene context that is ignored (people, vehicles, furniture...).
  4. Material check   - each plastic candidate crop is scored for plastic cues
                        (specular sheen, synthetic colour) vs natural cues (vegetation).
                        Only candidates whose combined plastic probability clears the
                        bar are counted as plastic.
  5. Fallback         - if no plastic object is confirmed, a colour/sheen heuristic
                        looks for weathered litter COCO cannot name (marked as
                        lower-evidence "heuristic" detections).
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from ultralytics import YOLO

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    import cv2
except ImportError:  # quality metrics fall back to numpy gradients
    cv2 = None

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
# Resolve weights against the repo root so uvicorn/streamlit work from any CWD
_WEIGHTS = ROOT / "yolov8n.pt"
_MODEL = YOLO(str(_WEIGHTS) if _WEIGHTS.is_file() else "yolov8n.pt")
TACO_ANNOTATIONS = ROOT / "data" / "taco" / "data" / "annotations.json"
TACO_MAPPING = ROOT / "data" / "taco_class_mapping.json"
TACO_TRAINING = ROOT / "data" / "taco_training"
TACO_SAMPLES_DIR = ROOT / "data" / "taco_samples"

# Refuse decompression bombs early (≈ 60 MP is far above any phone camera)
Image.MAX_IMAGE_PIXELS = 60_000_000

PLASTIC_CLASSES = ("plastic_bottle", "plastic_bag_wrapper", "other_plastic")

# ── Stage 3: COCO class policy ───────────────────────────────────────────────
# (role, project_class, plastic_prior, label). Roles:
#   plastic_candidate  - may be plastic litter; must pass the material check
#   non_plastic_waste  - litter, but not plastic (food/organic, paper)
#   anything not listed is scene context and ignored (people, cars, benches, beds...)
COCO_POLICY: Dict[int, Tuple[str, str, float, str]] = {
    39: ("plastic_candidate", "plastic_bottle", 0.80, "bottle"),
    41: ("plastic_candidate", "other_plastic", 0.70, "cup"),
    40: ("plastic_candidate", "other_plastic", 0.30, "wine glass (usually glass)"),
    42: ("plastic_candidate", "other_plastic", 0.55, "fork (disposable cutlery)"),
    43: ("plastic_candidate", "other_plastic", 0.45, "knife (disposable cutlery)"),
    44: ("plastic_candidate", "other_plastic", 0.55, "spoon (disposable cutlery)"),
    45: ("plastic_candidate", "other_plastic", 0.50, "bowl / food container"),
    24: ("plastic_candidate", "plastic_bag_wrapper", 0.40, "backpack-like sack"),
    26: ("plastic_candidate", "plastic_bag_wrapper", 0.50, "carry bag"),
    73: ("non_plastic_waste", "carton_paper", 0.0, "paper / carton"),
    46: ("non_plastic_waste", "organic", 0.0, "banana"),
    47: ("non_plastic_waste", "organic", 0.0, "apple"),
    48: ("non_plastic_waste", "organic", 0.0, "sandwich"),
    49: ("non_plastic_waste", "organic", 0.0, "orange"),
    50: ("non_plastic_waste", "organic", 0.0, "broccoli"),
    51: ("non_plastic_waste", "organic", 0.0, "carrot"),
    52: ("non_plastic_waste", "organic", 0.0, "hot dog"),
    53: ("non_plastic_waste", "organic", 0.0, "pizza"),
    54: ("non_plastic_waste", "organic", 0.0, "donut"),
    55: ("non_plastic_waste", "organic", 0.0, "cake"),
}
# Back-compat view used by older callers: COCO id -> project class (plastic candidates only)
COCO_TO_TACO = {cid: pol[1] for cid, pol in COCO_POLICY.items() if pol[0] == "plastic_candidate"}

YOLO_CANDIDATE_CONF = 0.15   # gather candidates
MIN_DETECTION_CONF = 0.25    # a candidate must reach this detector confidence
MIN_PLASTIC_PROBABILITY = 0.45
PRIOR_WEIGHT = 0.60          # plastic_prob = 0.60 * class prior + 0.40 * material score

# Color palette for detected bounding boxes
CLASS_COLORS = {
    "plastic_bottle": "#ef4444",      # Red - high clog danger
    "plastic_bag_wrapper": "#f97316", # Orange - high flood hazard
    "can_metal": "#eab308",           # Yellow - recyclable metal
    "carton_paper": "#06b6d4",        # Cyan - biodegradable paper/carton
    "other_plastic": "#a855f7",       # Purple - mixed polymer
    "organic": "#22c55e",             # Green - organic, not plastic
}

CLASS_SEVERITY_WEIGHTS = {
    "plastic_bottle": 1.25,
    "plastic_bag_wrapper": 1.40,      # Film clogs stormwater intake grates rapidly
    "can_metal": 0.85,
    "carton_paper": 0.70,
    "other_plastic": 1.00,
}

# ── Stage 1: quality gate ────────────────────────────────────────────────────
# Calibrated on the real photos in data/taco_samples (JPEG re-encoded): the edge-sharpness
# ratio (99.5th-percentile gradient at 640 px vs 160 px) is >= 0.64 for sharp photos and
# <= 0.50 after even a 1.5 px Gaussian blur; directional (motion) blur pushes edge
# isotropy below 0.55 while sharp photos sit at >= 0.73.
QUALITY_ANALYSIS_WIDTH = 640
QUALITY_TIERS: Dict[str, Dict[str, float]] = {
    "citizen": {   # lenient: doubtful photos go to the operator review queue
        "min_short_side_px": 240, "min_luminance": 35.0, "max_luminance": 235.0, "max_clipped_highlights": 0.45,
        "min_contrast": 12.0, "min_sharpness_ratio": 0.52, "min_edge_isotropy": 0.45, "min_scene_edges": 6.0,
    },
    "evidence": {  # strict: contractor before/after photos used for payment decisions
        "min_short_side_px": 480, "min_luminance": 45.0, "max_luminance": 215.0, "max_clipped_highlights": 0.25,
        "min_contrast": 20.0, "min_sharpness_ratio": 0.57, "min_edge_isotropy": 0.55, "min_scene_edges": 12.0,
    },
}
# Legacy constants kept for importers
DARK_MEAN_LUMINANCE_THRESHOLD = QUALITY_TIERS["citizen"]["min_luminance"]

REVIEW_STATUS = "PENDING_OPERATOR_VERIFICATION"

# Phone cameras deliver 12+ MP frames; YOLO runs at 640 px anyway, so cap the working
# copy and keep the annotated preview small enough to return quickly over a tunnel.
MAX_INFERENCE_SIDE = 1280
MAX_PREVIEW_SIDE = 960
MIN_IMAGE_SIDE = 32


def _open_upright(source: Any) -> Image.Image:
    """Open an image and apply its EXIF orientation (phone photos are often stored sideways)."""
    img = Image.open(source)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:  # corrupt EXIF must not block detection
        pass
    return img.convert("RGB")


def _cap_size(img: Image.Image, max_side: int) -> Image.Image:
    if max(img.size) <= max_side:
        return img
    scale = max_side / float(max(img.size))
    return img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)


def _gray(img: Image.Image, width: int) -> np.ndarray:
    g = img.convert("L")
    h = max(1, int(round(g.height * width / g.width)))
    return np.asarray(g.resize((width, h), Image.BILINEAR), dtype=np.float32)


def _gradients(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if cv2 is not None:
        return cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    gy, gx = np.gradient(a)
    return gx * 4.0, gy * 4.0


def evaluate_image_quality(
    img: Image.Image,
    strict: bool = False,
    original_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Quality gate run before any detection result is trusted.

    Args:
        img: RGB image.
        strict: use the "evidence" tier (contractor verification) instead of "citizen".
        original_size: (w, h) before any downscaling, for the resolution check.
    """
    tier_name = "evidence" if strict else "citizen"
    t = QUALITY_TIERS[tier_name]
    w, h = original_size or img.size

    fine = _gray(img, QUALITY_ANALYSIS_WIDTH)
    coarse = _gray(img, QUALITY_ANALYSIS_WIDTH // 4)
    mean_lum = float(fine.mean())
    contrast = float(fine.std())
    clipped = float((fine >= 250).mean())

    fx, fy = _gradients(fine)
    cx, cy = _gradients(coarse)
    coarse_edges = float(np.percentile(np.hypot(cx, cy), 98))
    # Blur flattens the strongest fine-scale edges much more than coarse-scale ones
    sharpness_ratio = float(np.percentile(np.hypot(fx, fy), 99.5)) / max(float(np.percentile(np.hypot(cx, cy), 99.5)), 1e-3)
    ax, ay = float(np.percentile(np.abs(fx), 98)), float(np.percentile(np.abs(fy), 98))
    isotropy = min(ax, ay) / max(ax, ay, 1e-3)

    featureless = coarse_edges < t["min_scene_edges"]
    checks = {
        "is_low_resolution": min(w, h) < t["min_short_side_px"],
        "is_dark": mean_lum < t["min_luminance"],
        "is_overexposed": mean_lum > t["max_luminance"] or clipped > t["max_clipped_highlights"],
        "is_low_contrast": contrast < t["min_contrast"],
        # Blur needs scene structure to measure; featureless frames are judged separately
        "is_blurry": (not featureless) and sharpness_ratio < t["min_sharpness_ratio"],
        "is_motion_blurred": (not featureless) and isotropy < t["min_edge_isotropy"],
        "is_featureless": featureless and strict,
    }
    labels = {
        "is_low_resolution": f"low resolution ({min(w, h)} px short side, need {t['min_short_side_px']:.0f})",
        "is_dark": f"too dark (luminance {mean_lum:.0f}, need {t['min_luminance']:.0f}+)",
        "is_overexposed": f"over-exposed (luminance {mean_lum:.0f}, {clipped:.0%} clipped highlights)",
        "is_low_contrast": f"hazy / low contrast (contrast {contrast:.0f}, need {t['min_contrast']:.0f}+)",
        "is_blurry": f"blurry / out of focus (sharpness {sharpness_ratio:.2f}, need {t['min_sharpness_ratio']:.2f}+)",
        "is_motion_blurred": f"motion blur (edge isotropy {isotropy:.2f}, need {t['min_edge_isotropy']:.2f}+)",
        "is_featureless": "no identifiable scene detail (cannot prove location or contents)",
    }
    reasons = [labels[k] for k, bad in checks.items() if bad]
    return {
        **checks,
        "needs_review": bool(reasons),
        "reasons": reasons,
        "tier": tier_name,
        "thresholds": t,
        "mean_luminance": round(mean_lum, 1),
        "contrast": round(contrast, 1),
        "clipped_highlights": round(clipped, 3),
        "sharpness_ratio": round(sharpness_ratio, 2),
        "edge_isotropy": round(isotropy, 2),
        "scene_edge_strength": round(coarse_edges, 1),
        # legacy key (dashboard & older reports display it)
        "blur_variance": round(sharpness_ratio, 2),
        "resolution": [int(w), int(h)],
    }


# ── Stage 4: material check ──────────────────────────────────────────────────
def _hsv(img: Image.Image) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    return hsv[..., 0] * 360.0, hsv[..., 1], hsv[..., 2]


def material_scores(crop: Image.Image) -> Dict[str, float]:
    """Plastic vs natural cues inside a detection box."""
    small = crop.convert("RGB").resize((64, 64))
    hue, sat, val = _hsv(small)
    vegetation = float(((hue >= 60) & (hue <= 170) & (sat > 0.25) & (val > 0.2)).mean())
    sheen = float(((sat < 0.18) & (val > 0.80)).mean())          # specular highlights / translucent film
    vivid = float(((sat > 0.5) & (val > 0.4) & ~((hue >= 60) & (hue <= 170))).mean())  # printed labels, dyed polymer
    # Clear PET lets background vegetation show through, so vegetation is a mild penalty
    material = max(0.0, min(1.0, 0.25 + 1.6 * sheen + 1.4 * vivid - 0.8 * vegetation))
    return {"vegetation": round(vegetation, 3), "sheen": round(sheen, 3), "vivid": round(vivid, 3), "material_score": round(material, 3)}


LITTER_ANALYSIS_WIDTH = 640
MIN_LITTER_MASS_FRAC = 0.0015   # an item must cover >= 0.15% of the frame (specks ignored)
MAX_LITTER_SPAN_FRAC = 0.30     # wider/taller structures are scenery (fences, kerbs, rails)
MAX_LITTER_ASPECT = 4.0         # long thin strips (fence bars, poles) are scenery


def plastic_signature_mask(img: Image.Image) -> np.ndarray:
    """Pixels that look like plastic: vivid non-vegetation colour or bright low-saturation sheen/film."""
    hue, sat, val = _hsv(img)
    vegetation = (hue >= 60.0) & (hue <= 170.0)
    mask = (((sat > 0.55) & (val > 0.40) & ~vegetation) | ((sat < 0.18) & (val > 0.85))).astype(np.uint8)
    if cv2 is not None:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def litter_blobs(img: Image.Image, ignore_mask: Optional[np.ndarray] = None, mask: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    """Compact plastic-signature items, in the coordinates of ``img``.

    Neighbouring fragments are joined (morphological close) so that a fence mesh becomes one
    large structure and is rejected by the span/aspect limits, while bottles, bags, cans and
    wrappers remain as compact items.
    """
    if cv2 is None:
        return []
    if mask is None:
        mask = plastic_signature_mask(img)
    if ignore_mask is not None:
        mask = mask * (~ignore_mask).astype(np.uint8)
    h_, w_ = mask.shape
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    hue, sat, val = _hsv(img) if img is not None else (None, None, None)
    out = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        component = labels == i
        mass = int(mask[component].sum())
        if mass < MIN_LITTER_MASS_FRAC * h_ * w_:
            continue
        if w > MAX_LITTER_SPAN_FRAC * w_ or h > MAX_LITTER_SPAN_FRAC * h_ or max(w, h) / max(1, min(w, h)) > MAX_LITTER_ASPECT:
            continue
        cls = "other_plastic"
        if hue is not None:
            px = component & (mask > 0)
            vivid = px & (sat > 0.55)
            cyan = vivid & (hue >= 170.0) & (hue <= 250.0)
            if vivid.sum() and cyan.sum() >= 0.5 * vivid.sum():
                cls = "plastic_bottle"
            elif (px & (sat < 0.18)).sum() >= 0.5 * px.sum():
                cls = "plastic_bag_wrapper"
        out.append({"box": [int(x), int(y), int(x + w), int(y + h)], "mass": mass, "mass_frac": mass / float(h_ * w_), "class": cls})
    out.sort(key=lambda b: -b["mass"])
    return out


def _analysis_frame(img: Image.Image) -> Image.Image:
    h = max(1, int(round(img.height * LITTER_ANALYSIS_WIDTH / img.width)))
    return img.convert("RGB").resize((LITTER_ANALYSIS_WIDTH, h))


def _check_synthetic_debris_heuristic(img: Image.Image, ignore_mask: Optional[np.ndarray] = None) -> Tuple[bool, str, float]:
    """Back-compat wrapper: (hit, dominant class, confidence) from compact litter items."""
    items = _heuristic_items(img, ignore_mask)
    if not items:
        return False, "none", 0.0
    return True, items[0]["class"], items[0]["confidence"]


def _heuristic_items(img: Image.Image, ignore_mask: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    """Colour/sheen fallback for weathered litter that YOLO's COCO classes cannot name.

    Returns real boxes (in ``img`` coordinates) for compact plastic-signature items; clean
    surfaces, foliage, fences and large uniform objects produce none.
    ``ignore_mask`` (120x160, True = ignore) removes people/vehicles found by YOLO.
    """
    frame = _analysis_frame(img)
    ign = None
    if ignore_mask is not None and cv2 is not None:
        ign = cv2.resize(ignore_mask.astype(np.uint8), frame.size, interpolation=cv2.INTER_NEAREST).astype(bool)
    sx, sy = img.width / frame.width, img.height / frame.height
    items = []
    for b in litter_blobs(frame, ign)[:6]:
        x0, y0, x1, y1 = b["box"]
        items.append({
            "class": b["class"],
            "box": [round(x0 * sx, 1), round(y0 * sy, 1), round(x1 * sx, 1), round(y1 * sy, 1)],
            "confidence": round(min(0.55, 0.36 + b["mass_frac"] * 20.0), 2),
            "mass_frac": round(b["mass_frac"], 4),
        })
    return items


def dataset_status(path: Path = TACO_ANNOTATIONS) -> dict:
    """Return dataset facts for provenance UI; never claim this is an opaque model."""
    if not path.is_file():
        return {"available": False, "message": "TACO annotations not downloaded"}
    try:
        payload = json.loads(path.read_text())
        categories = [item["name"] for item in payload["categories"]]
        plastic_labels = [label for label in categories if "plastic" in label.lower() or "bag" in label.lower()]
        image_root = path.parent
        downloaded_images = [item for item in image_root.rglob("*") if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        sample_images = list(TACO_SAMPLES_DIR.glob("*.jpg")) if TACO_SAMPLES_DIR.is_dir() else []
        trained_model = next(iter(TACO_TRAINING.glob("**/best.pt")), None)
        mapping = json.loads(TACO_MAPPING.read_text()) if TACO_MAPPING.is_file() else None
        return {
            "available": True,
            "images": len(payload.get("images", [])),
            "annotations": len(payload.get("annotations", [])),
            "categories": len(categories),
            "plastic_related_labels": len(plastic_labels),
            "mapping_ready": mapping is not None,
            "target_classes": mapping.get("project_classes", []) if mapping else [],
            "downloaded_images": len(downloaded_images),
            "sample_images": len(sample_images),
            "prepared_dataset": (TACO_TRAINING / "dataset.yaml").is_file(),
            "trained_model": str(trained_model) if trained_model else None,
            "model_state": "TACO fine-tuned" if trained_model else "COCO-pretrained YOLOv8n + TACO class mapping (not fine-tuned)",
            "license": "TACO dataset: CC BY 4.0; retain attribution for dataset use.",
        }
    except Exception as exc:
        logger.error("Failed to parse TACO dataset status: %s", exc)
        return {"available": False, "message": str(exc)}


def get_sample_images() -> List[Dict[str, Any]]:
    """Return catalog of available sample images for citizen report simulation."""
    if not TACO_SAMPLES_DIR.is_dir():
        return []
    samples = []
    descriptions = {
        "sample_bottles_drain.jpg": ("Discarded PET bottles accumulated at street curb", "plastic_bottle", 4.6),
        "sample_plastic_bag_curb.jpg": ("Polypropylene film blocking stormwater grate", "plastic_bag_wrapper", 4.8),
        "sample_beverage_cans_gutter.jpg": ("Aluminium beverage cans in gutter line", "can_metal", 3.4),
        "sample_food_wrappers_culvert.jpg": ("Food snack wrappers near culvert inlet", "plastic_bag_wrapper", 3.9),
        "sample_mixed_waste_grate.jpg": ("Mixed municipal waste & single-use packaging", "other_plastic", 4.3),
    }
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = [p for p in sorted(TACO_SAMPLES_DIR.iterdir()) if p.is_file() and p.suffix.lower() in valid_exts]
    
    for p in image_paths:
        if p.name in descriptions:
            desc, primary_class, severity = descriptions[p.name]
        else:
            # Auto-infer from filename for user-added custom images
            name_lower = p.stem.lower()
            if "bottle" in name_lower:
                primary_class = "plastic_bottle"
                severity = 4.4
                desc = "User-added sample: Plastic bottle accumulation"
            elif "bag" in name_lower or "film" in name_lower:
                primary_class = "plastic_bag_wrapper"
                severity = 4.7
                desc = "User-added sample: Plastic bag / film blockage"
            elif "can" in name_lower or "metal" in name_lower:
                primary_class = "can_metal"
                severity = 3.6
                desc = "User-added sample: Metal beverage cans in runoff"
            elif "wrapper" in name_lower or "snack" in name_lower:
                primary_class = "plastic_bag_wrapper"
                severity = 4.0
                desc = "User-added sample: Food wrappers near drain"
            else:
                primary_class = "other_plastic"
                severity = 3.8
                desc = f"Custom field sample: {p.stem.replace('_', ' ').title()}"

        samples.append({
            "filename": p.name,
            "path": str(p),
            "description": desc,
            "primary_class": primary_class,
            "default_severity": severity,
            "size_kb": round(p.stat().st_size / 1024, 1),
        })
    return samples


@lru_cache(maxsize=1)
def taco_annotation_stats() -> Dict[str, Any]:
    """Statistics computed from the real TACO annotation file and the project class mapping."""
    if not TACO_ANNOTATIONS.is_file() or not TACO_MAPPING.is_file():
        return {"available": False}
    payload = json.loads(TACO_ANNOTATIONS.read_text())
    mapping = json.loads(TACO_MAPPING.read_text())
    cat_name = {c["id"]: c["name"] for c in payload["categories"]}
    name_to_project = {lab: cls for cls, labels in mapping["mapping"].items() for lab in labels}
    images = {im["id"]: im for im in payload["images"]}
    per_class: Dict[str, Dict[str, Any]] = {}
    unmapped = 0
    for ann in payload["annotations"]:
        cls = name_to_project.get(cat_name.get(ann["category_id"], ""))
        if cls is None:
            unmapped += 1
            continue
        im = images.get(ann["image_id"], {})
        area_frac = ann.get("area", 0.0) / max(1.0, float(im.get("width", 1)) * float(im.get("height", 1)))
        d = per_class.setdefault(cls, {"annotations": 0, "images": set(), "area_fracs": [], "taco_labels": set()})
        d["annotations"] += 1
        d["images"].add(ann["image_id"])
        d["area_fracs"].append(area_frac)
        d["taco_labels"].add(cat_name.get(ann["category_id"]))
    total_mapped = sum(d["annotations"] for d in per_class.values())
    rows = []
    for cls in mapping["project_classes"]:
        d = per_class.get(cls)
        if not d:
            continue
        rows.append({
            "project_class": cls,
            "is_plastic": cls in PLASTIC_CLASSES,
            "taco_annotations": d["annotations"],
            "share_pct": round(100.0 * d["annotations"] / max(1, total_mapped), 1),
            "images_containing": len(d["images"]),
            "median_box_area_pct": round(100.0 * float(np.median(d["area_fracs"])), 2),
            "taco_labels_mapped": len(d["taco_labels"]),
        })
    plastic_share = sum(r["taco_annotations"] for r in rows if r["is_plastic"]) / max(1, total_mapped)
    return {
        "available": True,
        "images": len(payload["images"]),
        "annotations": len(payload["annotations"]),
        "categories": len(payload["categories"]),
        "mapped_annotations": total_mapped,
        "unmapped_annotations": unmapped,
        "plastic_share_pct": round(100.0 * plastic_share, 1),
        "per_class": rows,
    }


@lru_cache(maxsize=1)
def model_card() -> Dict[str, Any]:
    """Facts about the detector actually running (read from the loaded weights)."""
    net = getattr(_MODEL, "model", None)
    try:
        params = int(sum(p.numel() for p in net.parameters())) if net is not None else None
    except Exception:
        params = None
    weights = _WEIGHTS if _WEIGHTS.is_file() else None
    sha = hashlib.sha1(weights.read_bytes()).hexdigest()[:12] if weights else None
    names = getattr(_MODEL, "names", {}) or {}
    trained = next(iter(TACO_TRAINING.glob("**/best.pt")), None)
    return {
        "architecture": "YOLOv8n (Ultralytics)",
        "weights_file": weights.name if weights else "yolov8n.pt (downloaded)",
        "weights_sha1": sha,
        "weights_mb": round(weights.stat().st_size / 1e6, 1) if weights else None,
        "parameters": params,
        "pretraining": "COCO 2017 (80 classes)",
        "taco_fine_tuned": trained is not None,
        "taco_fine_tune_note": (
            f"Fine-tuned weights found at {trained}" if trained else
            "Not fine-tuned on TACO yet: TACO provides the taxonomy, class mapping and statistics. "
            "data/prepare_taco_yolo.py converts TACO to YOLO format once images are downloaded."
        ),
        "num_model_classes": len(names),
        "input_size_px": 640,
        "candidate_confidence": YOLO_CANDIDATE_CONF,
        "min_detection_confidence": MIN_DETECTION_CONF,
        "min_plastic_probability": MIN_PLASTIC_PROBABILITY,
        "plastic_probability_formula": f"{PRIOR_WEIGHT:.2f} x class prior + {1 - PRIOR_WEIGHT:.2f} x material score",
        "class_policy": [
            {"coco_id": cid, "coco_class": names.get(cid, label), "role": role, "maps_to": cls, "plastic_prior": prior}
            for cid, (role, cls, prior, label) in sorted(COCO_POLICY.items())
        ],
        "quality_tiers": QUALITY_TIERS,
    }


def _resolve_input(image_input: Union[bytes, str, Path, Image.Image], filename_hint: str) -> Image.Image:
    if isinstance(image_input, (bytes, bytearray)):
        if not image_input:
            raise ValueError("Empty image payload")
        return _open_upright(io.BytesIO(image_input))
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, (str, Path)):
        if isinstance(image_input, str) and image_input.startswith("data:image"):
            b64_data = image_input.split(",", 1)[1]
            return _open_upright(io.BytesIO(base64.b64decode(b64_data)))
        if Path(image_input).is_file():
            return _open_upright(Path(image_input))
        sample_path = TACO_SAMPLES_DIR / Path(filename_hint or "").name if filename_hint else None
        if sample_path is not None and sample_path.is_file():
            return _open_upright(sample_path)
        raise FileNotFoundError(f"Image not found: {image_input}")
    raise ValueError(f"Unsupported image input type: {type(image_input)}")


def detect_waste(
    image_input: Union[bytes, str, Path, Image.Image],
    filename_hint: str = "",
    strict: bool = False,
) -> Dict[str, Any]:
    """Detect plastic waste in a photo and explain every decision.

    Args:
        image_input: bytes, path, data-URI string or PIL image.
        filename_hint: sample filename fallback when a path is not found.
        strict: apply the evidence-grade quality tier (cleanup verification).
    """
    img = _resolve_input(image_input, filename_hint)
    if min(img.size) < MIN_IMAGE_SIDE:
        raise ValueError(f"Image too small ({img.width}x{img.height})")
    original_size = img.size
    img = _cap_size(img, MAX_INFERENCE_SIDE)

    # 1. Quality gate (before annotation boxes are drawn onto the frame)
    quality = evaluate_image_quality(img, strict=strict, original_size=original_size)

    # 2. YOLO candidates
    results = _MODEL(img, conf=YOLO_CANDIDATE_CONF, verbose=False)
    boxes_data = results[0].boxes
    names = results[0].names

    clean = img.copy()  # unannotated copy for material crops
    draw = ImageDraw.Draw(img)
    detected_boxes: List[Dict[str, Any]] = []
    non_plastic: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []
    ignore = np.zeros((120, 160), dtype=bool)
    W, H = clean.size

    for box in boxes_data:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        xyxy = [round(c, 1) for c in box.xyxy[0].tolist()]
        coco_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
        role, project_cls, prior, _label = COCO_POLICY.get(cls_id, ("context", None, 0.0, coco_name))
        entry = {"coco_class": coco_name, "detector_confidence": round(conf, 2), "box": xyxy, "role": role}

        if role == "context":
            # Remember where people/vehicles/furniture are so the colour heuristic ignores them
            x0, y0, x1, y1 = (int(xyxy[0] / W * 160), int(xyxy[1] / H * 120), int(xyxy[2] / W * 160), int(xyxy[3] / H * 120))
            ignore[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
            trace.append({**entry, "decision": "ignored", "reason": "scene context, not litter"})
            continue
        if conf < MIN_DETECTION_CONF:
            trace.append({**entry, "decision": "rejected", "reason": f"detector confidence {conf:.2f} < {MIN_DETECTION_CONF}"})
            continue
        if role == "non_plastic_waste":
            non_plastic.append({"class": project_cls, "coco_class": coco_name, "confidence": round(conf, 2), "box": xyxy})
            draw.rectangle(xyxy, outline=CLASS_COLORS.get(project_cls, "#22c55e"), width=2)
            trace.append({**entry, "decision": "non-plastic waste", "reason": f"{coco_name} is {project_cls}, excluded from plastic load"})
            continue

        # Plastic candidate: verify material inside the box (shrink 10% to avoid background)
        bw, bh = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
        crop_box = (int(xyxy[0] + 0.1 * bw), int(xyxy[1] + 0.1 * bh), int(xyxy[2] - 0.1 * bw), int(xyxy[3] - 0.1 * bh))
        mat = material_scores(clean.crop(crop_box)) if crop_box[2] - crop_box[0] > 4 and crop_box[3] - crop_box[1] > 4 else {"material_score": 0.25}
        plastic_prob = round(PRIOR_WEIGHT * prior + (1 - PRIOR_WEIGHT) * mat["material_score"], 2)
        entry.update({"plastic_prior": prior, "material": mat, "plastic_probability": plastic_prob})
        if plastic_prob < MIN_PLASTIC_PROBABILITY:
            trace.append({**entry, "decision": "rejected", "reason": f"plastic probability {plastic_prob:.2f} < {MIN_PLASTIC_PROBABILITY}"})
            continue

        detected_boxes.append({
            "class": project_cls,
            "confidence": round(conf, 2),
            "plastic_probability": plastic_prob,
            "box": xyxy,
            "source": "yolo",
        })
        draw.rectangle(xyxy, outline=CLASS_COLORS.get(project_cls, "#ef4444"), width=4)
        trace.append({**entry, "decision": "accepted as plastic", "reason": f"{project_cls}, plastic probability {plastic_prob:.2f}"})

    # 5. Heuristic fallback for weathered litter COCO cannot name
    if not detected_boxes:
        for item in _heuristic_items(clean, ignore):
            detected_boxes.append({
                "class": item["class"],
                "confidence": item["confidence"],
                "plastic_probability": item["confidence"],
                "box": item["box"],
                "source": "heuristic",
            })
            draw.rectangle(item["box"], outline=CLASS_COLORS.get(item["class"], "#ef4444"), width=4)
            trace.append({"coco_class": "—", "role": "heuristic", "box": item["box"], "decision": "accepted as plastic (heuristic)",
                          "reason": f"compact {item['class']} colour/sheen item covering {item['mass_frac']:.1%} of frame"})

    detected_classes = [b["class"] for b in detected_boxes]
    confidences = [b["confidence"] for b in detected_boxes]
    item_count = len(detected_classes)
    mean_conf = float(np.mean(confidences)) if item_count > 0 else 0.0
    total_severity = sum(CLASS_SEVERITY_WEIGHTS.get(b["class"], 1.0) * b["confidence"] for b in detected_boxes)
    severity_score = min(1.0, total_severity / 2.0) if item_count > 0 else 0.0
    evidence_level = (
        "none" if not item_count else ("heuristic" if all(b["source"] == "heuristic" for b in detected_boxes) else "model")
    )

    buf = io.BytesIO()
    _cap_size(img, MAX_PREVIEW_SIDE).save(buf, format="JPEG", quality=80)
    raw_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    annotated_b64 = "data:image/jpeg;base64," + raw_b64

    hazard = "HIGH" if severity_score > 0.5 else ("MEDIUM" if severity_score > 0.25 else "LOW")
    needs_review = quality["needs_review"]
    if not needs_review:
        review_message = ""
    elif quality["is_blurry"] or quality["is_dark"] or quality["is_motion_blurred"]:
        review_message = "⚠️ Photo appears blurry or dark. Submitted to Municipal Operator Review Queue."
    else:
        review_message = (
            f"⚠️ Photo is unclear ({quality['reasons'][0].split(' (')[0]}). Submitted to Municipal Operator Review Queue."
        )

    return {
        "status": REVIEW_STATUS if needs_review else "success",
        "waste_detected": item_count > 0,
        "item_count": item_count,
        "detected_classes": detected_classes,
        "dominant_class": detected_classes[0] if item_count > 0 else "none",
        "confidence": round(mean_conf, 2),
        "severity_score": round(severity_score, 2),
        "drain_choke_hazard": hazard,
        "boxes": detected_boxes,
        "non_plastic_items": non_plastic,
        "evidence_level": evidence_level,
        "trace": trace,
        "annotated_image": annotated_b64,
        "frame_size": [img.width, img.height],
        "model_provenance": "YOLOv8n (COCO-pretrained) + plastic class policy + material check",
        # Image quality guard
        "image_quality": quality,
        "needs_review": needs_review,
        "review_message": review_message,
        # Legacy schema consumed by pipeline, verification engine and dashboard
        "detected_items": detected_boxes,
        "mean_confidence": round(mean_conf, 2),
        "severity": round(severity_score * 5.0, 2),
        "hazard_level": hazard,
        "annotated_image_base64": raw_b64,
    }
