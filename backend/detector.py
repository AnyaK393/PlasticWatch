# backend/detector.py
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Load lightweight model (downloads once and caches locally in ~/.config/Ultralytics)
MODEL = YOLO("yolov8n.pt")

# Standard COCO classes related to municipal waste / litter
WASTE_CLASSES = {
    39: ("bottle", 1.5),      # plastic bottles have high blockage risk
    41: ("cup", 1.2),
    42: ("fork", 0.8),
    43: ("knife", 0.8),
    44: ("spoon", 0.8),
    45: ("bowl", 1.0),
    46: ("banana", 0.5),     # organic
    47: ("apple", 0.5),
    48: ("sandwich", 0.5),
    77: ("cell phone", 0.5),
}

def detect_waste(image_path: Path | str):
    """
    Runs YOLO inference on an image.
    Returns:
        detected (bool)
        waste_count (int)
        severity_score (float 0-1)
        labels (list of str)
    """
    results = MODEL(image_path, verbose=False)
    boxes = results[0].boxes
    
    detected_items = []
    total_weight = 0.0
    
    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        if conf >= 0.25 and cls_id in WASTE_CLASSES:
            label, weight = WASTE_CLASSES[cls_id]
            detected_items.append(label)
            total_weight += weight * conf

    # If general waste not found in standard classes but trash detected
    is_waste = len(detected_items) > 0
    severity = min(1.0, total_weight / 3.0) if is_waste else 0.0

    return {
        "waste_detected": is_waste,
        "item_count": len(detected_items),
        "items": detected_items,
        "severity": round(severity, 2),
        "confidence": round(float(boxes.conf.mean().item()), 2) if len(boxes) > 0 else 0.0
    }