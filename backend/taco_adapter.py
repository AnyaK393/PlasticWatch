"""Small, dependency-free reader for the official TACO COCO annotations."""
from __future__ import annotations

import json
from pathlib import Path


TACO_ANNOTATIONS = Path(__file__).parent.parent / "data" / "taco" / "data" / "annotations.json"
TACO_MAPPING = Path(__file__).parent.parent / "data" / "taco_class_mapping.json"
TACO_TRAINING = Path(__file__).parent.parent / "data" / "taco_training"


def dataset_status(path: Path = TACO_ANNOTATIONS) -> dict:
    """Return dataset facts for provenance UI; never claim this is a trained model."""
    if not path.is_file():
        return {"available": False, "message": "TACO annotations not downloaded"}
    payload = json.loads(path.read_text())
    categories = [item["name"] for item in payload["categories"]]
    plastic_labels = [label for label in categories if "plastic" in label.lower() or "bag" in label.lower()]
    image_root = path.parent
    downloaded_images = [item for item in image_root.rglob("*") if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    trained_model = next(iter(TACO_TRAINING.glob("**/best.pt")), None)
    mapping = json.loads(TACO_MAPPING.read_text()) if TACO_MAPPING.is_file() else None
    return {
        "available": True,
        "images": len(payload["images"]),
        "annotations": len(payload["annotations"]),
        "categories": len(categories),
        "plastic_related_labels": len(plastic_labels),
        "mapping_ready": mapping is not None,
        "target_classes": mapping.get("project_classes", []) if mapping else [],
        "downloaded_images": len(downloaded_images),
        "prepared_dataset": (TACO_TRAINING / "dataset.yaml").is_file(),
        "trained_model": str(trained_model) if trained_model else None,
        "model_state": "trained" if trained_model else "not trained",
        "license": "TACO dataset: CC BY 4.0; retain attribution for dataset use.",
    }
