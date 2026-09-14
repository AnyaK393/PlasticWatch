"""Prepare downloaded TACO COCO images for PlasticWatch's four-class YOLO experiment.

This script never modifies the official TACO folder. It copies usable images and
writes labels into ``data/taco_training`` (or a supplied output directory).
Run ``python data/prepare_taco_yolo.py --dry-run`` before downloading images to
audit the mapping. Run it without ``--dry-run`` only after TACO images exist.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS = ROOT / "data" / "taco" / "data" / "annotations.json"
DEFAULT_IMAGES = ROOT / "data" / "taco" / "data"
DEFAULT_MAPPING = ROOT / "data" / "taco_class_mapping.json"
DEFAULT_OUTPUT = ROOT / "data" / "taco_training"


def load_mapping(path: Path) -> tuple[list[str], dict[str, int], set[str]]:
    payload = json.loads(path.read_text())
    classes = payload["project_classes"]
    name_to_class: dict[str, int] = {}
    for index, class_name in enumerate(classes):
        for label in payload["mapping"].get(class_name, []):
            if label in name_to_class:
                raise ValueError(f"Category mapped more than once: {label}")
            name_to_class[label] = index
    excluded = set(payload.get("excluded_categories", []))
    overlap = excluded.intersection(name_to_class)
    if overlap:
        raise ValueError(f"Category cannot be mapped and excluded: {sorted(overlap)}")
    return classes, name_to_class, excluded


def find_image(images_root: Path, file_name: str) -> Path | None:
    direct = images_root / file_name
    if direct.is_file():
        return direct
    # TACO's downloader stores files in data/batch_*/; retain that convention.
    matches = list(images_root.glob(f"**/{Path(file_name).name}"))
    return matches[0] if matches else None


def yolo_box(bbox: list[float], image_width: int, image_height: int) -> tuple[float, float, float, float] | None:
    if image_width <= 0 or image_height <= 0 or len(bbox) != 4:
        return None
    x, y, width, height = (float(value) for value in bbox)
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(image_width), x + width), min(float(image_height), y + height)
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1 + x2) / 2 / image_width, (y1 + y2) / 2 / image_width, (x2 - x1) / image_width, (y2 - y1) / image_height)


def split_ids(image_ids: list[int], seed: int) -> dict[str, set[int]]:
    ids = list(image_ids)
    random.Random(seed).shuffle(ids)
    total = len(ids)
    train_end = round(total * 0.70)
    val_end = train_end + round(total * 0.20)
    return {"train": set(ids[:train_end]), "val": set(ids[train_end:val_end]), "test": set(ids[val_end:])}


def inspect(annotations: Path, images_root: Path, mapping_path: Path) -> tuple[dict, dict[int, int], list[str]]:
    payload = json.loads(annotations.read_text())
    classes, name_to_class, excluded = load_mapping(mapping_path)
    category_names = {category["id"]: category["name"] for category in payload["categories"]}
    category_to_class = {category_id: name_to_class[name] for category_id, name in category_names.items() if name in name_to_class}
    by_image: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
    counters = Counter()
    for annotation in payload["annotations"]:
        name = category_names.get(annotation["category_id"], "unknown")
        if name in excluded:
            counters["excluded_annotations"] += 1
        elif annotation["category_id"] in category_to_class:
            by_image[annotation["image_id"]].append((category_to_class[annotation["category_id"]], annotation["bbox"]))
            counters["mapped_annotations"] += 1
        else:
            counters["unmapped_annotations"] += 1
    image_by_id = {image["id"]: image for image in payload["images"]}
    found_ids = [image_id for image_id in by_image if find_image(images_root, image_by_id[image_id]["file_name"])]
    counters["metadata_images"] = len(payload["images"])
    counters["images_with_mapped_labels"] = len(by_image)
    counters["downloaded_usable_images"] = len(found_ids)
    counters["missing_downloaded_images"] = len(by_image) - len(found_ids)
    return {"payload": payload, "classes": classes, "by_image": by_image, "image_by_id": image_by_id}, dict(counters), found_ids


def prepare(args: argparse.Namespace) -> dict:
    context, counters, found_ids = inspect(args.annotations, args.images_root, args.mapping)
    report = {"classes": context["classes"], "seed": args.seed, "source_annotations": str(args.annotations), **counters}
    if args.dry_run:
        return report
    if not found_ids:
        raise RuntimeError("No downloaded TACO images found. Run the official data/taco/download.py first, then rerun preparation.")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output exists and is not empty: {args.output}. Choose a new path or pass --overwrite.")
    if args.overwrite and args.output.exists():
        shutil.rmtree(args.output)
    splits = split_ids(found_ids, args.seed)
    per_split = Counter()
    for split, ids in splits.items():
        for image_id in ids:
            image = context["image_by_id"][image_id]
            source = find_image(args.images_root, image["file_name"])
            assert source is not None
            target_image = args.output / "images" / split / source.name
            target_label = args.output / "labels" / split / f"{source.stem}.txt"
            target_image.parent.mkdir(parents=True, exist_ok=True)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for class_index, bbox in context["by_image"][image_id]:
                converted = yolo_box(bbox, image["width"], image["height"])
                if converted:
                    lines.append(f"{class_index} " + " ".join(f"{value:.6f}" for value in converted))
            if not lines:
                continue
            shutil.copy2(source, target_image)
            target_label.write_text("\n".join(lines) + "\n")
            per_split[split] += 1
    dataset_yaml = "\n".join([f"path: {args.output.resolve()}", "train: images/train", "val: images/val", "test: images/test", "names:"] + [f"  {index}: {name}" for index, name in enumerate(context["classes"])]) + "\n"
    (args.output / "dataset.yaml").write_text(dataset_yaml)
    report["prepared_images"] = dict(per_split)
    (args.output / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Delete only the selected output directory before writing.")
    args = parser.parse_args()
    report = prepare(args)
    print(json.dumps(report, indent=2))
    if args.dry_run and not report["downloaded_usable_images"]:
        print("\nPreparation is blocked only by missing image files; annotations and mapping were inspected successfully.")


if __name__ == "__main__":
    main()
