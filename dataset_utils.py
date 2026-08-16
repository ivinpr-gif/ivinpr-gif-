"""
Dataset Validation & Inspection Utilities for Waste AI Computer-Vision Pipeline.
Verifies YOLO dataset directory structure, label formatting, normalization, and class balance.
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Tuple

def validate_yolo_dataset(yaml_path: str) -> Dict[str, any]:
    """
    Validates a YOLO dataset configuration and label file structure.
    Checks data.yaml, image/label counts, label format [class_id x_center y_center w h],
    bounding box normalization range [0..1], and class distribution.
    """
    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        print(f"[ERROR] Dataset configuration file not found at: {yaml_path}")
        return {"valid": False, "error": f"Missing config file {yaml_path}"}

    with open(yaml_file, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except Exception as e:
            return {"valid": False, "error": f"YAML syntax error: {e}"}

    base_dir = yaml_file.parent
    class_names = config.get("names", {})
    if isinstance(class_names, list):
        class_map = {i: name for i, name in enumerate(class_names)}
    elif isinstance(class_names, dict):
        class_map = {int(k): v for k, v in class_names.items()}
    else:
        return {"valid": False, "error": "Invalid class 'names' section in data.yaml"}

    stats = {
        "dataset_name": yaml_file.name,
        "num_classes": len(class_map),
        "classes": class_map,
        "splits": {},
        "class_counts": {cid: 0 for cid in class_map.keys()},
        "total_images": 0,
        "total_labels": 0,
        "corrupted_files": 0,
        "valid": True
    }

    print("=" * 60)
    print(f"AUDITING WASTE DATASET: {yaml_path}")
    print(f"Total Configured Classes: {len(class_map)}")
    print("=" * 60)

    splits = ["train", "val", "test"]
    for split in splits:
        split_rel = config.get(split)
        if not split_rel:
            print(f"[WARN] Split '{split}' not defined in data.yaml")
            continue

        images_dir = base_dir / split_rel
        # Infer labels directory by replacing 'images' with 'labels'
        labels_dir = base_dir / str(split_rel).replace("images", "labels")

        img_count = 0
        lbl_count = 0
        box_count = 0

        if images_dir.exists():
            img_files = [f for f in images_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
            img_count = len(img_files)

        if labels_dir.exists():
            lbl_files = list(labels_dir.glob("*.txt"))
            lbl_count = len(lbl_files)

            for txt_path in lbl_files:
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cid = int(parts[0])
                                x_center, y_center, w, h = map(float, parts[1:5])

                                # Validate normalization bounds [0..1]
                                if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                                    stats["corrupted_files"] += 1
                                    print(f"[WARN] Unnormalized bbox in {txt_path.name}: {parts[1:5]}")
                                    continue

                                box_count += 1
                                if cid in stats["class_counts"]:
                                    stats["class_counts"][cid] += 1
                except Exception as e:
                    stats["corrupted_files"] += 1
                    print(f"[ERROR] Corrupted label file {txt_path.name}: {e}")

        stats["splits"][split] = {
            "images": img_count,
            "labels": lbl_count,
            "boxes": box_count
        }
        stats["total_images"] += img_count
        stats["total_labels"] += box_count

        print(f"Split '{split:5s}': {img_count:5d} images | {lbl_count:5d} label files | {box_count:5d} bounding boxes")

    print("-" * 60)
    print("CLASS DISTRIBUTION SUMMARY:")
    for cid, cname in class_map.items():
        count = stats["class_counts"].get(cid, 0)
        print(f"  Class {cid:2d} ({cname:20s}): {count:5d} annotations")
    print("=" * 60)

    if stats["total_images"] == 0:
        print("[NOTICE] No dataset images currently loaded in waste_dataset/ directory.")
        print("[NOTICE] Training scripts will utilize configured checkpoint until real annotated images are supplied.")

    return stats

if __name__ == "__main__":
    target_yaml = sys.argv[1] if len(sys.argv) > 1 else "../waste_dataset/data.yaml"
    if not os.path.exists(target_yaml):
        target_yaml = "waste_dataset/data.yaml"
    validate_yolo_dataset(target_yaml)
